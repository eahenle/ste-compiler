import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from ste_compiler import cli as cli_module
from ste_compiler.artifacts import (
    ArtifactFileV1,
    ArtifactPreflightResultV1,
    build_artifact_manifest,
)
from ste_compiler.cli import app, resources
from ste_compiler.ir.models import Quantity
from ste_compiler.ir.serialization import dumps_document, load_document
from ste_compiler.realizer import factory as realizer_factory
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.results import CompileSourceResult, SourceIdentity
from ste_compiler.training import TrainingRecordValidationError, build_training_record

ROOT = Path(__file__).parents[2]
runner = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_cli_runs_packaged_end_to_end_demo():
    result = runner.invoke(app, ["demo", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    CompileSourceResult.model_validate(payload)
    assert payload["schema_version"] == "compile-source-v1"
    assert payload["source"]["id"] == "hydraulic_warning.txt"
    assert payload["metadata"]["frontend"] == "offline-replay"
    assert payload["metadata"]["realizer"] == "deterministic"
    assert payload["validation"] == {"status": "accepted", "violations": []}
    assert payload["ir"]["sections"][0]["statements"][0]["id"] == "stop_pressure"
    assert payload["text"] == (
        "Warning: injury can occur when hydraulic pressure is more than 20 MPa.\n"
        "If hydraulic pressure is more than 20 MPa, stop the hydraulic pressure."
    )


def test_cli_prints_versioned_compile_source_json_schema():
    result = runner.invoke(app, ["schema", "compile-source"])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert schema["title"] == "CompileSourceResult"
    assert schema["properties"]["schema_version"]["const"] == "compile-source-v1"
    assert schema["properties"]["source"]["$ref"].endswith("/SourceIdentity")
    assert schema["$defs"]["SourceIdentity"]["properties"]["id"]["pattern"] == r"\S"
    assert schema["required"] == [
        "schema_version",
        "source",
        "text",
        "mappings",
        "validation",
        "metadata",
        "ir",
    ]

    payload = json.loads(runner.invoke(app, ["demo", "--json"]).stdout)
    payload.pop("schema_version")
    with pytest.raises(ValidationError):
        CompileSourceResult.model_validate(payload)
    with pytest.raises(ValidationError):
        SourceIdentity(id=" \t", sha256="0" * 64)


def test_cli_validates_and_prints_strict_realizer_config_schema():
    config = ROOT / "data/realizers/deterministic.yaml"

    validated = runner.invoke(
        app,
        ["validate-realizer-config", str(config), "--json"],
    )
    schema_result = runner.invoke(app, ["schema", "realizer-config"])

    assert validated.exit_code == 0, validated.output
    payload = json.loads(validated.stdout)
    assert payload["schema_version"] == "ste-realizer-config-v1"
    assert payload["architecture"] == "deterministic"
    assert payload["artifact_mode"] == "offline-cache-only"
    assert len(payload["config_sha256"]) == 64
    assert schema_result.exit_code == 0, schema_result.output
    schema = json.loads(schema_result.stdout)
    assert schema["discriminator"]["propertyName"] == "architecture"
    assert set(schema["discriminator"]["mapping"]) == {
        "decoder-only-lora",
        "decoder-only-lora-local-bundle",
        "deterministic",
        "encoder-decoder",
        "encoder-decoder-local-bundle",
    }

    local_validated = runner.invoke(
        app,
        [
            "validate-realizer-config",
            str(ROOT / "data/realizers/encoder-decoder-local-bundle-schema-example.yaml"),
            "--json",
        ],
    )
    assert local_validated.exit_code == 0, local_validated.output
    assert json.loads(local_validated.stdout)["artifact_mode"] == ("content-addressed-local-bundle")


@pytest.mark.parametrize(
    ("architecture", "artifact_type", "entrypoint", "validation_profile"),
    [
        (
            "encoder-decoder",
            "encoder-decoder-checkpoint",
            ".",
            "encoder-checkpoint-load-v1",
        ),
        (
            "decoder-only-lora",
            "decoder-only-lora-run",
            "adapter",
            "decoder-adapter-structure-v1",
        ),
    ],
)
def test_cli_routes_artifact_preflight_with_one_architecture_capture(
    monkeypatch,
    tmp_path,
    architecture,
    artifact_type,
    entrypoint,
    validation_profile,
):
    run_digest = "1" * 64
    artifact_digest = "2" * 64
    manifest = build_artifact_manifest(
        architecture=architecture,
        artifact_type=artifact_type,
        entrypoint=entrypoint,
        files=(
            ArtifactFileV1(
                path="run-manifest.json",
                sha256=run_digest,
                bytes=2,
            ),
            *(
                (
                    ArtifactFileV1(
                        path="adapter/adapter_config.json",
                        sha256="3" * 64,
                        bytes=2,
                    ),
                )
                if architecture == "decoder-only-lora"
                else ()
            ),
        ),
    )
    routing_calls = []
    capture_calls = []
    monkeypatch.setattr(
        cli_module,
        "read_artifact_manifest_for_routing",
        lambda root, digest: routing_calls.append((root, digest)) or manifest,
    )
    monkeypatch.setattr(
        cli_module,
        "preflight_encoder_decoder_artifact_bundle",
        lambda root, digest: capture_calls.append(("encoder-decoder", root, digest)),
    )
    monkeypatch.setattr(
        cli_module,
        "preflight_decoder_lora_artifact_bundle",
        lambda root, digest: capture_calls.append(("decoder-only-lora", root, digest)),
    )

    result = runner.invoke(
        app,
        [
            "preflight-artifact",
            str(tmp_path),
            "--manifest-sha256",
            artifact_digest,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    ArtifactPreflightResultV1.model_validate(payload)
    assert payload["architecture"] == architecture
    assert payload["artifact_type"] == artifact_type
    assert payload["artifact_manifest_sha256"] == artifact_digest
    assert payload["run_manifest_sha256"] == run_digest
    assert payload["validation_profile"] == validation_profile
    assert payload["network_access"] == "none"
    assert routing_calls == [(tmp_path, artifact_digest)]
    assert capture_calls == [(architecture, tmp_path, artifact_digest)]


def test_cli_prints_artifact_schemas():
    manifest_result = runner.invoke(app, ["schema", "artifact-manifest"])
    preflight_result = runner.invoke(app, ["schema", "artifact-preflight"])

    assert manifest_result.exit_code == 0, manifest_result.output
    assert preflight_result.exit_code == 0, preflight_result.output
    manifest_schema = json.loads(manifest_result.stdout)
    preflight_schema = json.loads(preflight_result.stdout)
    assert manifest_schema["properties"]["schema_version"]["const"] == "ste-artifact-bundle-v1"
    assert preflight_schema["properties"]["schema_version"]["const"] == "ste-artifact-preflight-v1"
    assert preflight_schema["properties"]["network_access"]["const"] == "none"


def test_decoder_training_cli_reports_retained_staged_bundle_digest(monkeypatch, tmp_path):
    retained_digest = "a" * 64
    run_manifest = SimpleNamespace(
        schema_version="ste-decoder-lora-run-v1",
        status="completed",
        optimizer_steps=2,
        training_losses=(1.0, 0.5),
        validation_loss=0.25,
        trainable_parameters=10,
    )
    monkeypatch.setattr(
        cli_module,
        "_decoder_training_inputs",
        lambda config, release: (object(), object()),
    )
    monkeypatch.setattr(
        cli_module,
        "run_decoder_lora_training_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            run_manifest=run_manifest,
            artifact_manifest_sha256=retained_digest,
        ),
    )

    result = runner.invoke(
        app,
        [
            "train-decoder-lora",
            str(tmp_path / "config.json"),
            str(tmp_path / "release"),
            str(tmp_path / "snapshot"),
            "b" * 64,
            str(tmp_path / "output"),
            "--source-checkout",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["artifact_manifest_sha256"] == retained_digest


def test_encoder_training_cli_reports_only_retained_staged_identities(monkeypatch, tmp_path):
    config = cli_module.load_training_config(
        ROOT / "data/training/encoder-decoder-schema-example.yaml"
    )
    retained_digest = "c" * 64

    class FakeRunManifest:
        schema_version = "encoder-decoder-run-manifest-v1"
        optimizer_steps = 2
        training_config_sha256 = "d" * 64

        def model_dump(self, *, mode):
            assert mode == "json"
            return {
                "schema_version": self.schema_version,
                "optimizer_steps": self.optimizer_steps,
            }

    run_manifest = FakeRunManifest()
    monkeypatch.setattr(cli_module, "load_training_config", lambda path: config)
    monkeypatch.setattr(
        cli_module,
        "run_encoder_decoder_training_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            run_manifest=run_manifest,
            artifact_manifest_sha256=retained_digest,
        ),
    )
    monkeypatch.setattr(cli_module, "canonical_run_manifest_json", lambda manifest: b"run\n")

    result = runner.invoke(
        app,
        [
            "train-encoder-decoder",
            str(tmp_path / "config.json"),
            str(tmp_path / "release"),
            "--output",
            str(tmp_path / "output"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["artifact_manifest_sha256"] == retained_digest
    assert payload["run_manifest_sha256"] == hashlib.sha256(b"run\n").hexdigest()


def test_cli_compiles_raw_source_with_verified_replay_fixture():
    example_root = ROOT / "data/end_to_end"
    result = runner.invoke(
        app,
        [
            "compile-source",
            str(example_root / "hydraulic_warning.txt"),
            "--ir-fixture",
            str(example_root / "hydraulic_warning.ir.yaml"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "compile-source-v1"
    assert len(payload["source"]["sha256"]) == 64
    assert payload["metadata"]["frontend"] == "offline-replay"
    assert payload["validation"]["status"] == "accepted"


def test_cli_routes_compile_source_through_versioned_deterministic_config():
    example_root = ROOT / "data/end_to_end"
    result = runner.invoke(
        app,
        [
            "compile-source",
            str(example_root / "hydraulic_warning.txt"),
            "--ir-fixture",
            str(example_root / "hydraulic_warning.ir.yaml"),
            "--realizer-config",
            str(ROOT / "data/realizers/deterministic.yaml"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["metadata"]["realizer"] == "deterministic"
    assert payload["metadata"]["artifact_mode"] == "offline-cache-only"
    assert len(payload["metadata"]["realizer_config_sha256"]) == 64
    assert payload["ir"]["metadata"]["realizer"] == "deterministic"


def test_cli_rejects_ambiguous_realizer_selectors_without_traceback():
    example_root = ROOT / "data/end_to_end"
    result = runner.invoke(
        app,
        [
            "compile-source",
            str(example_root / "hydraulic_warning.txt"),
            "--ir-fixture",
            str(example_root / "hydraulic_warning.ir.yaml"),
            "--realizer",
            "deterministic",
            "--realizer-config",
            str(ROOT / "data/realizers/deterministic.yaml"),
        ],
        color=False,
    )

    assert result.exit_code == 2
    assert "--realizer and --realizer-config cannot be combined" in ANSI_ESCAPE.sub(
        "",
        result.stderr,
    )
    assert "Traceback" not in result.output


def test_cli_replay_rejects_changed_source_without_traceback(tmp_path):
    example_root = ROOT / "data/end_to_end"
    changed = tmp_path / "hydraulic_warning.txt"
    changed.write_bytes(
        (example_root / "hydraulic_warning.txt").read_bytes()
        + b"\nDisconnect the pump before maintenance.\n"
    )

    result = runner.invoke(
        app,
        [
            "compile-source",
            str(changed),
            "--ir-fixture",
            str(example_root / "hydraulic_warning.ir.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "source SHA-256 does not match the fixture" in result.stderr
    assert "Traceback" not in result.output


def test_cli_replay_rejects_malformed_yaml_without_traceback(tmp_path):
    example_root = ROOT / "data/end_to_end"
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("sections: [unterminated")

    result = runner.invoke(
        app,
        [
            "compile-source",
            str(example_root / "hydraulic_warning.txt"),
            "--ir-fixture",
            str(malformed),
        ],
    )

    assert result.exit_code == 1
    assert "invalid replay IR fixture" in result.stderr
    assert "Traceback" not in result.output


def test_cli_preserves_crlf_source_offsets_and_hashes_original_bytes(tmp_path):
    prefix = "Unrepresented heading.\r\n"
    quote = "Install the access panel."
    source_bytes = f"{prefix}{quote}\r\n".encode()
    source = tmp_path / "windows-source.txt"
    source.write_bytes(source_bytes)
    proposal = yaml.safe_load((ROOT / "data/examples/installation.yaml").read_text())
    proposal["sections"][0]["statements"][0]["source_spans"] = [
        {
            "source_id": source.name,
            "start": len(prefix),
            "end": len(prefix) + len(quote),
            "quote": quote,
        }
    ]
    proposal["metadata"] = {
        "frontend": "forged-frontend",
        "frontend_version": "forged-frontend-version",
        "realizer": "forged-realizer",
        "realizer_version": "forged-realizer-version",
        "vocabulary_version": "forged-vocabulary",
        "terminology_version": "forged-terminology",
        "validator_profile": "forged-validator",
    }
    fixture = tmp_path / "windows-source.ir.yaml"
    fixture.write_text(
        yaml.safe_dump(
            {
                "schema_version": "replay-ir-v1",
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "ir": proposal,
            },
            sort_keys=False,
        )
    )

    result = runner.invoke(
        app,
        [
            "compile-source",
            str(source),
            "--ir-fixture",
            str(fixture),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert payload["metadata"] == {
        "frontend": "offline-replay",
        "frontend_version": "0.1.0",
        "realizer": "deterministic",
        "realizer_version": "0.2.0",
        "vocabulary_version": "demo-3",
        "terminology_version": "hydraulic-demo-1",
        "validator_profile": "strict-demo-1",
    }
    assert payload["ir"]["metadata"] == payload["metadata"]


def test_cli_realize_and_validate():
    result = runner.invoke(app, ["realize", str(ROOT / "data/examples/negative.yaml")])
    assert result.exit_code == 0
    assert result.stdout == "Do not open the shutoff valve.\n"
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(ROOT / "data/examples/invalid_semantic.txt"),
            "--ir",
            str(ROOT / "data/examples/negative.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "REQUIRED_NODE_OMITTED" in result.stdout
    assert "UNSUPPORTED_SEMANTIC_CHANGE" in result.stdout


def test_cli_compile_accepts_versioned_deterministic_config():
    result = runner.invoke(
        app,
        [
            "compile",
            str(ROOT / "data/examples/negative.yaml"),
            "--realizer-config",
            str(ROOT / "data/realizers/deterministic.yaml"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["text"] == "Do not open the shutoff valve."
    assert payload["metadata"]["artifact_mode"] == "offline-cache-only"
    assert len(payload["metadata"]["realizer_config_sha256"]) == 64


@pytest.mark.parametrize("architecture", ["encoder-decoder", "decoder-only-lora"])
def test_cli_routes_neural_configs_offline_through_compiler(
    architecture,
    tmp_path,
    monkeypatch,
):
    document = load_document(ROOT / "data/examples/negative.yaml")
    vocabulary, terminology = resources()
    reference = DeterministicRealizer().realize(document, vocabulary, terminology)
    plan = SymbolicLexicalizer(vocabulary, terminology).symbolize(reference.text)
    captured = {}

    class Generator:
        model_id = f"example/{architecture}"
        model_revision = "a" * 40
        base_model_revision = "a" * 40
        adapter_revision = "b" * 40

        def generate_symbols(self, serialized_ir, allowed_symbols):
            captured["serialized_ir"] = serialized_ir
            assert allowed_symbols == frozenset(plan.split())
            return plan

    def construct(runtime_config):
        captured["runtime_config"] = runtime_config
        return Generator()

    identity = {"repo_id": "example/model", "revision": "a" * 40}
    if architecture == "encoder-decoder":
        monkeypatch.setattr(
            realizer_factory,
            "TransformersEncoderDecoderSymbolGenerator",
            construct,
        )
        config_payload = {
            "schema_version": "ste-realizer-config-v1",
            "architecture": architecture,
            "checkpoint": identity,
        }
    else:
        monkeypatch.setattr(
            realizer_factory,
            "DecoderOnlyLoRASymbolGenerator",
            construct,
        )
        config_payload = {
            "schema_version": "ste-realizer-config-v1",
            "architecture": architecture,
            "base_model": identity,
            "adapter": {
                "repo_id": "example/adapter",
                "revision": "b" * 40,
            },
            "prompt_profile": "decoder-only-symbol-plan-v1",
        }
    config = tmp_path / f"{architecture}.json"
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "compile",
            str(ROOT / "data/examples/negative.yaml"),
            "--realizer-config",
            str(config),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["metadata"]["realizer"] == "symbolic-neural"
    assert payload["metadata"]["artifact_mode"] == "offline-cache-only"
    assert json.loads(captured["serialized_ir"])["metadata"]["realizer"] == "deterministic"
    assert captured["runtime_config"].local_files_only is True


def test_cli_rejects_invalid_realizer_config_without_traceback(tmp_path):
    config = tmp_path / "mutable-revision.yaml"
    config.write_text(
        "schema_version: ste-realizer-config-v1\n"
        "architecture: encoder-decoder\n"
        "checkpoint:\n"
        "  repo_id: example/model\n"
        "  revision: main\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "compile",
            str(ROOT / "data/examples/negative.yaml"),
            "--realizer-config",
            str(config),
        ],
    )

    assert result.exit_code == 1
    assert "revision" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "command",
    [
        [
            "compile",
            str(ROOT / "data/examples/negative.yaml"),
        ],
        [
            "compile-source",
            str(ROOT / "data/end_to_end/hydraulic_warning.txt"),
            "--ir-fixture",
            str(ROOT / "data/end_to_end/hydraulic_warning.ir.yaml"),
        ],
    ],
)
def test_cli_rejects_local_artifact_locators_without_portable_config(command, tmp_path):
    result = runner.invoke(
        app,
        [*command, "--artifact-bundle", str(tmp_path / "bundle")],
    )

    assert result.exit_code == 1
    assert "require --realizer-config" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("architecture", "config_payload", "include_snapshot"),
    [
        (
            "encoder-decoder-local-bundle",
            {
                "schema_version": "ste-realizer-config-v1",
                "architecture": "encoder-decoder-local-bundle",
                "artifact_manifest_sha256": "a" * 64,
                "intended_use": "mechanics-smoke",
            },
            False,
        ),
        (
            "decoder-only-lora-local-bundle",
            {
                "schema_version": "ste-realizer-config-v1",
                "architecture": "decoder-only-lora-local-bundle",
                "artifact_manifest_sha256": "a" * 64,
                "model_snapshot_manifest_sha256": "b" * 64,
                "base_model": {"repo_id": "example/base", "revision": "c" * 40},
                "tokenizer": {"repo_id": "example/base", "revision": "c" * 40},
                "intended_use": "mechanics-smoke",
                "prompt_profile": "decoder-only-symbol-plan-v1",
            },
            True,
        ),
    ],
)
def test_cli_keeps_local_locators_outside_config_and_routes_them_explicitly(
    architecture,
    config_payload,
    include_snapshot,
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / f"{architecture}.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    bundle = tmp_path / "bundle"
    snapshot = tmp_path / "snapshot"
    captured = {}

    def construct(config, **locators):
        captured["config"] = config
        captured["locators"] = locators
        return DeterministicRealizer()

    monkeypatch.setattr(cli_module, "build_realizer", construct)
    command = [
        "compile",
        str(ROOT / "data/examples/negative.yaml"),
        "--realizer-config",
        str(config_path),
        "--artifact-bundle",
        str(bundle),
        "--json",
    ]
    if include_snapshot:
        command.extend(["--model-snapshot", str(snapshot)])

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert captured["config"].architecture == architecture
    assert captured["locators"]["artifact_bundle"] == bundle
    assert captured["locators"]["model_snapshot"] == (snapshot if include_snapshot else None)
    assert "artifact_bundle" not in captured["config"].model_fields_set
    assert "model_snapshot" not in captured["config"].model_fields_set


def test_cli_exports_symbolic_training_record():
    result = runner.invoke(
        app,
        [
            "plan-symbols",
            str(ROOT / "data/examples/warning_pressure.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["document_id"] == "warning_pressure"
    assert "TERM_hydraulic_pressure|hydraulic%20pressure" in record["allowed_symbols"]
    assert "NUMBER_20" in record["symbols"]
    assert json.loads(record["serialized_ir"])["id"] == "warning_pressure"


def test_direct_training_record_rejects_forbidden_alias(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={"manner": "system pressure"}
    )

    with pytest.raises(TrainingRecordValidationError) as captured:
        build_training_record(document, vocab, terms)

    assert captured.value.report.status == "rejected"
    assert {item.code for item in captured.value.report.violations} == {"TERMINOLOGY_ALIAS"}


def test_cli_training_plan_rejects_forbidden_alias_consistently_with_compile(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={"manner": "system pressure"}
    )
    source = tmp_path / "forbidden_alias.json"
    source.write_text(dumps_document(document, as_json=True))

    compiled = runner.invoke(app, ["compile", str(source)])
    planned = runner.invoke(app, ["plan-symbols", str(source)])
    planned_json = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    expected = "ERROR TERMINOLOGY_ALIAS: Use the canonical term instead of 'system pressure'."
    assert compiled.exit_code == planned.exit_code == 1
    assert expected in compiled.stdout
    assert planned.stdout == f"{expected}\n"
    assert "PLAN_EXACT_WHITESPACE_V1" not in planned.stdout
    assert planned_json.exit_code == 1
    rejected_report = json.loads(planned_json.stdout)
    assert rejected_report["status"] == "rejected"
    assert [item["code"] for item in rejected_report["violations"]] == ["TERMINOLOGY_ALIAS"]
    assert "symbols" not in rejected_report


def test_cli_exports_reproducible_symbolic_corpus(tmp_path):
    output = tmp_path / "training"
    result = runner.invoke(
        app,
        [
            "export-symbolic-corpus",
            str(ROOT / "data/examples"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    manifest = json.loads((output / "current" / "manifest.json").read_text())
    assert manifest["schema_version"] == "symbolic-corpus-v1"
    assert manifest["record_count"] == 5
    assert manifest["corpus_sha256"] in result.stdout
    assert str(output / "current" / "corpus.jsonl") in result.stdout
    assert len((output / "current" / "corpus.jsonl").read_text().splitlines()) == 5


def test_cli_reconstructs_demonstration_corpus(tmp_path):
    output = tmp_path / "demonstration-corpus"

    result = runner.invoke(
        app,
        ["build-demonstration-corpus", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["record_count"] == 12
    assert manifest["split_counts"] == {
        "adversarial": 3,
        "test": 3,
        "train": 4,
        "validation": 2,
    }
    assert manifest["construction_sha256"] in result.stdout
    assert (output / "dataset-card.md").is_file()
    assert (output / "checksums.sha256").is_file()

    verified = runner.invoke(
        app,
        ["verify-demonstration-corpus", str(output)],
    )
    assert verified.exit_code == 0, verified.output
    assert "Verified 12 records" in verified.stdout

    repeated = runner.invoke(
        app,
        ["build-demonstration-corpus", "--output", str(output)],
    )
    assert repeated.exit_code == 1
    assert "output directory must be empty" in repeated.stderr
    assert "Traceback" not in repeated.output


def _training_config_payload() -> dict[str, object]:
    identity = {"repo_id": "example/tiny-model", "revision": "a" * 40}
    return {
        "schema_version": "ste-training-config-v1",
        "architecture": "encoder-decoder",
        "corpus": {
            "dataset_version": "demonstration-corpus-1",
            "manifest_sha256": ("f6ae4582669c4d7d06e33018088b900ffa0f8aa8b6e0d9f1beeccca2023faa7b"),
            "train_sha256": ("1772fbe01a15c28d174e139f93e5c3b0fd6744c01cf5c81b79fb842c9609ebd0"),
            "validation_sha256": (
                "ea16d6bae1f624c26581e05b02cb693282805d93f945bd6e00e73a48a79d15dd"
            ),
        },
        "base_model": identity,
        "tokenizer": identity,
        "seed": 1729,
        "max_steps": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {"learning_rate": 0.0001, "weight_decay": 0},
        "strategy": "full",
        "max_source_tokens": 1024,
        "max_target_tokens": 256,
    }


def test_cli_validates_training_config_and_hash_pinned_release(tmp_path):
    config = tmp_path / "training.json"
    config.write_text(json.dumps(_training_config_payload()))

    validated = runner.invoke(app, ["validate-training-config", str(config), "--json"])
    assert validated.exit_code == 0, validated.output
    validated_payload = json.loads(validated.stdout)
    assert validated_payload["architecture"] == "encoder-decoder"
    assert len(validated_payload["config_sha256"]) == 64

    verified = runner.invoke(
        app,
        [
            "verify-training-release",
            str(config),
            str(ROOT / "datasets/demonstration-corpus-1"),
            "--json",
        ],
    )
    assert verified.exit_code == 0, verified.output
    verified_payload = json.loads(verified.stdout)
    assert verified_payload["split_counts"] == {
        "adversarial": 3,
        "test": 3,
        "train": 4,
        "validation": 2,
    }
    assert (
        verified_payload["manifest_sha256"]
        == _training_config_payload()["corpus"]["manifest_sha256"]
    )
    assert verified_payload["symbol_count"] > 0


def test_decoder_smoke_cli_rejects_wrong_architecture_without_traceback(tmp_path):
    config = tmp_path / "encoder.json"
    config.write_text(json.dumps(_training_config_payload()))

    result = runner.invoke(
        app,
        [
            "prepare-decoder-smoke-fixture",
            str(config),
            str(ROOT / "datasets/demonstration-corpus-1"),
            str(tmp_path / "model"),
        ],
    )

    assert result.exit_code == 1
    assert "architecture must be decoder-only-lora" in result.stderr
    assert "Traceback" not in result.output


def test_cli_rejects_training_release_identity_mismatch(tmp_path):
    payload = _training_config_payload()
    payload["corpus"]["manifest_sha256"] = "0" * 64
    config = tmp_path / "training.json"
    config.write_text(json.dumps(payload))

    result = runner.invoke(
        app,
        [
            "verify-training-release",
            str(config),
            str(ROOT / "datasets/demonstration-corpus-1"),
        ],
    )

    assert result.exit_code == 1
    assert "manifest SHA-256 does not match" in result.stderr
    assert "Traceback" not in result.output


def test_cli_rejects_mismatched_corpus_profile(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    document.metadata.vocabulary_version = "unloaded-vocabulary"
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    output = tmp_path / "training"

    result = runner.invoke(
        app,
        ["export-symbolic-corpus", str(source), "--output", str(output)],
    )

    assert result.exit_code == 1
    assert "vocabulary_version='unloaded-vocabulary'" in result.stderr
    assert not output.exists()


def test_cli_corpus_export_rejects_unknown_term_without_traceback_or_artifacts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"][0]["object"] = {"term_id": "unknown_term"}
    document = type(document).model_validate(raw)
    (source / "unknown-term.json").write_text(
        dumps_document(document, as_json=True),
        encoding="utf-8",
    )
    output = tmp_path / "training"

    result = runner.invoke(
        app,
        ["export-symbolic-corpus", str(source), "--output", str(output)],
    )

    assert result.exit_code == 1
    assert "unknown_term" in result.stderr
    assert "Traceback" not in result.output
    assert not output.exists()


def test_cli_training_plan_preserves_negative_quantity_symbol(tmp_path):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    instruction = document.sections[0].statements[0]
    negative_quantity = Quantity(value=-20, unit="MPa", comparator="more_than")
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                instruction.quantity_constraints[0].model_copy(
                    update={"quantity": negative_quantity}
                )
            ]
        }
    )
    source = tmp_path / "negative_quantity.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert "NUMBER_-20" in record["allowed_symbols"]
    assert "NUMBER_-20" in record["symbols"].split()
    assert "PUNCT_U002D" not in record["symbols"].split()


def test_cli_training_plan_rejects_nonfinite_quantity(tmp_path):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"][0]["quantity_constraints"][0]["quantity"]["value"] = float(
        "nan"
    )
    source = tmp_path / "nonfinite_quantity.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 1
    assert "finite number" in result.output


def test_cli_training_plan_round_trips_quoted_manner(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": '"safe"'})
    source = tmp_path / "quoted_manner.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["text"] == 'Install the access panel "safe".'
    assert record["symbols"].endswith(
        "TERM_access_panel|access%20panel SPACE PUNCT_U0022 WORD_safe PUNCT_U0022 PERIOD"
    )
    assert "PUNCT_U0022" in record["allowed_symbols"]


def test_cli_training_plan_preserves_punctuation_adjacency(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": "safe;slowly"})
    source = tmp_path / "punctuation_adjacency.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["text"] == "Install the access panel safe;slowly."
    assert record["symbols"].startswith("PLAN_EXACT_WHITESPACE_V1 ")
    assert record["symbols"].endswith("WORD_safe PUNCT_U003B WORD_slowly PERIOD")
    assert {"PLAN_EXACT_WHITESPACE_V1", "PUNCT_U003B"} <= set(record["allowed_symbols"])


def test_cli_training_plan_preserves_exact_word_case_after_question(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": "safe? slowly"})
    source = tmp_path / "exact_word_case.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["text"] == "Install the access panel safe? slowly."
    assert record["symbols"].startswith("PLAN_EXACT_WHITESPACE_V1 WORD_Install SPACE")
    assert record["symbols"].endswith("WORD_safe QUESTION SPACE WORD_slowly PERIOD")
    assert {"WORD_Install", "WORD_slowly"} <= set(record["allowed_symbols"])


def test_cli_training_plan_preserves_capitalized_first_term_surface(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"] = [
        {
            "kind": "state",
            "id": "state_001",
            "subject": {"term_id": "access_panel"},
            "predicate": "is",
            "value": "safe",
            "source_spans": [],
        }
    ]
    document = type(document).model_validate(raw)
    source = tmp_path / "first_term_surface.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    exact_term = "TERM_access_panel|Access%20panel"
    assert record["text"] == "Access panel is safe."
    assert record["symbols"].startswith(f"PLAN_EXACT_WHITESPACE_V1 {exact_term} SPACE")
    assert exact_term in record["allowed_symbols"]


def test_cli_training_plan_preserves_unicode_casefold_expansion(
    tmp_path, monkeypatch, vocab, terms
):
    custom_terms = type(terms)(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "ß", "aliases": []})
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    monkeypatch.setattr("ste_compiler.cli.resources", lambda: (vocab, custom_terms))
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"] = [
        {
            "kind": "state",
            "id": "state_001",
            "subject": {"term_id": "access_panel"},
            "predicate": "is",
            "value": "safe",
            "source_spans": [],
        }
    ]
    document = type(document).model_validate(raw)
    source = tmp_path / "unicode_casefold.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    exact_term = "TERM_access_panel|SS"
    assert record["text"] == "SS is safe."
    assert record["symbols"].startswith(f"PLAN_EXACT_WHITESPACE_V1 {exact_term} SPACE")
    assert exact_term in record["allowed_symbols"]


def test_validate_text_does_not_inherit_expected_semantics(tmp_path):
    submitted = tmp_path / "submitted.txt"
    submitted.write_text("Open the shutoff valve.\n")
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(submitted),
            "--ir",
            str(ROOT / "data/examples/installation.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert '"status": "rejected"' in result.stdout
    assert "UNSUPPORTED_SEMANTIC_CHANGE" in result.stdout


def test_validate_text_accepts_exact_controlled_realization(tmp_path):
    submitted = tmp_path / "submitted.txt"
    submitted.write_text("Install the access panel.\n")
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(submitted),
            "--ir",
            str(ROOT / "data/examples/installation.yaml"),
        ],
    )
    assert result.exit_code == 0
    assert result.stdout == "accepted\n"


def test_cli_critical_failure_and_glossary():
    bad = runner.invoke(
        app, ["validate-text", str(ROOT / "data/examples/invalid_unauthorized.txt")]
    )
    assert bad.exit_code == 1
    good = runner.invoke(app, ["glossary", "check", str(ROOT / "data/demo_terminology.yaml")])
    assert good.exit_code == 0


def test_glossary_check_rejects_duplicate_ids(tmp_path):
    raw = yaml.safe_load((ROOT / "data/demo_terminology.yaml").read_text())
    raw["terms"][1]["id"] = raw["terms"][0]["id"]
    source = tmp_path / "duplicate_ids.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False))

    result = runner.invoke(app, ["glossary", "check", str(source)])

    assert result.exit_code == 1
    assert "duplicate terminology ID" in result.output


def test_glossary_check_rejects_replacement_cycles(tmp_path):
    raw = yaml.safe_load((ROOT / "data/demo_terminology.yaml").read_text())
    old_pressure = next(term for term in raw["terms"] if term["id"] == "old_pressure")
    old_pressure["replacement_term_id"] = "old_pressure"
    source = tmp_path / "replacement_cycle.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False))

    result = runner.invoke(app, ["glossary", "check", str(source)])

    assert result.exit_code == 1
    assert "replacement cycle" in result.output


def test_evaluation_reports(tmp_path):
    result = runner.invoke(
        app, ["evaluate", str(ROOT / "data/evaluation"), "--output", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()

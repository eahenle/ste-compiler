import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from ste_compiler.cli import app
from ste_compiler.ir.models import Quantity
from ste_compiler.ir.serialization import dumps_document, load_document
from ste_compiler.results import CompileSourceResult
from ste_compiler.training import TrainingRecordValidationError, build_training_record

ROOT = Path(__file__).parents[2]
runner = CliRunner()


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
    assert payload["text"].startswith("Warning: injury can occur")


def test_cli_prints_versioned_compile_source_json_schema():
    result = runner.invoke(app, ["schema", "compile-source"])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert schema["title"] == "CompileSourceResult"
    assert schema["properties"]["schema_version"]["const"] == "compile-source-v1"
    assert schema["properties"]["source"]["$ref"].endswith("/SourceIdentity")
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


def test_cli_replay_rejects_changed_source_without_traceback(tmp_path):
    example_root = ROOT / "data/end_to_end"
    changed = tmp_path / "hydraulic_warning.txt"
    changed.write_text(
        (example_root / "hydraulic_warning.txt").read_text().replace("injury", "damage")
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
    assert "quote does not match the source" in result.stderr
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
    fixture.write_text(yaml.safe_dump(proposal, sort_keys=False))

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
        "realizer_version": "0.1.0",
        "vocabulary_version": "demo-1",
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

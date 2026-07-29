import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_installed_wheel_contains_default_cli_data(tmp_path):
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("ste_compiler-*.whl"))

    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (installed / "ste_compiler/py.typed").is_file()
    assert next(installed.glob("ste_compiler-*.dist-info/licenses/LICENSE")).is_file()

    clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    clean_env["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

original_import = builtins.__import__

def import_without_fcntl(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fcntl":
        raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_fcntl
from ste_compiler.cli import app
from typer.testing import CliRunner

result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
""",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    command = [sys.executable, "-m", "ste_compiler.cli"]
    public_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ste_compiler.realizer import "
                "DecoderOnlyLoRAConfig, DecoderOnlyLoRAError, "
                "DecoderOnlyLoRALocalBundleRealizerConfigV1, "
                "DecoderOnlyLoRARealizerConfigV1, DeterministicRealizerConfigV1, "
                "DecoderOnlyLoRASymbolGenerator, EncoderDecoderConfig, "
                "EncoderDecoderError, EncoderDecoderLocalBundleConfig, "
                "EncoderDecoderLocalBundleRealizerConfigV1, EncoderDecoderRealizerConfigV1, "
                "LocalDecoderOnlyLoRARuntimeConfig, load_local_decoder_lora_generator, "
                "TransformersEncoderDecoderSymbolGenerator, build_realizer, "
                "load_realizer_config, realizer_config_sha256; "
                "assert DecoderOnlyLoRAConfig and DecoderOnlyLoRAError "
                "and DecoderOnlyLoRALocalBundleRealizerConfigV1 "
                "and DecoderOnlyLoRARealizerConfigV1 and DeterministicRealizerConfigV1 "
                "and DecoderOnlyLoRASymbolGenerator and EncoderDecoderConfig "
                "and EncoderDecoderError and EncoderDecoderLocalBundleConfig "
                "and EncoderDecoderLocalBundleRealizerConfigV1 "
                "and EncoderDecoderRealizerConfigV1 and LocalDecoderOnlyLoRARuntimeConfig "
                "and load_local_decoder_lora_generator "
                "and TransformersEncoderDecoderSymbolGenerator and build_realizer "
                "and load_realizer_config and realizer_config_sha256"
            ),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not public_import.stderr

    artifact_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ste_compiler.artifacts import "
                "ArtifactBundleManifestV1, ArtifactFileV1, ArtifactPreflightResultV1, "
                "open_verified_artifact_bundle, verify_artifact_bundle; "
                "from ste_compiler.training import "
                "decoder_lora_artifact_manifest_sha256, "
                "encoder_decoder_artifact_manifest_sha256, "
                "open_verified_decoder_lora_artifact_bundle, "
                "open_verified_encoder_decoder_artifact_bundle, "
                "preflight_decoder_lora_artifact_bundle, "
                "preflight_encoder_decoder_artifact_bundle; "
                "assert ArtifactBundleManifestV1 and ArtifactFileV1 "
                "and ArtifactPreflightResultV1 and open_verified_artifact_bundle "
                "and verify_artifact_bundle and decoder_lora_artifact_manifest_sha256 "
                "and encoder_decoder_artifact_manifest_sha256 "
                "and open_verified_decoder_lora_artifact_bundle "
                "and open_verified_encoder_decoder_artifact_bundle "
                "and preflight_decoder_lora_artifact_bundle "
                "and preflight_encoder_decoder_artifact_bundle"
            ),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not artifact_import.stderr

    reference_release_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ste_compiler.reference_release import "
                "ReferenceReleaseManifestV1, ReferenceReleaseMetadataV1, "
                "build_reference_release, read_verified_reference_release, "
                "verify_reference_release; "
                "assert ReferenceReleaseManifestV1 and ReferenceReleaseMetadataV1 "
                "and build_reference_release and read_verified_reference_release "
                "and verify_reference_release"
            ),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not reference_release_import.stderr

    demo = subprocess.run(
        [*command, "demo", "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    demo_payload = json.loads(demo.stdout)
    assert demo_payload["schema_version"] == "compile-source-v1"
    assert demo_payload["metadata"]["frontend"] == "offline-replay"
    assert demo_payload["validation"]["status"] == "accepted"

    realized = subprocess.run(
        [*command, "realize", str(ROOT / "data/examples/negative.yaml")],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert realized.stdout == "Do not open the shutoff valve.\n"

    planned = subprocess.run(
        [*command, "plan-symbols", str(ROOT / "data/examples/warning_pressure.yaml"), "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    training_record = json.loads(planned.stdout)
    assert "WORD_occur" in training_record["allowed_symbols"]
    assert "TERM_hydraulic_pressure|hydraulic%20pressure" in training_record["symbols"]

    corpus = tmp_path / "training"
    subprocess.run(
        [
            *command,
            "export-symbolic-corpus",
            str(ROOT / "data/examples"),
            "--output",
            str(corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((corpus / "current" / "manifest.json").read_text())["record_count"] == 5
    assert len((corpus / "current" / "corpus.jsonl").read_text().splitlines()) == 5

    demonstration_corpus = tmp_path / "demonstration-corpus"
    subprocess.run(
        [
            *command,
            "build-demonstration-corpus",
            "--output",
            str(demonstration_corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    release_manifest = json.loads((demonstration_corpus / "manifest.json").read_text())
    assert release_manifest["record_count"] == 12
    assert (demonstration_corpus / "terminology.json").is_file()
    assert (demonstration_corpus / "vocabulary.json").is_file()
    subprocess.run(
        [
            *command,
            "verify-demonstration-corpus",
            str(demonstration_corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )

    demonstration_corpus_v2 = tmp_path / "demonstration-corpus-2"
    subprocess.run(
        [
            *command,
            "build-demonstration-corpus",
            "--version",
            "2",
            "--output",
            str(demonstration_corpus_v2),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    release_manifest_v2 = json.loads((demonstration_corpus_v2 / "manifest.json").read_text())
    assert release_manifest_v2["dataset_version"] == "demonstration-corpus-2"
    assert release_manifest_v2["record_count"] == 24
    subprocess.run(
        [
            *command,
            "verify-demonstration-corpus",
            str(demonstration_corpus_v2),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )

    benchmark_fixture = installed / "ste_compiler/data/benchmark/v1"
    benchmark_report = tmp_path / "benchmark-report"
    generated_benchmark = subprocess.run(
        [
            *command,
            "benchmark-report",
            str(benchmark_fixture / "benchmark-spec.json"),
            str(benchmark_fixture / "failure-taxonomy.json"),
            str(benchmark_fixture / "prediction-manifest.json"),
            str(benchmark_fixture / "predictions.jsonl"),
            str(demonstration_corpus_v2),
            "--output",
            str(benchmark_report),
            "--json",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark_manifest = json.loads(generated_benchmark.stdout)
    assert benchmark_manifest["evidence_label"] == "deterministic_fixture_only"
    for expected in (benchmark_fixture / "expected-report").iterdir():
        assert (benchmark_report / expected.name).read_bytes() == expected.read_bytes()

    training_config = installed / "ste_compiler/data/training/encoder-decoder-schema-example.yaml"
    decoder_config = installed / "ste_compiler/data/training/decoder-only-lora-schema-example.yaml"
    reference_metadata = (
        installed / "ste_compiler/data/reference-release/synthetic-mechanics-metadata.json"
    )
    assert training_config.is_file()
    assert decoder_config.is_file()
    assert reference_metadata.is_file()
    validated_training = subprocess.run(
        [*command, "validate-training-config", str(training_config), "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated_training.stdout)["architecture"] == "encoder-decoder"
    validated_decoder = subprocess.run(
        [*command, "validate-training-config", str(decoder_config), "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated_decoder.stdout)["architecture"] == "decoder-only-lora"

    realizer_directory = installed / "ste_compiler/data/realizers"
    realizer_configs = {
        "deterministic": realizer_directory / "deterministic.yaml",
        "encoder-decoder": realizer_directory / "encoder-decoder-schema-example.yaml",
        "decoder-only-lora": realizer_directory / "decoder-only-lora-schema-example.yaml",
        "encoder-decoder-local-bundle": (
            realizer_directory / "encoder-decoder-local-bundle-schema-example.yaml"
        ),
        "decoder-only-lora-local-bundle": (
            realizer_directory / "decoder-only-lora-local-bundle-schema-example.yaml"
        ),
    }
    for architecture, realizer_config in realizer_configs.items():
        assert realizer_config.is_file()
        validated_realizer = subprocess.run(
            [
                *command,
                "validate-realizer-config",
                str(realizer_config),
                "--json",
            ],
            cwd=tmp_path,
            env=clean_env,
            check=True,
            capture_output=True,
            text=True,
        )
        realizer_payload = json.loads(validated_realizer.stdout)
        assert realizer_payload["architecture"] == architecture
        expected_mode = (
            "content-addressed-local-bundle"
            if architecture.endswith("-local-bundle")
            else "offline-cache-only"
        )
        assert realizer_payload["artifact_mode"] == expected_mode
        assert len(realizer_payload["config_sha256"]) == 64

    realizer_schema = subprocess.run(
        [*command, "schema", "realizer-config"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(realizer_schema.stdout)["discriminator"]["propertyName"] == "architecture"
    artifact_manifest_schema = subprocess.run(
        [*command, "schema", "artifact-manifest"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact_preflight_schema = subprocess.run(
        [*command, "schema", "artifact-preflight"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    reference_release_schema = subprocess.run(
        [*command, "schema", "reference-release"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        json.loads(artifact_manifest_schema.stdout)["properties"]["schema_version"]["const"]
        == "ste-artifact-bundle-v1"
    )
    assert (
        json.loads(artifact_preflight_schema.stdout)["properties"]["schema_version"]["const"]
        == "ste-artifact-preflight-v1"
    )
    assert (
        json.loads(reference_release_schema.stdout)["properties"]["schema_version"]["const"]
        == "ste-reference-artifact-release-v1"
    )
    packaged_example = installed / "ste_compiler/data/examples/negative.yaml"
    configured_compile = subprocess.run(
        [
            *command,
            "compile",
            str(packaged_example),
            "--realizer-config",
            str(realizer_configs["deterministic"]),
            "--json",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    configured_payload = json.loads(configured_compile.stdout)
    assert configured_payload["text"] == "Do not open the shutoff valve."
    assert configured_payload["metadata"]["artifact_mode"] == "offline-cache-only"

    verified_training = subprocess.run(
        [
            *command,
            "verify-training-release",
            str(training_config),
            str(demonstration_corpus),
            "--json",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verified_training.stdout)["split_counts"] == release_manifest["split_counts"]

    reports = tmp_path / "reports"
    subprocess.run(
        [*command, "evaluate", "--output", str(reports)],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (reports / "report.json").is_file()

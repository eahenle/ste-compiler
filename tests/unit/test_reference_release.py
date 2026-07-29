from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import ste_compiler.reference_release as release_module
from ste_compiler.cli import app
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.decoder_lora import DecoderOnlyLoRAError
from ste_compiler.realizer.encoder_decoder import EncoderDecoderError
from ste_compiler.reference_release import (
    REFERENCE_RELEASE_FILES,
    ReferenceReleaseError,
    ReferenceReleaseMetadataV1,
    ReferenceTrackAuthorizationV1,
    build_reference_release,
    read_verified_reference_release,
    verify_reference_release,
)
from ste_compiler.training import load_training_config

ROOT = Path(__file__).parents[2]
CORPUS = ROOT / "datasets/demonstration-corpus-1"
ENCODER_CONFIG = ROOT / "data/training/encoder-decoder-schema-example.yaml"
DECODER_CONFIG = ROOT / "data/training/decoder-only-lora-schema-example.yaml"
SNAPSHOT_DIGEST = "3" * 64
ENCODER_DIGEST = "1" * 64
DECODER_DIGEST = "2" * 64


def _metadata():
    encoder = load_training_config(ENCODER_CONFIG)
    decoder = load_training_config(DECODER_CONFIG)
    return ReferenceReleaseMetadataV1(
        schema_version="ste-reference-release-metadata-v1",
        release_id="synthetic-mechanics-1",
        intended_use="mechanics-smoke",
        no_quality_claim=True,
        tracks=(
            ReferenceTrackAuthorizationV1(
                architecture="encoder-decoder",
                base_model=encoder.base_model,
                base_model_origin="generated-by-repository-smoke-fixture",
                base_model_license="MIT",
                artifact_license="MIT",
                redistribution="external-artifact-not-included",
            ),
            ReferenceTrackAuthorizationV1(
                architecture="decoder-only-lora",
                base_model=decoder.base_model,
                base_model_origin="generated-by-repository-smoke-fixture",
                base_model_license="MIT",
                artifact_license="MIT",
                redistribution="external-artifact-not-included",
            ),
        ),
    )


def _install_release_seams(monkeypatch):
    encoder_config = load_training_config(ENCODER_CONFIG)
    decoder_config = load_training_config(DECODER_CONFIG)
    encoder_run = SimpleNamespace(training_config=encoder_config)
    decoder_run = SimpleNamespace(
        training_config=decoder_config,
        model_snapshot_manifest_sha256=SNAPSHOT_DIGEST,
    )

    @contextmanager
    def open_encoder(path, digest):
        assert digest == ENCODER_DIGEST
        yield SimpleNamespace(run_manifest=encoder_run, run_manifest_sha256="4" * 64)

    @contextmanager
    def open_decoder(path, digest):
        assert digest == DECODER_DIGEST
        yield SimpleNamespace(run_manifest=decoder_run, run_manifest_sha256="5" * 64)

    def read_snapshot(path, *, base_model, tokenizer, expected_manifest_sha256):
        assert base_model == decoder_config.base_model
        assert tokenizer == decoder_config.tokenizer
        assert expected_manifest_sha256 == SNAPSHOT_DIGEST
        return SimpleNamespace(manifest_sha256=SNAPSHOT_DIGEST)

    monkeypatch.setattr(
        release_module,
        "open_verified_encoder_decoder_artifact_bundle",
        open_encoder,
    )
    monkeypatch.setattr(
        release_module,
        "open_verified_decoder_lora_artifact_bundle",
        open_decoder,
    )
    monkeypatch.setattr(
        release_module,
        "read_verified_model_snapshot_for_identities",
        read_snapshot,
    )
    monkeypatch.setattr(
        release_module,
        "build_realizer",
        lambda config, **locators: DeterministicRealizer(),
    )


def _build(tmp_path, monkeypatch):
    _install_release_seams(monkeypatch)
    output = tmp_path / "release"
    result = build_reference_release(
        _metadata(),
        CORPUS,
        tmp_path / "encoder",
        ENCODER_DIGEST,
        tmp_path / "decoder",
        DECODER_DIGEST,
        tmp_path / "snapshot",
        SNAPSHOT_DIGEST,
        output,
    )
    return output, result


class _PreparedFailureRealizer:
    def __init__(self, error_type, *, fail_during):
        self.error_type = error_type
        self.fail_during = fail_during
        self.prepare_calls = 0
        self.realize_calls = 0

    def prepare(self):
        self.prepare_calls += 1
        if self.fail_during == "prepare":
            raise self.error_type("artifact loading failed")

    def realize(self, document, vocabulary, terminology, constraints=None):
        self.realize_calls += 1
        if self.fail_during == "realize":
            raise self.error_type("generation failed")
        return DeterministicRealizer().realize(document, vocabulary, terminology)


def _install_architecture_failure(monkeypatch, architecture, error_type, *, fail_during):
    _install_release_seams(monkeypatch)
    target_config_architecture = f"{architecture}-local-bundle"
    failing = _PreparedFailureRealizer(error_type, fail_during=fail_during)

    def build(config, **locators):
        if config.architecture == target_config_architecture:
            return failing
        return DeterministicRealizer()

    monkeypatch.setattr(release_module, "build_realizer", build)
    return failing


@pytest.mark.parametrize(
    ("architecture", "error_type"),
    (
        ("encoder-decoder", EncoderDecoderError),
        ("decoder-only-lora", DecoderOnlyLoRAError),
    ),
)
def test_architecture_generation_errors_become_rejected_predictions(
    tmp_path,
    monkeypatch,
    architecture,
    error_type,
):
    failing = _install_architecture_failure(
        monkeypatch,
        architecture,
        error_type,
        fail_during="realize",
    )
    output = tmp_path / "release"

    build_reference_release(
        _metadata(),
        CORPUS,
        tmp_path / "encoder",
        ENCODER_DIGEST,
        tmp_path / "decoder",
        DECODER_DIGEST,
        tmp_path / "snapshot",
        SNAPSHOT_DIGEST,
        output,
    )

    predictions = [
        json.loads(line)
        for line in (output / f"{architecture}-predictions.jsonl").read_text().splitlines()
    ]
    assert failing.prepare_calls == 1
    assert failing.realize_calls == len(predictions) == 6
    assert {prediction["outcome"] for prediction in predictions} == {"rejected"}
    assert {prediction["error_type"] for prediction in predictions} == {error_type.__name__}
    assert {prediction["error"] for prediction in predictions} == {"generation failed"}


@pytest.mark.parametrize(
    ("architecture", "error_type"),
    (
        ("encoder-decoder", EncoderDecoderError),
        ("decoder-only-lora", DecoderOnlyLoRAError),
    ),
)
def test_architecture_loading_errors_remain_fatal(
    tmp_path,
    monkeypatch,
    architecture,
    error_type,
):
    failing = _install_architecture_failure(
        monkeypatch,
        architecture,
        error_type,
        fail_during="prepare",
    )
    output = tmp_path / "release"

    with pytest.raises(error_type, match="artifact loading failed"):
        build_reference_release(
            _metadata(),
            CORPUS,
            tmp_path / "encoder",
            ENCODER_DIGEST,
            tmp_path / "decoder",
            DECODER_DIGEST,
            tmp_path / "snapshot",
            SNAPSHOT_DIGEST,
            output,
        )

    assert failing.prepare_calls == 1
    assert failing.realize_calls == 0
    assert not output.exists()


def test_dual_architecture_release_is_content_addressed_and_reproducible(
    tmp_path,
    monkeypatch,
):
    output, result = _build(tmp_path, monkeypatch)
    verified = read_verified_reference_release(output, result.manifest_sha256)

    assert result.manifest.no_quality_claim is True
    assert {item.architecture for item in result.manifest.architectures} == {
        "encoder-decoder",
        "decoder-only-lora",
    }
    assert {path for path, _ in verified.files} == REFERENCE_RELEASE_FILES
    assert len((output / "encoder-decoder-predictions.jsonl").read_text().splitlines()) == 6
    assert len((output / "decoder-only-lora-predictions.jsonl").read_text().splitlines()) == 6
    prediction = json.loads(
        (output / "encoder-decoder-predictions.jsonl").read_text().splitlines()[0]
    )
    assert prediction["outcome"] == "accepted"
    assert prediction["corpus_manifest_sha256"] == result.manifest.corpus_manifest_sha256
    assert "not a quality benchmark" in (output / "encoder-decoder-model-card.md").read_text()
    assert (
        "external artifact bytes are not included"
        in (output / "decoder-only-lora-model-card.md").read_text()
    )

    regenerated = verify_reference_release(
        output,
        result.manifest_sha256,
        corpus_release=CORPUS,
        encoder_bundle=tmp_path / "encoder",
        decoder_bundle=tmp_path / "decoder",
        decoder_model_snapshot=tmp_path / "snapshot",
        regenerate=True,
    )
    assert regenerated.manifest_sha256 == result.manifest_sha256


def test_release_verification_rejects_wrong_digest_and_changed_file(tmp_path, monkeypatch):
    output, result = _build(tmp_path, monkeypatch)

    with pytest.raises(ReferenceReleaseError, match="SHA-256 does not match"):
        read_verified_reference_release(output, "0" * 64)

    changed = tmp_path / "changed"
    shutil.copytree(output, changed)
    (changed / "encoder-decoder-model-card.md").write_text("changed")
    with pytest.raises(ReferenceReleaseError, match="file identity does not match"):
        read_verified_reference_release(changed, result.manifest_sha256)


def test_release_requires_exact_license_authorization_identity(tmp_path, monkeypatch):
    _install_release_seams(monkeypatch)
    metadata = _metadata()
    wrong_track = metadata.tracks[0].model_copy(
        update={
            "base_model": metadata.tracks[0].base_model.model_copy(update={"revision": "f" * 40})
        }
    )
    metadata = metadata.model_copy(update={"tracks": (wrong_track, metadata.tracks[1])})

    with pytest.raises(ReferenceReleaseError, match="authorization does not match"):
        build_reference_release(
            metadata,
            CORPUS,
            tmp_path / "encoder",
            ENCODER_DIGEST,
            tmp_path / "decoder",
            DECODER_DIGEST,
            tmp_path / "snapshot",
            SNAPSHOT_DIGEST,
            tmp_path / "release",
        )


def test_reference_release_schema_is_exposed_by_cli():
    result = CliRunner().invoke(app, ["schema", "reference-release"])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert schema["properties"]["schema_version"]["const"] == "ste-reference-artifact-release-v1"
    metadata = CliRunner().invoke(app, ["schema", "reference-release-metadata"])
    prediction = CliRunner().invoke(app, ["schema", "reference-prediction"])
    assert metadata.exit_code == prediction.exit_code == 0
    assert (
        json.loads(metadata.stdout)["properties"]["schema_version"]["const"]
        == "ste-reference-release-metadata-v1"
    )
    assert (
        json.loads(prediction.stdout)["properties"]["schema_version"]["const"]
        == "ste-reference-prediction-v1"
    )


def test_reference_release_build_and_verify_cli_commands(tmp_path, monkeypatch):
    _install_release_seams(monkeypatch)
    metadata = _metadata()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    output = tmp_path / "release"

    built = CliRunner().invoke(
        app,
        [
            "build-reference-release",
            str(metadata_path),
            str(CORPUS),
            str(tmp_path / "encoder"),
            ENCODER_DIGEST,
            str(tmp_path / "decoder"),
            DECODER_DIGEST,
            str(tmp_path / "snapshot"),
            SNAPSHOT_DIGEST,
            str(output),
            "--json",
        ],
    )

    assert built.exit_code == 0, built.output
    built_payload = json.loads(built.stdout)
    verified = CliRunner().invoke(
        app,
        [
            "verify-reference-release",
            str(output),
            built_payload["manifest_sha256"],
            "--json",
        ],
    )
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["status"] == "verified"


def test_release_manifest_digest_is_exact_file_hash(tmp_path, monkeypatch):
    output, result = _build(tmp_path, monkeypatch)

    assert (
        hashlib.sha256((output / "release-manifest.json").read_bytes()).hexdigest()
        == result.manifest_sha256
    )

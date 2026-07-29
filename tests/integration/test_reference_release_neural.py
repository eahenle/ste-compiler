from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from ste_compiler.reference_release import (
    ReferenceReleaseMetadataV1,
    ReferenceTrackAuthorizationV1,
    build_reference_release,
    read_verified_reference_release,
)
from ste_compiler.training import (
    DecoderOnlyLoRATrainingConfigV1,
    load_training_config,
    model_snapshot_manifest_sha256,
    prepare_decoder_smoke_fixture,
    read_training_release,
    run_decoder_lora_training_bundle,
    run_encoder_decoder_training_bundle,
)
from tests.integration.test_encoder_training import _config
from tests.neural_helpers import build_tiny_t5_snapshot

pytestmark = pytest.mark.neural

ROOT = Path(__file__).parents[2]
CORPUS = ROOT / "datasets/demonstration-corpus-1"
DECODER_CONFIG = ROOT / "data/training/decoder-only-lora-schema-example.yaml"


def _clean_source_checkout(root: Path) -> Path:
    source = root / "source-checkout"
    (source / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/ste_compiler", source / "src/ste_compiler")
    shutil.copy2(ROOT / "uv.lock", source / "uv.lock")
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "add",
            "src/ste_compiler",
            "uv.lock",
            "pyproject.toml",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=ste-compiler test",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture source",
        ],
        check=True,
    )
    return source


def test_real_dual_architecture_mechanics_release_uses_local_loaders(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def reject_network(*args, **kwargs):
        raise AssertionError("reference release attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    encoder_snapshot = build_tiny_t5_snapshot(tmp_path / "encoder-snapshot")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(encoder_snapshot),
    )
    source_checkout = _clean_source_checkout(tmp_path)
    _, encoder_config = _config(tmp_path)
    encoder_bundle = tmp_path / "encoder-bundle"
    encoder_result = run_encoder_decoder_training_bundle(
        encoder_config,
        CORPUS,
        encoder_bundle,
        source_root=source_checkout,
        dependency_lock=source_checkout / "uv.lock",
    )

    decoder_config = load_training_config(DECODER_CONFIG)
    assert isinstance(decoder_config, DecoderOnlyLoRATrainingConfigV1)
    corpus = read_training_release(CORPUS, decoder_config.corpus)
    decoder_snapshot = tmp_path / "decoder-snapshot"
    prepare_decoder_smoke_fixture(decoder_config, corpus, decoder_snapshot)
    snapshot_digest = model_snapshot_manifest_sha256(decoder_snapshot)
    decoder_bundle = tmp_path / "decoder-bundle"
    decoder_result = run_decoder_lora_training_bundle(
        decoder_config,
        corpus,
        decoder_snapshot,
        snapshot_digest,
        decoder_bundle,
        source_checkout=source_checkout,
        evaluation_command="ste-compiler evaluate-decoder-lora ...",
    )
    metadata = ReferenceReleaseMetadataV1(
        schema_version="ste-reference-release-metadata-v1",
        release_id="repository-synthetic-mechanics-1",
        intended_use="mechanics-smoke",
        no_quality_claim=True,
        tracks=(
            ReferenceTrackAuthorizationV1(
                architecture="encoder-decoder",
                base_model=encoder_config.base_model,
                base_model_origin="generated-by-tests.neural_helpers.build_tiny_t5_snapshot",
                base_model_license="MIT",
                artifact_license="MIT",
                redistribution="external-artifact-not-included",
            ),
            ReferenceTrackAuthorizationV1(
                architecture="decoder-only-lora",
                base_model=decoder_config.base_model,
                base_model_origin="generated-by-ste-compiler-prepare-decoder-smoke-fixture",
                base_model_license="MIT",
                artifact_license="MIT",
                redistribution="external-artifact-not-included",
            ),
        ),
    )
    release = tmp_path / "reference-release"

    result = build_reference_release(
        metadata,
        CORPUS,
        encoder_bundle,
        encoder_result.artifact_manifest_sha256,
        decoder_bundle,
        decoder_result.artifact_manifest_sha256,
        decoder_snapshot,
        snapshot_digest,
        release,
    )
    verified = read_verified_reference_release(release, result.manifest_sha256)

    assert verified.manifest.corpus_manifest_sha256 == corpus.manifest_sha256
    assert len((release / "encoder-decoder-predictions.jsonl").read_text().splitlines()) == 6
    assert len((release / "decoder-only-lora-predictions.jsonl").read_text().splitlines()) == 6
    assert {
        item.architecture: item.artifact_manifest_sha256 for item in verified.manifest.architectures
    } == {
        "encoder-decoder": encoder_result.artifact_manifest_sha256,
        "decoder-only-lora": decoder_result.artifact_manifest_sha256,
    }

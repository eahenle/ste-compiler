from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import ste_compiler.training.decoder_lora as decoder_training

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
from safetensors.torch import load_file, save_file

from ste_compiler.training import (
    DecoderLoRATrainingError,
    DecoderOnlyLoRATrainingConfigV1,
    evaluate_decoder_lora_adapter,
    load_training_config,
    model_snapshot_manifest_sha256,
    preflight_decoder_lora_artifact_bundle,
    prepare_decoder_smoke_fixture,
    read_training_release,
    read_verified_model_snapshot,
    run_decoder_lora_training,
    run_decoder_lora_training_bundle,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "data/training/decoder-only-lora-schema-example.yaml"
RELEASE = ROOT / "datasets/demonstration-corpus-1"
UNSAFE_SUFFIXES = {".bin", ".ckpt", ".joblib", ".pickle", ".pkl", ".pt", ".pth"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _offline(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def reject_network(*args, **kwargs):
        raise AssertionError("offline decoder training attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)


def _training_inputs():
    config = load_training_config(CONFIG)
    assert isinstance(config, DecoderOnlyLoRATrainingConfigV1)
    return config, read_training_release(RELEASE, config.corpus)


def _call_without_runtime_state_leak(operation):
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state().clone()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch_threads = torch.get_num_threads()

    result = operation()

    assert random.getstate() == python_random_state
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() is deterministic_warn_only
    assert torch.get_num_threads() == torch_threads
    return result


def test_decoder_lora_two_step_smoke_is_offline_deterministic_safe_and_reloadable(
    tmp_path,
    monkeypatch,
):
    _offline(monkeypatch)
    config, release = _training_inputs()
    model_snapshot = tmp_path / "model"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    snapshot_manifest = _call_without_runtime_state_leak(
        lambda: prepare_decoder_smoke_fixture(config, release, model_snapshot)
    )
    snapshot_digest = model_snapshot_manifest_sha256(model_snapshot)
    monkeypatch.setattr(
        decoder_training,
        "decoder_lora_artifact_manifest_sha256",
        lambda output: pytest.fail(
            f"bundle result rediscovered its digest after publication: {output}"
        ),
    )
    first_bundle = _call_without_runtime_state_leak(
        lambda: run_decoder_lora_training_bundle(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            first_output,
            source_checkout=ROOT,
            evaluation_command="ste-compiler evaluate-decoder-lora ...",
        )
    )
    first = first_bundle.run_manifest
    second = _call_without_runtime_state_leak(
        lambda: run_decoder_lora_training(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            second_output,
            source_checkout=ROOT,
            evaluation_command="ste-compiler evaluate-decoder-lora ...",
        )
    )

    assert snapshot_manifest.fixture_profile == "tiny-byte-bpe-gpt2-v1"
    assert first.optimizer_steps == second.optimizer_steps == 2
    assert first.sample_order == second.sample_order
    assert first.training_losses == second.training_losses
    assert first.validation_loss == second.validation_loss
    assert first.trainable_parameters > 0
    assert first.trainable_parameters < first.total_parameters
    assert (
        first.source.package_commit
        == subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert first.source.dependency_lock_sha256 == _sha256(ROOT / "uv.lock")
    assert first.source.dirty is False
    assert len(first.source.package_tree_sha256) == 64
    assert first.model_snapshot_manifest_sha256 == snapshot_digest
    dependency_names = {dependency.name for dependency in first.dependencies}
    assert {
        "accelerate",
        "huggingface-hub",
        "numpy",
        "peft",
        "safetensors",
        "ste-compiler",
        "tokenizers",
        "torch",
        "transformers",
    } <= dependency_names
    assert len(dependency_names) > 9
    assert _sha256(first_output / "adapter/adapter_model.safetensors") == _sha256(
        second_output / "adapter/adapter_model.safetensors"
    )
    assert (
        json.loads((first_output / "adapter/adapter_config.json").read_text())["revision"]
        == config.base_model.revision
    )
    assert {
        path.relative_to(first_output).as_posix()
        for path in first_output.rglob("*")
        if path.is_file()
    } == {
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "artifact-manifest.json",
        "checksums.sha256",
        "run-manifest.json",
        "training-config.json",
    }
    assert (
        not {path.suffix.casefold() for path in first_output.rglob("*") if path.is_file()}
        & UNSAFE_SUFFIXES
    )
    first_artifact_digest = first_bundle.artifact_manifest_sha256
    first_preflight = preflight_decoder_lora_artifact_bundle(
        first_output,
        first_artifact_digest,
    )
    assert first_preflight.run_manifest == first
    assert first_preflight.artifact_manifest_sha256 == first_artifact_digest
    assert (
        _call_without_runtime_state_leak(
            lambda: evaluate_decoder_lora_adapter(
                config,
                release,
                model_snapshot,
                snapshot_digest,
                first_output / "adapter",
            )
        )
        == first.validation_loss
    )
    manifest = json.loads((first_output / "run-manifest.json").read_text())
    assert manifest["schema_version"] == "ste-decoder-lora-run-v1"
    assert manifest["source"]["dirty"] is False
    assert manifest["hardware"]["device"] == "cpu"
    assert manifest["duration_seconds"] >= 0
    checksum_lines = (first_output / "checksums.sha256").read_text().splitlines()
    assert len(checksum_lines) == 5
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert digest == _sha256(first_output / relative)

    original_checksums = (first_output / "checksums.sha256").read_bytes()
    checksum_lines[0] = f"{'0' * 64}  {checksum_lines[0].split('  ', 1)[1]}"
    (first_output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    checksum_mismatch_digest = decoder_training._write_decoder_artifact_manifest(first_output)
    with pytest.raises(
        DecoderLoRATrainingError,
        match="checksums are not canonical or complete",
    ):
        preflight_decoder_lora_artifact_bundle(first_output, checksum_mismatch_digest)
    (first_output / "checksums.sha256").write_bytes(original_checksums)
    assert decoder_training._write_decoder_artifact_manifest(first_output) == first_artifact_digest

    adapter_weights = first_output / "adapter/adapter_model.safetensors"
    original_adapter_weights = adapter_weights.read_bytes()
    original_run_manifest = (first_output / "run-manifest.json").read_bytes()
    tensors = load_file(adapter_weights)
    removed_key = next(key for key in tensors if key.endswith(".lora_B.weight"))
    del tensors[removed_key]
    save_file(tensors, adapter_weights, metadata={"format": "pt"})
    run_manifest = decoder_training.DecoderLoRARunManifestV1.model_validate_json(
        original_run_manifest
    )
    adapter_identity = decoder_training._artifact(
        "adapter/adapter_model.safetensors",
        adapter_weights.read_bytes(),
    )
    run_manifest = run_manifest.model_copy(
        update={
            "output_artifacts": tuple(
                adapter_identity
                if artifact.path == "adapter/adapter_model.safetensors"
                else artifact
                for artifact in run_manifest.output_artifacts
            )
        }
    )
    (first_output / "run-manifest.json").write_bytes(
        decoder_training.canonical_decoder_lora_run_manifest_json(run_manifest)
    )
    decoder_training._write_checksums(first_output)
    unpaired_weights_digest = decoder_training._write_decoder_artifact_manifest(first_output)
    with pytest.raises(
        DecoderLoRATrainingError,
        match="complete LoRA A/B pairs",
    ):
        preflight_decoder_lora_artifact_bundle(first_output, unpaired_weights_digest)
    adapter_weights.write_bytes(original_adapter_weights)
    (first_output / "run-manifest.json").write_bytes(original_run_manifest)
    (first_output / "checksums.sha256").write_bytes(original_checksums)
    assert decoder_training._write_decoder_artifact_manifest(first_output) == first_artifact_digest

    linked_adapter = tmp_path / "linked-adapter"
    linked_adapter.symlink_to(first_output / "adapter", target_is_directory=True)
    with pytest.raises(DecoderLoRATrainingError, match="must be a real directory"):
        evaluate_decoder_lora_adapter(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            linked_adapter,
        )

    adapter_config = first_output / "adapter/adapter_config.json"
    tampered_config = json.loads(adapter_config.read_text())
    tampered_config["r"] = config.lora.rank + 1
    adapter_config.write_text(json.dumps(tampered_config))
    with pytest.raises(
        DecoderLoRATrainingError,
        match="artifact bundle verification failed",
    ):
        preflight_decoder_lora_artifact_bundle(first_output, first_artifact_digest)
    with pytest.raises(
        DecoderLoRATrainingError,
        match="does not match training LoRA fields: r",
    ):
        evaluate_decoder_lora_adapter(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            first_output / "adapter",
        )


def test_decoder_bundle_publication_rejects_stage_modification_after_preflight(
    tmp_path,
    monkeypatch,
):
    _offline(monkeypatch)
    config, release = _training_inputs()
    model_snapshot = tmp_path / "model"
    output = tmp_path / "changed-stage"
    prepare_decoder_smoke_fixture(config, release, model_snapshot)
    snapshot_digest = model_snapshot_manifest_sha256(model_snapshot)
    staged_digests = []
    real_write_manifest = decoder_training._write_decoder_artifact_manifest
    real_fsync_tree = decoder_training._fsync_tree

    def capture_manifest_digest(root):
        digest = real_write_manifest(root)
        staged_digests.append(digest)
        return digest

    def modify_after_fsync(root):
        real_fsync_tree(root)
        manifest = root / decoder_training.ARTIFACT_MANIFEST_NAME
        if manifest.is_file():
            manifest.write_bytes(b"changed after staged preflight\n")

    monkeypatch.setattr(
        decoder_training,
        "_write_decoder_artifact_manifest",
        capture_manifest_digest,
    )
    monkeypatch.setattr(decoder_training, "_fsync_tree", modify_after_fsync)

    with pytest.raises(
        DecoderLoRATrainingError,
        match="decoder artifact bundle verification failed",
    ):
        run_decoder_lora_training_bundle(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            output,
            source_checkout=ROOT,
            evaluation_command="ste-compiler evaluate-decoder-lora ...",
        )

    assert len(staged_digests) == 1
    assert len(staged_digests[0]) == 64
    assert not output.exists()
    assert not list(tmp_path.glob(".changed-stage.stage-*"))


def test_decoder_removes_invalid_output_after_post_rename_verification_failure(
    tmp_path,
    monkeypatch,
):
    _offline(monkeypatch)
    config, release = _training_inputs()
    model_snapshot = tmp_path / "model"
    output = tmp_path / "invalid-published"
    prepare_decoder_smoke_fixture(config, release, model_snapshot)
    snapshot_digest = model_snapshot_manifest_sha256(model_snapshot)
    real_verify = decoder_training.verify_artifact_bundle

    def mutate_published_then_verify(root, expected_manifest_sha256):
        if root == output:
            (root / decoder_training.ARTIFACT_MANIFEST_NAME).write_bytes(b"changed after rename\n")
        return real_verify(root, expected_manifest_sha256)

    monkeypatch.setattr(
        decoder_training,
        "verify_artifact_bundle",
        mutate_published_then_verify,
    )

    with pytest.raises(
        DecoderLoRATrainingError,
        match="published artifact bundle does not match",
    ):
        run_decoder_lora_training_bundle(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            output,
            source_checkout=ROOT,
            evaluation_command="ste-compiler evaluate-decoder-lora ...",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-published.stage-*"))


def test_decoder_fixture_is_content_bound_and_refuses_existing_output(
    tmp_path,
    monkeypatch,
):
    _offline(monkeypatch)
    config, release = _training_inputs()
    model_snapshot = tmp_path / "model"
    prepare_decoder_smoke_fixture(config, release, model_snapshot)
    snapshot_digest = model_snapshot_manifest_sha256(model_snapshot)

    marker = model_snapshot / "keep.txt"
    marker.write_text("do not overwrite")
    with pytest.raises(DecoderLoRATrainingError, match="output path already exists"):
        prepare_decoder_smoke_fixture(config, release, model_snapshot)
    assert marker.read_text() == "do not overwrite"
    marker.unlink()

    weights = model_snapshot / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(
        DecoderLoRATrainingError,
        match="artifact size does not match its manifest",
    ):
        read_verified_model_snapshot(model_snapshot, config, snapshot_digest)

    with pytest.raises(
        DecoderLoRATrainingError,
        match="manifest SHA-256 does not match",
    ):
        read_verified_model_snapshot(model_snapshot, config, "0" * 64)


def test_decoder_smoke_rejects_non_two_step_configuration(tmp_path, monkeypatch):
    _offline(monkeypatch)
    config, release = _training_inputs()
    config = config.model_copy(update={"max_steps": 3})

    with pytest.raises(DecoderLoRATrainingError, match="requires max_steps=2"):
        prepare_decoder_smoke_fixture(config, release, tmp_path / "model")


def test_installed_wheel_runs_decoder_training_and_evaluation_offline(tmp_path):
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_directory),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_directory.glob("ste_compiler-*.whl"))
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
    config = installed / "ste_compiler/data/training/decoder-only-lora-schema-example.yaml"
    model_snapshot = tmp_path / "model"
    run = tmp_path / "run"
    tripwire = tmp_path / "tripwire"
    tripwire.mkdir()
    (tripwire / "sitecustomize.py").write_text(
        "import socket\n"
        "def _reject(*args, **kwargs):\n"
        "    raise RuntimeError('network access denied by installed-wheel test')\n"
        "socket.socket.connect = _reject\n"
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join((str(tripwire), str(installed))),
    }
    command = [sys.executable, "-m", "ste_compiler.cli"]

    prepared = subprocess.run(
        [
            *command,
            "prepare-decoder-smoke-fixture",
            str(config),
            str(RELEASE),
            str(model_snapshot),
            "--json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    trained = subprocess.run(
        [
            *command,
            "train-decoder-lora",
            str(config),
            str(RELEASE),
            str(model_snapshot),
            json.loads(prepared.stdout)["manifest_sha256"],
            str(run),
            "--source-checkout",
            str(ROOT),
            "--json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    trained_payload = json.loads(trained.stdout)
    preflight = subprocess.run(
        [
            *command,
            "preflight-artifact",
            str(run),
            "--manifest-sha256",
            trained_payload["artifact_manifest_sha256"],
            "--json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    evaluated = subprocess.run(
        [
            *command,
            "evaluate-decoder-lora",
            str(config),
            str(RELEASE),
            str(model_snapshot),
            json.loads(prepared.stdout)["manifest_sha256"],
            str(run / "adapter"),
            "--json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(prepared.stdout)["fixture_profile"] == "tiny-byte-bpe-gpt2-v1"
    assert len(json.loads(prepared.stdout)["manifest_sha256"]) == 64
    preflight_payload = json.loads(preflight.stdout)
    evaluated_payload = json.loads(evaluated.stdout)
    assert trained_payload["optimizer_steps"] == 2
    assert preflight_payload["architecture"] == "decoder-only-lora"
    assert preflight_payload["validation_profile"] == "decoder-adapter-structure-v1"
    assert preflight_payload["network_access"] == "none"
    assert evaluated_payload["validation_loss"] == trained_payload["validation_loss"]
    assert (run / "adapter/adapter_model.safetensors").is_file()

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from ste_compiler.training import (
    DecoderLoRATrainingError,
    DecoderOnlyLoRATrainingConfigV1,
    evaluate_decoder_lora_adapter,
    load_training_config,
    model_snapshot_manifest_sha256,
    prepare_decoder_smoke_fixture,
    read_training_release,
    read_verified_model_snapshot,
    run_decoder_lora_training,
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


def test_decoder_lora_two_step_smoke_is_offline_deterministic_safe_and_reloadable(
    tmp_path,
    monkeypatch,
):
    _offline(monkeypatch)
    config, release = _training_inputs()
    model_snapshot = tmp_path / "model"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    snapshot_manifest = prepare_decoder_smoke_fixture(config, release, model_snapshot)
    snapshot_digest = model_snapshot_manifest_sha256(model_snapshot)
    first = run_decoder_lora_training(
        config,
        release,
        model_snapshot,
        snapshot_digest,
        first_output,
        source_checkout=ROOT,
        evaluation_command="ste-compiler evaluate-decoder-lora ...",
    )
    second = run_decoder_lora_training(
        config,
        release,
        model_snapshot,
        snapshot_digest,
        second_output,
        source_checkout=ROOT,
        evaluation_command="ste-compiler evaluate-decoder-lora ...",
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
    assert {dependency.name for dependency in first.dependencies} == {
        "huggingface-hub",
        "peft",
        "safetensors",
        "ste-compiler",
        "tokenizers",
        "torch",
        "transformers",
    }
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
        "checksums.sha256",
        "run-manifest.json",
        "training-config.json",
    }
    assert (
        not {path.suffix.casefold() for path in first_output.rglob("*") if path.is_file()}
        & UNSAFE_SUFFIXES
    )
    assert (
        evaluate_decoder_lora_adapter(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            first_output / "adapter",
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
        match="does not match training LoRA fields: r",
    ):
        evaluate_decoder_lora_adapter(
            config,
            release,
            model_snapshot,
            snapshot_digest,
            first_output / "adapter",
        )


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
    trained_payload = json.loads(trained.stdout)
    evaluated_payload = json.loads(evaluated.stdout)
    assert trained_payload["optimizer_steps"] == 2
    assert evaluated_payload["validation_loss"] == trained_payload["validation_loss"]
    assert (run / "adapter/adapter_model.safetensors").is_file()

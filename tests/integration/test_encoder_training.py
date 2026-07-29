from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from ste_compiler.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactFileV1,
    artifact_manifest_sha256,
    build_artifact_manifest,
    canonical_artifact_manifest_json,
    parse_canonical_artifact_manifest,
)
from ste_compiler.realizer import (
    EncoderDecoderError,
    EncoderDecoderLocalBundleConfig,
    EncoderDecoderLocalBundleRealizerConfigV1,
    TransformersEncoderDecoderSymbolGenerator,
    build_realizer,
)
from ste_compiler.realizer import encoder_decoder as runtime_module
from ste_compiler.training import (
    EncoderDecoderTrainingConfigV1,
    FileIdentityV1,
    canonical_run_manifest_json,
    encoder_decoder_artifact_manifest_sha256,
    evaluate_encoder_decoder_checkpoint,
    load_training_config,
    open_verified_encoder_decoder_artifact_bundle,
    preflight_encoder_decoder_artifact_bundle,
    run_encoder_decoder_training,
    run_encoder_decoder_training_bundle,
    verify_safe_encoder_decoder_checkpoint,
)
from ste_compiler.training import encoder_decoder as training_module
from ste_compiler.training.encoder_decoder import EncoderDecoderTrainingError
from tests.neural_helpers import (
    FIXTURE_REPO_ID,
    FIXTURE_REVISION,
    build_tiny_t5_snapshot,
)

pytestmark = pytest.mark.neural

ROOT = Path(__file__).parents[2]
RELEASE = ROOT / "datasets/demonstration-corpus-1"


def _config(tmp_path: Path) -> tuple[Path, EncoderDecoderTrainingConfigV1]:
    payload = json.loads(
        json.dumps(
            {
                "schema_version": "ste-training-config-v1",
                "architecture": "encoder-decoder",
                "corpus": {
                    "dataset_version": "demonstration-corpus-1",
                    "manifest_sha256": (
                        "f6ae4582669c4d7d06e33018088b900ffa0f8aa8b6e0d9f1beeccca2023faa7b"
                    ),
                    "train_sha256": (
                        "1772fbe01a15c28d174e139f93e5c3b0fd6744c01cf5c81b79fb842c9609ebd0"
                    ),
                    "validation_sha256": (
                        "ea16d6bae1f624c26581e05b02cb693282805d93f945bd6e00e73a48a79d15dd"
                    ),
                },
                "base_model": {
                    "repo_id": FIXTURE_REPO_ID,
                    "revision": FIXTURE_REVISION,
                },
                "tokenizer": {
                    "repo_id": FIXTURE_REPO_ID,
                    "revision": FIXTURE_REVISION,
                },
                "seed": 1729,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": {"learning_rate": 0.0001, "weight_decay": 0},
                "strategy": "full",
                "max_source_tokens": 8192,
                "max_target_tokens": 2048,
            }
        )
    )
    config_path = tmp_path / "encoder-training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_training_config(config_path)
    assert isinstance(config, EncoderDecoderTrainingConfigV1)
    return config_path, config


@pytest.fixture
def tiny_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = build_tiny_t5_snapshot(tmp_path / "snapshot")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    def block_network(self, address):
        raise AssertionError(f"unexpected network access: {address}")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    return snapshot, calls


def _weight_hash(checkpoint: Path) -> str:
    weight = next(checkpoint.glob("*.safetensors"))
    return hashlib.sha256(weight.read_bytes()).hexdigest()


def _run_manifest_hash(checkpoint: Path) -> str:
    return hashlib.sha256((checkpoint / "run-manifest.json").read_bytes()).hexdigest()


def _file_identities(checkpoint: Path) -> tuple[FileIdentityV1, ...]:
    return tuple(
        FileIdentityV1(
            path=path.relative_to(checkpoint).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
        )
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != ARTIFACT_MANIFEST_NAME
    )


def _rewrite_artifact_manifest(checkpoint: Path) -> str:
    identities = _file_identities(checkpoint)
    manifest = build_artifact_manifest(
        architecture="encoder-decoder",
        artifact_type="encoder-decoder-checkpoint",
        entrypoint=".",
        files=tuple(
            ArtifactFileV1(
                path=identity.path,
                sha256=identity.sha256,
                bytes=identity.bytes,
            )
            for identity in identities
        ),
    )
    (checkpoint / ARTIFACT_MANIFEST_NAME).write_bytes(canonical_artifact_manifest_json(manifest))
    return artifact_manifest_sha256(manifest)


def _call_without_runtime_state_leak(operation):
    import numpy
    import torch

    python_random_state = random.getstate()
    numpy_random_state = numpy.random.get_state()
    torch_random_state = torch.get_rng_state().clone()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch_threads = torch.get_num_threads()

    result = operation()

    assert random.getstate() == python_random_state
    assert all(
        left == right if isinstance(left, str | int | float) else (left == right).all()
        for left, right in zip(numpy.random.get_state(), numpy_random_state, strict=True)
    )
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() is deterministic_warn_only
    assert torch.get_num_threads() == torch_threads
    return result


def test_offline_two_step_trainer_is_deterministic_safe_and_reloadable(
    tmp_path,
    tiny_snapshot,
    monkeypatch,
):
    snapshot, calls = tiny_snapshot
    (snapshot / "README.md").write_text("cached model card", encoding="utf-8")
    (snapshot / ".gitattributes").write_text("*.bin filter=lfs", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"unsafe legacy weights")
    nested_tokenizer = snapshot / "tokenizer"
    nested_tokenizer.mkdir()
    shutil.copyfile(snapshot / "tokenizer.json", nested_tokenizer / "vocab.json")
    _, config = _config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = _call_without_runtime_state_leak(
        lambda: run_encoder_decoder_training(
            config,
            RELEASE,
            first,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )
    )
    second_manifest = _call_without_runtime_state_leak(
        lambda: run_encoder_decoder_training(
            config,
            RELEASE,
            second,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )
    )

    assert first_manifest.optimizer_steps == 2
    assert first_manifest.micro_steps == 2
    assert first_manifest.record_order == second_manifest.record_order
    assert first_manifest.optimizer_losses == second_manifest.optimizer_losses
    assert _weight_hash(first) == _weight_hash(second)
    assert (
        first_manifest.package.source_commit
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert (
        first_manifest.package.dependency_lock.sha256
        == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    dependency_names = dict(first_manifest.package.dependencies)
    assert {
        "huggingface-hub",
        "numpy",
        "safetensors",
        "ste-compiler",
        "tokenizers",
        "torch",
        "transformers",
    } <= dependency_names.keys()
    assert len(dependency_names) > 8
    assert first_manifest.hardware.device == "cpu"
    assert first_manifest.duration_seconds >= 0
    assert first_manifest.corpus.manifest_sha256 == config.corpus.manifest_sha256
    assert first_manifest.base_model_artifacts == first_manifest.tokenizer_artifacts
    assert {artifact.path for artifact in first_manifest.base_model_artifacts} >= {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    artifact_paths = tuple(artifact.path for artifact in first_manifest.base_model_artifacts)
    assert artifact_paths == tuple(sorted(artifact_paths))
    assert "tokenizer/vocab.json" in artifact_paths
    assert {"README.md", ".gitattributes", "pytorch_model.bin"}.isdisjoint(artifact_paths)
    assert first_manifest.package.source_dirty is False
    assert len(first_manifest.package.source_tree_sha256) == 64
    assert first_manifest.validation.record_count == 2
    assert len(calls) == 2
    assert all(
        call
        == {
            "repo_id": FIXTURE_REPO_ID,
            "revision": FIXTURE_REVISION,
            "local_files_only": True,
            "allow_patterns": [
                "*.codes",
                "*.json",
                "*.merges",
                "*.model",
                "*.safetensors",
                "*.spm",
                "*.tiktoken",
                "*.tokenizer",
                "*.txt",
                "*.vocab",
            ],
        }
        for call in calls
    )
    assert not list(tmp_path.glob(".first.stage-*"))
    assert not list(tmp_path.glob(".second.stage-*"))
    assert not any(
        path.suffix in {".bin", ".ckpt", ".pickle", ".pkl", ".pt", ".pth"}
        for path in first.rglob("*")
    )
    assert verify_safe_encoder_decoder_checkpoint(first)
    artifact_digest = encoder_decoder_artifact_manifest_sha256(first)
    artifact_manifest = parse_canonical_artifact_manifest(
        (first / ARTIFACT_MANIFEST_NAME).read_bytes()
    )
    preflight = preflight_encoder_decoder_artifact_bundle(first, artifact_digest)

    assert artifact_digest == artifact_manifest_sha256(artifact_manifest)
    assert preflight.artifact_manifest_sha256 == artifact_digest
    assert preflight.run_manifest == first_manifest
    assert artifact_manifest.architecture == "encoder-decoder"
    assert artifact_manifest.artifact_type == "encoder-decoder-checkpoint"
    assert artifact_manifest.entrypoint == "."
    assert artifact_manifest.run_manifest_sha256 == _run_manifest_hash(first)
    assert ARTIFACT_MANIFEST_NAME not in {identity.path for identity in artifact_manifest.files}
    assert {identity.path for identity in artifact_manifest.files} == {
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file() and path.name != ARTIFACT_MANIFEST_NAME
    }
    with open_verified_encoder_decoder_artifact_bundle(
        first,
        artifact_digest,
    ) as verified:
        scoped_path = verified.checkpoint_path
        assert scoped_path != first
        assert scoped_path.is_dir()
        assert verified.run_manifest == first_manifest
        assert verified.artifact_manifest_sha256 == artifact_digest
        assert verified.run_manifest_sha256 == _run_manifest_hash(first)
    assert not scoped_path.exists()

    generic_open_count = 0
    real_generic_open = training_module.open_verified_artifact_bundle
    real_import_module = runtime_module.import_module
    import transformers

    loaded_private_paths = []
    real_tokenizer_loader = transformers.AutoTokenizer.from_pretrained
    real_model_loader = transformers.AutoModelForSeq2SeqLM.from_pretrained

    def capture_tokenizer_path(path, **kwargs):
        loaded_private_paths.append(Path(path))
        return real_tokenizer_loader(path, **kwargs)

    def capture_model_path(path, **kwargs):
        loaded_private_paths.append(Path(path))
        return real_model_loader(path, **kwargs)

    @contextmanager
    def count_generic_open(root, expected_digest):
        nonlocal generic_open_count
        generic_open_count += 1
        with real_generic_open(root, expected_digest) as verified:
            yield verified

    def local_runtime_import(name):
        if name == "huggingface_hub":
            raise AssertionError("local bundle runtime must not import Hugging Face Hub")
        return real_import_module(name)

    monkeypatch.setattr(training_module, "open_verified_artifact_bundle", count_generic_open)
    monkeypatch.setattr(runtime_module, "import_module", local_runtime_import)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(capture_tokenizer_path),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        staticmethod(capture_model_path),
    )
    local_realizer = build_realizer(
        EncoderDecoderLocalBundleRealizerConfigV1(
            schema_version="ste-realizer-config-v1",
            architecture="encoder-decoder-local-bundle",
            artifact_manifest_sha256=artifact_digest,
            intended_use="mechanics-smoke",
        ),
        artifact_bundle=first,
    )
    local_generator = local_realizer._delegate.generator
    tokenizer, model = local_generator._get_components()

    assert tokenizer is not None
    assert model is not None
    assert local_generator._get_components() == (tokenizer, model)
    assert generic_open_count == 1
    assert len(loaded_private_paths) == 2
    assert loaded_private_paths[0] == loaded_private_paths[1]
    assert loaded_private_paths[0] != first
    assert not loaded_private_paths[0].exists()
    generated = model.generate(
        **tokenizer("{}", return_tensors="pt"),
        do_sample=False,
        max_new_tokens=1,
    )
    assert generated.shape[0] == 1
    assert local_generator.model_id == f"ste-artifact-bundle:sha256:{artifact_digest}"
    assert local_generator.artifact_manifest_sha256 == artifact_digest
    assert local_generator.run_manifest_sha256 == _run_manifest_hash(first)
    assert local_generator.artifact_intended_use == "mechanics-smoke"
    assert local_realizer._artifact_mode == "content-addressed-local-bundle"

    wrong_digest_generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderLocalBundleConfig(
            artifact_bundle=first,
            artifact_manifest_sha256="0" * 64,
        )
    )
    with pytest.raises(
        EncoderDecoderError,
        match="artifact verification failed",
    ):
        wrong_digest_generator._get_components()

    with pytest.raises(
        EncoderDecoderTrainingError,
        match="manifest SHA-256 does not match",
    ):
        preflight_encoder_decoder_artifact_bundle(first, "0" * 64)

    inventory_mismatch = shutil.copytree(first, tmp_path / "inventory-mismatch")
    mismatched_run = first_manifest.model_copy(
        update={"output_artifacts": first_manifest.output_artifacts[:-1]}
    )
    (inventory_mismatch / "run-manifest.json").write_bytes(
        canonical_run_manifest_json(mismatched_run)
    )
    mismatch_digest = _rewrite_artifact_manifest(inventory_mismatch)
    with pytest.raises(
        EncoderDecoderTrainingError,
        match="does not match its run manifest",
    ):
        preflight_encoder_decoder_artifact_bundle(inventory_mismatch, mismatch_digest)
    tampered_generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderLocalBundleConfig(
            artifact_bundle=inventory_mismatch,
            artifact_manifest_sha256=mismatch_digest,
        )
    )
    with pytest.raises(
        EncoderDecoderError,
        match="does not match its run manifest",
    ):
        tampered_generator._get_components()

    unsafe_weights = shutil.copytree(first, tmp_path / "unsafe-weights")
    (unsafe_weights / ARTIFACT_MANIFEST_NAME).unlink()
    weight = next(unsafe_weights.glob("*.safetensors"))
    weight.rename(weight.with_suffix(".safeweights"))
    unsafe_output_artifacts = tuple(
        identity
        for identity in _file_identities(unsafe_weights)
        if identity.path != "run-manifest.json"
    )
    unsafe_run = first_manifest.model_copy(update={"output_artifacts": unsafe_output_artifacts})
    (unsafe_weights / "run-manifest.json").write_bytes(canonical_run_manifest_json(unsafe_run))
    unsafe_digest = _rewrite_artifact_manifest(unsafe_weights)
    with pytest.raises(
        EncoderDecoderTrainingError,
        match="does not contain safetensors weights",
    ):
        preflight_encoder_decoder_artifact_bundle(unsafe_weights, unsafe_digest)

    with pytest.raises(EncoderDecoderTrainingError, match="manifest digest does not match"):
        evaluate_encoder_decoder_checkpoint(config, RELEASE, first, "0" * 64)

    reevaluated = _call_without_runtime_state_leak(
        lambda: evaluate_encoder_decoder_checkpoint(
            config,
            RELEASE,
            first,
            _run_manifest_hash(first),
        )
    )

    assert reevaluated == first_manifest.validation

    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    relocated_checkpoint = shutil.copytree(first, relocated_root / "checkpoint")
    relocated_release = shutil.copytree(RELEASE, relocated_root / "release")
    assert (
        _call_without_runtime_state_leak(
            lambda: evaluate_encoder_decoder_checkpoint(
                config,
                relocated_release,
                relocated_checkpoint,
                _run_manifest_hash(relocated_checkpoint),
            )
        )
        == first_manifest.validation
    )

    (second / "validation-metrics.json").write_bytes(b"{}\n")
    with pytest.raises(EncoderDecoderTrainingError, match="manifest identity does not match"):
        evaluate_encoder_decoder_checkpoint(
            config,
            RELEASE,
            second,
            _run_manifest_hash(second),
        )


def test_separate_process_runs_reproduce_order_loss_and_weights(tmp_path, tiny_snapshot):
    snapshot, _ = tiny_snapshot
    config_path, _ = _config(tmp_path)
    script = """
import hashlib
import json
import pathlib
import socket
import sys

import huggingface_hub

snapshot, config_path, release, output, source, lock = map(pathlib.Path, sys.argv[1:])
huggingface_hub.snapshot_download = lambda **kwargs: str(snapshot)
socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
    AssertionError(f"unexpected network access: {address}")
)

from ste_compiler.training import load_training_config, run_encoder_decoder_training

config = load_training_config(config_path)
manifest = run_encoder_decoder_training(
    config,
    release,
    output,
    source_root=source,
    dependency_lock=lock,
)
weight = next(output.glob("*.safetensors"))
print(json.dumps({
    "record_order": manifest.record_order,
    "optimizer_losses": manifest.optimizer_losses,
    "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    results = []
    for name in ("process-first", "process-second"):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(snapshot),
                str(config_path),
                str(RELEASE),
                str(tmp_path / name),
                str(ROOT),
                str(ROOT / "uv.lock"),
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout))

    assert results[0] == results[1]


def test_failed_checkpoint_save_leaves_no_partial_output(tmp_path, tiny_snapshot, monkeypatch):
    _, config = _config(tmp_path)
    output = tmp_path / "failed"

    def fail_save(model, stage):
        raise OSError("injected save failure")

    monkeypatch.setattr(training_module, "_safe_save_pretrained", fail_save)

    with pytest.raises(OSError, match="injected save failure"):
        run_encoder_decoder_training(
            config,
            RELEASE,
            output,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".failed.stage-*"))


def test_bundle_publication_rejects_stage_swap_after_preflight(
    tmp_path,
    tiny_snapshot,
    monkeypatch,
):
    _, config = _config(tmp_path)
    output = tmp_path / "bundle-result"
    staged_digests = []
    displaced_stage = tmp_path / "displaced-stage"
    real_preflight = training_module.preflight_encoder_decoder_artifact_bundle

    def verify_then_swap_stage(root, expected_manifest_sha256):
        result = real_preflight(root, expected_manifest_sha256)
        staged_digests.append(result.artifact_manifest_sha256)
        root.rename(displaced_stage)
        root.mkdir()
        (root / "counterfeit").write_bytes(b"not the verified stage")
        return result

    monkeypatch.setattr(
        training_module,
        "preflight_encoder_decoder_artifact_bundle",
        verify_then_swap_stage,
    )

    with pytest.raises(
        EncoderDecoderTrainingError,
        match="changed during staged artifact publication",
    ):
        run_encoder_decoder_training_bundle(
            config,
            RELEASE,
            output,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )

    assert len(staged_digests) == 1
    assert len(staged_digests[0]) == 64
    assert displaced_stage.is_dir()
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle-result.stage-*"))


def test_encoder_removes_invalid_output_after_post_rename_verification_failure(
    tmp_path,
    tiny_snapshot,
    monkeypatch,
):
    _, config = _config(tmp_path)
    output = tmp_path / "invalid-published"
    real_verify = training_module.verify_artifact_bundle

    def mutate_published_then_verify(root, expected_manifest_sha256):
        if root == output:
            (root / training_module.ARTIFACT_MANIFEST_NAME).write_bytes(b"changed after rename\n")
        return real_verify(root, expected_manifest_sha256)

    monkeypatch.setattr(
        training_module,
        "verify_artifact_bundle",
        mutate_published_then_verify,
    )

    with pytest.raises(
        EncoderDecoderTrainingError,
        match="published artifact bundle does not match",
    ):
        run_encoder_decoder_training_bundle(
            config,
            RELEASE,
            output,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-published.stage-*"))


def test_installed_wheel_runs_offline_encoder_training(tmp_path, tiny_snapshot):
    snapshot, _ = tiny_snapshot
    config_path, _ = _config(tmp_path)
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
    output = tmp_path / "wheel-output"
    script = """
import json
import pathlib
import socket
import sys

import huggingface_hub

installed, snapshot, config, release, output, source, lock = map(pathlib.Path, sys.argv[1:])
huggingface_hub.snapshot_download = lambda **kwargs: str(snapshot)
socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
    AssertionError(f"unexpected network access: {address}")
)

import ste_compiler
from ste_compiler.cli import app
from ste_compiler.realizer import EncoderDecoderLocalBundleRealizerConfigV1, build_realizer
from typer.testing import CliRunner

assert pathlib.Path(ste_compiler.__file__).is_relative_to(installed)
runner = CliRunner()
trained = runner.invoke(
    app,
    [
        "train-encoder-decoder",
        str(config),
        str(release),
        "--output",
        str(output),
        "--source-root",
        str(source),
        "--dependency-lock",
        str(lock),
        "--json",
    ],
)
assert trained.exit_code == 0, trained.output
manifest = json.loads(trained.stdout)
assert manifest["optimizer_steps"] == 2
manifest_sha256 = manifest["run_manifest_sha256"]
artifact_manifest_sha256 = manifest["artifact_manifest_sha256"]
preflight = runner.invoke(
    app,
    [
        "preflight-artifact",
        str(output),
        "--manifest-sha256",
        artifact_manifest_sha256,
        "--json",
    ],
)
assert preflight.exit_code == 0, preflight.output
preflight_payload = json.loads(preflight.stdout)
assert preflight_payload["architecture"] == "encoder-decoder"
assert preflight_payload["validation_profile"] == "encoder-checkpoint-load-v1"
assert preflight_payload["network_access"] == "none"
evaluated = runner.invoke(
    app,
    [
        "evaluate-encoder-decoder-checkpoint",
        str(output / "training-config.json"),
        str(release),
        str(output),
        "--run-manifest-sha256",
        manifest_sha256,
        "--json",
    ],
)
assert evaluated.exit_code == 0, evaluated.output
assert json.loads(evaluated.stdout)["record_count"] == 2
local_realizer = build_realizer(
    EncoderDecoderLocalBundleRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="encoder-decoder-local-bundle",
        artifact_manifest_sha256=artifact_manifest_sha256,
        intended_use="mechanics-smoke",
    ),
    artifact_bundle=output,
)
local_generator = local_realizer._delegate.generator
tokenizer, model = local_generator._get_components()
generated = model.generate(
    **tokenizer("{}", return_tensors="pt"),
    do_sample=False,
    max_new_tokens=1,
)
assert generated.shape[0] == 1
assert local_generator.model_id == (
    f"ste-artifact-bundle:sha256:{artifact_manifest_sha256}"
)
assert local_generator.artifact_manifest_sha256 == artifact_manifest_sha256
assert local_generator.run_manifest_sha256 == manifest_sha256
assert local_generator.artifact_intended_use == "mechanics-smoke"
assert local_realizer._artifact_mode == "content-addressed-local-bundle"
"""
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(installed),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(installed),
            str(snapshot),
            str(config_path),
            str(RELEASE),
            str(output),
            str(ROOT),
            str(ROOT / "uv.lock"),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "run-manifest.json").is_file()

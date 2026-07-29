from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ste_compiler.realizer import (
    LocalDecoderOnlyLoRAError,
    LocalDecoderOnlyLoRARuntimeConfig,
    load_local_decoder_lora_generator,
    local_decoder,
)
from ste_compiler.training import ArtifactIdentityV1, DecoderLoRATrainingError

BASE_REVISION = "0123456789abcdef0123456789abcdef01234567"
ARTIFACT_DIGEST = "a" * 64
RUN_DIGEST = "b" * 64
SNAPSHOT_DIGEST = "c" * 64
BASE_MODEL = ArtifactIdentityV1(repo_id="org/tiny-base", revision=BASE_REVISION)
TOKENIZER = ArtifactIdentityV1(repo_id="org/tiny-base", revision=BASE_REVISION)
SNAPSHOT_ARTIFACTS = (SimpleNamespace(path="model.safetensors", sha256="d" * 64, bytes=8),)


def _config(**updates):
    values = {
        "artifact_bundle": Path("/deployment/adapter"),
        "artifact_manifest_sha256": ARTIFACT_DIGEST,
        "model_snapshot": Path("/deployment/base"),
        "model_snapshot_manifest_sha256": SNAPSHOT_DIGEST,
        "base_model": BASE_MODEL,
        "tokenizer": TOKENIZER,
        "intended_use": "mechanics-smoke",
        "prompt_profile": "decoder-only-symbol-plan-v1",
        "max_new_tokens": 32,
        "max_symbols": 4,
    }
    values.update(updates)
    return LocalDecoderOnlyLoRARuntimeConfig(**values)


def _run_manifest(**updates):
    values = {
        "prompt_profile": "decoder-only-symbol-plan-v1",
        "training_config": SimpleNamespace(
            base_model=BASE_MODEL,
            tokenizer=TOKENIZER,
        ),
        "model_snapshot_manifest_sha256": SNAPSHOT_DIGEST,
        "model_snapshot_artifacts": SNAPSHOT_ARTIFACTS,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _snapshot(**updates):
    values = {
        "manifest": SimpleNamespace(artifacts=SNAPSHOT_ARTIFACTS),
        "manifest_sha256": SNAPSHOT_DIGEST,
        "files": (
            ("config.json", b"{}"),
            ("model.safetensors", b"safe"),
            ("tokenizer.json", b"{}"),
        ),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_local_decoder_loads_only_private_paths_and_cleans_materializations(
    monkeypatch,
):
    captured_paths = []
    calls = []

    @contextmanager
    def verified_bundle(root, digest):
        assert root == Path("/deployment/adapter")
        assert digest == ARTIFACT_DIGEST
        with tempfile.TemporaryDirectory(prefix="verified-decoder-test-") as temporary:
            private_root = Path(temporary)
            (private_root / "adapter").mkdir()
            captured_paths.append(private_root)
            yield SimpleNamespace(
                path=private_root,
                run_manifest=_run_manifest(),
                artifact_manifest_sha256=ARTIFACT_DIGEST,
                run_manifest_sha256=RUN_DIGEST,
            )

    def verified_snapshot(root, base_model, tokenizer, digest):
        assert root == Path("/deployment/base")
        assert base_model == BASE_MODEL
        assert tokenizer == TOKENIZER
        assert digest == SNAPSHOT_DIGEST
        return _snapshot()

    class Factory:
        def __init__(self, name, result):
            self.name = name
            self.result = result

        def from_pretrained(self, *args, **kwargs):
            local_path = Path(args[-1])
            assert local_path.exists()
            assert local_path not in {Path("/deployment/adapter"), Path("/deployment/base")}
            captured_paths.append(local_path)
            calls.append((self.name, args, kwargs))
            return self.result

    class Model:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True

    tokenizer = object()
    base_model = object()
    model = Model()
    monkeypatch.setattr(
        local_decoder,
        "open_verified_decoder_lora_artifact_bundle",
        verified_bundle,
    )
    monkeypatch.setattr(
        local_decoder,
        "read_verified_model_snapshot_for_identities",
        verified_snapshot,
    )
    monkeypatch.setattr(
        local_decoder,
        "_runtime_modules",
        lambda: SimpleNamespace(
            transformers=SimpleNamespace(
                AutoTokenizer=Factory("tokenizer", tokenizer),
                AutoModelForCausalLM=Factory("base", base_model),
            ),
            peft=SimpleNamespace(
                PeftConfig=Factory("adapter-config", object()),
                PeftModel=Factory("adapter", model),
            ),
        ),
    )

    generator = load_local_decoder_lora_generator(_config())

    assert generator.artifact_manifest_sha256 == ARTIFACT_DIGEST
    assert generator.run_manifest_sha256 == RUN_DIGEST
    assert generator.model_snapshot_manifest_sha256 == SNAPSHOT_DIGEST
    assert generator.adapter_revision is None
    assert generator.base_model_revision == BASE_REVISION
    assert f"peft-bundle:sha256:{ARTIFACT_DIGEST}" in generator.model_id
    assert f"model-snapshot:sha256:{SNAPSHOT_DIGEST}" in generator.model_id
    assert model.evaluated
    assert calls[0][0] == "adapter-config"
    assert calls[0][2] == {"local_files_only": True}
    assert calls[1][0] == "tokenizer"
    assert calls[1][2] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert calls[2][0] == "base"
    assert calls[2][2] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert calls[3][0] == "adapter"
    assert calls[3][2]["local_files_only"] is True
    assert calls[3][2]["is_trainable"] is False
    assert all(not path.exists() for path in captured_paths)


@pytest.mark.parametrize(
    ("run_update", "snapshot_update", "message"),
    [
        (
            {"prompt_profile": "other-profile"},
            {},
            "prompt profile",
        ),
        (
            {
                "training_config": SimpleNamespace(
                    base_model=ArtifactIdentityV1(
                        repo_id="org/swapped-base",
                        revision=BASE_REVISION,
                    ),
                    tokenizer=TOKENIZER,
                )
            },
            {},
            "model or tokenizer identity",
        ),
        (
            {"model_snapshot_manifest_sha256": "e" * 64},
            {},
            "does not bind the authorized model snapshot",
        ),
        (
            {},
            {
                "manifest": SimpleNamespace(
                    artifacts=(
                        SimpleNamespace(
                            path="model.safetensors",
                            sha256="f" * 64,
                            bytes=8,
                        ),
                    )
                )
            },
            "inventory does not match",
        ),
    ],
)
def test_local_decoder_rejects_prompt_identity_swapped_snapshot_and_inventory_mismatch(
    run_update,
    snapshot_update,
    message,
):
    with pytest.raises(LocalDecoderOnlyLoRAError, match=message):
        local_decoder._validate_cross_links(
            _config(),
            _run_manifest(**run_update),
            _snapshot(**snapshot_update),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_manifest_sha256", "not-a-digest"),
        ("model_snapshot_manifest_sha256", "A" * 64),
    ],
)
def test_local_decoder_config_rejects_untrusted_digest_syntax(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_local_decoder_config_rejects_prompt_protocol_mismatch():
    with pytest.raises(ValueError, match="prompt profile"):
        _config(prompt_profile="other-profile")


def test_local_decoder_v1_requires_one_base_and_tokenizer_identity():
    different = ArtifactIdentityV1(repo_id="org/other-tokenizer", revision=BASE_REVISION)

    with pytest.raises(ValueError, match="tokenizer identity to equal base_model"):
        _config(tokenizer=different)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("bundle", "artifact verification failed"),
        ("snapshot", "artifact verification failed"),
        ("target", "artifact verification failed"),
    ],
)
def test_local_decoder_wraps_wrong_tampered_digest_and_target_mismatch(
    monkeypatch,
    operation,
    message,
):
    @contextmanager
    def bundle(root, digest):
        if operation in {"bundle", "target"}:
            detail = "target_modules mismatch" if operation == "target" else "manifest mismatch"
            raise DecoderLoRATrainingError(detail)
        yield SimpleNamespace(
            path=Path("/private/bundle"),
            run_manifest=_run_manifest(),
            artifact_manifest_sha256=ARTIFACT_DIGEST,
            run_manifest_sha256=RUN_DIGEST,
        )

    def snapshot(*args):
        raise DecoderLoRATrainingError("snapshot manifest mismatch")

    monkeypatch.setattr(local_decoder, "open_verified_decoder_lora_artifact_bundle", bundle)
    if operation == "snapshot":
        monkeypatch.setattr(
            local_decoder,
            "read_verified_model_snapshot_for_identities",
            snapshot,
        )

    with pytest.raises(LocalDecoderOnlyLoRAError, match=message):
        load_local_decoder_lora_generator(_config())

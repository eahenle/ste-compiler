from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ste_compiler.realizer import (
    DecoderOnlyLoRARealizerConfigV1,
    DeterministicRealizerConfigV1,
    EncoderDecoderRealizerConfigV1,
    canonical_realizer_config_json,
    load_realizer_config,
    realizer_config_sha256,
)
from ste_compiler.realizer.config import (
    MAX_BEAMS,
    MAX_NEW_TOKENS,
    MAX_SOURCE_TOKENS,
    MAX_SYMBOLS,
    REALIZER_CONFIG_ADAPTER,
)
from ste_compiler.training import ArtifactIdentityV1

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "data/realizers"


def _identity(revision: str = "a" * 40) -> dict[str, str]:
    return {"repo_id": "example/tiny-model", "revision": revision}


def _encoder_config() -> dict[str, object]:
    return {
        "schema_version": "ste-realizer-config-v1",
        "architecture": "encoder-decoder",
        "checkpoint": _identity(),
        "max_source_tokens": 1024,
        "max_new_tokens": 256,
        "num_beams": 1,
    }


def _decoder_config() -> dict[str, object]:
    return {
        "schema_version": "ste-realizer-config-v1",
        "architecture": "decoder-only-lora",
        "base_model": _identity(),
        "adapter": _identity("b" * 40),
        "prompt_profile": "decoder-only-symbol-plan-v1",
        "max_new_tokens": 512,
        "max_symbols": 128,
    }


@pytest.mark.parametrize(
    ("filename", "expected_type", "architecture"),
    [
        (
            "deterministic.yaml",
            DeterministicRealizerConfigV1,
            "deterministic",
        ),
        (
            "encoder-decoder-schema-example.yaml",
            EncoderDecoderRealizerConfigV1,
            "encoder-decoder",
        ),
        (
            "decoder-only-lora-schema-example.yaml",
            DecoderOnlyLoRARealizerConfigV1,
            "decoder-only-lora",
        ),
    ],
)
def test_packaged_realizer_examples_are_strict_and_versioned(
    filename,
    expected_type,
    architecture,
):
    config = load_realizer_config(EXAMPLES / filename)

    assert isinstance(config, expected_type)
    assert config.schema_version == "ste-realizer-config-v1"
    assert config.architecture == architecture


def test_neural_configs_reuse_immutable_artifact_identity():
    encoder = REALIZER_CONFIG_ADAPTER.validate_python(_encoder_config())
    decoder = REALIZER_CONFIG_ADAPTER.validate_python(_decoder_config())

    assert isinstance(encoder, EncoderDecoderRealizerConfigV1)
    assert isinstance(encoder.checkpoint, ArtifactIdentityV1)
    assert isinstance(decoder, DecoderOnlyLoRARealizerConfigV1)
    assert isinstance(decoder.base_model, ArtifactIdentityV1)
    assert isinstance(decoder.adapter, ArtifactIdentityV1)

    with pytest.raises(ValidationError, match="frozen"):
        encoder.checkpoint.revision = "c" * 40


def test_canonical_identity_is_stable_across_json_and_yaml(tmp_path):
    payload = _encoder_config()
    yaml_path = tmp_path / "realizer.yaml"
    json_path = tmp_path / "realizer.json"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    json_path.write_text(json.dumps(payload, indent=2))

    yaml_config = load_realizer_config(yaml_path)
    json_config = load_realizer_config(json_path)
    canonical = canonical_realizer_config_json(yaml_config)

    assert yaml_config == json_config
    assert canonical.endswith(b"\n")
    assert canonical == canonical_realizer_config_json(json_config)
    assert realizer_config_sha256(yaml_config) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_version": "ste-realizer-config-v1",
                "architecture": "deterministic",
                "unknown": True,
            },
            "extra_forbidden",
        ),
        (
            {
                "schema_version": "wrong",
                "architecture": "deterministic",
            },
            "ste-realizer-config-v1",
        ),
        (
            {**_encoder_config(), "max_new_tokens": True},
            "valid integer",
        ),
        (
            {**_decoder_config(), "prompt_profile": "unversioned"},
            "decoder-only-symbol-plan-v1",
        ),
        (
            {**_decoder_config(), "adapter": _identity("main")},
            "string_pattern_mismatch",
        ),
    ],
)
def test_realizer_config_rejects_ambiguous_or_unversioned_values(payload, message):
    with pytest.raises(ValidationError, match=message):
        REALIZER_CONFIG_ADAPTER.validate_python(payload)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**_encoder_config(), "max_source_tokens": 0}, "max_source_tokens"),
        (
            {**_encoder_config(), "max_source_tokens": MAX_SOURCE_TOKENS + 1},
            "max_source_tokens",
        ),
        ({**_encoder_config(), "max_new_tokens": MAX_NEW_TOKENS + 1}, "max_new_tokens"),
        ({**_encoder_config(), "num_beams": MAX_BEAMS + 1}, "num_beams"),
        ({**_decoder_config(), "max_symbols": MAX_SYMBOLS + 1}, "max_symbols"),
    ],
)
def test_generation_limits_are_positive_and_bounded(payload, field):
    with pytest.raises(ValidationError, match=field):
        REALIZER_CONFIG_ADAPTER.validate_python(payload)


def test_deterministic_config_cannot_smuggle_neural_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        REALIZER_CONFIG_ADAPTER.validate_python(
            {
                "schema_version": "ste-realizer-config-v1",
                "architecture": "deterministic",
                "checkpoint": _identity(),
            }
        )


@pytest.mark.parametrize("suffix", [".txt", ""])
def test_loader_rejects_unsupported_file_types(tmp_path, suffix):
    path = tmp_path / f"realizer{suffix}"
    path.write_text("{}")

    with pytest.raises(ValueError, match="unsupported realizer config file type"):
        load_realizer_config(path)


def test_loader_uses_safe_yaml_without_constructing_python_objects(tmp_path):
    path = tmp_path / "realizer.yaml"
    path.write_text("!!python/object/apply:os.system ['false']\n")

    with pytest.raises(ValueError, match="invalid realizer config"):
        load_realizer_config(path)

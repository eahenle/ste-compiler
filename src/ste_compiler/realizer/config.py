"""Strict, versioned configuration for deterministic and neural realizers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ste_compiler.training.config import ArtifactIdentityV1

REALIZER_CONFIG_SCHEMA_VERSION: Final = "ste-realizer-config-v1"
MAX_SOURCE_TOKENS: Final = 65_536
MAX_NEW_TOKENS: Final = 4_096
MAX_BEAMS: Final = 16
MAX_SYMBOLS: Final = 2_048


class StrictRealizerConfigModel(BaseModel):
    """Reject coercion and unknown fields, and make loaded configuration immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DeterministicRealizerConfigV1(StrictRealizerConfigModel):
    schema_version: Literal["ste-realizer-config-v1"]
    architecture: Literal["deterministic"]


class EncoderDecoderRealizerConfigV1(StrictRealizerConfigModel):
    schema_version: Literal["ste-realizer-config-v1"]
    architecture: Literal["encoder-decoder"]
    checkpoint: ArtifactIdentityV1
    max_source_tokens: int = Field(default=1024, gt=0, le=MAX_SOURCE_TOKENS)
    max_new_tokens: int = Field(default=256, gt=0, le=MAX_NEW_TOKENS)
    num_beams: int = Field(default=1, gt=0, le=MAX_BEAMS)


class DecoderOnlyLoRARealizerConfigV1(StrictRealizerConfigModel):
    schema_version: Literal["ste-realizer-config-v1"]
    architecture: Literal["decoder-only-lora"]
    base_model: ArtifactIdentityV1
    adapter: ArtifactIdentityV1
    prompt_profile: Literal["decoder-only-symbol-plan-v1"]
    max_new_tokens: int = Field(default=512, gt=0, le=MAX_NEW_TOKENS)
    max_symbols: int = Field(default=128, gt=0, le=MAX_SYMBOLS)


RealizerConfigV1 = Annotated[
    DeterministicRealizerConfigV1
    | EncoderDecoderRealizerConfigV1
    | DecoderOnlyLoRARealizerConfigV1,
    Field(discriminator="architecture"),
]
REALIZER_CONFIG_ADAPTER: TypeAdapter[RealizerConfigV1] = TypeAdapter(RealizerConfigV1)


def load_realizer_config(path: Path) -> RealizerConfigV1:
    """Safely load and strictly validate one JSON or YAML realizer configuration."""

    suffix = path.suffix.casefold()
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            raw: object = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        else:
            raise ValueError(f"unsupported realizer config file type: {path.suffix or '<none>'}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid realizer config: {error}") from error
    return REALIZER_CONFIG_ADAPTER.validate_python(raw)


def canonical_realizer_config_json(config: RealizerConfigV1) -> bytes:
    """Return stable identity bytes for a validated realizer configuration."""

    return (
        json.dumps(
            config.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def realizer_config_sha256(config: RealizerConfigV1) -> str:
    """Return the SHA-256 identity of a validated realizer configuration."""

    return hashlib.sha256(canonical_realizer_config_json(config)).hexdigest()

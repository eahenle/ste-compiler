"""Versioned, architecture-specific training configuration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    field_validator,
    model_validator,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_HUB_COMPONENT = re.compile(r"(?![.-])[A-Za-z0-9._-]{1,96}(?<![.-])", re.ASCII)


class StrictTrainingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentityV1(StrictTrainingModel):
    repo_id: str = Field(min_length=1, max_length=96)
    revision: str = Field(pattern=COMMIT_PATTERN)

    @field_validator("repo_id")
    @classmethod
    def hub_repository_id(cls, repo_id: str) -> str:
        components = repo_id.split("/")
        if (
            repo_id != repo_id.strip()
            or len(components) not in {1, 2}
            or any(_HUB_COMPONENT.fullmatch(component) is None for component in components)
            or ".." in repo_id
            or "--" in repo_id
            or repo_id.endswith(".git")
        ):
            raise ValueError("repo_id must be a Hugging Face Hub repository ID")
        return repo_id


class CorpusSelectionV1(StrictTrainingModel):
    dataset_version: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    train_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("dataset_version")
    @classmethod
    def stripped_dataset_version(cls, dataset_version: str) -> str:
        if dataset_version != dataset_version.strip():
            raise ValueError("dataset_version must be stripped and nonblank")
        return dataset_version


class OptimizerConfigV1(StrictTrainingModel):
    learning_rate: FiniteFloat = Field(gt=0)
    weight_decay: FiniteFloat = Field(default=0, ge=0)


class CommonTrainingConfigV1(StrictTrainingModel):
    schema_version: Literal["ste-training-config-v1"]
    corpus: CorpusSelectionV1
    base_model: ArtifactIdentityV1
    tokenizer: ArtifactIdentityV1
    seed: int = Field(ge=0, le=2**63 - 1)
    max_steps: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    optimizer: OptimizerConfigV1


class EncoderDecoderTrainingConfigV1(CommonTrainingConfigV1):
    architecture: Literal["encoder-decoder"]
    strategy: Literal["full"] = "full"
    max_source_tokens: int = Field(gt=0)
    max_target_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def tokenizer_matches_base(self) -> EncoderDecoderTrainingConfigV1:
        if self.tokenizer != self.base_model:
            raise ValueError("encoder-decoder tokenizer identity must match the base model")
        return self


class LoRAConfigV1(StrictTrainingModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: FiniteFloat = Field(default=0, ge=0, lt=1)
    bias: Literal["none"] = "none"
    target_modules: tuple[str, ...] = Field(min_length=1)

    @field_validator("target_modules", mode="before")
    @classmethod
    def immutable_target_modules(cls, target_modules: object) -> object:
        if isinstance(target_modules, list):
            return tuple(target_modules)
        return target_modules

    @field_validator("target_modules")
    @classmethod
    def unique_target_modules(cls, target_modules: tuple[str, ...]) -> tuple[str, ...]:
        if any(not module.strip() or module != module.strip() for module in target_modules):
            raise ValueError("target_modules must contain stripped nonblank names")
        if len(set(target_modules)) != len(target_modules):
            raise ValueError("target_modules must be unique")
        return target_modules


class DecoderOnlyLoRATrainingConfigV1(CommonTrainingConfigV1):
    architecture: Literal["decoder-only-lora"]
    prompt_profile: Literal["decoder-only-symbol-plan-v1"]
    max_sequence_tokens: int = Field(gt=0)
    lora: LoRAConfigV1

    @model_validator(mode="after")
    def tokenizer_matches_base(self) -> DecoderOnlyLoRATrainingConfigV1:
        if self.tokenizer != self.base_model:
            raise ValueError("decoder-only tokenizer identity must match the base model")
        return self


TrainingConfigV1 = Annotated[
    EncoderDecoderTrainingConfigV1 | DecoderOnlyLoRATrainingConfigV1,
    Field(discriminator="architecture"),
]
TRAINING_CONFIG_ADAPTER: TypeAdapter[TrainingConfigV1] = TypeAdapter(TrainingConfigV1)


def load_training_config(path: Path) -> TrainingConfigV1:
    """Safely load and strictly validate one JSON or YAML training config."""

    suffix = path.suffix.casefold()
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            raw: object = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        else:
            raise ValueError(f"unsupported training config file type: {path.suffix or '<none>'}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid training config: {error}") from error
    return TRAINING_CONFIG_ADAPTER.validate_python(raw)


def canonical_training_config_json(config: TrainingConfigV1) -> bytes:
    """Return the architecture-independent canonical identity bytes."""

    return (
        json.dumps(
            config.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def training_config_sha256(config: TrainingConfigV1) -> str:
    return hashlib.sha256(canonical_training_config_json(config)).hexdigest()

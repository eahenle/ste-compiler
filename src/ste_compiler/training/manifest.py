"""Canonical provenance manifests for encoder-decoder training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, FiniteFloat

from .config import (
    SHA256_PATTERN,
    ArtifactIdentityV1,
    EncoderDecoderTrainingConfigV1,
    StrictTrainingModel,
)

RUN_MANIFEST_SCHEMA_VERSION: Final = "encoder-decoder-run-manifest-v1"
VALIDATION_SCHEMA_VERSION: Final = "encoder-decoder-validation-v1"


class FileIdentityV1(StrictTrainingModel):
    path: str = Field(min_length=1, pattern=r"\S")
    sha256: str = Field(pattern=SHA256_PATTERN)
    bytes: int = Field(ge=0)


class PackageProvenanceV1(StrictTrainingModel):
    distribution: Literal["ste-compiler"]
    version: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: Literal[False]
    source_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    dependency_lock: FileIdentityV1
    dependencies: tuple[tuple[str, str], ...]


class HardwareProvenanceV1(StrictTrainingModel):
    device: Literal["cpu"]
    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    logical_cpu_count: int | None = Field(default=None, gt=0)
    torch_threads: int = Field(gt=0)
    process_peak_rss_bytes: int | None = Field(default=None, ge=0)


class ParameterCountsV1(StrictTrainingModel):
    total: int = Field(ge=0)
    trainable: int = Field(ge=0)


class CorpusRunIdentityV1(StrictTrainingModel):
    dataset_version: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    artifacts: tuple[tuple[str, str], ...]


class ValidationMetricsV1(StrictTrainingModel):
    schema_version: Literal["encoder-decoder-validation-v1"]
    record_count: int = Field(gt=0)
    mean_loss: FiniteFloat = Field(ge=0)


class EncoderDecoderRunManifestV1(StrictTrainingModel):
    schema_version: Literal["encoder-decoder-run-manifest-v1"]
    architecture: Literal["encoder-decoder"]
    training_config_sha256: str = Field(pattern=SHA256_PATTERN)
    training_config: EncoderDecoderTrainingConfigV1
    package: PackageProvenanceV1
    corpus: CorpusRunIdentityV1
    base_model: ArtifactIdentityV1
    tokenizer: ArtifactIdentityV1
    base_model_artifacts: tuple[FileIdentityV1, ...] = Field(min_length=1)
    tokenizer_artifacts: tuple[FileIdentityV1, ...] = Field(min_length=1)
    seed: int = Field(ge=0)
    parameter_counts: ParameterCountsV1
    optimizer_steps: int = Field(gt=0)
    micro_steps: int = Field(gt=0)
    record_order: tuple[str, ...] = Field(min_length=1)
    optimizer_losses: tuple[FiniteFloat, ...] = Field(min_length=1)
    validation: ValidationMetricsV1
    hardware: HardwareProvenanceV1
    duration_seconds: FiniteFloat = Field(ge=0)
    output_artifacts: tuple[FileIdentityV1, ...] = Field(min_length=1)
    evaluation_command: tuple[str, ...] = Field(min_length=1)


def _canonical_model_json(model: StrictTrainingModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_validation_metrics_json(metrics: ValidationMetricsV1) -> bytes:
    return _canonical_model_json(metrics)


def canonical_run_manifest_json(manifest: EncoderDecoderRunManifestV1) -> bytes:
    return _canonical_model_json(manifest)


def load_run_manifest(path: Path) -> EncoderDecoderRunManifestV1:
    try:
        return EncoderDecoderRunManifestV1.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid encoder-decoder run manifest: {error}") from error

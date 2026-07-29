"""Deterministic, content-addressed release records for both local neural architectures."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ste_compiler.ir.models import Document
from ste_compiler.realizer.base import Realizer
from ste_compiler.realizer.config import (
    REALIZER_CONFIG_ADAPTER,
    DecoderOnlyLoRALocalBundleRealizerConfigV1,
    EncoderDecoderLocalBundleRealizerConfigV1,
    RealizerConfigV1,
    canonical_realizer_config_json,
    realizer_config_sha256,
)
from ste_compiler.realizer.decoder_lora import DecoderOnlyLoRAError
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.realizer.encoder_decoder import EncoderDecoderError
from ste_compiler.realizer.factory import build_realizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.models import TerminologyData, VocabularyData
from ste_compiler.training.config import ArtifactIdentityV1
from ste_compiler.training.decoder_lora import (
    open_verified_decoder_lora_artifact_bundle,
    read_verified_model_snapshot_for_identities,
)
from ste_compiler.training.encoder_decoder import (
    open_verified_encoder_decoder_artifact_bundle,
)
from ste_compiler.training.release_reader import (
    ReleasedTrainingRecordV1,
    TrainingReleaseSnapshot,
    read_training_release,
)
from ste_compiler.validators import LexicalValidator, ValidationPipeline

REFERENCE_RELEASE_SCHEMA_VERSION: Final = "ste-reference-artifact-release-v1"
REFERENCE_RELEASE_METADATA_SCHEMA_VERSION: Final = "ste-reference-release-metadata-v1"
REFERENCE_PREDICTION_SCHEMA_VERSION: Final = "ste-reference-prediction-v1"
REFERENCE_RELEASE_MANIFEST_NAME: Final = "release-manifest.json"
REFERENCE_RELEASE_FILES: Final = frozenset(
    {
        "REPRODUCE.md",
        "checksums.sha256",
        "decoder-only-lora-model-card.md",
        "decoder-only-lora-predictions.jsonl",
        "decoder-only-lora-realizer.json",
        "encoder-decoder-model-card.md",
        "encoder-decoder-predictions.jsonl",
        "encoder-decoder-realizer.json",
        "release-metadata.json",
    }
)
MAX_REFERENCE_RELEASE_FILES: Final = 16
MAX_REFERENCE_RELEASE_FILE_BYTES: Final = 16 * 1024 * 1024
MAX_REFERENCE_RELEASE_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAX_REFERENCE_RELEASE_MANIFEST_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+", re.ASCII)

Architecture = Literal["encoder-decoder", "decoder-only-lora"]


class ReferenceReleaseError(ValueError):
    """A reference release could not establish its identity or reproducibility."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReferenceTrackAuthorizationV1(_StrictModel):
    architecture: Architecture
    base_model: ArtifactIdentityV1
    base_model_origin: str = Field(min_length=1, max_length=512, pattern=r"\S")
    base_model_license: str = Field(min_length=1, max_length=128, pattern=r"\S")
    artifact_license: str = Field(min_length=1, max_length=128, pattern=r"\S")
    redistribution: Literal["external-artifact-not-included"]


class ReferenceReleaseMetadataV1(_StrictModel):
    schema_version: Literal["ste-reference-release-metadata-v1"]
    release_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    intended_use: Literal["mechanics-smoke"]
    no_quality_claim: Literal[True]
    tracks: tuple[ReferenceTrackAuthorizationV1, ReferenceTrackAuthorizationV1]

    @model_validator(mode="after")
    def exactly_one_authorization_per_architecture(self) -> ReferenceReleaseMetadataV1:
        if tuple(track.architecture for track in self.tracks) != (
            "encoder-decoder",
            "decoder-only-lora",
        ):
            raise ValueError(
                "release metadata tracks must be ordered encoder-decoder, decoder-only-lora"
            )
        return self


class ReferenceReleaseFileV1(_StrictModel):
    path: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0, le=MAX_REFERENCE_RELEASE_FILE_BYTES)


class ReferenceArchitectureReleaseV1(_StrictModel):
    architecture: Architecture
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    realizer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_model: ArtifactIdentityV1
    tokenizer: ArtifactIdentityV1
    model_snapshot_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def decoder_alone_has_snapshot(self) -> ReferenceArchitectureReleaseV1:
        if (self.architecture == "decoder-only-lora") != (
            self.model_snapshot_manifest_sha256 is not None
        ):
            raise ValueError("only decoder-only-lora releases require a model snapshot")
        return self


class ReferenceReleaseManifestV1(_StrictModel):
    schema_version: Literal["ste-reference-artifact-release-v1"]
    release_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    intended_use: Literal["mechanics-smoke"]
    no_quality_claim: Literal[True]
    corpus_version: str = Field(min_length=1, max_length=128, pattern=r"\S")
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    architectures: tuple[ReferenceArchitectureReleaseV1, ReferenceArchitectureReleaseV1]
    files: tuple[ReferenceReleaseFileV1, ...] = Field(
        min_length=1,
        max_length=MAX_REFERENCE_RELEASE_FILES,
    )
    total_bytes: int = Field(ge=0, le=MAX_REFERENCE_RELEASE_TOTAL_BYTES)

    @model_validator(mode="after")
    def internally_consistent(self) -> ReferenceReleaseManifestV1:
        if tuple(item.architecture for item in self.architectures) != (
            "encoder-decoder",
            "decoder-only-lora",
        ):
            raise ValueError("release architectures must be ordered and complete")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("release file inventory must be sorted and unique")
        if set(paths) != REFERENCE_RELEASE_FILES:
            raise ValueError("release file inventory is incomplete")
        if any(_SAFE_NAME.fullmatch(path) is None for path in paths):
            raise ValueError("release file inventory contains an unsafe path")
        if self.total_bytes != sum(item.bytes for item in self.files):
            raise ValueError("release total_bytes does not match its file inventory")
        by_path = {item.path: item for item in self.files}
        if by_path["release-metadata.json"].sha256 != self.metadata_sha256:
            raise ValueError("release metadata identity does not match its file identity")
        for item in self.architectures:
            prefix = item.architecture
            if by_path[f"{prefix}-predictions.jsonl"].sha256 != item.predictions_sha256:
                raise ValueError("prediction identity does not match its file identity")
            if by_path[f"{prefix}-model-card.md"].sha256 != item.model_card_sha256:
                raise ValueError("model-card identity does not match its file identity")
            if by_path[f"{prefix}-realizer.json"].sha256 != item.realizer_config_sha256:
                raise ValueError("realizer identity does not match its file identity")
        return self


class ReferencePredictionV1(_StrictModel):
    schema_version: Literal["ste-reference-prediction-v1"]
    architecture: Architecture
    record_id: str = Field(min_length=1)
    split: Literal["test", "adversarial"]
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    realizer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["accepted", "rejected"]
    text: str | None
    validation_status: Literal["accepted", "rejected"] | None
    diagnostic_codes: tuple[str, ...]
    metadata: dict[str, str]
    error_type: str | None
    error: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def outcome_fields_match(self) -> ReferencePredictionV1:
        rejected_by_generation = self.error is not None
        if self.outcome == "accepted":
            if rejected_by_generation or self.validation_status != "accepted" or self.text is None:
                raise ValueError("accepted predictions require accepted validation and text")
        elif rejected_by_generation:
            if (
                self.text is not None
                or self.validation_status is not None
                or self.error_type is None
            ):
                raise ValueError("generation rejections may contain only error details")
        elif self.validation_status != "rejected" or self.text is None:
            raise ValueError("validator rejections require rejected validation and text")
        return self


@dataclass(frozen=True)
class ReferenceReleaseBuildResult:
    manifest: ReferenceReleaseManifestV1
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedReferenceRelease:
    manifest: ReferenceReleaseManifestV1
    manifest_sha256: str
    files: tuple[tuple[str, bytes], ...]


def _canonical_json(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def canonical_reference_release_manifest_json(manifest: ReferenceReleaseManifestV1) -> bytes:
    return _canonical_json(manifest)


def reference_release_manifest_sha256(manifest: ReferenceReleaseManifestV1) -> str:
    return hashlib.sha256(canonical_reference_release_manifest_json(manifest)).hexdigest()


def load_reference_release_metadata(path: Path) -> ReferenceReleaseMetadataV1:
    try:
        data = path.read_bytes()
        metadata = ReferenceReleaseMetadataV1.model_validate_json(data)
    except (OSError, ValueError) as error:
        raise ReferenceReleaseError(f"reference release metadata is invalid: {error}") from error
    if data != _canonical_json(metadata):
        raise ReferenceReleaseError("reference release metadata must be canonical JSON")
    return metadata


def _resources(snapshot: TrainingReleaseSnapshot) -> tuple[Vocabulary, TerminologyRegistry]:
    try:
        vocabulary = Vocabulary(VocabularyData.model_validate_json(snapshot.vocabulary_json))
        terminology = TerminologyRegistry(
            TerminologyData.model_validate_json(snapshot.terminology_json)
        )
    except ValueError as error:  # The release reader has already validated these exact bytes.
        raise ReferenceReleaseError(f"verified corpus resources are invalid: {error}") from error
    return vocabulary, terminology


def _prediction(
    *,
    architecture: Architecture,
    record: ReleasedTrainingRecordV1,
    corpus_manifest_sha256: str,
    config_sha256: str,
    realizer: Realizer,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> ReferencePredictionV1:
    document = Document.model_validate_json(record.serialized_ir)
    metadata = document.metadata.model_copy(
        update={
            "realizer": "deterministic",
            "realizer_version": DeterministicRealizer.version,
            "vocabulary_version": vocabulary.data.version,
            "terminology_version": terminology.data.version,
            "validator_profile": ValidationPipeline.profile,
        }
    )
    document = document.model_copy(update={"metadata": metadata})
    try:
        result = realizer.realize(document, vocabulary, terminology)
    except (ValueError, EncoderDecoderError, DecoderOnlyLoRAError) as error:
        return ReferencePredictionV1(
            schema_version=REFERENCE_PREDICTION_SCHEMA_VERSION,
            architecture=architecture,
            record_id=record.record_id,
            split=record.split,  # type: ignore[arg-type]
            corpus_manifest_sha256=corpus_manifest_sha256,
            realizer_config_sha256=config_sha256,
            outcome="rejected",
            text=None,
            validation_status=None,
            diagnostic_codes=(),
            metadata={},
            error_type=type(error).__name__,
            error=str(error),
        )
    result_realizer = result.metadata.get("realizer")
    result_version = result.metadata.get("realizer_version")
    if result_realizer is None or result_version is None:
        raise ReferenceReleaseError("realizer result omitted its implementation identity")
    document = document.model_copy(
        update={
            "metadata": document.metadata.model_copy(
                update={"realizer": result_realizer, "realizer_version": result_version}
            )
        }
    )
    report = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        result.text,
        document,
        result,
    )
    codes = tuple(sorted({item.code for item in report.violations}))
    validation_status: Literal["accepted", "rejected"] = (
        "accepted" if report.status == "accepted" else "rejected"
    )
    return ReferencePredictionV1(
        schema_version=REFERENCE_PREDICTION_SCHEMA_VERSION,
        architecture=architecture,
        record_id=record.record_id,
        split=record.split,  # type: ignore[arg-type]
        corpus_manifest_sha256=corpus_manifest_sha256,
        realizer_config_sha256=config_sha256,
        outcome=validation_status,
        text=result.text,
        validation_status=validation_status,
        diagnostic_codes=codes,
        metadata=dict(sorted(result.metadata.items())),
        error_type=None,
        error=None,
    )


def _prediction_bytes(
    *,
    architecture: Architecture,
    snapshot: TrainingReleaseSnapshot,
    config: RealizerConfigV1,
    realizer: Realizer,
) -> bytes:
    prepare = getattr(realizer, "prepare", None)
    if callable(prepare):
        # Local neural loaders are intentionally lazy. Establish their artifact trust
        # boundaries outside the per-record rejection path so loading failures remain fatal.
        prepare()
    vocabulary, terminology = _resources(snapshot)
    digest = realizer_config_sha256(config)
    records = (*snapshot.test, *snapshot.adversarial)
    return b"".join(
        _canonical_json(
            _prediction(
                architecture=architecture,
                record=record,
                corpus_manifest_sha256=snapshot.manifest_sha256,
                config_sha256=digest,
                realizer=realizer,
                vocabulary=vocabulary,
                terminology=terminology,
            )
        )
        for record in records
    )


def _model_card(
    *,
    authorization: ReferenceTrackAuthorizationV1,
    artifact_manifest_sha256: str,
    run_manifest_sha256: str,
    corpus: TrainingReleaseSnapshot,
    predictions_sha256: str,
    snapshot_sha256: str | None,
) -> bytes:
    snapshot_line = (
        f"- Base snapshot manifest SHA-256: `{snapshot_sha256}`\n"
        if snapshot_sha256 is not None
        else ""
    )
    return (
        f"# {authorization.architecture} mechanics artifact\n\n"
        "## Intended use\n\n"
        "This artifact is only an offline `mechanics-smoke` demonstrator for safe loading, "
        "constrained generation, rejection handling, and provenance. It is not a quality "
        "benchmark, production model, ASD-STE100 implementation, certification, or compliance "
        "claim.\n\n"
        "## Immutable identities\n\n"
        f"- Artifact manifest SHA-256: `{artifact_manifest_sha256}`\n"
        f"- Run manifest SHA-256: `{run_manifest_sha256}`\n"
        f"{snapshot_line}"
        f"- Corpus: `{corpus.manifest.dataset_version}`\n"
        f"- Corpus manifest SHA-256: `{corpus.manifest_sha256}`\n"
        f"- Prediction JSONL SHA-256: `{predictions_sha256}`\n"
        f"- Base model: `{authorization.base_model.repo_id}"
        f"@{authorization.base_model.revision}`\n\n"
        "## Origin and license declaration\n\n"
        f"- Base origin: `{authorization.base_model_origin}`\n"
        f"- Base license: `{authorization.base_model_license}`\n"
        f"- Derived artifact license: `{authorization.artifact_license}`\n"
        "- Redistribution: external artifact bytes are not included in this metadata release.\n\n"
        "The release builder requires this declaration to match the exact run-manifest identity. "
        "Operators remain responsible for reviewing the declared origin and license before "
        "hosting or redistributing model bytes.\n"
    ).encode()


def _reproduce(
    *,
    manifest_sha256_placeholder: str,
    encoder_digest: str,
    decoder_digest: str,
    snapshot_digest: str,
) -> bytes:
    return (
        "# Reproduce this metadata release\n\n"
        "Install the exact reviewed `ste-compiler` wheel and its `neural` extra, prepare the "
        "externally stored artifacts at the retained digests, then run:\n\n"
        "```bash\n"
        "ste-compiler build-reference-release \\\n"
        "  <release-metadata.json> <corpus-release> \\\n"
        f"  <encoder-bundle> {encoder_digest} \\\n"
        f"  <decoder-bundle> {decoder_digest} \\\n"
        f"  <decoder-model-snapshot> {snapshot_digest} \\\n"
        "  <new-output-directory> --json\n"
        "```\n\n"
        "Verify and regenerate every prediction through the production local loaders:\n\n"
        "```bash\n"
        "ste-compiler verify-reference-release \\\n"
        f"  <release-directory> {manifest_sha256_placeholder} \\\n"
        "  --regenerate --corpus-release <corpus-release> \\\n"
        "  --encoder-bundle <encoder-bundle> --decoder-bundle <decoder-bundle> \\\n"
        "  --decoder-model-snapshot <decoder-model-snapshot> --json\n"
        "```\n\n"
        "The manifest digest is reported after construction because a manifest cannot include "
        "its own hash. Replace the placeholder with that externally retained digest.\n"
    ).encode()


def _identity(path: str, data: bytes) -> ReferenceReleaseFileV1:
    return ReferenceReleaseFileV1(
        path=path, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
    )


def _checksums(files: dict[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(files[path]).hexdigest()}  {path}\n"
        for path in sorted(files)
        if path != "checksums.sha256"
    ).encode()


def _write_atomic(output: Path, files: dict[str, bytes], manifest: bytes) -> None:
    if output.exists():
        raise ReferenceReleaseError(f"reference release output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ReferenceReleaseError(
            f"reference release parent must be a real directory: {output.parent}"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        (stage / REFERENCE_RELEASE_MANIFEST_NAME).write_bytes(manifest)
        for child in stage.iterdir():
            descriptor = os.open(child, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_no_replace(stage, output)
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        for child in stage.iterdir() if stage.exists() else ():
            child.unlink()
        if stage.exists():
            stage.rmdir()
        raise


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise ReferenceReleaseError(
            f"atomic no-replace publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ReferenceReleaseError(
            f"reference release output was created concurrently: {destination}"
        )
    raise ReferenceReleaseError(
        f"cannot atomically publish reference release: {os.strerror(error_number)}"
    )


def _authorization_by_architecture(
    metadata: ReferenceReleaseMetadataV1,
) -> dict[Architecture, ReferenceTrackAuthorizationV1]:
    return {track.architecture: track for track in metadata.tracks}


def build_reference_release(
    metadata: ReferenceReleaseMetadataV1,
    corpus_release: Path,
    encoder_bundle: Path,
    encoder_artifact_manifest_sha256: str,
    decoder_bundle: Path,
    decoder_artifact_manifest_sha256: str,
    decoder_model_snapshot: Path,
    decoder_model_snapshot_manifest_sha256: str,
    output: Path,
) -> ReferenceReleaseBuildResult:
    """Build one portable dual-architecture metadata release and prediction set."""

    _require_hardened_platform()
    authorization = _authorization_by_architecture(metadata)
    with open_verified_encoder_decoder_artifact_bundle(
        encoder_bundle,
        encoder_artifact_manifest_sha256,
    ) as encoder:
        encoder_run = encoder.run_manifest
        encoder_run_digest = encoder.run_manifest_sha256
    with open_verified_decoder_lora_artifact_bundle(
        decoder_bundle,
        decoder_artifact_manifest_sha256,
    ) as decoder:
        decoder_run = decoder.run_manifest
        decoder_run_digest = decoder.run_manifest_sha256
    encoder_config = encoder_run.training_config
    decoder_config = decoder_run.training_config
    if encoder_config.corpus != decoder_config.corpus:
        raise ReferenceReleaseError("both architectures must bind the same corpus identity")
    if authorization["encoder-decoder"].base_model != encoder_config.base_model:
        raise ReferenceReleaseError(
            "encoder-decoder authorization does not match the run-manifest base identity"
        )
    if authorization["decoder-only-lora"].base_model != decoder_config.base_model:
        raise ReferenceReleaseError(
            "decoder-only-lora authorization does not match the run-manifest base identity"
        )
    snapshot = read_verified_model_snapshot_for_identities(
        decoder_model_snapshot,
        base_model=decoder_config.base_model,
        tokenizer=decoder_config.tokenizer,
        expected_manifest_sha256=decoder_model_snapshot_manifest_sha256,
    )
    if decoder_run.model_snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ReferenceReleaseError("decoder run does not bind the supplied model snapshot")
    corpus = read_training_release(corpus_release, encoder_config.corpus)

    encoder_realizer_config = EncoderDecoderLocalBundleRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="encoder-decoder-local-bundle",
        artifact_manifest_sha256=encoder_artifact_manifest_sha256,
        intended_use="mechanics-smoke",
        max_source_tokens=encoder_config.max_source_tokens,
        max_new_tokens=min(32, encoder_config.max_target_tokens),
        num_beams=1,
    )
    decoder_realizer_config = DecoderOnlyLoRALocalBundleRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="decoder-only-lora-local-bundle",
        artifact_manifest_sha256=decoder_artifact_manifest_sha256,
        model_snapshot_manifest_sha256=decoder_model_snapshot_manifest_sha256,
        base_model=decoder_config.base_model,
        tokenizer=decoder_config.tokenizer,
        intended_use="mechanics-smoke",
        prompt_profile=decoder_config.prompt_profile,
        max_new_tokens=128,
        max_symbols=1,
    )
    encoder_predictions = _prediction_bytes(
        architecture="encoder-decoder",
        snapshot=corpus,
        config=encoder_realizer_config,
        realizer=build_realizer(
            encoder_realizer_config,
            artifact_bundle=encoder_bundle,
        ),
    )
    decoder_predictions = _prediction_bytes(
        architecture="decoder-only-lora",
        snapshot=corpus,
        config=decoder_realizer_config,
        realizer=build_realizer(
            decoder_realizer_config,
            artifact_bundle=decoder_bundle,
            model_snapshot=decoder_model_snapshot,
        ),
    )
    encoder_predictions_digest = hashlib.sha256(encoder_predictions).hexdigest()
    decoder_predictions_digest = hashlib.sha256(decoder_predictions).hexdigest()
    encoder_card = _model_card(
        authorization=authorization["encoder-decoder"],
        artifact_manifest_sha256=encoder_artifact_manifest_sha256,
        run_manifest_sha256=encoder_run_digest,
        corpus=corpus,
        predictions_sha256=encoder_predictions_digest,
        snapshot_sha256=None,
    )
    decoder_card = _model_card(
        authorization=authorization["decoder-only-lora"],
        artifact_manifest_sha256=decoder_artifact_manifest_sha256,
        run_manifest_sha256=decoder_run_digest,
        corpus=corpus,
        predictions_sha256=decoder_predictions_digest,
        snapshot_sha256=decoder_model_snapshot_manifest_sha256,
    )
    metadata_bytes = _canonical_json(metadata)
    files = {
        "REPRODUCE.md": _reproduce(
            manifest_sha256_placeholder="<release-manifest-sha256>",
            encoder_digest=encoder_artifact_manifest_sha256,
            decoder_digest=decoder_artifact_manifest_sha256,
            snapshot_digest=decoder_model_snapshot_manifest_sha256,
        ),
        "decoder-only-lora-model-card.md": decoder_card,
        "decoder-only-lora-predictions.jsonl": decoder_predictions,
        "decoder-only-lora-realizer.json": canonical_realizer_config_json(decoder_realizer_config),
        "encoder-decoder-model-card.md": encoder_card,
        "encoder-decoder-predictions.jsonl": encoder_predictions,
        "encoder-decoder-realizer.json": canonical_realizer_config_json(encoder_realizer_config),
        "release-metadata.json": metadata_bytes,
    }
    files["checksums.sha256"] = _checksums(files)
    identities = tuple(_identity(path, files[path]) for path in sorted(files))
    architectures = (
        ReferenceArchitectureReleaseV1(
            architecture="encoder-decoder",
            artifact_manifest_sha256=encoder_artifact_manifest_sha256,
            run_manifest_sha256=encoder_run_digest,
            realizer_config_sha256=realizer_config_sha256(encoder_realizer_config),
            predictions_sha256=encoder_predictions_digest,
            model_card_sha256=hashlib.sha256(encoder_card).hexdigest(),
            base_model=encoder_config.base_model,
            tokenizer=encoder_config.tokenizer,
        ),
        ReferenceArchitectureReleaseV1(
            architecture="decoder-only-lora",
            artifact_manifest_sha256=decoder_artifact_manifest_sha256,
            run_manifest_sha256=decoder_run_digest,
            realizer_config_sha256=realizer_config_sha256(decoder_realizer_config),
            predictions_sha256=decoder_predictions_digest,
            model_card_sha256=hashlib.sha256(decoder_card).hexdigest(),
            base_model=decoder_config.base_model,
            tokenizer=decoder_config.tokenizer,
            model_snapshot_manifest_sha256=decoder_model_snapshot_manifest_sha256,
        ),
    )
    manifest = ReferenceReleaseManifestV1(
        schema_version=REFERENCE_RELEASE_SCHEMA_VERSION,
        release_id=metadata.release_id,
        intended_use=metadata.intended_use,
        no_quality_claim=True,
        corpus_version=corpus.manifest.dataset_version,
        corpus_manifest_sha256=corpus.manifest_sha256,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        architectures=architectures,
        files=identities,
        total_bytes=sum(item.bytes for item in identities),
    )
    manifest_bytes = canonical_reference_release_manifest_json(manifest)
    digest = reference_release_manifest_sha256(manifest)
    _write_atomic(output, files, manifest_bytes)
    read_verified_reference_release(output, digest)
    return ReferenceReleaseBuildResult(manifest=manifest, manifest_sha256=digest)


def _read_regular(directory_fd: int, name: str, limit: int) -> bytes:
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ReferenceReleaseError(f"cannot safely open reference release file: {name}") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ReferenceReleaseError(
                f"reference release file must be a bounded single-link regular file: {name}"
            )
        chunks: list[bytes] = []
        count = 0
        while chunk := os.read(fd, min(1024 * 1024, limit + 1 - count)):
            chunks.append(chunk)
            count += len(chunk)
            if count > limit:
                raise ReferenceReleaseError(f"reference release file exceeds its limit: {name}")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        count != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ReferenceReleaseError(f"reference release file changed while read: {name}")
    return b"".join(chunks)


def _require_hardened_platform() -> None:
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_CLOEXEC")
        or not (sys.platform == "darwin" or sys.platform.startswith("linux"))
    ):
        raise ReferenceReleaseError(
            "reference release construction and verification require hardened POSIX "
            "directory handles and atomic no-replace publication"
        )


def _bounded_names(directory_fd: int) -> set[str]:
    names: set[str] = set()
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if len(names) >= MAX_REFERENCE_RELEASE_FILES + 1:
                raise ReferenceReleaseError("reference release contains too many entries")
            names.add(entry.name)
    return names


def _validate_release_payloads(
    manifest: ReferenceReleaseManifestV1,
    files: dict[str, bytes],
) -> None:
    try:
        metadata = ReferenceReleaseMetadataV1.model_validate_json(files["release-metadata.json"])
    except ValueError as error:
        raise ReferenceReleaseError(f"release metadata is invalid: {error}") from error
    if files["release-metadata.json"] != _canonical_json(metadata):
        raise ReferenceReleaseError("release metadata is not canonical JSON")
    if (
        metadata.release_id != manifest.release_id
        or metadata.intended_use != manifest.intended_use
        or metadata.no_quality_claim != manifest.no_quality_claim
    ):
        raise ReferenceReleaseError("release metadata does not match the release manifest")
    authorization = _authorization_by_architecture(metadata)
    for architecture in manifest.architectures:
        if authorization[architecture.architecture].base_model != architecture.base_model:
            raise ReferenceReleaseError(
                f"{architecture.architecture} release metadata identity does not match"
            )
        config_path = f"{architecture.architecture}-realizer.json"
        config_bytes = files[config_path]
        try:
            config = REALIZER_CONFIG_ADAPTER.validate_json(config_bytes)
        except ValueError as error:
            raise ReferenceReleaseError(
                f"{architecture.architecture} realizer config is invalid: {error}"
            ) from error
        if config_bytes != canonical_realizer_config_json(config):
            raise ReferenceReleaseError(
                f"{architecture.architecture} realizer config is not canonical JSON"
            )
        if realizer_config_sha256(config) != architecture.realizer_config_sha256:
            raise ReferenceReleaseError(
                f"{architecture.architecture} realizer config identity does not match"
            )
        if architecture.architecture == "encoder-decoder":
            if (
                not isinstance(config, EncoderDecoderLocalBundleRealizerConfigV1)
                or config.artifact_manifest_sha256 != architecture.artifact_manifest_sha256
            ):
                raise ReferenceReleaseError(
                    "encoder-decoder realizer config does not match its artifact identity"
                )
        elif (
            not isinstance(config, DecoderOnlyLoRALocalBundleRealizerConfigV1)
            or config.artifact_manifest_sha256 != architecture.artifact_manifest_sha256
            or config.model_snapshot_manifest_sha256 != architecture.model_snapshot_manifest_sha256
            or config.base_model != architecture.base_model
            or config.tokenizer != architecture.tokenizer
        ):
            raise ReferenceReleaseError(
                "decoder-only-lora realizer config does not match its artifact identities"
            )
        prediction_path = f"{architecture.architecture}-predictions.jsonl"
        lines = files[prediction_path].splitlines(keepends=True)
        if not lines:
            raise ReferenceReleaseError(
                f"{architecture.architecture} prediction set must not be empty"
            )
        seen: set[tuple[str, str]] = set()
        for line in lines:
            try:
                prediction = ReferencePredictionV1.model_validate_json(line)
            except ValueError as error:
                raise ReferenceReleaseError(
                    f"{architecture.architecture} prediction record is invalid: {error}"
                ) from error
            if line != _canonical_json(prediction):
                raise ReferenceReleaseError(
                    f"{architecture.architecture} prediction record is not canonical JSON"
                )
            identity = (prediction.split, prediction.record_id)
            if identity in seen:
                raise ReferenceReleaseError(
                    f"{architecture.architecture} prediction record is duplicated"
                )
            seen.add(identity)
            if (
                prediction.architecture != architecture.architecture
                or prediction.corpus_manifest_sha256 != manifest.corpus_manifest_sha256
                or prediction.realizer_config_sha256 != architecture.realizer_config_sha256
            ):
                raise ReferenceReleaseError(
                    f"{architecture.architecture} prediction identities do not match"
                )


def read_verified_reference_release(
    release: Path,
    expected_manifest_sha256: str,
) -> VerifiedReferenceRelease:
    """Read one exact release by external identity through descriptor-relative handles."""

    _require_hardened_platform()
    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise ReferenceReleaseError("release manifest SHA-256 must be lowercase hexadecimal")
    try:
        directory_fd = os.open(
            release,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ReferenceReleaseError(
            f"reference release must be a real directory: {release}"
        ) from error
    try:
        manifest_bytes = _read_regular(
            directory_fd,
            REFERENCE_RELEASE_MANIFEST_NAME,
            MAX_REFERENCE_RELEASE_MANIFEST_BYTES,
        )
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
            raise ReferenceReleaseError("release manifest SHA-256 does not match")
        try:
            manifest = ReferenceReleaseManifestV1.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise ReferenceReleaseError(f"release manifest is invalid: {error}") from error
        if manifest_bytes != canonical_reference_release_manifest_json(manifest):
            raise ReferenceReleaseError("release manifest is not canonical JSON")
        expected_names = {REFERENCE_RELEASE_MANIFEST_NAME, *(item.path for item in manifest.files)}
        if _bounded_names(directory_fd) != expected_names:
            raise ReferenceReleaseError("reference release file set does not match its manifest")
        captured: list[tuple[str, bytes]] = []
        for identity in manifest.files:
            data = _read_regular(directory_fd, identity.path, MAX_REFERENCE_RELEASE_FILE_BYTES)
            if len(data) != identity.bytes or hashlib.sha256(data).hexdigest() != identity.sha256:
                raise ReferenceReleaseError(
                    f"reference release file identity does not match: {identity.path}"
                )
            captured.append((identity.path, data))
        if _bounded_names(directory_fd) != expected_names:
            raise ReferenceReleaseError("reference release file set changed while read")
    finally:
        os.close(directory_fd)
    by_name = dict(captured)
    expected_checksums = _checksums(
        {path: data for path, data in captured if path != "checksums.sha256"}
    )
    if by_name["checksums.sha256"] != expected_checksums:
        raise ReferenceReleaseError("reference release checksum inventory is not canonical")
    _validate_release_payloads(manifest, by_name)
    return VerifiedReferenceRelease(
        manifest=manifest,
        manifest_sha256=expected_manifest_sha256,
        files=tuple(captured),
    )


def verify_reference_release(
    release: Path,
    expected_manifest_sha256: str,
    *,
    corpus_release: Path | None = None,
    encoder_bundle: Path | None = None,
    decoder_bundle: Path | None = None,
    decoder_model_snapshot: Path | None = None,
    regenerate: bool = False,
) -> VerifiedReferenceRelease:
    """Verify metadata alone or regenerate it through both exact local loaders."""

    verified = read_verified_reference_release(release, expected_manifest_sha256)
    if not regenerate:
        return verified
    if (
        corpus_release is None
        or encoder_bundle is None
        or decoder_bundle is None
        or decoder_model_snapshot is None
    ):
        raise ReferenceReleaseError(
            "regeneration requires corpus and all three local artifact locators"
        )
    files = dict(verified.files)
    metadata = ReferenceReleaseMetadataV1.model_validate_json(files["release-metadata.json"])
    encoder, decoder = verified.manifest.architectures
    with tempfile.TemporaryDirectory(prefix="ste-reference-release-verify-") as temporary:
        rebuilt_path = Path(temporary) / "release"
        rebuilt = build_reference_release(
            metadata,
            corpus_release,
            encoder_bundle,
            encoder.artifact_manifest_sha256,
            decoder_bundle,
            decoder.artifact_manifest_sha256,
            decoder_model_snapshot,
            decoder.model_snapshot_manifest_sha256 or "",
            rebuilt_path,
        )
        rebuilt_verified = read_verified_reference_release(
            rebuilt_path,
            rebuilt.manifest_sha256,
        )
    if (
        rebuilt.manifest_sha256 != expected_manifest_sha256
        or rebuilt_verified.files != verified.files
    ):
        raise ReferenceReleaseError("reference release does not reproduce byte for byte")
    return verified

"""Race-resistant reader for one hash-pinned demonstration-corpus release."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer.constrained import EXACT_PLAN_SYMBOL
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.models import TerminologyData, VocabularyData

from .config import CorpusSelectionV1, StrictTrainingModel
from .records import build_training_record
from .release import (
    EXPECTED_RELEASE_FILES,
    CorpusConstruction,
    document_features,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Split = Literal["train", "validation", "test", "adversarial"]
SPLIT_NAMES: tuple[Split, ...] = ("train", "validation", "test", "adversarial")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CHECKSUM_BYTES = 1024 * 1024
MAX_RELEASE_FILE_BYTES = 64 * 1024 * 1024
MAX_RELEASE_BYTES = 128 * 1024 * 1024


class _ReleasedSourceModel(StrictTrainingModel):
    id: str = Field(min_length=1, pattern=r"\S")
    text: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    license_id: str = Field(min_length=1, pattern=r"\S")


class _ReleasedTrainingRecordModel(StrictTrainingModel):
    schema_version: Literal["demonstration-corpus-record-v1"]
    record_id: str = Field(min_length=1, pattern=r"\S")
    split: Split
    source: _ReleasedSourceModel
    ir: Document
    serialized_ir: str = Field(min_length=1)
    text: str = Field(min_length=1)
    symbols: str = Field(min_length=1)
    allowed_symbols: tuple[str, ...] = Field(min_length=1)
    metadata: dict[str, str]
    features: tuple[str, ...]


class _ReleaseArtifactModel(StrictTrainingModel):
    path: str = Field(min_length=1, pattern=r"\S")
    sha256: str = Field(pattern=SHA256_PATTERN)
    bytes: int = Field(ge=0)


class _ReleaseManifestModel(StrictTrainingModel):
    schema_version: Literal["demonstration-corpus-release-v1"]
    dataset_version: str = Field(min_length=1, pattern=r"\S")
    construction_sha256: str = Field(pattern=SHA256_PATTERN)
    seed: int = Field(ge=0)
    record_count: int = Field(gt=0)
    split_counts: dict[str, int]
    profiles: tuple[dict[str, str], ...]
    artifacts: tuple[_ReleaseArtifactModel, ...]


@dataclass(frozen=True)
class ReleasedTrainingRecordV1:
    schema_version: str
    record_id: str
    split: Split
    source_id: str
    source_sha256: str
    source_license_id: str
    serialized_ir: str
    text: str
    symbols: str
    allowed_symbols: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]
    features: tuple[str, ...]


@dataclass(frozen=True)
class TrainingReleaseManifestV1:
    schema_version: str
    dataset_version: str
    construction_sha256: str
    seed: int
    record_count: int
    split_counts: tuple[tuple[str, int], ...]
    profiles: tuple[tuple[tuple[str, str], ...], ...]


@dataclass(frozen=True)
class TrainingReleaseSnapshot:
    manifest: TrainingReleaseManifestV1
    manifest_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    train: tuple[ReleasedTrainingRecordV1, ...]
    validation: tuple[ReleasedTrainingRecordV1, ...]
    test: tuple[ReleasedTrainingRecordV1, ...]
    adversarial: tuple[ReleasedTrainingRecordV1, ...]

    @property
    def symbol_inventory(self) -> frozenset[str]:
        return frozenset(
            symbol
            for record in (*self.train, *self.validation, *self.test, *self.adversarial)
            for symbol in record.allowed_symbols
        )


def _read_regular_entry(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"cannot open training release entry {name!r}: {error}") from error
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"training release entry must be a single-link regular file: {name}")
        if before.st_size > max_bytes:
            raise ValueError(f"training release entry exceeds its size limit: {name}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError(f"training release entry size does not match its manifest: {name}")
        chunks: list[bytes] = []
        byte_count = 0
        while chunk := os.read(file_fd, min(1024 * 1024, max_bytes + 1 - byte_count)):
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > max_bytes or (
                expected_bytes is not None and byte_count > expected_bytes
            ):
                raise ValueError(f"training release entry exceeds its size limit: {name}")
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        data = b"".join(chunks)
        if identity_before != identity_after or len(data) != before.st_size:
            raise ValueError(f"training release entry changed while it was read: {name}")
        return data
    except OSError as error:
        raise ValueError(f"cannot read training release entry {name!r}: {error}") from error
    finally:
        os.close(file_fd)


def _read_release_files(
    release: Path,
    expected: CorpusSelectionV1,
) -> tuple[dict[str, bytes], _ReleaseManifestModel, dict[str, _ReleaseArtifactModel]]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory_fd = os.open(release, flags)
    except OSError as error:
        raise ValueError(f"training release must be a real directory: {release}") from error
    try:
        names = set(os.listdir(directory_fd))
        if names != EXPECTED_RELEASE_FILES:
            missing = sorted(EXPECTED_RELEASE_FILES - names)
            unexpected = sorted(names - EXPECTED_RELEASE_FILES)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise ValueError("training release file set is invalid: " + "; ".join(details))
        manifest_data = _read_regular_entry(
            directory_fd,
            "manifest.json",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        if _sha256(manifest_data) != expected.manifest_sha256:
            raise ValueError("training release manifest SHA-256 does not match the configuration")
        try:
            manifest = _ReleaseManifestModel.model_validate_json(manifest_data)
        except ValueError as error:
            raise ValueError(f"invalid training release manifest: {error}") from error
        artifact_by_path = {artifact.path: artifact for artifact in manifest.artifacts}
        expected_artifact_paths = names - {"checksums.sha256", "manifest.json"}
        if (
            len(artifact_by_path) != len(manifest.artifacts)
            or set(artifact_by_path) != expected_artifact_paths
        ):
            raise ValueError("training release manifest has an invalid artifact set")
        if any(artifact.bytes > MAX_RELEASE_FILE_BYTES for artifact in manifest.artifacts):
            raise ValueError("training release manifest declares an oversized artifact")
        if sum(artifact.bytes for artifact in manifest.artifacts) > MAX_RELEASE_BYTES:
            raise ValueError("training release manifest exceeds the total release size limit")

        files = {"manifest.json": manifest_data}
        for name in sorted(names - {"manifest.json"}):
            artifact = artifact_by_path.get(name)
            files[name] = _read_regular_entry(
                directory_fd,
                name,
                max_bytes=(
                    MAX_CHECKSUM_BYTES if name == "checksums.sha256" else MAX_RELEASE_FILE_BYTES
                ),
                expected_bytes=None if artifact is None else artifact.bytes,
            )
        return files, manifest, artifact_by_path
    finally:
        os.close(directory_fd)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checksums(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("training release checksum inventory is not UTF-8") from error
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or not parts[1]
            or "/" in parts[1]
            or parts[1] in checksums
        ):
            raise ValueError("training release checksum inventory is invalid")
        checksums[parts[1]] = parts[0]
    return checksums


def _record_lines(data: bytes, split: Split) -> tuple[_ReleasedTrainingRecordModel, ...]:
    records: list[_ReleasedTrainingRecordModel] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            record = _ReleasedTrainingRecordModel.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"invalid {split} record at line {line_number}: {error}") from error
        if record.split != split:
            raise ValueError(
                f"record {record.record_id!r} belongs to {record.split!r}, not {split!r}"
            )
        if record.record_id != record.ir.id:
            raise ValueError(f"record {record.record_id!r} does not match its IR document id")
        if record.serialized_ir != canonical_document_json(record.ir):
            raise ValueError(
                f"record {record.record_id!r} does not contain canonical serialized IR"
            )
        if record.source.sha256 != _sha256(record.source.text.encode("utf-8")):
            raise ValueError(f"record {record.record_id!r} has an invalid source SHA-256")
        symbol_parts = record.symbols.split()
        if (
            not symbol_parts
            or symbol_parts[0] != EXACT_PLAN_SYMBOL
            or " ".join(symbol_parts) != record.symbols
        ):
            raise ValueError(f"record {record.record_id!r} has a noncanonical symbolic plan")
        if tuple(sorted(set(symbol_parts))) != record.allowed_symbols:
            raise ValueError(f"record {record.record_id!r} has an invalid allowed-symbol set")
        if tuple(sorted(set(record.features))) != record.features:
            raise ValueError(f"record {record.record_id!r} has noncanonical features")
        records.append(record)
    if not records:
        raise ValueError(f"training release split is empty: {split}")
    return tuple(records)


def _validate_source_spans(record: _ReleasedTrainingRecordModel) -> None:
    statements = [statement for section in record.ir.sections for statement in section.statements]
    claims = [*statements, *record.ir.ambiguities, *record.ir.causal_relations]
    if any(not claim.source_spans for claim in claims):
        raise ValueError(f"record {record.record_id!r} has a claim without a source span")
    for claim in claims:
        for span in claim.source_spans:
            if span.source_id != record.source.id:
                raise ValueError(
                    f"record {record.record_id!r} has a source span for the wrong source"
                )
            if span.end > len(record.source.text):
                raise ValueError(
                    f"record {record.record_id!r} has a source span outside the source"
                )
            if span.quote is None or not span.quote.strip():
                raise ValueError(f"record {record.record_id!r} has a source span without a quote")
            if record.source.text[span.start : span.end] != span.quote:
                raise ValueError(f"record {record.record_id!r} has a source span quote mismatch")


def _freeze_record(record: _ReleasedTrainingRecordModel) -> ReleasedTrainingRecordV1:
    return ReleasedTrainingRecordV1(
        schema_version=record.schema_version,
        record_id=record.record_id,
        split=record.split,
        source_id=record.source.id,
        source_sha256=record.source.sha256,
        source_license_id=record.source.license_id,
        serialized_ir=record.serialized_ir,
        text=record.text,
        symbols=record.symbols,
        allowed_symbols=record.allowed_symbols,
        metadata=tuple(sorted(record.metadata.items())),
        features=record.features,
    )


def read_training_release(
    release: Path,
    expected: CorpusSelectionV1,
) -> TrainingReleaseSnapshot:
    """Read and validate one exact release before returning immutable records."""

    files, manifest, artifact_by_path = _read_release_files(release, expected)
    manifest_sha256 = _sha256(files["manifest.json"])
    if manifest.dataset_version != expected.dataset_version:
        raise ValueError("training release dataset version does not match the configuration")

    checksum_inventory = _checksums(files["checksums.sha256"])
    expected_checksum_paths = set(files) - {"checksums.sha256"}
    if set(checksum_inventory) != expected_checksum_paths:
        raise ValueError("training release checksum inventory has an invalid file set")
    for path, digest in checksum_inventory.items():
        if _sha256(files[path]) != digest:
            raise ValueError(f"training release checksum does not match: {path}")

    for path, artifact in artifact_by_path.items():
        data = files[path]
        if artifact.sha256 != _sha256(data) or artifact.bytes != len(data):
            raise ValueError(f"training release artifact identity does not match: {path}")

    if artifact_by_path["train.jsonl"].sha256 != expected.train_sha256:
        raise ValueError("training release train SHA-256 does not match the configuration")
    if artifact_by_path["validation.jsonl"].sha256 != expected.validation_sha256:
        raise ValueError("training release validation SHA-256 does not match the configuration")
    construction_artifact = artifact_by_path["source-construction.json"]
    if manifest.construction_sha256 != construction_artifact.sha256:
        raise ValueError("training release construction SHA-256 does not match its manifest")
    try:
        construction = CorpusConstruction.model_validate_json(files["source-construction.json"])
    except ValueError as error:
        raise ValueError(f"training release construction is invalid: {error}") from error
    if (
        construction.dataset_version != manifest.dataset_version
        or construction.seed != manifest.seed
    ):
        raise ValueError("training release construction identity does not match its manifest")

    records = {split: _record_lines(files[f"{split}.jsonl"], split) for split in SPLIT_NAMES}
    record_ids = [record.record_id for split in SPLIT_NAMES for record in records[split]]
    source_ids = [record.source.id for split in SPLIT_NAMES for record in records[split]]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("training release record IDs must be unique")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("training release source IDs must be unique")
    actual_split_counts = {split: len(records[split]) for split in SPLIT_NAMES}
    if manifest.split_counts != actual_split_counts or manifest.record_count != sum(
        actual_split_counts.values()
    ):
        raise ValueError("training release manifest counts do not match its records")
    record_profiles = {
        json.dumps(
            record.metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ): record.metadata
        for split in SPLIT_NAMES
        for record in records[split]
    }
    expected_profiles = tuple(record_profiles[key] for key in sorted(record_profiles))
    if manifest.profiles != expected_profiles:
        raise ValueError("training release manifest profiles do not match its records")
    construction_by_id = {record.id: record for record in construction.records}
    if len(construction_by_id) != len(construction.records) or set(construction_by_id) != set(
        record_ids
    ):
        raise ValueError("training release construction records do not match released records")
    try:
        vocabulary = Vocabulary(VocabularyData.model_validate_json(files["vocabulary.json"]))
        terminology = TerminologyRegistry(
            TerminologyData.model_validate_json(files["terminology.json"])
        )
    except ValueError as error:
        raise ValueError(f"training release resources are invalid: {error}") from error
    for split in SPLIT_NAMES:
        for record in records[split]:
            construction_record = construction_by_id[record.record_id]
            if (
                construction_record.split != record.split
                or construction_record.source_id != record.source.id
                or construction_record.source_text != record.source.text
                or construction_record.license_id != record.source.license_id
            ):
                raise ValueError(
                    f"training release record {record.record_id!r} does not match its "
                    "construction source"
                )
            _validate_source_spans(record)
            if document_features(record.ir, record.source.text) != record.features:
                raise ValueError(
                    f"training release record {record.record_id!r} has invalid semantic features"
                )
            try:
                rebuilt = build_training_record(record.ir, vocabulary, terminology)
            except ValueError as error:
                raise ValueError(
                    f"training release record {record.record_id!r} does not rebuild: {error}"
                ) from error
            if (
                rebuilt["serialized_ir"] != record.serialized_ir
                or rebuilt["text"] != record.text
                or rebuilt["symbols"] != record.symbols
                or tuple(rebuilt["allowed_symbols"]) != record.allowed_symbols
                or rebuilt["metadata"] != record.metadata
            ):
                raise ValueError(
                    f"training release record {record.record_id!r} does not match its "
                    "deterministic training target"
                )

    return TrainingReleaseSnapshot(
        manifest=TrainingReleaseManifestV1(
            schema_version=manifest.schema_version,
            dataset_version=manifest.dataset_version,
            construction_sha256=manifest.construction_sha256,
            seed=manifest.seed,
            record_count=manifest.record_count,
            split_counts=tuple(sorted(manifest.split_counts.items())),
            profiles=tuple(tuple(sorted(profile.items())) for profile in manifest.profiles),
        ),
        manifest_sha256=manifest_sha256,
        artifact_sha256=tuple(
            (path, artifact_by_path[path].sha256) for path in sorted(artifact_by_path)
        ),
        train=tuple(_freeze_record(record) for record in records["train"]),
        validation=tuple(_freeze_record(record) for record in records["validation"]),
        test=tuple(_freeze_record(record) for record in records["test"]),
        adversarial=tuple(_freeze_record(record) for record in records["adversarial"]),
    )

"""Deterministic, licensed demonstration-corpus release construction."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ste_compiler.ir.models import (
    Condition,
    Document,
    EntityRef,
    Instruction,
    Quantity,
    ReproducibilityMetadata,
    StateAssertion,
    TermReference,
)
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators import ValidationPipeline

from .records import DETERMINISTIC_REALIZER_PROFILE, build_training_record

RELEASE_SCHEMA_VERSION = "demonstration-corpus-release-v1"
RECORD_SCHEMA_VERSION = "demonstration-corpus-record-v1"
CONSTRUCTION_SCHEMA_VERSION = "demonstration-corpus-construction-v1"
CONSTRUCTION_FRONTEND = "deterministic-corpus-construction"
CONSTRUCTION_FRONTEND_VERSION = "0.1.0"
SPLITS = ("train", "validation", "test", "adversarial")
Split = Literal["train", "validation", "test", "adversarial"]
EXPECTED_RELEASE_FILES = frozenset(
    {
        "adversarial.jsonl",
        "checksums.sha256",
        "dataset-card.md",
        "feature-coverage.json",
        "leakage-report.json",
        "license-inventory.json",
        "manifest.json",
        "source-construction.json",
        "terminology.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "vocabulary.json",
    }
)

REQUIRED_FEATURES = frozenset(
    {
        "ambiguity",
        "causal_relation",
        "condition",
        "condition.exception",
        "document.multi_section",
        "document.multi_statement",
        "hazard",
        "instruction.actor",
        "instruction.indirect_object",
        "instruction.manner",
        "instruction.negated",
        "instruction.object",
        "instruction.purpose",
        "instruction.required_false",
        "quantity.comparator.at_least",
        "quantity.comparator.at_most",
        "quantity.comparator.equal",
        "quantity.comparator.less_than",
        "quantity.comparator.more_than",
        "quantity.tolerance",
        "reference",
        "referent.entity",
        "referent.term",
        "section.caution",
        "section.description",
        "section.note",
        "section.procedure",
        "section.warning",
        "source.unicode",
        "source.whitespace_newline",
        "state.value.quantity",
        "state.value.string",
        "statement.instruction",
        "statement.state",
        "temporal.after",
        "temporal.before",
    }
)


class StrictConstructionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConstructionRecord(StrictConstructionModel):
    id: str = Field(min_length=1, pattern=r"\S")
    split: Split
    source_id: str = Field(min_length=1, pattern=r"\S")
    source_text: str = Field(min_length=1)
    source_quotes: dict[str, str]
    license_id: str = Field(min_length=1, pattern=r"\S")
    document: dict[str, object]


class ResourceProvenance(StrictConstructionModel):
    version: str = Field(min_length=1, pattern=r"\S")
    license: str = Field(min_length=1, pattern=r"\S")
    origin: str = Field(min_length=1, pattern=r"\S")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConstructionProvenance(StrictConstructionModel):
    license_id: str = Field(min_length=1, pattern=r"\S")
    origin: str = Field(min_length=1, pattern=r"\S")
    vocabulary: ResourceProvenance
    terminology: ResourceProvenance


class CorpusConstruction(StrictConstructionModel):
    schema_version: Literal["demonstration-corpus-construction-v1"]
    dataset_version: str = Field(min_length=1, pattern=r"\S")
    seed: int = Field(ge=0)
    provenance: ConstructionProvenance
    records: tuple[ConstructionRecord, ...] = Field(min_length=1)


class ArtifactIdentity(TypedDict):
    path: str
    sha256: str
    bytes: int


class DemonstrationCorpusManifest(TypedDict):
    schema_version: str
    dataset_version: str
    construction_sha256: str
    seed: int
    record_count: int
    split_counts: dict[str, int]
    profiles: list[dict[str, str]]
    artifacts: list[ArtifactIdentity]


def _canonical_json(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_span(record: ConstructionRecord, node_id: str) -> dict[str, object]:
    try:
        quote = record.source_quotes[node_id]
    except KeyError as error:
        raise ValueError(
            f"construction record {record.id!r} has no source quote for {node_id!r}"
        ) from error
    if not quote.strip():
        raise ValueError(
            f"construction record {record.id!r} has a blank source quote for {node_id!r}"
        )
    start = record.source_text.find(quote)
    if start < 0:
        raise ValueError(
            f"construction record {record.id!r} quote for {node_id!r} is not in the source"
        )
    if record.source_text.find(quote, start + 1) >= 0:
        raise ValueError(f"construction record {record.id!r} quote for {node_id!r} is not unique")
    return {
        "source_id": record.source_id,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
    }


def _materialize_document(
    record: ConstructionRecord,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> Document:
    raw = deepcopy(record.document)
    sections = raw.get("sections")
    if not isinstance(sections, list):
        raise TypeError(f"construction record {record.id!r} document must contain sections")
    node_ids: set[str] = set()
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("statements"), list):
            raise TypeError(f"construction record {record.id!r} contains an invalid section")
        for statement in section["statements"]:
            if not isinstance(statement, dict):
                raise TypeError(f"construction record {record.id!r} contains an invalid statement")
            node_id = statement.get("id")
            if not isinstance(node_id, str):
                raise TypeError(f"construction record {record.id!r} has a statement without an id")
            if node_id in node_ids:
                raise ValueError(f"construction record {record.id!r} repeats node id {node_id!r}")
            node_ids.add(node_id)
            statement["source_spans"] = [_source_span(record, node_id)]
    ambiguities = raw.get("ambiguities", [])
    if not isinstance(ambiguities, list):
        raise TypeError(f"construction record {record.id!r} ambiguities must be a list")
    for ambiguity in ambiguities:
        if not isinstance(ambiguity, dict):
            raise TypeError(f"construction record {record.id!r} contains an invalid ambiguity")
        node_id = ambiguity.get("id")
        if not isinstance(node_id, str):
            raise TypeError(f"construction record {record.id!r} has an ambiguity without an id")
        if node_id in node_ids:
            raise ValueError(f"construction record {record.id!r} repeats node id {node_id!r}")
        node_ids.add(node_id)
        ambiguity["source_spans"] = [_source_span(record, node_id)]
    causal_relations = raw.get("causal_relations", [])
    if not isinstance(causal_relations, list):
        raise TypeError(f"construction record {record.id!r} causal_relations must be a list")
    for relation in causal_relations:
        if not isinstance(relation, dict):
            raise TypeError(
                f"construction record {record.id!r} contains an invalid causal relation"
            )
        node_id = relation.get("id")
        if not isinstance(node_id, str):
            raise TypeError(
                f"construction record {record.id!r} has a causal relation without an id"
            )
        if node_id in node_ids:
            raise ValueError(f"construction record {record.id!r} repeats node id {node_id!r}")
        node_ids.add(node_id)
        relation["source_spans"] = [_source_span(record, node_id)]
    unexpected_quotes = sorted(record.source_quotes.keys() - node_ids)
    if unexpected_quotes:
        raise ValueError(
            f"construction record {record.id!r} has source quotes for unknown nodes: "
            + ", ".join(unexpected_quotes)
        )

    raw["metadata"] = ReproducibilityMetadata(
        frontend=CONSTRUCTION_FRONTEND,
        frontend_version=CONSTRUCTION_FRONTEND_VERSION,
        realizer=DETERMINISTIC_REALIZER_PROFILE,
        realizer_version=DeterministicRealizer.version,
        vocabulary_version=vocabulary.data.version,
        terminology_version=terminology.data.version,
        validator_profile=ValidationPipeline.profile,
    ).model_dump(mode="json")
    document = Document.model_validate(raw)
    if document.id != record.id:
        raise ValueError(
            f"construction record id {record.id!r} does not match document id {document.id!r}"
        )
    return document


def _referent_feature(referent: EntityRef | TermReference | None, features: set[str]) -> None:
    if isinstance(referent, EntityRef):
        features.add("referent.entity")
    elif isinstance(referent, TermReference):
        features.add("referent.term")


def _quantity_features(quantity: Quantity, features: set[str]) -> None:
    features.add(f"quantity.comparator.{quantity.comparator}")
    features.add("quantity.unit")
    if quantity.tolerance is not None:
        features.add("quantity.tolerance")


def _condition_features(condition: Condition, features: set[str]) -> None:
    features.add("condition")
    _referent_feature(condition.subject, features)
    if condition.exception:
        features.add("condition.exception")
    if isinstance(condition.value, Quantity):
        _quantity_features(condition.value, features)


def _document_features(document: Document, source_text: str) -> tuple[str, ...]:
    features: set[str] = set()
    statements = [statement for section in document.sections for statement in section.statements]
    if len(document.sections) > 1:
        features.add("document.multi_section")
    if len(statements) > 1:
        features.add("document.multi_statement")
    if any(ord(character) > 127 for character in source_text):
        features.add("source.unicode")
    if "\n" in source_text:
        features.add("source.whitespace_newline")
    if "\t" in source_text:
        features.add("source.whitespace_tab")
    if document.ambiguities:
        features.add("ambiguity")
    if document.references:
        features.add("reference")
    if document.causal_relations:
        features.add("causal_relation")

    for section in document.sections:
        features.add(f"section.{section.kind.value}")
        for statement in section.statements:
            features.add(f"statement.{statement.kind}")
            for span in statement.source_spans:
                if span.quote is not None:
                    features.add("source_span.quote")
            if isinstance(statement, Instruction):
                _referent_feature(statement.actor, features)
                _referent_feature(statement.object, features)
                _referent_feature(statement.indirect_object, features)
                if statement.actor is not None:
                    features.add("instruction.actor")
                if statement.object is not None:
                    features.add("instruction.object")
                if statement.indirect_object is not None:
                    features.add("instruction.indirect_object")
                if statement.manner is not None:
                    features.add("instruction.manner")
                if statement.purpose is not None:
                    features.add("instruction.purpose")
                if statement.negated:
                    features.add("instruction.negated")
                if not statement.required:
                    features.add("instruction.required_false")
                for condition in statement.conditions:
                    _condition_features(condition, features)
                for relation in statement.temporal_relations:
                    features.add(f"temporal.{relation.relation}")
                for constraint in statement.quantity_constraints:
                    features.add("instruction.quantity_constraint")
                    _quantity_features(constraint.quantity, features)
                for hazard in statement.hazards:
                    features.add("hazard")
                    if hazard.threshold is not None:
                        _quantity_features(hazard.threshold, features)
            elif isinstance(statement, StateAssertion):
                _referent_feature(statement.subject, features)
                if isinstance(statement.value, Quantity):
                    features.add("state.value.quantity")
                    _quantity_features(statement.value, features)
                else:
                    features.add("state.value.string")
    return tuple(sorted(features))


def _normalized_source(source_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", source_text).casefold()
    return " ".join(normalized.split())


def _composition(features: tuple[str, ...]) -> tuple[str, ...]:
    prefixes = (
        "causal_relation",
        "condition",
        "hazard",
        "instruction.",
        "quantity.",
        "section.",
        "state.",
        "statement.",
        "temporal.",
    )
    return tuple(feature for feature in features if feature.startswith(prefixes))


def _resource_artifacts(
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> dict[str, bytes]:
    return {
        "vocabulary.json": _canonical_json(
            vocabulary.data.model_dump(mode="json"),
            indent=2,
        ),
        "terminology.json": _canonical_json(
            terminology.data.model_dump(mode="json"),
            indent=2,
        ),
    }


def _validate_provenance(
    construction: CorpusConstruction,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
    resource_artifacts: dict[str, bytes],
) -> None:
    resources = (
        ("vocabulary", construction.provenance.vocabulary, vocabulary.data),
        ("terminology", construction.provenance.terminology, terminology.data),
    )
    for name, declared, actual in resources:
        if declared.version != actual.version:
            raise ValueError(
                f"{name} version does not match construction provenance: "
                f"{actual.version!r} != {declared.version!r}"
            )
        if declared.license != actual.license:
            raise ValueError(f"{name} license does not match construction provenance")
        actual_sha256 = _sha256(resource_artifacts[f"{name}.json"])
        if declared.sha256 != actual_sha256:
            raise ValueError(f"{name} SHA-256 does not match construction provenance")
    mismatched_licenses = [
        record.id
        for record in construction.records
        if record.license_id != construction.provenance.license_id
    ]
    if mismatched_licenses:
        raise ValueError(
            "record licenses do not match construction provenance: "
            + ", ".join(mismatched_licenses)
        )


def _dataset_card(
    construction: CorpusConstruction,
    split_counts: dict[str, int],
    feature_count: int,
) -> bytes:
    return (
        "# ste-compiler demonstration corpus\n\n"
        f"Version: `{construction.dataset_version}`\n"
        f"License: `{construction.provenance.license_id}`\n"
        f"Origin: {construction.provenance.origin}\n"
        f"Construction seed: `{construction.seed}`\n\n"
        "This small, synthetic corpus demonstrates auditable technical-source to semantic-IR to "
        "controlled-text workflows. It is not ASD-STE100 data, does not reproduce the standard or "
        "its dictionary, and is not evidence of certification or production model quality.\n\n"
        "## Splits\n\n"
        + "\n".join(f"- `{split}`: {split_counts[split]} records" for split in SPLITS)
        + "\n\n"
        f"The release covers {feature_count} machine-reported semantic features. "
        "All records are schema-validated, deterministically realized, symbolized, and validated "
        "during construction. Test and adversarial records are evaluation-only.\n"
    ).encode("utf-8")


def _artifact_identity(path: str, data: bytes) -> ArtifactIdentity:
    return {"path": path, "sha256": _sha256(data), "bytes": len(data)}


def _write_release(output: Path, artifacts: dict[str, bytes]) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"demonstration corpus output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path, data in artifacts.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _regular_release_files(release: Path) -> dict[str, bytes]:
    if not release.is_dir() or release.is_symlink():
        raise ValueError(f"demonstration corpus release must be a directory: {release}")
    files: dict[str, bytes] = {}
    for path in release.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"demonstration corpus release contains an invalid entry: {path.name}")
        files[path.name] = path.read_bytes()
    if set(files) != EXPECTED_RELEASE_FILES:
        missing = sorted(EXPECTED_RELEASE_FILES - files.keys())
        unexpected = sorted(files.keys() - EXPECTED_RELEASE_FILES)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("demonstration corpus release file set is invalid: " + "; ".join(details))
    return files


def build_demonstration_corpus(
    construction_path: Path,
    output: Path,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> DemonstrationCorpusManifest:
    """Build one byte-reproducible demonstration dataset release."""

    construction = CorpusConstruction.model_validate_json(construction_path.read_bytes())
    if len({record.id for record in construction.records}) != len(construction.records):
        raise ValueError("construction records must have unique ids")
    if len({record.source_id for record in construction.records}) != len(construction.records):
        raise ValueError("construction records must have unique source ids")
    resource_artifacts = _resource_artifacts(vocabulary, terminology)
    _validate_provenance(
        construction,
        vocabulary,
        terminology,
        resource_artifacts,
    )

    split_records: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    coverage: Counter[str] = Counter()
    feature_splits: dict[str, set[str]] = defaultdict(set)
    profile_by_json: dict[str, dict[str, str]] = {}
    normalized_sources: dict[str, tuple[str, str]] = {}
    document_splits: dict[str, str] = {}
    train_compositions: set[tuple[str, ...]] = set()
    evaluation_compositions: list[tuple[str, tuple[str, ...]]] = []
    license_entries: list[dict[str, str]] = []

    for source_record in construction.records:
        try:
            document = _materialize_document(source_record, vocabulary, terminology)
        except TypeError as error:
            raise ValueError(
                f"invalid construction record {source_record.id!r}: {error}"
            ) from error
        training_record = build_training_record(document, vocabulary, terminology)
        features = _document_features(document, source_record.source_text)
        for feature in features:
            coverage[feature] += 1
            feature_splits[feature].add(source_record.split)

        normalized = _normalized_source(source_record.source_text)
        previous = normalized_sources.get(normalized)
        if previous is not None and previous[1] != source_record.split:
            raise ValueError(
                f"normalized source duplicate crosses splits: {previous[0]} and {source_record.id}"
            )
        normalized_sources[normalized] = (source_record.id, source_record.split)
        previous_split = document_splits.get(document.id)
        if previous_split is not None and previous_split != source_record.split:
            raise ValueError(f"document id {document.id!r} crosses splits")
        document_splits[document.id] = source_record.split

        composition = _composition(features)
        if source_record.split == "train":
            train_compositions.add(composition)
        elif source_record.split in {"test", "adversarial"}:
            evaluation_compositions.append((source_record.id, composition))

        source_bytes = source_record.source_text.encode("utf-8")
        released: dict[str, object] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "record_id": source_record.id,
            "split": source_record.split,
            "source": {
                "id": source_record.source_id,
                "text": source_record.source_text,
                "sha256": _sha256(source_bytes),
                "license_id": source_record.license_id,
            },
            "ir": document.model_dump(mode="json"),
            "serialized_ir": training_record["serialized_ir"],
            "text": training_record["text"],
            "symbols": training_record["symbols"],
            "allowed_symbols": training_record["allowed_symbols"],
            "metadata": training_record["metadata"],
            "features": list(features),
        }
        split_records[source_record.split].append(released)
        profile_json = json.dumps(
            training_record["metadata"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        profile_by_json[profile_json] = training_record["metadata"]
        license_entries.append(
            {
                "record_id": source_record.id,
                "source_id": source_record.source_id,
                "origin": construction.provenance.origin,
                "license_id": source_record.license_id,
            }
        )

    leaked_compositions = [
        record_id
        for record_id, composition in evaluation_compositions
        if composition in train_compositions
    ]
    if leaked_compositions:
        raise ValueError(
            "evaluation compositions duplicate training compositions: "
            + ", ".join(leaked_compositions)
        )
    missing_features = sorted(REQUIRED_FEATURES - coverage.keys())
    if missing_features:
        raise ValueError(
            "demonstration corpus is missing required features: " + ", ".join(missing_features)
        )
    split_counts = {split: len(split_records[split]) for split in SPLITS}
    empty_splits = [split for split, count in split_counts.items() if count == 0]
    if empty_splits:
        raise ValueError("demonstration corpus has empty splits: " + ", ".join(empty_splits))

    artifacts: dict[str, bytes] = {
        f"{split}.jsonl": b"".join(_canonical_json(record) for record in split_records[split])
        for split in SPLITS
    }
    artifacts["source-construction.json"] = _canonical_json(
        construction.model_dump(mode="json"),
        indent=2,
    )
    artifacts["license-inventory.json"] = _canonical_json(
        {
            "schema_version": "license-inventory-v1",
            "dataset_version": construction.dataset_version,
            "resources": [
                {
                    "path": "vocabulary.json",
                    "origin": construction.provenance.vocabulary.origin,
                    "license": construction.provenance.vocabulary.license,
                    "version": construction.provenance.vocabulary.version,
                    "sha256": construction.provenance.vocabulary.sha256,
                },
                {
                    "path": "terminology.json",
                    "origin": construction.provenance.terminology.origin,
                    "license": construction.provenance.terminology.license,
                    "version": construction.provenance.terminology.version,
                    "sha256": construction.provenance.terminology.sha256,
                },
            ],
            "entries": license_entries,
        },
        indent=2,
    )
    artifacts.update(resource_artifacts)
    artifacts["feature-coverage.json"] = _canonical_json(
        {
            "schema_version": "feature-coverage-v1",
            "required_features": sorted(REQUIRED_FEATURES),
            "missing_features": [],
            "features": {
                feature: {
                    "count": coverage[feature],
                    "splits": sorted(feature_splits[feature]),
                }
                for feature in sorted(coverage)
            },
        },
        indent=2,
    )
    artifacts["leakage-report.json"] = _canonical_json(
        {
            "schema_version": "leakage-report-v1",
            "document_id_cross_split": [],
            "normalized_source_cross_split": [],
            "evaluation_composition_in_train": [],
        },
        indent=2,
    )
    artifacts["dataset-card.md"] = _dataset_card(
        construction,
        split_counts,
        len(coverage),
    )

    artifact_identities = [
        _artifact_identity(path, data) for path, data in sorted(artifacts.items())
    ]
    manifest = DemonstrationCorpusManifest(
        schema_version=RELEASE_SCHEMA_VERSION,
        dataset_version=construction.dataset_version,
        construction_sha256=_sha256(artifacts["source-construction.json"]),
        seed=construction.seed,
        record_count=sum(split_counts.values()),
        split_counts=split_counts,
        profiles=[profile_by_json[key] for key in sorted(profile_by_json)],
        artifacts=artifact_identities,
    )
    artifacts["manifest.json"] = _canonical_json(manifest, indent=2)
    artifacts["checksums.sha256"] = "".join(
        f"{_sha256(data)}  {path}\n" for path, data in sorted(artifacts.items())
    ).encode("utf-8")
    _write_release(output, artifacts)
    return manifest


def verify_demonstration_corpus(release: Path) -> DemonstrationCorpusManifest:
    """Reconstruct a release from its embedded inputs and require byte identity."""

    files = _regular_release_files(release)
    vocabulary = Vocabulary.load(release / "vocabulary.json")
    terminology = TerminologyRegistry.load(release / "terminology.json")
    with tempfile.TemporaryDirectory(prefix="ste-compiler-corpus-verify-") as temporary:
        rebuilt = Path(temporary) / "release"
        manifest = build_demonstration_corpus(
            release / "source-construction.json",
            rebuilt,
            vocabulary,
            terminology,
        )
        rebuilt_files = _regular_release_files(rebuilt)
    mismatches = [
        path for path in sorted(EXPECTED_RELEASE_FILES) if files[path] != rebuilt_files[path]
    ]
    if mismatches:
        raise ValueError(
            "demonstration corpus release does not reproduce: " + ", ".join(mismatches)
        )
    return manifest

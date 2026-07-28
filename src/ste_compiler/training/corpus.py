from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from ste_compiler.ir.serialization import load_document
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

from .records import TrainingRecord, build_training_record

CORPUS_SCHEMA_VERSION = "symbolic-corpus-v1"
IR_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
OUTPUT_ARTIFACTS = ("corpus.jsonl", "manifest.json")


class CorpusManifest(TypedDict):
    schema_version: str
    record_count: int
    corpus_sha256: str
    source_files: list[str]
    profiles: list[dict[str, str]]


def _input_paths(source: Path, output: Path) -> tuple[Path, list[Path]]:
    if source.is_symlink():
        raise ValueError(f"symbolic corpus source must not be a symlink: {source}")
    if source.is_file():
        if source.suffix.casefold() not in IR_SUFFIXES:
            raise ValueError(f"unsupported IR file type: {source.suffix or '<none>'}")
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError(f"IR source does not exist: {source}")

    source_root = source.resolve()
    output_root = output.resolve()
    output_is_nested_in_source = output_root != source_root and output_root.is_relative_to(
        source_root
    )
    artifact_locations = {output_root / artifact_name for artifact_name in OUTPUT_ARTIFACTS}
    candidates = sorted(
        (path for path in source.rglob("*") if path.suffix.casefold() in IR_SUFFIXES),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    paths: list[Path] = []
    for path in candidates:
        source_path = path.relative_to(source).as_posix()
        path_location = path.parent.resolve() / path.name
        if path_location in artifact_locations or (
            output_is_nested_in_source and path_location.is_relative_to(output_root)
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"symbolic corpus source contains a symlinked IR file: {source_path}")
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(source_root):
            raise ValueError(f"IR file resolves outside the corpus source: {source_path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"no YAML or JSON IR documents found in {source}")
    return source, paths


def _canonical_line(record: TrainingRecord) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def _reject_output_aliases(paths: list[Path], output: Path) -> None:
    artifacts = [output / artifact_name for artifact_name in OUTPUT_ARTIFACTS]
    for artifact in artifacts:
        for source in paths:
            if _paths_alias(artifact, source):
                raise ValueError(
                    f"symbolic corpus output artifact {artifact} aliases source IR file {source}"
                )
    first, second = artifacts
    if _paths_alias(first, second):
        raise ValueError(f"symbolic corpus output artifacts {first} and {second} alias each other")
    for artifact in artifacts:
        if artifact.is_symlink():
            raise ValueError(f"symbolic corpus output artifact must not be a symlink: {artifact}")
        if artifact.exists() and (not artifact.is_file() or artifact.stat().st_nlink > 1):
            raise ValueError(
                f"symbolic corpus output artifact must be a regular single-link file: {artifact}"
            )


def export_symbolic_corpus(
    source: Path,
    output: Path,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> CorpusManifest:
    root, paths = _input_paths(source, output)
    _reject_output_aliases(paths, output)
    records: list[TrainingRecord] = []
    document_sources: dict[str, str] = {}
    for path in paths:
        source_path = path.relative_to(root).as_posix()
        document = load_document(path)
        previous_source = document_sources.get(document.id)
        if previous_source is not None:
            raise ValueError(
                f"duplicate document id {document.id!r} in {previous_source} and {source_path}"
            )
        document_sources[document.id] = source_path
        records.append(
            build_training_record(
                document,
                vocabulary,
                terminology,
                source_path=source_path,
            )
        )

    corpus_bytes = "".join(_canonical_line(record) for record in records).encode("utf-8")
    profile_by_json = {
        json.dumps(record["metadata"], ensure_ascii=False, separators=(",", ":"), sort_keys=True): (
            record["metadata"]
        )
        for record in records
    }
    manifest = CorpusManifest(
        schema_version=CORPUS_SCHEMA_VERSION,
        record_count=len(records),
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        source_files=[record["source_path"] for record in records],
        profiles=[profile_by_json[key] for key in sorted(profile_by_json)],
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus.jsonl").write_bytes(corpus_bytes)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

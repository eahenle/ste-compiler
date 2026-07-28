from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Protocol, TypedDict, cast

from ste_compiler.ir.serialization import load_document
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

from .records import TrainingRecord, build_training_record

CORPUS_SCHEMA_VERSION = "symbolic-corpus-v1"
IR_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
OUTPUT_ARTIFACTS = ("corpus.jsonl", "manifest.json")
OUTPUT_LOCK = ".ste-compiler-corpus.lock"
TRANSACTION_JOURNAL = ".ste-compiler-corpus.transaction.json"
TRANSACTION_BACKUPS = {
    "corpus.jsonl": ".ste-compiler-corpus.jsonl.bak",
    "manifest.json": ".ste-compiler-manifest.json.bak",
}
TRANSACTION_KEYS = frozenset({"version", "had_previous_pair"})
MANIFEST_KEYS = frozenset(
    {"schema_version", "record_count", "corpus_sha256", "source_files", "profiles"}
)
PROFILE_KEYS = frozenset(
    {
        "frontend",
        "frontend_version",
        "realizer",
        "realizer_version",
        "vocabulary_version",
        "terminology_version",
        "validator_profile",
    }
)
CORPUS_RECORD_KEYS = frozenset(
    {
        "document_id",
        "serialized_ir",
        "symbols",
        "allowed_symbols",
        "text",
        "metadata",
        "source_path",
    }
)


class CorpusManifest(TypedDict):
    schema_version: str
    record_count: int
    corpus_sha256: str
    source_files: list[str]
    profiles: list[dict[str, str]]


class _TransactionJournal(TypedDict):
    version: int
    had_previous_pair: bool


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


def _string_mapping(value: object, keys: frozenset[str]) -> dict[str, str] | None:
    if not isinstance(value, dict) or set(value) != keys:
        return None
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return None
    return {key: item for key, item in value.items()}


def _is_generated_pair(manifest_bytes: bytes, corpus_bytes: bytes) -> bool:
    try:
        manifest: object = json.loads(manifest_bytes)
        records: list[object] = [json.loads(line) for line in corpus_bytes.splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        return False

    schema_version = manifest["schema_version"]
    record_count = manifest["record_count"]
    corpus_sha256 = manifest["corpus_sha256"]
    source_files = manifest["source_files"]
    profiles = manifest["profiles"]
    if (
        schema_version != CORPUS_SCHEMA_VERSION
        or type(record_count) is not int
        or record_count < 1
        or not isinstance(corpus_sha256, str)
        or len(corpus_sha256) != 64
        or any(character not in "0123456789abcdef" for character in corpus_sha256)
        or not isinstance(source_files, list)
        or not all(isinstance(source_path, str) for source_path in source_files)
        or not isinstance(profiles, list)
    ):
        return False
    normalized_profiles: list[dict[str, str]] = []
    for profile in profiles:
        normalized = _string_mapping(profile, PROFILE_KEYS)
        if normalized is None:
            return False
        normalized_profiles.append(normalized)

    record_sources: list[str] = []
    record_profiles: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != CORPUS_RECORD_KEYS:
            return False
        string_fields = ("document_id", "serialized_ir", "symbols", "text", "source_path")
        if not all(isinstance(record[field], str) for field in string_fields):
            return False
        allowed_symbols = record["allowed_symbols"]
        if not isinstance(allowed_symbols, list) or not all(
            isinstance(symbol, str) for symbol in allowed_symbols
        ):
            return False
        metadata = _string_mapping(record["metadata"], PROFILE_KEYS)
        if metadata is None:
            return False
        record_sources.append(record["source_path"])
        metadata_json = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        record_profiles[metadata_json] = metadata

    expected_profiles = [record_profiles[key] for key in sorted(record_profiles)]
    return (
        record_count == len(records)
        and source_files == record_sources
        and normalized_profiles == expected_profiles
        and corpus_sha256 == hashlib.sha256(corpus_bytes).hexdigest()
    )


def _is_prior_generated_manifest(manifest_path: Path, corpus_path: Path) -> bool:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory_fd = os.open(manifest_path.parent, directory_flags)
    except OSError:
        return False
    try:
        manifest_bytes = _read_regular_single_link_entry(directory_fd, manifest_path.name)
        corpus_bytes = _read_regular_single_link_entry(directory_fd, corpus_path.name)
    finally:
        os.close(directory_fd)
    return (
        manifest_bytes is not None
        and corpus_bytes is not None
        and _is_generated_pair(manifest_bytes, corpus_bytes)
    )


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
    paths: list[Path] = []
    pending_directories = [source]
    while pending_directories:
        directory = pending_directories.pop()
        child_directories: list[Path] = []
        for path in sorted(directory.iterdir(), key=lambda child: child.name):
            source_path = path.relative_to(source).as_posix()
            path_location = path.parent.resolve() / path.name
            if output_is_nested_in_source and path_location.is_relative_to(output_root):
                continue
            if path_location in artifact_locations:
                prior_manifest = (
                    output_root == source_root
                    and path.name == "manifest.json"
                    and _is_prior_generated_manifest(path, output_root / "corpus.jsonl")
                )
                if path.name != "manifest.json" or prior_manifest:
                    continue
            if path.is_symlink():
                if not path.exists() or path.is_dir():
                    raise ValueError(
                        f"symbolic corpus source contains a symlinked directory: {source_path}"
                    )
                if path.suffix.casefold() in IR_SUFFIXES:
                    raise ValueError(
                        f"symbolic corpus source contains a symlinked IR file: {source_path}"
                    )
                continue
            if path.is_dir():
                child_directories.append(path)
                continue
            if not path.is_file() or path.suffix.casefold() not in IR_SUFFIXES:
                continue
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(source_root):
                raise ValueError(f"IR file resolves outside the corpus source: {source_path}")
            paths.append(path)
        pending_directories.extend(reversed(child_directories))
    paths.sort(key=lambda path: path.relative_to(source).as_posix())
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


def _unlink_entry(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _write_temporary_artifact(directory_fd: int, data: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    for _ in range(128):
        name = f".ste-compiler-{secrets.token_hex(16)}.tmp"
        try:
            file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(file_fd, remaining)
                if written == 0:
                    raise OSError("failed to write complete corpus artifact")
                remaining = remaining[written:]
            os.fsync(file_fd)
            os.close(file_fd)
            file_fd = -1
        except BaseException:
            if file_fd >= 0:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
            _unlink_entry(directory_fd, name)
            raise
        return name
    raise FileExistsError("could not create a unique temporary corpus artifact")


def _atomic_replace(directory_fd: int, temporary_name: str, artifact_name: str) -> None:
    os.replace(
        temporary_name,
        artifact_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


def _fsync_directory(directory_fd: int) -> None:
    try:
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_regular_single_link_entry(directory_fd: int, name: str) -> bytes | None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ELOOP}:
            return None
        raise
    try:
        opened = os.fstat(file_fd)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            return None
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _fcntl_module() -> _FcntlModule:
    try:
        module = importlib.import_module("fcntl")
    except ModuleNotFoundError as error:
        raise ValueError(
            "symbolic corpus export requires POSIX fcntl file locking; "
            "other ste-compiler commands remain supported on this platform"
        ) from error
    return cast(_FcntlModule, module)


def _acquire_output_lock(directory_fd: int, output: Path) -> int:
    lock_module = _fcntl_module()
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        lock_fd = os.open(OUTPUT_LOCK, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(
            f"symbolic corpus output lock must be a regular single-link file: {output / OUTPUT_LOCK}"
        ) from error
    try:
        lock_module.flock(lock_fd, lock_module.LOCK_EX)
        opened = os.fstat(lock_fd)
        entry = os.stat(OUTPUT_LOCK, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            raise ValueError(
                "symbolic corpus output lock must be a regular single-link file: "
                f"{output / OUTPUT_LOCK}"
            )
    except BaseException:
        os.close(lock_fd)
        raise
    return lock_fd


def _release_output_lock(lock_fd: int) -> None:
    lock_module = _fcntl_module()
    try:
        lock_module.flock(lock_fd, lock_module.LOCK_UN)
    finally:
        os.close(lock_fd)


def _is_owned_temporary(name: str) -> bool:
    prefix = ".ste-compiler-"
    suffix = ".tmp"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix) : -len(suffix)]
    return len(token) == 32 and all(character in "0123456789abcdef" for character in token)


def _clean_owned_temporary_files(directory_fd: int) -> bool:
    removed = False
    for name in os.listdir(directory_fd):
        if _is_owned_temporary(name):
            _unlink_entry(directory_fd, name)
            removed = True
    return removed


def _write_transaction_journal(directory_fd: int, had_previous_pair: bool) -> None:
    journal = _TransactionJournal(version=1, had_previous_pair=had_previous_pair)
    journal_bytes = (
        json.dumps(journal, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name = _write_temporary_artifact(directory_fd, journal_bytes)
    try:
        _atomic_replace(directory_fd, temporary_name, TRANSACTION_JOURNAL)
    except BaseException:
        _unlink_entry(directory_fd, temporary_name)
        raise
    _fsync_directory(directory_fd)


def _read_transaction_journal(directory_fd: int) -> _TransactionJournal | None:
    if not _entry_exists(directory_fd, TRANSACTION_JOURNAL):
        return None
    journal_bytes = _read_regular_single_link_entry(directory_fd, TRANSACTION_JOURNAL)
    if journal_bytes is None:
        raise ValueError("symbolic corpus transaction journal must be a regular single-link file")
    try:
        value: object = json.loads(journal_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("symbolic corpus transaction journal is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != TRANSACTION_KEYS
        or value["version"] != 1
        or type(value["had_previous_pair"]) is not bool
    ):
        raise ValueError("symbolic corpus transaction journal is invalid")
    return _TransactionJournal(
        version=1,
        had_previous_pair=value["had_previous_pair"],
    )


def _final_pair_is_coherent(directory_fd: int) -> bool:
    corpus_bytes = _read_regular_single_link_entry(directory_fd, "corpus.jsonl")
    manifest_bytes = _read_regular_single_link_entry(directory_fd, "manifest.json")
    return (
        corpus_bytes is not None
        and manifest_bytes is not None
        and _is_generated_pair(manifest_bytes, corpus_bytes)
    )


def _remove_transaction_metadata(directory_fd: int) -> None:
    for backup_name in TRANSACTION_BACKUPS.values():
        _unlink_entry(directory_fd, backup_name)
    _fsync_directory(directory_fd)
    _unlink_entry(directory_fd, TRANSACTION_JOURNAL)
    _clean_owned_temporary_files(directory_fd)
    _fsync_directory(directory_fd)


def _recover_interrupted_publication(directory_fd: int) -> None:
    journal = _read_transaction_journal(directory_fd)
    if journal is None:
        orphaned_backups = [
            backup_name
            for backup_name in TRANSACTION_BACKUPS.values()
            if _entry_exists(directory_fd, backup_name)
        ]
        if orphaned_backups:
            raise ValueError(
                "symbolic corpus output contains transaction backups without a journal"
            )
        if _clean_owned_temporary_files(directory_fd):
            _fsync_directory(directory_fd)
        return

    if _final_pair_is_coherent(directory_fd):
        _remove_transaction_metadata(directory_fd)
        return

    if journal["had_previous_pair"]:
        for artifact_name, backup_name in TRANSACTION_BACKUPS.items():
            if _entry_exists(directory_fd, backup_name):
                if _read_regular_single_link_entry(directory_fd, backup_name) is None:
                    raise ValueError(
                        "symbolic corpus transaction backup must be a regular "
                        f"single-link file: {backup_name}"
                    )
                _unlink_entry(directory_fd, artifact_name)
                os.replace(
                    backup_name,
                    artifact_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            elif not _entry_exists(directory_fd, artifact_name):
                raise ValueError(
                    "symbolic corpus transaction cannot restore the previous artifact pair"
                )
        _fsync_directory(directory_fd)
    else:
        for artifact_name in OUTPUT_ARTIFACTS:
            _unlink_entry(directory_fd, artifact_name)
        for backup_name in TRANSACTION_BACKUPS.values():
            _unlink_entry(directory_fd, backup_name)
        _fsync_directory(directory_fd)

    _unlink_entry(directory_fd, TRANSACTION_JOURNAL)
    _clean_owned_temporary_files(directory_fd)
    _fsync_directory(directory_fd)


def _publish_artifacts(
    output: Path,
    source_paths: list[Path],
    artifacts: tuple[tuple[str, bytes], ...],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open(output, directory_flags)
    lock_fd = -1
    temporary_names: dict[str, str] = {}
    try:
        lock_fd = _acquire_output_lock(directory_fd, output)
        _recover_interrupted_publication(directory_fd)
        _reject_output_aliases(source_paths, output)
        existing = {
            artifact_name: _entry_exists(directory_fd, artifact_name)
            for artifact_name in OUTPUT_ARTIFACTS
        }
        if any(existing.values()) and not all(existing.values()):
            raise ValueError(
                "symbolic corpus output must contain both corpus.jsonl and manifest.json or neither"
            )
        had_previous_pair = all(existing.values())

        for artifact_name, data in artifacts:
            temporary_names[artifact_name] = _write_temporary_artifact(directory_fd, data)

        _write_transaction_journal(directory_fd, had_previous_pair)
        for artifact_name, _ in artifacts:
            if not _entry_exists(directory_fd, artifact_name):
                continue
            backup_name = TRANSACTION_BACKUPS[artifact_name]
            os.replace(
                artifact_name,
                backup_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            _fsync_directory(directory_fd)

        for artifact_name, _ in artifacts:
            temporary_name = temporary_names[artifact_name]
            _atomic_replace(directory_fd, temporary_name, artifact_name)
            del temporary_names[artifact_name]
            _fsync_directory(directory_fd)

        if not _final_pair_is_coherent(directory_fd):
            raise OSError("published symbolic corpus artifacts are not a coherent pair")
        _remove_transaction_metadata(directory_fd)
    except BaseException:
        if lock_fd >= 0:
            _recover_interrupted_publication(directory_fd)
        raise
    finally:
        for temporary_name in temporary_names.values():
            _unlink_entry(directory_fd, temporary_name)
        try:
            if lock_fd >= 0:
                _release_output_lock(lock_fd)
        finally:
            os.close(directory_fd)


def export_symbolic_corpus(
    source: Path,
    output: Path,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> CorpusManifest:
    _fcntl_module()
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

    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_artifacts(
        output,
        paths,
        (
            ("corpus.jsonl", corpus_bytes),
            ("manifest.json", manifest_bytes),
        ),
    )
    return manifest

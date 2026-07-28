from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import NamedTuple, Protocol, TypedDict, cast

from ste_compiler.ir.serialization import load_document
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

from .records import TrainingRecord, build_training_record

CORPUS_SCHEMA_VERSION = "symbolic-corpus-v1"
IR_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
OUTPUT_ARTIFACTS = ("corpus.jsonl", "manifest.json")
OUTPUT_LOCK = ".ste-compiler-corpus.lock"
OUTPUT_LOCK_BYTES = b"ste-compiler-symbolic-corpus-lock-v1\n"
CURRENT_SELECTOR = "current"
GENERATIONS_DIRECTORY = "generations"
GENERATIONS_MARKER = ".ste-compiler-owned"
GENERATIONS_MARKER_BYTES = b"ste-compiler-symbolic-corpus-generations-v1\n"
GENERATION_PREFIX = "sha256-"
GENERATION_STAGE_PREFIX = ".ste-compiler-generation-"
GENERATIONS_STAGE_PREFIX = ".ste-compiler-generations-"
STAGE_SUFFIX = ".stage"
CURRENT_TEMP_PREFIX = ".ste-compiler-current-"
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


class SymbolicCorpusSnapshot(NamedTuple):
    generation_id: str
    corpus_bytes: bytes
    manifest: CorpusManifest


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
    generations_path = output_root / GENERATIONS_DIRECTORY
    owned_generations = (
        output_root == source_root
        and generations_path.is_dir()
        and not generations_path.is_symlink()
        and (generations_path / GENERATIONS_MARKER).is_file()
        and not (generations_path / GENERATIONS_MARKER).is_symlink()
        and (generations_path / GENERATIONS_MARKER).read_bytes() == GENERATIONS_MARKER_BYTES
    )
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
            if (
                output_root == source_root
                and owned_generations
                and path.parent.resolve() == output_root
                and path.name in {CURRENT_SELECTOR, GENERATIONS_DIRECTORY}
            ):
                continue
            if (
                output_root == source_root
                and path_location == output_root / "manifest.json"
                and _is_prior_generated_manifest(path, output_root / "corpus.jsonl")
            ):
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


def _is_hex_token(value: str, length: int = 32) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _is_generation_id(value: str) -> bool:
    return value.startswith(GENERATION_PREFIX) and _is_hex_token(
        value[len(GENERATION_PREFIX) :], 64
    )


def _generation_id(corpus_bytes: bytes, manifest_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"ste-compiler-symbolic-corpus-generation-v1\0")
    for artifact_bytes in (corpus_bytes, manifest_bytes):
        digest.update(len(artifact_bytes).to_bytes(8, "big"))
        digest.update(artifact_bytes)
    return f"{GENERATION_PREFIX}{digest.hexdigest()}"


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_entry(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _fsync_directory(directory_fd: int) -> None:
    try:
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def _open_directory_entry(directory_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=directory_fd)


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


def _write_new_file(directory_fd: int, name: str, data: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    file_fd = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(file_fd, remaining)
            if written == 0:
                raise OSError(f"failed to write complete corpus artifact {name}")
            remaining = remaining[written:]
        os.fsync(file_fd)
    finally:
        os.close(file_fd)


def _atomic_replace(directory_fd: int, temporary_name: str, artifact_name: str) -> None:
    os.replace(
        temporary_name,
        artifact_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


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
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    created = False
    try:
        lock_fd = os.open(
            OUTPUT_LOCK,
            flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
    except FileExistsError:
        try:
            lock_fd = os.open(OUTPUT_LOCK, flags, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(
                "symbolic corpus output lock must be a regular single-link file: "
                f"{output / OUTPUT_LOCK}"
            ) from error
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
        os.lseek(lock_fd, 0, os.SEEK_SET)
        lock_bytes = os.read(lock_fd, len(OUTPUT_LOCK_BYTES) + 1)
        if created or (lock_bytes == b"" and _root_pair_is_coherent(directory_fd)):
            os.lseek(lock_fd, 0, os.SEEK_SET)
            if os.write(lock_fd, OUTPUT_LOCK_BYTES) != len(OUTPUT_LOCK_BYTES):
                raise OSError("failed to write complete symbolic corpus lock marker")
            os.fsync(lock_fd)
        elif lock_bytes != OUTPUT_LOCK_BYTES:
            raise ValueError(
                f"symbolic corpus output lock is not tool-owned: {output / OUTPUT_LOCK}"
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


def _root_pair_is_coherent(directory_fd: int) -> bool:
    corpus_bytes = _read_regular_single_link_entry(directory_fd, "corpus.jsonl")
    manifest_bytes = _read_regular_single_link_entry(directory_fd, "manifest.json")
    return (
        corpus_bytes is not None
        and manifest_bytes is not None
        and _is_generated_pair(manifest_bytes, corpus_bytes)
    )


def _stage_token(name: str, prefix: str, suffix: str) -> str | None:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    token = name[len(prefix) : -len(suffix)]
    return token if _is_hex_token(token) else None


def _remove_owned_stage(
    parent_fd: int,
    stage_name: str,
    *,
    allowed_entries: frozenset[str],
) -> None:
    try:
        stage_fd = _open_directory_entry(parent_fd, stage_name)
    except OSError as error:
        raise ValueError(
            f"symbolic corpus staging path is not a directory: {stage_name}"
        ) from error
    try:
        entries = set(os.listdir(stage_fd))
        unexpected = entries - allowed_entries
        if unexpected:
            raise ValueError(
                f"symbolic corpus staging directory contains unexpected entries: {stage_name}"
            )
        os.fchmod(stage_fd, 0o700)
        for entry in entries:
            _unlink_entry(stage_fd, entry)
        _fsync_directory(stage_fd)
    finally:
        os.close(stage_fd)
    os.rmdir(stage_name, dir_fd=parent_fd)
    _fsync_directory(parent_fd)


def _cleanup_output_staging(output_fd: int, generations_fd: int) -> None:
    changed = False
    for name in os.listdir(output_fd):
        if _stage_token(name, GENERATIONS_STAGE_PREFIX, STAGE_SUFFIX) is not None:
            _remove_owned_stage(
                output_fd,
                name,
                allowed_entries=frozenset({GENERATIONS_MARKER}),
            )
            continue
        if _stage_token(name, CURRENT_TEMP_PREFIX, ".tmp") is not None:
            entry = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
            if not stat.S_ISLNK(entry.st_mode):
                raise ValueError(f"symbolic corpus selector staging path is invalid: {name}")
            target = os.readlink(name, dir_fd=output_fd)
            prefix = f"{GENERATIONS_DIRECTORY}/"
            generation_id = target[len(prefix) :] if target.startswith(prefix) else ""
            if "/" in generation_id or not _is_generation_id(generation_id):
                raise ValueError(f"symbolic corpus selector staging path is invalid: {name}")
            _validate_generation(generations_fd, generation_id)
            _unlink_entry(output_fd, name)
            changed = True
            continue
    if changed:
        _fsync_directory(output_fd)


def _cleanup_generation_staging(generations_fd: int) -> None:
    for name in os.listdir(generations_fd):
        if _stage_token(name, GENERATION_STAGE_PREFIX, STAGE_SUFFIX) is not None:
            _remove_owned_stage(
                generations_fd,
                name,
                allowed_entries=frozenset(OUTPUT_ARTIFACTS),
            )


def _create_generations_directory(output_fd: int) -> None:
    for _ in range(128):
        stage_name = f"{GENERATIONS_STAGE_PREFIX}{secrets.token_hex(16)}{STAGE_SUFFIX}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=output_fd)
        except FileExistsError:
            continue
        stage_fd = _open_directory_entry(output_fd, stage_name)
        try:
            _write_new_file(
                stage_fd,
                GENERATIONS_MARKER,
                GENERATIONS_MARKER_BYTES,
            )
            _fsync_directory(stage_fd)
        finally:
            os.close(stage_fd)
        os.rename(
            stage_name,
            GENERATIONS_DIRECTORY,
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        _fsync_directory(output_fd)
        return
    raise FileExistsError("could not create corpus generations directory")


def _validate_generation(generations_fd: int, generation_id: str) -> tuple[bytes, bytes]:
    if not _is_generation_id(generation_id):
        raise ValueError(f"invalid symbolic corpus generation id: {generation_id}")
    try:
        generation_fd = _open_directory_entry(generations_fd, generation_id)
    except OSError as error:
        raise ValueError(
            f"symbolic corpus generation must be an immutable directory: {generation_id}"
        ) from error
    try:
        if set(os.listdir(generation_fd)) != set(OUTPUT_ARTIFACTS):
            raise ValueError(f"symbolic corpus generation has unexpected entries: {generation_id}")
        corpus_bytes = _read_regular_single_link_entry(generation_fd, "corpus.jsonl")
        manifest_bytes = _read_regular_single_link_entry(generation_fd, "manifest.json")
    finally:
        os.close(generation_fd)
    if (
        corpus_bytes is None
        or manifest_bytes is None
        or not _is_generated_pair(manifest_bytes, corpus_bytes)
        or _generation_id(corpus_bytes, manifest_bytes) != generation_id
    ):
        raise ValueError(f"symbolic corpus generation is not coherent: {generation_id}")
    return corpus_bytes, manifest_bytes


def _open_generations_directory(
    output_fd: int,
    *,
    create: bool,
    validate_all: bool,
) -> int:
    if not _entry_exists(output_fd, GENERATIONS_DIRECTORY):
        if not create:
            raise ValueError("symbolic corpus output does not contain a generations directory")
        _create_generations_directory(output_fd)
    try:
        generations_fd = _open_directory_entry(output_fd, GENERATIONS_DIRECTORY)
    except OSError as error:
        raise ValueError(
            "symbolic corpus generations path must be a tool-owned directory"
        ) from error
    try:
        marker = _read_regular_single_link_entry(generations_fd, GENERATIONS_MARKER)
        if marker != GENERATIONS_MARKER_BYTES:
            raise ValueError("symbolic corpus generations directory is not tool-owned")
        if validate_all:
            _cleanup_generation_staging(generations_fd)
            for name in os.listdir(generations_fd):
                if name != GENERATIONS_MARKER:
                    _validate_generation(generations_fd, name)
    except BaseException:
        os.close(generations_fd)
        raise
    return generations_fd


def _current_generation_id(
    output_fd: int,
    generations_fd: int,
    *,
    validate_generation: bool = True,
) -> str | None:
    if not _entry_exists(output_fd, CURRENT_SELECTOR):
        return None
    entry = os.stat(CURRENT_SELECTOR, dir_fd=output_fd, follow_symlinks=False)
    if not stat.S_ISLNK(entry.st_mode):
        raise ValueError("symbolic corpus current selector must be a tool-owned symbolic link")
    target = os.readlink(CURRENT_SELECTOR, dir_fd=output_fd)
    prefix = f"{GENERATIONS_DIRECTORY}/"
    if not target.startswith(prefix):
        raise ValueError("symbolic corpus current selector has an invalid target")
    generation_id = target[len(prefix) :]
    if "/" in generation_id or not _is_generation_id(generation_id):
        raise ValueError("symbolic corpus current selector has an invalid target")
    if validate_generation:
        _validate_generation(generations_fd, generation_id)
    return generation_id


def _preflight_management_paths(output_fd: int) -> None:
    generations_exists = _entry_exists(output_fd, GENERATIONS_DIRECTORY)
    current_exists = _entry_exists(output_fd, CURRENT_SELECTOR)
    if current_exists and not generations_exists:
        raise ValueError(
            "symbolic corpus current selector exists without a tool-owned generations directory"
        )
    if not generations_exists:
        return
    generations_fd = _open_generations_directory(
        output_fd,
        create=False,
        validate_all=False,
    )
    try:
        for name in os.listdir(generations_fd):
            if name == GENERATIONS_MARKER:
                continue
            if _stage_token(name, GENERATION_STAGE_PREFIX, STAGE_SUFFIX) is not None:
                continue
            _validate_generation(generations_fd, name)
        if current_exists:
            _current_generation_id(output_fd, generations_fd)
    finally:
        os.close(generations_fd)


def _ensure_generation(
    generations_fd: int,
    generation_id: str,
    artifacts: tuple[tuple[str, bytes], ...],
) -> None:
    expected = dict(artifacts)
    if _entry_exists(generations_fd, generation_id):
        corpus_bytes, manifest_bytes = _validate_generation(generations_fd, generation_id)
        if corpus_bytes != expected["corpus.jsonl"] or manifest_bytes != expected["manifest.json"]:
            raise ValueError(f"symbolic corpus generation id collision: {generation_id}")
        return

    for _ in range(128):
        stage_name = f"{GENERATION_STAGE_PREFIX}{secrets.token_hex(16)}{STAGE_SUFFIX}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=generations_fd)
        except FileExistsError:
            continue
        stage_fd = _open_directory_entry(generations_fd, stage_name)
        try:
            for artifact_name, data in artifacts:
                _write_new_file(stage_fd, artifact_name, data, mode=0o444)
            _fsync_directory(stage_fd)
            os.fchmod(stage_fd, 0o500)
        finally:
            os.close(stage_fd)
        os.rename(
            stage_name,
            generation_id,
            src_dir_fd=generations_fd,
            dst_dir_fd=generations_fd,
        )
        _fsync_directory(generations_fd)
        _validate_generation(generations_fd, generation_id)
        return
    raise FileExistsError("could not create a unique corpus generation staging directory")


def _switch_current_generation(output_fd: int, generation_id: str) -> None:
    target = f"{GENERATIONS_DIRECTORY}/{generation_id}"
    for _ in range(128):
        temporary_name = f"{CURRENT_TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
        try:
            os.symlink(target, temporary_name, dir_fd=output_fd)
        except FileExistsError:
            continue
        try:
            _atomic_replace(output_fd, temporary_name, CURRENT_SELECTOR)
        finally:
            _unlink_entry(output_fd, temporary_name)
        _fsync_directory(output_fd)
        return
    raise FileExistsError("could not create a unique corpus current selector")


def _publish_generation(
    output: Path,
    artifacts: tuple[tuple[str, bytes], ...],
) -> str:
    artifact_data = dict(artifacts)
    generation_id = _generation_id(
        artifact_data["corpus.jsonl"],
        artifact_data["manifest.json"],
    )
    output.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    output_fd = os.open(output, directory_flags)
    lock_fd = -1
    generations_fd = -1
    try:
        _preflight_management_paths(output_fd)
        lock_fd = _acquire_output_lock(output_fd, output)
        generations_fd = _open_generations_directory(
            output_fd,
            create=True,
            validate_all=True,
        )
        current_generation = _current_generation_id(output_fd, generations_fd)
        _cleanup_output_staging(output_fd, generations_fd)
        _ensure_generation(generations_fd, generation_id, artifacts)
        if current_generation != generation_id:
            _switch_current_generation(output_fd, generation_id)
        return generation_id
    finally:
        if generations_fd >= 0:
            os.close(generations_fd)
        try:
            if lock_fd >= 0:
                _release_output_lock(lock_fd)
        finally:
            os.close(output_fd)


def read_symbolic_corpus(output: Path) -> SymbolicCorpusSnapshot:
    """Read one pinned immutable corpus generation."""

    _fcntl_module()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        output_fd = os.open(output, directory_flags)
    except OSError as error:
        raise ValueError(f"symbolic corpus output directory is unavailable: {output}") from error
    generations_fd = -1
    generation_fd = -1
    try:
        generations_fd = _open_generations_directory(
            output_fd,
            create=False,
            validate_all=False,
        )
        generation_id = _current_generation_id(
            output_fd,
            generations_fd,
            validate_generation=False,
        )
        if generation_id is None:
            raise ValueError("symbolic corpus output does not have a current generation")
        generation_fd = _open_directory_entry(generations_fd, generation_id)
        corpus_bytes = _read_regular_single_link_entry(generation_fd, "corpus.jsonl")
        manifest_bytes = _read_regular_single_link_entry(generation_fd, "manifest.json")
        if (
            corpus_bytes is None
            or manifest_bytes is None
            or not _is_generated_pair(manifest_bytes, corpus_bytes)
        ):
            raise ValueError(f"symbolic corpus generation is not coherent: {generation_id}")
        manifest = cast(CorpusManifest, json.loads(manifest_bytes))
        return SymbolicCorpusSnapshot(generation_id, corpus_bytes, manifest)
    finally:
        if generation_fd >= 0:
            os.close(generation_fd)
        if generations_fd >= 0:
            os.close(generations_fd)
        os.close(output_fd)


def export_symbolic_corpus(
    source: Path,
    output: Path,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> CorpusManifest:
    _fcntl_module()
    root, paths = _input_paths(source, output)
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
    _publish_generation(
        output,
        (
            ("corpus.jsonl", corpus_bytes),
            ("manifest.json", manifest_bytes),
        ),
    )
    return manifest

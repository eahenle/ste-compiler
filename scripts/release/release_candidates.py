"""Build and verify deterministic dataset and benchmark release-candidate archives."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, Literal

import yaml

from ste_compiler.evaluation import generate_evidence_report
from ste_compiler.evaluation.evidence import REPORT_MANIFEST_SCHEMA_VERSION
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.training import (
    build_demonstration_corpus,
    verify_demonstration_corpus,
)
from ste_compiler.training.release import EXPECTED_RELEASE_FILES, RELEASE_SCHEMA_VERSION

if TYPE_CHECKING:
    from scripts.release.release_contract import ReleaseIdentity

CANDIDATE_SCHEMA: Final = "ste-release-candidate-v1"
IDENTITY_SCHEMA: Final = "ste-release-build-identity-v1"
DATASET_ID: Final = "demonstration-corpus-2"
REPORT_ID: Final = "ste-compiler-pipeline-fixture-1"
MAX_ARCHIVE_BYTES: Final = 64 * 1024 * 1024
MAX_MEMBER_BYTES: Final = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS: Final = 64
MAX_USTAR_NUMBER: Final = 0o77777777777
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
VERSION_PATTERN: Final = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")
COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")

ArtifactKind = Literal["dataset", "report"]
Payload = dict[str, bytes]

BENCHMARK_INPUTS: Final = (
    "benchmark-spec.json",
    "failure-taxonomy.json",
    "prediction-manifest.json",
    "predictions.jsonl",
)
BENCHMARK_REPORT_FILES: Final = (
    "metrics.json",
    "report-manifest.json",
    "report.md",
)


class ReleaseCandidateError(RuntimeError):
    """Raised when a release candidate is unsafe, inconsistent, or noncanonical."""


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity_dict(identity: ReleaseIdentity) -> dict[str, object]:
    try:
        payload = identity.as_dict()
    except AttributeError as error:
        raise ReleaseCandidateError(
            "release identity must expose canonical identity fields"
        ) from error
    if not isinstance(payload, dict):
        raise ReleaseCandidateError("release identity fields must be a dictionary")
    if set(payload) != {
        "commit",
        "mode",
        "schema_version",
        "source_date_epoch",
        "tag",
        "version",
    }:
        raise ReleaseCandidateError("release identity has an unexpected field inventory")
    version = payload["version"]
    commit = payload["commit"]
    mode = payload["mode"]
    epoch = payload["source_date_epoch"]
    tag = payload["tag"]
    if payload["schema_version"] != IDENTITY_SCHEMA:
        raise ReleaseCandidateError("release identity schema is unsupported")
    if not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None:
        raise ReleaseCandidateError("release identity version is invalid")
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise ReleaseCandidateError("release identity commit is invalid")
    if not isinstance(mode, str) or mode not in {"dry-run", "tag"}:
        raise ReleaseCandidateError("release identity mode is invalid")
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
        or epoch > MAX_USTAR_NUMBER
    ):
        raise ReleaseCandidateError("release identity source date epoch is invalid for USTAR")
    if mode == "dry-run":
        if tag is not None:
            raise ReleaseCandidateError("dry-run release identity must not contain a tag")
    elif not isinstance(tag, str) or tag != f"v{version}":
        raise ReleaseCandidateError("tag release identity must match its version")
    try:
        attributes = {
            field: getattr(identity, field)
            for field in (
                "commit",
                "mode",
                "schema_version",
                "source_date_epoch",
                "tag",
                "version",
            )
        }
    except AttributeError as error:
        raise ReleaseCandidateError(
            "release identity must expose canonical identity attributes"
        ) from error
    if attributes != payload:
        raise ReleaseCandidateError("release identity attributes and serialized fields differ")
    return payload


def _run_git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        )
        raise ReleaseCandidateError(
            f"cannot inspect release source Git identity: {detail}"
        ) from error
    return completed.stdout.strip()


def _run_git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise ReleaseCandidateError(
            f"cannot inspect release source Git content: {detail}"
        ) from error
    return completed.stdout


def _verified_source_path(
    source_root: Path,
    relative: str,
    *,
    directory: bool,
    label: str,
) -> Path:
    normalized = _validate_relative_path(relative, label=f"{label} path")
    current = source_root
    for component in normalized.parts:
        current = current / component
        if current.is_symlink():
            raise ReleaseCandidateError(f"{label} must not contain a symbolic-link component")
    try:
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise ReleaseCandidateError(f"{label} does not exist: {current}") from error
    if not resolved.is_relative_to(source_root):
        raise ReleaseCandidateError(f"{label} escapes the release source root")
    try:
        metadata = os.stat(current, follow_symlinks=False)
    except OSError as error:
        raise ReleaseCandidateError(f"cannot inspect {label}: {current}") from error
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseCandidateError(f"{label} must be a directory: {resolved}")
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseCandidateError(f"{label} must be a regular file: {resolved}")
        if metadata.st_nlink != 1:
            raise ReleaseCandidateError(f"{label} must not be hardlinked")
    return resolved


def _read_tracked_source_file(
    source_root: Path,
    commit: str,
    relative: str,
    *,
    label: str,
) -> bytes:
    normalized = _validate_relative_path(relative, label=f"{label} path").as_posix()
    path = _verified_source_path(
        source_root,
        normalized,
        directory=False,
        label=label,
    )
    entry = _run_git_bytes(
        source_root,
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        f":(literal){normalized}",
    )
    metadata, separator, listed_path = entry.partition(b"\t")
    fields = metadata.split()
    if (
        separator != b"\t"
        or not listed_path.endswith(b"\0")
        or listed_path[:-1] != normalized.encode("utf-8")
        or len(fields) != 3
        or fields[0] != b"100644"
        or fields[1] != b"blob"
    ):
        raise ReleaseCandidateError(f"{label} must be a tracked regular Git blob: {normalized}")
    committed = _run_git_bytes(source_root, "cat-file", "blob", fields[2].decode("ascii"))
    checked = _read_regular(path, label=label, require_single_link=True)
    if checked != committed:
        raise ReleaseCandidateError(f"{label} does not match its Git blob: {normalized}")
    return checked


def _bind_source_root(source_root: Path, identity: ReleaseIdentity) -> Path:
    identity_payload = _identity_dict(identity)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReleaseCandidateError(f"source root must be a real directory: {source_root}")
    try:
        root = source_root.resolve(strict=True)
    except OSError as error:
        raise ReleaseCandidateError(f"cannot resolve source root: {source_root}") from error
    if source_root.absolute() != root:
        raise ReleaseCandidateError("source root path must not contain symbolic-link ancestors")
    top_level = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ReleaseCandidateError("release source root must be the Git worktree top level")
    if _run_git(root, "rev-parse", "HEAD") != identity_payload["commit"]:
        raise ReleaseCandidateError("release source HEAD does not match the release identity")
    if _run_git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ReleaseCandidateError("release source worktree must be clean")
    try:
        commit_epoch = int(_run_git(root, "show", "-s", "--format=%ct", "HEAD"))
    except ValueError as error:
        raise ReleaseCandidateError("release source commit timestamp is invalid") from error
    if commit_epoch != identity_payload["source_date_epoch"]:
        raise ReleaseCandidateError(
            "release source commit timestamp does not match source_date_epoch"
        )
    project_bytes = _read_tracked_source_file(
        root,
        str(identity_payload["commit"]),
        "pyproject.toml",
        label="project metadata",
    )
    citation_bytes = _read_tracked_source_file(
        root,
        str(identity_payload["commit"]),
        "CITATION.cff",
        label="citation metadata",
    )
    try:
        project_version = tomllib.loads(project_bytes.decode("utf-8"))["project"]["version"]
        citation = yaml.safe_load(citation_bytes.decode("utf-8"))
        citation_version = citation["version"] if isinstance(citation, dict) else None
    except (
        KeyError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
    ) as error:
        raise ReleaseCandidateError(
            f"cannot read release source version metadata: {error}"
        ) from error
    if (
        project_version != identity_payload["version"]
        or citation_version != identity_payload["version"]
    ):
        raise ReleaseCandidateError("release source version does not match the release identity")
    return root


def _archive_stem(version: str, kind: ArtifactKind) -> str:
    artifact_id = DATASET_ID if kind == "dataset" else REPORT_ID
    return f"ste-compiler-{version}-{kind}-{artifact_id}"


def _archive_name(version: str, kind: ArtifactKind) -> str:
    return f"{_archive_stem(version, kind)}.tar"


def _validate_relative_path(path: str, *, label: str) -> PurePosixPath:
    if not path:
        raise ReleaseCandidateError(f"{label} must be nonempty")
    normalized = PurePosixPath(path)
    if (
        normalized.is_absolute()
        or normalized.as_posix() != path
        or "\\" in path
        or any(part in {"", ".", ".."} for part in normalized.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise ReleaseCandidateError(f"{label} must be a canonical relative POSIX path")
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ReleaseCandidateError(f"{label} must be valid UTF-8") from error
    return normalized


def _read_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
    require_single_link: bool = False,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReleaseCandidateError(f"cannot read {label}: {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or (require_single_link and opened.st_nlink != 1)
        ):
            raise ReleaseCandidateError(f"{label} must be a regular file: {path}")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ReleaseCandidateError(f"{label} exceeds the size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(None if max_bytes is None else max_bytes + 1)
        after = os.fstat(descriptor)
        if (
            len(data) != opened.st_size
            or (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
            )
            != identity
        ):
            raise ReleaseCandidateError(f"{label} changed while it was read: {path}")
        if max_bytes is not None and len(data) > max_bytes:
            raise ReleaseCandidateError(f"{label} exceeds the size limit")
        return data
    except OSError as error:
        raise ReleaseCandidateError(f"cannot read {label}: {path}: {error}") from error
    finally:
        os.close(descriptor)


def _read_tracked_directory(
    source_root: Path,
    commit: str,
    relative_root: str,
    *,
    expected_files: frozenset[str],
    expected_directories: frozenset[str] = frozenset(),
    label: str,
) -> Payload:
    root = _verified_source_path(
        source_root,
        relative_root,
        directory=True,
        label=label,
    )
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ReleaseCandidateError(f"cannot inspect {label}: {root}: {error}") from error
    names = {entry.name for entry in entries}
    expected = expected_files | expected_directories
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise ReleaseCandidateError(f"{label} has an invalid inventory: {'; '.join(detail)}")
    for entry in entries:
        if entry.is_symlink():
            raise ReleaseCandidateError(f"{label} must not contain symbolic links")
        if entry.name in expected_files and not entry.is_file():
            raise ReleaseCandidateError(f"{label} must contain only expected regular files")
        if entry.name in expected_directories and not entry.is_dir():
            raise ReleaseCandidateError(f"{label} must contain expected real directories")
    return {
        name: _read_tracked_source_file(
            source_root,
            commit,
            f"{relative_root}/{name}",
            label=f"{label} payload",
        )
        for name in sorted(expected_files, key=lambda value: value.encode("utf-8"))
    }


def _read_flat_directory(
    root: Path,
    *,
    expected: frozenset[str],
    label: str,
) -> Payload:
    if root.is_symlink() or not root.is_dir():
        raise ReleaseCandidateError(f"{label} must be a real directory: {root}")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ReleaseCandidateError(f"cannot inspect {label}: {root}: {error}") from error
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ReleaseCandidateError(f"{label} must contain only regular files")
    names = {path.name for path in entries}
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise ReleaseCandidateError(f"{label} has an invalid inventory: {'; '.join(detail)}")
    return {
        name: _read_regular(root / name, label=f"{label} payload")
        for name in sorted(expected, key=lambda value: value.encode("utf-8"))
    }


def _rebuild_dataset(
    source_root: Path,
    source_commit: str,
    temporary: Path,
) -> Payload:
    construction_files = _read_tracked_directory(
        source_root,
        source_commit,
        "data/demonstration_corpus/v2",
        expected_files=frozenset({"source-construction.json", "terminology.yaml"}),
        label="Corpus V2 construction directory",
    )
    checked_files = _read_tracked_directory(
        source_root,
        source_commit,
        "datasets/demonstration-corpus-2",
        expected_files=frozenset(EXPECTED_RELEASE_FILES),
        label="checked Corpus V2 release",
    )
    vocabulary_bytes = _read_tracked_source_file(
        source_root,
        source_commit,
        "data/demo_vocabulary.yaml",
        label="Corpus V2 vocabulary",
    )
    inputs = temporary / "tracked-dataset-inputs"
    inputs.mkdir()
    construction_path = inputs / "source-construction.json"
    construction_path.write_bytes(construction_files["source-construction.json"])
    vocabulary_path = inputs / "demo-vocabulary.yaml"
    vocabulary_path.write_bytes(vocabulary_bytes)
    terminology_path = inputs / "terminology.yaml"
    terminology_path.write_bytes(construction_files["terminology.yaml"])
    checked_root = temporary / "checked-dataset"
    _materialize_payload(checked_root, checked_files)

    try:
        vocabulary = Vocabulary.load(vocabulary_path)
        terminology = TerminologyRegistry.load(terminology_path)
        verify_demonstration_corpus(checked_root)
        rebuilt = temporary / "rebuilt-dataset"
        build_demonstration_corpus(
            construction_path,
            rebuilt,
            vocabulary,
            terminology,
        )
        verify_demonstration_corpus(rebuilt)
    except (OSError, ValueError) as error:
        raise ReleaseCandidateError(f"cannot reconstruct Corpus V2: {error}") from error

    rebuilt_files = _read_flat_directory(
        rebuilt,
        expected=frozenset(EXPECTED_RELEASE_FILES),
        label="rebuilt Corpus V2 release",
    )
    mismatches = [
        name
        for name in sorted(EXPECTED_RELEASE_FILES)
        if checked_files[name] != rebuilt_files[name]
    ]
    if mismatches:
        raise ReleaseCandidateError(
            "rebuilt Corpus V2 differs from checked release: " + ", ".join(mismatches)
        )
    return rebuilt_files


def _regenerate_report(
    source_root: Path,
    source_commit: str,
    dataset_root: Path,
    temporary: Path,
) -> Payload:
    input_files = _read_tracked_directory(
        source_root,
        source_commit,
        "data/benchmark/v1",
        expected_files=frozenset(BENCHMARK_INPUTS),
        expected_directories=frozenset({"expected-report"}),
        label="benchmark input directory",
    )
    checked_report = _read_tracked_directory(
        source_root,
        source_commit,
        "data/benchmark/v1/expected-report",
        expected_files=frozenset(BENCHMARK_REPORT_FILES),
        label="checked benchmark report directory",
    )
    benchmark_root = temporary / "tracked-benchmark-inputs"
    _materialize_payload(benchmark_root, input_files)
    generated = temporary / "generated-report"
    try:
        generate_evidence_report(
            specification_path=benchmark_root / "benchmark-spec.json",
            taxonomy_path=benchmark_root / "failure-taxonomy.json",
            prediction_manifest_path=benchmark_root / "prediction-manifest.json",
            predictions_path=benchmark_root / "predictions.jsonl",
            dataset_release=dataset_root,
            output=generated,
        )
    except (OSError, ValueError) as error:
        raise ReleaseCandidateError(f"cannot regenerate benchmark evidence: {error}") from error
    generated_report = _read_flat_directory(
        generated,
        expected=frozenset(BENCHMARK_REPORT_FILES),
        label="regenerated benchmark report",
    )
    mismatches = [
        name for name in BENCHMARK_REPORT_FILES if generated_report[name] != checked_report[name]
    ]
    if mismatches:
        raise ReleaseCandidateError(
            "regenerated benchmark evidence differs from checked report: " + ", ".join(mismatches)
        )
    return {**input_files, **generated_report}


def _payload_identities(payload: Mapping[str, bytes]) -> list[dict[str, object]]:
    paths = sorted(payload, key=lambda value: value.encode("utf-8"))
    if len(paths) != len(set(paths)):
        raise ReleaseCandidateError("candidate payload paths must be unique")
    identities: list[dict[str, object]] = []
    for path in paths:
        _validate_relative_path(path, label="candidate payload path")
        data = payload[path]
        identities.append(
            {
                "bytes": len(data),
                "path": path,
                "sha256": _sha256(data),
            }
        )
    return identities


def _content_identity(kind: ArtifactKind, payload: Mapping[str, bytes]) -> dict[str, object]:
    manifest_path = "manifest.json" if kind == "dataset" else "report-manifest.json"
    expected_schema = (
        RELEASE_SCHEMA_VERSION if kind == "dataset" else REPORT_MANIFEST_SCHEMA_VERSION
    )
    try:
        manifest_bytes = payload[manifest_path]
    except KeyError as error:
        raise ReleaseCandidateError(
            f"{kind} payload is missing its authoritative manifest: {manifest_path}"
        ) from error
    parsed = _json_object(manifest_bytes, label=f"{kind} content manifest")
    if parsed.get("schema_version") != expected_schema:
        raise ReleaseCandidateError(f"{kind} content manifest schema is invalid")
    return {
        "manifest_path": manifest_path,
        "manifest_schema_version": expected_schema,
        "manifest_sha256": _sha256(manifest_bytes),
    }


def _candidate_manifest(
    *,
    identity: ReleaseIdentity,
    kind: ArtifactKind,
    payload: Mapping[str, bytes],
    dependencies: list[dict[str, object]],
) -> dict[str, object]:
    identity_payload = _identity_dict(identity)
    artifact_id = DATASET_ID if kind == "dataset" else REPORT_ID
    return {
        "archive": _archive_name(identity.version, kind),
        "artifact_id": artifact_id,
        "artifact_kind": kind,
        "content_identity": _content_identity(kind, payload),
        "dependencies": dependencies,
        "payload": _payload_identities(payload),
        "release_identity": identity_payload,
        "schema_version": CANDIDATE_SCHEMA,
        "top_level_directory": _archive_stem(identity.version, kind),
    }


def _tar_info(
    name: str,
    *,
    identity: ReleaseIdentity,
    directory: bool,
    size: int = 0,
) -> tarfile.TarInfo:
    encoded = name.encode("utf-8")
    if len(encoded) > 100:
        raise ReleaseCandidateError(f"USTAR path exceeds the portable name field: {name}")
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.mtime = identity.source_date_epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.linkname = ""
    info.devmajor = 0
    info.devminor = 0
    return info


def _archive_bytes(
    *,
    identity: ReleaseIdentity,
    kind: ArtifactKind,
    payload: Mapping[str, bytes],
    manifest_bytes: bytes,
) -> bytes:
    top = _archive_stem(identity.version, kind)
    members: dict[str, bytes | None] = {top: None}
    members.update({f"{top}/{path}": data for path, data in payload.items()})
    members[f"{top}/release-candidate.json"] = manifest_bytes
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            data = members[name]
            if data is None:
                archive.addfile(_tar_info(name, identity=identity, directory=True))
            else:
                archive.addfile(
                    _tar_info(
                        name,
                        identity=identity,
                        directory=False,
                        size=len(data),
                    ),
                    io.BytesIO(data),
                )
    encoded = buffer.getvalue()
    if len(encoded) > MAX_ARCHIVE_BYTES:
        raise ReleaseCandidateError("release candidate archive exceeds the size limit")
    return encoded


def _write_candidate(
    *,
    output: Path,
    identity: ReleaseIdentity,
    kind: ArtifactKind,
    payload: Payload,
    dependencies: list[dict[str, object]],
) -> tuple[Path, bytes]:
    manifest = _candidate_manifest(
        identity=identity,
        kind=kind,
        payload=payload,
        dependencies=dependencies,
    )
    manifest_bytes = _canonical_json(manifest)
    archive_bytes = _archive_bytes(
        identity=identity,
        kind=kind,
        payload=payload,
        manifest_bytes=manifest_bytes,
    )
    destination = output / _archive_name(identity.version, kind)
    destination.write_bytes(archive_bytes)
    return destination, manifest_bytes


def _rename_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as error:
            raise ReleaseCandidateError(
                "atomic no-replace candidate publication is unavailable"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise ReleaseCandidateError(
                "atomic no-replace candidate publication is unavailable"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        raise ReleaseCandidateError(
            f"atomic no-replace candidate publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ReleaseCandidateError(
            f"candidate output was created concurrently: {destination_name}"
        )
    raise ReleaseCandidateError(
        f"cannot publish candidate directory atomically: {os.strerror(error_number)}"
    )


def _create_private_stage(parent_descriptor: int) -> tuple[str, int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for _ in range(128):
        name = f".ste-release-candidates-stage-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            return name, os.open(name, flags, dir_fd=parent_descriptor)
        except OSError:
            os.rmdir(name, dir_fd=parent_descriptor)
            raise
    raise ReleaseCandidateError("cannot allocate a private candidate stage")


def _write_stage_file(stage_descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o644, dir_fd=stage_descriptor)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("candidate stage write made no progress")
            view = view[written:]
    finally:
        os.close(descriptor)


def _remove_private_stage(
    parent_descriptor: int,
    stage_name: str,
    stage_descriptor: int,
    archive_names: tuple[str, ...],
) -> None:
    for name in archive_names:
        try:
            os.unlink(name, dir_fd=stage_descriptor)
        except FileNotFoundError:
            pass
    os.close(stage_descriptor)
    try:
        os.rmdir(stage_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        pass


def _remove_published_candidates(
    parent_descriptor: int,
    output_name: str,
    archive_names: tuple[str, ...],
) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(output_name, flags, dir_fd=parent_descriptor)
    try:
        for name in archive_names:
            os.unlink(name, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(output_name, dir_fd=parent_descriptor)


def build_candidate_directory(
    source_root: Path,
    identity: ReleaseIdentity,
    output: Path,
) -> tuple[Path, Path]:
    """Reconstruct, byte-verify, and package the two core release candidates."""

    _identity_dict(identity)
    if os.path.lexists(output):
        raise ReleaseCandidateError(f"candidate output path must not exist: {output}")
    if output.name in {"", ".", ".."}:
        raise ReleaseCandidateError("candidate output must name a directory")
    source_root = _bind_source_root(source_root, identity)
    if output.absolute().is_relative_to(source_root):
        raise ReleaseCandidateError("candidate output must be outside the release source root")
    output_parent = output.parent
    if output_parent.is_symlink() or not output_parent.is_dir():
        raise ReleaseCandidateError(
            f"candidate output parent must be an existing real directory: {output_parent}"
        )
    if output.absolute() != output.resolve():
        raise ReleaseCandidateError(
            "candidate output path must not contain symbolic-link ancestors"
        )
    if output_parent.resolve().is_relative_to(source_root):
        raise ReleaseCandidateError("candidate output must be outside the release source root")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(output_parent, flags)
    except OSError as error:
        raise ReleaseCandidateError(
            f"candidate output parent must be an existing real directory: {output_parent}"
        ) from error
    try:
        parent_identity = os.fstat(parent_descriptor)
        with tempfile.TemporaryDirectory(
            prefix="ste-release-candidates-build-",
        ) as build_raw:
            build = Path(build_raw)
            candidate_output = build / "candidate-stage"
            candidate_output.mkdir()
            work = build / "work"
            work.mkdir()
            source_commit = str(_identity_dict(identity)["commit"])
            dataset_payload = _rebuild_dataset(source_root, source_commit, work)
            dataset_materialized = work / "dataset-for-report"
            dataset_materialized.mkdir()
            for name, data in dataset_payload.items():
                (dataset_materialized / name).write_bytes(data)
            report_payload = _regenerate_report(
                source_root,
                source_commit,
                dataset_materialized,
                work,
            )
            dataset_archive, dataset_candidate_manifest = _write_candidate(
                output=candidate_output,
                identity=identity,
                kind="dataset",
                payload=dataset_payload,
                dependencies=[],
            )
            dataset_manifest = dataset_payload["manifest.json"]
            report_dependencies: list[dict[str, object]] = [
                {
                    "archive": dataset_archive.name,
                    "archive_sha256": _sha256(dataset_archive.read_bytes()),
                    "artifact_id": DATASET_ID,
                    "artifact_kind": "dataset",
                    "candidate_manifest_sha256": _sha256(dataset_candidate_manifest),
                    "corpus_manifest_sha256": _sha256(dataset_manifest),
                }
            ]
            report_archive, _ = _write_candidate(
                output=candidate_output,
                identity=identity,
                kind="report",
                payload=report_payload,
                dependencies=report_dependencies,
            )
            try:
                current_parent = os.stat(output_parent, follow_symlinks=False)
            except OSError as error:
                raise ReleaseCandidateError(
                    "candidate output parent changed during candidate construction"
                ) from error
            if (
                current_parent.st_dev,
                current_parent.st_ino,
                current_parent.st_mode,
            ) != (
                parent_identity.st_dev,
                parent_identity.st_ino,
                parent_identity.st_mode,
            ):
                raise ReleaseCandidateError(
                    "candidate output parent changed during candidate construction"
                )
            archive_names = (dataset_archive.name, report_archive.name)
            stage_name, stage_descriptor = _create_private_stage(parent_descriptor)
            published = False
            try:
                for archive in (dataset_archive, report_archive):
                    _write_stage_file(
                        stage_descriptor,
                        archive.name,
                        _read_regular(archive, label="built candidate archive"),
                    )
                _rename_no_replace(
                    parent_descriptor,
                    stage_name,
                    output.name,
                )
                published = True
                published_parent = os.stat(output_parent, follow_symlinks=False)
                if (
                    published_parent.st_dev,
                    published_parent.st_ino,
                    published_parent.st_mode,
                ) != (
                    parent_identity.st_dev,
                    parent_identity.st_ino,
                    parent_identity.st_mode,
                ):
                    raise ReleaseCandidateError(
                        "candidate output parent changed during candidate publication"
                    )
            except BaseException:
                if published:
                    os.close(stage_descriptor)
                    _remove_published_candidates(
                        parent_descriptor,
                        output.name,
                        archive_names,
                    )
                else:
                    _remove_private_stage(
                        parent_descriptor,
                        stage_name,
                        stage_descriptor,
                        archive_names,
                    )
                raise
            else:
                os.close(stage_descriptor)
            return output / dataset_archive.name, output / report_archive.name
    except ReleaseCandidateError:
        raise
    except (OSError, ValueError, tarfile.TarError) as error:
        raise ReleaseCandidateError(f"cannot build release candidate directory: {error}") from error
    finally:
        os.close(parent_descriptor)


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ReleaseCandidateError(f"candidate JSON contains duplicate field {key!r}")
        payload[key] = value
    return payload


def _json_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload: Any = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseCandidateError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ReleaseCandidateError(f"{label} must contain a JSON object")
    return payload


def _canonical_json_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    payload = _json_object(encoded, label=label)
    if encoded != _canonical_json(payload):
        raise ReleaseCandidateError(f"{label} is not canonical JSON")
    return payload


def _validated_payload_manifest(
    raw: Any,
    *,
    actual: Mapping[str, bytes],
) -> list[dict[str, object]]:
    if not isinstance(raw, list) or not raw:
        raise ReleaseCandidateError("candidate payload manifest must be a nonempty list")
    observed: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"bytes", "path", "sha256"}:
            raise ReleaseCandidateError("candidate payload identity has unexpected fields")
        path = item["path"]
        size = item["bytes"]
        digest = item["sha256"]
        if not isinstance(path, str):
            raise ReleaseCandidateError("candidate payload path must be a string")
        _validate_relative_path(path, label="candidate payload path")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReleaseCandidateError("candidate payload size must be a nonnegative integer")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseCandidateError("candidate payload SHA-256 must be lowercase hexadecimal")
        observed.append({"bytes": size, "path": path, "sha256": digest})
    paths = tuple(item["path"] for item in observed)
    if paths != tuple(sorted(set(paths), key=lambda value: str(value).encode("utf-8"))):
        raise ReleaseCandidateError("candidate payload manifest must be unique and UTF-8 sorted")
    if set(actual) != set(paths):
        raise ReleaseCandidateError("candidate archive payload does not match its manifest")
    for item in observed:
        path = str(item["path"])
        data = actual[path]
        if item["bytes"] != len(data) or item["sha256"] != _sha256(data):
            raise ReleaseCandidateError(f"candidate payload identity does not match: {path}")
    return observed


def _validate_dependencies(raw: Any, *, kind: ArtifactKind) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        raise ReleaseCandidateError("candidate dependencies must be a list")
    if kind == "dataset":
        if raw:
            raise ReleaseCandidateError("dataset candidate must not declare dependencies")
        return []
    if len(raw) != 1:
        raise ReleaseCandidateError("report candidate must declare one dataset dependency")
    dependency = raw[0]
    expected_keys = {
        "archive",
        "archive_sha256",
        "artifact_id",
        "artifact_kind",
        "candidate_manifest_sha256",
        "corpus_manifest_sha256",
    }
    if not isinstance(dependency, dict) or set(dependency) != expected_keys:
        raise ReleaseCandidateError("report dataset dependency has unexpected fields")
    if (
        dependency["artifact_id"] != DATASET_ID
        or dependency["artifact_kind"] != "dataset"
        or not isinstance(dependency["archive"], str)
        or not isinstance(dependency["archive_sha256"], str)
        or SHA256_PATTERN.fullmatch(dependency["archive_sha256"]) is None
        or not isinstance(dependency["candidate_manifest_sha256"], str)
        or SHA256_PATTERN.fullmatch(dependency["candidate_manifest_sha256"]) is None
        or not isinstance(dependency["corpus_manifest_sha256"], str)
        or SHA256_PATTERN.fullmatch(dependency["corpus_manifest_sha256"]) is None
    ):
        raise ReleaseCandidateError("report dataset dependency is invalid")
    return [dict(dependency)]


def _parse_candidate_manifest(
    encoded: bytes,
    *,
    identity: ReleaseIdentity,
    kind: ArtifactKind,
    actual_payload: Mapping[str, bytes],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    payload = _canonical_json_object(encoded, label="release-candidate.json")
    expected_keys = {
        "archive",
        "artifact_id",
        "artifact_kind",
        "content_identity",
        "dependencies",
        "payload",
        "release_identity",
        "schema_version",
        "top_level_directory",
    }
    if set(payload) != expected_keys:
        raise ReleaseCandidateError("candidate manifest has an unexpected field inventory")
    artifact_id = DATASET_ID if kind == "dataset" else REPORT_ID
    if (
        payload["schema_version"] != CANDIDATE_SCHEMA
        or payload["artifact_kind"] != kind
        or payload["artifact_id"] != artifact_id
        or payload["archive"] != _archive_name(identity.version, kind)
        or payload["top_level_directory"] != _archive_stem(identity.version, kind)
    ):
        raise ReleaseCandidateError("candidate manifest identity is inconsistent")
    if payload["release_identity"] != _identity_dict(identity):
        raise ReleaseCandidateError("candidate outer release identity does not match")
    _validated_payload_manifest(payload["payload"], actual=actual_payload)
    if payload["content_identity"] != _content_identity(kind, actual_payload):
        raise ReleaseCandidateError("candidate content manifest identity does not match")
    dependencies = _validate_dependencies(payload["dependencies"], kind=kind)
    return payload, dependencies


def _read_candidate_archive(
    path: Path,
    *,
    identity: ReleaseIdentity,
    kind: ArtifactKind,
) -> tuple[Payload, bytes, list[dict[str, object]], str]:
    encoded = _read_regular(
        path,
        label=f"{kind} candidate archive",
        max_bytes=MAX_ARCHIVE_BYTES,
        require_single_link=True,
    )
    if len(encoded) < tarfile.RECORDSIZE or len(encoded) % tarfile.RECORDSIZE != 0:
        raise ReleaseCandidateError(
            "candidate archive must use complete deterministic 10240-byte tar records"
        )
    try:
        with tarfile.open(fileobj=io.BytesIO(encoded), mode="r:") as archive:
            if archive.pax_headers:
                raise ReleaseCandidateError("candidate archive must not contain global PAX headers")
            members: list[tarfile.TarInfo] = []
            while True:
                member = archive.next()
                if member is None:
                    break
                if len(members) == MAX_ARCHIVE_MEMBERS:
                    raise ReleaseCandidateError("candidate archive contains too many members")
                members.append(member)
            if not members:
                raise ReleaseCandidateError("candidate archive cannot be empty")
            names = tuple(member.name for member in members)
            if len(names) != len(set(names)):
                raise ReleaseCandidateError("candidate archive contains duplicate paths")
            if names != tuple(sorted(names, key=lambda value: value.encode("utf-8"))):
                raise ReleaseCandidateError("candidate archive paths must be UTF-8 sorted")
            top = _archive_stem(identity.version, kind)
            payload: Payload = {}
            manifest_bytes: bytes | None = None
            directories: set[str] = set()
            for member in members:
                _validate_relative_path(member.name, label="candidate archive member path")
                if member.pax_headers:
                    raise ReleaseCandidateError("candidate archive must not contain PAX headers")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != identity.source_date_epoch
                    or member.linkname != ""
                    or member.devmajor != 0
                    or member.devminor != 0
                ):
                    raise ReleaseCandidateError("candidate archive contains noncanonical metadata")
                if member.type == tarfile.DIRTYPE:
                    if member.mode != 0o755 or member.size != 0:
                        raise ReleaseCandidateError(
                            "candidate archive directory metadata is noncanonical"
                        )
                    directories.add(member.name)
                    continue
                if member.type != tarfile.REGTYPE:
                    raise ReleaseCandidateError(
                        "candidate archive may contain only directories and regular files"
                    )
                if member.mode != 0o644 or member.size > MAX_MEMBER_BYTES:
                    raise ReleaseCandidateError("candidate archive file metadata is noncanonical")
                prefix = f"{top}/"
                if not member.name.startswith(prefix):
                    raise ReleaseCandidateError(
                        "candidate archive member is outside its top-level directory"
                    )
                relative = member.name.removeprefix(prefix)
                _validate_relative_path(relative, label="candidate payload path")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReleaseCandidateError("candidate regular file cannot be read")
                data = handle.read(MAX_MEMBER_BYTES + 1)
                if len(data) != member.size:
                    raise ReleaseCandidateError("candidate archive member size is inconsistent")
                if relative == "release-candidate.json":
                    manifest_bytes = data
                else:
                    payload[relative] = data
    except (OSError, tarfile.TarError) as error:
        raise ReleaseCandidateError(f"cannot read {kind} candidate archive: {error}") from error
    if directories != {_archive_stem(identity.version, kind)}:
        raise ReleaseCandidateError(
            "candidate archive must contain exactly one explicit top-level directory"
        )
    if manifest_bytes is None:
        raise ReleaseCandidateError("candidate archive is missing release-candidate.json")
    _, dependencies = _parse_candidate_manifest(
        manifest_bytes,
        identity=identity,
        kind=kind,
        actual_payload=payload,
    )
    canonical = _archive_bytes(
        identity=identity,
        kind=kind,
        payload=payload,
        manifest_bytes=manifest_bytes,
    )
    if encoded != canonical:
        raise ReleaseCandidateError("candidate archive is not canonical uncompressed USTAR")
    return payload, manifest_bytes, dependencies, _sha256(encoded)


def _materialize_payload(root: Path, payload: Mapping[str, bytes]) -> None:
    root.mkdir()
    for relative, data in payload.items():
        path = _validate_relative_path(relative, label="candidate payload path")
        destination = root.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _verify_dataset_payload(payload: Payload, temporary: Path) -> Path:
    if set(payload) != set(EXPECTED_RELEASE_FILES):
        raise ReleaseCandidateError("dataset candidate has an unexpected corpus payload inventory")
    materialized = temporary / "dataset"
    _materialize_payload(materialized, payload)
    try:
        verify_demonstration_corpus(materialized)
    except (OSError, ValueError) as error:
        raise ReleaseCandidateError(f"dataset candidate does not reconstruct: {error}") from error
    return materialized


def _verify_report_payload(
    payload: Payload,
    *,
    dataset_root: Path,
    temporary: Path,
) -> None:
    expected = set(BENCHMARK_INPUTS) | set(BENCHMARK_REPORT_FILES)
    if set(payload) != expected:
        raise ReleaseCandidateError(
            "report candidate has an unexpected benchmark payload inventory"
        )
    materialized = temporary / "report"
    _materialize_payload(materialized, payload)
    generated = temporary / "verified-report"
    try:
        generate_evidence_report(
            specification_path=materialized / "benchmark-spec.json",
            taxonomy_path=materialized / "failure-taxonomy.json",
            prediction_manifest_path=materialized / "prediction-manifest.json",
            predictions_path=materialized / "predictions.jsonl",
            dataset_release=dataset_root,
            output=generated,
        )
    except (OSError, ValueError) as error:
        raise ReleaseCandidateError(f"report candidate does not regenerate: {error}") from error
    for name in BENCHMARK_REPORT_FILES:
        rebuilt = _read_regular(generated / name, label="regenerated candidate report")
        if rebuilt != payload[name]:
            raise ReleaseCandidateError(
                f"report candidate output does not regenerate byte-for-byte: {name}"
            )


def verify_candidate_directory(
    path: Path,
    identity: ReleaseIdentity,
) -> tuple[Path, Path]:
    """Verify both untrusted core candidate archives and their dependency cross-link."""

    _identity_dict(identity)
    if path.is_symlink() or not path.is_dir():
        raise ReleaseCandidateError(f"candidate directory must be a real directory: {path}")
    if path.absolute() != path.resolve():
        raise ReleaseCandidateError(
            "candidate directory path must not contain symbolic-link ancestors"
        )
    dataset_name = _archive_name(identity.version, "dataset")
    report_name = _archive_name(identity.version, "report")
    try:
        entries = tuple(path.iterdir())
        invalid_entry = any(
            entry.is_symlink()
            or not entry.is_file()
            or os.stat(entry, follow_symlinks=False).st_nlink != 1
            for entry in entries
        )
    except OSError as error:
        raise ReleaseCandidateError(f"cannot inspect candidate directory: {error}") from error
    if invalid_entry:
        raise ReleaseCandidateError("candidate directory must contain only regular archive files")
    if {entry.name for entry in entries} != {dataset_name, report_name}:
        raise ReleaseCandidateError("candidate directory has an unexpected archive inventory")
    dataset_path = path / dataset_name
    report_path = path / report_name
    dataset_payload, dataset_manifest, dataset_dependencies, dataset_digest = (
        _read_candidate_archive(
            dataset_path,
            identity=identity,
            kind="dataset",
        )
    )
    report_payload, _, report_dependencies, _ = _read_candidate_archive(
        report_path,
        identity=identity,
        kind="report",
    )
    if dataset_dependencies:
        raise ReleaseCandidateError("dataset candidate dependency inventory is invalid")
    with tempfile.TemporaryDirectory(prefix="ste-release-candidate-verify-") as temporary_raw:
        temporary = Path(temporary_raw)
        dataset_root = _verify_dataset_payload(dataset_payload, temporary)
        expected_dependency = {
            "archive": dataset_name,
            "archive_sha256": dataset_digest,
            "artifact_id": DATASET_ID,
            "artifact_kind": "dataset",
            "candidate_manifest_sha256": _sha256(dataset_manifest),
            "corpus_manifest_sha256": _sha256(dataset_payload["manifest.json"]),
        }
        if report_dependencies != [expected_dependency]:
            raise ReleaseCandidateError("report candidate dataset dependency does not match")
        _verify_report_payload(
            report_payload,
            dataset_root=dataset_root,
            temporary=temporary,
        )
    return dataset_path, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--identity", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--path", type=Path, required=True)
    verify.add_argument("--identity", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        from scripts.release.release_contract import read_identity

        identity = read_identity(args.identity)
        if args.command == "build":
            dataset, report = build_candidate_directory(
                args.source_root,
                identity,
                args.output,
            )
        else:
            dataset, report = verify_candidate_directory(args.path, identity)
        print(
            json.dumps(
                {
                    "dataset": str(dataset),
                    "report": str(report),
                    "status": "built" if args.command == "build" else "verified",
                },
                sort_keys=True,
            )
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    main()

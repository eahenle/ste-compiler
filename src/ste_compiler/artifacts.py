"""Content-bound manifests and hardened local artifact-bundle verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ARTIFACT_MANIFEST_NAME: Final = "artifact-manifest.json"
ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = "ste-artifact-bundle-v1"
MAX_ARTIFACT_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_ARTIFACT_FILES: Final = 1024
MAX_ARTIFACT_FILE_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES: Final = 32 * 1024 * 1024 * 1024
MAX_ARTIFACT_PATH_BYTES: Final = 1024
MAX_ARTIFACT_COMPONENT_BYTES: Final = 255
MAX_ARTIFACT_PATH_DEPTH: Final = 8
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_PORTABLE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+", re.ASCII)
_STABLE_STAT_FIELDS: Final = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

ArtifactArchitecture = Literal["encoder-decoder", "decoder-only-lora"]
ArtifactType = Literal["encoder-decoder-checkpoint", "decoder-only-lora-run"]
ArtifactEntrypoint = Literal[".", "adapter"]
ArtifactValidationProfile = Literal[
    "encoder-checkpoint-load-v1",
    "decoder-adapter-structure-v1",
]


class ArtifactVerificationError(ValueError):
    """A local artifact bundle could not establish its declared identity."""


class _StrictArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(path: str) -> str:
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("artifact paths must contain only portable ASCII characters") from error
    if (
        not path
        or path != path.strip()
        or path.startswith("/")
        or "\\" in path
        or ":" in path
        or "\0" in path
        or len(encoded) > MAX_ARTIFACT_PATH_BYTES
    ):
        raise ValueError("artifact path must be a safe relative POSIX path")
    components = path.split("/")
    if len(components) > MAX_ARTIFACT_PATH_DEPTH:
        raise ValueError("artifact path exceeds the maximum depth")
    if any(
        component in {"", ".", ".."}
        or len(component.encode("ascii")) > MAX_ARTIFACT_COMPONENT_BYTES
        or _PORTABLE_COMPONENT.fullmatch(component) is None
        for component in components
    ):
        raise ValueError("artifact path contains an unsafe component")
    return path


class ArtifactFileV1(_StrictArtifactModel):
    """Immutable content identity for one regular file relative to a bundle root."""

    path: str = Field(min_length=1, max_length=MAX_ARTIFACT_PATH_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0, le=MAX_ARTIFACT_FILE_BYTES)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, path: str) -> str:
        return _validate_relative_path(path)


class ArtifactBundleManifestV1(_StrictArtifactModel):
    """Canonical identity of every file in one local training-output bundle."""

    schema_version: Literal["ste-artifact-bundle-v1"]
    architecture: ArtifactArchitecture
    artifact_type: ArtifactType
    intended_use: Literal["mechanics-smoke"]
    entrypoint: ArtifactEntrypoint
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(gt=0, le=MAX_ARTIFACT_FILES)
    total_bytes: int = Field(ge=0, le=MAX_ARTIFACT_TOTAL_BYTES)
    files: tuple[ArtifactFileV1, ...] = Field(min_length=1, max_length=MAX_ARTIFACT_FILES)

    @model_validator(mode="after")
    def internally_consistent(self) -> ArtifactBundleManifestV1:
        expected_profile = {
            "encoder-decoder": ("encoder-decoder-checkpoint", "."),
            "decoder-only-lora": ("decoder-only-lora-run", "adapter"),
        }[self.architecture]
        if (self.artifact_type, self.entrypoint) != expected_profile:
            raise ValueError("artifact type and entrypoint do not match the architecture")
        paths = tuple(identity.path for identity in self.files)
        if paths != tuple(sorted(paths)):
            raise ValueError("artifact files must be sorted by path")
        if len(set(paths)) != len(paths):
            raise ValueError("artifact file paths must be unique")
        folded_paths = tuple(path.casefold() for path in paths)
        if len(set(folded_paths)) != len(folded_paths):
            raise ValueError("artifact file paths must be unique under case folding")
        path_set = set(paths)
        for path in paths:
            components = path.split("/")
            if any("/".join(components[:end]) in path_set for end in range(1, len(components))):
                raise ValueError("an artifact path cannot be both a file and a directory")
        if ARTIFACT_MANIFEST_NAME in path_set:
            raise ValueError("artifact manifest must not include itself in the file inventory")
        if self.file_count != len(self.files):
            raise ValueError("artifact file_count does not match the file inventory")
        if self.total_bytes != sum(identity.bytes for identity in self.files):
            raise ValueError("artifact total_bytes does not match the file inventory")
        run_manifest = next(
            (identity for identity in self.files if identity.path == "run-manifest.json"),
            None,
        )
        if run_manifest is None:
            raise ValueError("artifact bundle must include run-manifest.json")
        if run_manifest.sha256 != self.run_manifest_sha256:
            raise ValueError("run manifest digest does not match its file identity")
        if self.entrypoint != "." and not any(
            path == self.entrypoint or path.startswith(f"{self.entrypoint}/") for path in paths
        ):
            raise ValueError("artifact entrypoint does not exist in the file inventory")
        return self


class ArtifactPreflightResultV1(_StrictArtifactModel):
    """Machine-readable result of one complete offline artifact preflight."""

    schema_version: Literal["ste-artifact-preflight-v1"]
    status: Literal["verified"]
    architecture: ArtifactArchitecture
    artifact_type: ArtifactType
    intended_use: Literal["mechanics-smoke"]
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(gt=0, le=MAX_ARTIFACT_FILES)
    total_bytes: int = Field(ge=0, le=MAX_ARTIFACT_TOTAL_BYTES)
    validation_profile: ArtifactValidationProfile
    network_access: Literal["none"]

    @model_validator(mode="after")
    def internally_consistent(self) -> ArtifactPreflightResultV1:
        expected_profile = {
            "encoder-decoder": (
                "encoder-decoder-checkpoint",
                "encoder-checkpoint-load-v1",
            ),
            "decoder-only-lora": (
                "decoder-only-lora-run",
                "decoder-adapter-structure-v1",
            ),
        }[self.architecture]
        if (self.artifact_type, self.validation_profile) != expected_profile:
            raise ValueError("artifact type and validation profile do not match the architecture")
        return self


@dataclass(frozen=True)
class VerifiedArtifactBundle:
    """A private, content-verified materialization valid only inside its context manager."""

    path: Path
    manifest: ArtifactBundleManifestV1
    manifest_sha256: str


def canonical_artifact_manifest_json(manifest: ArtifactBundleManifestV1) -> bytes:
    """Return the one accepted byte representation of an artifact manifest."""

    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def artifact_manifest_sha256(manifest: ArtifactBundleManifestV1) -> str:
    """Return the SHA-256 identity of a canonical artifact manifest."""

    return hashlib.sha256(canonical_artifact_manifest_json(manifest)).hexdigest()


def build_artifact_manifest(
    *,
    architecture: ArtifactArchitecture,
    artifact_type: ArtifactType,
    entrypoint: ArtifactEntrypoint,
    files: Sequence[ArtifactFileV1],
) -> ArtifactBundleManifestV1:
    """Build one canonical manifest from a previously captured file inventory."""

    ordered = tuple(sorted(files, key=lambda identity: identity.path))
    run_manifest = next(
        (identity for identity in ordered if identity.path == "run-manifest.json"),
        None,
    )
    if run_manifest is None:
        raise ValueError("captured file inventory must include run-manifest.json")
    return ArtifactBundleManifestV1(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        architecture=architecture,
        artifact_type=artifact_type,
        intended_use="mechanics-smoke",
        entrypoint=entrypoint,
        run_manifest_sha256=run_manifest.sha256,
        file_count=len(ordered),
        total_bytes=sum(identity.bytes for identity in ordered),
        files=ordered,
    )


def parse_canonical_artifact_manifest(data: bytes) -> ArtifactBundleManifestV1:
    """Parse bounded manifest bytes and reject alternate JSON representations."""

    if len(data) > MAX_ARTIFACT_MANIFEST_BYTES:
        raise ArtifactVerificationError("artifact manifest exceeds its size limit")
    try:
        manifest = ArtifactBundleManifestV1.model_validate_json(data)
    except ValueError as error:
        raise ArtifactVerificationError(f"artifact manifest is invalid: {error}") from error
    if data != canonical_artifact_manifest_json(manifest):
        raise ArtifactVerificationError("artifact manifest is not canonical JSON")
    return manifest


def _require_hardened_posix() -> None:
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_CLOEXEC")
    ):
        raise ArtifactVerificationError(
            "hardened artifact verification requires POSIX directory descriptors and O_NOFOLLOW"
        )


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(metadata, field)) for field in _STABLE_STAT_FIELDS)


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)


def _read_regular_bytes(
    directory_fd: int,
    name: str,
    *,
    display_path: str,
    max_bytes: int,
) -> bytes:
    try:
        entry_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        file_fd = os.open(name, _file_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise ArtifactVerificationError(f"cannot safely open artifact: {display_path}") from error
    try:
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(entry_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or entry_before.st_nlink != 1
            or before.st_nlink != 1
            or _stat_identity(entry_before) != _stat_identity(before)
        ):
            raise ArtifactVerificationError(
                f"artifact must be a single-link regular file: {display_path}"
            )
        if before.st_size > max_bytes:
            raise ArtifactVerificationError(f"artifact exceeds its size limit: {display_path}")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(file_fd, min(_COPY_CHUNK_BYTES, max_bytes + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise ArtifactVerificationError(f"artifact exceeds its size limit: {display_path}")
        after = os.fstat(file_fd)
        entry_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise ArtifactVerificationError(f"cannot safely read artifact: {display_path}") from error
    finally:
        os.close(file_fd)
    if (
        byte_count != before.st_size
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(entry_after)
    ):
        raise ArtifactVerificationError(f"artifact changed while read: {display_path}")
    return b"".join(chunks)


def _copy_regular_file(
    source_fd: int,
    destination_fd: int,
    name: str,
    *,
    relative_path: str,
    expected: ArtifactFileV1,
) -> None:
    try:
        entry_before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        source_file_fd = os.open(name, _file_flags(), dir_fd=source_fd)
    except OSError as error:
        raise ArtifactVerificationError(f"cannot safely open artifact: {relative_path}") from error
    destination_file_fd = -1
    try:
        before = os.fstat(source_file_fd)
        if (
            not stat.S_ISREG(entry_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or entry_before.st_nlink != 1
            or before.st_nlink != 1
            or _stat_identity(entry_before) != _stat_identity(before)
        ):
            raise ArtifactVerificationError(
                f"artifact must be a single-link regular file: {relative_path}"
            )
        if before.st_size != expected.bytes:
            raise ArtifactVerificationError(
                f"artifact size does not match its manifest: {relative_path}"
            )
        destination_file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_fd,
        )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(
                source_file_fd,
                min(_COPY_CHUNK_BYTES, MAX_ARTIFACT_FILE_BYTES + 1 - byte_count),
            )
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > MAX_ARTIFACT_FILE_BYTES:
                raise ArtifactVerificationError(f"artifact exceeds its size limit: {relative_path}")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_file_fd, view)
                if written <= 0:
                    raise OSError(f"short write while materializing {relative_path}")
                view = view[written:]
        os.fsync(destination_file_fd)
        after = os.fstat(source_file_fd)
        entry_after = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
    except ArtifactVerificationError:
        raise
    except OSError as error:
        raise ArtifactVerificationError(f"cannot materialize artifact: {relative_path}") from error
    finally:
        os.close(source_file_fd)
        if destination_file_fd >= 0:
            os.close(destination_file_fd)
    if (
        byte_count != expected.bytes
        or digest.hexdigest() != expected.sha256
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(entry_after)
    ):
        raise ArtifactVerificationError(
            f"artifact identity does not match its manifest: {relative_path}"
        )


_TreeKind = Literal["file", "directory"]


def _expected_tree(
    files: Sequence[ArtifactFileV1],
) -> tuple[dict[str, dict[str, _TreeKind]], dict[str, ArtifactFileV1]]:
    identities = {identity.path: identity for identity in files}
    tree: dict[str, dict[str, _TreeKind]] = {"": {}}
    for relative_path in identities:
        components = relative_path.split("/")
        parent = ""
        for component in components[:-1]:
            children = tree.setdefault(parent, {})
            existing = children.get(component)
            if existing == "file":
                raise ArtifactVerificationError(
                    "artifact manifest uses one path as both a file and directory"
                )
            children[component] = "directory"
            parent = f"{parent}/{component}".lstrip("/")
            tree.setdefault(parent, {})
        leaf = components[-1]
        children = tree.setdefault(parent, {})
        if leaf in children:
            raise ArtifactVerificationError("artifact manifest contains a conflicting path")
        children[leaf] = "file"
    return tree, identities


def _bounded_directory_names(
    directory_fd: int,
    *,
    expected_count: int,
    operation: str,
) -> tuple[str, ...]:
    if expected_count > MAX_ARTIFACT_FILES:
        raise ArtifactVerificationError("artifact directory exceeds the file-count limit")
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if len(names) >= expected_count or len(names) >= MAX_ARTIFACT_FILES:
                    raise ArtifactVerificationError(
                        "artifact directory does not match the manifest file set"
                    )
                names.append(entry.name)
    except OSError as error:
        raise ArtifactVerificationError(f"cannot {operation} artifact directory") from error
    return tuple(sorted(names))


def _capture_directory(
    source_fd: int,
    destination_fd: int,
    *,
    prefix: str,
    tree: dict[str, dict[str, _TreeKind]],
    identities: dict[str, ArtifactFileV1],
) -> None:
    before = os.fstat(source_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise ArtifactVerificationError("artifact tree contains a non-directory entry")
    expected_children = tree[prefix]
    names_before = _bounded_directory_names(
        source_fd,
        expected_count=len(expected_children),
        operation="enumerate",
    )
    if names_before != tuple(sorted(expected_children)):
        raise ArtifactVerificationError("artifact directory does not match the manifest file set")
    for name in names_before:
        relative_path = f"{prefix}/{name}" if prefix else name
        kind = expected_children[name]
        try:
            entry_before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as error:
            raise ArtifactVerificationError(f"cannot inspect artifact: {relative_path}") from error
        if kind == "file":
            if not stat.S_ISREG(entry_before.st_mode):
                raise ArtifactVerificationError(f"artifact must be a regular file: {relative_path}")
            _copy_regular_file(
                source_fd,
                destination_fd,
                name,
                relative_path=relative_path,
                expected=identities[relative_path],
            )
            continue
        if not stat.S_ISDIR(entry_before.st_mode):
            raise ArtifactVerificationError(f"artifact must be a real directory: {relative_path}")
        child_source_fd = -1
        child_destination_fd = -1
        try:
            os.mkdir(name, 0o700, dir_fd=destination_fd)
            child_source_fd = os.open(name, _directory_flags(), dir_fd=source_fd)
            child_destination_fd = os.open(name, _directory_flags(), dir_fd=destination_fd)
        except OSError as error:
            if child_source_fd >= 0:
                os.close(child_source_fd)
            if child_destination_fd >= 0:
                os.close(child_destination_fd)
            raise ArtifactVerificationError(
                f"cannot safely open artifact directory: {relative_path}"
            ) from error
        try:
            child_before = os.fstat(child_source_fd)
            if _stat_identity(entry_before) != _stat_identity(child_before):
                raise ArtifactVerificationError(
                    f"artifact directory changed while opened: {relative_path}"
                )
            _capture_directory(
                child_source_fd,
                child_destination_fd,
                prefix=relative_path,
                tree=tree,
                identities=identities,
            )
            child_after = os.fstat(child_source_fd)
            entry_after = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as error:
            raise ArtifactVerificationError(
                f"cannot verify artifact directory: {relative_path}"
            ) from error
        finally:
            os.close(child_source_fd)
            os.close(child_destination_fd)
        if _stat_identity(child_before) != _stat_identity(child_after) or _stat_identity(
            child_before
        ) != _stat_identity(entry_after):
            raise ArtifactVerificationError(
                f"artifact directory changed while read: {relative_path}"
            )
    after = os.fstat(source_fd)
    names_after = _bounded_directory_names(
        source_fd,
        expected_count=len(expected_children),
        operation="re-enumerate",
    )
    if names_after != names_before or _stat_identity(before) != _stat_identity(after):
        raise ArtifactVerificationError("artifact directory changed while read")


@contextmanager
def open_verified_artifact_bundle(
    root: Path,
    expected_manifest_sha256: str,
) -> Iterator[VerifiedArtifactBundle]:
    """Capture and verify an exact artifact tree, then yield its private materialization."""

    _require_hardened_posix()
    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise ArtifactVerificationError(
            "artifact manifest SHA-256 must be 64 lowercase hexadecimal characters"
        )
    try:
        source_fd = os.open(root, _directory_flags())
    except OSError as error:
        raise ArtifactVerificationError(
            f"artifact root must be a real directory: {root}"
        ) from error
    try:
        manifest_bytes = _read_regular_bytes(
            source_fd,
            ARTIFACT_MANIFEST_NAME,
            display_path=ARTIFACT_MANIFEST_NAME,
            max_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
        )
        observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise ArtifactVerificationError("artifact manifest SHA-256 does not match")
        manifest = parse_canonical_artifact_manifest(manifest_bytes)
        manifest_identity = ArtifactFileV1(
            path=ARTIFACT_MANIFEST_NAME,
            sha256=observed_manifest_sha256,
            bytes=len(manifest_bytes),
        )
        expected_files = (*manifest.files, manifest_identity)
        tree, identities = _expected_tree(expected_files)
        with tempfile.TemporaryDirectory(prefix="ste-artifact-") as temporary:
            materialized = Path(temporary) / "bundle"
            materialized.mkdir(mode=0o700)
            destination_fd = os.open(materialized, _directory_flags())
            try:
                _capture_directory(
                    source_fd,
                    destination_fd,
                    prefix="",
                    tree=tree,
                    identities=identities,
                )
            finally:
                os.close(destination_fd)
            yield VerifiedArtifactBundle(
                path=materialized,
                manifest=manifest,
                manifest_sha256=observed_manifest_sha256,
            )
    finally:
        os.close(source_fd)


def verify_artifact_bundle(
    root: Path,
    expected_manifest_sha256: str,
) -> ArtifactBundleManifestV1:
    """Verify a complete bundle and return its content-bound manifest."""

    with open_verified_artifact_bundle(root, expected_manifest_sha256) as verified:
        return verified.manifest

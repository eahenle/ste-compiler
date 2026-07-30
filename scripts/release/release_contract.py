"""Validate release refs and finalize deterministic release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

TAG_PATTERN = re.compile(r"v(?P<version>0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTITY_SCHEMA = "ste-release-build-identity-v1"
MANIFEST_SCHEMA = "ste-release-build-manifest-v1"
Mode = Literal["dry-run", "tag"]


class ReleaseContractError(RuntimeError):
    """Raised when release inputs do not meet the reviewed trust contract."""


@dataclass(frozen=True)
class ReleaseIdentity:
    """Immutable source identity accepted by the release workflow."""

    schema_version: str
    mode: Mode
    version: str
    commit: str
    source_date_epoch: int
    tag: str | None

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON-compatible identity fields."""

        return {
            "commit": self.commit,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "source_date_epoch": self.source_date_epoch,
            "tag": self.tag,
            "version": self.version,
        }


def _run(root: Path, *command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else ""
        detail = f": {stderr}" if stderr else ""
        raise ReleaseContractError(f"command failed: {command!r}{detail}") from error
    return completed.stdout.strip()


def _project_version(root: Path) -> str:
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        version = project["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ReleaseContractError(f"cannot read project version: {error}") from error
    if not isinstance(version, str) or TAG_PATTERN.fullmatch(f"v{version}") is None:
        raise ReleaseContractError("project version must be a stable three-component SemVer")
    return version


def _citation_version(root: Path) -> str:
    try:
        citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReleaseContractError(f"cannot read citation version: {error}") from error
    if not isinstance(citation, dict):
        raise ReleaseContractError("CITATION.cff must define a string version")
    version = citation.get("version")
    if not isinstance(version, str):
        raise ReleaseContractError("CITATION.cff must define a string version")
    return version


def _configured_signers(path: Path) -> tuple[str, ...]:
    try:
        lines = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as error:
        raise ReleaseContractError(f"cannot read trusted tag signers: {error}") from error
    if not lines:
        raise ReleaseContractError(
            "signed-tag releases are disabled until trusted-tag-signers contains a reviewed SSH key"
        )
    return lines


def validate_release_ref(
    root: Path,
    *,
    mode: Mode,
    commit: str,
    tag: str | None,
    allowed_signers: Path,
) -> ReleaseIdentity:
    """Validate a manual dry-run or one exact signed stable-version tag."""

    root = root.resolve()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise ReleaseContractError("release commit must be a full lowercase Git SHA-1")
    head = _run(root, "git", "rev-parse", "HEAD")
    if head != commit:
        raise ReleaseContractError(f"release commit {commit} does not equal checkout HEAD {head}")
    if _run(root, "git", "status", "--porcelain"):
        raise ReleaseContractError("release checkout must be clean before validation")

    version = _project_version(root)
    citation_version = _citation_version(root)
    if citation_version != version:
        raise ReleaseContractError(
            f"CITATION.cff version {citation_version!r} does not equal project version {version!r}"
        )

    if mode == "dry-run":
        if tag is not None:
            raise ReleaseContractError("manual dry-runs must not supply or claim a release tag")
    elif mode == "tag":
        if tag is None or TAG_PATTERN.fullmatch(tag) is None:
            raise ReleaseContractError("tag releases require an exact vMAJOR.MINOR.PATCH tag")
        if tag != f"v{version}":
            raise ReleaseContractError(
                f"release tag {tag!r} does not equal package version tag {f'v{version}'!r}"
            )
        object_type = _run(root, "git", "cat-file", "-t", f"refs/tags/{tag}")
        if object_type != "tag":
            raise ReleaseContractError("release tag must be annotated, not lightweight")
        tagged_commit = _run(root, "git", "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if tagged_commit != commit:
            raise ReleaseContractError(
                f"release tag resolves to {tagged_commit}, not requested commit {commit}"
            )
        _configured_signers(allowed_signers)
        _run(
            root,
            "git",
            "-c",
            "gpg.format=ssh",
            "-c",
            f"gpg.ssh.allowedSignersFile={allowed_signers.resolve()}",
            "verify-tag",
            tag,
        )
    else:  # pragma: no cover - argparse and the Mode type keep callers explicit.
        raise ReleaseContractError(f"unsupported release mode: {mode!r}")

    source_date_epoch_raw = _run(root, "git", "show", "-s", "--format=%ct", commit)
    try:
        source_date_epoch = int(source_date_epoch_raw)
    except ValueError as error:
        raise ReleaseContractError("release commit timestamp is not an integer") from error
    if source_date_epoch < 0:
        raise ReleaseContractError("release commit timestamp cannot be negative")
    return ReleaseIdentity(
        schema_version=IDENTITY_SCHEMA,
        mode=mode,
        version=version,
        commit=commit,
        source_date_epoch=source_date_epoch,
        tag=tag,
    )


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_identity(identity: ReleaseIdentity, output: Path) -> None:
    """Write a canonical identity file outside the release artifact directory."""

    if output.exists():
        raise ReleaseContractError("release identity output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json(identity.as_dict()))


def read_identity(path: Path) -> ReleaseIdentity:
    """Read a strict identity produced by ``validate-ref``."""

    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"cannot read release identity: {error}") from error
    return _parse_identity(payload)


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ReleaseContractError(f"JSON object contains duplicate field {key!r}")
        payload[key] = value
    return payload


def _parse_identity(payload: Any) -> ReleaseIdentity:
    expected_keys = {
        "commit",
        "mode",
        "schema_version",
        "source_date_epoch",
        "tag",
        "version",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ReleaseContractError("release identity has an unexpected field inventory")
    if payload["schema_version"] != IDENTITY_SCHEMA:
        raise ReleaseContractError(f"release identity schema must be {IDENTITY_SCHEMA!r}")
    mode = payload["mode"]
    if mode not in {"dry-run", "tag"}:
        raise ReleaseContractError("release identity mode must be dry-run or tag")
    tag = payload["tag"]
    if tag is not None and not isinstance(tag, str):
        raise ReleaseContractError("release identity tag must be null or a string")
    if (
        not isinstance(payload["version"], str)
        or TAG_PATTERN.fullmatch(f"v{payload['version']}") is None
        or not isinstance(payload["commit"], str)
        or COMMIT_PATTERN.fullmatch(payload["commit"]) is None
        or not isinstance(payload["source_date_epoch"], int)
        or isinstance(payload["source_date_epoch"], bool)
        or payload["source_date_epoch"] < 0
    ):
        raise ReleaseContractError("release identity contains invalid primitive values")
    if (mode == "tag") != (tag is not None):
        raise ReleaseContractError("only tag-mode release identities can contain a tag")
    if tag is not None and tag != f"v{payload['version']}":
        raise ReleaseContractError("release identity tag and version do not match")
    return ReleaseIdentity(
        schema_version=IDENTITY_SCHEMA,
        mode=mode,
        version=payload["version"],
        commit=payload["commit"],
        source_date_epoch=payload["source_date_epoch"],
        tag=tag,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_files(root: Path, excluded: frozenset[Path]) -> tuple[Path, ...]:
    resolved = root.resolve()
    entries = tuple(sorted(resolved.rglob("*")))
    if any(path.is_symlink() for path in entries):
        raise ReleaseContractError("release evidence must not contain symbolic links")
    files = tuple(path for path in entries if path.is_file() and path.resolve() not in excluded)
    if not files:
        raise ReleaseContractError("release evidence directory cannot be empty")
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(resolved):
            raise ReleaseContractError("release evidence must contain only in-tree regular files")
    return files


def _validate_release_layout(root: Path, identity: ReleaseIdentity) -> None:
    root_entries = tuple(root.iterdir())
    if any(path.is_symlink() for path in root_entries):
        raise ReleaseContractError("release evidence must not contain symbolic links")
    if {path.name for path in root_entries} != {
        "distributions",
        "ste-compiler.spdx.json",
    }:
        raise ReleaseContractError("release has an unexpected pre-finalization file inventory")
    distributions = root / "distributions"
    if not distributions.is_dir() or distributions.is_symlink():
        raise ReleaseContractError("release must contain a real distributions directory")
    distribution_files = tuple(sorted(distributions.iterdir()))
    if (
        any(not path.is_file() or path.is_symlink() for path in distribution_files)
        or len(distribution_files) != 2
    ):
        raise ReleaseContractError(
            "release must contain exactly one wheel and one source distribution"
        )
    expected_sdist = f"ste_compiler-{identity.version}.tar.gz"
    wheels = tuple(
        path
        for path in distribution_files
        if path.name.startswith(f"ste_compiler-{identity.version}-") and path.suffix == ".whl"
    )
    if len(wheels) != 1 or {path.name for path in distribution_files} != {
        wheels[0].name if wheels else "",
        expected_sdist,
    }:
        raise ReleaseContractError(
            "distribution filenames do not match the release identity version"
        )

    sbom = root / "ste-compiler.spdx.json"
    if not sbom.is_file() or sbom.is_symlink():
        raise ReleaseContractError("release must contain a real SPDX JSON SBOM")
    try:
        payload: Any = json.loads(sbom.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"cannot read release SPDX JSON SBOM: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("spdxVersion") != "SPDX-2.3"
        or not isinstance(payload.get("SPDXID"), str)
    ):
        raise ReleaseContractError("release SBOM must be an SPDX 2.3 JSON document")


def finalize_release(release_root: Path, identity_path: Path) -> tuple[Path, Path]:
    """Write canonical build metadata and checksums after SBOM generation."""

    if release_root.is_symlink():
        raise ReleaseContractError("release root must be a real existing directory")
    release_root = release_root.resolve()
    if not release_root.is_dir():
        raise ReleaseContractError("release root must be a real existing directory")
    identity = read_identity(identity_path)
    manifest_path = release_root / "release-build.json"
    checksums_path = release_root / "SHA256SUMS"
    if manifest_path.exists() or checksums_path.exists():
        raise ReleaseContractError("release metadata outputs must not already exist")
    _validate_release_layout(release_root, identity)

    subjects = _release_files(
        release_root,
        frozenset({manifest_path.resolve(), checksums_path.resolve()}),
    )
    manifest = {
        **identity.as_dict(),
        "artifacts": [
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(release_root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in subjects
        ],
        "identity_schema_version": identity.schema_version,
        "schema_version": MANIFEST_SCHEMA,
    }
    manifest_path.write_bytes(_canonical_json(manifest))
    checksum_subjects = (*subjects, manifest_path)
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(release_root).as_posix()}\n"
            for path in sorted(checksum_subjects)
        ),
        encoding="utf-8",
    )
    return manifest_path, checksums_path


def _read_release_manifest(
    manifest_path: Path,
) -> tuple[ReleaseIdentity, tuple[tuple[str, int, str], ...]]:
    try:
        encoded = manifest_path.read_bytes()
        payload: Any = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"cannot read release build manifest: {error}") from error
    expected_keys = {
        "artifacts",
        "commit",
        "identity_schema_version",
        "mode",
        "schema_version",
        "source_date_epoch",
        "tag",
        "version",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ReleaseContractError("release build manifest has an unexpected field inventory")
    if encoded != _canonical_json(payload):
        raise ReleaseContractError("release build manifest is not canonical JSON")
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise ReleaseContractError(f"release build manifest schema must be {MANIFEST_SCHEMA!r}")
    identity = _parse_identity(
        {
            "commit": payload["commit"],
            "mode": payload["mode"],
            "schema_version": payload["identity_schema_version"],
            "source_date_epoch": payload["source_date_epoch"],
            "tag": payload["tag"],
            "version": payload["version"],
        }
    )

    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ReleaseContractError("release build manifest must contain an artifact inventory")
    artifacts: list[tuple[str, int, str]] = []
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"bytes", "path", "sha256"}:
            raise ReleaseContractError("release artifact has an unexpected field inventory")
        path = artifact["path"]
        size = artifact["bytes"]
        digest = artifact["sha256"]
        if not isinstance(path, str) or not path:
            raise ReleaseContractError("release artifact path must be a nonempty string")
        normalized = PurePosixPath(path)
        if (
            normalized.is_absolute()
            or normalized.as_posix() != path
            or "\\" in path
            or any(part in {".", ".."} for part in normalized.parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ReleaseContractError("release artifact path must be canonical and relative")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReleaseContractError("release artifact size must be a nonnegative integer")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseContractError("release artifact SHA-256 must be lowercase hexadecimal")
        artifacts.append((path, size, digest))
    artifact_paths = tuple(artifact[0] for artifact in artifacts)
    if artifact_paths != tuple(sorted(set(artifact_paths))):
        raise ReleaseContractError("release artifact inventory must be unique and sorted")
    return identity, tuple(artifacts)


def _verified_release_root(release_root: Path) -> Path:
    if release_root.is_symlink():
        raise ReleaseContractError("release bundle root must be a real existing directory")
    root = release_root.resolve()
    if not root.is_dir():
        raise ReleaseContractError("release bundle root must be a real existing directory")
    root_entries = tuple(root.iterdir())
    if any(path.is_symlink() for path in root_entries):
        raise ReleaseContractError("release bundle must not contain symbolic links")
    if {path.name for path in root_entries} != {
        "SHA256SUMS",
        "distributions",
        "release-build.json",
        "ste-compiler.spdx.json",
    }:
        raise ReleaseContractError("release bundle has an unexpected file inventory")
    if not (root / "distributions").is_dir():
        raise ReleaseContractError("release bundle must contain a real distributions directory")
    for name in ("SHA256SUMS", "release-build.json", "ste-compiler.spdx.json"):
        if not (root / name).is_file():
            raise ReleaseContractError(f"release bundle must contain a real {name} file")
    return root


def _verified_release_subjects(root: Path, identity: ReleaseIdentity) -> tuple[Path, ...]:
    distributions = root / "distributions"
    distribution_files = tuple(sorted(distributions.iterdir()))
    if (
        any(not path.is_file() or path.is_symlink() for path in distribution_files)
        or len(distribution_files) != 2
    ):
        raise ReleaseContractError(
            "release bundle must contain exactly one wheel and one source distribution"
        )
    expected_sdist = f"ste_compiler-{identity.version}.tar.gz"
    wheels = tuple(
        path
        for path in distribution_files
        if path.name.startswith(f"ste_compiler-{identity.version}-") and path.suffix == ".whl"
    )
    if len(wheels) != 1 or {path.name for path in distribution_files} != {
        wheels[0].name if wheels else "",
        expected_sdist,
    }:
        raise ReleaseContractError(
            "release bundle distribution filenames do not match the release identity version"
        )
    manifest_path = root / "release-build.json"
    checksums_path = root / "SHA256SUMS"
    subjects = _release_files(
        root,
        frozenset({manifest_path.resolve(), checksums_path.resolve()}),
    )
    expected_paths = {
        f"distributions/{wheels[0].name}",
        f"distributions/{expected_sdist}",
        "ste-compiler.spdx.json",
    }
    if {path.relative_to(root).as_posix() for path in subjects} != expected_paths:
        raise ReleaseContractError("release bundle has an unexpected artifact inventory")
    return subjects


def verify_release_bundle(
    release_root: Path,
    *,
    expected_commit: str,
    expected_mode: Mode,
) -> ReleaseIdentity:
    """Verify an untrusted release bundle and return its validated identity."""

    if COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ReleaseContractError("expected release commit must be a full lowercase Git SHA-1")
    if expected_mode not in {"dry-run", "tag"}:
        raise ReleaseContractError("expected release mode must be dry-run or tag")
    root = _verified_release_root(release_root)
    manifest_path = root / "release-build.json"
    identity, artifacts = _read_release_manifest(manifest_path)
    if identity.mode != expected_mode:
        raise ReleaseContractError(
            f"release bundle mode {identity.mode!r} does not equal expected mode {expected_mode!r}"
        )
    if identity.commit != expected_commit:
        raise ReleaseContractError(
            f"release bundle commit {identity.commit} does not equal expected commit {expected_commit}"
        )

    subjects = _verified_release_subjects(root, identity)
    subject_by_path = {path.relative_to(root).as_posix(): path for path in subjects}
    if tuple(subject_by_path) != tuple(artifact[0] for artifact in artifacts):
        raise ReleaseContractError(
            "release build manifest artifact inventory does not match the release bundle"
        )
    for relative_path, expected_size, expected_digest in artifacts:
        artifact_path = subject_by_path[relative_path]
        if artifact_path.stat().st_size != expected_size:
            raise ReleaseContractError(
                f"release artifact size does not match manifest: {relative_path}"
            )
        if _sha256(artifact_path) != expected_digest:
            raise ReleaseContractError(
                f"release artifact SHA-256 does not match manifest: {relative_path}"
            )

    checksum_subjects = tuple(sorted((*subjects, manifest_path)))
    canonical_checksums = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_subjects
    ).encode()
    try:
        actual_checksums = (root / "SHA256SUMS").read_bytes()
    except OSError as error:
        raise ReleaseContractError(f"cannot read release bundle checksums: {error}") from error
    if actual_checksums != canonical_checksums:
        raise ReleaseContractError("SHA256SUMS is not the canonical release bundle checksum list")
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-ref")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--mode", choices=("dry-run", "tag"), required=True)
    validate.add_argument("--commit", required=True)
    validate.add_argument("--tag")
    validate.add_argument("--allowed-signers", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--release-root", type=Path, required=True)
    finalize.add_argument("--identity", type=Path, required=True)
    verify = subparsers.add_parser("verify-bundle")
    verify.add_argument("--release-root", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--expected-mode", choices=("dry-run", "tag"), required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "validate-ref":
            identity = validate_release_ref(
                args.root,
                mode=args.mode,
                commit=args.commit,
                tag=args.tag,
                allowed_signers=args.allowed_signers,
            )
            write_identity(identity, args.output)
            print(json.dumps(identity.as_dict(), sort_keys=True))
        elif args.command == "finalize":
            manifest, checksums = finalize_release(args.release_root, args.identity)
            print(
                json.dumps(
                    {
                        "checksums": str(checksums),
                        "manifest": str(manifest),
                        "status": "finalized",
                    },
                    sort_keys=True,
                )
            )
        else:
            identity = verify_release_bundle(
                args.release_root,
                expected_commit=args.expected_commit,
                expected_mode=args.expected_mode,
            )
            print(
                json.dumps(
                    identity.as_dict(),
                    sort_keys=True,
                )
            )
    except ReleaseContractError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

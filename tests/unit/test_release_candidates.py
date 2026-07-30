from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from scripts.release import release_candidates as candidates
from scripts.release.release_candidates import (
    BENCHMARK_INPUTS,
    BENCHMARK_REPORT_FILES,
    CANDIDATE_SCHEMA,
    DATASET_ID,
    REPORT_ID,
    ReleaseCandidateError,
    build_candidate_directory,
    verify_candidate_directory,
)
from scripts.release.release_contract import (
    IDENTITY_SCHEMA,
    ReleaseIdentity,
    write_identity,
)
from ste_compiler.training.release import EXPECTED_RELEASE_FILES

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def source_repository(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("candidate-source") / "repository"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--local",
            "--no-hardlinks",
            str(ROOT),
            str(source),
        ],
        check=True,
    )
    return source


def _identity_for_repository(
    source: Path,
    *,
    mode: Literal["dry-run", "tag"] = "dry-run",
) -> ReleaseIdentity:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    epoch = int(
        subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return ReleaseIdentity(
        schema_version=IDENTITY_SCHEMA,
        mode=mode,
        version="0.1.0",
        commit=commit,
        source_date_epoch=epoch,
        tag="v0.1.0" if mode == "tag" else None,
    )


@pytest.fixture(scope="module")
def identity(source_repository: Path) -> ReleaseIdentity:
    return _identity_for_repository(source_repository)


@pytest.fixture(scope="module")
def candidate_directory(
    tmp_path_factory: pytest.TempPathFactory,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> Path:
    output = tmp_path_factory.mktemp("release-candidates") / "candidates"
    build_candidate_directory(source_repository, identity, output)
    return output


def _dataset_name(identity: ReleaseIdentity) -> str:
    return f"ste-compiler-{identity.version}-dataset-{DATASET_ID}.tar"


def _report_name(identity: ReleaseIdentity) -> str:
    return f"ste-compiler-{identity.version}-report-{REPORT_ID}.tar"


def _archive_members(path: Path) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member) if member.isreg() else None
            members.append((copy.copy(member), handle.read() if handle is not None else None))
    return members


def _write_members(
    path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
    *,
    archive_format: int = tarfile.USTAR_FORMAT,
    mode: str = "w:",
    pax_headers: dict[str, str] | None = None,
) -> None:
    with tarfile.open(
        path,
        mode=mode,
        format=archive_format,
        pax_headers=pax_headers,
    ) as archive:
        for member, data in members:
            archive.addfile(member, None if data is None else candidates.io.BytesIO(data))


def _copy_candidates(candidate_directory: Path, tmp_path: Path) -> Path:
    copied = tmp_path / "candidates"
    shutil.copytree(candidate_directory, copied)
    return copied


def _manifest(
    archive_path: Path,
    identity: ReleaseIdentity,
    kind: candidates.ArtifactKind,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    payload, manifest_bytes, _, _ = candidates._read_candidate_archive(
        archive_path,
        identity=identity,
        kind=kind,
    )
    return payload, json.loads(manifest_bytes)


def _rewrite_manifest(
    archive_path: Path,
    identity: ReleaseIdentity,
    kind: candidates.ArtifactKind,
    mutate: Callable[[dict[str, Any]], None],
    *,
    canonical: bool = True,
) -> None:
    payload, manifest = _manifest(archive_path, identity, kind)
    mutate(manifest)
    if canonical:
        manifest_bytes = candidates._canonical_json(manifest)
    else:
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    archive_path.write_bytes(
        candidates._archive_bytes(
            identity=identity,
            kind=kind,
            payload=payload,
            manifest_bytes=manifest_bytes,
        )
    )


def _source_copy(source_repository: Path, tmp_path: Path, mtime: int) -> Path:
    source = tmp_path / f"source-{mtime}"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--local",
            "--no-hardlinks",
            str(source_repository),
            str(source),
        ],
        check=True,
    )
    for path in sorted(source.rglob("*"), reverse=True):
        os.utime(path, (mtime, mtime), follow_symlinks=False)
    os.utime(source, (mtime, mtime))
    return source


def test_build_is_byte_reproducible_across_roots_and_machine_mtimes(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    first_source = _source_copy(source_repository, tmp_path, 1_600_000_000)
    second_source = _source_copy(source_repository, tmp_path, 1_900_000_000)
    first_output = tmp_path / "first-candidates"
    second_output = tmp_path / "second-candidates"

    build_candidate_directory(first_source, identity, first_output)
    build_candidate_directory(second_source, identity, second_output)

    assert {path.name: path.read_bytes() for path in first_output.iterdir()} == {
        path.name: path.read_bytes() for path in second_output.iterdir()
    }
    verify_candidate_directory(first_output, identity)
    verify_candidate_directory(second_output, identity)


@pytest.mark.parametrize(
    ("kind", "expected_payload"),
    [
        ("dataset", frozenset(EXPECTED_RELEASE_FILES)),
        ("report", frozenset((*BENCHMARK_INPUTS, *BENCHMARK_REPORT_FILES))),
    ],
)
def test_archives_have_exact_ustar_inventory_metadata_and_outer_identity(
    candidate_directory: Path,
    identity: ReleaseIdentity,
    kind: candidates.ArtifactKind,
    expected_payload: frozenset[str],
) -> None:
    archive_name = _dataset_name(identity) if kind == "dataset" else _report_name(identity)
    archive_path = candidate_directory / archive_name
    top = archive_name.removesuffix(".tar")
    members = _archive_members(archive_path)
    names = [member.name for member, _ in members]
    expected_names = sorted(
        {
            top,
            f"{top}/release-candidate.json",
            *(f"{top}/{path}" for path in expected_payload),
        },
        key=lambda value: value.encode("utf-8"),
    )

    assert names == expected_names
    for member, data in members:
        assert member.uid == member.gid == 0
        assert member.uname == member.gname == ""
        assert member.mtime == identity.source_date_epoch
        assert not member.pax_headers
        if data is None:
            assert member.name == top
            assert member.isdir()
            assert member.mode == 0o755
            assert member.size == 0
        else:
            assert member.isreg()
            assert member.mode == 0o644
            assert member.size == len(data)

    payload, manifest = _manifest(archive_path, identity, kind)
    assert set(payload) == expected_payload
    assert manifest["schema_version"] == CANDIDATE_SCHEMA
    assert manifest["archive"] == archive_name
    assert manifest["artifact_kind"] == kind
    assert manifest["artifact_id"] == (DATASET_ID if kind == "dataset" else REPORT_ID)
    assert manifest["top_level_directory"] == top
    assert manifest["release_identity"] == identity.as_dict()
    content_manifest = "manifest.json" if kind == "dataset" else "report-manifest.json"
    assert manifest["content_identity"] == {
        "manifest_path": content_manifest,
        "manifest_schema_version": (
            "demonstration-corpus-release-v1"
            if kind == "dataset"
            else "ste-benchmark-report-manifest-v1"
        ),
        "manifest_sha256": candidates._sha256(payload[content_manifest]),
    }


def test_dataset_payload_is_exact_checked_corpus(
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    payload, _ = _manifest(candidate_directory / _dataset_name(identity), identity, "dataset")

    assert payload == {
        name: (ROOT / "datasets/demonstration-corpus-2" / name).read_bytes()
        for name in EXPECTED_RELEASE_FILES
    }


def test_report_payload_is_exact_checked_input_and_regenerated_report(
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    payload, _ = _manifest(candidate_directory / _report_name(identity), identity, "report")
    benchmark = ROOT / "data/benchmark/v1"

    assert {name: payload[name] for name in BENCHMARK_INPUTS} == {
        name: (benchmark / name).read_bytes() for name in BENCHMARK_INPUTS
    }
    assert {name: payload[name] for name in BENCHMARK_REPORT_FILES} == {
        name: (benchmark / "expected-report" / name).read_bytes() for name in BENCHMARK_REPORT_FILES
    }


def test_report_dependency_cross_links_dataset_archive_and_corpus_manifest(
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    dataset_payload, dataset_manifest = _manifest(
        candidate_directory / _dataset_name(identity),
        identity,
        "dataset",
    )
    _, report_manifest = _manifest(
        candidate_directory / _report_name(identity),
        identity,
        "report",
    )

    assert dataset_manifest["dependencies"] == []
    assert report_manifest["dependencies"] == [
        {
            "archive": _dataset_name(identity),
            "archive_sha256": candidates._sha256(
                (candidate_directory / _dataset_name(identity)).read_bytes()
            ),
            "artifact_id": DATASET_ID,
            "artifact_kind": "dataset",
            "candidate_manifest_sha256": candidates._sha256(
                candidates._canonical_json(dataset_manifest)
            ),
            "corpus_manifest_sha256": candidates._sha256(dataset_payload["manifest.json"]),
        }
    ]


def test_dry_run_and_tag_candidates_bind_distinct_full_identities(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    tagged = _identity_for_repository(source_repository, mode="tag")
    dry_output = tmp_path / "dry"
    tag_output = tmp_path / "tag"

    build_candidate_directory(source_repository, identity, dry_output)
    build_candidate_directory(source_repository, tagged, tag_output)

    assert (dry_output / _dataset_name(identity)).read_bytes() != (
        tag_output / _dataset_name(tagged)
    ).read_bytes()
    verify_candidate_directory(dry_output, identity)
    verify_candidate_directory(tag_output, tagged)
    with pytest.raises(ReleaseCandidateError, match="outer release identity"):
        verify_candidate_directory(dry_output, tagged)
    with pytest.raises(ReleaseCandidateError, match="outer release identity"):
        verify_candidate_directory(tag_output, identity)


def _add_link(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top = members[0][0].name
    link = copy.copy(members[1][0])
    link.name = f"{top}/link"
    link.type = tarfile.SYMTYPE
    link.linkname = "release-candidate.json"
    link.size = 0
    members.append((link, None))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _add_device(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top = members[0][0].name
    device = copy.copy(members[1][0])
    device.name = f"{top}/device"
    device.type = tarfile.CHRTYPE
    device.size = 0
    members.append((device, None))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _add_hardlink(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top = members[0][0].name
    link = copy.copy(members[1][0])
    link.name = f"{top}/hardlink"
    link.type = tarfile.LNKTYPE
    link.linkname = members[1][0].name
    link.size = 0
    members.append((link, None))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _add_fifo(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top = members[0][0].name
    fifo = copy.copy(members[1][0])
    fifo.name = f"{top}/fifo"
    fifo.type = tarfile.FIFOTYPE
    fifo.size = 0
    members.append((fifo, None))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _add_sparse(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].type = tarfile.GNUTYPE_SPARSE


def _use_areg_type(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].type = tarfile.AREGTYPE


def _use_unknown_type(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].type = b"Z"


def _traversal_path(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].name = "../escape"
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _backslash_path(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].name = members[1][0].name.replace("/", "\\", 1)
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _aliased_path(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top, relative = members[1][0].name.split("/", maxsplit=1)
    members[1][0].name = f"{top}//{relative}"
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _duplicate_path(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members.append((copy.copy(members[1][0]), members[1][1]))


def _extra_path(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    top = members[0][0].name
    extra = copy.copy(members[1][0])
    extra.name = f"{top}/extra.txt"
    extra.size = 5
    members.append((extra, b"extra"))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))


def _missing_payload(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    del members[1]


def _missing_directory(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    del members[0]


def _tamper_payload(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    member, data = members[1]
    assert data is not None
    tampered = data + b"x"
    member.size = len(tampered)
    members[1] = (member, tampered)


def _nonzero_owner(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].uid = 1


def _wrong_mode(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].mode = 0o600


def _wrong_mtime(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members[1][0].mtime += 1


def _reverse_order(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    members.reverse()


TarMutation = Callable[[list[tuple[tarfile.TarInfo, bytes | None]]], None]


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("link", _add_link),
        ("hardlink", _add_hardlink),
        ("device", _add_device),
        ("fifo", _add_fifo),
        ("sparse", _add_sparse),
        ("areg", _use_areg_type),
        ("unknown-type", _use_unknown_type),
        ("traversal", _traversal_path),
        ("backslash", _backslash_path),
        ("alias", _aliased_path),
        ("duplicate", _duplicate_path),
        ("extra", _extra_path),
        ("missing-payload", _missing_payload),
        ("missing-directory", _missing_directory),
        ("tamper", _tamper_payload),
        ("owner", _nonzero_owner),
        ("mode", _wrong_mode),
        ("mtime", _wrong_mtime),
        ("order", _reverse_order),
    ],
)
def test_verifier_rejects_hostile_or_noncanonical_tar_members(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    name: str,
    mutation: TarMutation,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    members = _archive_members(archive_path)
    mutation(members)
    _write_members(archive_path, members)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


@pytest.mark.parametrize("archive_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_verifier_rejects_pax_and_gnu_tar_formats(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    archive_format: int,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    members = _archive_members(archive_path)
    if archive_format == tarfile.PAX_FORMAT:
        members[1][0].pax_headers = {"comment": "not allowed"}
    _write_members(archive_path, members, archive_format=archive_format)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def test_verifier_rejects_global_pax_headers(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    members = _archive_members(archive_path)
    _write_members(
        archive_path,
        members,
        archive_format=tarfile.PAX_FORMAT,
        pax_headers={"comment": "not allowed"},
    )

    with pytest.raises(ReleaseCandidateError, match="PAX"):
        verify_candidate_directory(copied, identity)


@pytest.mark.parametrize("extension", ["longname", "longlink"])
def test_verifier_rejects_gnu_longname_and_longlink_extensions(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    extension: str,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    members = _archive_members(archive_path)
    top = members[0][0].name
    extended = copy.copy(members[1][0])
    extended.name = f"{top}/{'n' * 120}"
    data: bytes | None = b"x"
    extended.size = 1
    if extension == "longlink":
        extended.name = f"{top}/longlink"
        extended.type = tarfile.SYMTYPE
        extended.linkname = "l" * 120
        extended.size = 0
        data = None
    members.append((extended, data))
    members.sort(key=lambda item: item[0].name.encode("utf-8"))
    _write_members(archive_path, members, archive_format=tarfile.GNU_FORMAT)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def test_verifier_rejects_compressed_archive_under_tar_name(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    members = _archive_members(archive_path)
    _write_members(archive_path, members, mode="w:gz")

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def _recompute_tar_checksum(encoded: bytearray, header_offset: int = 0) -> None:
    header = encoded[header_offset : header_offset + tarfile.BLOCKSIZE]
    header[148:156] = b"        "
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    encoded[header_offset : header_offset + tarfile.BLOCKSIZE] = header


def _bad_magic(encoded: bytearray) -> None:
    encoded[257:263] = b"broken"
    _recompute_tar_checksum(encoded)


def _bad_version(encoded: bytearray) -> None:
    encoded[263:265] = b"99"
    _recompute_tar_checksum(encoded)


def _bad_checksum(encoded: bytearray) -> None:
    encoded[0] ^= 1


def _base256_number(encoded: bytearray) -> None:
    encoded[136:148] = b"\x80" + b"\0" * 11
    _recompute_tar_checksum(encoded)


def _noncanonical_octal(encoded: bytearray) -> None:
    encoded[100:108] = b"0000755 "
    _recompute_tar_checksum(encoded)


def _truncate_archive(encoded: bytearray) -> None:
    del encoded[-tarfile.BLOCKSIZE :]


def _nonzero_padding(encoded: bytearray) -> None:
    encoded[-1] = 1


def _trailing_garbage(encoded: bytearray) -> None:
    encoded.extend(b"garbage")


def _concatenate_archive(encoded: bytearray) -> None:
    encoded.extend(bytes(encoded))


RawTarMutation = Callable[[bytearray], None]


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("magic", _bad_magic),
        ("version", _bad_version),
        ("checksum", _bad_checksum),
        ("base256", _base256_number),
        ("octal", _noncanonical_octal),
        ("truncated", _truncate_archive),
        ("padding", _nonzero_padding),
        ("trailing", _trailing_garbage),
        ("concatenated", _concatenate_archive),
    ],
)
def test_verifier_rejects_noncanonical_raw_tar_encoding(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    name: str,
    mutation: RawTarMutation,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    encoded = bytearray(archive_path.read_bytes())
    mutation(encoded)
    archive_path.write_bytes(encoded)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def test_verifier_enforces_archive_size_and_member_count_bounds(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size_copy = _copy_candidates(candidate_directory, tmp_path / "size")
    dataset_path = size_copy / _dataset_name(identity)
    with monkeypatch.context() as size_patch:
        size_patch.setattr(
            candidates,
            "MAX_ARCHIVE_BYTES",
            dataset_path.stat().st_size - 1,
        )

        def forbid_path_read(*args: object, **kwargs: object) -> bytes:
            pytest.fail("oversized archive was read before its size was rejected")

        size_patch.setattr(Path, "read_bytes", forbid_path_read)
        with pytest.raises(ReleaseCandidateError, match="size limit"):
            verify_candidate_directory(size_copy, identity)

    count_copy = _copy_candidates(candidate_directory, tmp_path / "count")
    with monkeypatch.context() as count_patch:
        count_patch.setattr(candidates, "MAX_ARCHIVE_MEMBERS", 1)

        def forbid_getmembers(*args: object, **kwargs: object) -> list[tarfile.TarInfo]:
            pytest.fail("tar member inventory was materialized before its cap")

        count_patch.setattr(tarfile.TarFile, "getmembers", forbid_getmembers)
        with pytest.raises(ReleaseCandidateError, match="too many members"):
            verify_candidate_directory(count_copy, identity)


def test_verifier_reads_each_candidate_archive_once(
    candidate_directory: Path,
    identity: ReleaseIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read = candidates._read_regular
    reads: dict[str, int] = {}

    def count_reads(path: Path, **kwargs: object) -> bytes:
        if path.suffix == ".tar":
            reads[path.name] = reads.get(path.name, 0) + 1
        return real_read(path, **kwargs)

    monkeypatch.setattr(candidates, "_read_regular", count_reads)
    verify_candidate_directory(candidate_directory, identity)

    assert reads == {
        _dataset_name(identity): 1,
        _report_name(identity): 1,
    }


def test_canonical_archives_have_two_zero_end_blocks_and_record_padding(
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    for archive_name in (_dataset_name(identity), _report_name(identity)):
        archive_path = candidate_directory / archive_name
        encoded = archive_path.read_bytes()
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            last = members[-1]
            payload_blocks = (last.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
            content_end = last.offset_data + payload_blocks * tarfile.BLOCKSIZE
        suffix = encoded[content_end:]
        expected_padding = tarfile.RECORDSIZE - (content_end % tarfile.RECORDSIZE)

        assert len(encoded) % tarfile.RECORDSIZE == 0
        assert len(suffix) == expected_padding
        assert len(suffix) >= 2 * tarfile.BLOCKSIZE
        assert suffix == b"\0" * len(suffix)


@pytest.mark.parametrize(
    ("name", "mutate", "canonical"),
    [
        ("extra-field", lambda manifest: manifest.update({"extra": True}), True),
        ("missing-field", lambda manifest: manifest.pop("artifact_id"), True),
        (
            "identity",
            lambda manifest: manifest["release_identity"].update({"commit": "0" * 40}),
            True,
        ),
        (
            "content-manifest-hash",
            lambda manifest: manifest["content_identity"].update({"manifest_sha256": "0" * 64}),
            True,
        ),
        (
            "content-manifest-schema",
            lambda manifest: manifest["content_identity"].update(
                {"manifest_schema_version": "invented"}
            ),
            True,
        ),
        ("noncanonical-json", lambda manifest: None, False),
    ],
)
def test_verifier_rejects_hostile_candidate_manifests(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    canonical: bool,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    _rewrite_manifest(archive_path, identity, "dataset", mutate, canonical=canonical)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def test_verifier_rejects_duplicate_candidate_manifest_json_field(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _dataset_name(identity)
    payload, manifest = _manifest(archive_path, identity, "dataset")
    canonical = candidates._canonical_json(manifest)
    duplicate = canonical.replace(
        b"{\n",
        b'{\n  "schema_version": "ste-release-candidate-v1",\n',
        1,
    )
    archive_path.write_bytes(
        candidates._archive_bytes(
            identity=identity,
            kind="dataset",
            payload=payload,
            manifest_bytes=duplicate,
        )
    )

    with pytest.raises(ReleaseCandidateError, match="duplicate field"):
        verify_candidate_directory(copied, identity)


@pytest.mark.parametrize(
    "field",
    [
        "archive_sha256",
        "candidate_manifest_sha256",
        "corpus_manifest_sha256",
    ],
)
def test_verifier_rejects_tampered_report_dependency_cross_link(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    field: str,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    archive_path = copied / _report_name(identity)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["dependencies"][0][field] = "0" * 64

    _rewrite_manifest(archive_path, identity, "report", mutate)

    with pytest.raises(ReleaseCandidateError, match="dependency does not match"):
        verify_candidate_directory(copied, identity)


@pytest.mark.parametrize("change", ["extra", "missing", "symlink", "hardlink"])
def test_verifier_rejects_candidate_directory_inventory_and_links(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
    change: str,
) -> None:
    copied = _copy_candidates(candidate_directory, tmp_path)
    if change == "extra":
        (copied / "unexpected.tar").write_bytes(b"")
    elif change == "missing":
        (copied / _report_name(identity)).unlink()
    elif change == "symlink":
        dataset = copied / _dataset_name(identity)
        dataset.unlink()
        dataset.symlink_to(candidate_directory / _dataset_name(identity))
    else:
        dataset = copied / _dataset_name(identity)
        external = tmp_path / "dataset.tar"
        dataset.replace(external)
        os.link(external, dataset)

    with pytest.raises(ReleaseCandidateError):
        verify_candidate_directory(copied, identity)


def test_verifier_rejects_symlinked_candidate_directory_ancestor(
    tmp_path: Path,
    candidate_directory: Path,
    identity: ReleaseIdentity,
) -> None:
    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(candidate_directory.parent, target_is_directory=True)

    with pytest.raises(ReleaseCandidateError, match="ancestors"):
        verify_candidate_directory(parent_link / candidate_directory.name, identity)


def test_builder_rejects_symlinked_candidate_directory_ancestor(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    real_nested = real_parent / "nested"
    real_nested.mkdir()
    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ReleaseCandidateError, match="ancestors"):
        build_candidate_directory(
            source_repository,
            identity,
            parent_link / "nested" / "candidates",
        )

    assert list(real_nested.iterdir()) == []


def test_builder_rejects_output_parent_swap_without_redirecting_writes(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "parent"
    output_parent.mkdir()
    displaced = tmp_path / "displaced-parent"
    real_rebuild = candidates._rebuild_dataset

    def swap_parent(*args: object, **kwargs: object) -> candidates.Payload:
        payload = real_rebuild(*args, **kwargs)
        output_parent.rename(displaced)
        output_parent.mkdir()
        return payload

    monkeypatch.setattr(candidates, "_rebuild_dataset", swap_parent)

    with pytest.raises(ReleaseCandidateError, match="parent changed"):
        build_candidate_directory(
            source_repository,
            identity,
            output_parent / "candidates",
        )

    assert list(output_parent.iterdir()) == []
    assert list(displaced.iterdir()) == []


def test_builder_rejects_published_name_swap_without_deleting_replacement(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidates"
    displaced = tmp_path / "displaced-candidates"
    real_rename = candidates._rename_no_replace
    replacements = {
        _dataset_name(identity): b"replacement dataset",
        _report_name(identity): b"replacement report",
    }

    def swap_published_name(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        real_rename(parent_descriptor, source_name, destination_name)
        output.rename(displaced)
        output.mkdir()
        for name, data in replacements.items():
            (output / name).write_bytes(data)

    monkeypatch.setattr(candidates, "_rename_no_replace", swap_published_name)

    with pytest.raises(ReleaseCandidateError, match="changed before verification"):
        build_candidate_directory(source_repository, identity, output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == replacements
    verify_candidate_directory(displaced, identity)


def test_build_rejects_existing_output_without_modifying_it(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ReleaseCandidateError, match="must not exist"):
        build_candidate_directory(source_repository, identity, output)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_rejects_output_inside_release_source(
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    output = source_repository / "new-output-parent/candidates"
    with pytest.raises(ReleaseCandidateError, match="outside"):
        build_candidate_directory(
            source_repository,
            identity,
            output,
        )
    assert not output.parent.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"commit": "0" * 40}, "HEAD does not match"),
        ({"source_date_epoch": 1}, "timestamp does not match"),
        ({"version": "0.2.0"}, "version does not match"),
    ],
)
def test_build_binds_source_git_and_version_identity(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    mismatched = replace(identity, **mutation)

    with pytest.raises(ReleaseCandidateError, match=message):
        build_candidate_directory(source_repository, mismatched, tmp_path / "candidates")


def test_build_rejects_dirty_source_including_untracked_files(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_710_000_000)
    (source / "untracked").write_text("not release input", encoding="utf-8")

    with pytest.raises(ReleaseCandidateError, match="worktree must be clean"):
        build_candidate_directory(source, identity, tmp_path / "candidates")


def test_build_rejects_skip_worktree_modified_normative_blob(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_715_000_000)
    relative = "data/demo_vocabulary.yaml"
    subprocess.run(
        ["git", "update-index", "--skip-worktree", relative],
        cwd=source,
        check=True,
    )
    vocabulary = source / relative
    vocabulary.write_bytes(vocabulary.read_bytes() + b"\n")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    with pytest.raises(ReleaseCandidateError, match="does not match its Git blob"):
        build_candidate_directory(source, identity, tmp_path / "candidates")


def test_build_rejects_ignored_extra_normative_input(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_716_000_000)
    ignored = source / "data/benchmark/v1/ignored-predictions.jsonl"
    ignored.write_text("{}\n", encoding="utf-8")
    exclude = source / ".git/info/exclude"
    exclude.write_text(
        exclude.read_text(encoding="utf-8") + "\ndata/benchmark/v1/ignored-predictions.jsonl\n",
        encoding="utf-8",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    with pytest.raises(ReleaseCandidateError, match="invalid inventory"):
        build_candidate_directory(source, identity, tmp_path / "candidates")


def test_build_rejects_nonregular_git_mode_for_normative_blob(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_716_500_000)
    relative = "data/demo_vocabulary.yaml"
    os.chmod(source / relative, 0o755)
    subprocess.run(["git", "update-index", "--chmod=+x", relative], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.com",
            "commit",
            "-m",
            "make normative input executable",
        ],
        cwd=source,
        check=True,
        capture_output=True,
    )
    executable_identity = _identity_for_repository(source)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    with pytest.raises(ReleaseCandidateError, match="tracked regular Git blob"):
        build_candidate_directory(
            source,
            executable_identity,
            tmp_path / "candidates",
        )


def test_build_uses_commit_bound_snapshot_after_source_read(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_717_000_000)
    output = tmp_path / "candidates"
    real_read = candidates._read_tracked_source_file
    mutated = False

    def mutate_after_read(
        source_root: Path,
        commit: str,
        relative: str,
        *,
        label: str,
    ) -> bytes:
        nonlocal mutated
        data = real_read(source_root, commit, relative, label=label)
        if relative == "data/demo_vocabulary.yaml":
            (source_root / relative).write_bytes(data + b"\n")
            mutated = True
        return data

    monkeypatch.setattr(candidates, "_read_tracked_source_file", mutate_after_read)
    build_candidate_directory(source, identity, output)

    assert mutated
    verify_candidate_directory(output, identity)


def test_build_rejects_committed_symlinked_normative_directory(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_720_000_000)
    benchmark = source / "data/benchmark/v1"
    external = tmp_path / "external-benchmark"
    shutil.copytree(benchmark, external)
    shutil.rmtree(benchmark)
    benchmark.symlink_to(external, target_is_directory=True)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.com",
            "commit",
            "-m",
            "symlink benchmark",
        ],
        cwd=source,
        check=True,
        capture_output=True,
    )
    symlink_identity = _identity_for_repository(source)

    with pytest.raises(ReleaseCandidateError, match="symbolic-link component"):
        build_candidate_directory(source, symlink_identity, tmp_path / "candidates")


def test_build_rejects_hardlinked_normative_input(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_730_000_000)
    specification = source / "data/benchmark/v1/benchmark-spec.json"
    external = tmp_path / "benchmark-spec.json"
    external.write_bytes(specification.read_bytes())
    specification.unlink()
    os.link(external, specification)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    with pytest.raises(ReleaseCandidateError, match="must not be hardlinked"):
        build_candidate_directory(source, identity, tmp_path / "candidates")


def test_builder_accepts_release_identity_loaded_as_a_separate_script_module(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    output = tmp_path / "candidates"
    code = """
import runpy
import sys
from pathlib import Path
from scripts.release.release_candidates import build_candidate_directory, verify_candidate_directory

namespace = runpy.run_path(sys.argv[1])
identity = namespace["ReleaseIdentity"](
    schema_version=sys.argv[4],
    mode="dry-run",
    version="0.1.0",
    commit=sys.argv[5],
    source_date_epoch=int(sys.argv[6]),
    tag=None,
)
build_candidate_directory(Path(sys.argv[2]), identity, Path(sys.argv[3]))
verify_candidate_directory(Path(sys.argv[3]), identity)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT / "scripts/release/release_contract.py"),
            str(source_repository),
            str(output),
            IDENTITY_SCHEMA,
            identity.commit,
            str(identity.source_date_epoch),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_atomic_publication_preserves_concurrently_created_destination(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidates"
    real_rename = candidates._rename_no_replace

    def attack(parent_descriptor: int, source_name: str, destination_name: str) -> None:
        output.mkdir()
        (output / "attacker").write_text("keep", encoding="utf-8")
        real_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(candidates, "_rename_no_replace", attack)

    with pytest.raises(ReleaseCandidateError, match="created concurrently"):
        build_candidate_directory(source_repository, identity, output)

    assert (output / "attacker").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".ste-release-candidates-*"))


def test_build_failure_cleans_private_stage_and_does_not_publish(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidates"

    def fail_report(*args: object, **kwargs: object) -> candidates.Payload:
        raise ReleaseCandidateError("injected report failure")

    monkeypatch.setattr(candidates, "_regenerate_report", fail_report)

    with pytest.raises(ReleaseCandidateError, match="injected report failure"):
        build_candidate_directory(source_repository, identity, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".ste-release-candidates-*"))


def test_build_rejects_tampered_checked_source_and_cleans_output(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    source = _source_copy(source_repository, tmp_path, 1_700_000_000)
    corpus_file = source / "datasets/demonstration-corpus-2/train.jsonl"
    corpus_file.write_bytes(corpus_file.read_bytes() + b"\n")
    output = tmp_path / "candidates"

    with pytest.raises(ReleaseCandidateError, match="worktree must be clean"):
        build_candidate_directory(source, identity, output)

    assert not output.exists()


def test_direct_script_build_and_verify_cli(
    tmp_path: Path,
    identity: ReleaseIdentity,
    source_repository: Path,
) -> None:
    identity_path = tmp_path / "identity.json"
    output = tmp_path / "candidates"
    write_identity(identity, identity_path)
    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/release_candidates.py"),
            "build",
            "--source-root",
            str(source_repository),
            "--identity",
            str(identity_path),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    verify = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/release_candidates.py"),
            "verify",
            "--path",
            str(output),
            "--identity",
            str(identity_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(build.stdout)["status"] == "built"
    assert json.loads(verify.stdout)["status"] == "verified"

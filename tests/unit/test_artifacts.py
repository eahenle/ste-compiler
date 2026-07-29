from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ste_compiler.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactBundleManifestV1,
    ArtifactFileV1,
    ArtifactPreflightResultV1,
    ArtifactVerificationError,
    artifact_manifest_sha256,
    build_artifact_manifest,
    canonical_artifact_manifest_json,
    open_verified_artifact_bundle,
    parse_canonical_artifact_manifest,
    verify_artifact_bundle,
)


def _identity(path: str, data: bytes) -> ArtifactFileV1:
    return ArtifactFileV1(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


def _write_bundle(root: Path, *, architecture: str = "encoder-decoder") -> str:
    files = {
        "run-manifest.json": b'{"architecture":"test"}\n',
        (
            "model.safetensors"
            if architecture == "encoder-decoder"
            else "adapter/adapter_model.safetensors"
        ): b"safe weights",
    }
    for relative_path, data in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    identities = tuple(_identity(path, data) for path, data in files.items())
    if architecture == "encoder-decoder":
        manifest = build_artifact_manifest(
            architecture="encoder-decoder",
            artifact_type="encoder-decoder-checkpoint",
            entrypoint=".",
            files=identities,
        )
    else:
        manifest = build_artifact_manifest(
            architecture="decoder-only-lora",
            artifact_type="decoder-only-lora-run",
            entrypoint="adapter",
            files=identities,
        )
    (root / ARTIFACT_MANIFEST_NAME).write_bytes(canonical_artifact_manifest_json(manifest))
    return artifact_manifest_sha256(manifest)


def test_manifest_construction_is_canonical_and_stable():
    run = _identity("run-manifest.json", b"run")
    weights = _identity("model.safetensors", b"weights")
    manifest = build_artifact_manifest(
        architecture="encoder-decoder",
        artifact_type="encoder-decoder-checkpoint",
        entrypoint=".",
        files=(weights, run),
    )
    canonical = canonical_artifact_manifest_json(manifest)

    assert tuple(file.path for file in manifest.files) == (
        "model.safetensors",
        "run-manifest.json",
    )
    assert manifest.file_count == 2
    assert manifest.total_bytes == len(b"runweights")
    assert canonical.endswith(b"\n")
    assert parse_canonical_artifact_manifest(canonical) == manifest
    assert artifact_manifest_sha256(manifest) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute",
        "../escape",
        "nested/../escape",
        "nested//empty",
        "windows\\path",
        "drive:path",
        "white space",
        "unicodé.json",
    ],
)
def test_file_identity_rejects_unsafe_or_nonportable_paths(path):
    with pytest.raises(ValidationError, match="path"):
        ArtifactFileV1(path=path, sha256="a" * 64, bytes=0)


def test_manifest_rejects_duplicates_case_collisions_and_prefix_conflicts():
    run = _identity("run-manifest.json", b"run")
    with pytest.raises(ValidationError, match="unique"):
        ArtifactBundleManifestV1(
            schema_version="ste-artifact-bundle-v1",
            architecture="encoder-decoder",
            artifact_type="encoder-decoder-checkpoint",
            intended_use="mechanics-smoke",
            entrypoint=".",
            run_manifest_sha256=run.sha256,
            file_count=3,
            total_bytes=6,
            files=(_identity("A.json", b"a"), _identity("a.json", b"b"), run),
        )
    with pytest.raises(ValidationError, match="file and a directory"):
        build_artifact_manifest(
            architecture="encoder-decoder",
            artifact_type="encoder-decoder-checkpoint",
            entrypoint=".",
            files=(run, _identity("tokenizer", b"a"), _identity("tokenizer/vocab.json", b"b")),
        )


def test_manifest_rejects_wrong_profile_counts_and_run_digest():
    run = _identity("run-manifest.json", b"run")
    payload = {
        "schema_version": "ste-artifact-bundle-v1",
        "architecture": "encoder-decoder",
        "artifact_type": "decoder-only-lora-run",
        "intended_use": "mechanics-smoke",
        "entrypoint": ".",
        "run_manifest_sha256": "b" * 64,
        "file_count": 2,
        "total_bytes": run.bytes,
        "files": [run.model_dump(mode="json")],
    }

    with pytest.raises(ValidationError):
        ArtifactBundleManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"artifact_type": "decoder-only-lora-run"},
        {"validation_profile": "decoder-adapter-structure-v1"},
    ],
)
def test_preflight_result_rejects_profile_fields_for_another_architecture(update):
    payload = {
        "schema_version": "ste-artifact-preflight-v1",
        "status": "verified",
        "architecture": "encoder-decoder",
        "artifact_type": "encoder-decoder-checkpoint",
        "intended_use": "mechanics-smoke",
        "artifact_manifest_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "file_count": 1,
        "total_bytes": 1,
        "validation_profile": "encoder-checkpoint-load-v1",
        "network_access": "none",
    }
    payload.update(update)

    with pytest.raises(ValidationError, match="do not match the architecture"):
        ArtifactPreflightResultV1.model_validate(payload)


def test_parser_rejects_noncanonical_and_oversized_json():
    run = _identity("run-manifest.json", b"run")
    manifest = build_artifact_manifest(
        architecture="encoder-decoder",
        artifact_type="encoder-decoder-checkpoint",
        entrypoint=".",
        files=(run,),
    )
    noncanonical = json.dumps(manifest.model_dump(mode="json"), indent=2).encode()

    with pytest.raises(ArtifactVerificationError, match="not canonical"):
        parse_canonical_artifact_manifest(noncanonical)


@pytest.mark.parametrize("architecture", ["encoder-decoder", "decoder-only-lora"])
def test_verifier_yields_private_exact_materialization_and_cleans_it(tmp_path, architecture):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source, architecture=architecture)

    with open_verified_artifact_bundle(source, digest) as verified:
        materialized = verified.path
        assert materialized != source
        assert verified.manifest.architecture == architecture
        assert verified.manifest_sha256 == digest
        assert {
            path.relative_to(materialized).as_posix()
            for path in materialized.rglob("*")
            if path.is_file()
        } == {
            ARTIFACT_MANIFEST_NAME,
            "run-manifest.json",
            (
                "model.safetensors"
                if architecture == "encoder-decoder"
                else "adapter/adapter_model.safetensors"
            ),
        }
        assert (materialized / "run-manifest.json").read_bytes() == (
            source / "run-manifest.json"
        ).read_bytes()
        (source / "run-manifest.json").write_bytes(b"source changed after capture")
        assert (materialized / "run-manifest.json").read_bytes() == b'{"architecture":"test"}\n'
    assert not materialized.exists()


def test_convenience_verifier_returns_the_bound_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)

    manifest = verify_artifact_bundle(source, digest)

    assert manifest.architecture == "encoder-decoder"
    assert artifact_manifest_sha256(manifest) == digest


def test_verifier_requires_external_digest_and_canonical_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)

    with (
        pytest.raises(ArtifactVerificationError, match="64 lowercase"),
        open_verified_artifact_bundle(source, "invalid"),
    ):
        pass
    with (
        pytest.raises(ArtifactVerificationError, match="does not match"),
        open_verified_artifact_bundle(source, "0" * 64),
    ):
        pass

    manifest = json.loads((source / ARTIFACT_MANIFEST_NAME).read_text())
    (source / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    altered_digest = hashlib.sha256((source / ARTIFACT_MANIFEST_NAME).read_bytes()).hexdigest()
    assert altered_digest != digest
    with (
        pytest.raises(ArtifactVerificationError, match="not canonical"),
        open_verified_artifact_bundle(source, altered_digest),
    ):
        pass


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_verifier_rejects_linked_manifest(tmp_path, link_kind):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)
    manifest = source / ARTIFACT_MANIFEST_NAME
    target = tmp_path / "manifest"
    manifest.replace(target)
    if link_kind == "symlink":
        manifest.symlink_to(target)
    else:
        os.link(target, manifest)

    with pytest.raises(ArtifactVerificationError), open_verified_artifact_bundle(source, digest):
        pass


@pytest.mark.parametrize("change", ["tamper", "missing", "extra", "extra-directory"])
def test_verifier_requires_the_exact_manifest_tree(tmp_path, change):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)
    if change == "tamper":
        (source / "model.safetensors").write_bytes(b"evil weights")
    elif change == "missing":
        (source / "model.safetensors").unlink()
    elif change == "extra":
        (source / "extra.json").write_text("{}")
    else:
        (source / "empty").mkdir()

    with pytest.raises(ArtifactVerificationError), open_verified_artifact_bundle(source, digest):
        pass


def test_verifier_stops_enumerating_after_the_first_excess_entry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)
    real_scandir = os.scandir
    intercepted = False
    injected_entries = 0

    class InjectedDirectoryEntries:
        def __init__(self, entries):
            self.entries = entries

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.entries.close()

        def __iter__(self):
            nonlocal injected_entries
            yield from self.entries
            injected_entries += 1
            yield SimpleNamespace(name="unexpected-0")
            raise AssertionError("verifier continued enumerating unexpected entries")

    def inject_excess_entries(path):
        nonlocal intercepted
        entries = real_scandir(path)
        if isinstance(path, int) and not intercepted:
            intercepted = True
            return InjectedDirectoryEntries(entries)
        return entries

    monkeypatch.setattr(os, "scandir", inject_excess_entries)

    with (
        pytest.raises(ArtifactVerificationError, match="does not match the manifest file set"),
        open_verified_artifact_bundle(source, digest),
    ):
        pass

    assert intercepted
    assert injected_entries == 1


def test_verifier_rejects_symlinked_root_file_and_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source, architecture="decoder-only-lora")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(source, target_is_directory=True)

    with (
        pytest.raises(ArtifactVerificationError, match="real directory"),
        open_verified_artifact_bundle(linked_root, digest),
    ):
        pass

    weights = source / "adapter/adapter_model.safetensors"
    target = tmp_path / "weights"
    weights.replace(target)
    weights.symlink_to(target)
    with pytest.raises(ArtifactVerificationError), open_verified_artifact_bundle(source, digest):
        pass

    weights.unlink()
    weights.write_bytes(target.read_bytes())
    adapter = source / "adapter"
    real_adapter = tmp_path / "real-adapter"
    adapter.replace(real_adapter)
    adapter.symlink_to(real_adapter, target_is_directory=True)
    with pytest.raises(ArtifactVerificationError), open_verified_artifact_bundle(source, digest):
        pass


def test_verifier_rejects_hard_links_and_fifo_without_blocking(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    digest = _write_bundle(source)
    weights = source / "model.safetensors"
    hardlink = tmp_path / "hardlink"
    os.link(weights, hardlink)

    with (
        pytest.raises(ArtifactVerificationError, match="single-link"),
        open_verified_artifact_bundle(source, digest),
    ):
        pass

    hardlink.unlink()
    weights.unlink()
    os.mkfifo(weights)
    with (
        pytest.raises(ArtifactVerificationError, match="regular file"),
        open_verified_artifact_bundle(source, digest),
    ):
        pass

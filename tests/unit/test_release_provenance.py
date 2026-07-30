from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.distribution_smoke import _publish_verified_distributions
from scripts.release.release_contract import (
    IDENTITY_SCHEMA,
    MANIFEST_SCHEMA,
    ReleaseContractError,
    ReleaseIdentity,
    finalize_release,
    validate_release_ref,
    verify_release_bundle,
    write_identity,
)

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github/workflows/release-provenance.yml"
ATTESTATION_WORKFLOW = ROOT / ".github/workflows/release-attestation.yml"
TRUSTED_SIGNERS = ROOT / ".github/release/trusted-tag-signers"
ACTION_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _run(root: Path, *command: str) -> str:
    return subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _run(root, "git", "init", "-b", "main")
    _run(root, "git", "config", "user.name", "Release Test")
    _run(root, "git", "config", "user.email", "release@example.com")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "ste-compiler"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: ste-compiler\nversion: "0.1.0"\n',
        encoding="utf-8",
    )
    _run(root, "git", "add", ".")
    _run(root, "git", "commit", "-m", "release source")
    return root


def _release_bundle(
    tmp_path: Path,
    *,
    mode: str = "tag",
    commit: str = "a" * 40,
) -> Path:
    release = tmp_path / "release"
    distributions = release / "distributions"
    distributions.mkdir(parents=True)
    (distributions / "ste_compiler-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (distributions / "ste_compiler-0.1.0.tar.gz").write_bytes(b"sdist")
    (release / "ste-compiler.spdx.json").write_text(
        '{"SPDXID":"SPDXRef-DOCUMENT","spdxVersion":"SPDX-2.3"}\n',
        encoding="utf-8",
    )
    identity_path = tmp_path / "identity.json"
    identity_mode = "tag" if mode == "tag" else "dry-run"
    write_identity(
        ReleaseIdentity(
            schema_version=IDENTITY_SCHEMA,
            mode=identity_mode,
            version="0.1.0",
            commit=commit,
            source_date_epoch=1729,
            tag="v0.1.0" if identity_mode == "tag" else None,
        ),
        identity_path,
    )
    finalize_release(release, identity_path)
    return release


def _manifest(release: Path) -> dict[str, Any]:
    payload = json.loads((release / "release-build.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(release: Path, payload: dict[str, Any]) -> None:
    (release / "release-build.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_manual_dry_run_binds_clean_head_and_version_without_tag(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    commit = _run(root, "git", "rev-parse", "HEAD")

    identity = validate_release_ref(
        root,
        mode="dry-run",
        commit=commit,
        tag=None,
        allowed_signers=tmp_path / "not-required-for-dry-run",
    )

    assert identity.schema_version == IDENTITY_SCHEMA
    assert identity.mode == "dry-run"
    assert identity.version == "0.1.0"
    assert identity.commit == commit
    assert identity.tag is None
    assert identity.source_date_epoch > 0


def test_release_ref_rejects_dirty_source_and_version_mismatch(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    commit = _run(root, "git", "rev-parse", "HEAD")
    (root / "untracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="checkout must be clean"):
        validate_release_ref(
            root,
            mode="dry-run",
            commit=commit,
            tag=None,
            allowed_signers=tmp_path / "unused",
        )

    (root / "untracked").unlink()
    (root / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: ste-compiler\nversion: "0.2.0"\n',
        encoding="utf-8",
    )
    _run(root, "git", "add", "CITATION.cff")
    _run(root, "git", "commit", "-m", "mismatched citation")
    with pytest.raises(ReleaseContractError, match="does not equal project version"):
        validate_release_ref(
            root,
            mode="dry-run",
            commit=_run(root, "git", "rev-parse", "HEAD"),
            tag=None,
            allowed_signers=tmp_path / "unused",
        )


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen is required")
def test_signed_tag_requires_annotated_allowed_ssh_signature(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    private_key = tmp_path / "release-signing-key"
    _run(
        tmp_path,
        "ssh-keygen",
        "-q",
        "-t",
        "ed25519",
        "-N",
        "",
        "-C",
        "release@example.com",
        "-f",
        str(private_key),
    )
    allowed_signers = root / "trusted-tag-signers"
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers.write_text(
        f'release@example.com namespaces="git" {public_key}\n',
        encoding="utf-8",
    )
    _run(root, "git", "add", "trusted-tag-signers")
    _run(root, "git", "commit", "-m", "authorize release signer")
    commit = _run(root, "git", "rev-parse", "HEAD")
    _run(root, "git", "config", "gpg.format", "ssh")
    _run(root, "git", "config", "user.signingkey", str(private_key))
    _run(root, "git", "tag", "-s", "v0.1.0", "-m", "v0.1.0")

    identity = validate_release_ref(
        root,
        mode="tag",
        commit=commit,
        tag="v0.1.0",
        allowed_signers=allowed_signers,
    )
    assert identity.mode == "tag"
    assert identity.tag == "v0.1.0"

    unauthorized_key = tmp_path / "unauthorized-key"
    _run(
        tmp_path,
        "ssh-keygen",
        "-q",
        "-t",
        "ed25519",
        "-N",
        "",
        "-C",
        "other@example.com",
        "-f",
        str(unauthorized_key),
    )
    unauthorized = tmp_path / "unauthorized-signers"
    unauthorized.write_text(
        (
            'other@example.com namespaces="git" '
            f"{unauthorized_key.with_suffix('.pub').read_text(encoding='utf-8').strip()}\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReleaseContractError, match="command failed"):
        validate_release_ref(
            root,
            mode="tag",
            commit=commit,
            tag="v0.1.0",
            allowed_signers=unauthorized,
        )


def test_tag_mode_rejects_lightweight_tag_and_disabled_signer_policy(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    commit = _run(root, "git", "rev-parse", "HEAD")
    _run(root, "git", "tag", "v0.1.0")
    signers = tmp_path / "signers"
    signers.write_text("# intentionally disabled\n", encoding="utf-8")

    with pytest.raises(ReleaseContractError, match="must be annotated"):
        validate_release_ref(
            root,
            mode="tag",
            commit=commit,
            tag="v0.1.0",
            allowed_signers=signers,
        )

    _run(root, "git", "tag", "-d", "v0.1.0")
    _run(root, "git", "tag", "-a", "v0.1.0", "-m", "unsigned annotated tag")
    with pytest.raises(ReleaseContractError, match="disabled until trusted-tag-signers"):
        validate_release_ref(
            root,
            mode="tag",
            commit=commit,
            tag="v0.1.0",
            allowed_signers=signers,
        )


def test_finalize_release_writes_canonical_inventory_and_checksums(tmp_path: Path) -> None:
    release = tmp_path / "release"
    distributions = release / "distributions"
    distributions.mkdir(parents=True)
    (distributions / "ste_compiler-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (distributions / "ste_compiler-0.1.0.tar.gz").write_bytes(b"sdist")
    (release / "ste-compiler.spdx.json").write_text(
        '{"SPDXID":"SPDXRef-DOCUMENT","spdxVersion":"SPDX-2.3"}\n'
    )
    identity_path = tmp_path / "identity.json"
    write_identity(
        ReleaseIdentity(
            schema_version=IDENTITY_SCHEMA,
            mode="dry-run",
            version="0.1.0",
            commit="a" * 40,
            source_date_epoch=1729,
            tag=None,
        ),
        identity_path,
    )

    manifest_path, checksums_path = finalize_release(release, identity_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["identity_schema_version"] == IDENTITY_SCHEMA
    assert manifest["mode"] == "dry-run"
    assert [artifact["path"] for artifact in manifest["artifacts"]] == [
        "distributions/ste_compiler-0.1.0-py3-none-any.whl",
        "distributions/ste_compiler-0.1.0.tar.gz",
        "ste-compiler.spdx.json",
    ]
    checksum_paths = [line.split("  ", 1)[1] for line in checksums_path.read_text().splitlines()]
    assert checksum_paths == [
        "distributions/ste_compiler-0.1.0-py3-none-any.whl",
        "distributions/ste_compiler-0.1.0.tar.gz",
        "release-build.json",
        "ste-compiler.spdx.json",
    ]
    with pytest.raises(ReleaseContractError, match="must not already exist"):
        finalize_release(release, identity_path)


def test_verified_distribution_copy_requires_new_output(tmp_path: Path) -> None:
    wheel = tmp_path / "package.whl"
    sdist = tmp_path / "package.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    output = tmp_path / "verified"

    copied = _publish_verified_distributions((wheel, sdist), output)
    assert tuple(path.read_bytes() for path in copied) == (b"wheel", b"sdist")
    with pytest.raises(RuntimeError, match="must not already exist"):
        _publish_verified_distributions((wheel, sdist), output)


def test_finalize_release_rejects_symbolic_link_entries(tmp_path: Path) -> None:
    release = tmp_path / "release"
    distributions = release / "distributions"
    distributions.mkdir(parents=True)
    (distributions / "ste_compiler-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (distributions / "ste_compiler-0.1.0.tar.gz").write_bytes(b"sdist")
    (release / "ste-compiler.spdx.json").write_text(
        '{"SPDXID":"SPDXRef-DOCUMENT","spdxVersion":"SPDX-2.3"}\n'
    )
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    (release / "linked").symlink_to(target)
    identity_path = tmp_path / "identity.json"
    write_identity(
        ReleaseIdentity(
            schema_version=IDENTITY_SCHEMA,
            mode="dry-run",
            version="0.1.0",
            commit="a" * 40,
            source_date_epoch=1729,
            tag=None,
        ),
        identity_path,
    )

    with pytest.raises(ReleaseContractError, match="must not contain symbolic links"):
        finalize_release(release, identity_path)


def test_finalize_release_rejects_symbolic_link_root(tmp_path: Path) -> None:
    release = tmp_path / "release"
    distributions = release / "distributions"
    distributions.mkdir(parents=True)
    (distributions / "ste_compiler-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (distributions / "ste_compiler-0.1.0.tar.gz").write_bytes(b"sdist")
    (release / "ste-compiler.spdx.json").write_text(
        '{"SPDXID":"SPDXRef-DOCUMENT","spdxVersion":"SPDX-2.3"}\n'
    )
    linked_release = tmp_path / "linked-release"
    linked_release.symlink_to(release, target_is_directory=True)
    identity_path = tmp_path / "identity.json"
    write_identity(
        ReleaseIdentity(
            schema_version=IDENTITY_SCHEMA,
            mode="dry-run",
            version="0.1.0",
            commit="a" * 40,
            source_date_epoch=1729,
            tag=None,
        ),
        identity_path,
    )

    with pytest.raises(ReleaseContractError, match="release root must be a real"):
        finalize_release(linked_release, identity_path)


def test_finalize_release_rejects_distribution_version_mismatch(tmp_path: Path) -> None:
    release = tmp_path / "release"
    distributions = release / "distributions"
    distributions.mkdir(parents=True)
    (distributions / "ste_compiler-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (distributions / "ste_compiler-0.1.0.tar.gz").write_bytes(b"sdist")
    (release / "ste-compiler.spdx.json").write_text(
        '{"SPDXID":"SPDXRef-DOCUMENT","spdxVersion":"SPDX-2.3"}\n'
    )
    identity_path = tmp_path / "identity.json"
    write_identity(
        ReleaseIdentity(
            schema_version=IDENTITY_SCHEMA,
            mode="dry-run",
            version="0.2.0",
            commit="a" * 40,
            source_date_epoch=1729,
            tag=None,
        ),
        identity_path,
    )

    with pytest.raises(ReleaseContractError, match="filenames do not match"):
        finalize_release(release, identity_path)


def test_verify_release_bundle_returns_identity_and_prints_safe_json(tmp_path: Path) -> None:
    release = _release_bundle(tmp_path)

    identity = verify_release_bundle(
        release,
        expected_commit="a" * 40,
        expected_mode="tag",
    )

    assert identity == ReleaseIdentity(
        schema_version=IDENTITY_SCHEMA,
        mode="tag",
        version="0.1.0",
        commit="a" * 40,
        source_date_epoch=1729,
        tag="v0.1.0",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/release_contract.py"),
            "verify-bundle",
            "--release-root",
            str(release),
            "--expected-commit",
            "a" * 40,
            "--expected-mode",
            "tag",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == identity.as_dict()
    assert completed.stderr == ""


def test_verify_release_bundle_accepts_only_the_expected_mode_and_commit(
    tmp_path: Path,
) -> None:
    release = _release_bundle(tmp_path, mode="dry-run")

    identity = verify_release_bundle(
        release,
        expected_commit="a" * 40,
        expected_mode="dry-run",
    )

    assert identity.mode == "dry-run"
    assert identity.tag is None
    with pytest.raises(ReleaseContractError, match="does not equal expected mode"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )
    with pytest.raises(ReleaseContractError, match="does not equal expected commit"):
        verify_release_bundle(
            release,
            expected_commit="b" * 40,
            expected_mode="dry-run",
        )
    with pytest.raises(ReleaseContractError, match="full lowercase Git SHA-1"):
        verify_release_bundle(
            release,
            expected_commit="A" * 40,
            expected_mode="dry-run",
        )


def test_verify_release_bundle_rejects_symlinks_and_unexpected_inventory(
    tmp_path: Path,
) -> None:
    release = _release_bundle(tmp_path / "linked-entry")
    sbom = release / "ste-compiler.spdx.json"
    sbom.unlink()
    sbom.symlink_to(tmp_path / "outside")
    with pytest.raises(ReleaseContractError, match="must not contain symbolic links"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )

    release = _release_bundle(tmp_path / "extra-entry")
    (release / "unexpected").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="unexpected file inventory"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )

    release = _release_bundle(tmp_path / "linked-root")
    linked_root = tmp_path / "bundle-link"
    linked_root.symlink_to(release, target_is_directory=True)
    with pytest.raises(ReleaseContractError, match="root must be a real"):
        verify_release_bundle(
            linked_root,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("identity_schema_version", "unexpected", "identity schema"),
        ("schema_version", "unexpected", "manifest schema"),
        ("mode", "unexpected", "mode must be"),
        ("mode", "dry-run", "only tag-mode"),
        ("source_date_epoch", True, "invalid primitive"),
        ("tag", None, "only tag-mode"),
        ("tag", "v0.2.0", "tag and version"),
    ],
)
def test_verify_release_bundle_strictly_parses_manifest_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    release = _release_bundle(tmp_path)
    payload = _manifest(release)
    payload[field] = value
    _write_manifest(release, payload)

    with pytest.raises(ReleaseContractError, match=message):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


def test_verify_release_bundle_rejects_duplicate_and_unexpected_manifest_fields(
    tmp_path: Path,
) -> None:
    release = _release_bundle(tmp_path / "duplicate")
    manifest_path = release / "release-build.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            '  "mode": "tag",\n',
            '  "mode": "tag",\n  "mode": "tag",\n',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReleaseContractError, match="duplicate field"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )

    release = _release_bundle(tmp_path / "unexpected")
    payload = _manifest(release)
    payload["untrusted"] = "value"
    _write_manifest(release, payload)
    with pytest.raises(ReleaseContractError, match="unexpected field inventory"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "../outside", "canonical and relative"),
        (
            "path",
            "distributions/ste_compiler-0.1.0-aaa.whl",
            "does not match the release bundle",
        ),
        ("bytes", True, "nonnegative integer"),
        ("sha256", "A" * 64, "lowercase hexadecimal"),
    ],
)
def test_verify_release_bundle_rejects_invalid_artifact_records(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    release = _release_bundle(tmp_path)
    payload = _manifest(release)
    payload["artifacts"][0][field] = value
    _write_manifest(release, payload)

    with pytest.raises(ReleaseContractError, match=message):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


def test_verify_release_bundle_checks_artifact_size_and_sha256(tmp_path: Path) -> None:
    release = _release_bundle(tmp_path / "size")
    wheel = release / "distributions/ste_compiler-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"changed-wheel")
    with pytest.raises(ReleaseContractError, match="size does not match"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )

    release = _release_bundle(tmp_path / "digest")
    wheel = release / "distributions/ste_compiler-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"same!")
    with pytest.raises(ReleaseContractError, match="SHA-256 does not match"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


def test_verify_release_bundle_requires_canonical_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    release = _release_bundle(tmp_path / "manifest")
    manifest_path = release / "release-build.json"
    payload = _manifest(release)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="not canonical JSON"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )

    release = _release_bundle(tmp_path / "checksums")
    checksums = release / "SHA256SUMS"
    checksums.write_text(
        checksums.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReleaseContractError, match="not the canonical"):
        verify_release_bundle(
            release,
            expected_commit="a" * 40,
            expected_mode="tag",
        )


def test_release_workflow_is_immutable_least_privilege_and_nonpublishing() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert workflow["on"] == {
        "push": {"tags": ["v*"]},
        "workflow_dispatch": "",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert set(workflow["jobs"]) == {"build"}
    checkout = workflow["jobs"]["build"]["steps"][0]
    assert checkout["with"]["persist-credentials"] == "false"
    identity_gate = workflow["jobs"]["build"]["steps"][3]["run"]
    assert "git merge-base --is-ancestor" in identity_gate
    assert "refs/remotes/origin/${default_branch}" in identity_gate
    assert "git show" in identity_gate
    assert '"${policy_ref}:.github/release/trusted-tag-signers"' in identity_gate
    assert '--allowed-signers "${allowed_signers}"' in identity_gate
    uses = [
        step["uses"] for job in workflow["jobs"].values() for step in job["steps"] if "uses" in step
    ]
    assert uses
    assert all(ACTION_PIN.fullmatch(action) for action in uses)
    raw = WORKFLOW.read_text(encoding="utf-8").casefold()
    assert "id-token: write" not in raw
    assert "attestations: write" not in raw
    assert "pypi" not in raw
    assert "contents: write" not in raw
    assert "packages: write" not in raw
    assert "gh release" not in raw
    assert "ste-compiler-release-${{ github.sha }}-${{ github.run_attempt }}" in raw

    attestation = yaml.load(
        ATTESTATION_WORKFLOW.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert attestation["on"] == {
        "workflow_run": {
            "workflows": ["Release provenance"],
            "types": ["completed"],
        }
    }
    assert attestation["permissions"] == {"contents": "read"}
    verify_job = attestation["jobs"]["verify"]
    assert "workflow_run.conclusion == 'success'" in verify_job["if"]
    assert "workflow_run.head_repository.full_name == github.repository" in verify_job["if"]
    assert "workflow_run.event == 'workflow_dispatch'" in verify_job["if"]
    assert verify_job["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert "id-token" not in verify_job["permissions"]
    attest_job = attestation["jobs"]["attest"]
    assert "workflow_run.event == 'push'" in attest_job["if"]
    assert "needs.verify.outputs.mode == 'tag'" in attest_job["if"]
    assert attest_job["permissions"] == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    trusted_checkout, source_checkout = verify_job["steps"][:2]
    assert trusted_checkout["with"]["persist-credentials"] == "false"
    assert source_checkout["with"]["persist-credentials"] == "false"
    assert source_checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"
    assert source_checkout["with"]["path"] == "release-source"
    attestation_uses = [
        step["uses"]
        for job in attestation["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert attestation_uses
    assert all(ACTION_PIN.fullmatch(action) for action in attestation_uses)
    attestation_raw = ATTESTATION_WORKFLOW.read_text(encoding="utf-8")
    assert "run-id: ${{ github.event.workflow_run.id }}" in attestation_raw
    assert "github.event.workflow_run.run_attempt" in attestation_raw
    assert attestation_raw.count("github.run_attempt") == 2
    assert "verify-bundle" in attestation_raw
    assert "--source-root release-source" in attestation_raw
    assert "cmp \\" in attestation_raw
    assert "enable-cache: false" in attestation_raw
    assert "git merge-base --is-ancestor" in attestation_raw
    assert "--allowed-signers .github/release/trusted-tag-signers" in attestation_raw
    assert [step["name"] for step in attest_job["steps"]] == [
        "Download trusted verification bundle",
        "Attest build provenance",
        "Attest SPDX SBOM",
    ]


def test_repository_signer_policy_is_explicitly_closed_pending_authorization() -> None:
    configured = [
        line
        for line in TRUSTED_SIGNERS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert configured == []

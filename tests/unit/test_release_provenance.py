from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

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
    write_identity,
)

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github/workflows/release-provenance.yml"
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


def test_release_workflow_is_immutable_least_privilege_and_nonpublishing() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert workflow["on"] == {
        "push": {"tags": ["v*"]},
        "workflow_dispatch": "",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert workflow["jobs"]["attest"]["if"] == "needs.build.outputs.mode == 'tag'"
    assert workflow["jobs"]["attest"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    checkout = workflow["jobs"]["build"]["steps"][0]
    assert checkout["with"]["persist-credentials"] == "false"
    uses = [
        step["uses"] for job in workflow["jobs"].values() for step in job["steps"] if "uses" in step
    ]
    assert uses
    assert all(ACTION_PIN.fullmatch(action) for action in uses)
    raw = WORKFLOW.read_text(encoding="utf-8").casefold()
    assert "pypi" not in raw
    assert "contents: write" not in raw
    assert "packages: write" not in raw
    assert "gh release" not in raw


def test_repository_signer_policy_is_explicitly_closed_pending_authorization() -> None:
    configured = [
        line
        for line in TRUSTED_SIGNERS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert configured == []

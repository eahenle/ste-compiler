from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/ci/check_dependency_policy.py"
POLICY = ROOT / "policy/dependency-audit-policy.json"
AS_OF = "2026-07-29"


def _policy() -> dict[str, object]:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _vulnerability(vulnerability_id: str = "PYSEC-2099-1") -> dict[str, object]:
    return {
        "id": vulnerability_id,
        "fix_versions": ["2.0"],
        "aliases": ["CVE-2099-0001"],
        "description": "Synthetic offline vulnerability fixture.",
    }


def _audit_report(
    *packages: tuple[str, str, list[dict[str, object]]],
) -> dict[str, object]:
    return {
        "dependencies": [
            {"name": name, "version": version, "vulns": vulnerabilities}
            for name, version, vulnerabilities in packages
        ],
        "fixes": [],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_vulnerabilities(
    tmp_path: Path,
    report: object,
    *,
    policy: object | None = None,
    scanner_exit_code: int = 0,
    profile: str = "core",
) -> subprocess.CompletedProcess[str]:
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "audit.json"
    _write_json(policy_path, _policy() if policy is None else policy)
    _write_json(report_path, report)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "vulnerabilities",
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
            "--scanner-exit-code",
            str(scanner_exit_code),
            "--profile",
            profile,
            "--as-of",
            AS_OF,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_licenses(
    tmp_path: Path,
    report: object,
    expected: object,
    *,
    policy: object | None = None,
) -> subprocess.CompletedProcess[str]:
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "licenses.json"
    expected_path = tmp_path / "audit.json"
    _write_json(policy_path, _policy() if policy is None else policy)
    _write_json(report_path, report)
    _write_json(expected_path, expected)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "licenses",
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
            "--expected-audit-report",
            str(expected_path),
            "--profile",
            "all",
            "--as-of",
            AS_OF,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_clean_vulnerability_report_passes(tmp_path: Path) -> None:
    result = _run_vulnerabilities(
        tmp_path,
        _audit_report(("example", "1.0", [])),
    )

    assert result.returncode == 0
    assert result.stdout == (
        "dependency vulnerability policy passed for core: 1 packages; 0 suppressed findings\n"
    )
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("report", "scanner_exit_code", "message"),
    [
        (
            _audit_report(("example", "1.0", [_vulnerability()])),
            0,
            "scanner exit code does not match report findings",
        ),
        (
            _audit_report(("example", "1.0", [])),
            1,
            "scanner exit code does not match report findings",
        ),
        (
            {"dependencies": [], "fixes": []},
            0,
            "audit report must contain at least one dependency",
        ),
        (
            {
                "dependencies": [
                    {"name": "example", "version": "1.0", "vulns": []},
                    {"name": "Example", "version": "1.0", "vulns": []},
                ],
                "fixes": [],
            },
            0,
            "audit report contains duplicate package",
        ),
        (
            {
                "dependencies": [{"name": "example", "version": "1.0", "vulns": []}],
                "fixes": [{"name": "example"}],
            },
            0,
            "audit report.fixes must be empty",
        ),
    ],
)
def test_vulnerability_report_fails_closed(
    tmp_path: Path,
    report: object,
    scanner_exit_code: int,
    message: str,
) -> None:
    result = _run_vulnerabilities(
        tmp_path,
        report,
        scanner_exit_code=scanner_exit_code,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_unsuppressed_vulnerability_fails(tmp_path: Path) -> None:
    result = _run_vulnerabilities(
        tmp_path,
        _audit_report(("example", "1.0", [_vulnerability()])),
        scanner_exit_code=1,
    )

    assert result.returncode == 1
    assert "unsuppressed vulnerability: example==1.0 PYSEC-2099-1" in result.stderr


def test_exact_unexpired_vulnerability_suppression_passes(tmp_path: Path) -> None:
    policy = _policy()
    policy["vulnerability_suppressions"] = [
        {
            "package": "example",
            "version": "1.0",
            "vulnerability_id": "PYSEC-2099-1",
            "reason": "Synthetic acceptance fixture.",
            "expires": "2026-07-30",
        }
    ]

    result = _run_vulnerabilities(
        tmp_path,
        _audit_report(
            (
                "example",
                "1.0",
                [_vulnerability(), _vulnerability()],
            )
        ),
        policy=policy,
        scanner_exit_code=1,
    )

    assert result.returncode == 0
    assert "2 suppressed findings" in result.stdout


@pytest.mark.parametrize(
    ("expires", "report", "message"),
    [
        (
            "2026-07-28",
            _audit_report(("example", "1.0", [_vulnerability()])),
            "suppression expired",
        ),
        (
            "2026-07-30",
            _audit_report(("example", "1.0", [])),
            "unused vulnerability suppression",
        ),
    ],
)
def test_vulnerability_suppressions_expire_and_cannot_go_stale(
    tmp_path: Path,
    expires: str,
    report: object,
    message: str,
) -> None:
    policy = _policy()
    policy["vulnerability_suppressions"] = [
        {
            "package": "example",
            "version": "1.0",
            "vulnerability_id": "PYSEC-2099-1",
            "reason": "Synthetic acceptance fixture.",
            "expires": expires,
        }
    ]

    result = _run_vulnerabilities(
        tmp_path,
        report,
        policy=policy,
        scanner_exit_code=1 if "expired" in message else 0,
    )

    assert result.returncode == 1
    assert message in result.stderr


def test_profile_specific_suppression_is_enforced_only_where_package_is_present(
    tmp_path: Path,
) -> None:
    policy = _policy()
    policy["vulnerability_suppressions"] = [
        {
            "package": "neural-package",
            "version": "1.0",
            "vulnerability_id": "PYSEC-2099-1",
            "reason": "Synthetic profile-scoping fixture.",
            "expires": "2026-07-30",
        }
    ]

    core = _run_vulnerabilities(
        tmp_path,
        _audit_report(("core-package", "1.0", [])),
        policy=policy,
        profile="core",
    )
    all_without_neural = _run_vulnerabilities(
        tmp_path,
        _audit_report(("core-package", "1.0", [])),
        policy=policy,
        profile="all",
    )
    all_with_neural = _run_vulnerabilities(
        tmp_path,
        _audit_report(
            ("core-package", "1.0", []),
            ("neural-package", "1.0", [_vulnerability()]),
        ),
        policy=policy,
        scanner_exit_code=1,
        profile="all",
    )

    assert core.returncode == 0
    assert all_without_neural.returncode == 1
    assert "unused vulnerability suppression" in all_without_neural.stderr
    assert all_with_neural.returncode == 0


def test_reviewed_complete_license_inventory_passes(tmp_path: Path) -> None:
    expected = _audit_report(
        ("example-one", "1.0", []),
        ("example-two", "2.0", []),
    )
    report = [
        {"Name": "Example_One", "Version": "1.0", "License": "MIT"},
        {"Name": "example-two", "Version": "2.0", "License": "Apache-2.0"},
    ]

    result = _run_licenses(tmp_path, report, expected)

    assert result.returncode == 0
    assert result.stdout == ("dependency license policy passed for all: 2 packages; 0 exceptions\n")


@pytest.mark.parametrize(
    ("report", "expected", "message"),
    [
        (
            [{"Name": "example", "Version": "1.0", "License": "UNKNOWN"}],
            _audit_report(("example", "1.0", [])),
            "unreviewed license",
        ),
        (
            [{"Name": "example", "Version": "1.0", "License": "GPL-3.0-only"}],
            _audit_report(("example", "1.0", [])),
            "denied license marker",
        ),
        (
            [{"Name": "example", "Version": "2.0", "License": "MIT"}],
            _audit_report(("example", "1.0", [])),
            "license inventory does not match audited dependency inventory",
        ),
        (
            [
                {"Name": "example-name", "Version": "1.0", "License": "MIT"},
                {"Name": "Example_Name", "Version": "1.0", "License": "MIT"},
            ],
            _audit_report(("example-name", "1.0", [])),
            "license report contains duplicate package",
        ),
    ],
)
def test_license_report_fails_closed(
    tmp_path: Path,
    report: object,
    expected: object,
    message: str,
) -> None:
    result = _run_licenses(tmp_path, report, expected)

    assert result.returncode in {1, 2}
    assert message in result.stderr


def test_exact_unexpired_license_exception_can_override_denial(tmp_path: Path) -> None:
    policy = _policy()
    policy["license_exceptions"] = [
        {
            "package": "example",
            "version": "1.0",
            "license": "GPL-3.0-only",
            "reason": "Synthetic acceptance fixture.",
            "expires": "2026-07-30",
        }
    ]

    result = _run_licenses(
        tmp_path,
        [{"Name": "example", "Version": "1.0", "License": "GPL-3.0-only"}],
        _audit_report(("example", "1.0", [])),
        policy=policy,
    )

    assert result.returncode == 0
    assert "1 exceptions" in result.stdout


@pytest.mark.parametrize(
    ("expires", "report", "expected", "message"),
    [
        (
            "2026-07-28",
            [{"Name": "example", "Version": "1.0", "License": "GPL-3.0-only"}],
            _audit_report(("example", "1.0", [])),
            "suppression expired",
        ),
        (
            "2026-07-30",
            [{"Name": "other", "Version": "1.0", "License": "MIT"}],
            _audit_report(("other", "1.0", [])),
            "unused license exception",
        ),
    ],
)
def test_license_exceptions_expire_and_cannot_go_stale(
    tmp_path: Path,
    expires: str,
    report: object,
    expected: object,
    message: str,
) -> None:
    policy = _policy()
    policy["license_exceptions"] = [
        {
            "package": "example",
            "version": "1.0",
            "license": "GPL-3.0-only",
            "reason": "Synthetic acceptance fixture.",
            "expires": expires,
        }
    ]

    result = _run_licenses(
        tmp_path,
        report,
        expected,
        policy=policy,
    )

    assert result.returncode == 1
    assert message in result.stderr


def test_policy_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"schema_version":"ste-dependency-audit-policy-v1",'
        '"schema_version":"ste-dependency-audit-policy-v1"}',
        encoding="utf-8",
    )
    report_path = tmp_path / "audit.json"
    _write_json(report_path, _audit_report(("example", "1.0", [])))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "vulnerabilities",
            "--policy",
            str(policy_path),
            "--report",
            str(report_path),
            "--scanner-exit-code",
            "0",
            "--profile",
            "core",
            "--as-of",
            AS_OF,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "duplicate key 'schema_version'" in result.stderr


def test_policy_rejects_redundant_license_exception(tmp_path: Path) -> None:
    policy = copy.deepcopy(_policy())
    policy["license_exceptions"] = [
        {
            "package": "example",
            "version": "1.0",
            "license": "MIT",
            "reason": "This exception must be rejected as redundant.",
            "expires": "2026-07-30",
        }
    ]

    result = _run_licenses(
        tmp_path,
        [{"Name": "example", "Version": "1.0", "License": "MIT"}],
        _audit_report(("example", "1.0", [])),
        policy=policy,
    )

    assert result.returncode == 2
    assert "must not duplicate an allowed license expression" in result.stderr


def test_license_inventory_install_is_hash_locked_across_reviewed_indexes() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["dependency-policy"]["steps"]
    install = next(
        step for step in steps if step["name"] == "Build isolated all-extras license inventory"
    )
    command = install["run"]

    assert "--index https://download.pytorch.org/whl/cpu" in command
    assert "--index-strategy unsafe-best-match" in command
    assert "--require-hashes" in command
    assert "--requirement .dependency-audit/all.requirements.txt" in command

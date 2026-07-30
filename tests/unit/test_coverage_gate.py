from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_coverage.py"


def _run_gate(tmp_path: Path, totals: object, *arguments: str) -> subprocess.CompletedProcess[str]:
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "meta": {"branch_coverage": True},
                "totals": totals,
                "files": {"src/ste_compiler/example.py": {"summary": totals}},
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_gate_accepts_counts_exactly_at_both_floors(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": 88,
            "missing_lines": 12,
            "num_statements": 100,
            "covered_branches": 76,
            "missing_branches": 24,
            "num_branches": 100,
        },
    )

    assert result.returncode == 0
    assert result.stdout == (
        "line coverage: 88.00% (88/100; minimum 88%)\n"
        "branch coverage: 76.00% (76/100; minimum 76%)\n"
        "coverage gate passed\n"
    )
    assert result.stderr == ""


def test_gate_fails_when_either_floor_is_missed(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": 87,
            "missing_lines": 13,
            "num_statements": 100,
            "covered_branches": 80,
            "missing_branches": 20,
            "num_branches": 100,
        },
    )

    assert result.returncode == 1
    assert "line coverage: 87.00%" in result.stdout
    assert "branch coverage: 80.00%" in result.stdout
    assert result.stderr == "coverage gate failed\n"


def test_gate_uses_exact_count_ratios_instead_of_rounded_percentages(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": 8799,
            "missing_lines": 1201,
            "num_statements": 10000,
            "covered_branches": 7599,
            "missing_branches": 2401,
            "num_branches": 10000,
        },
    )

    assert result.returncode == 1
    assert "line coverage: 87.99%" in result.stdout
    assert "branch coverage: 75.99%" in result.stdout


def test_gate_supports_explicit_floors(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": 7,
            "missing_lines": 1,
            "num_statements": 8,
            "covered_branches": 3,
            "missing_branches": 1,
            "num_branches": 4,
        },
        "--line-floor",
        "87.5",
        "--branch-floor",
        "75",
    )

    assert result.returncode == 0
    assert "minimum 87.5%" in result.stdout


def test_gate_rejects_malformed_counts(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": True,
            "missing_lines": 12,
            "num_statements": 100,
            "covered_branches": 76,
            "missing_branches": 24,
            "num_branches": 100,
        },
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "coverage gate error: totals.covered_lines must be a non-negative integer\n"
    )


def test_gate_rejects_reports_without_branch_opportunities(tmp_path: Path) -> None:
    result = _run_gate(
        tmp_path,
        {
            "covered_lines": 1,
            "missing_lines": 0,
            "num_statements": 1,
            "covered_branches": 0,
            "missing_branches": 0,
            "num_branches": 0,
        },
    )

    assert result.returncode == 2
    assert result.stderr == "coverage gate error: coverage report contains no branches\n"


@pytest.mark.parametrize(
    ("totals", "message"),
    [
        (
            {
                "covered_lines": 88,
                "num_statements": 100,
                "covered_branches": 76,
                "missing_branches": 24,
                "num_branches": 100,
            },
            "totals.missing_lines must be a non-negative integer",
        ),
        (
            {
                "covered_lines": 88,
                "missing_lines": 999,
                "num_statements": 100,
                "covered_branches": 76,
                "missing_branches": 24,
                "num_branches": 100,
            },
            "totals covered and missing line counts do not equal total statements",
        ),
        (
            {
                "covered_lines": 88,
                "missing_lines": 12,
                "num_statements": 100,
                "covered_branches": 76,
                "missing_branches": 999,
                "num_branches": 100,
            },
            "totals covered and missing branch counts do not equal total branches",
        ),
    ],
)
def test_gate_rejects_incomplete_or_contradictory_reports(
    totals: dict[str, int],
    message: str,
    tmp_path: Path,
) -> None:
    result = _run_gate(tmp_path, totals)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"coverage gate error: {message}\n"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "totals": {
                    "covered_lines": 88,
                    "missing_lines": 12,
                    "num_statements": 100,
                    "covered_branches": 76,
                    "missing_branches": 24,
                    "num_branches": 100,
                },
                "files": {},
            },
            "coverage JSON must report branch_coverage=true",
        ),
        (
            {
                "meta": {"branch_coverage": True},
                "totals": {
                    "covered_lines": 88,
                    "missing_lines": 12,
                    "num_statements": 100,
                    "covered_branches": 76,
                    "missing_branches": 24,
                    "num_branches": 100,
                },
                "files": {},
            },
            "coverage JSON must contain a non-empty files object",
        ),
        (
            {
                "meta": {"branch_coverage": True},
                "totals": {
                    "covered_lines": 88,
                    "missing_lines": 12,
                    "num_statements": 100,
                    "covered_branches": 76,
                    "missing_branches": 24,
                    "num_branches": 100,
                },
                "files": {
                    "src/ste_compiler/example.py": {
                        "summary": {
                            "covered_lines": 89,
                            "missing_lines": 11,
                            "num_statements": 100,
                            "covered_branches": 76,
                            "missing_branches": 24,
                            "num_branches": 100,
                        }
                    }
                },
            },
            "aggregate file counts do not equal report totals",
        ),
    ],
)
def test_gate_rejects_incomplete_or_inconsistent_report_envelopes(
    payload: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"coverage gate error: {message}\n"

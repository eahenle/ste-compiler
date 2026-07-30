from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_coverage.py"


def _run_gate(tmp_path: Path, totals: object, *arguments: str) -> subprocess.CompletedProcess[str]:
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"totals": totals}), encoding="utf-8")
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
            "num_statements": 100,
            "covered_branches": 76,
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
            "num_statements": 100,
            "covered_branches": 80,
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
            "num_statements": 10000,
            "covered_branches": 7599,
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
            "num_statements": 8,
            "covered_branches": 3,
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
            "num_statements": 100,
            "covered_branches": 76,
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
            "num_statements": 1,
            "covered_branches": 0,
            "num_branches": 0,
        },
    )

    assert result.returncode == 2
    assert result.stderr == "coverage gate error: coverage report contains no branches\n"

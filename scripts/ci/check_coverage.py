"""Enforce deterministic line and branch coverage floors from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_LINE_FLOOR = Decimal(88)
DEFAULT_BRANCH_FLOOR = Decimal(76)


class CoverageGateError(ValueError):
    """The coverage report cannot be evaluated safely."""


def _percentage(value: str) -> Decimal:
    try:
        percentage = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}") from error
    if not percentage.is_finite() or percentage < 0 or percentage > 100:
        raise argparse.ArgumentTypeError("percentage must be between 0 and 100")
    return percentage


def _count(summary: dict[str, Any], name: str, *, location: str) -> int:
    value = summary.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CoverageGateError(f"{location}.{name} must be a non-negative integer")
    return value


def _summary_counts(summary: dict[str, Any], *, location: str) -> tuple[int, int, int, int]:
    covered_lines = _count(summary, "covered_lines", location=location)
    missing_lines = _count(summary, "missing_lines", location=location)
    num_statements = _count(summary, "num_statements", location=location)
    covered_branches = _count(summary, "covered_branches", location=location)
    missing_branches = _count(summary, "missing_branches", location=location)
    num_branches = _count(summary, "num_branches", location=location)
    if covered_lines > num_statements:
        raise CoverageGateError(f"{location} covered lines exceed total statements")
    if covered_branches > num_branches:
        raise CoverageGateError(f"{location} covered branches exceed total branches")
    if covered_lines + missing_lines != num_statements:
        raise CoverageGateError(
            f"{location} covered and missing line counts do not equal total statements"
        )
    if covered_branches + missing_branches != num_branches:
        raise CoverageGateError(
            f"{location} covered and missing branch counts do not equal total branches"
        )
    return covered_lines, num_statements, covered_branches, num_branches


def _load_counts(path: Path) -> tuple[int, int, int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageGateError(f"cannot read coverage JSON {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("totals"), dict):
        raise CoverageGateError("coverage JSON must contain an object at totals")
    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("branch_coverage") is not True:
        raise CoverageGateError("coverage JSON must report branch_coverage=true")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise CoverageGateError("coverage JSON must contain a non-empty files object")

    counts = _summary_counts(payload["totals"], location="totals")
    _, num_statements, _, num_branches = counts
    if num_statements == 0:
        raise CoverageGateError("coverage report contains no statements")
    if num_branches == 0:
        raise CoverageGateError("coverage report contains no branches")

    aggregate = [0, 0, 0, 0]
    for filename, file_record in files.items():
        if not isinstance(filename, str) or not filename:
            raise CoverageGateError("coverage file names must be nonblank strings")
        if not isinstance(file_record, dict) or not isinstance(file_record.get("summary"), dict):
            raise CoverageGateError(f"coverage file {filename!r} must contain a summary object")
        file_counts = _summary_counts(
            file_record["summary"],
            location=f"files[{filename!r}].summary",
        )
        aggregate = [total + value for total, value in zip(aggregate, file_counts, strict=True)]
    if tuple(aggregate) != counts:
        raise CoverageGateError("aggregate file counts do not equal report totals")
    return counts


def _display(covered: int, total: int) -> str:
    return f"{Decimal(covered * 100) / Decimal(total):.2f}"


def _passes(covered: int, total: int, floor: Decimal) -> bool:
    return Decimal(covered * 100) >= floor * Decimal(total)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    parser.add_argument("--line-floor", type=_percentage, default=DEFAULT_LINE_FLOOR)
    parser.add_argument("--branch-floor", type=_percentage, default=DEFAULT_BRANCH_FLOOR)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        covered_lines, num_statements, covered_branches, num_branches = _load_counts(
            arguments.report
        )
    except CoverageGateError as error:
        print(f"coverage gate error: {error}", file=sys.stderr)
        return 2

    line_result = _passes(covered_lines, num_statements, arguments.line_floor)
    branch_result = _passes(covered_branches, num_branches, arguments.branch_floor)
    print(
        "line coverage: "
        f"{_display(covered_lines, num_statements)}% "
        f"({covered_lines}/{num_statements}; minimum {arguments.line_floor}%)"
    )
    print(
        "branch coverage: "
        f"{_display(covered_branches, num_branches)}% "
        f"({covered_branches}/{num_branches}; minimum {arguments.branch_floor}%)"
    )
    if not line_result or not branch_result:
        print("coverage gate failed", file=sys.stderr)
        return 1
    print("coverage gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

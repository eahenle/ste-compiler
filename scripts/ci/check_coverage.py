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


def _count(totals: dict[str, Any], name: str) -> int:
    value = totals.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CoverageGateError(f"totals.{name} must be a non-negative integer")
    return value


def _load_counts(path: Path) -> tuple[int, int, int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageGateError(f"cannot read coverage JSON {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("totals"), dict):
        raise CoverageGateError("coverage JSON must contain an object at totals")
    totals = payload["totals"]
    covered_lines = _count(totals, "covered_lines")
    num_statements = _count(totals, "num_statements")
    covered_branches = _count(totals, "covered_branches")
    num_branches = _count(totals, "num_branches")
    if num_statements == 0:
        raise CoverageGateError("coverage report contains no statements")
    if num_branches == 0:
        raise CoverageGateError("coverage report contains no branches")
    if covered_lines > num_statements:
        raise CoverageGateError("covered lines exceed total statements")
    if covered_branches > num_branches:
        raise CoverageGateError("covered branches exceed total branches")
    return covered_lines, num_statements, covered_branches, num_branches


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

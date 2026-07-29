"""Execute the portable example catalog from an installed package."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CATALOG_NAME = "examples/manifest.yaml"
CATALOG_SCHEMA = "ste-executable-examples-v1"
WHEEL_FIXTURE_BASE = "ste_compiler"


class CatalogExecutionError(RuntimeError):
    """Raised when the installed example catalog is unsafe or does not reproduce."""


@dataclass(frozen=True)
class CommandResult:
    """Captured result from one catalog command."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CatalogResult:
    """Summary proving which portable scenarios and commands were evaluated."""

    execution: tuple[str, ...]
    scenario_ids: tuple[int, ...]
    command_count: int

    def as_json(self) -> str:
        """Return the stable machine-readable installed-catalog summary."""

        return json.dumps(
            {
                "command_count": self.command_count,
                "execution": list(self.execution),
                "scenario_ids": list(self.scenario_ids),
            },
            sort_keys=True,
        )


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogExecutionError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise CatalogExecutionError(f"{description} keys must be strings")
    return value


def _require_string_sequence(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise CatalogExecutionError(f"{description} must be a sequence of strings")
    if not all(isinstance(item, str) for item in value):
        raise CatalogExecutionError(f"{description} must be a sequence of strings")
    return tuple(value)


def load_catalog(package_root: Path) -> Mapping[str, Any]:
    """Load the manifest shipped inside one installed package."""

    manifest_path = _safe_package_path(package_root, CATALOG_NAME, "catalog")
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CatalogExecutionError(f"cannot load installed example catalog: {error}") from error
    return _require_mapping(loaded, "catalog")


def _safe_package_path(package_root: Path, raw_path: str, description: str) -> Path:
    root = package_root.resolve()
    if Path(raw_path).is_absolute():
        raise CatalogExecutionError(f"{description} must be relative to the installed package")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        raise CatalogExecutionError(f"{description} escapes the installed package: {raw_path!r}")
    return path


def _safe_output_path(temporary: Path, raw_path: str) -> Path:
    if not raw_path.startswith("{tmp}/"):
        raise CatalogExecutionError(
            f"expected artifact path must start with '{{tmp}}/': {raw_path!r}"
        )
    root = temporary.resolve()
    path = Path(raw_path.replace("{tmp}", str(root), 1)).resolve()
    if not path.is_relative_to(root):
        raise CatalogExecutionError(f"expected artifact path escapes its scenario: {raw_path!r}")
    return path


def _json_path(payload: object, dotted_path: str) -> object:
    if not dotted_path or any(not component for component in dotted_path.split(".")):
        raise CatalogExecutionError(f"invalid dotted JSON path: {dotted_path!r}")
    current = payload
    for component in dotted_path.split("."):
        if isinstance(current, list):
            if not component.isdecimal():
                raise CatalogExecutionError(
                    f"JSON path {dotted_path!r} uses a non-index list component"
                )
            index = int(component)
            if index >= len(current):
                raise CatalogExecutionError(f"JSON path {dotted_path!r} index is out of range")
            current = current[index]
        elif isinstance(current, Mapping):
            if component not in current:
                raise CatalogExecutionError(f"JSON path {dotted_path!r} does not exist")
            current = current[component]
        else:
            raise CatalogExecutionError(
                f"JSON path {dotted_path!r} traverses a non-container value"
            )
    return current


def _format_argument(
    argument: str,
    *,
    package_root: Path,
    temporary: Path,
    fixtures: frozenset[str],
) -> str:
    if "{tmp}" in argument:
        if not argument.startswith("{tmp}"):
            raise CatalogExecutionError(f"{{tmp}} must start a catalog argument: {argument!r}")
        expanded = argument.replace("{tmp}", str(temporary.resolve()), 1)
        output = Path(expanded).resolve()
        if not output.is_relative_to(temporary.resolve()):
            raise CatalogExecutionError(f"catalog argument escapes its scenario: {argument!r}")
        return str(output)
    candidate = _safe_package_path(package_root, argument, "catalog argument")
    if candidate.exists():
        if argument not in fixtures:
            raise CatalogExecutionError(
                f"package-relative catalog argument is not a declared fixture: {argument!r}"
            )
        return str(candidate)
    return argument


def _installed_command(
    argv: object,
    *,
    package_root: Path,
    temporary: Path,
    fixtures: frozenset[str],
) -> tuple[str, ...]:
    arguments = _require_string_sequence(argv, "command argv")
    if not arguments:
        raise CatalogExecutionError("command argv cannot be empty")
    if arguments[0] == "ste-compiler":
        tail = arguments[1:]
        prefix = (sys.executable, "-m", "ste_compiler.cli")
    elif arguments == ("python", "examples/custom_resources.py"):
        tail = ()
        prefix = (sys.executable, "-m", "ste_compiler.examples.custom_resources")
    else:
        raise CatalogExecutionError(f"unsupported portable catalog argv: {arguments!r}")
    return (
        *prefix,
        *(
            _format_argument(
                argument,
                package_root=package_root,
                temporary=temporary,
                fixtures=fixtures,
            )
            for argument in tail
        ),
    )


def _execute_command(command: tuple[str, ...], *, cwd: Path) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _parse_json(raw: str, description: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise CatalogExecutionError(f"{description} is not valid JSON: {error}") from error


def _assert_json_paths(
    payload: object,
    raw_paths: object,
    *,
    description: str,
) -> None:
    paths = _require_mapping(raw_paths, description)
    for dotted_path, expected_value in paths.items():
        observed = _json_path(payload, dotted_path)
        if observed != expected_value:
            raise CatalogExecutionError(
                f"{description} {dotted_path!r} expected {expected_value!r}, got {observed!r}"
            )


def _assert_expected(
    expected_raw: object,
    result: CommandResult,
    temporary: Path,
    *,
    package_root: Path,
    fixtures: frozenset[str],
) -> None:
    expected = _require_mapping(expected_raw, "command expectation")
    expected_exit_code = expected.get("exit_code")
    if not isinstance(expected_exit_code, int):
        raise CatalogExecutionError("command expectation exit_code must be an integer")
    if result.exit_code != expected_exit_code:
        detail = result.stderr or result.stdout
        raise CatalogExecutionError(
            f"command expected exit code {expected_exit_code}, got {result.exit_code}: {detail}"
        )

    stdout_contains = expected.get("stdout_contains")
    if stdout_contains is not None:
        if not isinstance(stdout_contains, str):
            raise CatalogExecutionError("stdout_contains must be a string")
        if stdout_contains not in result.stdout:
            raise CatalogExecutionError(f"stdout does not contain {stdout_contains!r}")

    stdout_paths = expected.get("stdout_json_paths")
    if stdout_paths is not None:
        _assert_json_paths(
            _parse_json(result.stdout, "command stdout"),
            stdout_paths,
            description="stdout_json_paths",
        )

    file_paths = expected.get("file_json_paths", {})
    for raw_path, paths in _require_mapping(file_paths, "file_json_paths").items():
        artifact = _safe_output_path(temporary, raw_path)
        try:
            payload = _parse_json(artifact.read_text(encoding="utf-8"), str(artifact))
        except OSError as error:
            raise CatalogExecutionError(
                f"cannot read expected artifact {artifact}: {error}"
            ) from error
        _assert_json_paths(payload, paths, description=f"file_json_paths[{raw_path!r}]")

    file_matches = _require_mapping(expected.get("file_matches", {}), "file_matches")
    for raw_path, frozen_path_raw in file_matches.items():
        if not isinstance(frozen_path_raw, str):
            raise CatalogExecutionError("file_matches values must be package-relative strings")
        if frozen_path_raw not in fixtures:
            raise CatalogExecutionError(
                f"frozen expected artifact is not a declared fixture: {frozen_path_raw!r}"
            )
        artifact = _safe_output_path(temporary, raw_path)
        frozen = _safe_package_path(package_root, frozen_path_raw, "frozen expected artifact")
        try:
            observed_bytes = artifact.read_bytes()
            expected_bytes = frozen.read_bytes()
        except OSError as error:
            raise CatalogExecutionError(f"cannot compare frozen artifact: {error}") from error
        if observed_bytes != expected_bytes:
            raise CatalogExecutionError(
                f"generated artifact {raw_path!r} does not match {frozen_path_raw!r}"
            )


def _portable_execution(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    distribution = _require_mapping(catalog.get("distribution"), "catalog distribution")
    if distribution.get("wheel_fixture_base") != WHEEL_FIXTURE_BASE:
        raise CatalogExecutionError(
            f"wheel_fixture_base must be {WHEEL_FIXTURE_BASE!r} for installed execution"
        )
    execution = _require_string_sequence(
        distribution.get("portable_execution"),
        "portable_execution",
    )
    if not execution:
        raise CatalogExecutionError("portable_execution cannot be empty")
    return execution


def _validate_fixtures(scenario: Mapping[str, Any], package_root: Path) -> frozenset[str]:
    fixtures = _require_string_sequence(scenario.get("fixtures"), "scenario fixtures")
    if len(set(fixtures)) != len(fixtures):
        raise CatalogExecutionError("scenario fixtures must be unique")
    for raw_fixture in fixtures:
        fixture = _safe_package_path(package_root, raw_fixture, "scenario fixture")
        if not fixture.exists():
            raise CatalogExecutionError(
                f"installed scenario fixture does not exist: {raw_fixture!r}"
            )
    return frozenset(fixtures)


def _execute_and_evaluate(
    command_raw: object,
    *,
    package_root: Path,
    temporary: Path,
    fixtures: frozenset[str],
) -> None:
    command = _require_mapping(command_raw, "scenario command")
    argv = _installed_command(
        command.get("argv"),
        package_root=package_root,
        temporary=temporary,
        fixtures=fixtures,
    )
    result = _execute_command(argv, cwd=temporary)
    _assert_expected(
        command.get("expected"),
        result,
        temporary,
        package_root=package_root,
        fixtures=fixtures,
    )


def run_portable_catalog(
    catalog: Mapping[str, Any],
    *,
    package_root: Path,
    temporary_root: Path,
) -> CatalogResult:
    """Execute and evaluate every command in every portable manifest scenario."""

    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise CatalogExecutionError(f"catalog schema must be {CATALOG_SCHEMA!r}")
    execution = _portable_execution(catalog)
    scenarios_raw = catalog.get("scenarios")
    if not isinstance(scenarios_raw, Sequence) or isinstance(scenarios_raw, str | bytes):
        raise CatalogExecutionError("catalog scenarios must be a sequence")

    temporary_root.mkdir(parents=True, exist_ok=False)
    selected: list[int] = []
    command_count = 0
    for scenario_raw in scenarios_raw:
        scenario = _require_mapping(scenario_raw, "scenario")
        if scenario.get("execution") not in execution:
            continue
        scenario_id = scenario.get("id")
        slug = scenario.get("slug")
        if not isinstance(scenario_id, int) or not isinstance(slug, str):
            raise CatalogExecutionError("portable scenario must have an integer id and string slug")
        fixtures = _validate_fixtures(scenario, package_root)
        commands = scenario.get("commands")
        if not isinstance(commands, Sequence) or isinstance(commands, str | bytes) or not commands:
            raise CatalogExecutionError(f"portable scenario {scenario_id} must define commands")
        temporary = temporary_root / f"{scenario_id}-{slug}"
        temporary.mkdir()
        for command in commands:
            _execute_and_evaluate(
                command,
                package_root=package_root,
                temporary=temporary,
                fixtures=fixtures,
            )
            command_count += 1
        selected.append(scenario_id)

    if not selected:
        raise CatalogExecutionError("catalog does not select any portable scenarios")
    return CatalogResult(execution, tuple(selected), command_count)


def main() -> None:
    """Execute the catalog shipped beside this module."""

    package_root = Path(__file__).resolve().parents[1]
    if len(sys.argv) != 2:
        raise CatalogExecutionError("usage: python -m ste_compiler.examples.catalog_runner OUTPUT")
    result = run_portable_catalog(
        load_catalog(package_root),
        package_root=package_root,
        temporary_root=Path(sys.argv[1]),
    )
    print(result.as_json())


if __name__ == "__main__":
    main()

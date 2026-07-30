from __future__ import annotations

import ast
import copy
import json
import os
import runpy
import socket
import subprocess
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ste_compiler.cli import app
from ste_compiler.examples import catalog_runner
from tests.executable_example_helpers import (
    forbid_network,
    load_example_manifest,
)

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "examples/manifest.yaml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _load_manifest() -> dict[str, Any]:
    return load_example_manifest()


def _pytest_node(command: dict[str, Any]) -> str | None:
    nodes = [argument for argument in command["argv"] if "::" in argument]
    if command["argv"][0] != "pytest":
        assert not nodes
        return None
    assert len(nodes) == 1, "manifest pytest commands require exactly one explicit node ID"
    return nodes[0]


def _owner_defines_test(owner: Path, test_name: str) -> bool:
    module = ast.parse(owner.read_text(encoding="utf-8"), filename=str(owner))
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == test_name
        for node in module.body
    )


def _ci_job_runs(job: dict[str, Any]) -> str:
    return "\n".join(run for step in job["steps"] if isinstance((run := step.get("run")), str))


def _pytest_targets(command: str) -> tuple[PurePosixPath, ...]:
    return tuple(
        PurePosixPath(token)
        for token in command.replace('"', "").replace("'", "").split()
        if token == "tests" or token.startswith("tests/")
    )


def _target_contains_owner(target: PurePosixPath, owner: PurePosixPath) -> bool:
    return target == owner or target in owner.parents


def _json_path(value: object, path: str) -> object:
    current = value
    for component in path.split("."):
        if isinstance(current, list):
            current = current[int(component)]
        elif isinstance(current, Mapping):
            current = current[component]
        else:  # pragma: no cover - assertion below reports the full failing path.
            raise TypeError(f"{path!r} traverses a non-container value")
    return current


def _format_arguments(arguments: list[str], temporary: Path) -> list[str]:
    return [argument.replace("{tmp}", str(temporary)) for argument in arguments]


def _run_python_example(path: str) -> tuple[int, str, str]:
    output = StringIO()
    try:
        with redirect_stdout(output):
            namespace = runpy.run_path(str(ROOT / path))
            namespace["main"]()
    except SystemExit as error:
        return int(error.code or 0), output.getvalue(), ""
    return 0, output.getvalue(), ""


def _assert_expected(
    *,
    expected: dict[str, Any],
    exit_code: int,
    stdout: str,
    stderr: str,
    temporary: Path,
    fixtures: set[str],
) -> None:
    assert exit_code == expected["exit_code"], stderr or stdout
    if contains := expected.get("stdout_contains"):
        assert contains in stdout
    if paths := expected.get("stdout_json_paths"):
        payload = json.loads(stdout)
        for path, value in paths.items():
            observed = _json_path(payload, path)
            assert catalog_runner.strict_json_equal(observed, value), (
                f"{path!r}: expected {value!r} ({type(value).__name__}), "
                f"got {observed!r} ({type(observed).__name__})"
            )
    for raw_path, paths in expected.get("file_json_paths", {}).items():
        artifact = Path(raw_path.replace("{tmp}", str(temporary)))
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        for path, value in paths.items():
            observed = _json_path(payload, path)
            assert catalog_runner.strict_json_equal(observed, value), (
                f"{path!r}: expected {value!r} ({type(value).__name__}), "
                f"got {observed!r} ({type(observed).__name__})"
            )
    for raw_path, frozen_path in expected.get("file_matches", {}).items():
        assert frozen_path in fixtures
        artifact = Path(raw_path.replace("{tmp}", str(temporary)))
        assert artifact.read_bytes() == (ROOT / frozen_path).read_bytes()


def test_example_manifest_is_complete_and_honest():
    manifest = _load_manifest()

    assert manifest["schema_version"] == "ste-executable-examples-v1"
    assert manifest["distribution"] == {
        "wheel_catalog": "ste_compiler/examples/manifest.yaml",
        "wheel_fixture_base": "ste_compiler",
        "portable_execution": ["portable-ci", "posix-ci"],
        "portable_execution_overrides": {"win32": ["portable-ci"]},
        "source_only_execution": ["existing-ci", "neural-ci"],
    }
    scenarios = manifest["scenarios"]
    assert [scenario["id"] for scenario in scenarios] == list(range(1, 14))
    assert len({scenario["slug"] for scenario in scenarios}) == 13

    for scenario in scenarios:
        assert scenario["status"] in {"tested", "partial", "gated"}
        assert scenario["network"] in {"forbidden", "optional"}
        for fixture in scenario["fixtures"]:
            assert (ROOT / fixture).exists(), fixture
        if scenario["status"] == "gated":
            assert scenario["commands"] == []
            assert scenario["gate"]
        else:
            assert scenario["commands"]
            if scenario["status"] == "partial":
                assert scenario["limitation"]
            for command in scenario["commands"]:
                assert command["argv"]
                assert "exit_code" in command["expected"]
        if scenario["execution"] in {"existing-ci", "neural-ci"}:
            assert (ROOT / scenario["pytest_owner"]).is_file()
            assert scenario["ci_job"]


def test_manifest_declares_cli_defaults_and_every_package_relative_argument():
    scenarios = {scenario["id"]: scenario for scenario in _load_manifest()["scenarios"]}

    for scenario in scenarios.values():
        fixtures = set(scenario["fixtures"])
        assert len(fixtures) == len(scenario["fixtures"])
        for command in scenario["commands"]:
            catalog_runner.validate_command_contract(
                command,
                fixtures=frozenset(fixtures),
            )
            if command["argv"] != ["python", "examples/custom_resources.py"]:
                for argument in command["argv"][1:]:
                    if "{tmp}" not in argument and (ROOT / argument).exists():
                        assert argument in fixtures
            for frozen_path in command["expected"].get("file_matches", {}).values():
                assert frozen_path in fixtures


def test_manifest_pytest_nodes_are_owned_and_selected_by_their_ci_jobs():
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]

    for scenario in _load_manifest()["scenarios"]:
        pytest_commands = [
            (command, node)
            for command in scenario["commands"]
            if (node := _pytest_node(command)) is not None
        ]
        if not pytest_commands:
            assert "pytest_owner" not in scenario
            assert "ci_job" not in scenario
            continue

        owner = PurePosixPath(scenario["pytest_owner"])
        assert scenario["ci_job"] in jobs
        ci_runs = _ci_job_runs(jobs[scenario["ci_job"]])
        ci_targets = _pytest_targets(ci_runs)
        assert ci_targets, f"CI job {scenario['ci_job']} has no explicit pytest test target"
        assert any(_target_contains_owner(target, owner) for target in ci_targets), (
            f"CI job {scenario['ci_job']} does not select {owner}"
        )

        for _, node in pytest_commands:
            node_owner, separator, test_name = node.partition("::")
            assert separator
            assert PurePosixPath(node_owner) == owner
            assert "::" not in test_name, "only module-level pytest functions are supported"
            assert _owner_defines_test(ROOT / owner, test_name), (
                f"pytest node ID has rotted: {node}"
            )


INSTALLED_SCENARIOS = tuple(
    scenario
    for scenario in _load_manifest()["scenarios"]
    if scenario["execution"] in {"portable-ci", "posix-ci"}
)


def test_network_tripwire_blocks_connection_and_datagram_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    forbid_network(monkeypatch)

    operations = (
        lambda: socket.create_connection(("example.invalid", 443)),
        lambda: socket.getaddrinfo("example.invalid", 443),
        lambda: socket.gethostbyname("example.invalid"),
        lambda: socket.gethostbyname_ex("example.invalid"),
        lambda: socket.gethostbyaddr("192.0.2.1"),
        lambda: socket.getnameinfo(("192.0.2.1", 443), 0),
        lambda: socket.socket().connect(("127.0.0.1", 9)),
        lambda: socket.socket().connect_ex(("127.0.0.1", 9)),
        lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", 9)),
    )
    for operation in operations:
        with pytest.raises(AssertionError, match="unexpected network access"):
            operation()


@pytest.mark.parametrize(
    "scenario",
    INSTALLED_SCENARIOS,
    ids=[scenario["slug"] for scenario in INSTALLED_SCENARIOS],
)
def test_installed_executable_example(
    scenario: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    forbid_network(monkeypatch)
    monkeypatch.chdir(ROOT)
    runner = CliRunner()
    temporary = tmp_path / scenario["slug"]
    temporary.mkdir()

    for command in scenario["commands"]:
        arguments = _format_arguments(command["argv"], temporary)
        if arguments[0] == "ste-compiler":
            result = runner.invoke(app, arguments[1:])
            exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
        elif arguments[:2] == ["python", "examples/custom_resources.py"]:
            exit_code, stdout, stderr = _run_python_example(arguments[1])
        else:  # pragma: no cover - manifest validation keeps the command set explicit.
            raise AssertionError(f"unsupported core example command: {arguments}")
        _assert_expected(
            expected=command["expected"],
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            temporary=temporary,
            fixtures=set(scenario["fixtures"]),
        )


def test_installed_executable_examples_run_when_pytest_starts_outside_checkout(
    tmp_path: Path,
) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{Path(__file__).resolve()}::test_installed_executable_example",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_installed_catalog_runner_selects_every_default_platform_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    manifest = _load_manifest()
    portable = set(manifest["distribution"]["portable_execution"])
    expected = [scenario for scenario in manifest["scenarios"] if scenario["execution"] in portable]
    evaluated: list[object] = []

    def record_command(
        command: object,
        *,
        package_root: Path,
        temporary: Path,
        fixtures: frozenset[str],
    ) -> None:
        assert package_root == ROOT
        assert temporary.is_relative_to(tmp_path)
        assert fixtures
        evaluated.append(command)

    monkeypatch.setattr(catalog_runner, "_execute_and_evaluate", record_command)
    result = catalog_runner.run_portable_catalog(
        manifest,
        package_root=ROOT,
        temporary_root=tmp_path / "installed-catalog",
        platform=platform,
    )

    assert result.execution == tuple(manifest["distribution"]["portable_execution"])
    assert result.scenario_ids == tuple(scenario["id"] for scenario in expected)
    assert result.command_count == sum(len(scenario["commands"]) for scenario in expected)
    assert evaluated == [command for scenario in expected for command in scenario["commands"]]


def test_installed_catalog_runner_applies_win32_portable_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _load_manifest()
    evaluated: list[object] = []

    def record_command(
        command: object,
        *,
        package_root: Path,
        temporary: Path,
        fixtures: frozenset[str],
    ) -> None:
        assert package_root == ROOT
        assert temporary.is_relative_to(tmp_path)
        assert fixtures
        evaluated.append(command)

    monkeypatch.setattr(catalog_runner.sys, "platform", "win32")
    monkeypatch.setattr(catalog_runner, "_execute_and_evaluate", record_command)
    result = catalog_runner.run_portable_catalog(
        manifest,
        package_root=ROOT,
        temporary_root=tmp_path / "win32-installed-catalog",
    )
    expected = [
        scenario for scenario in manifest["scenarios"] if scenario["execution"] == "portable-ci"
    ]

    assert result.execution == ("portable-ci",)
    assert result.scenario_ids == (1, 2, 4, 5, 10, 11)
    assert result.command_count == 9
    assert evaluated == [command for scenario in expected for command in scenario["commands"]]


def test_installed_catalog_runner_fails_a_mutated_manifest_expectation(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["commands"][0]["expected"]["stdout_json_paths"]["validation.status"] = "rejected"

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="validation.status.*expected 'rejected', got 'accepted'",
    ):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "mutated-catalog",
        )


def test_installed_catalog_runner_rejects_unsupported_argv(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["commands"][0]["argv"] = ["sh", "-c", "exit 0"]

    with pytest.raises(catalog_runner.CatalogExecutionError, match="unsupported portable catalog"):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "unsupported-command",
        )


def test_installed_catalog_runner_rejects_fixture_escape(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["fixtures"] = ["../pyproject.toml"]

    with pytest.raises(catalog_runner.CatalogExecutionError, match="escapes the installed package"):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "escaping-fixture",
        )


def test_installed_catalog_runner_rejects_undeclared_package_argument(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][1]
    manifest["scenarios"] = [scenario]
    scenario["fixtures"].remove("data/end_to_end/hydraulic_warning.txt")

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="package-relative catalog argument is not a declared fixture",
    ):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "undeclared-argument",
        )


@pytest.mark.parametrize(
    ("duplicate_field", "replacement", "message"),
    [
        ("id", 1, "scenario id is duplicated"),
        ("slug", "raw-source-deterministic", "scenario slug is duplicated"),
    ],
)
def test_installed_catalog_runner_rejects_duplicate_scenario_identity(
    duplicate_field: str,
    replacement: object,
    message: str,
    tmp_path: Path,
) -> None:
    manifest = copy.deepcopy(_load_manifest())
    first = manifest["scenarios"][0]
    duplicate = copy.deepcopy(manifest["scenarios"][1])
    duplicate[duplicate_field] = replacement
    manifest["scenarios"] = [first, duplicate]

    with pytest.raises(catalog_runner.CatalogExecutionError, match=message):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "duplicate-scenario",
        )


def test_installed_catalog_runner_rejects_slug_escape_before_creating_output(
    tmp_path: Path,
) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["slug"] = "seed/../../outside"
    output = tmp_path / "catalog-output"

    with pytest.raises(catalog_runner.CatalogExecutionError, match="lowercase kebab-case"):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=output,
        )

    assert not output.exists()
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "escape_kind",
    ["absolute", "traversal"],
)
def test_installed_catalog_runner_rejects_equals_form_output_escape(
    escape_kind: str,
    tmp_path: Path,
) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][4]
    manifest["scenarios"] = [scenario]
    outside = tmp_path / "ste-compiler-catalog-escape"
    unsafe_value = (
        str(outside) if escape_kind == "absolute" else "../../ste-compiler-catalog-escape"
    )
    unsafe_option = f"--output={unsafe_value}"
    argv = scenario["commands"][0]["argv"]
    output_index = argv.index("--output")
    argv[output_index : output_index + 2] = [unsafe_option]

    with pytest.raises(catalog_runner.CatalogExecutionError, match="not equals form"):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "equals-output",
        )
    assert not outside.exists()


@pytest.mark.parametrize(
    ("scenario_id", "required_fixture", "wrong_fixture"),
    [
        (
            4,
            "examples/resources/custom_vocabulary.yaml",
            "data/demo_vocabulary.yaml",
        ),
        (
            5,
            "data/demonstration_corpus/v2/source-construction.json",
            "data/demonstration_corpus/v1/source-construction.json",
        ),
        (
            12,
            "data/demo_vocabulary.yaml",
            "data/training/encoder-decoder-schema-example.yaml",
        ),
    ],
)
def test_source_manifest_validation_binds_implicit_command_resources(
    scenario_id: int,
    required_fixture: str,
    wrong_fixture: str,
) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = next(item for item in manifest["scenarios"] if item["id"] == scenario_id)
    scenario["fixtures"].remove(required_fixture)
    scenario["fixtures"].append(wrong_fixture)
    fixtures = frozenset(scenario["fixtures"])

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="implicit fixtures are not declared",
    ):
        for command in scenario["commands"]:
            catalog_runner.validate_command_contract(command, fixtures=fixtures)


def test_installed_catalog_validation_binds_implicit_command_resources(
    tmp_path: Path,
) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][4]
    manifest["scenarios"] = [scenario]
    scenario["fixtures"].remove("data/demonstration_corpus/v2/terminology.yaml")
    scenario["fixtures"].append("data/demonstration_corpus/v1/terminology.yaml")

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="implicit fixtures are not declared.*v2/terminology",
    ):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "wrong-implicit-fixture",
        )


def test_installed_catalog_runner_rejects_duplicate_fixtures(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["fixtures"].append(scenario["fixtures"][0])

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="scenario fixtures must be unique",
    ):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "duplicate-fixture",
        )


def test_installed_catalog_runner_rejects_missing_fixture(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_load_manifest())
    scenario = manifest["scenarios"][0]
    manifest["scenarios"] = [scenario]
    scenario["fixtures"] = ["data/examples/does-not-exist.yaml"]

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="installed scenario fixture does not exist",
    ):
        catalog_runner.run_portable_catalog(
            manifest,
            package_root=ROOT,
            temporary_root=tmp_path / "missing-fixture",
        )


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (True, 1),
        (1, 1.0),
        ({"nested": [True, 1, 1.0]}, {"nested": [1, 1.0, 1]}),
    ],
)
def test_installed_json_expectations_reject_recursive_type_coercion(
    observed: object,
    expected: object,
) -> None:
    with pytest.raises(catalog_runner.CatalogExecutionError, match="expected"):
        catalog_runner._assert_json_paths(
            {"value": observed},
            {"value": expected},
            description="strict paths",
        )


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (True, 1),
        (1, 1.0),
        ({"nested": [True, 1, 1.0]}, {"nested": [1, 1.0, 1]}),
    ],
)
def test_source_json_expectations_reject_recursive_type_coercion(
    observed: object,
    expected: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError, match="expected"):
        _assert_expected(
            expected={
                "exit_code": 0,
                "stdout_json_paths": {"value": expected},
            },
            exit_code=0,
            stdout=json.dumps({"value": observed}),
            stderr="",
            temporary=tmp_path,
            fixtures=set(),
        )


def test_installed_and_source_json_expectations_accept_same_type_values(
    tmp_path: Path,
) -> None:
    value = {"nested": [True, 1, 1.0, "one", None]}
    catalog_runner._assert_json_paths(
        {"value": value},
        {"value": value},
        description="strict paths",
    )
    _assert_expected(
        expected={
            "exit_code": 0,
            "stdout_json_paths": {"value": value},
        },
        exit_code=0,
        stdout=json.dumps({"value": value}),
        stderr="",
        temporary=tmp_path,
        fixtures=set(),
    )


@pytest.mark.parametrize(
    ("artifact_path", "message"),
    (
        ("report.json", "expected artifact path must start with"),
        ("{tmp}/../escaped.json", "expected artifact path escapes its scenario"),
    ),
)
def test_catalog_expectation_rejects_unconfined_artifact_paths(
    tmp_path: Path,
    artifact_path: str,
    message: str,
) -> None:
    with pytest.raises(catalog_runner.CatalogExecutionError, match=message):
        catalog_runner._assert_expected(
            {
                "exit_code": 0,
                "file_json_paths": {artifact_path: {"status": "accepted"}},
            },
            catalog_runner.CommandResult(0, "", ""),
            tmp_path,
            package_root=ROOT,
            fixtures=frozenset(),
        )


def test_catalog_expectation_rejects_missing_json_artifact(tmp_path: Path) -> None:
    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="cannot read expected artifact",
    ):
        catalog_runner._assert_expected(
            {
                "exit_code": 0,
                "file_json_paths": {"{tmp}/missing.json": {"status": "accepted"}},
            },
            catalog_runner.CommandResult(0, "", ""),
            tmp_path,
            package_root=ROOT,
            fixtures=frozenset(),
        )


def test_catalog_expectation_rejects_invalid_json_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid.json"
    artifact.write_text("not JSON", encoding="utf-8")

    with pytest.raises(catalog_runner.CatalogExecutionError, match="is not valid JSON"):
        catalog_runner._assert_expected(
            {
                "exit_code": 0,
                "file_json_paths": {"{tmp}/invalid.json": {"status": "accepted"}},
            },
            catalog_runner.CommandResult(0, "", ""),
            tmp_path,
            package_root=ROOT,
            fixtures=frozenset(),
        )


def test_catalog_expectation_rejects_undeclared_frozen_artifact(tmp_path: Path) -> None:
    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="frozen expected artifact is not a declared fixture",
    ):
        catalog_runner._assert_expected(
            {
                "exit_code": 0,
                "file_matches": {
                    "{tmp}/report.json": "data/examples/undeclared-report.json",
                },
            },
            catalog_runner.CommandResult(0, "", ""),
            tmp_path,
            package_root=ROOT,
            fixtures=frozenset(),
        )


def test_catalog_expectation_rejects_frozen_artifact_byte_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_bytes(b"observed\n")
    frozen_path = "data/benchmark/v1/expected-report/metrics.json"

    with pytest.raises(
        catalog_runner.CatalogExecutionError,
        match="generated artifact.*does not match",
    ):
        catalog_runner._assert_expected(
            {
                "exit_code": 0,
                "file_matches": {"{tmp}/report.json": frozen_path},
            },
            catalog_runner.CommandResult(0, "", ""),
            tmp_path,
            package_root=ROOT,
            fixtures=frozenset({frozen_path}),
        )

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from ste_compiler.cli import (
    DEFAULT_TERMINOLOGY_RELATIVE,
    DEFAULT_VOCABULARY_RELATIVE,
    DEMO_IR_RELATIVE,
    DEMO_SOURCE_RELATIVE,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "examples/manifest.yaml"


def load_example_manifest() -> dict[str, Any]:
    loaded = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def example_scenario(scenario_id: int) -> dict[str, Any]:
    scenarios = load_example_manifest()["scenarios"]
    matches = [scenario for scenario in scenarios if scenario["id"] == scenario_id]
    assert len(matches) == 1, f"expected exactly one executable example scenario {scenario_id}"
    return matches[0]


def example_fixture(scenario_id: int, fixture_index: int) -> Path:
    fixture = example_scenario(scenario_id)["fixtures"][fixture_index]
    path = ROOT / fixture
    assert path.exists(), f"scenario {scenario_id} fixture does not exist: {fixture}"
    return path


def data_fixture(relative: Path) -> str:
    """Return the catalog path for one package data default."""

    return (Path("data") / relative).as_posix()


DEFAULT_VOCABULARY_FIXTURE = data_fixture(DEFAULT_VOCABULARY_RELATIVE)
DEFAULT_TERMINOLOGY_FIXTURE = data_fixture(DEFAULT_TERMINOLOGY_RELATIVE)
DEMO_SOURCE_FIXTURE = data_fixture(DEMO_SOURCE_RELATIVE)
DEMO_IR_FIXTURE = data_fixture(DEMO_IR_RELATIVE)


def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every supported Python socket connection, datagram, and resolver path."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected network access: args={args!r}, kwargs={kwargs!r}")

    for method in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, method):
            monkeypatch.setattr(socket.socket, method, reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    for resolver in (
        "getaddrinfo",
        "gethostbyaddr",
        "gethostbyname",
        "gethostbyname_ex",
        "getnameinfo",
    ):
        if hasattr(socket, resolver):
            monkeypatch.setattr(socket, resolver, reject_network)

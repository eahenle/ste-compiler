"""Build, inspect, reproduce, and execute the public Python distributions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHEEL_SUFFIXES = (
    "ste_compiler/py.typed",
    "ste_compiler/data/demonstration_corpus/v1/source-construction.json",
    "ste_compiler/data/demonstration_corpus/v2/source-construction.json",
    "ste_compiler/data/realizers/decoder-only-lora-local-bundle-schema-example.yaml",
    "ste_compiler/data/realizers/encoder-decoder-local-bundle-schema-example.yaml",
    "ste_compiler/examples/__init__.py",
    "ste_compiler/examples/custom_resources.py",
    "ste_compiler/examples/manifest.yaml",
    "ste_compiler/examples/resources/custom_installation.yaml",
    "ste_compiler/examples/resources/custom_terminology.yaml",
    "ste_compiler/examples/resources/custom_vocabulary.yaml",
    ".dist-info/METADATA",
    ".dist-info/entry_points.txt",
)
SDIST_SUFFIXES = (
    "/LICENSE",
    "/README.md",
    "/src/ste_compiler/py.typed",
    "/datasets/demonstration-corpus-1/manifest.json",
    "/datasets/demonstration-corpus-2/manifest.json",
    "/docs/v1-implementation-plan.md",
    "/examples/__init__.py",
    "/examples/custom_resources.py",
    "/examples/manifest.yaml",
    "/examples/resources/custom_installation.yaml",
    "/examples/resources/custom_terminology.yaml",
    "/examples/resources/custom_vocabulary.yaml",
    "/scripts/ci/distribution_smoke.py",
    "/tests/integration/test_executable_examples.py",
)


def _run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"command failed: {command!r}\nstdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    return completed.stdout


def _source_date_epoch() -> str:
    return _run("git", "show", "-s", "--format=%ct", "HEAD").strip()


def _build(output: Path, environment: dict[str, str]) -> tuple[Path, Path]:
    _run(
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        "--sdist",
        "--wheel",
        "--outdir",
        str(output),
        str(ROOT),
        env=environment,
    )
    wheels = tuple(output.glob("*.whl"))
    sdists = tuple(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("distribution build must produce exactly one wheel and one sdist")
    return wheels[0], sdists[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_suffixes(names: tuple[str, ...], suffixes: tuple[str, ...], artifact: str) -> None:
    for suffix in suffixes:
        if not any(name.endswith(suffix) for name in names):
            raise RuntimeError(f"{artifact} is missing required member ending in {suffix!r}")


def _inspect(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = tuple(archive.namelist())
        _require_suffixes(wheel_names, WHEEL_SUFFIXES, "wheel")
        if any(name.endswith((".pyc", ".pyo")) or "__pycache__/" in name for name in wheel_names):
            raise RuntimeError("wheel contains generated Python cache artifacts")
    with tarfile.open(sdist) as archive:
        sdist_names = tuple(archive.getnames())
        _require_suffixes(sdist_names, SDIST_SUFFIXES, "sdist")
        if any("/.git/" in name or "/.venv/" in name for name in sdist_names):
            raise RuntimeError("sdist contains checkout-local metadata")


def _build_wheel_from_sdist(
    sdist: Path,
    temporary_root: Path,
    environment: dict[str, str],
) -> Path:
    extracted = temporary_root / "extracted-sdist"
    extracted.mkdir()
    with tarfile.open(sdist) as archive:
        archive.extractall(extracted, filter="data")
    source_roots = tuple(path for path in extracted.iterdir() if path.is_dir())
    if len(source_roots) != 1:
        raise RuntimeError("sdist must contain exactly one top-level source directory")
    output = temporary_root / "sdist-wheel"
    _run(
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        "--wheel",
        "--outdir",
        str(output),
        str(source_roots[0]),
        env=environment,
    )
    wheels = tuple(output.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("sdist build must produce exactly one wheel")
    return wheels[0]


def _assert_reproducible(
    first: tuple[Path, Path],
    second: tuple[Path, Path],
) -> dict[str, str]:
    first_by_name = {path.name: path for path in first}
    second_by_name = {path.name: path for path in second}
    if first_by_name.keys() != second_by_name.keys():
        raise RuntimeError("repeated builds produced different distribution filenames")
    checksums: dict[str, str] = {}
    for name, first_path in sorted(first_by_name.items()):
        first_digest = _sha256(first_path)
        second_digest = _sha256(second_by_name[name])
        if first_digest != second_digest:
            raise RuntimeError(f"distribution is not reproducible with SOURCE_DATE_EPOCH: {name}")
        checksums[name] = first_digest
    return checksums


def _smoke_installed_wheel(wheel: Path, temporary_root: Path) -> None:
    installed = temporary_root / "installed"
    tripwire = temporary_root / "tripwire"
    tripwire.mkdir()
    (tripwire / "sitecustomize.py").write_text(
        "import socket\n"
        "def _reject(*args, **kwargs):\n"
        "    raise RuntimeError('distribution smoke test forbids network access')\n"
        "for _method in ('connect', 'connect_ex', 'sendto', 'sendmsg'):\n"
        "    if hasattr(socket.socket, _method):\n"
        "        setattr(socket.socket, _method, _reject)\n"
        "socket.create_connection = _reject\n"
        "for _resolver in "
        "('getaddrinfo', 'gethostbyaddr', 'gethostbyname', 'gethostbyname_ex', 'getnameinfo'):\n"
        "    if hasattr(socket, _resolver):\n"
        "        setattr(socket, _resolver, _reject)\n",
        encoding="utf-8",
    )
    _run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(installed),
        str(wheel),
        cwd=temporary_root,
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join((str(tripwire), str(installed))),
        "TRANSFORMERS_OFFLINE": "1",
    }
    script = """
import json
import pathlib
import subprocess
import sys

import yaml
import ste_compiler

installed, output = map(pathlib.Path, sys.argv[1:])
package_root = pathlib.Path(ste_compiler.__file__).parent.resolve()
assert package_root.is_relative_to(installed.resolve())

manifest = yaml.safe_load(
    (package_root / "examples/manifest.yaml").read_text(encoding="utf-8")
)
assert manifest["schema_version"] == "ste-executable-examples-v1"
assert manifest["distribution"]["wheel_fixture_base"] == "ste_compiler"
assert manifest["distribution"]["portable_execution"] == ["core-ci"]
assert [scenario["id"] for scenario in manifest["scenarios"]] == list(range(1, 14))
network_probe = subprocess.run(
    [sys.executable, "-c", "import socket; socket.gethostbyname('example.invalid')"],
    capture_output=True,
    text=True,
)
assert network_probe.returncode != 0
assert "distribution smoke test forbids network access" in network_probe.stderr
catalog_result = subprocess.run(
    [sys.executable, "-m", "ste_compiler.examples.catalog_runner", str(output)],
    check=True,
    capture_output=True,
    text=True,
)
catalog_payload = json.loads(catalog_result.stdout)
portable = set(manifest["distribution"]["portable_execution"])
portable_scenarios = [
    scenario for scenario in manifest["scenarios"] if scenario["execution"] in portable
]
assert catalog_payload["execution"] == manifest["distribution"]["portable_execution"]
assert catalog_payload["scenario_ids"] == [
    scenario["id"] for scenario in portable_scenarios
]
assert catalog_payload["command_count"] == sum(
    len(scenario["commands"]) for scenario in portable_scenarios
)
"""
    _run(
        sys.executable,
        "-c",
        script,
        str(installed),
        str(temporary_root / "installed-catalog-output"),
        cwd=temporary_root,
        env=environment,
    )


def main() -> None:
    environment = {
        **os.environ,
        "SOURCE_DATE_EPOCH": _source_date_epoch(),
    }
    with tempfile.TemporaryDirectory(prefix="ste-compiler-distribution-") as directory:
        temporary_root = Path(directory)
        first = _build(temporary_root / "first", environment)
        second = _build(temporary_root / "second", environment)
        _inspect(*first)
        checksums = _assert_reproducible(first, second)
        sdist_wheel = _build_wheel_from_sdist(first[1], temporary_root, environment)
        if _sha256(sdist_wheel) != _sha256(first[0]):
            raise RuntimeError("wheel rebuilt from sdist differs from the direct wheel")
        _smoke_installed_wheel(sdist_wheel, temporary_root)
    print(json.dumps({"distributions": checksums, "status": "verified"}, sort_keys=True))


if __name__ == "__main__":
    main()

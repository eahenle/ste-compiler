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
    "/scripts/ci/distribution_smoke.py",
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
        "socket.socket.connect = _reject\n",
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

import ste_compiler

installed, output = map(pathlib.Path, sys.argv[1:])
assert pathlib.Path(ste_compiler.__file__).is_relative_to(installed)
command = [sys.executable, "-m", "ste_compiler.cli"]
help_result = subprocess.run(
    [*command, "--help"], check=True, capture_output=True, text=True
)
assert "compile-source" in help_result.stdout
demo = subprocess.run(
    [*command, "demo", "--json"], check=True, capture_output=True, text=True
)
assert json.loads(demo.stdout)["validation"]["status"] == "accepted"
subprocess.run(
    [*command, "build-demonstration-corpus", "--version", "2", "--output", str(output)],
    check=True,
)
verified = subprocess.run(
    [*command, "verify-demonstration-corpus", str(output)],
    check=True,
    capture_output=True,
    text=True,
)
assert verified.stdout.startswith("Verified 24 records")
"""
    _run(
        sys.executable,
        "-c",
        script,
        str(installed),
        str(temporary_root / "corpus-v2"),
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
        _smoke_installed_wheel(first[0], temporary_root)
    print(json.dumps({"distributions": checksums, "status": "verified"}, sort_keys=True))


if __name__ == "__main__":
    main()

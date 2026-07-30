from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.ci import distribution_smoke


def test_source_date_epoch_uses_selected_source_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "reviewed-source"
    source_root.mkdir()
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(*command: str, cwd: Path, env=None) -> str:
        del env
        calls.append((command, cwd))
        return "1729\n"

    monkeypatch.setattr(distribution_smoke, "_run", fake_run)

    assert distribution_smoke._source_date_epoch(source_root) == "1729"
    assert calls == [
        (("git", "show", "-s", "--format=%ct", "HEAD"), source_root),
    ]


def test_build_uses_selected_source_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "reviewed-source"
    source_root.mkdir()
    output = tmp_path / "dist"
    environment = {"SOURCE_DATE_EPOCH": "1729"}
    calls: list[tuple[tuple[str, ...], Path, dict[str, str] | None]] = []

    def fake_run(
        *command: str,
        cwd: Path = distribution_smoke.ROOT,
        env: dict[str, str] | None = None,
    ) -> str:
        calls.append((command, cwd, env))
        output.mkdir()
        (output / "ste_compiler-0.1.0-py3-none-any.whl").touch()
        (output / "ste_compiler-0.1.0.tar.gz").touch()
        return ""

    monkeypatch.setattr(distribution_smoke, "_run", fake_run)

    wheel, sdist = distribution_smoke._build(output, environment, source_root)

    assert wheel.name == "ste_compiler-0.1.0-py3-none-any.whl"
    assert sdist.name == "ste_compiler-0.1.0.tar.gz"
    assert calls == [
        (
            (
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--sdist",
                "--wheel",
                "--outdir",
                str(output),
                str(source_root),
            ),
            distribution_smoke.ROOT,
            environment,
        )
    ]


def test_main_uses_trusted_script_with_selected_source_root(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "reviewed-source"
    source_root.mkdir()
    marker = tmp_path / "untrusted-script-ran"
    untrusted_script = source_root / "scripts/ci/distribution_smoke.py"
    untrusted_script.parent.mkdir(parents=True)
    untrusted_script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    source_roots: list[Path] = []
    epochs: list[Path] = []

    def fake_epoch(selected: Path) -> str:
        epochs.append(selected)
        return "1729"

    def fake_build(
        output: Path,
        environment: dict[str, str],
        selected: Path,
    ) -> tuple[Path, Path]:
        assert environment["SOURCE_DATE_EPOCH"] == "1729"
        source_roots.append(selected)
        return output / "package.whl", output / "package.tar.gz"

    monkeypatch.setattr(distribution_smoke, "_source_date_epoch", fake_epoch)
    monkeypatch.setattr(distribution_smoke, "_build", fake_build)
    monkeypatch.setattr(distribution_smoke, "_inspect", lambda *_args: None)
    monkeypatch.setattr(
        distribution_smoke,
        "_assert_reproducible",
        lambda _first, _second: {"package.whl": "digest"},
    )
    monkeypatch.setattr(
        distribution_smoke,
        "_build_wheel_from_sdist",
        lambda _sdist, temporary_root, _environment: temporary_root / "sdist.whl",
    )
    monkeypatch.setattr(distribution_smoke, "_sha256", lambda _path: "digest")
    monkeypatch.setattr(distribution_smoke, "_smoke_installed_wheel", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["distribution_smoke.py", "--source-root", str(source_root)],
    )

    distribution_smoke.main()

    assert epochs == [source_root.resolve()]
    assert source_roots == [source_root.resolve(), source_root.resolve()]
    assert not marker.exists()
    assert '"status": "verified"' in capsys.readouterr().out


def test_parser_preserves_trusted_checkout_as_default() -> None:
    args = distribution_smoke._parser().parse_args([])

    assert args.source_root == distribution_smoke.ROOT


def test_installed_catalog_failure_surfaces_nested_stdout_and_stderr(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = (distribution_smoke.ROOT / "examples/manifest.yaml").read_text(encoding="utf-8")

    def fake_run(
        *command: str,
        cwd: Path = distribution_smoke.ROOT,
        env: dict[str, str] | None = None,
    ) -> str:
        del cwd, env
        if command[1:4] == ("-m", "pip", "install"):
            package_root = tmp_path / "installed/ste_compiler/examples"
            package_root.mkdir(parents=True)
            (package_root / "manifest.yaml").write_text(manifest, encoding="utf-8")
            return ""
        if command[1:3] == ("-m", "ste_compiler.examples.catalog_runner"):
            raise RuntimeError(
                "command failed\nstdout:\nnested catalog stdout\nstderr:\nnested catalog stderr"
            )
        if command[1] == "-c":
            return ""
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(distribution_smoke, "_run", fake_run)

    with pytest.raises(
        RuntimeError,
        match=r"nested catalog stdout[\s\S]*nested catalog stderr",
    ):
        distribution_smoke._smoke_installed_wheel(tmp_path / "package.whl", tmp_path)

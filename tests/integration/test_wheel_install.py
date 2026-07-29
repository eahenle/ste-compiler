import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_installed_wheel_contains_default_cli_data(tmp_path):
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("ste_compiler-*.whl"))

    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    clean_env["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

original_import = builtins.__import__

def import_without_fcntl(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fcntl":
        raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_fcntl
from ste_compiler.cli import app
from typer.testing import CliRunner

result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
""",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    command = [sys.executable, "-m", "ste_compiler.cli"]
    public_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ste_compiler.realizer import "
                "DecoderOnlyLoRAConfig, DecoderOnlyLoRAError, "
                "DecoderOnlyLoRASymbolGenerator, EncoderDecoderConfig, "
                "EncoderDecoderError, TransformersEncoderDecoderSymbolGenerator; "
                "assert DecoderOnlyLoRAConfig and DecoderOnlyLoRAError "
                "and DecoderOnlyLoRASymbolGenerator and EncoderDecoderConfig "
                "and EncoderDecoderError and TransformersEncoderDecoderSymbolGenerator"
            ),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not public_import.stderr

    demo = subprocess.run(
        [*command, "demo", "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    demo_payload = json.loads(demo.stdout)
    assert demo_payload["schema_version"] == "compile-source-v1"
    assert demo_payload["metadata"]["frontend"] == "offline-replay"
    assert demo_payload["validation"]["status"] == "accepted"

    realized = subprocess.run(
        [*command, "realize", str(ROOT / "data/examples/negative.yaml")],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert realized.stdout == "Do not open the shutoff valve.\n"

    planned = subprocess.run(
        [*command, "plan-symbols", str(ROOT / "data/examples/warning_pressure.yaml"), "--json"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    training_record = json.loads(planned.stdout)
    assert "WORD_occur" in training_record["allowed_symbols"]
    assert "TERM_hydraulic_pressure|hydraulic%20pressure" in training_record["symbols"]

    corpus = tmp_path / "training"
    subprocess.run(
        [
            *command,
            "export-symbolic-corpus",
            str(ROOT / "data/examples"),
            "--output",
            str(corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((corpus / "current" / "manifest.json").read_text())["record_count"] == 5
    assert len((corpus / "current" / "corpus.jsonl").read_text().splitlines()) == 5

    demonstration_corpus = tmp_path / "demonstration-corpus"
    subprocess.run(
        [
            *command,
            "build-demonstration-corpus",
            "--output",
            str(demonstration_corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    release_manifest = json.loads((demonstration_corpus / "manifest.json").read_text())
    assert release_manifest["record_count"] == 12
    assert (demonstration_corpus / "terminology.json").is_file()
    assert (demonstration_corpus / "vocabulary.json").is_file()
    subprocess.run(
        [
            *command,
            "verify-demonstration-corpus",
            str(demonstration_corpus),
        ],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )

    reports = tmp_path / "reports"
    subprocess.run(
        [*command, "evaluate", "--output", str(reports)],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (reports / "report.json").is_file()

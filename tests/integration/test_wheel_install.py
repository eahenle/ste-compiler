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
    command = [sys.executable, "-m", "ste_compiler.cli"]
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
    assert json.loads((corpus / "manifest.json").read_text())["record_count"] == 5
    assert len((corpus / "corpus.jsonl").read_text().splitlines()) == 5

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

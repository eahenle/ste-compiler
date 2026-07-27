from pathlib import Path

from typer.testing import CliRunner

from ste_compiler.cli import app

ROOT = Path(__file__).parents[2]
runner = CliRunner()


def test_cli_realize_and_validate():
    result = runner.invoke(app, ["realize", str(ROOT / "data/examples/negative.yaml")])
    assert result.exit_code == 0
    assert result.stdout == "Do not open the shutoff valve.\n"
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(ROOT / "data/examples/invalid_semantic.txt"),
            "--ir",
            str(ROOT / "data/examples/negative.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "REQUIRED_NODE_OMITTED" in result.stdout
    assert "UNSUPPORTED_SEMANTIC_CHANGE" in result.stdout


def test_validate_text_does_not_inherit_expected_semantics(tmp_path):
    submitted = tmp_path / "submitted.txt"
    submitted.write_text("Open the shutoff valve.\n")
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(submitted),
            "--ir",
            str(ROOT / "data/examples/installation.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert '"status": "rejected"' in result.stdout
    assert "UNSUPPORTED_SEMANTIC_CHANGE" in result.stdout


def test_validate_text_accepts_exact_controlled_realization(tmp_path):
    submitted = tmp_path / "submitted.txt"
    submitted.write_text("Install the access panel.\n")
    result = runner.invoke(
        app,
        [
            "validate-text",
            str(submitted),
            "--ir",
            str(ROOT / "data/examples/installation.yaml"),
        ],
    )
    assert result.exit_code == 0
    assert result.stdout == "accepted\n"


def test_cli_critical_failure_and_glossary():
    bad = runner.invoke(
        app, ["validate-text", str(ROOT / "data/examples/invalid_unauthorized.txt")]
    )
    assert bad.exit_code == 1
    good = runner.invoke(app, ["glossary", "check", str(ROOT / "data/demo_terminology.yaml")])
    assert good.exit_code == 0


def test_evaluation_reports(tmp_path):
    result = runner.invoke(
        app, ["evaluate", str(ROOT / "data/evaluation"), "--output", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()

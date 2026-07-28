import json
from pathlib import Path

from typer.testing import CliRunner

from ste_compiler.cli import app
from ste_compiler.ir.models import Quantity
from ste_compiler.ir.serialization import dumps_document, load_document

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


def test_cli_exports_symbolic_training_record():
    result = runner.invoke(
        app,
        [
            "plan-symbols",
            str(ROOT / "data/examples/warning_pressure.yaml"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["document_id"] == "warning_pressure"
    assert "TERM_hydraulic_pressure" in record["allowed_symbols"]
    assert "NUMBER_20" in record["symbols"]
    assert json.loads(record["serialized_ir"])["id"] == "warning_pressure"


def test_cli_training_plan_preserves_negative_quantity_symbol(tmp_path):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    instruction = document.sections[0].statements[0]
    negative_quantity = Quantity(value=-20, unit="MPa", comparator="more_than")
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                instruction.quantity_constraints[0].model_copy(
                    update={"quantity": negative_quantity}
                )
            ]
        }
    )
    source = tmp_path / "negative_quantity.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert "NUMBER_-20" in record["allowed_symbols"]
    assert "NUMBER_-20" in record["symbols"].split()
    assert "PUNCT_U002D" not in record["symbols"].split()


def test_cli_training_plan_round_trips_quoted_manner(tmp_path):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": '"safe"'})
    source = tmp_path / "quoted_manner.json"
    source.write_text(dumps_document(document, as_json=True))

    result = runner.invoke(app, ["plan-symbols", str(source), "--json"])

    assert result.exit_code == 0
    record = json.loads(result.stdout)
    assert record["text"] == 'Install the access panel "safe".'
    assert record["symbols"].endswith("TERM_access_panel PUNCT_U0022 WORD_safe PUNCT_U0022 PERIOD")
    assert "PUNCT_U0022" in record["allowed_symbols"]


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

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from ste_compiler.diagnostics import ValidationReport
from ste_compiler.evaluation import evaluate as run_evaluation
from ste_compiler.evaluation import write_reports
from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.training import (
    TrainingRecordValidationError,
    build_training_record,
    export_symbolic_corpus,
)
from ste_compiler.validators import LexicalValidator, ValidationPipeline, align_controlled_text

app = typer.Typer(help="Compile semantic IR to auditable STE-inspired controlled English.")
glossary_app = typer.Typer()
app.add_typer(glossary_app, name="glossary")
ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DATA = Path(__file__).with_name("data")
DATA_ROOT = PACKAGE_DATA if PACKAGE_DATA.is_dir() else ROOT / "data"
CONTROLLED_INPUT_ERRORS = (KeyError, ValidationError, ValueError)


def resources(
    vocabulary: Path | None = None, terminology: Path | None = None
) -> tuple[Vocabulary, TerminologyRegistry]:
    return (
        Vocabulary.load(vocabulary or DATA_ROOT / "demo_vocabulary.yaml"),
        TerminologyRegistry.load(terminology or DATA_ROOT / "demo_terminology.yaml"),
    )


def emit_report(report: ValidationReport, json_output: bool) -> None:
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    elif report.violations:
        for item in report.violations:
            typer.echo(f"{item.severity.value.upper()} {item.code}: {item.message}")
    else:
        typer.echo("accepted")


@app.command("validate-ir")
def validate_ir(path: Path) -> None:
    try:
        load_document(path)
    except (ValidationError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo("valid")


@app.command()
def realize(path: Path, metadata: bool = typer.Option(False, help="Print JSON mappings.")) -> None:
    doc = load_document(path)
    vocab, terms = resources()
    result = DeterministicRealizer().realize(doc, vocab, terms)
    typer.echo(result.text)
    if metadata:
        typer.echo(
            json.dumps(
                {"mappings": [vars(m) for m in result.mappings], "metadata": result.metadata},
                indent=2,
            )
        )


@app.command("plan-symbols")
def plan_symbols(
    path: Path, json_output: bool = typer.Option(False, "--json", help="Print a training record.")
) -> None:
    """Create the deterministic symbolic target for one IR document."""

    try:
        doc = load_document(path)
        vocab, terms = resources()
        record = build_training_record(doc, vocab, terms)
    except TrainingRecordValidationError as error:
        emit_report(error.report, json_output)
        raise typer.Exit(1) from error
    except CONTROLLED_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(json.dumps(record, indent=2))
    else:
        typer.echo(record["symbols"])


@app.command("export-symbolic-corpus")
def export_corpus(
    source: Annotated[Path, typer.Argument(help="IR file or directory.")],
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("training-corpus"),
) -> None:
    """Write deterministic JSONL training records and a SHA-256 manifest."""

    vocab, terms = resources()
    try:
        manifest = export_symbolic_corpus(source, output, vocab, terms)
    except CONTROLLED_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Wrote {manifest['record_count']} records to {output / 'corpus.jsonl'} "
        f"(sha256: {manifest['corpus_sha256']})"
    )


@app.command("validate-text")
def validate_text(
    path: Path, ir: Path | None = None, json_output: bool = typer.Option(False, "--json")
) -> None:
    text = path.read_text()
    vocab, terms = resources()
    document = load_document(ir) if ir else None
    realization = None
    if document:
        expected = DeterministicRealizer().realize(document, vocab, terms)
        realization = align_controlled_text(text, expected)
    report = ValidationPipeline(LexicalValidator(vocab, terms)).validate(
        text, document, realization
    )
    emit_report(report, json_output)
    if report.status == "rejected":
        raise typer.Exit(1)


@app.command()
def compile(
    source: Path, frontend: str = "manual", json_output: bool = typer.Option(False, "--json")
) -> None:
    if frontend != "manual":
        raise typer.BadParameter("Only the offline manual frontend is configured.")
    doc = load_document(source)
    vocab, terms = resources()
    result = DeterministicRealizer().realize(doc, vocab, terms)
    report = ValidationPipeline(LexicalValidator(vocab, terms)).validate(result.text, doc, result)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "text": result.text,
                    "mappings": [vars(m) for m in result.mappings],
                    "validation": report.model_dump(mode="json"),
                    "metadata": result.metadata,
                },
                indent=2,
            )
        )
    else:
        typer.echo(result.text)
        emit_report(report, False)
    if report.status == "rejected":
        raise typer.Exit(1)


@app.command()
def evaluate(
    corpus: Annotated[Path, typer.Argument()] = DATA_ROOT / "evaluation",
    output: Annotated[Path, typer.Option()] = Path("reports"),
) -> None:
    vocab, terms = resources()
    results = run_evaluation(corpus, vocab, terms)
    write_reports(results, output)
    typer.echo(f"Wrote {output / 'report.json'} and {output / 'report.md'}")


@glossary_app.command("check")
def glossary_check(path: Path) -> None:
    try:
        registry = TerminologyRegistry.load(path)
        [registry.get(t.id) for t in registry.data.terms if t.status == "approved"]
    except (ValidationError, ValueError, KeyError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"valid: {len(registry.data.terms)} terms, version {registry.data.version}")


if __name__ == "__main__":
    app()

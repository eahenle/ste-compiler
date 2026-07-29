import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from ste_compiler.diagnostics import ValidationReport
from ste_compiler.evaluation import evaluate as run_evaluation
from ste_compiler.evaluation import write_reports
from ste_compiler.frontend import LLMFrontend, ReplayIRProvider
from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.base import RealizationResult
from ste_compiler.results import CompileSourceResult
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
SOURCE_INPUT_ERRORS = (*CONTROLLED_INPUT_ERRORS, OSError)
schema_app = typer.Typer(help="Print versioned machine-readable result schemas.")
app.add_typer(schema_app, name="schema")


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


def compile_document(
    document: Document,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> tuple[RealizationResult, ValidationReport]:
    result = DeterministicRealizer().realize(document, vocabulary, terminology)
    report = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        result.text,
        document,
        result,
    )
    return result, report


def compiled_payload(
    document: Document,
    result: RealizationResult,
    report: ValidationReport,
) -> dict[str, object]:
    return {
        "text": result.text,
        "mappings": [vars(mapping) for mapping in result.mappings],
        "validation": report.model_dump(mode="json"),
        "metadata": result.metadata,
        "ir": document.model_dump(mode="json"),
    }


def compile_replayed_source(
    source: Path,
    ir_fixture: Path,
    source_id: str,
) -> tuple[str, Document, RealizationResult, ValidationReport]:
    source_text = source.read_text(encoding="utf-8")
    provider = ReplayIRProvider.from_path(ir_fixture)
    document = LLMFrontend(provider).parse(source_text, source_id=source_id)
    vocabulary, terminology = resources()
    result, report = compile_document(document, vocabulary, terminology)
    return source_text, document, result, report


def emit_source_compilation(
    source_text: str,
    source_id: str,
    document: Document,
    result: RealizationResult,
    report: ValidationReport,
    json_output: bool,
) -> None:
    if json_output:
        payload = CompileSourceResult.from_compilation(
            source_text=source_text,
            source_id=source_id,
            document=document,
            realization=result,
            validation=report,
        )
        typer.echo(payload.model_dump_json(indent=2))
    else:
        typer.echo(result.text)
        emit_report(report, False)
    if report.status == "rejected":
        raise typer.Exit(1)


def run_replay_compilation(
    source: Path,
    ir_fixture: Path,
    source_id: str,
    json_output: bool,
) -> None:
    try:
        source_text, document, result, report = compile_replayed_source(
            source,
            ir_fixture,
            source_id,
        )
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    emit_source_compilation(
        source_text,
        source_id,
        document,
        result,
        report,
        json_output,
    )


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
        f"Wrote {manifest['record_count']} records to {output / 'current' / 'corpus.jsonl'} "
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
    result, report = compile_document(doc, vocab, terms)
    if json_output:
        payload = compiled_payload(doc, result, report)
        payload.pop("ir")
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(result.text)
        emit_report(report, False)
    if report.status == "rejected":
        raise typer.Exit(1)


@app.command("compile-source")
def compile_source(
    source: Annotated[Path, typer.Argument(help="Raw UTF-8 technical source.")],
    ir_fixture: Annotated[
        Path,
        typer.Option(
            "--ir-fixture",
            help="Gold YAML or JSON IR proposal to replay through the frontend boundary.",
        ),
    ],
    frontend: Annotated[str, typer.Option(help="Source frontend implementation.")] = "replay",
    realizer: Annotated[str, typer.Option(help="Realization implementation.")] = "deterministic",
    source_id: Annotated[
        str | None,
        typer.Option(help="Expected source_span source_id; defaults to the source filename."),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compile raw source through a reproducible offline end-to-end workflow."""

    if frontend != "replay":
        raise typer.BadParameter("Only the offline replay source frontend is configured.")
    if realizer != "deterministic":
        raise typer.BadParameter(
            "Only the deterministic realizer is configured for compile-source."
        )
    expected_source_id = source_id or source.name
    run_replay_compilation(source, ir_fixture, expected_source_id, json_output)


@app.command()
def demo(json_output: bool = typer.Option(False, "--json")) -> None:
    """Run the packaged, credential-free raw-source reference workflow."""

    example_root = DATA_ROOT / "end_to_end"
    run_replay_compilation(
        example_root / "hydraulic_warning.txt",
        example_root / "hydraulic_warning.ir.yaml",
        "hydraulic_warning.txt",
        json_output,
    )


@schema_app.command("compile-source")
def compile_source_schema() -> None:
    """Print the JSON Schema for `compile-source --json` and `demo --json`."""

    typer.echo(json.dumps(CompileSourceResult.model_json_schema(), indent=2))


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

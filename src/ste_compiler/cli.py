from pathlib import Path
import json
import typer
from pydantic import ValidationError
from ste_compiler.diagnostics import ValidationReport
from ste_compiler.evaluation import evaluate as run_evaluation, write_reports
from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.base import RealizationResult, SentenceMapping
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators import LexicalValidator, ValidationPipeline

app = typer.Typer(help="Compile semantic IR to auditable STE-inspired controlled English.")
glossary_app = typer.Typer()
app.add_typer(glossary_app, name="glossary")
ROOT = Path(__file__).resolve().parents[2]


def resources(
    vocabulary: Path | None = None, terminology: Path | None = None
) -> tuple[Vocabulary, TerminologyRegistry]:
    return (
        Vocabulary.load(vocabulary or ROOT / "data/demo_vocabulary.yaml"),
        TerminologyRegistry.load(terminology or ROOT / "data/demo_terminology.yaml"),
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
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        mappings = tuple(
            SentenceMapping(
                m.sentence, sentences[i] if i < len(sentences) else "", m.ir_node_ids, m.features
            )
            for i, m in enumerate(expected.mappings)
        )
        realization = RealizationResult(text, mappings, expected.metadata)
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
    corpus: Path = typer.Argument(ROOT / "data/evaluation"),
    output: Path = typer.Option(ROOT / "reports"),
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

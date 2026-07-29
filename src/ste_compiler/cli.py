import hashlib
import json
import shlex
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError

from ste_compiler.artifacts import (
    ArtifactBundleManifestV1,
    ArtifactPreflightResultV1,
    ArtifactValidationProfile,
    read_artifact_manifest_for_routing,
)
from ste_compiler.diagnostics import ValidationReport
from ste_compiler.evaluation import evaluate as run_evaluation
from ste_compiler.evaluation import write_reports
from ste_compiler.frontend import LLMFrontend, ManualFrontend, ReplayIRProvider
from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.base import RealizationResult, Realizer
from ste_compiler.realizer.config import (
    REALIZER_CONFIG_ADAPTER,
    load_realizer_config,
    realizer_config_sha256,
)
from ste_compiler.realizer.decoder_lora import DecoderOnlyLoRAError
from ste_compiler.realizer.encoder_decoder import EncoderDecoderError
from ste_compiler.realizer.factory import build_realizer
from ste_compiler.realizer.neural import NeuralRealizerUnavailable
from ste_compiler.reference_release import (
    ReferencePredictionV1,
    ReferenceReleaseManifestV1,
    ReferenceReleaseMetadataV1,
    build_reference_release,
    load_reference_release_metadata,
    verify_reference_release,
)
from ste_compiler.results import CompileSourceResult
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.training import (
    DecoderLoRATrainingError,
    DecoderOnlyLoRATrainingConfigV1,
    EncoderDecoderTrainingConfigV1,
    TrainingRecordValidationError,
    TrainingReleaseSnapshot,
    build_demonstration_corpus,
    build_training_record,
    canonical_run_manifest_json,
    evaluate_decoder_lora_adapter,
    evaluate_encoder_decoder_checkpoint,
    export_symbolic_corpus,
    load_training_config,
    model_snapshot_manifest_sha256,
    preflight_decoder_lora_artifact_bundle,
    preflight_encoder_decoder_artifact_bundle,
    prepare_decoder_smoke_fixture,
    read_training_release,
    run_decoder_lora_training_bundle,
    run_encoder_decoder_training_bundle,
    training_config_sha256,
    verify_demonstration_corpus,
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
REALIZER_INPUT_ERRORS = (
    *SOURCE_INPUT_ERRORS,
    DecoderOnlyLoRAError,
    EncoderDecoderError,
    NeuralRealizerUnavailable,
)
TRAINING_INPUT_ERRORS = (
    *SOURCE_INPUT_ERRORS,
    DecoderLoRATrainingError,
    NeuralRealizerUnavailable,
)
REFERENCE_RELEASE_INPUT_ERRORS = (
    *TRAINING_INPUT_ERRORS,
    DecoderOnlyLoRAError,
    EncoderDecoderError,
)
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
    *,
    frontend: str,
    frontend_version: str,
    realizer: Realizer | None = None,
) -> tuple[Document, RealizationResult, ValidationReport]:
    selected_realizer = realizer or DeterministicRealizer()
    metadata = document.metadata.model_copy(
        update={
            "frontend": frontend,
            "frontend_version": frontend_version,
            "realizer": "deterministic",
            "realizer_version": DeterministicRealizer.version,
            "vocabulary_version": vocabulary.data.version,
            "terminology_version": terminology.data.version,
            "validator_profile": ValidationPipeline.profile,
        }
    )
    document = document.model_copy(update={"metadata": metadata})
    result = selected_realizer.realize(document, vocabulary, terminology)
    result_realizer = result.metadata.get("realizer")
    result_realizer_version = result.metadata.get("realizer_version")
    if result_realizer is None or result_realizer_version is None:
        raise ValueError("realizer result must identify its implementation and version")
    metadata = document.metadata.model_copy(
        update={
            "realizer": result_realizer,
            "realizer_version": result_realizer_version,
        }
    )
    document = document.model_copy(update={"metadata": metadata})
    report = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        result.text,
        document,
        result,
    )
    return document, result, report


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
    *,
    realizer: Realizer | None = None,
) -> tuple[bytes, str, Document, RealizationResult, ValidationReport]:
    source_bytes = source.read_bytes()
    source_text = source_bytes.decode("utf-8")
    provider = ReplayIRProvider.from_path(ir_fixture)
    document = LLMFrontend(provider).parse(source_text, source_id=source_id)
    vocabulary, terminology = resources()
    document, result, report = compile_document(
        document,
        vocabulary,
        terminology,
        frontend=document.metadata.frontend,
        frontend_version=document.metadata.frontend_version,
        realizer=realizer,
    )
    return source_bytes, source_text, document, result, report


def configured_realizer(
    realizer_config: Path | None,
    *,
    artifact_bundle: Path | None = None,
    model_snapshot: Path | None = None,
) -> Realizer | None:
    """Load one portable config and bind any untrusted local artifact locators."""

    if realizer_config is None:
        if artifact_bundle is not None or model_snapshot is not None:
            raise ValueError("--artifact-bundle and --model-snapshot require --realizer-config")
        return None
    config = load_realizer_config(realizer_config)
    if artifact_bundle is None and model_snapshot is None:
        return build_realizer(config)
    return build_realizer(
        config,
        artifact_bundle=artifact_bundle,
        model_snapshot=model_snapshot,
    )


def emit_source_compilation(
    source_bytes: bytes,
    source_id: str,
    document: Document,
    result: RealizationResult,
    report: ValidationReport,
    json_output: bool,
) -> None:
    if json_output:
        payload = CompileSourceResult.from_compilation(
            source_bytes=source_bytes,
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
    *,
    realizer_config: Path | None = None,
    artifact_bundle: Path | None = None,
    model_snapshot: Path | None = None,
) -> None:
    try:
        selected_realizer = configured_realizer(
            realizer_config,
            artifact_bundle=artifact_bundle,
            model_snapshot=model_snapshot,
        )
        source_bytes, _, document, result, report = compile_replayed_source(
            source,
            ir_fixture,
            source_id,
            realizer=selected_realizer,
        )
    except REALIZER_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    emit_source_compilation(
        source_bytes,
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


@app.command("build-demonstration-corpus")
def build_corpus_release(
    output: Annotated[
        Path | None,
        typer.Option(help="Empty output directory for the reconstructed release."),
    ] = None,
    version: Annotated[
        Literal["1", "2"],
        typer.Option(help="Versioned demonstration-corpus construction to build."),
    ] = "1",
) -> None:
    """Reconstruct the licensed, split demonstration dataset."""

    corpus_root = DATA_ROOT / f"demonstration_corpus/v{version}"
    construction = corpus_root / "source-construction.json"
    terminology_path = corpus_root / "terminology.yaml"
    destination = output or Path(f"demonstration-corpus-v{version}")
    vocab, terms = resources(terminology=terminology_path)
    try:
        manifest = build_demonstration_corpus(construction, destination, vocab, terms)
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Wrote {manifest['record_count']} records to {destination} "
        f"(construction sha256: {manifest['construction_sha256']})"
    )


@app.command("verify-demonstration-corpus")
def verify_corpus_release(release: Path) -> None:
    """Reconstruct and verify every byte of a demonstration dataset release."""

    try:
        manifest = verify_demonstration_corpus(release)
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Verified {manifest['record_count']} records in {release} "
        f"(construction sha256: {manifest['construction_sha256']})"
    )


@app.command("validate-training-config")
def validate_training_config(
    path: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate and identify one versioned neural-training configuration."""

    try:
        config = load_training_config(path)
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": config.schema_version,
        "architecture": config.architecture,
        "config_sha256": training_config_sha256(config),
        "dataset_version": config.corpus.dataset_version,
        "manifest_sha256": config.corpus.manifest_sha256,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Validated {config.architecture} training config (sha256: {payload['config_sha256']})"
        )


@app.command("validate-realizer-config")
def validate_realizer_configuration(
    path: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate and identify one versioned offline realizer configuration."""

    try:
        config = load_realizer_config(path)
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": config.schema_version,
        "architecture": config.architecture,
        "config_sha256": realizer_config_sha256(config),
        "artifact_mode": (
            "content-addressed-local-bundle"
            if config.architecture.endswith("-local-bundle")
            else "offline-cache-only"
        ),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Validated {config.architecture} realizer config (sha256: {payload['config_sha256']})"
        )


@app.command("verify-training-release")
def verify_training_release(
    config_path: Path,
    release: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Read one exact release through the immutable training-data boundary."""

    try:
        config = load_training_config(config_path)
        snapshot = read_training_release(release, config.corpus)
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": snapshot.manifest.schema_version,
        "dataset_version": snapshot.manifest.dataset_version,
        "manifest_sha256": snapshot.manifest_sha256,
        "split_counts": dict(snapshot.manifest.split_counts),
        "symbol_count": len(snapshot.symbol_inventory),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Verified training release {snapshot.manifest.dataset_version} "
            f"({snapshot.manifest.record_count} records, "
            f"manifest sha256: {snapshot.manifest_sha256})"
        )


def _decoder_training_inputs(
    config_path: Path,
    release_path: Path,
) -> tuple[DecoderOnlyLoRATrainingConfigV1, TrainingReleaseSnapshot]:
    config = load_training_config(config_path)
    if not isinstance(config, DecoderOnlyLoRATrainingConfigV1):
        raise DecoderLoRATrainingError(
            "training configuration architecture must be decoder-only-lora"
        )
    return config, read_training_release(release_path, config.corpus)


@app.command("prepare-decoder-smoke-fixture")
def prepare_decoder_fixture(
    config_path: Path,
    release: Path,
    output: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Create a tiny safe local causal-LM/tokenizer fixture for offline smoke training."""

    try:
        config, snapshot = _decoder_training_inputs(config_path, release)
        manifest = prepare_decoder_smoke_fixture(config, snapshot, output)
    except TRAINING_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": manifest.schema_version,
        "fixture_profile": manifest.fixture_profile,
        "manifest_sha256": model_snapshot_manifest_sha256(output),
        "artifact_count": len(manifest.artifacts),
        "output": str(output),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Prepared decoder smoke fixture at {output}")


@app.command("train-decoder-lora")
def train_decoder_lora(
    config_path: Path,
    release: Path,
    model_snapshot: Path,
    model_snapshot_manifest_sha256: str,
    output: Path,
    source_checkout: Annotated[
        Path,
        typer.Option(
            "--source-checkout",
            help="Exact ste-compiler Git checkout used to derive commit and uv.lock provenance.",
        ),
    ],
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run deterministic offline decoder-only LoRA smoke training."""

    evaluation_command = shlex.join(
        (
            "ste-compiler",
            "evaluate-decoder-lora",
            str(config_path),
            str(release),
            str(model_snapshot),
            model_snapshot_manifest_sha256,
            str(output / "adapter"),
        )
    )
    try:
        config, snapshot = _decoder_training_inputs(config_path, release)
        bundle = run_decoder_lora_training_bundle(
            config,
            snapshot,
            model_snapshot,
            model_snapshot_manifest_sha256,
            output,
            source_checkout=source_checkout,
            evaluation_command=evaluation_command,
        )
        manifest = bundle.run_manifest
        artifact_manifest_digest = bundle.artifact_manifest_sha256
    except TRAINING_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": manifest.schema_version,
        "status": manifest.status,
        "optimizer_steps": manifest.optimizer_steps,
        "training_losses": manifest.training_losses,
        "validation_loss": manifest.validation_loss,
        "trainable_parameters": manifest.trainable_parameters,
        "artifact_manifest_sha256": artifact_manifest_digest,
        "output": str(output),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Completed {manifest.optimizer_steps} decoder LoRA optimizer steps at {output} "
            f"(artifact manifest sha256: {artifact_manifest_digest})"
        )


@app.command("evaluate-decoder-lora")
def evaluate_decoder_lora(
    config_path: Path,
    release: Path,
    model_snapshot: Path,
    model_snapshot_manifest_sha256: str,
    adapter: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Reload a safe decoder LoRA adapter and evaluate its validation loss."""

    try:
        config, snapshot = _decoder_training_inputs(config_path, release)
        validation_loss = evaluate_decoder_lora_adapter(
            config,
            snapshot,
            model_snapshot,
            model_snapshot_manifest_sha256,
            adapter,
        )
    except TRAINING_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": "ste-decoder-lora-evaluation-v1",
        "split": "validation",
        "validation_loss": validation_loss,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Decoder LoRA validation loss: {validation_loss}")


@app.command("train-encoder-decoder")
def train_encoder_decoder(
    config_path: Path,
    release: Path,
    output: Annotated[
        Path,
        typer.Option(help="New output directory for the atomic safe checkpoint."),
    ],
    source_root: Annotated[
        Path,
        typer.Option(help="Git checkout used to derive package commit and dirty state."),
    ] = Path("."),
    dependency_lock: Annotated[
        Path,
        typer.Option(help="Exact dependency lock file hashed into the run manifest."),
    ] = Path("uv.lock"),
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Optional local Hugging Face cache directory."),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run deterministic offline CPU encoder-decoder training."""

    try:
        config = load_training_config(config_path)
        if not isinstance(config, EncoderDecoderTrainingConfigV1):
            raise typer.BadParameter(
                "training config architecture must be encoder-decoder",
                param_hint="config_path",
            )
        bundle = run_encoder_decoder_training_bundle(
            config,
            release,
            output,
            source_root=source_root,
            dependency_lock=dependency_lock,
            cache_dir=cache_dir,
        )
        manifest = bundle.run_manifest
        run_manifest_sha256 = hashlib.sha256(canonical_run_manifest_json(manifest)).hexdigest()
        artifact_manifest_digest = bundle.artifact_manifest_sha256
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    if json_output:
        payload = manifest.model_dump(mode="json")
        payload["run_manifest_sha256"] = run_manifest_sha256
        payload["artifact_manifest_sha256"] = artifact_manifest_digest
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Wrote {manifest.optimizer_steps}-step encoder-decoder checkpoint to {output} "
            f"(config sha256: {manifest.training_config_sha256}; "
            f"run manifest sha256: {run_manifest_sha256}; "
            f"artifact manifest sha256: {artifact_manifest_digest})"
        )


@app.command("preflight-artifact")
def preflight_artifact(
    root: Path,
    manifest_sha256: Annotated[
        str,
        typer.Option(
            help="Externally retained SHA-256 of artifact-manifest.json.",
        ),
    ],
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Verify one exact local training-output bundle without network access."""

    try:
        manifest = read_artifact_manifest_for_routing(root, manifest_sha256)
        validation_profile: ArtifactValidationProfile
        if manifest.architecture == "encoder-decoder":
            preflight_encoder_decoder_artifact_bundle(root, manifest_sha256)
            validation_profile = "encoder-checkpoint-load-v1"
        else:
            preflight_decoder_lora_artifact_bundle(root, manifest_sha256)
            validation_profile = "decoder-adapter-structure-v1"
    except TRAINING_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    result = ArtifactPreflightResultV1(
        schema_version="ste-artifact-preflight-v1",
        status="verified",
        architecture=manifest.architecture,
        artifact_type=manifest.artifact_type,
        intended_use=manifest.intended_use,
        artifact_manifest_sha256=manifest_sha256,
        run_manifest_sha256=manifest.run_manifest_sha256,
        file_count=manifest.file_count,
        total_bytes=manifest.total_bytes,
        validation_profile=validation_profile,
        network_access="none",
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"Verified {result.artifact_type} bundle "
            f"(artifact manifest sha256: {result.artifact_manifest_sha256})"
        )


@app.command("build-reference-release")
def build_reference_artifact_release(
    metadata_path: Path,
    corpus_release: Path,
    encoder_bundle: Path,
    encoder_artifact_manifest_sha256: str,
    decoder_bundle: Path,
    decoder_artifact_manifest_sha256: str,
    decoder_model_snapshot: Path,
    decoder_model_snapshot_manifest_sha256: str,
    output: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Build a dual-architecture mechanics release through production local loaders."""

    try:
        result = build_reference_release(
            load_reference_release_metadata(metadata_path),
            corpus_release,
            encoder_bundle,
            encoder_artifact_manifest_sha256,
            decoder_bundle,
            decoder_artifact_manifest_sha256,
            decoder_model_snapshot,
            decoder_model_snapshot_manifest_sha256,
            output,
        )
    except REFERENCE_RELEASE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": result.manifest.schema_version,
        "release_id": result.manifest.release_id,
        "manifest_sha256": result.manifest_sha256,
        "corpus_manifest_sha256": result.manifest.corpus_manifest_sha256,
        "architectures": [item.architecture for item in result.manifest.architectures],
        "output": str(output),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Wrote reference release {result.manifest.release_id} to {output} "
            f"(manifest sha256: {result.manifest_sha256})"
        )


@app.command("verify-reference-release")
def verify_reference_artifact_release(
    release: Path,
    release_manifest_sha256: str,
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        help="Regenerate predictions and every metadata byte through both local loaders.",
    ),
    corpus_release: Annotated[Path | None, typer.Option()] = None,
    encoder_bundle: Annotated[Path | None, typer.Option()] = None,
    decoder_bundle: Annotated[Path | None, typer.Option()] = None,
    decoder_model_snapshot: Annotated[Path | None, typer.Option()] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Verify one externally identified reference release, optionally by regeneration."""

    try:
        verified = verify_reference_release(
            release,
            release_manifest_sha256,
            corpus_release=corpus_release,
            encoder_bundle=encoder_bundle,
            decoder_bundle=decoder_bundle,
            decoder_model_snapshot=decoder_model_snapshot,
            regenerate=regenerate,
        )
    except REFERENCE_RELEASE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    payload = {
        "schema_version": verified.manifest.schema_version,
        "release_id": verified.manifest.release_id,
        "manifest_sha256": verified.manifest_sha256,
        "corpus_manifest_sha256": verified.manifest.corpus_manifest_sha256,
        "architectures": [item.architecture for item in verified.manifest.architectures],
        "regenerated": regenerate,
        "status": "verified",
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Verified reference release {verified.manifest.release_id} "
            f"(manifest sha256: {verified.manifest_sha256})"
        )


@app.command("evaluate-encoder-decoder-checkpoint")
def evaluate_encoder_decoder(
    config_path: Path,
    release: Path,
    checkpoint: Path,
    run_manifest_sha256: Annotated[
        str,
        typer.Option(
            help="Externally retained SHA-256 of checkpoint/run-manifest.json.",
        ),
    ],
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Reload and score one safe checkpoint on the pinned validation split."""

    try:
        config = load_training_config(config_path)
        if not isinstance(config, EncoderDecoderTrainingConfigV1):
            raise typer.BadParameter(
                "training config architecture must be encoder-decoder",
                param_hint="config_path",
            )
        metrics = evaluate_encoder_decoder_checkpoint(
            config,
            release,
            checkpoint,
            run_manifest_sha256,
        )
    except SOURCE_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(metrics.model_dump_json(indent=2))
    else:
        typer.echo(f"Validation loss: {metrics.mean_loss:.6f} ({metrics.record_count} records)")


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
    source: Path,
    frontend: str = "manual",
    realizer_config: Annotated[
        Path | None,
        typer.Option(help="Strict offline realizer configuration."),
    ] = None,
    artifact_bundle: Annotated[
        Path | None,
        typer.Option(
            "--artifact-bundle",
            help="Untrusted local bundle locator; content identity comes from the realizer config.",
        ),
    ] = None,
    model_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--model-snapshot",
            help="Untrusted local decoder base-snapshot locator; identity comes from the config.",
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if frontend != "manual":
        raise typer.BadParameter("Only the offline manual frontend is configured.")
    try:
        selected_realizer = configured_realizer(
            realizer_config,
            artifact_bundle=artifact_bundle,
            model_snapshot=model_snapshot,
        )
        doc = load_document(source)
        vocab, terms = resources()
        doc, result, report = compile_document(
            doc,
            vocab,
            terms,
            frontend="manual",
            frontend_version=ManualFrontend.version,
            realizer=selected_realizer,
        )
    except REALIZER_INPUT_ERRORS as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
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
    realizer: Annotated[
        str | None,
        typer.Option(help="Legacy realization implementation selector."),
    ] = None,
    realizer_config: Annotated[
        Path | None,
        typer.Option(help="Strict offline realizer configuration."),
    ] = None,
    artifact_bundle: Annotated[
        Path | None,
        typer.Option(
            "--artifact-bundle",
            help="Untrusted local bundle locator; content identity comes from the realizer config.",
        ),
    ] = None,
    model_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--model-snapshot",
            help="Untrusted local decoder base-snapshot locator; identity comes from the config.",
        ),
    ] = None,
    source_id: Annotated[
        str | None,
        typer.Option(help="Expected source_span source_id; defaults to the source filename."),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compile raw source through a reproducible offline end-to-end workflow."""

    if frontend != "replay":
        raise typer.BadParameter("Only the offline replay source frontend is configured.")
    if realizer_config is not None and realizer is not None:
        raise typer.BadParameter("--realizer and --realizer-config cannot be combined.")
    if realizer not in {None, "deterministic"}:
        raise typer.BadParameter(
            "Only the deterministic realizer is configured for compile-source."
        )
    expected_source_id = source_id or source.name
    run_replay_compilation(
        source,
        ir_fixture,
        expected_source_id,
        json_output,
        realizer_config=realizer_config,
        artifact_bundle=artifact_bundle,
        model_snapshot=model_snapshot,
    )


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


@schema_app.command("realizer-config")
def realizer_config_schema() -> None:
    """Print the JSON Schema for strict versioned realizer configuration."""

    typer.echo(json.dumps(REALIZER_CONFIG_ADAPTER.json_schema(), indent=2))


@schema_app.command("artifact-manifest")
def artifact_manifest_schema() -> None:
    """Print the JSON Schema for content-bound local artifact bundles."""

    typer.echo(json.dumps(ArtifactBundleManifestV1.model_json_schema(), indent=2))


@schema_app.command("artifact-preflight")
def artifact_preflight_schema() -> None:
    """Print the JSON Schema for `preflight-artifact --json`."""

    typer.echo(json.dumps(ArtifactPreflightResultV1.model_json_schema(), indent=2))


@schema_app.command("reference-release")
def reference_release_schema() -> None:
    """Print the JSON Schema for dual-architecture reference release manifests."""

    typer.echo(json.dumps(ReferenceReleaseManifestV1.model_json_schema(), indent=2))


@schema_app.command("reference-release-metadata")
def reference_release_metadata_schema() -> None:
    """Print the JSON Schema for reviewed release origin and license declarations."""

    typer.echo(json.dumps(ReferenceReleaseMetadataV1.model_json_schema(), indent=2))


@schema_app.command("reference-prediction")
def reference_prediction_schema() -> None:
    """Print the JSON Schema for one canonical neural prediction or rejection."""

    typer.echo(json.dumps(ReferencePredictionV1.model_json_schema(), indent=2))


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

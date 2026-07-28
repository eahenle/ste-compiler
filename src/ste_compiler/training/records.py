from typing import NotRequired, TypedDict

from ste_compiler.diagnostics import ValidationReport
from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators import LexicalValidator, ValidationPipeline

DETERMINISTIC_REALIZER_PROFILE = "deterministic"


class TrainingRecord(TypedDict):
    document_id: str
    serialized_ir: str
    symbols: str
    allowed_symbols: list[str]
    text: str
    metadata: dict[str, str]
    source_path: NotRequired[str]


class TrainingRecordValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        codes = ", ".join(sorted({diagnostic.code for diagnostic in report.violations}))
        super().__init__(f"deterministic training target was rejected: {codes}")


def build_training_record(
    document: Document,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
    *,
    source_path: str | None = None,
) -> TrainingRecord:
    realizer = DeterministicRealizer()
    actual_profile = {
        "realizer": DETERMINISTIC_REALIZER_PROFILE,
        "realizer_version": realizer.version,
        "vocabulary_version": vocabulary.data.version,
        "terminology_version": terminology.data.version,
        "validator_profile": ValidationPipeline.profile,
    }
    claimed_profile = {
        "realizer": document.metadata.realizer,
        "realizer_version": document.metadata.realizer_version,
        "vocabulary_version": document.metadata.vocabulary_version,
        "terminology_version": document.metadata.terminology_version,
        "validator_profile": document.metadata.validator_profile,
    }
    mismatches = [
        f"{key}={claimed_profile[key]!r} (expected {actual_profile[key]!r})"
        for key in actual_profile
        if claimed_profile[key] != actual_profile[key]
    ]
    if mismatches:
        raise ValueError(
            f"document {document.id!r} metadata does not match the corpus export runtime: "
            + ", ".join(mismatches)
        )

    result = realizer.realize(document, vocabulary, terminology)
    report = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        result.text,
        document,
        result,
    )
    if report.status == "rejected":
        raise TrainingRecordValidationError(report)

    symbols = SymbolicLexicalizer(vocabulary, terminology).symbolize(result.text)
    record = TrainingRecord(
        document_id=document.id,
        serialized_ir=canonical_document_json(document),
        symbols=symbols,
        allowed_symbols=sorted(set(symbols.split())),
        text=result.text,
        metadata={**result.metadata, **actual_profile},
    )
    if source_path is not None:
        record["source_path"] = source_path
    return record

from typing import TypedDict

from ste_compiler.diagnostics import ValidationReport
from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators import LexicalValidator, ValidationPipeline


class TrainingRecord(TypedDict):
    document_id: str
    serialized_ir: str
    symbols: str
    allowed_symbols: list[str]
    text: str
    metadata: dict[str, str]


class TrainingRecordValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        codes = ", ".join(sorted({diagnostic.code for diagnostic in report.violations}))
        super().__init__(f"deterministic training target was rejected: {codes}")


def build_training_record(
    document: Document,
    vocabulary: Vocabulary,
    terminology: TerminologyRegistry,
) -> TrainingRecord:
    result = DeterministicRealizer().realize(document, vocabulary, terminology)
    report = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        result.text,
        document,
        result,
    )
    if report.status == "rejected":
        raise TrainingRecordValidationError(report)

    symbols = SymbolicLexicalizer(vocabulary, terminology).symbolize(result.text)
    return TrainingRecord(
        document_id=document.id,
        serialized_ir=canonical_document_json(document),
        symbols=symbols,
        allowed_symbols=sorted(set(symbols.split())),
        text=result.text,
        metadata=result.metadata,
    )

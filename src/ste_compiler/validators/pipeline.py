from ste_compiler.diagnostics import ValidationReport
from ste_compiler.ir.models import Document
from ste_compiler.realizer.base import RealizationResult

from .lexical import LexicalValidator
from .semantic import SemanticValidator
from .structural import StructuralValidator


class ValidationPipeline:
    def __init__(self, lexical: LexicalValidator, structural: StructuralValidator | None = None):
        self.lexical = lexical
        self.structural = structural or StructuralValidator()

    def validate(
        self,
        text: str,
        document: Document | None = None,
        realization: RealizationResult | None = None,
    ) -> ValidationReport:
        items = self.lexical.validate(text) + self.structural.validate(text)
        if document is not None and realization is not None:
            items += SemanticValidator().validate(document, realization)
        return ValidationReport.from_diagnostics(items)

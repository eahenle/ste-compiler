"""Provider-neutral, schema-validating semantic extraction (never final prose)."""

from typing import Protocol
from pydantic import ValidationError
from ste_compiler.ir.models import Document


class StructuredIRProvider(Protocol):
    model_id: str

    def extract_ir(
        self, source: str, schema: dict[str, object], feedback: str | None
    ) -> dict[str, object]: ...


class LLMFrontend:
    version = "0.1.0"

    def __init__(self, provider: StructuredIRProvider, retries: int = 2):
        self.provider, self.retries = provider, retries

    def parse(self, source: str) -> Document:
        feedback = None
        for attempt in range(self.retries + 1):
            try:
                doc = Document.model_validate(
                    self.provider.extract_ir(source, Document.model_json_schema(), feedback)
                )
                if any(
                    not span.quote
                    for sec in doc.sections
                    for statement in sec.statements
                    for span in statement.source_spans
                ):
                    raise ValueError("all extracted claims must include quoted source spans")
                return doc
            except (ValidationError, ValueError) as error:
                feedback = f"Schema/provenance validation failed: {error}"
                if attempt == self.retries:
                    raise
        raise AssertionError("unreachable")

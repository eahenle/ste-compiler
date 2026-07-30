"""Provider-neutral, schema-validating semantic extraction (never final prose)."""

from typing import Protocol

from pydantic import ValidationError

from ste_compiler.ir.models import Document, SourceSpan


class StructuredIRProvider(Protocol):
    model_id: str

    def extract_ir(
        self, source: str, schema: dict[str, object], feedback: str | None
    ) -> dict[str, object]: ...


class LLMFrontend:
    version = "0.1.0"

    def __init__(self, provider: StructuredIRProvider, retries: int = 2):
        if not provider.model_id or not provider.model_id.strip():
            raise ValueError("frontend provider model_id must be nonblank")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("frontend retries must be a non-negative integer")
        self.provider, self.retries = provider, retries

    @staticmethod
    def _verify_span(span: SourceSpan, source: str, source_id: str | None) -> None:
        if source_id is not None and span.source_id != source_id:
            raise ValueError(f"source span identifies {span.source_id!r}; expected {source_id!r}")
        if span.end > len(source):
            raise ValueError(
                f"source span {span.start}:{span.end} exceeds source length {len(source)}"
            )
        if span.quote is None or not span.quote.strip():
            raise ValueError("all extracted claims must include nonblank source quotes")
        observed = source[span.start : span.end]
        if observed != span.quote:
            raise ValueError(f"source span {span.start}:{span.end} quote does not match the source")

    def parse(self, source: str, *, source_id: str | None = None) -> Document:
        feedback = None
        for attempt in range(self.retries + 1):
            proposal = self.provider.extract_ir(
                source,
                Document.model_json_schema(),
                feedback,
            )
            try:
                doc = Document.model_validate(proposal)
                statements = [statement for sec in doc.sections for statement in sec.statements]
                if any(not statement.source_spans for statement in statements):
                    raise ValueError("all extracted claims must include quoted source spans")
                spans = [
                    span
                    for claim in [
                        *statements,
                        *doc.ambiguities,
                        *doc.causal_relations,
                    ]
                    for span in claim.source_spans
                ]
                for span in spans:
                    self._verify_span(span, source, source_id)
                metadata = doc.metadata.model_copy(
                    update={
                        "frontend": self.provider.model_id,
                        "frontend_version": self.version,
                    }
                )
                doc = doc.model_copy(update={"metadata": metadata})
                return doc
            except (ValidationError, ValueError) as error:
                feedback = f"Schema/provenance validation failed: {error}"
                if attempt == self.retries:
                    raise
        raise AssertionError("unreachable")

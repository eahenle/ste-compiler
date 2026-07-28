from typing import Protocol

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer.base import (
    DEFAULT_CONSTRAINTS,
    RealizationConstraints,
    RealizationResult,
)
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators.alignment import align_controlled_text


class SymbolGenerator(Protocol):
    """Vendor-neutral SLM adapter: generate symbols, never production prose."""

    model_id: str

    def generate_symbols(self, serialized_ir: str, allowed_symbols: frozenset[str]) -> str: ...


class NeuralRealizerUnavailable(RuntimeError):
    pass


class NeuralRealizer:
    """Constrain a generator to symbols and independently align its controlled text."""

    version = "0.1.0"

    def __init__(self, generator: SymbolGenerator):
        self.generator = generator

    def realize(
        self,
        document: Document,
        vocabulary: Vocabulary,
        terminology: TerminologyRegistry,
        constraints: RealizationConstraints = DEFAULT_CONSTRAINTS,
    ) -> RealizationResult:
        deterministic = DeterministicRealizer().realize(
            document, vocabulary, terminology, constraints
        )
        lexicalizer = SymbolicLexicalizer(vocabulary, terminology)
        reference_plan = lexicalizer.symbolize(deterministic.text)
        allowed_symbols = frozenset(reference_plan.split())
        generated_plan = self.generator.generate_symbols(
            canonical_document_json(document), allowed_symbols
        )
        text = lexicalizer.lexicalize(
            generated_plan,
            allowed_symbols=allowed_symbols,
            capitalize_sentences=True,
        )
        aligned = align_controlled_text(text, deterministic)
        return RealizationResult(
            text=aligned.text,
            mappings=aligned.mappings,
            metadata={
                **aligned.metadata,
                "realizer": "symbolic-neural",
                "realizer_version": self.version,
                "model_id": self.generator.model_id,
                "symbol_profile": "plan-specific-v1",
            },
        )

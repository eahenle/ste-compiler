from dataclasses import replace
from typing import Protocol

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer.base import (
    DEFAULT_CONSTRAINTS,
    RealizationConstraints,
    RealizationResult,
)
from ste_compiler.realizer.constrained import EXACT_PLAN_SYMBOL, SymbolicLexicalizer
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators.alignment import (
    align_controlled_text,
    has_exact_whitespace_layout,
)


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
        generated_symbols = generated_plan.split()
        if not generated_symbols or generated_symbols[0] != EXACT_PLAN_SYMBOL:
            raise ValueError(f"generated plan must begin with {EXACT_PLAN_SYMBOL}")
        text = lexicalizer.lexicalize(
            generated_plan,
            allowed_symbols=allowed_symbols,
            capitalize_sentences=True,
        )
        whitespace_layout_preserved = has_exact_whitespace_layout(text, deterministic.text)
        alignment_reference = (
            deterministic if text == deterministic.text else replace(deterministic, mappings=())
        )
        aligned = align_controlled_text(text, alignment_reference)
        metadata = {
            **aligned.metadata,
            "realizer": "symbolic-neural",
            "realizer_version": self.version,
            "model_id": self.generator.model_id,
            "symbol_profile": "plan-specific-v1",
            "whitespace_alignment": "exact-layout-v1",
            "whitespace_layout_preserved": str(whitespace_layout_preserved).lower(),
        }
        model_revision = getattr(self.generator, "model_revision", None)
        if isinstance(model_revision, str):
            metadata["model_revision"] = model_revision
        return RealizationResult(
            text=aligned.text,
            mappings=aligned.mappings,
            metadata=metadata,
        )

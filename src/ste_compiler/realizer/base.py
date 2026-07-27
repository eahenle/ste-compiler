from dataclasses import dataclass, field
from typing import Protocol

from ste_compiler.ir.models import Document
from ste_compiler.terminology import TerminologyRegistry, Vocabulary


@dataclass(frozen=True)
class RealizationConstraints:
    max_sentence_words: int = 25


DEFAULT_CONSTRAINTS = RealizationConstraints()


@dataclass(frozen=True)
class SentenceMapping:
    sentence: int
    text: str
    ir_node_ids: tuple[str, ...]
    features: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RealizationResult:
    text: str
    mappings: tuple[SentenceMapping, ...]
    metadata: dict[str, str]


class Realizer(Protocol):
    def realize(
        self,
        document: Document,
        vocabulary: Vocabulary,
        terminology: TerminologyRegistry,
        constraints: RealizationConstraints = DEFAULT_CONSTRAINTS,
    ) -> RealizationResult: ...

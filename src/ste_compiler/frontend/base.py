from typing import Protocol

from ste_compiler.ir.models import Document


class SemanticFrontend(Protocol):
    version: str

    def parse(self, source: str) -> Document: ...

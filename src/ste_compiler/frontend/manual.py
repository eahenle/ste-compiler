from pathlib import Path

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import loads_document


class ManualFrontend:
    version = "0.1.0"

    def __init__(self, suffix: str = ".yaml"):
        self.suffix = suffix

    def parse(self, source: str) -> Document:
        return loads_document(source, self.suffix)

    def parse_file(self, path: Path) -> Document:
        return ManualFrontend(path.suffix).parse(path.read_text())

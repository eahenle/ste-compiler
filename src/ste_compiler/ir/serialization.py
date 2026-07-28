import json
from pathlib import Path

import yaml

from ste_compiler.ir.models import Document


def loads_document(data: str, suffix: str = ".yaml") -> Document:
    raw = json.loads(data) if suffix.lower() == ".json" else yaml.safe_load(data)
    return Document.model_validate(raw)


def load_document(path: Path) -> Document:
    return loads_document(path.read_text(encoding="utf-8"), path.suffix)


def dumps_document(document: Document, *, as_json: bool = False) -> str:
    obj = document.model_dump(mode="json")
    return json.dumps(obj, indent=2) if as_json else yaml.safe_dump(obj, sort_keys=False)


def canonical_document_json(document: Document) -> str:
    """Serialize IR deterministically for model inputs, hashes, and training records."""

    return json.dumps(
        document.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

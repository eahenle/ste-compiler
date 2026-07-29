"""Load custom resource files and realize one typed IR document."""

from __future__ import annotations

import json
from pathlib import Path

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators import LexicalValidator, ValidationPipeline


def _resource_root() -> Path:
    packaged = Path(__file__).resolve().parent / "resources"
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "examples/resources"
    if checkout.is_dir():
        return checkout
    raise RuntimeError("custom example resources are not installed")


def build_result() -> dict[str, object]:
    """Run the custom-resource example and return its auditable result."""

    resource_root = _resource_root()
    document = load_document(resource_root / "custom_installation.yaml")
    vocabulary = Vocabulary.load(resource_root / "custom_vocabulary.yaml")
    terminology = TerminologyRegistry.load(resource_root / "custom_terminology.yaml")
    realization = DeterministicRealizer().realize(document, vocabulary, terminology)
    validation = ValidationPipeline(LexicalValidator(vocabulary, terminology)).validate(
        realization.text,
        document,
        realization,
    )
    return {
        "text": realization.text,
        "mappings": [
            {
                "sentence": mapping.sentence,
                "ir_node_ids": list(mapping.ir_node_ids),
            }
            for mapping in realization.mappings
        ],
        "validation": validation.model_dump(mode="json"),
        "vocabulary_version": vocabulary.data.version,
        "terminology_version": terminology.data.version,
    }


def main() -> None:
    print(json.dumps(build_result(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

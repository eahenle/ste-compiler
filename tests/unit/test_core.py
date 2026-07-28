from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ste_compiler.frontend.llm import LLMFrontend
from ste_compiler.ir.models import Document, EntityRef, Quantity
from ste_compiler.ir.serialization import (
    canonical_document_json,
    dumps_document,
    load_document,
    loads_document,
)
from ste_compiler.realizer import DeterministicRealizer, NeuralRealizer
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.validators.lexical import LexicalValidator
from ste_compiler.validators.semantic import SemanticValidator
from ste_compiler.validators.structural import StructuralValidator

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("installation", "Install the access panel."),
        ("negative", "Do not open the shutoff valve."),
        ("conditional", "If hydraulic pressure is high, close the shutoff valve."),
        ("sequence", "Disconnect the access panel before the test after the check."),
        (
            "warning_pressure",
            (
                "Warning: injury can occur when hydraulic pressure is more than 20 MPa.\n"
                "Stop the hydraulic pressure to more than 20 MPa."
            ),
        ),
    ],
)
def test_realization_patterns(name, expected, vocab, terms):
    document = load_document(ROOT / f"data/examples/{name}.yaml")
    result = DeterministicRealizer().realize(document, vocab, terms)
    assert result.text == expected
    assert result.mappings[0].ir_node_ids
    assert DeterministicRealizer().realize(document, vocab, terms) == result


def test_schema_and_round_trip():
    document = load_document(ROOT / "data/examples/conditional.yaml")
    assert loads_document(dumps_document(document)).model_dump() == document.model_dump()
    invalid = document.model_dump()
    invalid["invented"] = True
    with pytest.raises(ValidationError):
        Document.model_validate(invalid)


def test_terminology_normalization(terms):
    assert terms.normalize("system pressure") == "hydraulic pressure"
    assert terms.get("old_pressure").id == "hydraulic_pressure"


def test_vocabulary_and_symbolic_lexicalizer(vocab, terms):
    assert vocab.contains("make") and not vocab.contains("commence")
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    assert (
        lexicalizer.lexicalize("WORD_make WORD_sure WORD_that TERM_hydraulic_pressure PERIOD")
        == "make sure that hydraulic pressure."
    )
    with pytest.raises(ValueError):
        lexicalizer.lexicalize("WORD_commence PERIOD")


@pytest.mark.parametrize(
    "name",
    ["installation", "negative", "conditional", "sequence", "warning_pressure"],
)
def test_symbolic_plan_round_trip(name, vocab, terms):
    document = load_document(ROOT / f"data/examples/{name}.yaml")
    expected = DeterministicRealizer().realize(document, vocab, terms).text
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    symbols = lexicalizer.symbolize(expected)
    assert lexicalizer.lexicalize(symbols, capitalize_sentences=True) == expected


def test_symbolic_plan_allowlist_blocks_invented_quantity(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    allowed = frozenset({"NUMBER_20", "UNIT_MPa", "PERIOD"})
    with pytest.raises(ValueError, match="not allowed"):
        lexicalizer.lexicalize(
            "NUMBER_21 UNIT_MPa PERIOD",
            allowed_symbols=allowed,
        )


def test_neural_realizer_accepts_only_aligned_symbolic_output(vocab, terms):
    document = load_document(ROOT / "data/examples/negative.yaml")
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)

    class Generator:
        model_id = "offline-test-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            assert serialized_ir == canonical_document_json(document)
            assert allowed_symbols == frozenset(expected_plan.split())
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)
    assert result.text == expected.text
    assert result.metadata["model_id"] == "offline-test-generator"
    assert result.metadata["alignment"] == "deterministic-surface-v1"
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_does_not_trust_reordered_allowed_symbols(vocab, terms):
    document = load_document(ROOT / "data/examples/negative.yaml")

    class Generator:
        model_id = "offline-corrupt-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert "WORD_not" in allowed_symbols
            return "WORD_open WORD_do WORD_not WORD_the TERM_shutoff_valve PERIOD"

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)
    diagnostics = SemanticValidator().validate(document, result)
    assert not result.mappings[0].ir_node_ids
    assert {diagnostic.code for diagnostic in diagnostics} == {
        "REQUIRED_NODE_OMITTED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    }


def test_hazard_sentence_does_not_substitute_for_required_instruction(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    result = DeterministicRealizer().realize(document, vocab, terms)
    hazard_only = replace(result, text=result.mappings[0].text, mappings=result.mappings[:1])
    codes = {item.code for item in SemanticValidator().validate(document, hazard_only)}
    assert "REQUIRED_NODE_OMITTED" in codes


def test_instruction_cannot_omit_its_hazard_sentence(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    result = DeterministicRealizer().realize(document, vocab, terms)
    instruction_only = replace(result, text=result.mappings[1].text, mappings=result.mappings[1:])
    codes = {item.code for item in SemanticValidator().validate(document, instruction_only)}
    assert "HAZARD_NOT_PRESERVED" in codes


def test_lexical_and_structural_diagnostics(vocab, terms):
    lexical = LexicalValidator(vocab, terms).validate("Commence with the system pressure.")
    assert {x.code for x in lexical} == {"UNAUTHORIZED_WORD", "TERMINOLOGY_ALIAS"}
    structural = StructuralValidator(max_sentence_words=2).validate("Open it now.")
    assert {x.code for x in structural} == {"SENTENCE_TOO_LONG", "AMBIGUOUS_PRONOUN"}
    assert lexical[0].model_dump_json()


def test_semantic_corruption_detected(vocab, terms):
    document = load_document(ROOT / "data/examples/negative.yaml")
    result = DeterministicRealizer().realize(document, vocab, terms)
    bad_mapping = replace(
        result.mappings[0],
        text="Open the shutoff valve.",
        features={**result.mappings[0].features, "negated": False},
    )
    bad = replace(result, text=bad_mapping.text, mappings=(bad_mapping,))
    assert "NEGATION_NOT_PRESERVED" in {x.code for x in SemanticValidator().validate(document, bad)}


def test_actor_instruction_preserves_negation(vocab, terms):
    document = load_document(ROOT / "data/examples/negative.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={"actor": EntityRef(id="technician", name="technician")}
    )
    result = DeterministicRealizer().realize(document, vocab, terms)
    assert result.text == "Technician must not open the shutoff valve."
    assert not SemanticValidator().validate(document, result)


def test_llm_frontend_rejects_statements_without_source_spans():
    document = load_document(ROOT / "data/examples/installation.yaml").model_dump(mode="json")
    document["sections"][0]["statements"][0]["source_spans"] = []

    class Provider:
        model_id = "test"

        def extract_ir(self, source, schema, feedback):
            del source, schema, feedback
            return document

    with pytest.raises(ValueError, match="quoted source spans"):
        LLMFrontend(Provider(), retries=0).parse("Install the access panel.")


@given(st.integers(min_value=0, max_value=1_000_000))
def test_quantity_format_is_stable(value):
    quantity = Quantity(value=float(value), unit="Pa")
    realizer = DeterministicRealizer()
    assert realizer._quantity(quantity) == f"{value} Pa"

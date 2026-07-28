from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ste_compiler.frontend.llm import LLMFrontend
from ste_compiler.ir.models import Document, EntityRef, Quantity, QuantityConstraint
from ste_compiler.ir.serialization import (
    canonical_document_json,
    dumps_document,
    load_document,
    loads_document,
)
from ste_compiler.realizer import DeterministicRealizer, NeuralRealizer
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
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
    with pytest.raises(ValueError, match="invalid output symbol"):
        lexicalizer.lexicalize("PUNCT_UFFFFFF")


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


def test_symbolic_plan_preserves_configured_nonword_units(vocab, terms):
    custom_vocab = Vocabulary(
        vocab.data.model_copy(
            update={"units": [*vocab.data.units, "%", "°C", "m/s", "degrees Celsius"]}
        )
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)
    text = "20 % 5 °C 3 m/s 2 degrees Celsius."
    symbols = lexicalizer.symbolize(text)
    assert symbols == (
        "PLAN_EXACT_WHITESPACE_V1 NUMBER_20 SPACE UNIT_%25 SPACE NUMBER_5 "
        "SPACE UNIT_%C2%B0C SPACE "
        "NUMBER_3 SPACE UNIT_m%2Fs SPACE NUMBER_2 SPACE UNIT_degrees%20Celsius PERIOD"
    )
    assert lexicalizer.lexicalize(symbols) == text


@given(punctuation=st.from_regex(r"[^\w\s]", fullmatch=True))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_symbolic_plan_represents_every_accepted_punctuation(punctuation, vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    symbols = lexicalizer.symbolize(punctuation)
    assert lexicalizer.lexicalize(symbols) == punctuation


def test_symbolic_plan_round_trips_parentheses_semicolon_and_word_case(vocab, terms):
    apu_entry = vocab.data.entries[0].model_copy(update={"lemma": "APU", "inflections": []})
    custom_vocab = Vocabulary(
        vocab.data.model_copy(update={"entries": [*vocab.data.entries, apu_entry]})
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)
    text = "APU (test); APU."

    symbols = lexicalizer.symbolize(text)

    assert symbols == (
        "PLAN_EXACT_WHITESPACE_V1 WORD_APU SPACE PUNCT_U0028 WORD_test "
        "PUNCT_U0029 PUNCT_U003B SPACE WORD_APU PERIOD"
    )
    assert lexicalizer.lexicalize(symbols, capitalize_sentences=True) == text


@pytest.mark.parametrize(
    ("text", "expected_symbols"),
    [
        (
            "safe — slowly",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe SPACE PUNCT_U2014 SPACE WORD_slowly",
        ),
        (
            "safe ; slowly",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe SPACE PUNCT_U003B SPACE WORD_slowly",
        ),
        (
            "safe;slowly",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe PUNCT_U003B WORD_slowly",
        ),
        (
            "safe\t;\tslowly",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe WS_U0009 PUNCT_U003B WS_U0009 WORD_slowly",
        ),
    ],
)
def test_symbolic_plan_preserves_explicit_whitespace(text, expected_symbols, vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)

    symbols = lexicalizer.symbolize(text)

    assert symbols == expected_symbols
    assert lexicalizer.lexicalize(symbols, allowed_symbols=frozenset(symbols.split())) == text


def test_markerless_symbolic_plans_retain_legacy_implicit_spacing(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)

    assert lexicalizer.lexicalize("WORD_safe PUNCT_U003B WORD_slowly") == "safe; slowly"


def test_symbolic_plan_escapes_terminology_ids(vocab, terms):
    encoded_id = "access panel/v1"
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"id": encoded_id})
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    lexicalizer = SymbolicLexicalizer(vocab, custom_terms)

    symbols = lexicalizer.symbolize("access panel")

    assert symbols == "PLAN_EXACT_WHITESPACE_V1 TERM_access%20panel%2Fv1"
    assert (
        lexicalizer.lexicalize(symbols, allowed_symbols=frozenset(symbols.split()))
        == "access panel"
    )


@pytest.mark.parametrize(
    "text",
    [
        'Install the access panel "safe".',
        "Install the access panel “safe”.",
        "Install the access panel ‘safe’.",
    ],
)
def test_symbolic_plan_round_trips_quoted_approved_text(text, vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    symbols = lexicalizer.symbolize(text)
    assert lexicalizer.lexicalize(symbols, capitalize_sentences=True) == text


def test_quote_spacing_does_not_break_apostrophe_joining(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    assert lexicalizer.lexicalize("WORD_do PUNCT_U0027 WORD_not") == "do'not"
    assert lexicalizer.lexicalize("WORD_do PUNCT_U2019 WORD_not") == "do’not"


def test_symbolic_plan_preserves_unit_case_and_rejects_noncanonical_spelling(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    assert (
        lexicalizer.symbolize("20 MPa.")
        == "PLAN_EXACT_WHITESPACE_V1 NUMBER_20 SPACE UNIT_MPa PERIOD"
    )
    with pytest.raises(ValueError, match="unauthorized word"):
        lexicalizer.symbolize("20 mPa.")
    with pytest.raises(ValueError, match="unauthorized unit symbol"):
        lexicalizer.lexicalize("NUMBER_20 UNIT_mPa PERIOD")
    assert {
        diagnostic.code for diagnostic in LexicalValidator(vocab, terms).validate("20 mPa.")
    } == {"UNAUTHORIZED_WORD"}


def test_symbolic_plan_supports_scientific_notation(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    symbols = lexicalizer.symbolize("1e-07 MPa.")
    assert symbols == "PLAN_EXACT_WHITESPACE_V1 NUMBER_1e-07 SPACE UNIT_MPa PERIOD"
    assert lexicalizer.lexicalize(symbols) == "1e-07 MPa."
    assert not LexicalValidator(vocab, terms).validate("1e-07 MPa.")


@pytest.mark.parametrize(
    ("text", "expected_symbol"),
    [
        (
            "-20 MPa.",
            "PLAN_EXACT_WHITESPACE_V1 NUMBER_-20 SPACE UNIT_MPa PERIOD",
        ),
        (
            "-1e-07 MPa.",
            "PLAN_EXACT_WHITESPACE_V1 NUMBER_-1e-07 SPACE UNIT_MPa PERIOD",
        ),
    ],
)
def test_symbolic_plan_parses_signed_numbers_before_punctuation(
    text, expected_symbol, vocab, terms
):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    symbols = lexicalizer.symbolize(text)
    assert symbols == expected_symbol
    assert lexicalizer.lexicalize(symbols) == text


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


def test_neural_realizer_aligns_decimal_quantities(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    instruction = document.sections[0].statements[0]
    decimal_quantity = Quantity(value=20.5, unit="MPa", comparator="more_than")
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                instruction.quantity_constraints[0].model_copy(
                    update={"quantity": decimal_quantity}
                )
            ],
            "hazards": [instruction.hazards[0].model_copy(update={"threshold": decimal_quantity})],
        }
    )
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)

    class Generator:
        model_id = "offline-decimal-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert "NUMBER_20.5" in allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)
    assert result.text == expected.text
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_scientific_notation_quantities(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    instruction = document.sections[0].statements[0]
    scientific_quantity = Quantity(value=1e-7, unit="MPa", comparator="more_than")
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                instruction.quantity_constraints[0].model_copy(
                    update={"quantity": scientific_quantity}
                )
            ],
            "hazards": [
                instruction.hazards[0].model_copy(update={"threshold": scientific_quantity})
            ],
        }
    )
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert "NUMBER_1e-07" in expected_plan.split()

    class Generator:
        model_id = "offline-scientific-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert "NUMBER_1e-07" in allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)
    assert result.text == expected.text
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_negative_scientific_quantities(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    instruction = document.sections[0].statements[0]
    negative_quantity = Quantity(value=-1e-7, unit="MPa", comparator="more_than")
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                instruction.quantity_constraints[0].model_copy(
                    update={"quantity": negative_quantity}
                )
            ],
            "hazards": [instruction.hazards[0].model_copy(update={"threshold": negative_quantity})],
        }
    )
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert "NUMBER_-1e-07" in expected_plan.split()
    assert "PUNCT_U002D" not in expected_plan.split()

    class Generator:
        model_id = "offline-negative-scientific-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert "NUMBER_-1e-07" in allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)
    assert result.text == expected.text
    assert result.mappings == expected.mappings
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_opaque_casing_and_internal_periods(vocab, terms):
    apu_entry = vocab.data.entries[0].model_copy(update={"lemma": "APU", "inflections": []})
    custom_vocab = Vocabulary(
        vocab.data.model_copy(
            update={
                "entries": [*vocab.data.entries, apu_entry],
                "units": [*vocab.data.units, "N.m"],
            }
        )
    )
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "No. valve"})
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "quantity_constraints": [
                QuantityConstraint(
                    property="torque",
                    quantity=Quantity(value=20, unit="N.m"),
                )
            ],
            "manner": "(APU); APU",
        }
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, custom_terms)
    expected = DeterministicRealizer().realize(document, custom_vocab, custom_terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected.text == "Install the No. valve to 20 N.m (APU); APU."

    class Generator:
        model_id = "offline-opaque-symbol-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert {"WORD_APU", "UNIT_N.m", "TERM_access_panel"} <= allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, custom_vocab, custom_terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_quoted_manner(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": '"safe"'})
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected.text == 'Install the access panel "safe".'

    class Generator:
        model_id = "offline-quoted-manner-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert {"PUNCT_U0022", "WORD_safe"} <= allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_spaced_punctuation(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": "safe — slowly"})
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected.text == "Install the access panel safe — slowly."

    class Generator:
        model_id = "offline-spaced-punctuation-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert {"SPACE", "PUNCT_U2014", "WORD_safe", "WORD_slowly"} <= allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_aligns_adjacency_and_escaped_term_id(vocab, terms):
    encoded_id = "access panel/v1"
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"id": encoded_id})
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={
            "object": instruction.object.model_copy(update={"term_id": encoded_id}),
            "manner": "safe;slowly",
        }
    )
    lexicalizer = SymbolicLexicalizer(vocab, custom_terms)
    expected = DeterministicRealizer().realize(document, vocab, custom_terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected.text == "Install the access panel safe;slowly."

    class Generator:
        model_id = "offline-exact-plan-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert {
                "PLAN_EXACT_WHITESPACE_V1",
                "TERM_access%20panel%2Fv1",
                "PUNCT_U003B",
            } <= allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, custom_terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings
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


def test_terminology_schema_rejects_empty_canonical_form(terms):
    data = terms.data.model_dump()
    data["terms"][0]["canonical_form"] = ""

    with pytest.raises(ValidationError):
        type(terms.data).model_validate(data)


def test_vocabulary_schema_rejects_empty_unit(vocab):
    data = vocab.data.model_dump()
    data["units"].append("")

    with pytest.raises(ValidationError):
        type(vocab.data).model_validate(data)


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

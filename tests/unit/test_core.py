from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ste_compiler.frontend.llm import LLMFrontend
from ste_compiler.ir.models import (
    CausalRelation,
    Document,
    EntityRef,
    Quantity,
    QuantityConstraint,
)
from ste_compiler.ir.serialization import (
    canonical_document_json,
    dumps_document,
    load_document,
    loads_document,
)
from ste_compiler.realizer import DeterministicRealizer, NeuralRealizer
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.boundaries import whole_casefold_spans
from ste_compiler.validators.alignment import align_controlled_text
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


@pytest.mark.parametrize(
    ("text", "expected_symbols"),
    [
        (
            "20°C.",
            "PLAN_EXACT_WHITESPACE_V1 NUMBER_20 UNIT_%C2%B0C PERIOD",
        ),
        (
            "20m/s.",
            "PLAN_EXACT_WHITESPACE_V1 NUMBER_20 UNIT_m%2Fs PERIOD",
        ),
        (
            "°C.",
            "PLAN_EXACT_WHITESPACE_V1 UNIT_%C2%B0C PERIOD",
        ),
        (
            "safe °C.",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe SPACE UNIT_%C2%B0C PERIOD",
        ),
        (
            "safe,°C.",
            "PLAN_EXACT_WHITESPACE_V1 WORD_safe COMMA UNIT_%C2%B0C PERIOD",
        ),
    ],
)
def test_symbolic_plan_uses_units_at_valid_numeric_and_surface_boundaries(
    text, expected_symbols, vocab, terms
):
    custom_vocab = Vocabulary(
        vocab.data.model_copy(update={"units": [*vocab.data.units, "°C", "m/s"]})
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)

    symbols = lexicalizer.symbolize(text)

    assert symbols == expected_symbols
    assert lexicalizer.lexicalize(symbols) == text
    assert not LexicalValidator(custom_vocab, terms).validate(text)


@pytest.mark.parametrize("text", ["safe°C.", "safem/s.", "safe_°C."])
def test_symbolic_plan_rejects_units_attached_to_alphabetic_or_underscore(text, vocab, terms):
    custom_vocab = Vocabulary(
        vocab.data.model_copy(update={"units": [*vocab.data.units, "°C", "m/s"]})
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)

    with pytest.raises(ValueError, match="cannot symbolize|unauthorized word"):
        lexicalizer.symbolize(text)
    assert "UNAUTHORIZED_WORD" in {
        diagnostic.code for diagnostic in LexicalValidator(custom_vocab, terms).validate(text)
    }


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


def test_exact_symbolic_plan_preserves_first_word_and_internal_sentence_case(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    text = "Safe? slowly"

    symbols = lexicalizer.symbolize(text)

    assert symbols == ("PLAN_EXACT_WHITESPACE_V1 WORD_Safe QUESTION SPACE WORD_slowly")
    assert lexicalizer.lexicalize(symbols, capitalize_sentences=True) == text


def test_exact_symbolic_plan_preserves_acronym_and_mixed_case(vocab, terms):
    template = vocab.data.entries[0]
    custom_vocab = Vocabulary(
        vocab.data.model_copy(
            update={
                "entries": [
                    *vocab.data.entries,
                    template.model_copy(update={"lemma": "APU", "inflections": []}),
                    template.model_copy(update={"lemma": "eBay", "inflections": []}),
                ]
            }
        )
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)
    text = "APU eBay safe? slowly"

    symbols = lexicalizer.symbolize(text)

    assert "WORD_APU" in symbols.split()
    assert "WORD_eBay" in symbols.split()
    assert lexicalizer.lexicalize(symbols, capitalize_sentences=True) == text


def test_markerless_symbolic_plan_retains_legacy_capitalization(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)

    assert (
        lexicalizer.lexicalize(
            "WORD_safe QUESTION WORD_slowly",
            capitalize_sentences=True,
        )
        == "Safe? Slowly"
    )


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
    encoded_id = "access|panel/v1"
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(
                        update={
                            "id": encoded_id,
                            "canonical_form": "access|panel",
                        }
                    )
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    lexicalizer = SymbolicLexicalizer(vocab, custom_terms)

    symbols = lexicalizer.symbolize("aCcEsS|PaNeL")

    assert symbols == ("PLAN_EXACT_WHITESPACE_V1 TERM_access%7Cpanel%2Fv1|aCcEsS%7CPaNeL")
    assert (
        lexicalizer.lexicalize(symbols, allowed_symbols=frozenset(symbols.split()))
        == "aCcEsS|PaNeL"
    )


def test_exact_term_surface_rejects_tampering_after_allowlist_check(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    valid = "TERM_access_panel|Access%20panel"
    tampered = "TERM_access_panel|shutoff%20valve"
    allowed = frozenset({"PLAN_EXACT_WHITESPACE_V1", valid})

    with pytest.raises(ValueError, match="not allowed"):
        lexicalizer.lexicalize(
            f"PLAN_EXACT_WHITESPACE_V1 {tampered}",
            allowed_symbols=allowed,
        )
    with pytest.raises(ValueError, match="unauthorized term symbol"):
        lexicalizer.lexicalize(
            f"PLAN_EXACT_WHITESPACE_V1 {tampered}",
        )


def test_markerless_term_symbol_retains_legacy_canonical_surface(vocab, terms):
    lexicalizer = SymbolicLexicalizer(vocab, terms)

    assert lexicalizer.lexicalize("TERM_access_panel") == "access panel"


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
    assert result.metadata["whitespace_alignment"] == "exact-layout-v1"
    assert result.metadata["whitespace_layout_preserved"] == "true"
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_rejects_markerless_legacy_casing_bypass(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"object": None})
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    allowed_symbols = frozenset(expected_plan.split())
    markerless_plan = "WORD_Install PERIOD"
    assert set(markerless_plan.split()) <= allowed_symbols
    assert (
        lexicalizer.lexicalize(
            markerless_plan,
            allowed_symbols=allowed_symbols,
            capitalize_sentences=True,
        )
        == expected.text
    )

    class Generator:
        model_id = "offline-markerless-casing-generator"

        def generate_symbols(self, serialized_ir, supplied_symbols):
            del serialized_ir
            assert supplied_symbols == allowed_symbols
            return markerless_plan

    with pytest.raises(
        ValueError,
        match="must begin with PLAN_EXACT_WHITESPACE_V1",
    ):
        NeuralRealizer(Generator()).realize(document, vocab, terms)


def test_neural_realizer_withholds_mappings_for_swapped_surface_casing(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"] = [
        {
            "kind": "state",
            "id": "state_001",
            "subject": {"term_id": "access_panel"},
            "predicate": "is",
            "value": "access panel safe Safe",
            "source_spans": [],
        }
    ]
    document = Document.model_validate(raw)
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    generated_symbols = lexicalizer.symbolize(expected.text).split()
    first_term = generated_symbols.index("TERM_access_panel|Access%20panel")
    second_term = generated_symbols.index("TERM_access_panel|access%20panel")
    lower_word = generated_symbols.index("WORD_safe")
    upper_word = generated_symbols.index("WORD_Safe")
    generated_symbols[first_term], generated_symbols[second_term] = (
        generated_symbols[second_term],
        generated_symbols[first_term],
    )
    generated_symbols[lower_word], generated_symbols[upper_word] = (
        generated_symbols[upper_word],
        generated_symbols[lower_word],
    )
    generated_plan = " ".join(generated_symbols)

    class Generator:
        model_id = "offline-swapped-case-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert set(generated_symbols) <= allowed_symbols
            return generated_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert expected.text == "Access panel is access panel safe Safe."
    assert result.text == "access panel is Access panel Safe safe."
    assert result.metadata["whitespace_layout_preserved"] == "true"
    assert all(
        mapping.ir_node_ids for mapping in align_controlled_text(result.text, expected).mappings
    )
    assert all(not mapping.ir_node_ids for mapping in result.mappings)
    assert {diagnostic.code for diagnostic in SemanticValidator().validate(document, result)} == {
        "REQUIRED_NODE_OMITTED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    }


def test_neural_realizer_rejects_alphabetically_attached_configured_unit(vocab, terms):
    custom_vocab = Vocabulary(vocab.data.model_copy(update={"units": [*vocab.data.units, "°C"]}))
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(update={"manner": "safe°C"})

    class Generator:
        model_id = "must-not-run"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            raise AssertionError(
                f"generator called for invalid plan: {serialized_ir}, {allowed_symbols}"
            )

    with pytest.raises(ValueError, match="unauthorized word"):
        NeuralRealizer(Generator()).realize(document, custom_vocab, terms)


def test_neural_realizer_preserves_capitalized_first_term_surface(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"] = [
        {
            "kind": "state",
            "id": "state_001",
            "subject": {"term_id": "access_panel"},
            "predicate": "is",
            "value": "safe",
            "source_spans": [],
        }
    ]
    document = Document.model_validate(raw)
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    expected_term = "TERM_access_panel|Access%20panel"
    assert expected.text == "Access panel is safe."
    assert expected_term in expected_plan.split()

    class Generator:
        model_id = "offline-first-term-surface-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert expected_term in allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_preserves_exact_word_case_across_internal_punctuation(vocab, terms):
    template = vocab.data.entries[0]
    custom_vocab = Vocabulary(
        vocab.data.model_copy(
            update={
                "entries": [
                    *vocab.data.entries,
                    template.model_copy(update={"lemma": "APU", "inflections": []}),
                    template.model_copy(update={"lemma": "eBay", "inflections": []}),
                ]
            }
        )
    )
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={"manner": "APU eBay safe? slowly"}
    )
    lexicalizer = SymbolicLexicalizer(custom_vocab, terms)
    expected = DeterministicRealizer().realize(document, custom_vocab, terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected_plan.startswith("PLAN_EXACT_WHITESPACE_V1 WORD_Install SPACE")
    assert {
        "WORD_APU",
        "WORD_eBay",
        "WORD_slowly",
    } <= set(expected_plan.split())

    class Generator:
        model_id = "offline-exact-case-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert set(expected_plan.split()) <= allowed_symbols
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, custom_vocab, terms)

    assert result.text == "Install the access panel APU eBay safe? slowly."
    assert result.mappings == expected.mappings
    assert result.metadata["whitespace_layout_preserved"] == "true"
    assert not SemanticValidator().validate(document, result)


def test_neural_realizer_rejects_moved_and_retyped_newline_layout(vocab, terms):
    document = load_document(ROOT / "data/examples/warning_pressure.yaml")
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    generated_symbols = lexicalizer.symbolize(expected.text).split()
    original_newline = generated_symbols.index("NEWLINE")
    moved_newline = generated_symbols.index("WORD_injury") + 1
    assert generated_symbols[moved_newline] == "SPACE"
    generated_symbols[moved_newline] = "NEWLINE"
    generated_symbols[original_newline] = "SPACE"
    generated_plan = " ".join(generated_symbols)

    class Generator:
        model_id = "offline-moved-newline-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert {"SPACE", "NEWLINE"} <= allowed_symbols
            assert set(generated_symbols) <= allowed_symbols
            return generated_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert result.text != expected.text
    assert all(
        mapping.ir_node_ids for mapping in align_controlled_text(result.text, expected).mappings
    )
    assert result.metadata["whitespace_layout_preserved"] == "false"
    assert all(not mapping.ir_node_ids for mapping in result.mappings)
    assert {diagnostic.code for diagnostic in SemanticValidator().validate(document, result)} == {
        "REQUIRED_NODE_OMITTED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    }


def test_neural_realizer_rejects_repeated_space_hidden_by_normalization(vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    lexicalizer = SymbolicLexicalizer(vocab, terms)
    expected = DeterministicRealizer().realize(document, vocab, terms)
    generated_symbols = lexicalizer.symbolize(expected.text).split()
    first_space = generated_symbols.index("SPACE")
    generated_symbols.insert(first_space, "SPACE")
    generated_plan = " ".join(generated_symbols)

    class Generator:
        model_id = "offline-repeated-space-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert set(generated_symbols) <= allowed_symbols
            return generated_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, terms)

    assert result.text == expected.text.replace(" ", "  ", 1)
    assert all(
        mapping.ir_node_ids for mapping in align_controlled_text(result.text, expected).mappings
    )
    assert result.metadata["whitespace_layout_preserved"] == "false"
    assert all(not mapping.ir_node_ids for mapping in result.mappings)
    assert {diagnostic.code for diagnostic in SemanticValidator().validate(document, result)} == {
        "REQUIRED_NODE_OMITTED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    }


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
            assert {
                "WORD_APU",
                "UNIT_N.m",
                "TERM_access_panel|No.%20valve",
            } <= allowed_symbols
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
    encoded_id = "access|panel/v1"
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
                "TERM_access%7Cpanel%2Fv1|access%20panel",
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
            term_symbol = next(
                symbol for symbol in allowed_symbols if symbol.startswith("TERM_shutoff_valve|")
            )
            return (
                "PLAN_EXACT_WHITESPACE_V1 WORD_open SPACE WORD_Do SPACE "
                f"WORD_not SPACE WORD_the SPACE {term_symbol} PERIOD"
            )

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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_form", " "),
        ("canonical_form", " 20 "),
        ("canonical_form", "20"),
        ("aliases", ["\t"]),
        ("aliases", ["-1e-07"]),
    ],
)
def test_terminology_schema_rejects_degenerate_or_numeric_forms(field, value, terms):
    data = terms.data.model_dump()
    data["terms"][0][field] = value

    with pytest.raises(
        ValidationError,
        match="nonblank|leading or trailing whitespace|numeric-only",
    ):
        type(terms.data).model_validate(data)


@pytest.mark.parametrize("unit", [" ", "\tMPa", "20", "-1e-07"])
def test_vocabulary_schema_rejects_degenerate_or_numeric_units(unit, vocab):
    data = vocab.data.model_dump()
    data["units"].append(unit)

    with pytest.raises(
        ValidationError,
        match="nonblank|leading or trailing whitespace|numeric-only",
    ):
        type(vocab.data).model_validate(data)


@pytest.mark.parametrize("word", ["", " ", "20", "two_words", "three-part-word"])
def test_vocabulary_schema_rejects_unrepresentable_word_forms(word, vocab):
    data = vocab.data.model_dump()
    data["entries"][0]["lemma"] = word

    with pytest.raises(ValidationError, match="nonblank|ASCII word"):
        type(vocab.data).model_validate(data)


def test_terminology_schema_rejects_duplicate_ids(terms):
    data = terms.data.model_dump()
    data["terms"][1]["id"] = data["terms"][0]["id"]

    with pytest.raises(ValidationError, match="duplicate terminology ID"):
        type(terms.data).model_validate(data)


def test_terminology_schema_rejects_casefold_form_collisions(terms):
    data = terms.data.model_dump()
    data["terms"][1]["aliases"].append(data["terms"][0]["canonical_form"].upper())

    with pytest.raises(
        ValidationError,
        match="duplicate case-insensitive terminology form",
    ):
        type(terms.data).model_validate(data)


def test_vocabulary_schema_rejects_casefold_word_form_collisions(vocab):
    data = vocab.data.model_dump()
    data["entries"][1]["inflections"].append(data["entries"][0]["lemma"].upper())

    with pytest.raises(
        ValidationError,
        match="duplicate case-insensitive vocabulary form",
    ):
        type(vocab.data).model_validate(data)


def test_registries_revalidate_unvalidated_model_copies(vocab, terms):
    duplicate_terms = [
        *terms.data.terms,
        terms.data.terms[0].model_copy(),
    ]
    invalid_terms = terms.data.model_copy(update={"terms": duplicate_terms})
    invalid_vocab = vocab.data.model_copy(update={"units": [*vocab.data.units, "20"]})

    with pytest.raises(ValidationError, match="duplicate terminology ID"):
        TerminologyRegistry(invalid_terms)
    with pytest.raises(ValidationError, match="numeric-only"):
        Vocabulary(invalid_vocab)


@pytest.mark.parametrize("replacement", ["missing", "old_pressure"])
def test_terminology_schema_rejects_missing_or_cyclic_replacements(replacement, terms):
    data = terms.data.model_dump()
    old_pressure = next(term for term in data["terms"] if term["id"] == "old_pressure")
    old_pressure["replacement_term_id"] = replacement

    with pytest.raises(
        ValidationError,
        match="does not exist|replacement cycle",
    ):
        type(terms.data).model_validate(data)


def test_terminology_lookup_guards_against_post_validation_cycle(terms):
    registry = TerminologyRegistry(terms.data)
    old_pressure = next(term for term in registry.data.terms if term.id == "old_pressure")
    old_pressure.replacement_term_id = "old_pressure"

    with pytest.raises(ValueError, match="replacement cycle"):
        registry.get("old_pressure")


@pytest.mark.parametrize("alias", ["%", "C++", "°C", "(old)"])
def test_lexical_validator_detects_punctuation_delimited_aliases(alias, vocab, terms):
    data = terms.data.model_dump()
    data["terms"][0]["aliases"] = [alias]
    registry = TerminologyRegistry(type(terms.data).model_validate(data))

    diagnostics = LexicalValidator(vocab, registry).validate(alias)

    assert [diagnostic.code for diagnostic in diagnostics] == ["TERMINOLOGY_ALIAS"]


def test_lexical_validator_masks_unicode_casefolded_canonical_form(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "Straße", "aliases": []})
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )

    assert not LexicalValidator(vocab, custom_terms).validate("STRASSE.")


def test_lexical_validator_detects_unicode_casefolded_alias(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(
                        update={
                            "canonical_form": "street",
                            "aliases": ["Straße"],
                        }
                    )
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )

    diagnostics = LexicalValidator(vocab, custom_terms).validate("STRASSE.")

    assert [diagnostic.code for diagnostic in diagnostics] == ["TERMINOLOGY_ALIAS"]
    assert diagnostics[0].suggestions == ["street"]


def test_unicode_casefold_masking_retains_original_diagnostic_spans(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "Straße", "aliases": []})
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )

    diagnostics = LexicalValidator(vocab, custom_terms).validate("Straße commence.")

    unauthorized = next(
        diagnostic for diagnostic in diagnostics if diagnostic.code == "UNAUTHORIZED_WORD"
    )
    assert unauthorized.span is not None
    assert unauthorized.span.start == 7
    assert unauthorized.span.end == 15


def test_unicode_casefold_matching_requires_original_character_boundaries():
    assert whole_casefold_spans("Straße", "strasse") == ((0, 6),)
    assert whole_casefold_spans("ß", "ss") == ((0, 1),)
    assert whole_casefold_spans("ß", "s") == ()
    assert whole_casefold_spans("XSTRASSE", "strasse") == ()
    assert whole_casefold_spans("STRASSE_y", "strasse") == ()


def test_symbolizer_uses_original_span_for_unicode_casefold_expansion(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "Straße", "aliases": []})
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )

    symbols = SymbolicLexicalizer(vocab, custom_terms).symbolize("STRASSE.")

    assert symbols == "PLAN_EXACT_WHITESPACE_V1 TERM_hydraulic_pressure|STRASSE PERIOD"


@pytest.mark.parametrize(
    ("canonical_form", "observed_surface"),
    [
        ("Straße", "STRASSE"),
        ("STRASSE", "Straße"),
    ],
)
def test_unicode_casefold_term_symbols_round_trip(canonical_form, observed_surface, vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": canonical_form, "aliases": []})
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    lexicalizer = SymbolicLexicalizer(vocab, custom_terms)
    text = f"{observed_surface}."

    symbols = lexicalizer.symbolize(text)

    assert lexicalizer.lexicalize(symbols) == text


def test_neural_realizer_handles_sentence_initial_casefold_expansion(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "ß", "aliases": []})
                    if term.id == "access_panel"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    document = load_document(ROOT / "data/examples/installation.yaml")
    raw = document.model_dump(mode="json")
    raw["sections"][0]["statements"] = [
        {
            "kind": "state",
            "id": "state_001",
            "subject": {"term_id": "access_panel"},
            "predicate": "is",
            "value": "safe",
            "source_spans": [],
        }
    ]
    document = Document.model_validate(raw)
    lexicalizer = SymbolicLexicalizer(vocab, custom_terms)
    expected = DeterministicRealizer().realize(document, vocab, custom_terms)
    expected_plan = lexicalizer.symbolize(expected.text)
    assert expected.text == "SS is safe."
    assert "TERM_access_panel|SS" in expected_plan.split()

    class Generator:
        model_id = "offline-unicode-casefold-generator"

        def generate_symbols(self, serialized_ir, allowed_symbols):
            assert serialized_ir == canonical_document_json(document)
            assert allowed_symbols == frozenset(expected_plan.split())
            return expected_plan

    result = NeuralRealizer(Generator()).realize(document, vocab, custom_terms)

    assert result.text == expected.text
    assert result.mappings == expected.mappings


def test_symbolizer_does_not_match_casefolded_terms_inside_words(vocab, terms):
    custom_terms = TerminologyRegistry(
        terms.data.model_copy(
            update={
                "terms": [
                    term.model_copy(update={"canonical_form": "Straße", "aliases": []})
                    if term.id == "hydraulic_pressure"
                    else term
                    for term in terms.data.terms
                ]
            }
        )
    )
    template = vocab.data.entries[0]
    custom_vocab = Vocabulary(
        vocab.data.model_copy(
            update={
                "entries": [
                    *vocab.data.entries,
                    template.model_copy(update={"lemma": "XSTRASSE", "inflections": []}),
                    template.model_copy(update={"lemma": "STRASSEsafe", "inflections": []}),
                ]
            }
        )
    )

    symbols = SymbolicLexicalizer(custom_vocab, custom_terms).symbolize("XSTRASSE STRASSEsafe")

    assert symbols == ("PLAN_EXACT_WHITESPACE_V1 WORD_XSTRASSE SPACE WORD_STRASSEsafe")


@pytest.mark.parametrize("text", ["20°C.", "20m/s."])
def test_lexical_validator_accepts_units_adjacent_to_numbers(text, vocab, terms):
    custom_vocab = Vocabulary(
        vocab.data.model_copy(update={"units": [*vocab.data.units, "°C", "m/s"]})
    )

    assert not LexicalValidator(custom_vocab, terms).validate(text)


@pytest.mark.parametrize("text", ["safe°Cslowly.", "am/safe."])
def test_lexical_validator_does_not_mask_units_inside_alphabetic_words(text, vocab, terms):
    custom_vocab = Vocabulary(
        vocab.data.model_copy(update={"units": [*vocab.data.units, "°C", "m/s"]})
    )

    assert "UNAUTHORIZED_WORD" in {
        diagnostic.code for diagnostic in LexicalValidator(custom_vocab, terms).validate(text)
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_quantity_schema_rejects_nonfinite_values(value):
    with pytest.raises(ValidationError, match="finite number"):
        Quantity(value=value, unit="Pa")


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf")])
def test_quantity_schema_rejects_nonfinite_tolerances(tolerance):
    with pytest.raises(ValidationError, match="finite number"):
        Quantity(value=1, unit="Pa", tolerance=tolerance)


@pytest.mark.parametrize("unit", ["", " ", "\tMPa", "20", "-1e-07"])
def test_quantity_schema_rejects_degenerate_or_numeric_units(unit):
    with pytest.raises(
        ValidationError,
        match="nonblank|leading or trailing whitespace|numeric-only",
    ):
        Quantity(value=1, unit=unit)


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


def _causal_document() -> Document:
    raw = load_document(ROOT / "data/examples/installation.yaml").model_dump(mode="json")
    raw["sections"][0]["statements"][0]["source_spans"] = [
        {
            "source_id": "causal.txt",
            "start": 0,
            "end": 25,
            "quote": "Install the access panel.",
        }
    ]
    raw["sections"][0]["statements"].append(
        {
            "kind": "instruction",
            "id": "inspect_pump",
            "action": {"id": "inspect", "lemma": "inspect"},
            "object": {"id": "pump", "name": "pump"},
            "source_spans": [
                {
                    "source_id": "causal.txt",
                    "start": 75,
                    "end": 92,
                    "quote": "Inspect the pump.",
                }
            ],
        }
    )
    raw["causal_relations"] = [
        {
            "id": "installation_causes_inspection",
            "cause_node_id": "inst_001",
            "effect_node_id": "inspect_pump",
            "source_spans": [
                {
                    "source_id": "causal.txt",
                    "start": 26,
                    "end": 74,
                    "quote": "Installation of the panel causes pump inspection.",
                }
            ],
        }
    ]
    return Document.model_validate(raw)


def test_causal_relation_has_explicit_controlled_realization(vocab, terms):
    document = _causal_document()

    result = DeterministicRealizer().realize(document, vocab, terms)

    assert result.text == (
        "Install the access panel.\n"
        "Inspect the pump.\n"
        "Cause: Install the access panel; effect: Inspect the pump."
    )
    assert result.mappings[-1].ir_node_ids == (
        "installation_causes_inspection",
        "inst_001",
        "inspect_pump",
    )
    assert not SemanticValidator().validate(document, result)


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {"cause_node_id": "missing"},
            "causal relation endpoints must refer to statements",
        ),
        (
            {"effect_node_id": "inst_001"},
            "causal relation endpoints must be different",
        ),
        (
            {"id": "inst_001"},
            "causal relation id 'inst_001' is not unique",
        ),
    ],
)
def test_causal_relation_schema_rejects_invalid_graph(replacement, expected_message):
    document = _causal_document().model_dump(mode="json")
    document["causal_relations"][0].update(replacement)

    with pytest.raises(ValidationError, match=expected_message):
        Document.model_validate(document)


def test_causal_relation_schema_requires_claim_level_provenance():
    with pytest.raises(ValidationError, match="source_spans"):
        CausalRelation(
            id="causal",
            cause_node_id="cause",
            effect_node_id="effect",
        )


def test_causal_relation_schema_rejects_duplicate_pairs():
    document = _causal_document().model_dump(mode="json")
    duplicate = {
        **document["causal_relations"][0],
        "id": "duplicate_relation",
    }
    document["causal_relations"].append(duplicate)

    with pytest.raises(ValidationError, match="cause and effect pairs must be unique"):
        Document.model_validate(document)


def test_semantic_validator_rejects_changed_or_omitted_causal_relation(vocab, terms):
    document = _causal_document()
    result = DeterministicRealizer().realize(document, vocab, terms)
    changed_mapping = replace(result.mappings[-1], text="Cause: Inspect the pump.")
    changed = replace(
        result,
        text="\n".join([*(mapping.text for mapping in result.mappings[:-1]), changed_mapping.text]),
        mappings=(*result.mappings[:-1], changed_mapping),
    )
    omitted = replace(
        result,
        text="\n".join(mapping.text for mapping in result.mappings[:-1]),
        mappings=result.mappings[:-1],
    )

    assert "CAUSAL_RELATION_NOT_PRESERVED" in {
        diagnostic.code for diagnostic in SemanticValidator().validate(document, changed)
    }
    assert "CAUSAL_RELATION_NOT_PRESERVED" in {
        diagnostic.code for diagnostic in SemanticValidator().validate(document, omitted)
    }


def test_neural_alignment_cannot_inherit_changed_causal_relation(vocab, terms):
    document = _causal_document()
    expected = DeterministicRealizer().realize(document, vocab, terms)
    changed = expected.text.replace("effect: Inspect", "effect: Open")

    result = align_controlled_text(changed, expected)

    assert not result.mappings[-1].ir_node_ids
    assert {
        "CAUSAL_RELATION_NOT_PRESERVED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    } <= {diagnostic.code for diagnostic in SemanticValidator().validate(document, result)}


def test_llm_frontend_verifies_causal_relation_source_spans():
    document = _causal_document().model_dump(mode="json")
    document["causal_relations"][0]["source_spans"][0]["quote"] = "wrong"
    source = (
        "Install the access panel. "
        "Installation of the panel causes pump inspection. "
        "Inspect the pump."
    )

    class Provider:
        model_id = "test"

        def extract_ir(self, source, schema, feedback):
            del source, schema, feedback
            return document

    with pytest.raises(ValueError, match="quote does not match"):
        LLMFrontend(Provider(), retries=0).parse(source, source_id="causal.txt")


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

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ste_compiler.realizer.constrained import EXACT_PLAN_SYMBOL, SymbolicLexicalizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

ROOT = Path(__file__).parents[2]
VOCABULARY = Vocabulary.load(ROOT / "data/demo_vocabulary.yaml")
TERMINOLOGY = TerminologyRegistry.load(ROOT / "data/demo_terminology.yaml")
LEXICALIZER = SymbolicLexicalizer(VOCABULARY, TERMINOLOGY)

CONTROLLED_ATOMS = st.sampled_from(
    [
        "safe",
        "Safe",
        "open",
        "inspect",
        "access panel",
        "ACCESS PANEL",
        "hydraulic pressure",
        "Shutoff Valve",
        "20 MPa",
        "-1e-07 Pa",
    ]
)
SEPARATORS = st.sampled_from([" ", "\t", "\u2003", "\n"])
TRAILING_PUNCTUATION = st.sampled_from(["", ".", ",", ":", ";", "?", "!", "—"])


@st.composite
def controlled_text(draw: st.DrawFn) -> str:
    atoms = draw(st.lists(CONTROLLED_ATOMS, min_size=1, max_size=8))
    punctuation = draw(
        st.lists(
            TRAILING_PUNCTUATION,
            min_size=len(atoms),
            max_size=len(atoms),
        )
    )
    separators = draw(
        st.lists(
            SEPARATORS,
            min_size=max(0, len(atoms) - 1),
            max_size=max(0, len(atoms) - 1),
        )
    )
    pieces: list[str] = []
    for index, (atom, suffix) in enumerate(zip(atoms, punctuation, strict=True)):
        pieces.append(atom + suffix)
        if index < len(separators):
            pieces.append(separators[index])
    return "".join(pieces)


@settings(max_examples=80, deadline=None)
@given(text=controlled_text())
def test_exact_symbolic_plans_are_lossless_and_stable(text: str) -> None:
    plan = LEXICALIZER.symbolize(text)
    allowed_symbols = frozenset(plan.split())

    assert plan.split()[0] == EXACT_PLAN_SYMBOL
    assert LEXICALIZER.lexicalize(plan, allowed_symbols=allowed_symbols) == text
    assert LEXICALIZER.symbolize(LEXICALIZER.lexicalize(plan)) == plan


INVALID_SYMBOLS = st.sampled_from(
    [
        "WORD_commence",
        "TERM_missing|access%20panel",
        "UNIT_psi",
        "PUNCT_U0041",
        "WS_U000A",
        "NUMBER_nan",
    ]
)


@settings(max_examples=40, deadline=None)
@given(text=controlled_text(), invalid_symbol=INVALID_SYMBOLS)
def test_unauthorized_or_mutated_plan_symbols_fail_closed(
    text: str,
    invalid_symbol: str,
) -> None:
    plan = LEXICALIZER.symbolize(text)
    allowed_symbols = frozenset(plan.split())
    mutated = f"{plan} {invalid_symbol}"

    with pytest.raises(ValueError, match="not allowed"):
        LEXICALIZER.lexicalize(mutated, allowed_symbols=allowed_symbols)
    with pytest.raises(ValueError):
        LEXICALIZER.lexicalize(mutated)


@settings(max_examples=40, deadline=None)
@given(text=controlled_text())
def test_exact_plan_marker_cannot_be_moved_or_duplicated(text: str) -> None:
    plan = LEXICALIZER.symbolize(text)
    symbols = plan.split()
    payload = " ".join(symbols[1:])

    with pytest.raises(ValueError):
        LEXICALIZER.lexicalize(f"{payload} {EXACT_PLAN_SYMBOL}")
    with pytest.raises(ValueError, match="invalid output symbol"):
        LEXICALIZER.lexicalize(f"{plan} {EXACT_PLAN_SYMBOL}")

import re

from ste_compiler.diagnostics import Diagnostic, Severity
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.boundaries import mask_unit_surface

NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
TOKEN = re.compile(rf"{NUMBER}|[A-Za-z]+(?:-[A-Za-z]+)?|[^\w\s]")
WORD_CHARACTER = re.compile(r"\w")


def _casefold_view(text: str) -> tuple[str, dict[int, int]]:
    folded_parts: list[str] = []
    original_boundaries = {0: 0}
    folded_length = 0
    for index, character in enumerate(text):
        folded_character = character.casefold()
        folded_parts.append(folded_character)
        folded_length += len(folded_character)
        original_boundaries[folded_length] = index + 1
    return "".join(folded_parts), original_boundaries


def _whole_casefold_spans(text: str, form: str) -> tuple[tuple[int, int], ...]:
    folded_text, original_boundaries = _casefold_view(text)
    folded_form = form.casefold()
    if not folded_form:
        return ()

    spans: list[tuple[int, int]] = []
    position = 0
    while (start := folded_text.find(folded_form, position)) >= 0:
        end = start + len(folded_form)
        original_start = original_boundaries.get(start)
        original_end = original_boundaries.get(end)
        if original_start is not None and original_end is not None:
            left_is_word = (
                original_start > 0
                and WORD_CHARACTER.fullmatch(text[original_start - 1]) is not None
            )
            right_is_word = (
                original_end < len(text)
                and WORD_CHARACTER.fullmatch(text[original_end]) is not None
            )
            if not left_is_word and not right_is_word:
                spans.append((original_start, original_end))
        position = start + 1
    return tuple(spans)


def _mask_casefold_form(text: str, form: str) -> str:
    output = list(text)
    for start, end in _whole_casefold_spans(text, form):
        output[start:end] = " " * (end - start)
    return "".join(output)


class LexicalValidator:
    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    def validate(self, text: str) -> list[Diagnostic]:
        masked = text
        diagnostics: list[Diagnostic] = []
        for alias in sorted(self.terminology.aliases, key=len, reverse=True):
            if _whole_casefold_spans(masked, alias):
                diagnostics.append(
                    Diagnostic(
                        code="TERMINOLOGY_ALIAS",
                        severity=Severity.ERROR,
                        message=f"Use the canonical term instead of '{alias}'.",
                        suggestions=[self.terminology.normalize(alias) or ""],
                    )
                )
        for form in sorted(
            self.terminology.canonical_forms | self.terminology.aliases, key=len, reverse=True
        ):
            masked = _mask_casefold_form(masked, form)
        for unit in sorted(self.vocabulary.units, key=len, reverse=True):
            masked = mask_unit_surface(masked, unit)
        sentence = 1
        for match in TOKEN.finditer(masked):
            token = match.group()
            if token == ".":
                sentence += 1
            if re.fullmatch(NUMBER, token) or re.fullmatch(r"[^\w\s]", token):
                continue
            if self.vocabulary.contains(token):
                continue
            diagnostics.append(
                Diagnostic(
                    code="UNAUTHORIZED_WORD",
                    severity=Severity.ERROR,
                    message=f"The word '{token}' is not authorized.",
                    span={"sentence": sentence, "start": match.start(), "end": match.end()},
                )
            )
        return diagnostics

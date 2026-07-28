import re

from ste_compiler.diagnostics import Diagnostic, Severity
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.boundaries import (
    mask_casefold_form,
    mask_unit_surface,
    whole_casefold_spans,
)

NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
TOKEN = re.compile(rf"{NUMBER}|[A-Za-z]+(?:-[A-Za-z]+)?|[^\w\s]")


class LexicalValidator:
    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    def validate(self, text: str) -> list[Diagnostic]:
        masked = text
        diagnostics: list[Diagnostic] = []
        for alias in sorted(self.terminology.aliases, key=len, reverse=True):
            if whole_casefold_spans(masked, alias):
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
            masked = mask_casefold_form(masked, form)
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

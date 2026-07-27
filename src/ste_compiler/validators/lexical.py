import re
from ste_compiler.diagnostics import Diagnostic, Severity
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

TOKEN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")


class LexicalValidator:
    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    def validate(self, text: str) -> list[Diagnostic]:
        masked = text.casefold()
        diagnostics: list[Diagnostic] = []
        for alias in sorted(self.terminology.aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", masked):
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
            masked = re.sub(rf"\b{re.escape(form)}\b", " ", masked)
        sentence = 1
        for match in TOKEN.finditer(masked):
            token = match.group()
            if token == ".":
                sentence += 1
            if token.isnumeric() or re.fullmatch(r"\d+\.\d+|[^\w\s]", token):
                continue
            if self.vocabulary.contains(token) or token.casefold() in self.vocabulary.units:
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

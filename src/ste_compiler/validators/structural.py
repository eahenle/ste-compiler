import re
from ste_compiler.diagnostics import Diagnostic, Severity


class StructuralValidator:
    def __init__(self, max_sentence_words: int = 25, max_paragraph_sentences: int = 6):
        self.max_sentence_words = max_sentence_words
        self.max_paragraph_sentences = max_paragraph_sentences

    def validate(self, text: str) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        for number, sentence in enumerate(sentences, 1):
            if len(re.findall(r"\b[\w-]+\b", sentence)) > self.max_sentence_words:
                out.append(
                    Diagnostic(
                        code="SENTENCE_TOO_LONG",
                        severity=Severity.ERROR,
                        message=f"Sentence {number} exceeds {self.max_sentence_words} words.",
                    )
                )
            if re.search(r"\b(it|they|this|that)\b", sentence, re.I):
                out.append(
                    Diagnostic(
                        code="AMBIGUOUS_PRONOUN",
                        severity=Severity.WARNING,
                        message=f"Sentence {number} can contain an ambiguous pronoun.",
                    )
                )
            if re.search(r"\b(is|are|was|were|be|been)\s+\w+ed\b", sentence, re.I):
                out.append(
                    Diagnostic(
                        code="PASSIVE_VOICE",
                        severity=Severity.WARNING,
                        message=f"Sentence {number} can contain passive voice.",
                    )
                )
        for paragraph in text.split("\n\n"):
            if len(re.findall(r"[.!?](?:\s|$)", paragraph)) > self.max_paragraph_sentences:
                out.append(
                    Diagnostic(
                        code="PARAGRAPH_TOO_LONG",
                        severity=Severity.ERROR,
                        message="A paragraph has too many sentences.",
                    )
                )
        return out

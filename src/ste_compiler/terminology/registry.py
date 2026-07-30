from pathlib import Path

import yaml

from ste_compiler.terminology.models import Term, TerminologyData, VocabularyData


class TerminologyRegistry:
    def __init__(self, data: TerminologyData):
        self.data = TerminologyData.model_validate(data.model_dump())
        self._terms = {term.id: term for term in self.data.terms}
        self._forms = {
            form.casefold(): term
            for term in self.data.terms
            for form in [term.canonical_form, *term.aliases]
        }

    @classmethod
    def load(cls, path: Path) -> "TerminologyRegistry":
        return cls(TerminologyData.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))

    def get(self, term_id: str) -> Term:
        visited: set[str] = set()
        term = self._terms[term_id]
        while term.status == "deprecated" and term.replacement_term_id:
            if term.id in visited:
                raise ValueError(f"deprecated terminology replacement cycle includes {term.id!r}")
            visited.add(term.id)
            term = self._terms[term.replacement_term_id]
        if term.status != "approved":
            raise ValueError(f"term {term_id!r} is not approved")
        return term

    def normalize(self, form: str) -> str | None:
        term = self._forms.get(form.casefold())
        return self.get(term.id).canonical_form if term else None

    @property
    def canonical_forms(self) -> set[str]:
        return {t.canonical_form.casefold() for t in self.data.terms if t.status == "approved"}

    @property
    def aliases(self) -> set[str]:
        return {a.casefold() for t in self.data.terms for a in t.aliases}

    @property
    def approved_terms(self) -> tuple[Term, ...]:
        return tuple(term for term in self.data.terms if term.status == "approved")


class Vocabulary:
    def __init__(self, data: VocabularyData):
        self.data = VocabularyData.model_validate(data.model_dump())
        self.word_forms: dict[str, str] = {}
        for entry in self.data.entries:
            for word in [entry.lemma, *entry.inflections]:
                self.word_forms[word.casefold()] = word
        self.words = set(self.word_forms)
        self.unit_forms = {unit: unit for unit in self.data.units}
        self.units = set(self.unit_forms)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(VocabularyData.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))

    def contains(self, word: str) -> bool:
        return word.casefold() in self.words

    def canonical_word(self, word: str) -> str | None:
        return self.word_forms.get(word.casefold())

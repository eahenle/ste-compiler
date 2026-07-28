from pathlib import Path

import yaml

from ste_compiler.terminology.models import Term, TerminologyData, VocabularyData


class TerminologyRegistry:
    def __init__(self, data: TerminologyData):
        self.data = data
        self._terms = {term.id: term for term in data.terms}
        self._forms = {
            form.casefold(): term
            for term in data.terms
            for form in [term.canonical_form, *term.aliases]
        }

    @classmethod
    def load(cls, path: Path) -> "TerminologyRegistry":
        return cls(TerminologyData.model_validate(yaml.safe_load(path.read_text())))

    def get(self, term_id: str) -> Term:
        term = self._terms[term_id]
        if term.status == "deprecated" and term.replacement_term_id:
            return self.get(term.replacement_term_id)
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
        self.data = data
        self.words = {
            word.casefold() for entry in data.entries for word in [entry.lemma, *entry.inflections]
        }
        self.unit_forms = {unit.casefold(): unit for unit in data.units}
        self.units = set(self.unit_forms)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(VocabularyData.model_validate(yaml.safe_load(path.read_text())))

    def contains(self, word: str) -> bool:
        return word.casefold() in self.words

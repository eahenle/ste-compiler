import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

NUMBER_SURFACE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
WORD_SURFACE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")


def _stripped_nonblank(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("must be nonblank and have no leading or trailing whitespace")
    return value


def _non_numeric_surface(value: str) -> str:
    if NUMBER_SURFACE.fullmatch(value):
        raise ValueError("must not be a numeric-only surface form")
    return value


def _word_surface(value: str) -> str:
    if WORD_SURFACE.fullmatch(value) is None:
        raise ValueError("must be one ASCII word with at most one internal hyphen")
    return value


NonEmptyString = Annotated[str, AfterValidator(_stripped_nonblank)]
TerminologyForm = Annotated[
    str,
    AfterValidator(_stripped_nonblank),
    AfterValidator(_non_numeric_surface),
]
WordForm = Annotated[
    str,
    AfterValidator(_stripped_nonblank),
    AfterValidator(_word_surface),
]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    reference: str


class Term(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: NonEmptyString
    canonical_form: TerminologyForm
    definition: str
    domain: str
    allowed_roles: list[str] = Field(min_length=1)
    aliases: list[TerminologyForm] = []
    source: Provenance
    status: Literal["approved", "deprecated", "draft"] = "approved"
    version: int = Field(ge=1)
    replacement_term_id: NonEmptyString | None = None


class TerminologyData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    license: str
    terms: list[Term]

    @model_validator(mode="after")
    def validate_registry_invariants(self) -> Self:
        terms_by_id: dict[str, Term] = {}
        for term in self.terms:
            if term.id in terms_by_id:
                raise ValueError(f"duplicate terminology ID: {term.id!r}")
            terms_by_id[term.id] = term

        form_owners: dict[str, tuple[str, str]] = {}
        for term in self.terms:
            for kind, form in [
                ("canonical form", term.canonical_form),
                *(("alias", alias) for alias in term.aliases),
            ]:
                key = form.casefold()
                prior = form_owners.get(key)
                if prior is not None:
                    prior_id, prior_kind = prior
                    raise ValueError(
                        f"duplicate case-insensitive terminology form {form!r}: "
                        f"{prior_kind} of {prior_id!r} and {kind} of {term.id!r}"
                    )
                form_owners[key] = (term.id, kind)

        for term in self.terms:
            replacement = term.replacement_term_id
            if replacement is not None and replacement not in terms_by_id:
                raise ValueError(f"replacement term {replacement!r} for {term.id!r} does not exist")

        for term in self.terms:
            path: set[str] = set()
            current = term
            while current.status == "deprecated" and current.replacement_term_id is not None:
                if current.id in path:
                    raise ValueError(
                        f"deprecated terminology replacement cycle includes {current.id!r}"
                    )
                path.add(current.id)
                current = terms_by_id[current.replacement_term_id]
        return self


class VocabularyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lemma: WordForm
    parts_of_speech: list[str]
    meaning_id: str | None = None
    inflections: list[WordForm] = []
    example: str | None = None
    forbidden_confusions: list[str] = []


class VocabularyData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    license: str
    units: list[TerminologyForm] = []
    structural_tokens: list[str] = []
    entries: list[VocabularyEntry]

    @model_validator(mode="after")
    def validate_vocabulary_invariants(self) -> Self:
        word_owners: dict[str, tuple[int, str]] = {}
        for index, entry in enumerate(self.entries):
            for kind, word in [
                ("lemma", entry.lemma),
                *(("inflection", inflection) for inflection in entry.inflections),
            ]:
                key = word.casefold()
                prior = word_owners.get(key)
                if prior is not None:
                    prior_index, prior_kind = prior
                    raise ValueError(
                        f"duplicate case-insensitive vocabulary form {word!r}: "
                        f"{prior_kind} of entry {prior_index} and {kind} of entry {index}"
                    )
                word_owners[key] = (index, kind)

        seen_units: set[str] = set()
        for unit in self.units:
            if unit in seen_units:
                raise ValueError(f"duplicate unit surface: {unit!r}")
            seen_units.add(unit)
        return self

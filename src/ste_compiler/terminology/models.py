from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    reference: str


class Term(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    canonical_form: str
    definition: str
    domain: str
    allowed_roles: list[str] = Field(min_length=1)
    aliases: list[str] = []
    source: Provenance
    status: Literal["approved", "deprecated", "draft"] = "approved"
    version: int = Field(ge=1)
    replacement_term_id: str | None = None


class TerminologyData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    license: str
    terms: list[Term]


class VocabularyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lemma: str
    parts_of_speech: list[str]
    meaning_id: str | None = None
    inflections: list[str] = []
    example: str | None = None
    forbidden_confusions: list[str] = []


class VocabularyData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    license: str
    units: list[str] = []
    structural_tokens: list[str] = []
    entries: list[VocabularyEntry]

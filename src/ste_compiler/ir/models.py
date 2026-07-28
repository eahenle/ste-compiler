"""Typed semantic intermediate representation; no final prose belongs here."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

NUMBER_SURFACE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SectionKind(StrEnum):
    PROCEDURE = "procedure"
    DESCRIPTION = "description"
    WARNING = "warning"
    CAUTION = "caution"
    NOTE = "note"


class SourceSpan(StrictModel):
    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str | None = None

    @model_validator(mode="after")
    def ordered(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be after start")
        return self


class EntityRef(StrictModel):
    id: str
    name: str


class ActionRef(StrictModel):
    id: str
    lemma: str


class TermReference(StrictModel):
    term_id: str


class Quantity(StrictModel):
    value: FiniteFloat
    unit: str
    tolerance: FiniteFloat | None = Field(default=None, ge=0)
    comparator: Literal["equal", "less_than", "more_than", "at_most", "at_least"] = "equal"

    @field_validator("unit")
    @classmethod
    def symbolic_unit_surface(cls, unit: str) -> str:
        if not unit or unit != unit.strip():
            raise ValueError("unit must be nonblank and have no leading or trailing whitespace")
        if NUMBER_SURFACE.fullmatch(unit):
            raise ValueError("unit must not be a numeric-only surface form")
        return unit


class Measurement(StrictModel):
    property: str
    quantity: Quantity


class Condition(StrictModel):
    id: str
    subject: EntityRef | TermReference
    predicate: str
    value: str | Quantity
    exception: bool = False


class TemporalRelation(StrictModel):
    relation: Literal["before", "after"]
    event: str


class CausalRelation(StrictModel):
    cause_node_id: str
    effect_node_id: str


class Hazard(StrictModel):
    id: str
    severity: Literal["warning", "caution"]
    consequence: str
    threshold: Quantity | None = None


class Reference(StrictModel):
    target: str
    label: str | None = None


class Ambiguity(StrictModel):
    id: str
    description: str
    alternatives: list[str] = Field(min_length=1)
    source_spans: list[SourceSpan] = Field(min_length=1)


class QuantityConstraint(StrictModel):
    property: str
    quantity: Quantity


Referent = EntityRef | TermReference


class Instruction(StrictModel):
    kind: Literal["instruction"] = "instruction"
    id: str
    actor: EntityRef | None = None
    action: ActionRef
    object: Referent | None = None
    indirect_object: Referent | None = None
    conditions: list[Condition] = []
    temporal_relations: list[TemporalRelation] = []
    manner: str | None = None
    purpose: str | None = None
    negated: bool = False
    quantity_constraints: list[QuantityConstraint] = []
    hazards: list[Hazard] = []
    source_spans: list[SourceSpan] = []
    required: bool = True


class StateAssertion(StrictModel):
    kind: Literal["state"] = "state"
    id: str
    subject: Referent
    predicate: str
    value: str | Quantity
    source_spans: list[SourceSpan] = []


Statement = Annotated[Instruction | StateAssertion, Field(discriminator="kind")]


class Section(StrictModel):
    id: str
    kind: SectionKind
    title: str | None = None
    statements: list[Statement] = Field(min_length=1)


class ReproducibilityMetadata(StrictModel):
    frontend: str = "manual"
    frontend_version: str = "0.1.0"
    realizer: str = "deterministic"
    realizer_version: str = "0.1.0"
    vocabulary_version: str = "demo-1"
    terminology_version: str = "hydraulic-demo-1"
    validator_profile: str = "strict-demo-1"


class Document(StrictModel):
    id: str
    title: str | None = None
    sections: list[Section] = Field(min_length=1)
    ambiguities: list[Ambiguity] = []
    causal_relations: list[CausalRelation] = []
    references: list[Reference] = []
    metadata: ReproducibilityMetadata = ReproducibilityMetadata()

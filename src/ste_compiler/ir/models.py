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
    source_id: str = Field(min_length=1, pattern=r"\S")
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
    id: str
    cause_node_id: str
    effect_node_id: str
    source_spans: list[SourceSpan] = Field(min_length=1)


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
    realizer_version: str = "0.2.0"
    vocabulary_version: str = "demo-3"
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

    @model_validator(mode="after")
    def valid_causal_graph(self) -> Document:
        ordered_statement_ids = [
            statement.id for section in self.sections for statement in section.statements
        ]
        statement_ids = set(ordered_statement_ids)
        if len(statement_ids) != len(ordered_statement_ids):
            duplicate_ids = sorted(
                {
                    statement_id
                    for statement_id in statement_ids
                    if ordered_statement_ids.count(statement_id) > 1
                }
            )
            raise ValueError("statement ids must be unique: " + ", ".join(duplicate_ids))
        occupied_ids = set(statement_ids)
        occupied_ids.update(
            hazard.id
            for section in self.sections
            for statement in section.statements
            if isinstance(statement, Instruction)
            for hazard in statement.hazards
        )
        occupied_ids.update(ambiguity.id for ambiguity in self.ambiguities)
        relation_pairs: set[tuple[str, str]] = set()
        for relation in self.causal_relations:
            if relation.id in occupied_ids:
                raise ValueError(f"causal relation id {relation.id!r} is not unique")
            occupied_ids.add(relation.id)
            if relation.cause_node_id == relation.effect_node_id:
                raise ValueError("causal relation endpoints must be different")
            missing = {
                endpoint
                for endpoint in (relation.cause_node_id, relation.effect_node_id)
                if endpoint not in statement_ids
            }
            if missing:
                raise ValueError(
                    "causal relation endpoints must refer to statements: "
                    + ", ".join(sorted(missing))
                )
            pair = (relation.cause_node_id, relation.effect_node_id)
            if pair in relation_pairs:
                raise ValueError(
                    "causal relation cause and effect pairs must be unique: "
                    f"{relation.cause_node_id!r} -> {relation.effect_node_id!r}"
                )
            relation_pairs.add(pair)
        return self

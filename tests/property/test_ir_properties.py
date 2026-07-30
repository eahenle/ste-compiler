from copy import deepcopy
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from ste_compiler.ir.models import (
    ActionRef,
    Ambiguity,
    CausalRelation,
    Condition,
    Document,
    EntityRef,
    Hazard,
    Instruction,
    Measurement,
    Quantity,
    QuantityConstraint,
    Reference,
    ReproducibilityMetadata,
    Section,
    SectionKind,
    SourceSpan,
    StateAssertion,
    TemporalRelation,
    TermReference,
)
from ste_compiler.ir.serialization import (
    canonical_document_json,
    dumps_document,
    loads_document,
)

Path = tuple[str | int, ...]


def complete_document() -> Document:
    """Return one valid document containing every Document-reachable IR model."""
    return Document(
        id="complete_document",
        title="Complete semantic IR",
        sections=[
            Section(
                id="procedure",
                kind=SectionKind.PROCEDURE,
                title="Complete instruction",
                statements=[
                    Instruction(
                        id="inspect_panel",
                        actor=EntityRef(id="technician", name="technician"),
                        action=ActionRef(id="inspect", lemma="inspect"),
                        object=TermReference(term_id="access_panel"),
                        indirect_object=EntityRef(id="assembly", name="assembly"),
                        conditions=[
                            Condition(
                                id="pressure_condition",
                                subject=EntityRef(id="system", name="system"),
                                predicate="has pressure",
                                value=Quantity(
                                    value=20,
                                    unit="MPa",
                                    tolerance=0.5,
                                    comparator="at_most",
                                ),
                                exception=True,
                            ),
                            Condition(
                                id="mode_condition",
                                subject=TermReference(term_id="operating_mode"),
                                predicate="is",
                                value="maintenance",
                            ),
                        ],
                        temporal_relations=[
                            TemporalRelation(relation="before", event="the test"),
                            TemporalRelation(relation="after", event="isolation"),
                        ],
                        manner="carefully",
                        purpose="find damage",
                        negated=True,
                        quantity_constraints=[
                            QuantityConstraint(
                                property="clearance",
                                quantity=Quantity(
                                    value=3,
                                    unit="mm",
                                    tolerance=0.25,
                                    comparator="equal",
                                ),
                            )
                        ],
                        hazards=[
                            Hazard(
                                id="pressure_hazard",
                                severity="warning",
                                consequence="injury",
                                threshold=Quantity(
                                    value=25,
                                    unit="MPa",
                                    comparator="more_than",
                                ),
                            )
                        ],
                        source_spans=[
                            SourceSpan(
                                source_id="complete.txt",
                                start=0,
                                end=10,
                                quote="Inspect it",
                            )
                        ],
                        required=False,
                    )
                ],
            ),
            Section(
                id="description",
                kind=SectionKind.DESCRIPTION,
                statements=[
                    StateAssertion(
                        id="pressure_state",
                        subject=TermReference(term_id="hydraulic_pressure"),
                        predicate="is",
                        value=Quantity(
                            value=15,
                            unit="MPa",
                            comparator="at_least",
                        ),
                        source_spans=[
                            SourceSpan(
                                source_id="complete.txt",
                                start=11,
                                end=20,
                                quote="15 MPa.",
                            )
                        ],
                    ),
                    StateAssertion(
                        id="unit_state",
                        subject=EntityRef(id="unit", name="unit"),
                        predicate="is",
                        value="safe",
                        source_spans=[
                            SourceSpan(
                                source_id="complete.txt",
                                start=21,
                                end=30,
                                quote="is safe.",
                            )
                        ],
                    ),
                ],
            ),
        ],
        ambiguities=[
            Ambiguity(
                id="switch_position",
                description="The switch position is unresolved.",
                alternatives=["on", "off"],
                source_spans=[
                    SourceSpan(
                        source_id="complete.txt",
                        start=31,
                        end=40,
                        quote="on or off",
                    )
                ],
            )
        ],
        causal_relations=[
            CausalRelation(
                id="inspection_reveals_pressure",
                cause_node_id="inspect_panel",
                effect_node_id="pressure_state",
                source_spans=[
                    SourceSpan(
                        source_id="complete.txt",
                        start=41,
                        end=50,
                        quote="causes it",
                    )
                ],
            )
        ],
        references=[
            Reference(
                target="synthetic-manual-section-1",
                label="system manual",
            )
        ],
        metadata=ReproducibilityMetadata(
            frontend="property-test",
            frontend_version="1",
            realizer="deterministic",
            realizer_version="1",
            vocabulary_version="vocabulary-1",
            terminology_version="terminology-1",
            validator_profile="validator-1",
        ),
    )


@st.composite
def documents(draw: st.DrawFn) -> Document:
    indexes = draw(
        st.lists(
            st.integers(min_value=0, max_value=1_000_000),
            min_size=2,
            max_size=6,
            unique=True,
        )
    )
    statement_ids = [f"node_{index}" for index in indexes]
    entity_names = draw(
        st.lists(
            st.from_regex(r"[A-Za-z][A-Za-z -]{0,24}", fullmatch=True),
            min_size=len(statement_ids),
            max_size=len(statement_ids),
        )
    )
    statements = [
        Instruction(
            id=statement_id,
            actor=EntityRef(id=f"actor_{index}", name="technician"),
            action=ActionRef(id=f"action_{index}", lemma="inspect"),
            object=EntityRef(id=f"entity_{index}", name=entity_name),
            required=draw(st.booleans()),
            negated=draw(st.booleans()),
            source_spans=[
                SourceSpan(
                    source_id="property-source",
                    start=index,
                    end=index + 1,
                    quote=chr(ord("a") + index % 26),
                )
            ],
        )
        if index % 2 == 0
        else StateAssertion(
            id=statement_id,
            subject=EntityRef(id=f"entity_{index}", name=entity_name),
            predicate="is",
            value=draw(st.sampled_from(["open", "closed", "safe"])),
            source_spans=[
                SourceSpan(
                    source_id="property-source",
                    start=index,
                    end=index + 1,
                    quote=chr(ord("a") + index % 26),
                )
            ],
        )
        for index, (statement_id, entity_name) in enumerate(
            zip(statement_ids, entity_names, strict=True)
        )
    ]
    return Document(
        id=f"document_{indexes[0]}",
        title=draw(st.one_of(st.none(), st.sampled_from(["Procedure", "Inspection"]))),
        sections=[
            Section(
                id="section_0",
                kind=draw(st.sampled_from(list(SectionKind))),
                statements=statements,
            )
        ],
        causal_relations=[
            CausalRelation(
                id=f"relation_{indexes[0]}_{indexes[1]}",
                cause_node_id=statement_ids[0],
                effect_node_id=statement_ids[1],
                source_spans=[
                    SourceSpan(
                        source_id="property-source",
                        start=len(statement_ids),
                        end=len(statement_ids) + 1,
                        quote="z",
                    )
                ],
            )
        ],
    )


def _assert_document_round_trip(document: Document) -> None:
    json_round_trip = loads_document(dumps_document(document, as_json=True), ".json")
    yaml_round_trip = loads_document(dumps_document(document), ".yaml")

    assert json_round_trip == document
    assert yaml_round_trip == document
    assert canonical_document_json(json_round_trip) == canonical_document_json(document)
    assert canonical_document_json(yaml_round_trip) == canonical_document_json(document)


@settings(max_examples=50, deadline=None, derandomize=True)
@given(document=documents())
def test_strict_ir_round_trips_through_json_and_yaml(document: Document) -> None:
    _assert_document_round_trip(document)


def test_complete_ir_round_trips_through_json_and_yaml() -> None:
    _assert_document_round_trip(complete_document())


NESTED_MODEL_LOCATIONS: tuple[tuple[str, Path], ...] = (
    ("Document", ()),
    ("Section", ("sections", 0)),
    ("Instruction", ("sections", 0, "statements", 0)),
    ("Instruction.actor.EntityRef", ("sections", 0, "statements", 0, "actor")),
    ("Instruction.action.ActionRef", ("sections", 0, "statements", 0, "action")),
    ("Instruction.object.TermReference", ("sections", 0, "statements", 0, "object")),
    (
        "Instruction.indirect_object.EntityRef",
        ("sections", 0, "statements", 0, "indirect_object"),
    ),
    ("Condition", ("sections", 0, "statements", 0, "conditions", 0)),
    (
        "Condition.subject.EntityRef",
        ("sections", 0, "statements", 0, "conditions", 0, "subject"),
    ),
    (
        "Condition.value.Quantity",
        ("sections", 0, "statements", 0, "conditions", 0, "value"),
    ),
    (
        "Condition.subject.TermReference",
        ("sections", 0, "statements", 0, "conditions", 1, "subject"),
    ),
    ("TemporalRelation", ("sections", 0, "statements", 0, "temporal_relations", 0)),
    ("QuantityConstraint", ("sections", 0, "statements", 0, "quantity_constraints", 0)),
    (
        "QuantityConstraint.quantity.Quantity",
        ("sections", 0, "statements", 0, "quantity_constraints", 0, "quantity"),
    ),
    ("Hazard", ("sections", 0, "statements", 0, "hazards", 0)),
    (
        "Hazard.threshold.Quantity",
        ("sections", 0, "statements", 0, "hazards", 0, "threshold"),
    ),
    ("Instruction.SourceSpan", ("sections", 0, "statements", 0, "source_spans", 0)),
    ("StateAssertion", ("sections", 1, "statements", 0)),
    (
        "StateAssertion.subject.TermReference",
        ("sections", 1, "statements", 0, "subject"),
    ),
    (
        "StateAssertion.value.Quantity",
        ("sections", 1, "statements", 0, "value"),
    ),
    ("StateAssertion.SourceSpan", ("sections", 1, "statements", 0, "source_spans", 0)),
    (
        "StateAssertion.subject.EntityRef",
        ("sections", 1, "statements", 1, "subject"),
    ),
    ("Ambiguity", ("ambiguities", 0)),
    ("Ambiguity.SourceSpan", ("ambiguities", 0, "source_spans", 0)),
    ("CausalRelation", ("causal_relations", 0)),
    ("CausalRelation.SourceSpan", ("causal_relations", 0, "source_spans", 0)),
    ("Reference", ("references", 0)),
    ("ReproducibilityMetadata", ("metadata",)),
)

DOCUMENT_MODEL_TYPES: tuple[type[BaseModel], ...] = (
    ActionRef,
    Ambiguity,
    CausalRelation,
    Condition,
    Document,
    EntityRef,
    Hazard,
    Instruction,
    Quantity,
    QuantityConstraint,
    Reference,
    ReproducibilityMetadata,
    Section,
    SourceSpan,
    StateAssertion,
    TemporalRelation,
    TermReference,
)
DOCUMENT_MODEL_BY_NAME = {model.__name__: model for model in DOCUMENT_MODEL_TYPES}


def test_nested_boundary_matrix_tracks_every_document_schema_model() -> None:
    schema_models = set(Document.model_json_schema()["$defs"]) - {"SectionKind"}
    covered_models = {
        boundary.rsplit(".", maxsplit=1)[-1] for boundary, _ in NESTED_MODEL_LOCATIONS
    }

    assert covered_models == schema_models | {"Document"}
    assert set(DOCUMENT_MODEL_BY_NAME) == covered_models


def _model_field_cases(
    locations: tuple[tuple[str, Path], ...],
    models: dict[str, type[BaseModel]],
    *,
    required: bool,
) -> tuple[tuple[str, Path, str], ...]:
    return tuple(
        (boundary, location, field_name)
        for boundary, location in locations
        for field_name, field in models[boundary.rsplit(".", maxsplit=1)[-1]].model_fields.items()
        if field.is_required() is required and not (field_name == "kind" and not required)
    )


def _nested_value(raw: object, location: Path) -> object:
    current = raw
    for component in location:
        if isinstance(component, int):
            assert isinstance(current, list)
            current = current[component]
        else:
            assert isinstance(current, dict)
            current = current[component]
    return current


def _nested_object(raw: dict[str, object], location: Path) -> dict[str, object]:
    current = _nested_value(raw, location)
    assert isinstance(current, dict)
    return current


REQUIRED_FIELD_CASES = _model_field_cases(
    NESTED_MODEL_LOCATIONS,
    DOCUMENT_MODEL_BY_NAME,
    required=True,
)

DEFAULTED_FIELD_CASES = _model_field_cases(
    NESTED_MODEL_LOCATIONS,
    DOCUMENT_MODEL_BY_NAME,
    required=False,
)


@pytest.mark.parametrize(
    ("boundary", "location", "field_name"),
    REQUIRED_FIELD_CASES,
    ids=[f"{boundary}.{field_name}" for boundary, _, field_name in REQUIRED_FIELD_CASES],
)
def test_every_required_field_fails_closed_when_deleted(
    boundary: str,
    location: Path,
    field_name: str,
) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    del _nested_object(raw, location)[field_name]

    with pytest.raises(ValidationError) as raised:
        Document.model_validate(raw)

    assert any(
        error["type"] == "missing" and error["loc"][-1] == field_name
        for error in raised.value.errors()
    ), boundary


@pytest.mark.parametrize(
    ("boundary", "location", "field_name"),
    DEFAULTED_FIELD_CASES,
    ids=[f"{boundary}.{field_name}" for boundary, _, field_name in DEFAULTED_FIELD_CASES],
)
def test_every_defaulted_field_accepts_omission(
    boundary: str,
    location: Path,
    field_name: str,
) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    del _nested_object(raw, location)[field_name]

    validated = Document.model_validate(raw)
    restored = _nested_object(validated.model_dump(mode="json"), location)

    assert field_name in restored, boundary


@pytest.mark.parametrize(
    "location",
    [
        ("sections", 0, "statements", 0),
        ("sections", 1, "statements", 0),
    ],
    ids=["Instruction.kind", "StateAssertion.kind"],
)
def test_discriminated_statement_requires_kind_even_though_models_default_it(
    location: Path,
) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    del _nested_object(raw, location)["kind"]

    with pytest.raises(ValidationError) as raised:
        Document.model_validate(raw)

    assert "union_tag_not_found" in {error["type"] for error in raised.value.errors()}


@pytest.mark.parametrize(
    ("boundary", "location"),
    NESTED_MODEL_LOCATIONS,
    ids=[boundary for boundary, _ in NESTED_MODEL_LOCATIONS],
)
def test_ir_models_reject_unknown_fields_at_every_nested_boundary(
    boundary: str,
    location: Path,
) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    _nested_object(raw, location)["unrecognized_property"] = boundary

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Document.model_validate(raw)


@dataclass(frozen=True)
class SchemaMutation:
    name: str
    path: Path
    replacement: object
    error_type: str


SCHEMA_MUTATIONS = (
    SchemaMutation("document_requires_sections", ("sections",), [], "too_short"),
    SchemaMutation(
        "section_requires_known_kind",
        ("sections", 0, "kind"),
        "appendix",
        "enum",
    ),
    SchemaMutation(
        "section_requires_statements",
        ("sections", 0, "statements"),
        [],
        "too_short",
    ),
    SchemaMutation(
        "statement_requires_known_discriminator",
        ("sections", 0, "statements", 0, "kind"),
        "command",
        "union_tag_invalid",
    ),
    SchemaMutation(
        "source_span_requires_nonblank_source",
        ("sections", 0, "statements", 0, "source_spans", 0, "source_id"),
        " ",
        "string_pattern_mismatch",
    ),
    SchemaMutation(
        "source_span_start_is_nonnegative",
        ("sections", 0, "statements", 0, "source_spans", 0, "start"),
        -1,
        "greater_than_equal",
    ),
    SchemaMutation(
        "source_span_end_is_positive",
        ("sections", 0, "statements", 0, "source_spans", 0, "end"),
        0,
        "greater_than",
    ),
    SchemaMutation(
        "source_span_end_follows_start",
        ("sections", 0, "statements", 0, "source_spans", 0, "start"),
        10,
        "value_error",
    ),
    SchemaMutation(
        "quantity_value_is_finite",
        ("sections", 0, "statements", 0, "quantity_constraints", 0, "quantity", "value"),
        float("inf"),
        "finite_number",
    ),
    SchemaMutation(
        "quantity_tolerance_is_nonnegative",
        (
            "sections",
            0,
            "statements",
            0,
            "quantity_constraints",
            0,
            "quantity",
            "tolerance",
        ),
        -0.1,
        "greater_than_equal",
    ),
    SchemaMutation(
        "quantity_tolerance_is_finite",
        (
            "sections",
            0,
            "statements",
            0,
            "quantity_constraints",
            0,
            "quantity",
            "tolerance",
        ),
        float("inf"),
        "finite_number",
    ),
    SchemaMutation(
        "quantity_unit_is_nonblank",
        ("sections", 0, "statements", 0, "quantity_constraints", 0, "quantity", "unit"),
        "",
        "value_error",
    ),
    SchemaMutation(
        "quantity_unit_is_trimmed",
        ("sections", 0, "statements", 0, "quantity_constraints", 0, "quantity", "unit"),
        " MPa",
        "value_error",
    ),
    SchemaMutation(
        "quantity_unit_is_not_numeric",
        ("sections", 0, "statements", 0, "quantity_constraints", 0, "quantity", "unit"),
        "12",
        "value_error",
    ),
    SchemaMutation(
        "quantity_requires_known_comparator",
        (
            "sections",
            0,
            "statements",
            0,
            "quantity_constraints",
            0,
            "quantity",
            "comparator",
        ),
        "approximately",
        "literal_error",
    ),
    SchemaMutation(
        "temporal_relation_requires_known_relation",
        ("sections", 0, "statements", 0, "temporal_relations", 0, "relation"),
        "during",
        "literal_error",
    ),
    SchemaMutation(
        "hazard_requires_known_severity",
        ("sections", 0, "statements", 0, "hazards", 0, "severity"),
        "danger",
        "literal_error",
    ),
    SchemaMutation(
        "ambiguity_requires_alternative",
        ("ambiguities", 0, "alternatives"),
        [],
        "too_short",
    ),
    SchemaMutation(
        "ambiguity_requires_provenance",
        ("ambiguities", 0, "source_spans"),
        [],
        "too_short",
    ),
    SchemaMutation(
        "causal_relation_requires_provenance",
        ("causal_relations", 0, "source_spans"),
        [],
        "too_short",
    ),
)


def _apply_schema_mutation(raw: dict[str, object], mutation: SchemaMutation) -> None:
    parent = _nested_value(raw, mutation.path[:-1])
    leaf = mutation.path[-1]
    if isinstance(leaf, int):
        assert isinstance(parent, list)
        parent[leaf] = mutation.replacement
    else:
        assert isinstance(parent, dict)
        parent[leaf] = mutation.replacement


@pytest.mark.parametrize(
    "mutation",
    SCHEMA_MUTATIONS,
    ids=[mutation.name for mutation in SCHEMA_MUTATIONS],
)
def test_nested_schema_mutations_fail_closed(mutation: SchemaMutation) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    _apply_schema_mutation(raw, mutation)

    with pytest.raises(ValidationError) as raised:
        Document.model_validate(raw)

    assert mutation.error_type in {error["type"] for error in raised.value.errors()}


CAUSAL_MUTATIONS = (
    "missing_endpoint",
    "self_loop",
    "duplicate_pair",
    "duplicate_relation_id",
    "statement_id_collision",
    "hazard_id_collision",
    "ambiguity_id_collision",
    "duplicate_statement_id",
)


@pytest.mark.parametrize("mutation", CAUSAL_MUTATIONS)
def test_causal_graph_mutations_fail_closed(mutation: str) -> None:
    raw = deepcopy(complete_document().model_dump(mode="json"))
    relation = raw["causal_relations"][0]
    assert isinstance(relation, dict)
    statements = raw["sections"][1]["statements"]
    assert isinstance(statements, list)
    assert isinstance(statements[0], dict)
    assert isinstance(statements[1], dict)

    if mutation == "missing_endpoint":
        relation["effect_node_id"] = "missing_node"
    elif mutation == "self_loop":
        relation["effect_node_id"] = relation["cause_node_id"]
    elif mutation == "duplicate_pair":
        duplicate = deepcopy(relation)
        duplicate["id"] = "second_relation"
        raw["causal_relations"].append(duplicate)
    elif mutation == "duplicate_relation_id":
        duplicate = deepcopy(relation)
        duplicate["cause_node_id"] = "pressure_state"
        duplicate["effect_node_id"] = "unit_state"
        raw["causal_relations"].append(duplicate)
    elif mutation == "statement_id_collision":
        relation["id"] = "inspect_panel"
    elif mutation == "hazard_id_collision":
        relation["id"] = "pressure_hazard"
    elif mutation == "ambiguity_id_collision":
        relation["id"] = "switch_position"
    else:
        statements[1]["id"] = statements[0]["id"]

    with pytest.raises(ValidationError):
        Document.model_validate(raw)


def test_measurement_round_trips_with_strict_nested_quantity() -> None:
    measurement = _complete_measurement()

    assert Measurement.model_validate_json(measurement.model_dump_json()) == measurement


def _complete_measurement() -> Measurement:
    return Measurement(
        property="pressure",
        quantity=Quantity(
            value=20,
            unit="MPa",
            tolerance=0.5,
            comparator="at_most",
        ),
    )


MEASUREMENT_MODEL_LOCATIONS: tuple[tuple[str, Path], ...] = (
    ("Measurement", ()),
    ("Measurement.quantity.Quantity", ("quantity",)),
)
MEASUREMENT_MODEL_BY_NAME: dict[str, type[BaseModel]] = {
    "Measurement": Measurement,
    "Quantity": Quantity,
}
MEASUREMENT_REQUIRED_FIELD_CASES = _model_field_cases(
    MEASUREMENT_MODEL_LOCATIONS,
    MEASUREMENT_MODEL_BY_NAME,
    required=True,
)
MEASUREMENT_DEFAULTED_FIELD_CASES = _model_field_cases(
    MEASUREMENT_MODEL_LOCATIONS,
    MEASUREMENT_MODEL_BY_NAME,
    required=False,
)


@pytest.mark.parametrize(
    ("boundary", "location", "field_name"),
    MEASUREMENT_REQUIRED_FIELD_CASES,
    ids=[
        f"{boundary}.{field_name}" for boundary, _, field_name in MEASUREMENT_REQUIRED_FIELD_CASES
    ],
)
def test_every_required_measurement_field_fails_closed_when_deleted(
    boundary: str,
    location: Path,
    field_name: str,
) -> None:
    raw = _complete_measurement().model_dump(mode="json")
    del _nested_object(raw, location)[field_name]

    with pytest.raises(ValidationError) as raised:
        Measurement.model_validate(raw)

    assert any(
        error["type"] == "missing" and error["loc"][-1] == field_name
        for error in raised.value.errors()
    ), boundary


@pytest.mark.parametrize(
    ("boundary", "location", "field_name"),
    MEASUREMENT_DEFAULTED_FIELD_CASES,
    ids=[
        f"{boundary}.{field_name}" for boundary, _, field_name in MEASUREMENT_DEFAULTED_FIELD_CASES
    ],
)
def test_every_defaulted_measurement_field_accepts_omission(
    boundary: str,
    location: Path,
    field_name: str,
) -> None:
    raw = _complete_measurement().model_dump(mode="json")
    del _nested_object(raw, location)[field_name]

    validated = Measurement.model_validate(raw)
    restored = _nested_object(validated.model_dump(mode="json"), location)

    assert field_name in restored, boundary


@pytest.mark.parametrize(
    ("operation", "location", "replacement", "error_type"),
    [
        ("extra", (), True, "extra_forbidden"),
        ("extra", ("quantity",), True, "extra_forbidden"),
        ("replace", ("quantity", "value"), float("nan"), "finite_number"),
        ("replace", ("quantity", "tolerance"), -1, "greater_than_equal"),
        ("replace", ("quantity", "unit"), "20", "value_error"),
    ],
    ids=[
        "measurement_unknown_field",
        "measurement_quantity_unknown_field",
        "measurement_nonfinite_value",
        "measurement_negative_tolerance",
        "measurement_numeric_unit",
    ],
)
def test_measurement_mutations_fail_closed(
    operation: str,
    location: Path,
    replacement: object,
    error_type: str,
) -> None:
    raw = _complete_measurement().model_dump(mode="json")
    if operation == "extra":
        _nested_object(raw, location)["unrecognized_property"] = replacement
    else:
        _apply_schema_mutation(
            raw,
            SchemaMutation(
                name="measurement_mutation",
                path=location,
                replacement=replacement,
                error_type=error_type,
            ),
        )

    with pytest.raises(ValidationError) as raised:
        Measurement.model_validate(raw)

    assert error_type in {error["type"] for error in raised.value.errors()}

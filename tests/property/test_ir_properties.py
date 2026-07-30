from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ste_compiler.ir.models import (
    ActionRef,
    CausalRelation,
    Document,
    EntityRef,
    Instruction,
    Section,
    SectionKind,
    SourceSpan,
    StateAssertion,
)
from ste_compiler.ir.serialization import (
    canonical_document_json,
    dumps_document,
    loads_document,
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


@settings(max_examples=50, deadline=None)
@given(document=documents())
def test_strict_ir_round_trips_through_json_and_yaml(document: Document) -> None:
    json_round_trip = loads_document(dumps_document(document, as_json=True), ".json")
    yaml_round_trip = loads_document(dumps_document(document), ".yaml")

    assert json_round_trip == document
    assert yaml_round_trip == document
    assert canonical_document_json(json_round_trip) == canonical_document_json(document)
    assert canonical_document_json(yaml_round_trip) == canonical_document_json(document)


UNKNOWN_FIELD_LOCATIONS = st.sampled_from(
    [
        (),
        ("sections", 0),
        ("sections", 0, "statements", 0),
        ("sections", 0, "statements", 0, "source_spans", 0),
        ("causal_relations", 0),
        ("causal_relations", 0, "source_spans", 0),
        ("metadata",),
    ]
)


def _nested_object(raw: dict[str, object], location: tuple[str | int, ...]) -> dict[str, object]:
    current: object = raw
    for component in location:
        if isinstance(component, int):
            assert isinstance(current, list)
            current = current[component]
        else:
            assert isinstance(current, dict)
            current = current[component]
    assert isinstance(current, dict)
    return current


@settings(max_examples=40, deadline=None)
@given(document=documents(), location=UNKNOWN_FIELD_LOCATIONS)
def test_ir_models_reject_unknown_fields_at_nested_boundaries(
    document: Document,
    location: tuple[str | int, ...],
) -> None:
    raw = deepcopy(document.model_dump(mode="json"))
    _nested_object(raw, location)["unrecognized_property"] = True

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Document.model_validate(raw)


CAUSAL_MUTATIONS = st.sampled_from(
    [
        "missing_endpoint",
        "self_loop",
        "duplicate_pair",
        "relation_id_collision",
        "duplicate_statement_id",
    ]
)


@settings(max_examples=40, deadline=None)
@given(document=documents(), mutation=CAUSAL_MUTATIONS)
def test_causal_graph_mutations_fail_closed(document: Document, mutation: str) -> None:
    raw = deepcopy(document.model_dump(mode="json"))
    relation = raw["causal_relations"][0]
    assert isinstance(relation, dict)
    statements = raw["sections"][0]["statements"]
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
    elif mutation == "relation_id_collision":
        relation["id"] = statements[0]["id"]
    else:
        statements[1]["id"] = statements[0]["id"]

    with pytest.raises(ValidationError):
        Document.model_validate(raw)

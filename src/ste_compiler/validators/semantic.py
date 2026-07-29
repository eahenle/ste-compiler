from ste_compiler.diagnostics import Diagnostic, Severity
from ste_compiler.ir.models import Document, Instruction
from ste_compiler.realizer.base import RealizationResult, SentenceMapping


class SemanticValidator:
    """Metadata-backed checks; untrusted neural output must create independently verified mappings."""

    def validate(self, document: Document, result: RealizationResult) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        for unmatched_mapping in result.mappings:
            if not unmatched_mapping.ir_node_ids:
                out.append(
                    Diagnostic(
                        code="UNSUPPORTED_SEMANTIC_CHANGE",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Sentence {unmatched_mapping.sentence} does not match the supplied IR."
                        ),
                    )
                )
        mappings: dict[str, list[SentenceMapping]] = {}
        for result_mapping in result.mappings:
            for node in result_mapping.ir_node_ids:
                mappings.setdefault(node, []).append(result_mapping)
        for section in document.sections:
            for item in section.statements:
                primary_mapping = next(
                    (
                        candidate
                        for candidate in mappings.get(item.id, [])
                        if candidate.ir_node_ids == (item.id,)
                    ),
                    None,
                )
                if primary_mapping is None:
                    if isinstance(item, Instruction) and not item.required:
                        continue
                    out.append(
                        Diagnostic(
                            code="REQUIRED_NODE_OMITTED",
                            severity=Severity.CRITICAL,
                            message=f"Required node {item.id} was not expressed.",
                            ir_node_id=item.id,
                        )
                    )
                    continue
                expected = item.model_dump(mode="json")
                actual = primary_mapping.features
                checks = [
                    ("negated", "NEGATION_NOT_PRESERVED"),
                    ("quantity_constraints", "QUANTITY_NOT_PRESERVED"),
                    ("conditions", "CONDITION_NOT_PRESERVED"),
                    ("temporal_relations", "TEMPORAL_RELATION_NOT_PRESERVED"),
                    ("hazards", "HAZARD_NOT_PRESERVED"),
                ]
                if isinstance(item, Instruction):
                    for field, code in checks:
                        if expected.get(field) != actual.get(field):
                            out.append(
                                Diagnostic(
                                    code=code,
                                    severity=Severity.CRITICAL,
                                    message=f"{field} changed for {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
                    for key in ("actor", "action", "object", "indirect_object"):
                        if expected.get(key) != actual.get(key):
                            out.append(
                                Diagnostic(
                                    code="UNSUPPORTED_SEMANTIC_CHANGE",
                                    severity=Severity.CRITICAL,
                                    message=f"{key} changed for {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
                    for hazard in item.hazards:
                        hazard_mapping = next(
                            (
                                candidate
                                for candidate in mappings.get(hazard.id, [])
                                if candidate.features == expected
                            ),
                            None,
                        )
                        if hazard_mapping is None:
                            out.append(
                                Diagnostic(
                                    code="HAZARD_NOT_PRESERVED",
                                    severity=Severity.CRITICAL,
                                    message=f"Output omitted hazard {hazard.id} from {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
                    sentence = primary_mapping.text.casefold()
                    if item.negated and "do not" not in sentence and "must not" not in sentence:
                        out.append(
                            Diagnostic(
                                code="NEGATION_NOT_PRESERVED",
                                severity=Severity.CRITICAL,
                                message=f"Output omitted negation from {item.id}.",
                                ir_node_id=item.id,
                            )
                        )
                    for constraint in item.quantity_constraints:
                        value = str(constraint.quantity.value).removesuffix(".0")
                        if (
                            value not in sentence
                            or constraint.quantity.unit.casefold() not in sentence
                        ):
                            out.append(
                                Diagnostic(
                                    code="QUANTITY_NOT_PRESERVED",
                                    severity=Severity.CRITICAL,
                                    message=f"Output omitted a quantity from {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
                    for temporal_relation in item.temporal_relations:
                        if (
                            temporal_relation.relation not in sentence
                            or temporal_relation.event.casefold() not in sentence
                        ):
                            out.append(
                                Diagnostic(
                                    code="TEMPORAL_RELATION_NOT_PRESERVED",
                                    severity=Severity.CRITICAL,
                                    message=f"Output omitted ordering from {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
        for causal_relation in document.causal_relations:
            expected_ids = (
                causal_relation.id,
                causal_relation.cause_node_id,
                causal_relation.effect_node_id,
            )
            relation_mapping = next(
                (
                    candidate
                    for candidate in mappings.get(causal_relation.id, [])
                    if candidate.ir_node_ids == expected_ids
                ),
                None,
            )
            cause_mapping = next(
                (
                    candidate
                    for candidate in mappings.get(causal_relation.cause_node_id, [])
                    if candidate.ir_node_ids == (causal_relation.cause_node_id,)
                ),
                None,
            )
            effect_mapping = next(
                (
                    candidate
                    for candidate in mappings.get(causal_relation.effect_node_id, [])
                    if candidate.ir_node_ids == (causal_relation.effect_node_id,)
                ),
                None,
            )
            if relation_mapping is None or cause_mapping is None or effect_mapping is None:
                out.append(
                    Diagnostic(
                        code="CAUSAL_RELATION_NOT_PRESERVED",
                        severity=Severity.CRITICAL,
                        message=f"Output omitted causal relation {causal_relation.id}.",
                        ir_node_id=causal_relation.id,
                    )
                )
                continue
            cause_clause = cause_mapping.text.removesuffix(".")
            effect_clause = effect_mapping.text.removesuffix(".")
            expected_text = f"Cause: {cause_clause}; effect: {effect_clause}."
            if (
                relation_mapping.features != causal_relation.model_dump(mode="json")
                or relation_mapping.text != expected_text
            ):
                out.append(
                    Diagnostic(
                        code="CAUSAL_RELATION_NOT_PRESERVED",
                        severity=Severity.CRITICAL,
                        message=f"Causal relation {causal_relation.id} changed.",
                        ir_node_id=causal_relation.id,
                    )
                )
        return out

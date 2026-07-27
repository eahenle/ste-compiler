from ste_compiler.diagnostics import Diagnostic, Severity
from ste_compiler.ir.models import Document, Instruction
from ste_compiler.realizer.base import RealizationResult


class SemanticValidator:
    """Metadata-backed checks; untrusted neural output must create independently verified mappings."""

    def validate(self, document: Document, result: RealizationResult) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        mappings = {node: m for m in result.mappings for node in m.ir_node_ids}
        for section in document.sections:
            for item in section.statements:
                mapping = mappings.get(item.id)
                if mapping is None:
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
                actual = mapping.features
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
                    sentence = mapping.text.casefold()
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
                    for relation in item.temporal_relations:
                        if (
                            relation.relation not in sentence
                            or relation.event.casefold() not in sentence
                        ):
                            out.append(
                                Diagnostic(
                                    code="TEMPORAL_RELATION_NOT_PRESERVED",
                                    severity=Severity.CRITICAL,
                                    message=f"Output omitted ordering from {item.id}.",
                                    ir_node_id=item.id,
                                )
                            )
        return out

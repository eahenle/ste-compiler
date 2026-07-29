from ste_compiler.ir.models import (
    Condition,
    Document,
    EntityRef,
    Instruction,
    Quantity,
    StateAssertion,
    TermReference,
)
from ste_compiler.realizer.base import (
    DEFAULT_CONSTRAINTS,
    RealizationConstraints,
    RealizationResult,
    SentenceMapping,
)
from ste_compiler.terminology import TerminologyRegistry, Vocabulary


class DeterministicRealizer:
    version = "0.2.0"

    def _ref(self, ref: EntityRef | TermReference, terms: TerminologyRegistry) -> str:
        return ref.name if isinstance(ref, EntityRef) else terms.get(ref.term_id).canonical_form

    @staticmethod
    def _number(value: float) -> str:
        return str(int(value)) if value.is_integer() else str(value)

    def _quantity(self, q: Quantity) -> str:
        operators = {
            "less_than": "less than",
            "more_than": "more than",
            "at_most": "not more than",
            "at_least": "not less than",
            "equal": "",
        }
        parts = [operators[q.comparator], self._number(q.value), q.unit]
        if q.tolerance is not None:
            parts += ["with a tolerance of", self._number(q.tolerance), q.unit]
        return " ".join(x for x in parts if x)

    def _condition(self, condition: Condition, terms: TerminologyRegistry) -> str:
        value = (
            self._quantity(condition.value)
            if isinstance(condition.value, Quantity)
            else condition.value
        )
        prefix = "except when" if condition.exception else "if"
        return f"{prefix} {self._ref(condition.subject, terms)} {condition.predicate} {value}"

    def _instruction(self, item: Instruction, terms: TerminologyRegistry) -> str:
        command = item.action.lemma
        if item.object:
            command += " the " + self._ref(item.object, terms)
        if item.indirect_object:
            command += " to the " + self._ref(item.indirect_object, terms)
        for constraint in item.quantity_constraints:
            command += f" to {self._quantity(constraint.quantity)}"
        if item.manner:
            command += " " + item.manner
        if item.purpose:
            command += " to " + item.purpose
        if item.actor:
            modal = " must not " if item.negated else " must "
            command = self._ref(item.actor, terms) + modal + command
        elif item.negated:
            command = "do not " + command
        for relation in item.temporal_relations:
            command += f" {relation.relation} {relation.event}"
        for condition in reversed(item.conditions):
            command = self._condition(condition, terms) + ", " + command
        return command[:1].upper() + command[1:] + "."

    def _state(self, item: StateAssertion, terms: TerminologyRegistry) -> str:
        value = self._quantity(item.value) if isinstance(item.value, Quantity) else item.value
        text = f"{self._ref(item.subject, terms)} {item.predicate} {value}."
        return text[:1].upper() + text[1:]

    @staticmethod
    def _causal_relation(cause_text: str, effect_text: str) -> str:
        cause_clause = cause_text.removesuffix(".")
        effect_clause = effect_text.removesuffix(".")
        return f"Cause: {cause_clause}; effect: {effect_clause}."

    def realize(
        self,
        document: Document,
        vocabulary: Vocabulary,
        terminology: TerminologyRegistry,
        constraints: RealizationConstraints = DEFAULT_CONSTRAINTS,
    ) -> RealizationResult:
        del vocabulary, constraints
        mappings: list[SentenceMapping] = []
        statement_text: dict[str, str] = {}
        for section in document.sections:
            for item in section.statements:
                if isinstance(item, Instruction):
                    for hazard in item.hazards:
                        threshold = (
                            f" when hydraulic pressure is {self._quantity(hazard.threshold)}"
                            if hazard.threshold
                            else ""
                        )
                        hazard_text = (
                            f"{hazard.severity}: {hazard.consequence} can occur{threshold}."
                        )
                        hazard_text = hazard_text[:1].upper() + hazard_text[1:]
                        mappings.append(
                            SentenceMapping(
                                len(mappings) + 1,
                                hazard_text,
                                (item.id, hazard.id),
                                item.model_dump(mode="json"),
                            )
                        )
                text = (
                    self._instruction(item, terminology)
                    if isinstance(item, Instruction)
                    else self._state(item, terminology)
                )
                features = item.model_dump(mode="json")
                mappings.append(SentenceMapping(len(mappings) + 1, text, (item.id,), features))
                statement_text[item.id] = text
        for relation in document.causal_relations:
            text = self._causal_relation(
                statement_text[relation.cause_node_id],
                statement_text[relation.effect_node_id],
            )
            mappings.append(
                SentenceMapping(
                    len(mappings) + 1,
                    text,
                    (relation.id, relation.cause_node_id, relation.effect_node_id),
                    relation.model_dump(mode="json"),
                )
            )
        metadata = {k: str(v) for k, v in document.metadata.model_dump().items()}
        return RealizationResult("\n".join(m.text for m in mappings), tuple(mappings), metadata)

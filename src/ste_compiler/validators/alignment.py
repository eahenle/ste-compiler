import re

from ste_compiler.realizer.base import RealizationResult, SentenceMapping

SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")


def _sentences(text: str) -> list[str]:
    return [match.group().strip() for match in SENTENCE.finditer(text) if match.group().strip()]


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def align_controlled_text(text: str, expected: RealizationResult) -> RealizationResult:
    """Map only exact deterministic sentences back to their verified IR nodes.

    Arbitrary text must not inherit the expected realization's node IDs or feature
    snapshot. Unmatched sentences remain explicitly unassociated so semantic
    validation can reject both changed and extra content.
    """

    mappings: list[SentenceMapping] = []
    for index, sentence in enumerate(_sentences(text)):
        expected_mapping = expected.mappings[index] if index < len(expected.mappings) else None
        if expected_mapping and _normalized(sentence) == _normalized(expected_mapping.text):
            mappings.append(
                SentenceMapping(
                    sentence=index + 1,
                    text=sentence,
                    ir_node_ids=expected_mapping.ir_node_ids,
                    features=expected_mapping.features,
                )
            )
        else:
            mappings.append(
                SentenceMapping(
                    sentence=index + 1,
                    text=sentence,
                    ir_node_ids=(),
                    features={},
                )
            )
    return RealizationResult(
        text=text,
        mappings=tuple(mappings),
        metadata={**expected.metadata, "alignment": "deterministic-surface-v1"},
    )

from ste_compiler.realizer.base import RealizationResult, SentenceMapping


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    position = 0
    while position < len(text):
        character = text[position]
        if (
            character == "."
            and position > 0
            and position + 1 < len(text)
            and text[position - 1].isdigit()
            and text[position + 1].isdigit()
        ):
            position += 1
            continue
        if character not in ".!?":
            position += 1
            continue
        end = position + 1
        while end < len(text) and text[end] in ".!?":
            end += 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        position = end
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def align_controlled_text(text: str, expected: RealizationResult) -> RealizationResult:
    """Map only exact deterministic sentences back to their verified IR nodes.

    Arbitrary text must not inherit the expected realization's node IDs or feature
    snapshot. Unmatched sentences remain explicitly unassociated so semantic
    validation can reject both changed and extra content.
    """

    if text == expected.text:
        return RealizationResult(
            text=text,
            mappings=expected.mappings,
            metadata={**expected.metadata, "alignment": "deterministic-surface-v1"},
        )

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

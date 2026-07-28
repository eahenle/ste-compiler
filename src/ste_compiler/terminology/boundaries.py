import re


def _casefold_view(text: str) -> tuple[str, dict[int, int]]:
    """Return casefolded text and its complete original-character boundaries."""

    folded_parts: list[str] = []
    original_boundaries = {0: 0}
    folded_length = 0
    for index, character in enumerate(text):
        folded_character = character.casefold()
        folded_parts.append(folded_character)
        folded_length += len(folded_character)
        original_boundaries[folded_length] = index + 1
    return "".join(folded_parts), original_boundaries


def whole_casefold_spans(text: str, form: str) -> tuple[tuple[int, int], ...]:
    """Find whole-form matches and map folded offsets to original text spans."""

    folded_text, original_boundaries = _casefold_view(text)
    folded_form = form.casefold()
    if not folded_form:
        return ()

    spans: list[tuple[int, int]] = []
    position = 0
    while (start := folded_text.find(folded_form, position)) >= 0:
        end = start + len(folded_form)
        original_start = original_boundaries.get(start)
        original_end = original_boundaries.get(end)
        if original_start is not None and original_end is not None:
            left_is_word = original_start > 0 and (
                text[original_start - 1].isalnum() or text[original_start - 1] == "_"
            )
            right_is_word = original_end < len(text) and (
                text[original_end].isalnum() or text[original_end] == "_"
            )
            if not left_is_word and not right_is_word:
                spans.append((original_start, original_end))
        position = start + 1
    return tuple(spans)


def mask_casefold_form(text: str, form: str) -> str:
    """Mask whole casefold-equivalent forms without changing original offsets."""

    output = list(text)
    for start, end in whole_casefold_spans(text, form):
        output[start:end] = " " * (end - start)
    return "".join(output)


def has_unit_boundaries(text: str, start: int, end: int) -> bool:
    """Allow numeric adjacency but not alphabetic or underscore attachment."""

    left_attached = start > 0 and (text[start - 1].isalpha() or text[start - 1] == "_")
    right_attached = end < len(text) and (text[end].isalpha() or text[end] == "_")
    return not left_attached and not right_attached


def mask_unit_surface(text: str, unit: str) -> str:
    """Mask exact configured unit occurrences that have valid unit boundaries."""

    output = list(text)
    for match in re.finditer(re.escape(unit), text):
        if has_unit_boundaries(text, match.start(), match.end()):
            output[match.start() : match.end()] = " " * len(unit)
    return "".join(output)

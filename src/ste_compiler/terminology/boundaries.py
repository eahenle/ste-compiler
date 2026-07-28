import re


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

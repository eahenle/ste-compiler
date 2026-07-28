"""Symbolic output avoids the false assumption that BPE tokens equal approved words."""

import re
from urllib.parse import quote, unquote

from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.terminology.boundaries import has_unit_boundaries, whole_casefold_spans

NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
NUMBER_SYMBOL = re.compile(rf"NUMBER_{NUMBER}")
NUMBER_TEXT = re.compile(NUMBER)
WORD_TEXT = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
PUNCTUATION = {
    "PERIOD": ".",
    "COMMA": ",",
    "COLON": ":",
    "QUESTION": "?",
    "EXCLAMATION": "!",
}
TEXT_PUNCTUATION = {value: key for key, value in PUNCTUATION.items()}
PUNCTUATION_TEXT = re.compile(r"[^\w\s]")
PUNCTUATION_SYMBOL = re.compile(r"PUNCT_U([0-9A-F]{4,6})")
WHITESPACE_SYMBOL = re.compile(r"WS_U([0-9A-F]{4,6})")
OPENING_PUNCTUATION = frozenset("([{“‘")
JOINING_PUNCTUATION = frozenset({"'", "-", "/", "\\", "–", "—", "’"})
EXACT_PLAN_SYMBOL = "PLAN_EXACT_WHITESPACE_V1"
TERM_SURFACE_SEPARATOR = "|"


def _unit_symbol(unit: str) -> str:
    return f"UNIT_{quote(unit, safe='')}"


def _word_symbol(word: str) -> str:
    return f"WORD_{quote(word, safe='')}"


def _term_symbol(term_id: str, surface: str) -> str:
    return f"TERM_{quote(term_id, safe='')}{TERM_SURFACE_SEPARATOR}{quote(surface, safe='')}"


def _punctuation_symbol(punctuation: str) -> str:
    return f"PUNCT_U{ord(punctuation):04X}"


def _punctuation_text(symbol: str) -> str | None:
    if symbol in PUNCTUATION:
        return PUNCTUATION[symbol]
    match = PUNCTUATION_SYMBOL.fullmatch(symbol)
    if match is None:
        return None
    try:
        punctuation = chr(int(match.group(1), 16))
    except (OverflowError, ValueError):
        return None
    return punctuation if PUNCTUATION_TEXT.fullmatch(punctuation) else None


def _whitespace_symbol(whitespace: str) -> str:
    return "SPACE" if whitespace == " " else f"WS_U{ord(whitespace):04X}"


def _whitespace_text(symbol: str) -> str | None:
    if symbol == "SPACE":
        return " "
    match = WHITESPACE_SYMBOL.fullmatch(symbol)
    if match is None:
        return None
    try:
        whitespace = chr(int(match.group(1), 16))
    except (OverflowError, ValueError):
        return None
    return whitespace if whitespace != "\n" and whitespace.isspace() else None


class SymbolicLexicalizer:
    """Translate between controlled text and an auditable symbolic plan."""

    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    def lexicalize(
        self,
        symbols: str,
        *,
        allowed_symbols: frozenset[str] | None = None,
        capitalize_sentences: bool = False,
    ) -> str:
        """Copy approved forms from symbols, optionally enforcing a plan-specific allowlist."""

        output = ""
        join_next = False
        explicit_whitespace = False
        at_sentence_start = capitalize_sentences
        ascii_double_quote_open = False
        curly_single_quote_open = False
        plan_symbols = symbols.split()
        exact_whitespace = bool(plan_symbols and plan_symbols[0] == EXACT_PLAN_SYMBOL)
        for index, symbol in enumerate(plan_symbols):
            if allowed_symbols is not None and symbol not in allowed_symbols:
                raise ValueError(f"symbol is not allowed for this document: {symbol}")
            if symbol == EXACT_PLAN_SYMBOL:
                if index != 0:
                    raise ValueError(f"invalid output symbol: {symbol}")
                continue
            if symbol == "NEWLINE":
                output = (
                    output if exact_whitespace or explicit_whitespace else output.rstrip()
                ) + "\n"
                join_next = False
                explicit_whitespace = False
                continue

            whitespace = _whitespace_text(symbol)
            if whitespace is not None:
                output += whitespace
                join_next = False
                explicit_whitespace = True
                continue

            punctuation = _punctuation_text(symbol)
            if punctuation is not None:
                if exact_whitespace:
                    output += punctuation
                    if punctuation in ".!?":
                        at_sentence_start = capitalize_sentences
                    elif at_sentence_start:
                        at_sentence_start = False
                    continue

                opening_quote = False
                closing_quote = False
                if punctuation == '"':
                    opening_quote = not ascii_double_quote_open
                    closing_quote = ascii_double_quote_open
                    ascii_double_quote_open = opening_quote
                elif punctuation == "‘":
                    opening_quote = True
                    curly_single_quote_open = True
                elif punctuation == "’" and curly_single_quote_open:
                    closing_quote = True
                    curly_single_quote_open = False
                elif punctuation == "“":
                    opening_quote = True
                elif punctuation == "”":
                    closing_quote = True

                if punctuation in OPENING_PUNCTUATION or opening_quote:
                    if output and not output[-1].isspace() and not join_next:
                        output += " "
                    output += punctuation
                    join_next = True
                else:
                    output = (output if explicit_whitespace else output.rstrip()) + punctuation
                    join_next = punctuation in JOINING_PUNCTUATION and not closing_quote
                explicit_whitespace = False
                if punctuation in ".!?":
                    at_sentence_start = capitalize_sentences
                elif at_sentence_start:
                    at_sentence_start = False
                continue

            value: str
            if symbol.startswith("WORD_"):
                word_surface = unquote(symbol[5:])
                value = (
                    word_surface
                    if exact_whitespace and self.vocabulary.contains(word_surface)
                    else self.vocabulary.canonical_word(word_surface) or ""
                )
                if not value or WORD_TEXT.fullmatch(value) is None:
                    raise ValueError(f"unauthorized word symbol: {symbol}")
            elif symbol.startswith("TERM_"):
                try:
                    payload = symbol[5:]
                    if exact_whitespace:
                        fields = payload.split(TERM_SURFACE_SEPARATOR)
                        if len(fields) != 2:
                            raise ValueError("exact term symbol requires identity and surface")
                        term_id, observed_surface = (unquote(field) for field in fields)
                        canonical_form = self.terminology.get(term_id).canonical_form
                        if observed_surface.casefold() != canonical_form.casefold():
                            raise ValueError(
                                "exact term surface does not match its canonical identity"
                            )
                        value = observed_surface
                    else:
                        value = self.terminology.get(unquote(payload)).canonical_form
                except (KeyError, ValueError) as error:
                    raise ValueError(f"unauthorized term symbol: {symbol}") from error
            elif symbol.startswith("UNIT_"):
                value = self.vocabulary.unit_forms.get(unquote(symbol[5:])) or ""
                if not value:
                    raise ValueError(f"unauthorized unit symbol: {symbol}")
            elif NUMBER_SYMBOL.fullmatch(symbol):
                value = symbol[7:]
            else:
                raise ValueError(f"invalid output symbol: {symbol}")

            if at_sentence_start and not exact_whitespace:
                value = value[:1].upper() + value[1:]
            at_sentence_start = False
            output += (
                "" if exact_whitespace or not output or output[-1].isspace() or join_next else " "
            ) + value
            join_next = False
            explicit_whitespace = False
        return output

    def symbolize(self, text: str) -> str:
        """Create a lossless symbolic plan from already-controlled text."""

        symbols = [EXACT_PLAN_SYMBOL]
        position = 0
        terms = sorted(
            self.terminology.approved_terms,
            key=lambda term: len(term.canonical_form),
            reverse=True,
        )
        term_matches = {
            start: (end, term)
            for term in reversed(terms)
            for start, end in whole_casefold_spans(text, term.canonical_form)
        }
        units = sorted(self.vocabulary.unit_forms.values(), key=len, reverse=True)
        while position < len(text):
            if text[position].isspace():
                if text[position] == "\n":
                    symbols.append("NEWLINE")
                else:
                    symbols.append(_whitespace_symbol(text[position]))
                position += 1
                continue

            matched_term = term_matches.get(position)
            if matched_term is not None:
                end, term = matched_term
                symbols.append(_term_symbol(term.id, text[position:end]))
                position = end
                continue

            matched_unit = None
            for unit_form in units:
                end = position + len(unit_form)
                if text[position:end] != unit_form:
                    continue
                if not has_unit_boundaries(text, position, end):
                    continue
                matched_unit = unit_form
                break
            if matched_unit is not None:
                symbols.append(_unit_symbol(matched_unit))
                position += len(matched_unit)
                continue

            number = NUMBER_TEXT.match(text, position)
            if number is not None:
                symbols.append(f"NUMBER_{number.group()}")
                position = number.end()
                continue

            character = text[position]
            if character in TEXT_PUNCTUATION:
                symbols.append(TEXT_PUNCTUATION[character])
                position += 1
                continue
            if PUNCTUATION_TEXT.fullmatch(character):
                symbols.append(_punctuation_symbol(character))
                position += 1
                continue

            word = WORD_TEXT.match(text, position)
            if word is None:
                raise ValueError(f"cannot symbolize text at offset {position}: {text[position]!r}")
            token = word.group()
            canonical_unit = self.vocabulary.unit_forms.get(token)
            if canonical_unit is not None:
                symbols.append(_unit_symbol(canonical_unit))
            elif self.vocabulary.contains(token):
                symbols.append(_word_symbol(token))
            else:
                raise ValueError(f"unauthorized word in controlled text: {token}")
            position = word.end()
        return " ".join(symbols)

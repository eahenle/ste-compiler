"""Symbolic output avoids the false assumption that BPE tokens equal approved words."""

import re

from ste_compiler.terminology import TerminologyRegistry, Vocabulary

NUMBER_SYMBOL = re.compile(r"NUMBER_-?\d+(?:\.\d+)?")
NUMBER_TEXT = re.compile(r"-?\d+(?:\.\d+)?")
WORD_TEXT = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
PUNCTUATION = {
    "PERIOD": ".",
    "COMMA": ",",
    "COLON": ":",
    "QUESTION": "?",
    "EXCLAMATION": "!",
}
TEXT_PUNCTUATION = {value: key for key, value in PUNCTUATION.items()}


class SymbolicLexicalizer:
    """Translate between controlled text and an auditable symbolic plan."""

    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    @staticmethod
    def _sentence_case(text: str) -> str:
        output: list[str] = []
        at_sentence_start = True
        for character in text:
            if at_sentence_start and character.isalpha():
                character = character.upper()
                at_sentence_start = False
            elif not character.isspace():
                at_sentence_start = False
            output.append(character)
            if character in ".!?":
                at_sentence_start = True
        return "".join(output)

    def lexicalize(
        self,
        symbols: str,
        *,
        allowed_symbols: frozenset[str] | None = None,
        capitalize_sentences: bool = False,
    ) -> str:
        """Copy approved forms from symbols, optionally enforcing a plan-specific allowlist."""

        output = ""
        for symbol in symbols.split():
            if allowed_symbols is not None and symbol not in allowed_symbols:
                raise ValueError(f"symbol is not allowed for this document: {symbol}")
            if symbol == "NEWLINE":
                output = output.rstrip() + "\n"
            elif symbol in PUNCTUATION:
                output = output.rstrip() + PUNCTUATION[symbol]
            elif symbol.startswith("WORD_"):
                word = symbol[5:]
                if not WORD_TEXT.fullmatch(word) or not self.vocabulary.contains(word):
                    raise ValueError(f"unauthorized word symbol: {symbol}")
                output += ("" if not output or output.endswith(("\n", " ")) else " ") + word
            elif symbol.startswith("TERM_"):
                try:
                    term = self.terminology.get(symbol[5:]).canonical_form
                except (KeyError, ValueError) as error:
                    raise ValueError(f"unauthorized term symbol: {symbol}") from error
                output += ("" if not output or output.endswith(("\n", " ")) else " ") + term
            elif symbol.startswith("UNIT_"):
                unit = self.vocabulary.unit_forms.get(symbol[5:].casefold())
                if unit is None:
                    raise ValueError(f"unauthorized unit symbol: {symbol}")
                output += ("" if not output or output.endswith(("\n", " ")) else " ") + unit
            elif NUMBER_SYMBOL.fullmatch(symbol):
                output += ("" if not output or output.endswith(("\n", " ")) else " ") + symbol[7:]
            else:
                raise ValueError(f"invalid output symbol: {symbol}")
        return self._sentence_case(output) if capitalize_sentences else output

    def symbolize(self, text: str) -> str:
        """Create a lossless symbolic plan from already-controlled text."""

        symbols: list[str] = []
        position = 0
        terms = sorted(
            self.terminology.approved_terms,
            key=lambda term: len(term.canonical_form),
            reverse=True,
        )
        units = sorted(self.vocabulary.unit_forms.values(), key=len, reverse=True)
        while position < len(text):
            if text[position].isspace():
                if text[position] == "\n":
                    symbols.append("NEWLINE")
                position += 1
                continue

            matched_term = None
            for term in terms:
                end = position + len(term.canonical_form)
                if text[position:end].casefold() != term.canonical_form.casefold():
                    continue
                if end < len(text) and (text[end].isalnum() or text[end] in "_-"):
                    continue
                matched_term = term
                break
            if matched_term is not None:
                symbols.append(f"TERM_{matched_term.id}")
                position += len(matched_term.canonical_form)
                continue

            matched_unit = None
            for unit_form in units:
                end = position + len(unit_form)
                if text[position:end].casefold() != unit_form.casefold():
                    continue
                if end < len(text) and unit_form[-1].isalnum() and text[end].isalnum():
                    continue
                matched_unit = unit_form
                break
            if matched_unit is not None:
                symbols.append(f"UNIT_{matched_unit}")
                position += len(matched_unit)
                continue

            character = text[position]
            if character in TEXT_PUNCTUATION:
                symbols.append(TEXT_PUNCTUATION[character])
                position += 1
                continue

            number = NUMBER_TEXT.match(text, position)
            if number is not None:
                symbols.append(f"NUMBER_{number.group()}")
                position = number.end()
                continue

            word = WORD_TEXT.match(text, position)
            if word is None:
                raise ValueError(f"cannot symbolize text at offset {position}: {text[position]!r}")
            token = word.group()
            canonical_unit = self.vocabulary.unit_forms.get(token.casefold())
            if canonical_unit is not None:
                symbols.append(f"UNIT_{canonical_unit}")
            elif self.vocabulary.contains(token):
                symbols.append(f"WORD_{token.casefold()}")
            else:
                raise ValueError(f"unauthorized word in controlled text: {token}")
            position = word.end()
        return " ".join(symbols)

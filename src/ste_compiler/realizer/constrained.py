"""Symbolic output avoids the false assumption that BPE tokens equal approved words."""

import re
from ste_compiler.terminology import TerminologyRegistry, Vocabulary


class SymbolicLexicalizer:
    """Validate symbolic word IDs, then deterministically copy approved forms."""

    def __init__(self, vocabulary: Vocabulary, terminology: TerminologyRegistry):
        self.vocabulary, self.terminology = vocabulary, terminology

    def lexicalize(self, symbols: str) -> str:
        output: list[str] = []
        for symbol in symbols.split():
            if symbol == "PERIOD":
                output.append(".")
            elif symbol == "COMMA":
                output.append(",")
            elif symbol.startswith("WORD_"):
                word = symbol[5:].replace("_", " ")
                if not all(self.vocabulary.contains(w) for w in word.split()):
                    raise ValueError(f"unauthorized word symbol: {symbol}")
                output.append(word)
            elif symbol.startswith("TERM_"):
                output.append(self.terminology.get(symbol[5:]).canonical_form)
            elif re.fullmatch(r"NUMBER_-?\d+(?:\.\d+)?", symbol):
                output.append(symbol[7:])
            else:
                raise ValueError(f"invalid output symbol: {symbol}")
        return " ".join(output).replace(" ,", ",").replace(" .", ".")

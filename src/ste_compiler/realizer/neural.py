from typing import Protocol


class SymbolGenerator(Protocol):
    """Future vendor-neutral SLM adapter: generate symbols, never production prose."""

    model_id: str

    def generate_symbols(self, serialized_ir: str, allowed_symbols: frozenset[str]) -> str: ...


class NeuralRealizerUnavailable(RuntimeError):
    pass

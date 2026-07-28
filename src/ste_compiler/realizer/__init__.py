from typing import TYPE_CHECKING

from .base import RealizationResult
from .deterministic import DeterministicRealizer

if TYPE_CHECKING:
    from .neural import NeuralRealizer, SymbolGenerator

__all__ = ["DeterministicRealizer", "NeuralRealizer", "RealizationResult", "SymbolGenerator"]


def __getattr__(name: str) -> object:
    if name == "NeuralRealizer":
        from .neural import NeuralRealizer

        return NeuralRealizer
    if name == "SymbolGenerator":
        from .neural import SymbolGenerator

        return SymbolGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

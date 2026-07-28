from typing import TYPE_CHECKING

from .base import RealizationResult
from .deterministic import DeterministicRealizer

if TYPE_CHECKING:
    from .encoder_decoder import (
        EncoderDecoderConfig,
        EncoderDecoderUnavailable,
        InvalidSymbolGeneration,
        TransformersEncoderDecoderSymbolGenerator,
    )
    from .neural import NeuralRealizer, SymbolGenerator

__all__ = [
    "DeterministicRealizer",
    "EncoderDecoderConfig",
    "EncoderDecoderUnavailable",
    "InvalidSymbolGeneration",
    "NeuralRealizer",
    "RealizationResult",
    "SymbolGenerator",
    "TransformersEncoderDecoderSymbolGenerator",
]


def __getattr__(name: str) -> object:
    if name in {
        "EncoderDecoderConfig",
        "EncoderDecoderUnavailable",
        "InvalidSymbolGeneration",
        "TransformersEncoderDecoderSymbolGenerator",
    }:
        from . import encoder_decoder

        return getattr(encoder_decoder, name)
    if name == "NeuralRealizer":
        from .neural import NeuralRealizer

        return NeuralRealizer
    if name == "SymbolGenerator":
        from .neural import SymbolGenerator

        return SymbolGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from typing import TYPE_CHECKING

from .base import RealizationResult
from .deterministic import DeterministicRealizer

if TYPE_CHECKING:
    from .decoder_lora import (
        DecoderOnlyLoRAConfig,
        DecoderOnlyLoRAError,
        DecoderOnlyLoRASymbolGenerator,
    )
    from .encoder_decoder import (
        EncoderDecoderConfig,
        EncoderDecoderError,
        EncoderDecoderUnavailable,
        InvalidSymbolGeneration,
        TransformersEncoderDecoderSymbolGenerator,
    )
    from .neural import NeuralRealizer, SymbolGenerator

__all__ = [
    "DecoderOnlyLoRAConfig",
    "DecoderOnlyLoRAError",
    "DecoderOnlyLoRASymbolGenerator",
    "DeterministicRealizer",
    "EncoderDecoderConfig",
    "EncoderDecoderError",
    "EncoderDecoderUnavailable",
    "InvalidSymbolGeneration",
    "NeuralRealizer",
    "RealizationResult",
    "SymbolGenerator",
    "TransformersEncoderDecoderSymbolGenerator",
]


def __getattr__(name: str) -> object:
    if name in {
        "DecoderOnlyLoRAConfig",
        "DecoderOnlyLoRAError",
        "DecoderOnlyLoRASymbolGenerator",
    }:
        from . import decoder_lora

        return getattr(decoder_lora, name)
    if name in {
        "EncoderDecoderConfig",
        "EncoderDecoderError",
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

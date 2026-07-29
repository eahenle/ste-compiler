from typing import TYPE_CHECKING

from .base import RealizationResult
from .deterministic import DeterministicRealizer

if TYPE_CHECKING:
    from .config import (
        DecoderOnlyLoRALocalBundleRealizerConfigV1,
        DecoderOnlyLoRARealizerConfigV1,
        DeterministicRealizerConfigV1,
        EncoderDecoderLocalBundleRealizerConfigV1,
        EncoderDecoderRealizerConfigV1,
        RealizerConfigV1,
    )
    from .decoder_lora import (
        DecoderOnlyLoRAConfig,
        DecoderOnlyLoRAError,
        DecoderOnlyLoRASymbolGenerator,
    )
    from .encoder_decoder import (
        EncoderDecoderConfig,
        EncoderDecoderError,
        EncoderDecoderLocalBundleConfig,
        EncoderDecoderRuntimeConfig,
        EncoderDecoderUnavailable,
        InvalidSymbolGeneration,
        TransformersEncoderDecoderSymbolGenerator,
    )
    from .factory import build_realizer
    from .local_decoder import (
        LocalDecoderOnlyLoRAError,
        LocalDecoderOnlyLoRARuntimeConfig,
        load_local_decoder_lora_generator,
    )
    from .neural import NeuralRealizer, SymbolGenerator

__all__ = [
    "REALIZER_CONFIG_ADAPTER",
    "DecoderOnlyLoRAConfig",
    "DecoderOnlyLoRAError",
    "DecoderOnlyLoRALocalBundleRealizerConfigV1",
    "DecoderOnlyLoRARealizerConfigV1",
    "DecoderOnlyLoRASymbolGenerator",
    "DeterministicRealizer",
    "DeterministicRealizerConfigV1",
    "EncoderDecoderConfig",
    "EncoderDecoderError",
    "EncoderDecoderLocalBundleConfig",
    "EncoderDecoderLocalBundleRealizerConfigV1",
    "EncoderDecoderRealizerConfigV1",
    "EncoderDecoderRuntimeConfig",
    "EncoderDecoderUnavailable",
    "InvalidSymbolGeneration",
    "LocalDecoderOnlyLoRAError",
    "LocalDecoderOnlyLoRARuntimeConfig",
    "NeuralRealizer",
    "RealizationResult",
    "RealizerConfigV1",
    "SymbolGenerator",
    "TransformersEncoderDecoderSymbolGenerator",
    "build_realizer",
    "canonical_realizer_config_json",
    "load_local_decoder_lora_generator",
    "load_realizer_config",
    "realizer_config_sha256",
]


def __getattr__(name: str) -> object:
    if name in {
        "DecoderOnlyLoRALocalBundleRealizerConfigV1",
        "DecoderOnlyLoRARealizerConfigV1",
        "DeterministicRealizerConfigV1",
        "EncoderDecoderLocalBundleRealizerConfigV1",
        "EncoderDecoderRealizerConfigV1",
        "REALIZER_CONFIG_ADAPTER",
        "RealizerConfigV1",
        "canonical_realizer_config_json",
        "load_realizer_config",
        "realizer_config_sha256",
    }:
        from . import config

        return getattr(config, name)
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
        "EncoderDecoderLocalBundleConfig",
        "EncoderDecoderRuntimeConfig",
        "EncoderDecoderUnavailable",
        "InvalidSymbolGeneration",
        "TransformersEncoderDecoderSymbolGenerator",
    }:
        from . import encoder_decoder

        return getattr(encoder_decoder, name)
    if name == "build_realizer":
        from .factory import build_realizer

        return build_realizer
    if name in {
        "LocalDecoderOnlyLoRAError",
        "LocalDecoderOnlyLoRARuntimeConfig",
        "load_local_decoder_lora_generator",
    }:
        from . import local_decoder

        return getattr(local_decoder, name)
    if name == "NeuralRealizer":
        from .neural import NeuralRealizer

        return NeuralRealizer
    if name == "SymbolGenerator":
        from .neural import SymbolGenerator

        return SymbolGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

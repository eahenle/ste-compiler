"""Construct offline-only realizers from strict versioned configuration."""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property
from typing import assert_never

from ste_compiler.ir.models import Document
from ste_compiler.realizer.base import (
    DEFAULT_CONSTRAINTS,
    RealizationConstraints,
    RealizationResult,
    Realizer,
)
from ste_compiler.realizer.config import (
    DecoderOnlyLoRARealizerConfigV1,
    DeterministicRealizerConfigV1,
    EncoderDecoderRealizerConfigV1,
    RealizerConfigV1,
    realizer_config_sha256,
)
from ste_compiler.realizer.decoder_lora import (
    DecoderOnlyLoRAConfig,
    DecoderOnlyLoRASymbolGenerator,
)
from ste_compiler.realizer.decoder_protocol import DECODER_PROMPT_PROFILE
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.realizer.encoder_decoder import (
    EncoderDecoderConfig,
    TransformersEncoderDecoderSymbolGenerator,
)
from ste_compiler.realizer.neural import NeuralRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

OFFLINE_ARTIFACT_MODE = "offline-cache-only"


class _ConfiguredRealizer:
    """Attach configuration identity without changing an underlying realizer."""

    def __init__(self, delegate: Realizer, config_sha256: str):
        self._delegate = delegate
        self._config_sha256 = config_sha256

    def realize(
        self,
        document: Document,
        vocabulary: Vocabulary,
        terminology: TerminologyRegistry,
        constraints: RealizationConstraints = DEFAULT_CONSTRAINTS,
    ) -> RealizationResult:
        result = self._delegate.realize(document, vocabulary, terminology, constraints)
        return replace(
            result,
            metadata={
                **result.metadata,
                "realizer_config_sha256": self._config_sha256,
                "artifact_mode": OFFLINE_ARTIFACT_MODE,
            },
        )


class _LazyDecoderOnlyLoRASymbolGenerator:
    """Keep factory construction independent of optional neural dependencies."""

    def __init__(self, config: DecoderOnlyLoRAConfig):
        self.config = config
        self.model_id = (
            f"{config.base_model_id}@{config.base_model_revision}"
            f"+peft:{config.adapter_id}@{config.adapter_revision}"
        )
        self.base_model_revision = config.base_model_revision
        self.adapter_revision = config.adapter_revision

    @cached_property
    def _generator(self) -> DecoderOnlyLoRASymbolGenerator:
        return DecoderOnlyLoRASymbolGenerator(self.config)

    def generate_symbols(
        self,
        serialized_ir: str,
        allowed_symbols: frozenset[str],
    ) -> str:
        return self._generator.generate_symbols(serialized_ir, allowed_symbols)


def build_realizer(config: RealizerConfigV1) -> Realizer:
    """Build one realizer whose model artifacts can only come from the local cache."""

    delegate: Realizer
    if isinstance(config, DeterministicRealizerConfigV1):
        delegate = DeterministicRealizer()
    elif isinstance(config, EncoderDecoderRealizerConfigV1):
        generator = TransformersEncoderDecoderSymbolGenerator(
            EncoderDecoderConfig(
                model_id=config.checkpoint.repo_id,
                revision=config.checkpoint.revision,
                max_source_tokens=config.max_source_tokens,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                local_files_only=True,
            )
        )
        delegate = NeuralRealizer(generator)
    elif isinstance(config, DecoderOnlyLoRARealizerConfigV1):
        if config.prompt_profile != DECODER_PROMPT_PROFILE:
            raise ValueError(
                "decoder-only realizer prompt profile does not match the runtime protocol"
            )
        decoder_generator = _LazyDecoderOnlyLoRASymbolGenerator(
            DecoderOnlyLoRAConfig(
                base_model_id=config.base_model.repo_id,
                base_model_revision=config.base_model.revision,
                adapter_id=config.adapter.repo_id,
                adapter_revision=config.adapter.revision,
                max_new_tokens=config.max_new_tokens,
                max_symbols=config.max_symbols,
                local_files_only=True,
            )
        )
        delegate = NeuralRealizer(decoder_generator)
    else:
        assert_never(config)
    return _ConfiguredRealizer(delegate, realizer_config_sha256(config))

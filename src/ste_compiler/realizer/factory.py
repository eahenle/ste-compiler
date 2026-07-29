"""Construct offline-only realizers from strict versioned configuration."""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import assert_never

from ste_compiler.ir.models import Document
from ste_compiler.realizer.base import (
    DEFAULT_CONSTRAINTS,
    RealizationConstraints,
    RealizationResult,
    Realizer,
)
from ste_compiler.realizer.config import (
    DecoderOnlyLoRALocalBundleRealizerConfigV1,
    DecoderOnlyLoRARealizerConfigV1,
    DeterministicRealizerConfigV1,
    EncoderDecoderLocalBundleRealizerConfigV1,
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
    EncoderDecoderLocalBundleConfig,
    TransformersEncoderDecoderSymbolGenerator,
)
from ste_compiler.realizer.local_decoder import (
    LocalDecoderOnlyLoRARuntimeConfig,
    load_local_decoder_lora_generator,
)
from ste_compiler.realizer.neural import NeuralRealizer
from ste_compiler.terminology import TerminologyRegistry, Vocabulary

OFFLINE_ARTIFACT_MODE = "offline-cache-only"
LOCAL_BUNDLE_ARTIFACT_MODE = "content-addressed-local-bundle"


class _ConfiguredRealizer:
    """Attach configuration identity without changing an underlying realizer."""

    def __init__(self, delegate: Realizer, config_sha256: str, artifact_mode: str):
        self._delegate = delegate
        self._config_sha256 = config_sha256
        self._artifact_mode = artifact_mode

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
                "artifact_mode": self._artifact_mode,
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


class _LazyLocalDecoderOnlyLoRASymbolGenerator:
    """Defer optional decoder dependencies while retaining content provenance."""

    def __init__(self, config: LocalDecoderOnlyLoRARuntimeConfig):
        self.config = config
        self.model_id = (
            f"{config.base_model.repo_id}@{config.base_model.revision}"
            f"+peft-bundle:sha256:{config.artifact_manifest_sha256}"
            f"+model-snapshot:sha256:{config.model_snapshot_manifest_sha256}"
        )
        self.base_model_revision = config.base_model.revision
        self.adapter_revision = None
        self.artifact_manifest_sha256 = config.artifact_manifest_sha256
        self.model_snapshot_manifest_sha256 = config.model_snapshot_manifest_sha256
        self.artifact_intended_use = config.intended_use

    @cached_property
    def _generator(self) -> DecoderOnlyLoRASymbolGenerator:
        return load_local_decoder_lora_generator(self.config)

    @property
    def run_manifest_sha256(self) -> str:
        digest = self._generator.run_manifest_sha256
        if digest is None:  # pragma: no cover - local loader always supplies this identity.
            raise ValueError("local decoder generator did not retain its run-manifest identity")
        return digest

    def generate_symbols(
        self,
        serialized_ir: str,
        allowed_symbols: frozenset[str],
    ) -> str:
        return self._generator.generate_symbols(serialized_ir, allowed_symbols)


def build_realizer(
    config: RealizerConfigV1,
    *,
    artifact_bundle: Path | None = None,
    model_snapshot: Path | None = None,
) -> Realizer:
    """Build one realizer whose model artifacts can only come from the local cache."""

    delegate: Realizer
    artifact_mode = OFFLINE_ARTIFACT_MODE
    if isinstance(config, DeterministicRealizerConfigV1):
        if artifact_bundle is not None or model_snapshot is not None:
            raise ValueError("deterministic realization does not accept local artifact locators")
        delegate = DeterministicRealizer()
    elif isinstance(config, EncoderDecoderRealizerConfigV1):
        if artifact_bundle is not None or model_snapshot is not None:
            raise ValueError("Hub encoder-decoder realization does not accept local locators")
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
    elif isinstance(config, EncoderDecoderLocalBundleRealizerConfigV1):
        if artifact_bundle is None:
            raise ValueError("encoder-decoder-local-bundle realization requires --artifact-bundle")
        if model_snapshot is not None:
            raise ValueError(
                "encoder-decoder-local-bundle realization does not accept --model-snapshot"
            )
        local_generator = TransformersEncoderDecoderSymbolGenerator(
            EncoderDecoderLocalBundleConfig(
                artifact_bundle=artifact_bundle,
                artifact_manifest_sha256=config.artifact_manifest_sha256,
                intended_use=config.intended_use,
                max_source_tokens=config.max_source_tokens,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
            )
        )
        delegate = NeuralRealizer(local_generator)
        artifact_mode = LOCAL_BUNDLE_ARTIFACT_MODE
    elif isinstance(config, DecoderOnlyLoRARealizerConfigV1):
        if artifact_bundle is not None or model_snapshot is not None:
            raise ValueError("Hub decoder-only LoRA realization does not accept local locators")
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
    elif isinstance(config, DecoderOnlyLoRALocalBundleRealizerConfigV1):
        if artifact_bundle is None or model_snapshot is None:
            raise ValueError(
                "decoder-only-lora-local-bundle realization requires "
                "--artifact-bundle and --model-snapshot"
            )
        local_decoder_generator = _LazyLocalDecoderOnlyLoRASymbolGenerator(
            LocalDecoderOnlyLoRARuntimeConfig(
                artifact_bundle=artifact_bundle,
                artifact_manifest_sha256=config.artifact_manifest_sha256,
                model_snapshot=model_snapshot,
                model_snapshot_manifest_sha256=config.model_snapshot_manifest_sha256,
                base_model=config.base_model,
                tokenizer=config.tokenizer,
                intended_use=config.intended_use,
                prompt_profile=config.prompt_profile,
                max_new_tokens=config.max_new_tokens,
                max_symbols=config.max_symbols,
            )
        )
        delegate = NeuralRealizer(local_decoder_generator)
        artifact_mode = LOCAL_BUNDLE_ARTIFACT_MODE
    else:
        assert_never(config)
    return _ConfiguredRealizer(
        delegate,
        realizer_config_sha256(config),
        artifact_mode,
    )

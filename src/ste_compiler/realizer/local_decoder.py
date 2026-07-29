"""Content-addressed local loading for verified decoder-only LoRA artifacts."""

from __future__ import annotations

import importlib
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from ste_compiler.realizer.decoder_lora import (
    DecoderOnlyLoRAError,
    DecoderOnlyLoRASymbolGenerator,
)
from ste_compiler.realizer.decoder_protocol import DECODER_PROMPT_PROFILE
from ste_compiler.realizer.neural import NeuralRealizerUnavailable
from ste_compiler.training.config import ArtifactIdentityV1
from ste_compiler.training.decoder_lora import (
    DecoderLoRARunManifestV1,
    DecoderLoRATrainingError,
    VerifiedModelSnapshot,
    open_verified_decoder_lora_artifact_bundle,
    read_verified_model_snapshot_for_identities,
)

_SHA256: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)


class LocalDecoderOnlyLoRAError(DecoderOnlyLoRAError):
    """A content-addressed local decoder runtime could not be loaded safely."""


@dataclass(frozen=True)
class LocalDecoderOnlyLoRARuntimeConfig:
    """Trust identities, local locators, protocol, and inference bounds for one runtime."""

    artifact_bundle: Path
    artifact_manifest_sha256: str
    model_snapshot: Path
    model_snapshot_manifest_sha256: str
    base_model: ArtifactIdentityV1
    tokenizer: ArtifactIdentityV1
    intended_use: Literal["mechanics-smoke"]
    prompt_profile: Literal["decoder-only-symbol-plan-v1"]
    max_new_tokens: int = 512
    max_symbols: int = 128

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_manifest_sha256",
            "model_snapshot_manifest_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")
        if self.intended_use != "mechanics-smoke":
            raise ValueError("local decoder artifact intended_use must be mechanics-smoke")
        if self.prompt_profile != DECODER_PROMPT_PROFILE:
            raise ValueError("local decoder prompt profile does not match the runtime protocol")
        if self.tokenizer != self.base_model:
            raise ValueError("local decoder V1 requires tokenizer identity to equal base_model")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")


@dataclass(frozen=True)
class _RuntimeModules:
    transformers: Any
    peft: Any


def _runtime_modules() -> _RuntimeModules:
    try:
        return _RuntimeModules(
            transformers=importlib.import_module("transformers"),
            peft=importlib.import_module("peft"),
        )
    except ImportError as error:
        raise NeuralRealizerUnavailable(
            "local decoder-only LoRA inference requires the 'neural' extra: "
            "install ste-compiler[neural]"
        ) from error


def _validate_cross_links(
    config: LocalDecoderOnlyLoRARuntimeConfig,
    run_manifest: DecoderLoRARunManifestV1,
    snapshot: VerifiedModelSnapshot,
) -> None:
    training_config = run_manifest.training_config
    if run_manifest.prompt_profile != config.prompt_profile:
        raise LocalDecoderOnlyLoRAError(
            "decoder run prompt profile does not match the authorized runtime protocol"
        )
    if (
        training_config.base_model != config.base_model
        or training_config.tokenizer != config.tokenizer
    ):
        raise LocalDecoderOnlyLoRAError(
            "decoder run model or tokenizer identity does not match the authorized runtime"
        )
    if run_manifest.model_snapshot_manifest_sha256 != config.model_snapshot_manifest_sha256:
        raise LocalDecoderOnlyLoRAError(
            "decoder run does not bind the authorized model snapshot manifest"
        )
    if run_manifest.model_snapshot_artifacts != snapshot.manifest.artifacts:
        raise LocalDecoderOnlyLoRAError(
            "decoder run model snapshot inventory does not match the verified snapshot"
        )


@contextmanager
def _materialized_model_snapshot(snapshot: VerifiedModelSnapshot) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="ste-local-decoder-base-") as temporary:
        root = Path(temporary)
        for name, data in snapshot.files:
            (root / name).write_bytes(data)
        yield root


def _load_components(
    modules: _RuntimeModules,
    snapshot_path: Path,
    adapter_path: Path,
) -> tuple[object, object]:
    adapter_config = modules.peft.PeftConfig.from_pretrained(
        str(adapter_path),
        local_files_only=True,
    )
    tokenizer = modules.transformers.AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model = modules.transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = modules.peft.PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        config=adapter_config,
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    return tokenizer, model


def load_local_decoder_lora_generator(
    config: LocalDecoderOnlyLoRARuntimeConfig,
) -> DecoderOnlyLoRASymbolGenerator:
    """Load a local adapter and base snapshot only after exact cross-bound verification."""

    try:
        with open_verified_decoder_lora_artifact_bundle(
            config.artifact_bundle,
            config.artifact_manifest_sha256,
        ) as bundle:
            snapshot = read_verified_model_snapshot_for_identities(
                config.model_snapshot,
                config.base_model,
                config.tokenizer,
                config.model_snapshot_manifest_sha256,
            )
            _validate_cross_links(config, bundle.run_manifest, snapshot)
            modules = _runtime_modules()
            with _materialized_model_snapshot(snapshot) as snapshot_path:
                tokenizer, model = _load_components(
                    modules,
                    snapshot_path,
                    bundle.path / "adapter",
                )
            return DecoderOnlyLoRASymbolGenerator.from_loaded_components(
                tokenizer=tokenizer,
                model=model,
                base_model_id=config.base_model.repo_id,
                base_model_revision=config.base_model.revision,
                artifact_manifest_sha256=bundle.artifact_manifest_sha256,
                run_manifest_sha256=bundle.run_manifest_sha256,
                model_snapshot_manifest_sha256=snapshot.manifest_sha256,
                max_new_tokens=config.max_new_tokens,
                max_symbols=config.max_symbols,
            )
    except (LocalDecoderOnlyLoRAError, NeuralRealizerUnavailable):
        raise
    except DecoderLoRATrainingError as error:
        raise LocalDecoderOnlyLoRAError(
            "content-addressed decoder-only LoRA artifact verification failed"
        ) from error
    except Exception as error:
        raise LocalDecoderOnlyLoRAError(
            "content-addressed decoder-only LoRA artifacts could not be loaded safely"
        ) from error

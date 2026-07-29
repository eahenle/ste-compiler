"""Constrained encoder-decoder adapter for symbolic sentence plans."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from operator import index
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Protocol, cast

from ste_compiler.training.encoder_decoder import (
    EncoderDecoderTrainingError,
    open_verified_encoder_decoder_artifact_bundle,
)

_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_HUB_COMPONENT = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?", re.ASCII)
_SAFE_SNAPSHOT_PATTERNS = [
    "*.codes",
    "*.json",
    "*.merges",
    "*.model",
    "*.safetensors",
    "*.spm",
    "*.tiktoken",
    "*.tokenizer",
    "*.txt",
    "*.vocab",
]


class _Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> Mapping[str, Any]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class _Model(Protocol):
    config: Any

    def generate(self, **kwargs: Any) -> Any: ...


class EncoderDecoderError(RuntimeError):
    """Raised when the encoder-decoder trust boundary cannot be established."""


class EncoderDecoderUnavailable(EncoderDecoderError):
    """Raised when the optional Transformers runtime is not installed."""


class InvalidSymbolGeneration(ValueError):
    """Raised when a model output violates the symbolic decoding contract."""


@dataclass(frozen=True)
class EncoderDecoderConfig:
    """Immutable identity and bounded inference settings for a seq2seq model."""

    model_id: str
    revision: str
    max_source_tokens: int = 1024
    max_new_tokens: int = 256
    num_beams: int = 1
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be blank")
        _reject_local_model_id(self.model_id)
        if not self.revision.strip():
            raise ValueError("revision must not be blank")
        if _COMMIT_REVISION.fullmatch(self.revision) is None:
            raise ValueError("revision must be a full lowercase 40-character commit digest")
        if self.max_source_tokens < 1:
            raise ValueError("max_source_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.num_beams < 1:
            raise ValueError("num_beams must be positive")


@dataclass(frozen=True)
class EncoderDecoderLocalBundleConfig:
    """Content identity, untrusted locator, and bounded settings for a local bundle."""

    artifact_bundle: Path
    artifact_manifest_sha256: str
    intended_use: Literal["mechanics-smoke"] = "mechanics-smoke"
    max_source_tokens: int = 1024
    max_new_tokens: int = 256
    num_beams: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_bundle, Path):
            raise TypeError("artifact_bundle must be a pathlib.Path locator")
        if _SHA256.fullmatch(self.artifact_manifest_sha256) is None:
            raise ValueError("artifact_manifest_sha256 must be 64 lowercase hexadecimal characters")
        if self.intended_use != "mechanics-smoke":
            raise ValueError("local encoder-decoder bundles support mechanics-smoke use only")
        if self.max_source_tokens < 1:
            raise ValueError("max_source_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.num_beams < 1:
            raise ValueError("num_beams must be positive")


EncoderDecoderRuntimeConfig = EncoderDecoderConfig | EncoderDecoderLocalBundleConfig
ComponentLoader = Callable[[EncoderDecoderRuntimeConfig], tuple[_Tokenizer, _Model]]


def _reject_local_model_id(model_id: str) -> None:
    """Require a Hub repository ID whose resolution cannot currently select a local path."""

    path = Path(model_id).expanduser()
    windows_path = PureWindowsPath(model_id)
    components = model_id.split("/")
    valid_hub_id = (
        1 <= len(model_id) <= 96
        and len(components) in {1, 2}
        and all(_HUB_COMPONENT.fullmatch(component) for component in components)
        and "--" not in model_id
        and ".." not in model_id
        and not model_id.endswith(".git")
    )
    explicitly_local = (
        model_id != model_id.strip()
        or model_id in {".", "..", "~"}
        or model_id.startswith(("file:", "./", "../", "~/", ".\\", "..\\", "~\\"))
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or not valid_hub_id
    )
    try:
        resolves_locally = path.exists() or path.is_symlink()
    except OSError:
        resolves_locally = explicitly_local
    if explicitly_local or resolves_locally:
        raise EncoderDecoderError(
            "local filesystem model paths are not supported; model_id must be a Hugging Face "
            "Hub repository ID such as 'org/model'"
        )


class _SymbolTokenConstraint:
    """Accept token prefixes in ``SYMBOL (SPACE SYMBOL)* EOS``."""

    def __init__(
        self,
        tokenizer: _Tokenizer,
        allowed_symbols: frozenset[str],
        *,
        decoder_start_token_id: int,
        eos_token_id: int,
    ):
        if not allowed_symbols:
            raise ValueError("allowed_symbols must not be empty")
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self._first = self._encode_variants(tokenizer, allowed_symbols, prefix="")
        self._following = self._encode_variants(tokenizer, allowed_symbols, prefix=" ")

        forbidden_ids = {eos_token_id}
        unknown_token_id = getattr(tokenizer, "unk_token_id", None)
        if unknown_token_id is not None:
            forbidden_ids.add(int(unknown_token_id))
        if any(forbidden_ids.intersection(encoded) for encoded in (*self._first, *self._following)):
            raise ValueError("an allowed symbol cannot be represented exactly by this tokenizer")

    @staticmethod
    def _encode_variants(
        tokenizer: _Tokenizer,
        allowed_symbols: frozenset[str],
        *,
        prefix: str,
    ) -> tuple[tuple[int, ...], ...]:
        variants: set[tuple[int, ...]] = set()
        for symbol in sorted(allowed_symbols):
            expected = f"{prefix}{symbol}"
            encoded = tuple(
                int(token_id) for token_id in tokenizer.encode(expected, add_special_tokens=False)
            )
            if not encoded:
                raise ValueError(f"tokenizer produced no tokens for allowed symbol: {symbol}")
            decoded = tokenizer.decode(
                encoded,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if decoded != expected:
                raise ValueError(
                    f"tokenizer cannot losslessly encode allowed symbolic form: {expected!r}"
                )
            variants.add(encoded)
        return tuple(sorted(variants))

    def __call__(self, batch_id: int, input_ids: Sequence[int]) -> list[int]:
        del batch_id
        generated = tuple(int(token_id) for token_id in input_ids)
        if generated and generated[0] == self.decoder_start_token_id:
            generated = generated[1:]
        allowed = self._allowed_next(generated)
        if not allowed:
            raise InvalidSymbolGeneration("decoder reached a prefix outside the symbolic grammar")
        return sorted(allowed)

    def validate_complete(self, token_ids: Sequence[int]) -> None:
        """Reject any pre-EOS token path that the prefix grammar could not generate."""

        generated: tuple[int, ...] = ()
        for token_id in token_ids:
            if token_id not in self._allowed_next(generated):
                raise InvalidSymbolGeneration(
                    "model output contains a token path outside the symbolic grammar before EOS"
                )
            generated = (*generated, token_id)
        if self.eos_token_id not in self._allowed_next(generated):
            raise InvalidSymbolGeneration(
                "model output does not end at a complete symbolic boundary before EOS"
            )

    def _allowed_next(self, generated: tuple[int, ...]) -> set[int]:
        allowed: set[int] = set()
        visited: set[tuple[int, bool]] = set()

        def visit(position: int, first_symbol: bool) -> None:
            state = (position, first_symbol)
            if state in visited:
                return
            visited.add(state)
            variants = self._first if first_symbol else self._following
            remaining = generated[position:]
            if not remaining:
                allowed.update(variant[0] for variant in variants)
                if not first_symbol:
                    allowed.add(self.eos_token_id)
                return

            for variant in variants:
                available = min(len(remaining), len(variant))
                if remaining[:available] != variant[:available]:
                    continue
                if len(remaining) < len(variant):
                    allowed.add(variant[len(remaining)])
                else:
                    visit(position + len(variant), False)

        visit(0, True)
        return allowed


class TransformersEncoderDecoderSymbolGenerator:
    """Generate allowlisted symbolic plans with a pinned seq2seq Transformers model."""

    def __init__(
        self,
        config: EncoderDecoderRuntimeConfig,
        *,
        component_loader: ComponentLoader | None = None,
    ):
        self.config = config
        if isinstance(config, EncoderDecoderConfig):
            _reject_local_model_id(config.model_id)
            self.model_id = f"{config.model_id}@{config.revision}"
            self.model_revision: str | None = config.revision
            default_loader = self._load_transformers_components
        else:
            if component_loader is not None:
                raise ValueError(
                    "local encoder-decoder bundles do not accept injected component loaders"
                )
            self.model_id = f"ste-artifact-bundle:sha256:{config.artifact_manifest_sha256}"
            self.model_revision = None
            self.artifact_manifest_sha256 = config.artifact_manifest_sha256
            self.artifact_intended_use = config.intended_use
            self.run_manifest_sha256: str | None = None
            default_loader = self._load_local_bundle_components
        self._component_loader = component_loader or default_loader
        self._components: tuple[_Tokenizer, _Model] | None = None

    @staticmethod
    def _load_transformers_components(
        config: EncoderDecoderRuntimeConfig,
    ) -> tuple[_Tokenizer, _Model]:
        if not isinstance(config, EncoderDecoderConfig):
            raise EncoderDecoderError("Hub component loader received a local bundle configuration")
        _reject_local_model_id(config.model_id)
        try:
            transformers = import_module("transformers")
            huggingface_hub = import_module("huggingface_hub")
        except ImportError as error:
            raise EncoderDecoderUnavailable(
                "install ste-compiler[neural] to use the encoder-decoder adapter"
            ) from error

        snapshot = TransformersEncoderDecoderSymbolGenerator._resolve_safe_model_snapshot(
            config,
            huggingface_hub,
        )
        common = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(snapshot),
                **common,
            )
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                str(snapshot),
                **common,
                use_safetensors=True,
            )
        except Exception as error:
            raise EncoderDecoderError(
                "configured encoder-decoder artifacts could not be loaded safely"
            ) from error
        return tokenizer, model

    def _load_local_bundle_components(
        self,
        config: EncoderDecoderRuntimeConfig,
    ) -> tuple[_Tokenizer, _Model]:
        if not isinstance(config, EncoderDecoderLocalBundleConfig):
            raise EncoderDecoderError("local bundle component loader received a Hub configuration")
        try:
            with open_verified_encoder_decoder_artifact_bundle(
                config.artifact_bundle,
                config.artifact_manifest_sha256,
            ) as verified:
                if (
                    config.max_source_tokens
                    > verified.run_manifest.training_config.max_source_tokens
                    or config.max_new_tokens
                    > verified.run_manifest.training_config.max_target_tokens
                ):
                    raise EncoderDecoderError(
                        "runtime token limits exceed the verified training bundle limits"
                    )
                try:
                    transformers = import_module("transformers")
                except ImportError as error:
                    raise EncoderDecoderUnavailable(
                        "install ste-compiler[neural] to use the encoder-decoder adapter"
                    ) from error
                common = {
                    "local_files_only": True,
                    "trust_remote_code": False,
                }
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        str(verified.checkpoint_path),
                        **common,
                    )
                    loaded = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                        str(verified.checkpoint_path),
                        **common,
                        use_safetensors=True,
                        output_loading_info=True,
                        low_cpu_mem_usage=False,
                        device_map=None,
                    )
                except Exception as error:
                    raise EncoderDecoderError(
                        "verified local encoder-decoder bundle could not be loaded safely"
                    ) from error
                if (
                    not isinstance(loaded, tuple)
                    or len(loaded) != 2
                    or not isinstance(loaded[1], dict)
                ):
                    raise EncoderDecoderError(
                        "local encoder-decoder loader did not return loading diagnostics"
                    )
                model, loading_info = loaded
                failures = {
                    key: loading_info.get(key)
                    for key in (
                        "missing_keys",
                        "unexpected_keys",
                        "mismatched_keys",
                        "error_msgs",
                    )
                    if loading_info.get(key)
                }
                if failures:
                    raise EncoderDecoderError(
                        "local encoder-decoder checkpoint does not exactly match "
                        f"the model architecture: {failures}"
                    )
                self.run_manifest_sha256 = verified.run_manifest_sha256
                return tokenizer, model
        except EncoderDecoderTrainingError as error:
            raise EncoderDecoderError(
                f"local encoder-decoder artifact verification failed: {error}"
            ) from error

    @staticmethod
    def _resolve_safe_model_snapshot(
        config: EncoderDecoderConfig,
        huggingface_hub: object,
    ) -> Path:
        download = cast(Callable[..., object], cast(Any, huggingface_hub).snapshot_download)
        try:
            snapshot = Path(
                cast(
                    str,
                    download(
                        repo_id=config.model_id,
                        revision=config.revision,
                        local_files_only=config.local_files_only,
                        allow_patterns=_SAFE_SNAPSHOT_PATTERNS,
                    ),
                )
            ).resolve(strict=True)
        except Exception as error:
            raise EncoderDecoderError(
                "model revision could not be resolved to a safe local snapshot"
            ) from error
        if not snapshot.is_dir():
            raise EncoderDecoderError("model snapshot resolver did not return a directory")
        if snapshot.name != config.revision:
            raise EncoderDecoderError(
                "model snapshot resolver did not return the configured commit revision"
            )
        if not (snapshot / "config.json").is_file():
            raise EncoderDecoderError("model snapshot is missing config.json")
        if not any(snapshot.glob("*.safetensors")):
            raise EncoderDecoderError("model snapshot is missing safetensors weights")
        return snapshot

    def _get_components(self) -> tuple[_Tokenizer, _Model]:
        if self._components is None:
            if isinstance(self.config, EncoderDecoderConfig):
                _reject_local_model_id(self.config.model_id)
            self._components = self._component_loader(self.config)
        return self._components

    def prepare(self) -> None:
        """Load and verify model components before processing any records."""

        self._get_components()

    def generate_symbols(self, serialized_ir: str, allowed_symbols: frozenset[str]) -> str:
        try:
            return self._generate_symbols(serialized_ir, allowed_symbols)
        except (EncoderDecoderError, InvalidSymbolGeneration):
            raise
        except Exception as error:
            raise EncoderDecoderError("encoder-decoder inference failed safely") from error

    def _generate_symbols(self, serialized_ir: str, allowed_symbols: frozenset[str]) -> str:
        tokenizer, model = self._get_components()
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError("model must define decoder_start_token_id")

        constraint = _SymbolTokenConstraint(
            tokenizer,
            allowed_symbols,
            decoder_start_token_id=int(decoder_start_token_id),
            eos_token_id=int(eos_token_id),
        )
        encoded_source = tokenizer(
            serialized_ir,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_tokens,
        )
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "eos_token_id": int(eos_token_id),
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": self.config.max_new_tokens,
            "num_beams": self.config.num_beams,
            "num_beam_groups": 1,
            "num_return_sequences": 1,
            "decoder_start_token_id": int(decoder_start_token_id),
            "penalty_alpha": None,
            "dola_layers": None,
            "constraints": None,
            "force_words_ids": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "bad_words_ids": None,
            "sequence_bias": None,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
            "diversity_penalty": 0.0,
            "length_penalty": 1.0,
            "early_stopping": False,
            "exponential_decay_length_penalty": None,
            "renormalize_logits": False,
            "remove_invalid_values": False,
            "assistant_model": None,
            "prompt_lookup_num_tokens": None,
            "min_length": 0,
            "min_new_tokens": 0,
            "max_time": None,
            "stop_strings": None,
            "watermarking_config": None,
            "guidance_scale": None,
            "prefix_allowed_tokens_fn": constraint,
            "return_dict_in_generate": False,
        }
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "forced_decoder_ids"):
            generation_kwargs["forced_decoder_ids"] = None
        if generation_config is not None and hasattr(generation_config, "token_healing"):
            generation_kwargs["token_healing"] = False
        generated = model.generate(**encoded_source, **generation_kwargs)
        token_ids = self._first_sequence(generated)
        if token_ids and token_ids[0] == int(decoder_start_token_id):
            token_ids = token_ids[1:]
        try:
            termination = token_ids.index(int(eos_token_id))
        except ValueError as error:
            raise InvalidSymbolGeneration("model output did not terminate with EOS") from error
        constraint.validate_complete(token_ids[:termination])
        trailing_token_ids = token_ids[termination + 1 :]
        pad_token_id = tokenizer.pad_token_id
        if trailing_token_ids and (
            pad_token_id is None
            or any(token_id != int(pad_token_id) for token_id in trailing_token_ids)
        ):
            raise InvalidSymbolGeneration("model output contains non-padding tokens after EOS")
        decoded = tokenizer.decode(
            token_ids[:termination],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        symbols = decoded.split()
        if not symbols:
            raise InvalidSymbolGeneration("model generated an empty symbolic plan")
        unauthorized = sorted(set(symbols).difference(allowed_symbols))
        if unauthorized:
            raise InvalidSymbolGeneration(
                f"model generated symbols outside the document allowlist: {unauthorized}"
            )
        return " ".join(symbols)

    @staticmethod
    def _first_sequence(generated: Any) -> list[int]:
        sequences = getattr(generated, "sequences", generated)
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        try:
            if isinstance(sequences, (str, bytes)):
                raise TypeError
            sequence_count = len(sequences)
        except TypeError as error:
            raise InvalidSymbolGeneration(
                "model returned an invalid generated sequence container"
            ) from error
        if sequence_count == 0:
            raise InvalidSymbolGeneration("model returned no generated sequence")
        if sequence_count != 1:
            raise InvalidSymbolGeneration(
                "model returned multiple generated sequences; exactly one is required"
            )
        try:
            sequence = sequences[0]
        except (IndexError, KeyError, TypeError) as error:
            raise InvalidSymbolGeneration(
                "model returned an invalid generated sequence container"
            ) from error
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()
        try:
            if isinstance(sequence, (str, bytes)):
                raise TypeError
            token_ids = []
            for token_id in sequence:
                if isinstance(token_id, bool):
                    raise TypeError
                token_ids.append(index(token_id))
            return token_ids
        except TypeError as error:
            raise InvalidSymbolGeneration(
                "model returned a generated sequence that is not one-dimensional integer token IDs"
            ) from error

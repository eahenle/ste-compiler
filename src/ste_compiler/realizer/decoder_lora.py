"""Concrete decoder-only Transformers + PEFT adapter for symbolic generation."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, cast

from ste_compiler.realizer.neural import NeuralRealizerUnavailable

PROMPT_PROFILE = "decoder-only-symbol-plan-v1"
ADAPTER_CONFIG = "adapter_config.json"
SAFE_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}", re.ASCII)


def _is_local_artifact_id(identifier: str) -> bool:
    path = Path(identifier).expanduser()
    return path.is_absolute() or path.exists() or identifier.startswith(("./", "../", "~"))


class DecoderOnlyLoRAError(RuntimeError):
    """The decoder-only adapter could not produce a constrained symbolic plan."""


class _Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class DecoderOnlyLoRAConfig:
    """Immutable model and inference provenance for one adapter deployment."""

    base_model_id: str
    base_model_revision: str
    adapter_id: str
    adapter_revision: str
    max_new_tokens: int = 512
    max_symbols: int = 128
    local_files_only: bool = False

    def __post_init__(self) -> None:
        required = {
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
        }
        for field_name, value in required.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("base_model_id", "adapter_id"):
            identifier = required[field_name]
            if _is_local_artifact_id(identifier):
                raise ValueError(f"{field_name} must be a Hub repository ID, not a local path")
        for field_name in ("base_model_revision", "adapter_revision"):
            revision = required[field_name]
            if _COMMIT_REVISION.fullmatch(revision) is None:
                raise ValueError(
                    f"{field_name} must be a full lowercase 40-character commit digest"
                )
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")


@dataclass(frozen=True)
class _Candidate:
    tokens: tuple[int, ...]
    offset: int
    completed_symbols: int


class _SymbolTokenGrammar:
    """Recognize repetitions of explicitly encoded symbol strings plus EOS."""

    def __init__(
        self,
        tokenizer: _Tokenizer,
        symbols: frozenset[str],
        *,
        max_symbols: int,
    ):
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise DecoderOnlyLoRAError("the tokenizer must define eos_token_id")
        if not symbols:
            raise DecoderOnlyLoRAError("allowed_symbols must not be empty")

        self.eos_token_id = eos_token_id
        self.max_symbols = max_symbols
        self._first = self._encode_forms(tokenizer, symbols, prefix="")
        self._continued = self._encode_forms(tokenizer, symbols, prefix=" ")

    @staticmethod
    def _encode_forms(
        tokenizer: _Tokenizer,
        symbols: frozenset[str],
        *,
        prefix: str,
    ) -> tuple[tuple[int, ...], ...]:
        encoded: set[tuple[int, ...]] = set()
        for symbol in sorted(symbols):
            text = prefix + symbol
            token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            if not token_ids:
                raise DecoderOnlyLoRAError(f"tokenizer produced no tokens for {text!r}")
            round_trip = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if round_trip != text:
                raise DecoderOnlyLoRAError(
                    f"tokenizer does not losslessly encode symbolic form {text!r}"
                )
            encoded.add(token_ids)
        return tuple(sorted(encoded))

    def allowed_next(self, generated: Sequence[int]) -> list[int]:
        candidates = {
            _Candidate(tokens=tokens, offset=0, completed_symbols=0) for tokens in self._first
        }
        at_boundary: set[int] = set()

        for token in generated:
            next_candidates: set[_Candidate] = set()
            next_boundaries: set[int] = set()
            for candidate in candidates:
                if candidate.tokens[candidate.offset] != token:
                    continue
                next_offset = candidate.offset + 1
                if next_offset == len(candidate.tokens):
                    next_boundaries.add(candidate.completed_symbols + 1)
                else:
                    next_candidates.add(
                        _Candidate(
                            tokens=candidate.tokens,
                            offset=next_offset,
                            completed_symbols=candidate.completed_symbols,
                        )
                    )

            for completed_symbols in at_boundary:
                if completed_symbols >= self.max_symbols:
                    continue
                for tokens in self._continued:
                    if tokens[0] != token:
                        continue
                    if len(tokens) == 1:
                        next_boundaries.add(completed_symbols + 1)
                    else:
                        next_candidates.add(
                            _Candidate(
                                tokens=tokens,
                                offset=1,
                                completed_symbols=completed_symbols,
                            )
                        )

            candidates = next_candidates
            at_boundary = next_boundaries
            if not candidates and not at_boundary:
                return []

        allowed = {
            candidate.tokens[candidate.offset]
            for candidate in candidates
            if candidate.offset < len(candidate.tokens)
        }
        if at_boundary:
            allowed.add(self.eos_token_id)
        for completed_symbols in at_boundary:
            if completed_symbols < self.max_symbols:
                allowed.update(tokens[0] for tokens in self._continued)
        return sorted(allowed)

    def validate_complete(self, token_ids: Sequence[int]) -> None:
        """Reject any pre-EOS token path that the prefix grammar could not generate."""

        generated: list[int] = []
        for token_id in token_ids:
            if token_id not in self.allowed_next(generated):
                raise DecoderOnlyLoRAError(
                    "model output contains a token path outside the symbolic grammar before EOS"
                )
            generated.append(token_id)
        if self.eos_token_id not in self.allowed_next(generated):
            raise DecoderOnlyLoRAError(
                "model output does not end at a complete symbolic boundary before EOS"
            )


def _integer_sequence(value: object, *, batched: bool = False) -> list[int]:
    if hasattr(value, "tolist"):
        value = cast(Any, value).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DecoderOnlyLoRAError("model returned an unsupported token sequence")
    if batched:
        if not value:
            raise DecoderOnlyLoRAError("model returned no token sequences")
        if len(value) != 1:
            raise DecoderOnlyLoRAError("model returned multiple token sequences")
        value = cast(Sequence[object], value)[0]
        if hasattr(value, "tolist"):
            value = cast(Any, value).tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise DecoderOnlyLoRAError(
                "model output must contain exactly one batch dimension of token IDs"
            )
    elif value and isinstance(value[0], Sequence):
        raise DecoderOnlyLoRAError("token sequence must be one-dimensional")

    token_ids: list[int] = []
    try:
        for item in cast(Sequence[object], value):
            if isinstance(item, bool):
                raise TypeError
            token_ids.append(index(cast(SupportsIndex, item)))
    except TypeError as error:
        raise DecoderOnlyLoRAError("token sequence must contain only integer token IDs") from error
    return token_ids


class DecoderOnlyLoRASymbolGenerator:
    """Generate an allowlisted symbol plan with a pinned PEFT LoRA adapter."""

    def __init__(
        self,
        config: DecoderOnlyLoRAConfig,
        *,
        tokenizer: object | None = None,
        model: object | None = None,
    ):
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be supplied together")
        self.config = config
        if tokenizer is None:
            tokenizer, model = self._load_runtime(config)
        self._tokenizer = cast(_Tokenizer, tokenizer)
        self._model = model

    @property
    def model_id(self) -> str:
        """Revision-bearing provenance included in NeuralRealizer metadata."""

        return (
            f"{self.config.base_model_id}@{self.config.base_model_revision}"
            f"+peft:{self.config.adapter_id}@{self.config.adapter_revision}"
        )

    @property
    def base_model_revision(self) -> str:
        """Exact base-model commit digest included in realization metadata."""

        return self.config.base_model_revision

    @property
    def adapter_revision(self) -> str:
        """Exact PEFT adapter commit digest included in realization metadata."""

        return self.config.adapter_revision

    @staticmethod
    def _load_runtime(config: DecoderOnlyLoRAConfig) -> tuple[object, object]:
        for field_name in ("base_model_id", "adapter_id"):
            if _is_local_artifact_id(getattr(config, field_name)):
                raise DecoderOnlyLoRAError(
                    f"{field_name} resolved to a local path after configuration"
                )
        try:
            transformers = importlib.import_module("transformers")
            peft = importlib.import_module("peft")
            huggingface_hub = importlib.import_module("huggingface_hub")
        except ImportError as error:
            raise NeuralRealizerUnavailable(
                "decoder-only LoRA inference requires the 'neural' extra: "
                "install ste-compiler[neural]"
            ) from error

        adapter_snapshot = DecoderOnlyLoRASymbolGenerator._resolve_safe_adapter_snapshot(
            config,
            huggingface_hub,
        )
        adapter_config = peft.PeftConfig.from_pretrained(
            str(adapter_snapshot),
            local_files_only=True,
        )
        DecoderOnlyLoRASymbolGenerator._validate_adapter_config(config, adapter_config)
        common = {
            "revision": config.base_model_revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_id, **common)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            **common,
            use_safetensors=True,
        )
        model = peft.PeftModel.from_pretrained(
            base_model,
            str(adapter_snapshot),
            config=adapter_config,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        return tokenizer, model

    @staticmethod
    def _resolve_safe_adapter_snapshot(
        config: DecoderOnlyLoRAConfig,
        huggingface_hub: object,
    ) -> Path:
        download = cast(Callable[..., object], cast(Any, huggingface_hub).snapshot_download)
        try:
            snapshot = Path(
                cast(
                    str,
                    download(
                        repo_id=config.adapter_id,
                        revision=config.adapter_revision,
                        local_files_only=config.local_files_only,
                        allow_patterns=[ADAPTER_CONFIG, SAFE_ADAPTER_WEIGHTS],
                    ),
                )
            )
        except Exception as error:
            raise DecoderOnlyLoRAError(
                "adapter revision could not be resolved to a safe local snapshot"
            ) from error
        if not snapshot.is_dir():
            raise DecoderOnlyLoRAError("adapter snapshot resolver did not return a directory")
        missing = [
            filename
            for filename in (ADAPTER_CONFIG, SAFE_ADAPTER_WEIGHTS)
            if not (snapshot / filename).is_file()
        ]
        if missing:
            raise DecoderOnlyLoRAError(
                "adapter snapshot is missing required safe artifacts: " + ", ".join(missing)
            )
        return snapshot

    @staticmethod
    def _validate_adapter_config(
        config: DecoderOnlyLoRAConfig,
        adapter_config: object,
    ) -> None:
        def value(name: str) -> str | None:
            item = getattr(adapter_config, name, None)
            if item is None:
                return None
            return str(getattr(item, "value", item))

        if value("peft_type") != "LORA":
            raise DecoderOnlyLoRAError("adapter configuration must use PEFT type LORA")
        if value("task_type") != "CAUSAL_LM":
            raise DecoderOnlyLoRAError("adapter configuration must target the CAUSAL_LM task")
        declared_base_model = value("base_model_name_or_path")
        if declared_base_model != config.base_model_id:
            raise DecoderOnlyLoRAError(
                "adapter configuration does not target the configured base model"
            )
        declared_base_revision = value("revision")
        if declared_base_revision != config.base_model_revision:
            raise DecoderOnlyLoRAError(
                "adapter configuration must declare the configured base model revision"
            )

    @staticmethod
    def _prompt(serialized_ir: str) -> str:
        envelope = json.dumps(
            {"profile": PROMPT_PROFILE, "serialized_ir": serialized_ir},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"{envelope}\nSYMBOLS\n"

    def generate_symbols(
        self,
        serialized_ir: str,
        allowed_symbols: frozenset[str],
    ) -> str:
        grammar = _SymbolTokenGrammar(
            self._tokenizer,
            allowed_symbols,
            max_symbols=self.config.max_symbols,
        )
        encoded = dict(
            self._tokenizer(
                self._prompt(serialized_ir),
                return_tensors="pt",
                add_special_tokens=True,
            )
        )
        prompt_ids = _integer_sequence(encoded["input_ids"], batched=True)
        device = getattr(self._model, "device", None)
        if device is not None:
            encoded = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in encoded.items()
            }

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: object) -> list[int]:
            if batch_id != 0:
                raise DecoderOnlyLoRAError("batched constrained generation is not supported")
            all_ids = _integer_sequence(input_ids)
            if all_ids[: len(prompt_ids)] != prompt_ids:
                raise DecoderOnlyLoRAError("generation did not preserve the prompt prefix")
            return grammar.allowed_next(all_ids[len(prompt_ids) :])

        generate = cast(Callable[..., object], cast(Any, self._model).generate)
        pad_token_id = self._tokenizer.pad_token_id
        output = generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            num_beam_groups=1,
            num_return_sequences=1,
            return_dict_in_generate=False,
            penalty_alpha=None,
            dola_layers=None,
            constraints=None,
            force_words_ids=None,
            assistant_model=None,
            prompt_lookup_num_tokens=None,
            min_length=0,
            min_new_tokens=0,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            bad_words_ids=None,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            max_time=None,
            stop_strings=None,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=grammar.eos_token_id if pad_token_id is None else pad_token_id,
            eos_token_id=grammar.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        output_ids = _integer_sequence(output, batched=True)
        if output_ids[: len(prompt_ids)] != prompt_ids:
            raise DecoderOnlyLoRAError("generation did not preserve the prompt prefix")
        continuation = output_ids[len(prompt_ids) :]
        if not continuation or continuation[-1] != grammar.eos_token_id:
            raise DecoderOnlyLoRAError("generation did not terminate with EOS")
        if grammar.eos_token_id in continuation[:-1]:
            raise DecoderOnlyLoRAError("generation returned tokens after EOS")
        grammar.validate_complete(continuation[:-1])

        symbols = self._tokenizer.decode(
            continuation[:-1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_symbols = symbols.split()
        if not output_symbols:
            raise DecoderOnlyLoRAError("generation returned an empty symbolic plan")
        if len(output_symbols) > self.config.max_symbols:
            raise DecoderOnlyLoRAError("generation exceeded max_symbols")
        unauthorized = sorted(set(output_symbols) - allowed_symbols)
        if unauthorized:
            raise DecoderOnlyLoRAError(
                f"generation escaped the symbol allowlist: {', '.join(unauthorized)}"
            )
        if symbols != " ".join(output_symbols):
            raise DecoderOnlyLoRAError("generation did not use canonical symbol spacing")
        return symbols

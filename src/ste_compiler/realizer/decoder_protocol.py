"""Shared versioned text and token protocol for decoder-only symbol plans."""

from __future__ import annotations

import json
from collections.abc import Sequence
from operator import index
from typing import Protocol, SupportsIndex, cast

DECODER_PROMPT_PROFILE = "decoder-only-symbol-plan-v1"


class DecoderProtocolError(ValueError):
    """A prompt, symbol plan, or tokenizer violates the decoder protocol."""


class DecoderProtocolTokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


def canonical_decoder_prompt(serialized_ir: str) -> str:
    """Return the exact versioned prompt shared by training and inference."""

    envelope = json.dumps(
        {
            "profile": DECODER_PROMPT_PROFILE,
            "serialized_ir": serialized_ir,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{envelope}\nSYMBOLS\n"


def lossless_symbol_tokens(
    tokenizer: DecoderProtocolTokenizer,
    text: str,
) -> tuple[int, ...]:
    """Encode one first or space-prefixed symbol form without hidden special tokens."""

    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        token_ids = tuple(index(cast(SupportsIndex, token_id)) for token_id in raw_ids)
    except TypeError as error:
        raise DecoderProtocolError("tokenizer returned a non-integer symbol token ID") from error
    if not token_ids:
        raise DecoderProtocolError(f"tokenizer produced no tokens for {text!r}")
    if any(isinstance(token_id, bool) for token_id in raw_ids):
        raise DecoderProtocolError("tokenizer returned a Boolean symbol token ID")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise DecoderProtocolError("the tokenizer must define eos_token_id")
    if eos_token_id in token_ids:
        raise DecoderProtocolError(
            f"symbolic form encodes the EOS token before termination: {text!r}"
        )
    round_trip = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if round_trip != text:
        raise DecoderProtocolError(f"tokenizer does not losslessly encode symbolic form {text!r}")
    return token_ids


def segmented_symbol_plan_tokens(
    tokenizer: DecoderProtocolTokenizer,
    symbols: str,
) -> tuple[int, ...]:
    """Encode a canonical plan exactly as inference stitches grammar candidates."""

    symbol_parts = symbols.split()
    if not symbol_parts or " ".join(symbol_parts) != symbols:
        raise DecoderProtocolError("symbol plan must use canonical single-space separation")
    encoded: list[int] = []
    for position, symbol in enumerate(symbol_parts):
        encoded.extend(
            lossless_symbol_tokens(
                tokenizer,
                symbol if position == 0 else f" {symbol}",
            )
        )
    decoded = tokenizer.decode(
        encoded,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != symbols:
        raise DecoderProtocolError("segmented symbol plan does not round-trip canonically")
    return tuple(encoded)


def validate_lora_adapter_identity(
    base_model_id: str,
    base_model_revision: str,
    adapter_config: object,
) -> None:
    """Require a causal-LM LoRA adapter for one exact base-model commit."""

    def value(name: str) -> str | None:
        item = getattr(adapter_config, name, None)
        if item is None:
            return None
        return str(getattr(item, "value", item))

    if value("peft_type") != "LORA":
        raise DecoderProtocolError("adapter configuration must use PEFT type LORA")
    if value("task_type") != "CAUSAL_LM":
        raise DecoderProtocolError("adapter configuration must target the CAUSAL_LM task")
    if value("base_model_name_or_path") != base_model_id:
        raise DecoderProtocolError(
            "adapter configuration does not target the configured base model"
        )
    if value("revision") != base_model_revision:
        raise DecoderProtocolError(
            "adapter configuration must declare the configured base model revision"
        )

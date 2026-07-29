from __future__ import annotations

import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import ste_compiler.training.decoder_lora as decoder_training
from ste_compiler.realizer import DecoderOnlyLoRASymbolGenerator
from ste_compiler.realizer.decoder_protocol import (
    DECODER_PROMPT_PROFILE,
    DecoderProtocolError,
    canonical_decoder_prompt,
    segmented_symbol_plan_tokens,
)
from ste_compiler.training import (
    DecoderLoRATrainingError,
    ReleasedTrainingRecordV1,
    build_decoder_training_example,
)
from ste_compiler.training.decoder_lora import _collate


class CharacterTrainingTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1

    def encode(self, text, *, add_special_tokens):
        encoded = [ord(character) + 2 for character in text]
        return [self.bos_token_id, *encoded] if add_special_tokens else encoded

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        special = {self.eos_token_id, self.pad_token_id, self.bos_token_id}
        return "".join(chr(token_id - 2) for token_id in token_ids if token_id not in special)


def _record(symbols: str = "PLAN_EXACT_WHITESPACE_V1 WORD_Open") -> ReleasedTrainingRecordV1:
    return ReleasedTrainingRecordV1(
        schema_version="demonstration-corpus-record-v1",
        record_id="record",
        split="train",
        source_id="record.txt",
        source_sha256="a" * 64,
        source_license_id="MIT",
        serialized_ir='{"id":"caf\u00e9"}',
        text="Open.",
        symbols=symbols,
        allowed_symbols=tuple(sorted(set(symbols.split()))),
        metadata=(("realizer", "deterministic"),),
        features=("statement.instruction",),
    )


def test_decoder_prompt_is_one_shared_canonical_versioned_contract():
    serialized_ir = '{"title":"Caf\u00e9","quote":"a\\\\b"}'

    prompt = canonical_decoder_prompt(serialized_ir)

    assert prompt == (
        '{"profile":"decoder-only-symbol-plan-v1",'
        '"serialized_ir":"{\\"title\\":\\"Caf\u00e9\\",\\"quote\\":\\"a\\\\\\\\b\\"}"}'
        "\nSYMBOLS\n"
    )
    assert DECODER_PROMPT_PROFILE in prompt
    assert DecoderOnlyLoRASymbolGenerator._prompt(serialized_ir) == prompt


def test_segmented_plan_encoding_matches_inference_boundaries_not_whole_text_encoding():
    class MergeSensitiveTokenizer:
        eos_token_id = 0

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {
                "FIRST SECOND": [99],
                "FIRST": [10],
                " SECOND": [20],
            }[text]

        def decode(self, token_ids, **kwargs):
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return {
                (99,): "FIRST SECOND",
                (10,): "FIRST",
                (20,): " SECOND",
                (10, 20): "FIRST SECOND",
            }[tuple(token_ids)]

    tokenizer = MergeSensitiveTokenizer()

    assert tokenizer.encode("FIRST SECOND", add_special_tokens=False) == [99]
    assert segmented_symbol_plan_tokens(tokenizer, "FIRST SECOND") == (10, 20)


def test_decoder_training_example_masks_prompt_and_supervises_exactly_one_eos():
    tokenizer = CharacterTrainingTokenizer()
    record = _record()

    example = build_decoder_training_example(
        record,
        tokenizer,
        max_sequence_tokens=4096,
    )

    assert example.record_id == record.record_id
    assert example.input_ids[-1] == tokenizer.eos_token_id
    assert example.labels[: example.prompt_length] == (-100,) * example.prompt_length
    assert example.labels[example.prompt_length :] == example.input_ids[example.prompt_length :]
    assert example.labels[example.prompt_length :].count(tokenizer.eos_token_id) == 1
    assert example.attention_mask == (1,) * len(example.input_ids)
    with pytest.raises(FrozenInstanceError):
        example.prompt_length = 0


def test_decoder_training_example_fails_instead_of_truncating_terminal_eos():
    tokenizer = CharacterTrainingTokenizer()
    record = _record()
    complete = build_decoder_training_example(
        record,
        tokenizer,
        max_sequence_tokens=4096,
    )

    with pytest.raises(DecoderLoRATrainingError, match="exceeding max_sequence_tokens"):
        build_decoder_training_example(
            record,
            tokenizer,
            max_sequence_tokens=len(complete.input_ids) - 1,
        )


def test_decoder_batch_padding_never_supervises_pad_when_pad_equals_eos():
    class FakeTensor:
        def __init__(self, values):
            self.values = values

    class FakeTorch:
        long = object()

        @staticmethod
        def tensor(values, *, dtype):
            assert dtype is FakeTorch.long
            return FakeTensor(values)

    tokenizer = CharacterTrainingTokenizer()
    shorter = build_decoder_training_example(
        _record("WORD_Open"),
        tokenizer,
        max_sequence_tokens=4096,
    )
    longer = build_decoder_training_example(
        _record("PLAN_EXACT_WHITESPACE_V1 WORD_Open"),
        tokenizer,
        max_sequence_tokens=4096,
    )

    batch = _collate(
        FakeTorch,
        (shorter, longer),
        pad_token_id=tokenizer.eos_token_id,
    )

    assert batch["input_ids"].values[0][-1] == tokenizer.eos_token_id
    assert batch["labels"].values[0][-1] == -100
    assert batch["attention_mask"].values[0][-1] == 0
    assert batch["labels"].values[0].count(tokenizer.eos_token_id) == 1


def test_decoder_protocol_rejects_eos_hidden_inside_a_symbol_encoding():
    class EosInsideTokenizer:
        eos_token_id = 0

        def encode(self, text, *, add_special_tokens):
            return [1, self.eos_token_id, 2]

        def decode(self, token_ids, **kwargs):
            return "WORD_Open"

    with pytest.raises(DecoderProtocolError, match="EOS token before termination"):
        segmented_symbol_plan_tokens(EosInsideTokenizer(), "WORD_Open")


def test_source_provenance_requires_clean_checkout_bound_to_runtime_package(
    tmp_path,
    monkeypatch,
):
    checkout = tmp_path / "checkout"
    checkout_package = checkout / "src/ste_compiler"
    checkout_package.parent.mkdir(parents=True)
    runtime_package = Path(decoder_training.__file__).parents[1]
    shutil.copytree(runtime_package, checkout_package)
    (checkout / "pyproject.toml").write_text("[project]\nname='ste-compiler'\n")
    (checkout / "uv.lock").write_text("version = 1\n")
    git_status = {"value": ""}

    def fake_git(command, **kwargs):
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=git_status["value"])
        raise AssertionError(command)

    monkeypatch.setattr(decoder_training.subprocess, "run", fake_git)

    provenance = decoder_training._source_provenance(checkout)
    assert provenance.dirty is False
    assert provenance.package_tree_sha256 == decoder_training._package_tree_sha256(runtime_package)

    git_status["value"] = " M src/ste_compiler/cli.py\n"
    with pytest.raises(DecoderLoRATrainingError, match="must be clean"):
        decoder_training._source_provenance(checkout)

    git_status["value"] = ""
    with (checkout_package / "cli.py").open("a") as file:
        file.write("\n# changed checkout\n")
    with pytest.raises(DecoderLoRATrainingError, match="does not match the code executing"):
        decoder_training._source_provenance(checkout)

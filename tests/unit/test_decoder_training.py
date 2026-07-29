from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import ste_compiler.training.decoder_lora as decoder_training
from ste_compiler.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactFileV1,
    artifact_manifest_sha256,
    build_artifact_manifest,
    canonical_artifact_manifest_json,
    parse_canonical_artifact_manifest,
)
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
from ste_compiler.training.decoder_lora import _atomic_output_directory, _collate


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


def test_atomic_output_refuses_destination_created_during_staging(tmp_path):
    output = tmp_path / "output"

    def build(stage):
        (stage / "artifact.txt").write_text("complete")
        output.mkdir()
        return "built"

    with pytest.raises(DecoderLoRATrainingError, match="output path already exists"):
        _atomic_output_directory(output, build)

    assert output.is_dir()
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".output.stage-*"))


def test_invalid_output_cleanup_preserves_replacement_path(tmp_path, monkeypatch):
    output = tmp_path / "output"
    displaced = tmp_path / "displaced"
    output.mkdir()
    (output / "original").write_bytes(b"invalid")
    pinned = decoder_training._open_pinned_output_directory(output)
    real_assert = decoder_training._assert_pinned_output_directory

    def replace_after_identity_check(directory, pinned_directory, *, operation):
        real_assert(directory, pinned_directory, operation=operation)
        output.rename(displaced)
        output.mkdir()
        (output / "replacement").write_bytes(b"concurrent")

    monkeypatch.setattr(
        decoder_training,
        "_assert_pinned_output_directory",
        replace_after_identity_check,
    )
    try:
        with pytest.raises(DecoderLoRATrainingError, match="changed during invalid"):
            decoder_training._remove_invalid_pinned_output(output, pinned)
    finally:
        decoder_training.os.close(pinned.descriptor)

    assert (output / "replacement").read_bytes() == b"concurrent"
    assert displaced.is_dir()
    assert not list(displaced.iterdir())


def test_decoder_bundle_manifest_is_last_and_content_binds_the_complete_run(tmp_path):
    root = tmp_path / "run"
    for relative_path in decoder_training.DECODER_CHECKSUM_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{relative_path}\n".encode())
    decoder_training._write_checksums(root)

    digest = decoder_training._write_decoder_artifact_manifest(root)

    assert len(digest) == 64
    assert decoder_training.decoder_lora_artifact_manifest_sha256(root) == digest
    manifest_bytes = (root / ARTIFACT_MANIFEST_NAME).read_bytes()
    manifest = parse_canonical_artifact_manifest(manifest_bytes)
    assert manifest.architecture == "decoder-only-lora"
    assert manifest.artifact_type == "decoder-only-lora-run"
    assert manifest.entrypoint == "adapter"
    assert {identity.path for identity in manifest.files} == decoder_training.DECODER_BUNDLE_FILES
    checksum_paths = {
        line.split("  ", 1)[1] for line in (root / "checksums.sha256").read_text().splitlines()
    }
    assert checksum_paths == decoder_training.DECODER_CHECKSUM_FILES
    assert ARTIFACT_MANIFEST_NAME not in checksum_paths

    payload = json.loads(manifest_bytes)
    payload["entrypoint"] = "."
    (root / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(payload))
    with pytest.raises(DecoderLoRATrainingError, match="artifact manifest"):
        decoder_training.decoder_lora_artifact_manifest_sha256(root)


def test_decoder_checksum_construction_streams_covered_members(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for relative_path in decoder_training.DECODER_CHECKSUM_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{relative_path}\n".encode())

    def reject_read_bytes(self):
        raise AssertionError(f"checksum construction read all of {self}")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    checksum_bytes = decoder_training._checksum_bytes(root)

    assert checksum_bytes.count(b"\n") == len(decoder_training.DECODER_CHECKSUM_FILES)


@pytest.mark.parametrize(
    ("relative_path", "size_limit"),
    [
        ("run-manifest.json", decoder_training.MAX_DECODER_RUN_MANIFEST_BYTES),
        ("training-config.json", decoder_training.MAX_DECODER_TRAINING_CONFIG_BYTES),
        ("checksums.sha256", decoder_training.MAX_DECODER_CHECKSUM_BYTES),
    ],
)
def test_decoder_preflight_rejects_oversized_sparse_metadata_before_capture(
    tmp_path,
    monkeypatch,
    relative_path,
    size_limit,
):
    root = tmp_path / "run"
    identities = []
    for path_name in sorted(decoder_training.DECODER_BUNDLE_FILES):
        path = root / path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        byte_count = size_limit + 1 if path_name == relative_path else 0
        with path.open("wb") as file:
            file.truncate(byte_count)
        identities.append(
            ArtifactFileV1(
                path=path_name,
                sha256="0" * 64,
                bytes=byte_count,
            )
        )
    manifest = build_artifact_manifest(
        architecture="decoder-only-lora",
        artifact_type="decoder-only-lora-run",
        entrypoint="adapter",
        files=tuple(identities),
    )
    (root / ARTIFACT_MANIFEST_NAME).write_bytes(canonical_artifact_manifest_json(manifest))
    monkeypatch.setattr(
        decoder_training,
        "_runtime_modules",
        lambda: pytest.fail("oversized metadata must fail before neural runtime loading"),
    )

    with pytest.raises(
        DecoderLoRATrainingError,
        match=rf"oversized file: {relative_path}",
    ):
        decoder_training.preflight_decoder_lora_artifact_bundle(
            root,
            artifact_manifest_sha256(manifest),
        )


class _FakeSafeTensorSlice:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _FakeSafeTensorState:
    def __init__(self, tensors):
        self._tensors = tensors

    def keys(self):
        return self._tensors.keys()

    def get_slice(self, key):
        return _FakeSafeTensorSlice(self._tensors[key])


def _adapter_training_config(*, targets=("c_attn",), rank=8):
    return SimpleNamespace(
        lora=SimpleNamespace(
            rank=rank,
            target_modules=targets,
        )
    )


def test_lora_safetensors_state_dict_requires_complete_pairs_for_configured_targets():
    tensors = _FakeSafeTensorState(
        {
            "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": (8, 16),
            "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight": (48, 8),
            "base_model.model.transformer.h.1.attn.c_attn.lora_A.weight": (8, 16),
            "base_model.model.transformer.h.1.attn.c_attn.lora_B.weight": (48, 8),
        }
    )

    decoder_training._validate_lora_safetensors_state_dict(
        tensors,
        _adapter_training_config(),
    )


@pytest.mark.parametrize(
    ("tensors", "config", "message"),
    [
        (
            {
                "base_model.model.transformer.h.0.attn.c_attn.weight": (16, 16),
            },
            _adapter_training_config(),
            "invalid LoRA state-dict keys",
        ),
        (
            {
                "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": (8, 16),
            },
            _adapter_training_config(),
            "complete LoRA A/B pairs",
        ),
        (
            {
                "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": (4, 16),
                "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight": (48, 4),
            },
            _adapter_training_config(rank=8),
            "do not match configured rank 8",
        ),
        (
            {
                "base_model.model.transformer.h.0.attn.c_proj.lora_A.weight": (8, 16),
                "base_model.model.transformer.h.0.attn.c_proj.lora_B.weight": (16, 8),
            },
            _adapter_training_config(),
            "outside configured target_modules",
        ),
        (
            {
                "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": (8, 16),
                "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight": (48, 8),
            },
            _adapter_training_config(targets=("c_attn", "c_proj")),
            "do not represent configured target_modules: c_proj",
        ),
    ],
)
def test_lora_safetensors_state_dict_rejects_malformed_or_misdirected_weights(
    tensors,
    config,
    message,
):
    with pytest.raises(DecoderLoRATrainingError, match=message):
        decoder_training._validate_lora_safetensors_state_dict(
            _FakeSafeTensorState(tensors),
            config,
        )

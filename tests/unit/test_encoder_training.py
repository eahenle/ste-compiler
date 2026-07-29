from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ste_compiler.training import (
    EncoderDecoderTrainingConfigV1,
    load_training_config,
    preflight_encoder_decoder_tokenizer,
    read_training_release,
    run_encoder_decoder_training,
    verify_safe_encoder_decoder_checkpoint,
)
from ste_compiler.training import encoder_decoder as training_module
from ste_compiler.training.encoder_decoder import EncoderDecoderTrainingError

ROOT = Path(__file__).parents[2]
RELEASE = ROOT / "datasets/demonstration-corpus-1"
CONFIG = ROOT / "data/training/encoder-decoder-schema-example.yaml"


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = None

    def encode(self, text, *, add_special_tokens):
        encoded = [ord(character) + 10 for character in text]
        return [*encoded, self.eos_token_id] if add_special_tokens else encoded

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(
            chr(token_id - 10)
            for token_id in token_ids
            if token_id not in {self.eos_token_id, self.pad_token_id}
        )


def _config(**updates) -> EncoderDecoderTrainingConfigV1:
    loaded = load_training_config(CONFIG)
    assert isinstance(loaded, EncoderDecoderTrainingConfigV1)
    return loaded.model_copy(
        update={
            "max_source_tokens": 100_000,
            "max_target_tokens": 100_000,
            **updates,
        }
    )


def test_tokenizer_preflight_covers_every_split_and_appends_exact_eos():
    config = _config()
    release = read_training_release(RELEASE, config.corpus)

    prepared = preflight_encoder_decoder_tokenizer(
        CharacterTokenizer(),
        release,
        config,
    )

    assert len(prepared) == release.manifest.record_count
    assert {record.split for record in prepared} == {
        "train",
        "validation",
        "test",
        "adversarial",
    }
    assert all(record.labels[-1] == CharacterTokenizer.eos_token_id for record in prepared)


def test_tokenizer_preflight_rejects_lossy_leading_space():
    class SpaceNormalizingTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **kwargs):
            return super().decode(token_ids, **kwargs).lstrip()

    config = _config()
    release = read_training_release(RELEASE, config.corpus)

    with pytest.raises(EncoderDecoderTrainingError, match="losslessly encode symbolic form"):
        preflight_encoder_decoder_tokenizer(SpaceNormalizingTokenizer(), release, config)


def test_tokenizer_preflight_rejects_lossy_or_unknown_sources():
    class UnknownSourceTokenizer(CharacterTokenizer):
        unk_token_id = 2

        def encode(self, text, *, add_special_tokens):
            encoded = super().encode(text, add_special_tokens=add_special_tokens)
            if text.startswith("{"):
                encoded[0] = self.unk_token_id
            return encoded

    config = _config()
    release = read_training_release(RELEASE, config.corpus)

    with pytest.raises(EncoderDecoderTrainingError, match="losslessly encode source"):
        preflight_encoder_decoder_tokenizer(UnknownSourceTokenizer(), release, config)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("max_source_tokens", "exceeds max_source_tokens"),
        ("max_target_tokens", "exceeds max_target_tokens"),
    ],
)
def test_tokenizer_preflight_rejects_overflow_instead_of_truncating(field, message):
    config = _config(**{field: 1})
    release = read_training_release(RELEASE, config.corpus)

    with pytest.raises(EncoderDecoderTrainingError, match=message):
        preflight_encoder_decoder_tokenizer(CharacterTokenizer(), release, config)


def test_safe_checkpoint_rejects_pickle_capable_artifacts(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "optimizer.pt").touch()

    with pytest.raises(EncoderDecoderTrainingError, match="pickle-capable"):
        verify_safe_encoder_decoder_checkpoint(tmp_path)


def test_safe_checkpoint_returns_canonical_content_identities(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"safe weights")

    identities = verify_safe_encoder_decoder_checkpoint(tmp_path)

    assert [identity.path for identity in identities] == [
        "config.json",
        "model.safetensors",
    ]
    assert identities[1].bytes == len(b"safe weights")


def test_safe_checkpoint_rejects_symlink_even_when_target_is_regular(tmp_path):
    target = tmp_path / "weights"
    target.write_bytes(b"safe weights")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").symlink_to(target)

    with pytest.raises(EncoderDecoderTrainingError, match="non-regular entry"):
        verify_safe_encoder_decoder_checkpoint(checkpoint)


def test_snapshot_capture_supports_real_cache_symlinks_and_binds_content(tmp_path):
    blob = tmp_path / "cache" / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b'{"model_type":"t5"}')
    snapshot = tmp_path / "cache" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to(blob)

    captured = training_module._capture_tree(
        snapshot,
        tmp_path / "private",
        allow_file_symlinks=True,
    )

    assert (captured.path / "config.json").read_bytes() == blob.read_bytes()
    assert captured.artifacts[0].sha256 == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_model_tokenizer_compatibility_rejects_vocabulary_mismatch():
    class SizedTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 2
        model_max_length = 100_000

        def __len__(self):
            return 4

    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=5,
            pad_token_id=0,
            eos_token_id=1,
            unk_token_id=2,
        ),
        get_input_embeddings=lambda: SimpleNamespace(num_embeddings=5),
        get_output_embeddings=lambda: SimpleNamespace(num_embeddings=5),
    )

    with pytest.raises(EncoderDecoderTrainingError, match="vocabulary capacities"):
        training_module._validate_model_tokenizer_compatibility(
            SizedTokenizer(),
            model,
            _config(),
        )


def test_source_provenance_rejects_dirty_or_unbound_checkout(tmp_path, monkeypatch):
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")

    def git_result(command, **kwargs):
        output = "a" * 40 + "\n" if command[1] == "rev-parse" else " M changed.py\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(training_module.subprocess, "run", git_result)
    with pytest.raises(EncoderDecoderTrainingError, match="must be clean"):
        training_module._git_package_provenance(tmp_path, lock)

    def clean_git_result(command, **kwargs):
        output = "a" * 40 + "\n" if command[1] == "rev-parse" else ""
        return SimpleNamespace(stdout=output)

    hashes = iter(("b" * 64, "c" * 64))
    monkeypatch.setattr(training_module.subprocess, "run", clean_git_result)
    monkeypatch.setattr(training_module, "_package_tree_sha256", lambda path: next(hashes))
    with pytest.raises(EncoderDecoderTrainingError, match="does not match the executing"):
        training_module._git_package_provenance(tmp_path, lock)


def test_trainer_rejects_existing_output_before_loading_optional_runtime(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(EncoderDecoderTrainingError, match="must not already exist"):
        run_encoder_decoder_training(
            _config(),
            RELEASE,
            output,
            source_root=ROOT,
            dependency_lock=ROOT / "uv.lock",
        )

    assert sentinel.read_bytes() == b"unchanged"
    assert set(output.iterdir()) == {sentinel}


def test_no_replace_publication_preserves_concurrent_destination(tmp_path):
    source = tmp_path / "stage"
    destination = tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    (source / "artifact").write_bytes(b"complete")
    (destination / "sentinel").write_bytes(b"concurrent")

    with pytest.raises(EncoderDecoderTrainingError, match="created concurrently"):
        training_module._rename_no_replace(source, destination)

    assert (source / "artifact").read_bytes() == b"complete"
    assert (destination / "sentinel").read_bytes() == b"concurrent"

import hashlib
import json
import os
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.realizer.constrained import EXACT_PLAN_SYMBOL
from ste_compiler.training import (
    CorpusSelectionV1,
    DecoderOnlyLoRATrainingConfigV1,
    EncoderDecoderTrainingConfigV1,
    canonical_training_config_json,
    load_training_config,
    read_training_release,
    training_config_sha256,
)
from ste_compiler.training.config import TRAINING_CONFIG_ADAPTER

ROOT = Path(__file__).parents[2]
RELEASE = ROOT / "datasets/demonstration-corpus-1"
TRAINING_EXAMPLES = ROOT / "data/training"
MANIFEST_SHA256 = "f6ae4582669c4d7d06e33018088b900ffa0f8aa8b6e0d9f1beeccca2023faa7b"
TRAIN_SHA256 = "1772fbe01a15c28d174e139f93e5c3b0fd6744c01cf5c81b79fb842c9609ebd0"
VALIDATION_SHA256 = "ea16d6bae1f624c26581e05b02cb693282805d93f945bd6e00e73a48a79d15dd"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selection(
    *,
    manifest_sha256: str = MANIFEST_SHA256,
    train_sha256: str = TRAIN_SHA256,
    validation_sha256: str = VALIDATION_SHA256,
) -> CorpusSelectionV1:
    return CorpusSelectionV1(
        dataset_version="demonstration-corpus-1",
        manifest_sha256=manifest_sha256,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )


def _common_config() -> dict[str, object]:
    identity = {"repo_id": "example/tiny-model", "revision": "a" * 40}
    return {
        "schema_version": "ste-training-config-v1",
        "corpus": _selection().model_dump(mode="json"),
        "base_model": identity,
        "tokenizer": identity,
        "seed": 1729,
        "max_steps": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {"learning_rate": 0.0001, "weight_decay": 0},
    }


def _encoder_config() -> dict[str, object]:
    return {
        **_common_config(),
        "architecture": "encoder-decoder",
        "strategy": "full",
        "max_source_tokens": 1024,
        "max_target_tokens": 256,
    }


def _decoder_config() -> dict[str, object]:
    return {
        **_common_config(),
        "architecture": "decoder-only-lora",
        "prompt_profile": "decoder-only-symbol-plan-v1",
        "max_sequence_tokens": 1280,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"],
        },
    }


def test_training_configs_are_strict_versioned_and_canonical(tmp_path):
    encoder = TRAINING_CONFIG_ADAPTER.validate_python(_encoder_config())
    decoder = TRAINING_CONFIG_ADAPTER.validate_python(_decoder_config())

    assert isinstance(encoder, EncoderDecoderTrainingConfigV1)
    assert isinstance(decoder, DecoderOnlyLoRATrainingConfigV1)
    assert canonical_training_config_json(encoder).endswith(b"\n")
    assert training_config_sha256(encoder) == _sha256(canonical_training_config_json(encoder))

    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        json.dumps(_decoder_config()),
        encoding="utf-8",
    )
    loaded = load_training_config(config_path)
    assert loaded == decoder
    assert training_config_sha256(loaded) == training_config_sha256(decoder)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), "1729"),
        (("micro_batch_size",), True),
        (("max_source_tokens",), 1024.0),
        (("corpus", "dataset_version"), " demonstration-corpus-1 "),
    ],
)
def test_training_config_rejects_coercion_and_unstripped_identity(path, value):
    raw = _encoder_config()
    target = raw
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        TRAINING_CONFIG_ADAPTER.validate_python(raw)


def test_training_config_rejects_self_reported_execution_provenance():
    raw = _encoder_config()
    raw["package_commit"] = "b" * 40
    raw["dependency_lock_sha256"] = "c" * 64

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TRAINING_CONFIG_ADAPTER.validate_python(raw)


@pytest.mark.parametrize(
    ("name", "architecture"),
    [
        ("encoder-decoder-schema-example.yaml", "encoder-decoder"),
        ("decoder-only-lora-schema-example.yaml", "decoder-only-lora"),
    ],
)
def test_packaged_training_schema_examples_are_valid_and_pin_current_release(
    name,
    architecture,
):
    config = load_training_config(TRAINING_EXAMPLES / name)

    assert config.architecture == architecture
    snapshot = read_training_release(RELEASE, config.corpus)
    assert snapshot.manifest.record_count == 12


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "extra_forbidden"),
        (
            lambda raw: raw["base_model"].update({"revision": "main"}),
            "string_pattern_mismatch",
        ),
        (
            lambda raw: raw["base_model"].update({"repo_id": "../model"}),
            "Hugging Face Hub",
        ),
        (
            lambda raw: raw["optimizer"].update({"learning_rate": float("inf")}),
            "finite number",
        ),
    ],
)
def test_encoder_training_config_rejects_unreproducible_values(mutate, message):
    raw = _encoder_config()
    mutate(raw)

    with pytest.raises(ValidationError, match=message):
        TRAINING_CONFIG_ADAPTER.validate_python(raw)


def test_decoder_training_config_requires_base_tokenizer_identity():
    raw = _decoder_config()
    raw["tokenizer"] = {**raw["tokenizer"], "revision": "d" * 40}

    with pytest.raises(ValidationError, match="tokenizer identity must match"):
        TRAINING_CONFIG_ADAPTER.validate_python(raw)


def test_decoder_training_config_rejects_duplicate_lora_targets():
    raw = _decoder_config()
    raw["lora"]["target_modules"] = ["q_proj", "q_proj"]

    with pytest.raises(ValidationError, match="target_modules must be unique"):
        TRAINING_CONFIG_ADAPTER.validate_python(raw)


def test_training_config_loader_rejects_unknown_file_type(tmp_path):
    path = tmp_path / "config.txt"
    path.write_text("{}")

    with pytest.raises(ValueError, match="unsupported training config file type"):
        load_training_config(path)


def test_hash_pinned_release_reader_returns_verified_ordered_records():
    snapshot = read_training_release(RELEASE, _selection())

    assert snapshot.manifest.record_count == 12
    assert [record.record_id for record in snapshot.train] == [
        "train_install_panel",
        "train_negative_valve",
        "train_conditional_close",
        "train_panel_sequence",
    ]
    assert len(snapshot.validation) == 2
    assert len(snapshot.test) == 3
    assert len(snapshot.adversarial) == 3
    assert EXACT_PLAN_SYMBOL in snapshot.symbol_inventory
    assert ("train.jsonl", TRAIN_SHA256) in snapshot.artifact_sha256


def test_hash_pinned_release_snapshot_is_deeply_immutable():
    snapshot = read_training_release(RELEASE, _selection())

    with pytest.raises(FrozenInstanceError):
        snapshot.train[0].record_id = "changed"
    with pytest.raises(TypeError):
        snapshot.train[0].metadata[0] = ("changed", "value")
    with pytest.raises(TypeError):
        snapshot.manifest.split_counts[0] = ("train", 999)


@pytest.mark.parametrize(
    ("field", "digest", "message"),
    [
        ("manifest_sha256", "0" * 64, "manifest SHA-256"),
        ("train_sha256", "0" * 64, "train SHA-256"),
        ("validation_sha256", "0" * 64, "validation SHA-256"),
    ],
)
def test_release_reader_rejects_wrong_configured_identity(field, digest, message):
    values = {
        "manifest_sha256": MANIFEST_SHA256,
        "train_sha256": TRAIN_SHA256,
        "validation_sha256": VALIDATION_SHA256,
    }
    values[field] = digest

    with pytest.raises(ValueError, match=message):
        read_training_release(RELEASE, _selection(**values))


def test_release_reader_rejects_changed_artifact(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    with (release / "train.jsonl").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(ValueError, match="size does not match its manifest: train.jsonl"):
        read_training_release(release, _selection())


def test_release_reader_rejects_symlinked_artifact(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    (release / "train.jsonl").unlink()
    (release / "train.jsonl").symlink_to(RELEASE / "train.jsonl")

    with pytest.raises(ValueError, match="cannot open training release entry 'train.jsonl'"):
        read_training_release(release, _selection())


def test_release_reader_rejects_fifo_without_blocking(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    (release / "train.jsonl").unlink()
    os.mkfifo(release / "train.jsonl")

    with pytest.raises(ValueError, match="single-link regular file: train.jsonl"):
        read_training_release(release, _selection())


def test_release_reader_rejects_hardlinked_artifact(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    target = tmp_path / "validation-copy.jsonl"
    shutil.copyfile(RELEASE / "validation.jsonl", target)
    (release / "validation.jsonl").unlink()
    os.link(target, release / "validation.jsonl")

    with pytest.raises(ValueError, match="single-link regular file: validation.jsonl"):
        read_training_release(release, _selection())


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_release_reader_requires_exact_file_set(tmp_path, change):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    if change == "missing":
        (release / "dataset-card.md").unlink()
    else:
        (release / "unexpected.txt").write_text("unexpected")

    with pytest.raises(ValueError, match=change):
        read_training_release(release, _selection())


def _resign_release(release: Path, artifact_path: str) -> CorpusSelectionV1:
    artifact = release / artifact_path
    artifact_data = artifact.read_bytes()
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = next(item for item in manifest["artifacts"] if item["path"] == artifact_path)
    identity["sha256"] = _sha256(artifact_data)
    identity["bytes"] = len(artifact_data)
    if artifact_path == "source-construction.json":
        manifest["construction_sha256"] = identity["sha256"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    checksum_paths = sorted(
        path.name
        for path in release.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (release / "checksums.sha256").write_text(
        "".join(f"{_sha256((release / path).read_bytes())}  {path}\n" for path in checksum_paths)
    )
    artifacts = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    return _selection(
        manifest_sha256=_sha256(manifest_path.read_bytes()),
        train_sha256=artifacts["train.jsonl"],
        validation_sha256=artifacts["validation.jsonl"],
    )


def test_release_reader_revalidates_record_symbol_contract(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_path = release / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    records[0]["allowed_symbols"] = records[0]["allowed_symbols"][:-1]
    train_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    selection = _resign_release(release, "train.jsonl")

    with pytest.raises(ValueError, match="invalid allowed-symbol set"):
        read_training_release(release, selection)


def test_release_reader_rebuilds_deterministic_training_targets(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_path = release / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    records[0]["text"] = "Install the access panel slowly."
    train_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    selection = _resign_release(release, "train.jsonl")

    with pytest.raises(ValueError, match="does not match its deterministic training target"):
        read_training_release(release, selection)


def test_release_reader_revalidates_source_spans(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_path = release / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    span = records[0]["ir"]["sections"][0]["statements"][0]["source_spans"][0]
    span["source_id"] = "wrong-source.txt"
    records[0]["serialized_ir"] = canonical_document_json(Document.model_validate(records[0]["ir"]))
    train_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    selection = _resign_release(release, "train.jsonl")

    with pytest.raises(ValueError, match="IR does not match its construction input"):
        read_training_release(release, selection)


def test_release_reader_revalidates_semantic_features(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_path = release / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    records[0]["features"].append("state.value.string")
    records[0]["features"].sort()
    train_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    selection = _resign_release(release, "train.jsonl")

    with pytest.raises(ValueError, match="invalid semantic features"):
        read_training_release(release, selection)


def test_release_reader_rejects_incoherent_construction_hash(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["construction_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    checksum_path = release / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256((release / path).read_bytes())}  {path}\n"
            for path in sorted(
                item.name
                for item in release.iterdir()
                if item.is_file() and item.name != "checksums.sha256"
            )
        )
    )
    artifacts = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    selection = _selection(
        manifest_sha256=_sha256(manifest_path.read_bytes()),
        train_sha256=artifacts["train.jsonl"],
        validation_sha256=artifacts["validation.jsonl"],
    )

    with pytest.raises(ValueError, match="construction SHA-256 does not match"):
        read_training_release(release, selection)


def test_release_reader_rejects_manifest_declared_oversized_artifact(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train = next(item for item in manifest["artifacts"] if item["path"] == "train.jsonl")
    train["bytes"] = 64 * 1024 * 1024 + 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    selection = _selection(manifest_sha256=_sha256(manifest_path.read_bytes()))

    with pytest.raises(ValueError, match="declares an oversized artifact"):
        read_training_release(release, selection)


def test_release_reader_rematerializes_ir_from_construction(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_path = release / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    records[0]["ir"]["title"] = "Altered gold IR"
    records[0]["serialized_ir"] = canonical_document_json(Document.model_validate(records[0]["ir"]))
    train_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    selection = _resign_release(release, "train.jsonl")

    with pytest.raises(ValueError, match="IR does not match its construction input"):
        read_training_release(release, selection)


def test_release_reader_rechecks_normalized_source_leakage(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_record = json.loads((release / "train.jsonl").read_text().splitlines()[0])
    validation_path = release / "validation.jsonl"
    validation_records = [json.loads(line) for line in validation_path.read_text().splitlines()]
    validation_records[0]["source"]["text"] = train_record["source"]["text"]
    validation_records[0]["source"]["sha256"] = _sha256(train_record["source"]["text"].encode())
    validation_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in validation_records
        )
    )
    selection = _resign_release(release, "validation.jsonl")

    with pytest.raises(ValueError, match="normalized source duplicate crosses splits"):
        read_training_release(release, selection)


def test_release_reader_rechecks_compositional_leakage(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train_record = json.loads((release / "train.jsonl").read_text().splitlines()[0])
    test_path = release / "test.jsonl"
    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    test_records[0]["features"] = train_record["features"]
    test_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in test_records
        )
    )
    selection = _resign_release(release, "test.jsonl")

    with pytest.raises(ValueError, match="evaluation compositions duplicate training"):
        read_training_release(release, selection)


def test_release_reader_revalidates_construction_resource_provenance(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    construction_path = release / "source-construction.json"
    construction = json.loads(construction_path.read_text())
    construction["provenance"]["vocabulary"]["sha256"] = "0" * 64
    construction_path.write_text(
        json.dumps(construction, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    selection = _resign_release(release, "source-construction.json")

    with pytest.raises(ValueError, match="vocabulary SHA-256"):
        read_training_release(release, selection)

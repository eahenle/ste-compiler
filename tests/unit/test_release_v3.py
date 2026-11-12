"""Frozen additive corpus, source grounding and semantic holdout contracts."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ste_compiler.cli import app
from ste_compiler.terminology import TerminologyRegistry
from ste_compiler.training import (
    CorpusSelectionV1,
    build_demonstration_corpus,
    read_training_release,
)
from ste_compiler.training.release import (
    V2_REQUIRED_FEATURES,
    feature_composition,
    validate_release_profile,
)

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "data/demonstration_corpus/v3/source-construction.json"
RELEASE = ROOT / "datasets/demonstration-corpus-3"
COUNTS = {"train": 32, "validation": 8, "test": 12, "adversarial": 12}


def test_v3_rejects_unfrozen_counts():
    with pytest.raises(ValueError, match="exactly 64"):
        validate_release_profile(
            "demonstration-corpus-3", {**COUNTS, "train": 31}, set(V2_REQUIRED_FEATURES)
        )
    with pytest.raises(ValueError, match="split counts"):
        validate_release_profile(
            "demonstration-corpus-3", {**COUNTS, "train": 31, "test": 13}, set(V2_REQUIRED_FEATURES)
        )


def test_v3_requires_terminology_boundaries():
    with pytest.raises(ValueError, match="terminology.alias_surface"):
        validate_release_profile(
            "demonstration-corpus-3",
            COUNTS,
            set(V2_REQUIRED_FEATURES) - {"terminology.alias_surface"},
        )


def test_v3_build_and_select(tmp_path, vocab):
    output = tmp_path / "release"
    terms = TerminologyRegistry.load(ROOT / "data/demonstration_corpus/v3/terminology.yaml")
    manifest = build_demonstration_corpus(SOURCE, output, vocab, terms)
    assert manifest["record_count"] == 64
    assert manifest["split_counts"] == COUNTS
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        p.name: p.read_bytes() for p in RELEASE.iterdir()
    }
    selection = CorpusSelectionV1(
        dataset_version="demonstration-corpus-3",
        **{
            key + "_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()
            for key, name in [
                ("manifest", "manifest.json"),
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
            ]
        },
    )
    snapshot = read_training_release(output, selection)
    assert snapshot.manifest.record_count == 64
    assert (
        CliRunner()
        .invoke(
            app, ["build-demonstration-corpus", "--version", "3", "--output", str(tmp_path / "cli")]
        )
        .exit_code
        == 0
    )


def test_v3_adds_compositions_not_just_lexical_variants():
    def records(version):
        return {
            split: [
                json.loads(line)
                for line in (ROOT / f"datasets/demonstration-corpus-{version}" / f"{split}.jsonl")
                .read_text()
                .splitlines()
            ]
            for split in COUNTS
        }

    old, new = records(2), records(3)

    def compositions(records):
        return {feature_composition(tuple(r["features"])) for r in records}

    old_compositions = compositions([r for rs in old.values() for r in rs])
    new_compositions = compositions([r for rs in new.values() for r in rs])
    assert len(new_compositions - old_compositions) >= 10
    train = compositions(new["train"])
    assert train.isdisjoint(compositions(new["test"] + new["adversarial"]))
    # Retained examples stay in their original splits. The 40 additions are unique sources.
    old_ids = {r["record_id"]: split for split, rs in old.items() for r in rs}
    new_ids = {r["record_id"]: split for split, rs in new.items() for r in rs}
    assert old_ids.items() <= new_ids.items()
    assert len(new_ids.keys() - old_ids.keys()) == 40
    additions = [r for rs in new.values() for r in rs if r["record_id"] not in old_ids]
    # Object-renamed variants do not qualify as additional semantic examples.
    assert len(compositions(additions)) == len(additions)
    texts = [r["source"]["text"] for rs in new.values() for r in rs]
    assert len(set(texts)) == 64


@pytest.mark.parametrize(
    "mutation", ["schema", "span", "license", "source_leakage", "composition_leakage"]
)
def test_v3_rejects_invalid_construction(tmp_path, vocab, mutation):
    from copy import deepcopy

    raw = json.loads(SOURCE.read_text())
    record = raw["records"][-1]
    if mutation == "schema":
        record["unrecognized"] = True
    elif mutation == "span":
        record["source_quotes"]["step"] = "This quote does not occur."
    elif mutation == "license":
        record["license_id"] = "unlicensed"
    elif mutation == "source_leakage":
        replacement = deepcopy(raw["records"][0])
        replacement.update(id=record["id"], source_id=record["source_id"], split="adversarial")
        replacement["document"]["id"] = record["id"]
        raw["records"][-1] = replacement
    else:
        # Same feature composition with a distinct, correctly grounded source.
        replacement = deepcopy(raw["records"][0])
        replacement.update(id=record["id"], source_id=record["source_id"], split="adversarial")
        replacement["document"]["id"] = record["id"]
        replacement["source_text"] = "Install the panel."
        replacement["source_quotes"] = {"install_panel": "Install the panel."}
        raw["records"][-1] = replacement
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(raw))
    terms = TerminologyRegistry.load(ROOT / "data/demonstration_corpus/v3/terminology.yaml")
    with pytest.raises(ValueError):
        build_demonstration_corpus(source, tmp_path / "output", vocab, terms)
    assert not (tmp_path / "output").exists()


def test_v3_verification_rejects_target_tampering(tmp_path):
    import shutil

    from ste_compiler.training import verify_demonstration_corpus

    copy = tmp_path / "copy"
    shutil.copytree(RELEASE, copy)
    path = copy / "test.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["text"] = "Open the shutoff valve."
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_demonstration_corpus(copy)


def test_v3_frozen_construction_identity():
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    assert (
        manifest["construction_sha256"]
        == "3b74c35afe6e60c3add4534e8e454c56e72623d4a1230b4d93eadca2f0810cb2"
    )
    assert (
        hashlib.sha256((RELEASE / "source-construction.json").read_bytes()).hexdigest()
        == manifest["construction_sha256"]
    )


def test_added_instruction_qualifiers_are_explicit_in_source():
    raw = json.loads(SOURCE.read_text())
    for record in raw["records"][24:]:
        source = record["source_text"].casefold()
        for section in record["document"]["sections"]:
            for statement in section["statements"]:
                if statement["kind"] != "instruction":
                    continue
                for field in ("manner", "purpose"):
                    if field in statement:
                        assert statement[field] in source, record["id"]
                for relation in statement.get("temporal_relations", []):
                    assert relation["relation"] + " " + relation["event"] in source
                if statement.get("required") is False:
                    assert "can " in source
                if statement.get("negated"):
                    assert "not " in source

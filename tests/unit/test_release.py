import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.training import (
    CorpusSelectionV1,
    build_demonstration_corpus,
    build_training_record,
    read_training_release,
    verify_demonstration_corpus,
)
from ste_compiler.training.release import feature_composition

ROOT = Path(__file__).parents[2]
CONSTRUCTION = ROOT / "data/demonstration_corpus/v1/source-construction.json"
TERMINOLOGY = ROOT / "data/demonstration_corpus/v1/terminology.yaml"
RELEASE = ROOT / "datasets/demonstration-corpus-1"
V2_CONSTRUCTION = ROOT / "data/demonstration_corpus/v2/source-construction.json"
V2_TERMINOLOGY = ROOT / "data/demonstration_corpus/v2/terminology.yaml"
V2_RELEASE = ROOT / "datasets/demonstration-corpus-2"


def _corpus_terms() -> TerminologyRegistry:
    return TerminologyRegistry.load(TERMINOLOGY)


def _v2_corpus_terms() -> TerminologyRegistry:
    return TerminologyRegistry.load(V2_TERMINOLOGY)


def _release_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_demonstration_corpus_reconstructs_byte_for_byte(tmp_path, vocab):
    output = tmp_path / "release"

    manifest = build_demonstration_corpus(CONSTRUCTION, output, vocab, _corpus_terms())

    assert manifest["schema_version"] == "demonstration-corpus-release-v1"
    assert manifest["dataset_version"] == "demonstration-corpus-1"
    assert manifest["seed"] == 1729
    assert manifest["record_count"] == 12
    assert manifest["split_counts"] == {
        "train": 4,
        "validation": 2,
        "test": 3,
        "adversarial": 3,
    }
    assert _release_files(output) == _release_files(RELEASE)


def test_demonstration_corpus_v2_reconstructs_byte_for_byte(tmp_path, vocab):
    output = tmp_path / "release"

    manifest = build_demonstration_corpus(
        V2_CONSTRUCTION,
        output,
        vocab,
        _v2_corpus_terms(),
    )

    assert manifest["dataset_version"] == "demonstration-corpus-2"
    assert manifest["seed"] == 2718
    assert manifest["record_count"] == 24
    assert manifest["split_counts"] == {
        "train": 12,
        "validation": 4,
        "test": 4,
        "adversarial": 4,
    }
    coverage = json.loads((output / "feature-coverage.json").read_text())
    assert {
        "source.casing_upper",
        "source.punctuation_colon",
        "source.whitespace_tab",
        "terminology.alias_surface",
        "terminology.canonical_surface",
        "terminology.deprecated_reference",
    } <= set(coverage["required_features"])
    assert coverage["missing_features"] == []
    test_records = [json.loads(line) for line in (output / "test.jsonl").read_text().splitlines()]
    holdout = next(
        record
        for record in test_records
        if record["record_id"] == "test_negated_quantity_condition"
    )
    instruction = holdout["ir"]["sections"][0]["statements"][0]
    assert holdout["source"]["text"] == (
        "If hydraulic pressure is at least 8 MPa, do not open the shutoff valve to more than 2 mm."
    )
    assert instruction["negated"] is True
    assert instruction["conditions"][0]["value"]["comparator"] == "at_least"
    assert instruction["quantity_constraints"][0]["quantity"]["comparator"] == "more_than"
    assert holdout["text"] == (
        "If hydraulic pressure is not less than 8 MPa, do not open the shutoff valve "
        "to more than 2 mm."
    )
    assert _release_files(output) == _release_files(V2_RELEASE)


def test_demonstration_corpus_v2_enforces_benchmark_profile(tmp_path, vocab):
    construction = json.loads(V2_CONSTRUCTION.read_text())
    construction["records"] = [
        record for record in construction["records"] if record["id"] != "train_attach_cover"
    ]
    undersized = tmp_path / "construction.json"
    undersized.write_text(json.dumps(construction))

    with pytest.raises(ValueError, match="requires exactly 24 records; received 23"):
        build_demonstration_corpus(
            undersized,
            tmp_path / "release",
            vocab,
            _v2_corpus_terms(),
        )


def test_demonstration_corpus_v2_enforces_frozen_split_counts(tmp_path, vocab):
    construction = json.loads(V2_CONSTRUCTION.read_text())
    next(record for record in construction["records"] if record["id"] == "train_attach_cover")[
        "split"
    ] = "validation"
    rebalanced = tmp_path / "construction.json"
    rebalanced.write_text(json.dumps(construction))

    with pytest.raises(
        ValueError,
        match="split counts are not frozen: train=11.*validation=5",
    ):
        build_demonstration_corpus(
            rebalanced,
            tmp_path / "release",
            vocab,
            _v2_corpus_terms(),
        )


def test_demonstration_corpus_v2_is_a_valid_hash_pinned_training_release():
    manifest_bytes = (V2_RELEASE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    selection = CorpusSelectionV1(
        dataset_version="demonstration-corpus-2",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        train_sha256=artifacts["train.jsonl"]["sha256"],
        validation_sha256=artifacts["validation.jsonl"]["sha256"],
    )

    snapshot = read_training_release(V2_RELEASE, selection)

    assert snapshot.manifest.record_count == 24
    assert len(snapshot.train) == 12
    assert len(snapshot.validation) == 4


def test_hash_pinned_reader_enforces_v2_required_feature_profile(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(V2_RELEASE, release)
    records = [json.loads(line) for line in (release / "train.jsonl").read_text().splitlines()]
    deprecated = next(
        record for record in records if record["record_id"] == "train_deprecated_panel"
    )
    deprecated["ir"]["sections"][0]["statements"][0]["object"]["term_id"] = "access_panel"
    deprecated["serialized_ir"] = canonical_document_json(Document.model_validate(deprecated["ir"]))
    deprecated["features"].remove("terminology.deprecated_reference")
    deprecated["features"].remove("terminology.alias_surface")
    train_bytes = b"".join(
        (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        for record in records
    )
    (release / "train.jsonl").write_bytes(train_bytes)
    construction = json.loads((release / "source-construction.json").read_text())
    construction_record = next(
        record for record in construction["records"] if record["id"] == "train_deprecated_panel"
    )
    construction_record["document"]["sections"][0]["statements"][0]["object"]["term_id"] = (
        "access_panel"
    )
    construction_bytes = (
        json.dumps(construction, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (release / "source-construction.json").write_bytes(construction_bytes)

    manifest = json.loads((release / "manifest.json").read_text())
    train_artifact = next(
        artifact for artifact in manifest["artifacts"] if artifact["path"] == "train.jsonl"
    )
    train_artifact["sha256"] = hashlib.sha256(train_bytes).hexdigest()
    train_artifact["bytes"] = len(train_bytes)
    construction_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == "source-construction.json"
    )
    construction_artifact["sha256"] = hashlib.sha256(construction_bytes).hexdigest()
    construction_artifact["bytes"] = len(construction_bytes)
    manifest["construction_sha256"] = construction_artifact["sha256"]
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (release / "manifest.json").write_bytes(manifest_bytes)
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(release.iterdir())
        if path.name != "checksums.sha256"
    )
    (release / "checksums.sha256").write_text(checksums)
    selection = CorpusSelectionV1(
        dataset_version="demonstration-corpus-2",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        train_sha256=train_artifact["sha256"],
        validation_sha256=next(
            artifact["sha256"]
            for artifact in manifest["artifacts"]
            if artifact["path"] == "validation.jsonl"
        ),
    )

    with pytest.raises(ValueError, match="missing required features.*deprecated_reference"):
        read_training_release(release, selection)


def test_release_checksums_coverage_leakage_and_records_are_coherent(vocab):
    files = _release_files(RELEASE)
    manifest = json.loads(files["manifest.json"])
    checksums = {}
    for line in files["checksums.sha256"].decode().splitlines():
        digest, path = line.split("  ", 1)
        checksums[path] = digest
    assert set(checksums) == set(files) - {"checksums.sha256"}
    assert all(
        hashlib.sha256(files[path]).hexdigest() == digest for path, digest in checksums.items()
    )

    identities = {item["path"]: item for item in manifest["artifacts"]}
    assert set(identities) == set(files) - {"checksums.sha256", "manifest.json"}
    for path, identity in identities.items():
        assert identity["sha256"] == hashlib.sha256(files[path]).hexdigest()
        assert identity["bytes"] == len(files[path])

    coverage = json.loads(files["feature-coverage.json"])
    assert coverage["missing_features"] == []
    assert set(coverage["required_features"]) <= set(coverage["features"])
    leakage = json.loads(files["leakage-report.json"])
    assert leakage == {
        "schema_version": "leakage-report-v1",
        "document_id_cross_split": [],
        "normalized_source_cross_split": [],
        "evaluation_composition_in_train": [],
    }

    terms = _corpus_terms()
    record_ids: set[str] = set()
    source_hashes: set[str] = set()
    records_by_id: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test", "adversarial"):
        records = [json.loads(line) for line in files[f"{split}.jsonl"].splitlines()]
        for record in records:
            assert record["schema_version"] == "demonstration-corpus-record-v1"
            assert record["split"] == split
            assert record["record_id"] not in record_ids
            record_ids.add(record["record_id"])
            records_by_id[record["record_id"]] = record
            source = record["source"]
            source_bytes = source["text"].encode()
            assert source["sha256"] == hashlib.sha256(source_bytes).hexdigest()
            assert source["sha256"] not in source_hashes
            source_hashes.add(source["sha256"])
            document = Document.model_validate(record["ir"])
            assert record["serialized_ir"] == canonical_document_json(document)
            rebuilt = build_training_record(document, vocab, terms)
            assert rebuilt["text"] == record["text"]
            assert rebuilt["symbols"] == record["symbols"]
            assert rebuilt["allowed_symbols"] == record["allowed_symbols"]
            for section in document.sections:
                for statement in section.statements:
                    assert len(statement.source_spans) == 1
                    span = statement.source_spans[0]
                    assert source["text"][span.start : span.end] == span.quote
            for relation in document.causal_relations:
                assert len(relation.source_spans) == 1
                span = relation.source_spans[0]
                assert source["text"][span.start : span.end] == span.quote

    multi = Document.model_validate(records_by_id["test_multisection_state"]["ir"])
    multi_quotes = [
        statement.source_spans[0].quote
        for section in multi.sections
        for statement in section.statements
    ]
    assert multi_quotes == [
        "Hydraulic pressure is not more than 15 MPa.",
        "The unit is safe.",
        "Keep the access panel fully tight.",
    ]
    assert all(
        quote != records_by_id["test_multisection_state"]["source"]["text"]
        for quote in multi_quotes
    )

    ambiguity = Document.model_validate(records_by_id["adversarial_ambiguity"]["ir"])
    assert ambiguity.ambiguities[0].source_spans[0].quote == "ON or OFF"

    causal = Document.model_validate(records_by_id["adversarial_reference_sequence"]["ir"])
    assert causal.causal_relations[0].id == "open_panel_causes_inspection"
    assert causal.causal_relations[0].source_spans[0].quote == (
        "Opening the access panel causes inspection of the pump."
    )
    assert records_by_id["adversarial_reference_sequence"]["text"].endswith(
        "\n\nCause: Open the access panel before the test.\nEffect: Inspect the pump."
    )


def test_causal_relation_is_part_of_compositional_holdout_identity():
    assert feature_composition(("causal_relation", "reference", "statement.instruction")) == (
        "causal_relation",
        "statement.instruction",
    )


def test_release_rejects_cross_split_source_duplicates(tmp_path, vocab):
    construction = json.loads(CONSTRUCTION.read_text())
    construction["records"][4]["source_text"] = construction["records"][0]["source_text"]
    construction["records"][4]["source_quotes"]["tighten_fastener"] = construction["records"][0][
        "source_text"
    ]
    tampered = tmp_path / "construction.json"
    tampered.write_text(json.dumps(construction))

    with pytest.raises(ValueError, match="normalized source duplicate crosses splits"):
        build_demonstration_corpus(
            tampered,
            tmp_path / "release",
            vocab,
            _corpus_terms(),
        )


def test_release_rejects_missing_required_coverage(tmp_path, vocab):
    construction = json.loads(CONSTRUCTION.read_text())
    construction["records"] = [
        record for record in construction["records"] if record["id"] != "adversarial_ambiguity"
    ]
    incomplete = tmp_path / "construction.json"
    incomplete.write_text(json.dumps(construction))

    with pytest.raises(ValueError, match="missing required features.*ambiguity"):
        build_demonstration_corpus(
            incomplete,
            tmp_path / "release",
            vocab,
            _corpus_terms(),
        )


def test_release_rejects_license_claims_that_do_not_match_provenance(tmp_path, vocab):
    construction = json.loads(CONSTRUCTION.read_text())
    construction["records"][0]["license_id"] = "Proprietary"
    mismatched = tmp_path / "construction.json"
    mismatched.write_text(json.dumps(construction))

    with pytest.raises(ValueError, match="record licenses do not match construction provenance"):
        build_demonstration_corpus(
            mismatched,
            tmp_path / "release",
            vocab,
            _corpus_terms(),
        )


def test_release_rejects_replacement_resources_with_reused_identity(tmp_path, vocab):
    raw_vocabulary = vocab.data.model_dump(mode="json")
    raw_vocabulary["entries"][0]["meaning_id"] = "replacement-content"
    replacement = Vocabulary(type(vocab.data).model_validate(raw_vocabulary))

    with pytest.raises(ValueError, match="vocabulary SHA-256 does not match"):
        build_demonstration_corpus(
            CONSTRUCTION,
            tmp_path / "release",
            replacement,
            _corpus_terms(),
        )


def test_release_refuses_nonempty_output_without_modifying_it(tmp_path, vocab):
    output = tmp_path / "release"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve me")

    with pytest.raises(ValueError, match="output directory must be empty"):
        build_demonstration_corpus(CONSTRUCTION, output, vocab, _corpus_terms())

    assert sentinel.read_text() == "preserve me"
    assert set(output.iterdir()) == {sentinel}


def test_release_verifier_reconstructs_every_artifact_and_rejects_tampering(tmp_path):
    manifest = verify_demonstration_corpus(RELEASE)
    assert manifest["record_count"] == 12

    tampered = tmp_path / "tampered"
    shutil.copytree(RELEASE, tampered)
    (tampered / "dataset-card.md").write_text("tampered\n")

    with pytest.raises(ValueError, match="does not reproduce: dataset-card.md"):
        verify_demonstration_corpus(tampered)

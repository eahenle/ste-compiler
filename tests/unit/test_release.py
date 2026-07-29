import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ste_compiler.ir.models import Document
from ste_compiler.ir.serialization import canonical_document_json
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.training import (
    build_demonstration_corpus,
    build_training_record,
    verify_demonstration_corpus,
)

ROOT = Path(__file__).parents[2]
CONSTRUCTION = ROOT / "data/demonstration_corpus/v1/source-construction.json"
TERMINOLOGY = ROOT / "data/demonstration_corpus/v1/terminology.yaml"
RELEASE = ROOT / "datasets/demonstration-corpus-1"


def _corpus_terms() -> TerminologyRegistry:
    return TerminologyRegistry.load(TERMINOLOGY)


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

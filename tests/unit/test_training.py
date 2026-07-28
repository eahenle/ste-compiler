import hashlib
import json
from pathlib import Path

import pytest

from ste_compiler.ir.serialization import dumps_document, load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.training import export_symbolic_corpus

ROOT = Path(__file__).parents[2]


def test_symbolic_corpus_export_is_byte_reproducible(tmp_path, vocab, terms):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = export_symbolic_corpus(ROOT / "data/examples", first_output, vocab, terms)
    second = export_symbolic_corpus(ROOT / "data/examples", second_output, vocab, terms)

    first_bytes = (first_output / "corpus.jsonl").read_bytes()
    assert first == second
    assert first_bytes == (second_output / "corpus.jsonl").read_bytes()
    assert (first_output / "manifest.json").read_bytes() == (
        second_output / "manifest.json"
    ).read_bytes()
    assert first["record_count"] == 5
    assert first["source_files"] == [
        "conditional.yaml",
        "installation.yaml",
        "negative.yaml",
        "sequence.yaml",
        "warning_pressure.yaml",
    ]
    assert first["corpus_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert first["profiles"][0]["realizer"] == "deterministic"
    assert first["profiles"][0]["realizer_version"] == DeterministicRealizer.version
    assert first["profiles"][0]["vocabulary_version"] == vocab.data.version
    assert first["profiles"][0]["terminology_version"] == terms.data.version
    assert first["profiles"][0]["validator_profile"] == "strict-demo-1"

    records = [json.loads(line) for line in first_bytes.splitlines()]
    assert [record["source_path"] for record in records] == first["source_files"]
    assert all(record["serialized_ir"] and record["symbols"] for record in records)


def test_corpus_output_nested_inside_source_is_not_reingested(tmp_path, vocab, terms):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    output = source / "generated"

    first = export_symbolic_corpus(source, output, vocab, terms)
    second = export_symbolic_corpus(source, output, vocab, terms)

    assert first == second
    assert first["record_count"] == 1


def test_corpus_output_equal_to_source_processes_inputs_and_replaces_artifacts(
    tmp_path, vocab, terms
):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    (source / "manifest.json").write_text('{"stale":true}\n', encoding="utf-8")
    (source / "corpus.jsonl").write_text('{"stale":true}\n', encoding="utf-8")

    first = export_symbolic_corpus(source, source, vocab, terms)
    second = export_symbolic_corpus(source, source, vocab, terms)

    assert first == second
    assert first["record_count"] == 1
    assert first["source_files"] == ["installation.yaml"]
    assert json.loads((source / "manifest.json").read_text()) == first
    records = [json.loads(line) for line in (source / "corpus.jsonl").read_text().splitlines()]
    assert [record["source_path"] for record in records] == ["installation.yaml"]


@pytest.mark.parametrize("placement", ["ancestor", "sibling"])
def test_corpus_output_outside_source_does_not_suppress_inputs(
    tmp_path,
    vocab,
    terms,
    placement,
):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    output = tmp_path if placement == "ancestor" else tmp_path / "output"

    manifest = export_symbolic_corpus(source, output, vocab, terms)

    assert manifest["record_count"] == 1
    assert manifest["source_files"] == ["installation.yaml"]
    assert (output / "corpus.jsonl").is_file()


def test_corpus_export_rejects_duplicate_document_ids(tmp_path, vocab, terms):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    serialized = dumps_document(document)
    (source / "first.yaml").write_text(serialized, encoding="utf-8")
    (source / "second.yaml").write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document id.*first.yaml.*second.yaml"):
        export_symbolic_corpus(source, tmp_path / "output", vocab, terms)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("realizer", "claimed-realizer"),
        ("realizer_version", "claimed-realizer-version"),
        ("vocabulary_version", "claimed-vocabulary-version"),
        ("terminology_version", "claimed-terminology-version"),
        ("validator_profile", "claimed-validator-profile"),
    ],
)
def test_corpus_export_rejects_mismatched_runtime_profile(
    tmp_path,
    vocab,
    terms,
    field,
    invalid_value,
):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    setattr(document.metadata, field, invalid_value)
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"metadata does not match the corpus export runtime: {field}=",
    ):
        export_symbolic_corpus(source, tmp_path / "output", vocab, terms)


def test_corpus_export_rejects_symlinked_ir_file(tmp_path, vocab, terms):
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    external_ir = outside / "installation.yaml"
    external_ir.write_text(dumps_document(document), encoding="utf-8")
    (source / "linked.yaml").symlink_to(external_ir)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="symlinked IR file: linked.yaml"):
        export_symbolic_corpus(source, output, vocab, terms)
    assert not output.exists()


def test_corpus_export_rejects_direct_symlinked_ir_source(tmp_path, vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    actual_source = tmp_path / "installation.yaml"
    actual_source.write_text(dumps_document(document), encoding="utf-8")
    linked_source = tmp_path / "linked.yaml"
    linked_source.symlink_to(actual_source)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="source must not be a symlink"):
        export_symbolic_corpus(linked_source, output, vocab, terms)
    assert not output.exists()


def test_corpus_export_rejects_direct_manifest_source_without_partial_write(tmp_path, vocab, terms):
    source = tmp_path / "manifest.json"
    document = load_document(ROOT / "data/examples/installation.yaml")
    source.write_text(dumps_document(document, as_json=True), encoding="utf-8")
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="output artifact.*aliases source IR file"):
        export_symbolic_corpus(source, tmp_path, vocab, terms)

    assert source.read_bytes() == original_bytes
    assert not (tmp_path / "corpus.jsonl").exists()


def test_corpus_export_rejects_output_directory_symlink_to_source_parent(tmp_path, vocab, terms):
    actual_output = tmp_path / "actual-output"
    actual_output.mkdir()
    source = actual_output / "manifest.json"
    document = load_document(ROOT / "data/examples/installation.yaml")
    source.write_text(dumps_document(document, as_json=True), encoding="utf-8")
    original_bytes = source.read_bytes()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(actual_output, target_is_directory=True)

    with pytest.raises(ValueError, match="output artifact.*aliases source IR file"):
        export_symbolic_corpus(source, linked_output, vocab, terms)

    assert source.read_bytes() == original_bytes
    assert not (actual_output / "corpus.jsonl").exists()


@pytest.mark.parametrize("artifact_name", ["manifest.json", "corpus.jsonl"])
def test_corpus_export_rejects_symlinked_artifact_alias_without_partial_write(
    tmp_path, vocab, terms, artifact_name
):
    source = tmp_path / "source.json"
    document = load_document(ROOT / "data/examples/installation.yaml")
    source.write_text(dumps_document(document, as_json=True), encoding="utf-8")
    original_bytes = source.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / artifact_name
    artifact.symlink_to(source)
    other_artifact = output / (
        "manifest.json" if artifact_name == "corpus.jsonl" else "corpus.jsonl"
    )

    with pytest.raises(ValueError, match="output artifact.*aliases source IR file"):
        export_symbolic_corpus(source, output, vocab, terms)

    assert source.read_bytes() == original_bytes
    assert artifact.is_symlink()
    assert not other_artifact.exists()


@pytest.mark.parametrize("artifact_name", ["manifest.json", "corpus.jsonl"])
def test_corpus_export_rejects_hardlinked_artifact_alias_without_partial_write(
    tmp_path, vocab, terms, artifact_name
):
    source = tmp_path / "source.json"
    document = load_document(ROOT / "data/examples/installation.yaml")
    source.write_text(dumps_document(document, as_json=True), encoding="utf-8")
    original_bytes = source.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / artifact_name
    artifact.hardlink_to(source)
    other_artifact = output / (
        "manifest.json" if artifact_name == "corpus.jsonl" else "corpus.jsonl"
    )

    with pytest.raises(ValueError, match="output artifact.*aliases source IR file"):
        export_symbolic_corpus(source, output, vocab, terms)

    assert source.read_bytes() == original_bytes
    assert artifact.samefile(source)
    assert not other_artifact.exists()


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
@pytest.mark.parametrize("linked_artifact", ["manifest.json", "corpus.jsonl"])
def test_corpus_export_rejects_output_artifact_aliases_without_writes(
    tmp_path, vocab, terms, link_type, linked_artifact
):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    linked = output / linked_artifact
    target = output / ("corpus.jsonl" if linked_artifact == "manifest.json" else "manifest.json")
    target.write_bytes(f"{target.name} before export".encode())
    if link_type == "symlink":
        linked.symlink_to(target.name)
    else:
        linked.hardlink_to(target)
    before = {
        artifact_name: (output / artifact_name).read_bytes()
        for artifact_name in ("manifest.json", "corpus.jsonl")
    }

    with pytest.raises(ValueError, match="output artifacts.*alias each other"):
        export_symbolic_corpus(source, output, vocab, terms)

    assert {
        artifact_name: (output / artifact_name).read_bytes()
        for artifact_name in ("manifest.json", "corpus.jsonl")
    } == before
    if link_type == "symlink":
        assert linked.is_symlink()
    else:
        assert linked.samefile(target)


def test_corpus_export_rejects_symlinked_source_root(tmp_path, vocab, terms):
    actual_source = tmp_path / "actual-source"
    actual_source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (actual_source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    linked_source = tmp_path / "linked-source"
    linked_source.symlink_to(actual_source, target_is_directory=True)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="source must not be a symlink"):
        export_symbolic_corpus(linked_source, output, vocab, terms)
    assert not output.exists()


def test_training_record_rejects_invalid_deterministic_target(tmp_path, vocab, terms):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    instruction = document.sections[0].statements[0]
    document.sections[0].statements[0] = instruction.model_copy(
        update={"manner": " ".join(["fully"] * 30)}
    )
    (source / "overlong.yaml").write_text(dumps_document(document), encoding="utf-8")

    with pytest.raises(ValueError, match="was rejected: SENTENCE_TOO_LONG"):
        export_symbolic_corpus(source, tmp_path / "output", vocab, terms)

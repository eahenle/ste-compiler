import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ste_compiler.ir.serialization import dumps_document, load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.training import corpus as corpus_module
from ste_compiler.training import export_symbolic_corpus, read_symbolic_corpus

ROOT = Path(__file__).parents[2]


def _current_artifact(output: Path, artifact_name: str) -> Path:
    return output / corpus_module.CURRENT_SELECTOR / artifact_name


def _generation_directories(output: Path) -> list[Path]:
    generations = output / corpus_module.GENERATIONS_DIRECTORY
    return sorted(
        path
        for path in generations.iterdir()
        if path.name.startswith(corpus_module.GENERATION_PREFIX)
    )


def _assert_current_pair(output: Path, expected_manifest) -> None:
    snapshot = read_symbolic_corpus(output)
    assert snapshot.manifest == expected_manifest
    assert hashlib.sha256(snapshot.corpus_bytes).hexdigest() == expected_manifest["corpus_sha256"]
    assert json.loads(_current_artifact(output, "manifest.json").read_text()) == expected_manifest
    assert _current_artifact(output, "corpus.jsonl").read_bytes() == snapshot.corpus_bytes


def test_symbolic_corpus_export_is_byte_reproducible(tmp_path, vocab, terms):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = export_symbolic_corpus(ROOT / "data/examples", first_output, vocab, terms)
    second = export_symbolic_corpus(ROOT / "data/examples", second_output, vocab, terms)
    first_snapshot = read_symbolic_corpus(first_output)
    second_snapshot = read_symbolic_corpus(second_output)

    assert first == second
    assert first_snapshot == second_snapshot
    assert first_snapshot.generation_id.startswith(corpus_module.GENERATION_PREFIX)
    assert not (first_output / "corpus.jsonl").exists()
    assert not (first_output / "manifest.json").exists()
    assert first["record_count"] == 5
    assert first["source_files"] == [
        "conditional.yaml",
        "installation.yaml",
        "negative.yaml",
        "sequence.yaml",
        "warning_pressure.yaml",
    ]
    assert first["corpus_sha256"] == hashlib.sha256(first_snapshot.corpus_bytes).hexdigest()
    assert first["profiles"][0]["realizer"] == "deterministic"
    assert first["profiles"][0]["realizer_version"] == DeterministicRealizer.version
    assert first["profiles"][0]["vocabulary_version"] == vocab.data.version
    assert first["profiles"][0]["terminology_version"] == terms.data.version
    assert first["profiles"][0]["validator_profile"] == "strict-demo-1"
    records = [json.loads(line) for line in first_snapshot.corpus_bytes.splitlines()]
    assert [record["source_path"] for record in records] == first["source_files"]
    assert all(record["serialized_ir"] and record["symbols"] for record in records)


def test_repeated_identical_export_reuses_generation(tmp_path, vocab, terms):
    output = tmp_path / "output"
    first = export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    first_target = os.readlink(output / corpus_module.CURRENT_SELECTOR)
    generation = output / first_target
    first_inode = generation.stat().st_ino

    second = export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)

    assert second == first
    assert os.readlink(output / corpus_module.CURRENT_SELECTOR) == first_target
    assert generation.stat().st_ino == first_inode
    assert len(_generation_directories(output)) == 1
    _assert_current_pair(output, first)


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
    assert first["source_files"] == ["installation.yaml"]


def test_source_equal_to_output_preserves_legitimate_manifest_input(tmp_path, vocab, terms):
    source = tmp_path / "source"
    source.mkdir()
    manifest_document = load_document(ROOT / "data/examples/installation.yaml")
    sequence_document = load_document(ROOT / "data/examples/sequence.yaml")
    manifest_path = source / "manifest.json"
    other_path = source / "other.yaml"
    manifest_path.write_text(dumps_document(manifest_document, as_json=True), encoding="utf-8")
    other_path.write_text(dumps_document(sequence_document), encoding="utf-8")
    original_inputs = {
        manifest_path.name: manifest_path.read_bytes(),
        other_path.name: other_path.read_bytes(),
    }

    first = export_symbolic_corpus(source, source, vocab, terms)
    second = export_symbolic_corpus(source, source, vocab, terms)

    assert first == second
    assert first["source_files"] == ["manifest.json", "other.yaml"]
    assert {
        manifest_path.name: manifest_path.read_bytes(),
        other_path.name: other_path.read_bytes(),
    } == original_inputs
    assert len(_generation_directories(source)) == 1
    _assert_current_pair(source, first)


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
    assert _current_artifact(output, "corpus.jsonl").is_file()


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
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    external_ir = tmp_path / "installation.yaml"
    external_ir.write_text(dumps_document(document), encoding="utf-8")
    (source / "linked.yaml").symlink_to(external_ir)

    with pytest.raises(ValueError, match="symlinked IR file: linked.yaml"):
        export_symbolic_corpus(source, tmp_path / "output", vocab, terms)


@pytest.mark.parametrize("broken", [False, True])
def test_corpus_export_rejects_symlinked_ir_directory(tmp_path, vocab, terms, broken):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "missing" if broken else tmp_path / "documents"
    if not broken:
        target.mkdir()
        document = load_document(ROOT / "data/examples/installation.yaml")
        (target / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    (source / "linked").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked directory: linked"):
        export_symbolic_corpus(source, tmp_path / "output", vocab, terms)


def test_nested_output_subtree_is_pruned_before_symlink_inspection(tmp_path, vocab, terms):
    source = tmp_path / "source"
    source.mkdir()
    document = load_document(ROOT / "data/examples/installation.yaml")
    (source / "installation.yaml").write_text(dumps_document(document), encoding="utf-8")
    output = source / "generated"
    output.mkdir()
    internal_link = output / "linked"
    internal_link.symlink_to(tmp_path / "missing-directory", target_is_directory=True)

    manifest = export_symbolic_corpus(source, output, vocab, terms)

    assert manifest["source_files"] == ["installation.yaml"]
    assert internal_link.is_symlink()


def test_corpus_export_rejects_direct_symlinked_ir_source(tmp_path, vocab, terms):
    document = load_document(ROOT / "data/examples/installation.yaml")
    actual_source = tmp_path / "installation.yaml"
    actual_source.write_text(dumps_document(document), encoding="utf-8")
    linked_source = tmp_path / "linked.yaml"
    linked_source.symlink_to(actual_source)

    with pytest.raises(ValueError, match="source must not be a symlink"):
        export_symbolic_corpus(linked_source, tmp_path / "output", vocab, terms)


def test_output_directory_symlink_is_rejected_without_touching_target(tmp_path, vocab, terms):
    actual_output = tmp_path / "actual-output"
    actual_output.mkdir()
    sentinel = actual_output / "sentinel"
    sentinel.write_bytes(b"unchanged")
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(actual_output, target_is_directory=True)

    with pytest.raises(OSError):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            linked_output,
            vocab,
            terms,
        )

    assert sentinel.read_bytes() == b"unchanged"
    assert set(actual_output.iterdir()) == {sentinel}


@pytest.mark.parametrize("link_type", ["plain", "symlink", "hardlink"])
def test_unrelated_root_artifact_pair_is_never_overwritten(tmp_path, vocab, terms, link_type):
    output = tmp_path / "output"
    output.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"unrelated bytes")
    corpus_path = output / "corpus.jsonl"
    manifest_path = output / "manifest.json"
    corpus_path.write_bytes(b"root corpus bytes")
    if link_type == "plain":
        manifest_path.write_bytes(b"root manifest bytes")
    elif link_type == "symlink":
        manifest_path.symlink_to(unrelated)
    else:
        manifest_path.hardlink_to(unrelated)
    before_corpus = corpus_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )

    assert corpus_path.read_bytes() == before_corpus
    assert manifest_path.read_bytes() == before_manifest
    assert unrelated.read_bytes() == b"unrelated bytes"
    if link_type == "symlink":
        assert manifest_path.is_symlink()
    elif link_type == "hardlink":
        assert manifest_path.samefile(unrelated)
    _assert_current_pair(output, manifest)


@pytest.mark.parametrize(
    "reserved_name", [corpus_module.CURRENT_SELECTOR, corpus_module.GENERATIONS_DIRECTORY]
)
@pytest.mark.parametrize("entry_type", ["file", "symlink"])
def test_unowned_reserved_control_path_is_rejected_unchanged(
    tmp_path,
    vocab,
    terms,
    reserved_name,
    entry_type,
):
    output = tmp_path / "output"
    output.mkdir()
    reserved = output / reserved_name
    target = tmp_path / "unrelated"
    target.write_bytes(b"unrelated control bytes")
    if entry_type == "file":
        reserved.write_bytes(b"foreign reserved bytes")
    else:
        reserved.symlink_to(target)
    before = reserved.read_bytes()

    with pytest.raises(ValueError, match="current selector|generations"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    assert reserved.read_bytes() == before
    assert target.read_bytes() == b"unrelated control bytes"
    if entry_type == "symlink":
        assert reserved.is_symlink()
    assert not (output / corpus_module.OUTPUT_LOCK).exists()


@pytest.mark.parametrize("marker_state", ["missing", "wrong", "hardlink"])
def test_unowned_generations_directory_marker_is_rejected_unchanged(
    tmp_path, vocab, terms, marker_state
):
    output = tmp_path / "output"
    generations = output / corpus_module.GENERATIONS_DIRECTORY
    generations.mkdir(parents=True)
    marker = generations / corpus_module.GENERATIONS_MARKER
    unrelated = tmp_path / "unrelated-marker"
    unrelated.write_bytes(b"unrelated marker bytes")
    if marker_state == "wrong":
        marker.write_bytes(b"foreign marker bytes")
    elif marker_state == "hardlink":
        marker.hardlink_to(unrelated)
    before = {path.name: path.read_bytes() for path in generations.iterdir()}

    with pytest.raises(ValueError, match="generations directory is not tool-owned"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    assert {path.name: path.read_bytes() for path in generations.iterdir()} == before
    assert unrelated.read_bytes() == b"unrelated marker bytes"
    assert not (output / corpus_module.OUTPUT_LOCK).exists()


@pytest.mark.parametrize("link_type", ["plain", "symlink", "hardlink"])
def test_corpus_export_rejects_linked_output_lock_without_following_target(
    tmp_path, vocab, terms, link_type
):
    output = tmp_path / "output"
    output.mkdir()
    unrelated = tmp_path / "unrelated.lock"
    unrelated.write_bytes(b"unrelated lock bytes")
    lock_path = output / corpus_module.OUTPUT_LOCK
    if link_type == "plain":
        lock_path.write_bytes(b"foreign lock bytes")
    elif link_type == "symlink":
        lock_path.symlink_to(unrelated)
    else:
        lock_path.hardlink_to(unrelated)

    with pytest.raises(ValueError, match="output lock"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    assert unrelated.read_bytes() == b"unrelated lock bytes"
    assert not (output / corpus_module.GENERATIONS_DIRECTORY).exists()


def test_previous_tool_layout_migrates_without_overwriting_root_pair(tmp_path, vocab, terms):
    seed = tmp_path / "seed"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", seed, vocab, terms)
    old_pair = {
        artifact_name: _current_artifact(seed, artifact_name).read_bytes()
        for artifact_name in corpus_module.OUTPUT_ARTIFACTS
    }
    output = tmp_path / "output"
    output.mkdir()
    for artifact_name, data in old_pair.items():
        (output / artifact_name).write_bytes(data)
    (output / corpus_module.OUTPUT_LOCK).write_bytes(corpus_module.OUTPUT_LOCK_BYTES)

    manifest = export_symbolic_corpus(
        ROOT / "data/examples/sequence.yaml",
        output,
        vocab,
        terms,
    )

    assert {
        artifact_name: (output / artifact_name).read_bytes()
        for artifact_name in corpus_module.OUTPUT_ARTIFACTS
    } == old_pair
    assert (output / corpus_module.OUTPUT_LOCK).read_bytes() == corpus_module.OUTPUT_LOCK_BYTES
    _assert_current_pair(output, manifest)


def test_invalid_current_target_is_rejected_without_replacement(tmp_path, vocab, terms):
    output = tmp_path / "output"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    current = output / corpus_module.CURRENT_SELECTOR
    current.unlink()
    current.symlink_to("../unrelated")

    with pytest.raises(ValueError, match="current selector has an invalid target"):
        export_symbolic_corpus(ROOT / "data/examples/sequence.yaml", output, vocab, terms)

    assert current.is_symlink()
    assert os.readlink(current) == "../unrelated"
    assert len(_generation_directories(output)) == 1


def test_corrupt_existing_generation_is_rejected_without_replacement(tmp_path, vocab, terms):
    output = tmp_path / "output"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    generation = output / os.readlink(output / corpus_module.CURRENT_SELECTOR)
    manifest_path = generation / "manifest.json"
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(b"corrupt manifest")

    with pytest.raises(ValueError, match="generation is not coherent"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    assert manifest_path.read_bytes() == b"corrupt manifest"


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_linked_generation_artifact_is_rejected_without_following_target(
    tmp_path, vocab, terms, link_type
):
    output = tmp_path / "output"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    generation = output / os.readlink(output / corpus_module.CURRENT_SELECTOR)
    generation.chmod(0o700)
    manifest_path = generation / "manifest.json"
    manifest_path.unlink()
    unrelated = tmp_path / "unrelated-generation-file"
    unrelated.write_bytes(b"unrelated generation bytes")
    if link_type == "symlink":
        manifest_path.symlink_to(unrelated)
    else:
        manifest_path.hardlink_to(unrelated)

    with pytest.raises(ValueError, match="generation is not coherent"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    assert unrelated.read_bytes() == b"unrelated generation bytes"
    if link_type == "symlink":
        assert manifest_path.is_symlink()
    else:
        assert manifest_path.samefile(unrelated)


def test_generation_id_collision_is_rejected_without_replacing_generation(
    tmp_path, monkeypatch, vocab, terms
):
    output = tmp_path / "output"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    generation_id = read_symbolic_corpus(output).generation_id
    before = {
        artifact_name: _current_artifact(output, artifact_name).read_bytes()
        for artifact_name in corpus_module.OUTPUT_ARTIFACTS
    }
    monkeypatch.setattr(corpus_module, "_generation_id", lambda *_: generation_id)

    with pytest.raises(ValueError, match="generation id collision"):
        export_symbolic_corpus(ROOT / "data/examples/sequence.yaml", output, vocab, terms)

    assert {
        artifact_name: _current_artifact(output, artifact_name).read_bytes()
        for artifact_name in corpus_module.OUTPUT_ARTIFACTS
    } == before


def test_reader_rejects_coherent_pair_under_wrong_generation_id(tmp_path, vocab, terms):
    output = tmp_path / "output"
    export_symbolic_corpus(ROOT / "data/examples/installation.yaml", output, vocab, terms)
    snapshot = read_symbolic_corpus(output)
    wrong_generation_id = f"{corpus_module.GENERATION_PREFIX}{'0' * 64}"
    assert wrong_generation_id != snapshot.generation_id
    wrong_generation = output / corpus_module.GENERATIONS_DIRECTORY / wrong_generation_id
    wrong_generation.mkdir()
    for artifact_name in corpus_module.OUTPUT_ARTIFACTS:
        (wrong_generation / artifact_name).write_bytes(
            _current_artifact(output, artifact_name).read_bytes()
        )
    current = output / corpus_module.CURRENT_SELECTOR
    current.unlink()
    current.symlink_to(f"{corpus_module.GENERATIONS_DIRECTORY}/{wrong_generation_id}")

    with pytest.raises(ValueError, match="generation is not coherent"):
        read_symbolic_corpus(output)


def test_reader_pins_generation_across_current_switch(tmp_path, monkeypatch, vocab, terms):
    output = tmp_path / "output"
    old_manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )
    old_generation = read_symbolic_corpus(output).generation_id
    reader_generation_opened = threading.Event()
    allow_reader = threading.Event()
    original_open = corpus_module._open_directory_entry
    reader_generation_opens = 0

    def pause_after_reader_pins(directory_fd, name):
        nonlocal reader_generation_opens
        result = original_open(directory_fd, name)
        if threading.current_thread().name.startswith("pinned-reader") and name == old_generation:
            reader_generation_opens += 1
            if reader_generation_opens == 1:
                reader_generation_opened.set()
                assert allow_reader.wait(timeout=10)
        return result

    monkeypatch.setattr(corpus_module, "_open_directory_entry", pause_after_reader_pins)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pinned-reader") as executor:
        reading = executor.submit(read_symbolic_corpus, output)
        assert reader_generation_opened.wait(timeout=10)
        new_manifest = export_symbolic_corpus(
            ROOT / "data/examples/sequence.yaml",
            output,
            vocab,
            terms,
        )
        allow_reader.set()
        pinned = reading.result(timeout=10)

    assert pinned.generation_id == old_generation
    assert pinned.manifest == old_manifest
    assert hashlib.sha256(pinned.corpus_bytes).hexdigest() == old_manifest["corpus_sha256"]
    current = read_symbolic_corpus(output)
    assert current.manifest == new_manifest
    assert current.generation_id != old_generation


def test_concurrent_publishers_leave_one_coherent_generation_selected(
    tmp_path, monkeypatch, vocab, terms
):
    output = tmp_path / "output"
    first_at_switch = threading.Event()
    release_first = threading.Event()
    second_attempted_lock = threading.Event()
    publisher = threading.local()
    original_switch = corpus_module._switch_current_generation
    original_acquire = corpus_module._acquire_output_lock

    def observed_acquire(directory_fd, output_path):
        if publisher.name == "second":
            second_attempted_lock.set()
        return original_acquire(directory_fd, output_path)

    def paused_switch(directory_fd, generation_id):
        if publisher.name == "first":
            first_at_switch.set()
            assert release_first.wait(timeout=10)
        original_switch(directory_fd, generation_id)

    def publish(name, source):
        publisher.name = name
        return export_symbolic_corpus(source, output, vocab, terms)

    monkeypatch.setattr(corpus_module, "_acquire_output_lock", observed_acquire)
    monkeypatch.setattr(corpus_module, "_switch_current_generation", paused_switch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            publish,
            "first",
            ROOT / "data/examples/installation.yaml",
        )
        assert first_at_switch.wait(timeout=10)
        second = executor.submit(
            publish,
            "second",
            ROOT / "data/examples/sequence.yaml",
        )
        assert second_attempted_lock.wait(timeout=10)
        release_first.set()
        first.result(timeout=10)
        second_manifest = second.result(timeout=10)

    snapshot = read_symbolic_corpus(output)
    assert snapshot.manifest == second_manifest
    assert snapshot.manifest["source_files"] == ["sequence.yaml"]
    assert len(_generation_directories(output)) == 2


@pytest.mark.parametrize("interruption", ["before_install", "after_install"])
def test_interrupted_initial_lock_install_is_recovered(
    tmp_path,
    monkeypatch,
    vocab,
    terms,
    interruption,
):
    output = tmp_path / "output"
    original_install = corpus_module._install_output_lock

    def interrupted_install(directory_fd, temporary_name):
        if interruption == "after_install":
            original_install(directory_fd, temporary_name)
        raise OSError(f"injected {interruption}")

    monkeypatch.setattr(corpus_module, "_install_output_lock", interrupted_install)

    with pytest.raises(OSError, match=f"injected {interruption}"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )

    lock_path = output / corpus_module.OUTPUT_LOCK
    temporary_pattern = f"{corpus_module.LOCK_INIT_TEMP_PREFIX}*.tmp"
    if interruption == "before_install":
        assert not lock_path.exists()
    else:
        assert lock_path.read_bytes() == corpus_module.OUTPUT_LOCK_BYTES
        assert lock_path.stat().st_nlink == 2
    assert len(list(output.glob(temporary_pattern))) == 1

    monkeypatch.setattr(corpus_module, "_install_output_lock", original_install)
    manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )

    assert lock_path.read_bytes() == corpus_module.OUTPUT_LOCK_BYTES
    assert lock_path.stat().st_nlink == 1
    assert not list(output.glob(temporary_pattern))
    _assert_current_pair(output, manifest)


def test_concurrent_first_lock_creation_uses_one_valid_winner(tmp_path, monkeypatch, vocab, terms):
    output = tmp_path / "output"
    installers_ready = threading.Barrier(2)
    original_install = corpus_module._install_output_lock

    def racing_install(directory_fd, temporary_name):
        installers_ready.wait(timeout=10)
        original_install(directory_fd, temporary_name)

    monkeypatch.setattr(corpus_module, "_install_output_lock", racing_install)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            export_symbolic_corpus,
            ROOT / "data/examples/installation.yaml",
            output,
            vocab,
            terms,
        )
        second = executor.submit(
            export_symbolic_corpus,
            ROOT / "data/examples/sequence.yaml",
            output,
            vocab,
            terms,
        )
        first.result(timeout=10)
        second.result(timeout=10)

    lock_path = output / corpus_module.OUTPUT_LOCK
    assert lock_path.read_bytes() == corpus_module.OUTPUT_LOCK_BYTES
    assert lock_path.stat().st_nlink == 1
    assert not list(output.glob(f"{corpus_module.LOCK_INIT_TEMP_PREFIX}*.tmp"))
    snapshot = read_symbolic_corpus(output)
    assert snapshot.manifest["source_files"] in [["installation.yaml"], ["sequence.yaml"]]
    assert len(_generation_directories(output)) == 2


def test_failed_generation_staging_keeps_current_and_is_cleaned_on_retry(
    tmp_path, monkeypatch, vocab, terms
):
    output = tmp_path / "output"
    old_manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )
    old_target = os.readlink(output / corpus_module.CURRENT_SELECTOR)
    original_write = corpus_module._write_new_file

    def fail_manifest_write(directory_fd, name, data, *, mode=0o600):
        if name == "manifest.json":
            raise OSError("injected staging interruption")
        original_write(directory_fd, name, data, mode=mode)

    monkeypatch.setattr(corpus_module, "_write_new_file", fail_manifest_write)

    with pytest.raises(OSError, match="injected staging interruption"):
        export_symbolic_corpus(ROOT / "data/examples/sequence.yaml", output, vocab, terms)

    assert os.readlink(output / corpus_module.CURRENT_SELECTOR) == old_target
    assert read_symbolic_corpus(output).manifest == old_manifest
    generations = output / corpus_module.GENERATIONS_DIRECTORY
    assert any(path.name.endswith(corpus_module.STAGE_SUFFIX) for path in generations.iterdir())

    monkeypatch.setattr(corpus_module, "_write_new_file", original_write)
    new_manifest = export_symbolic_corpus(
        ROOT / "data/examples/sequence.yaml",
        output,
        vocab,
        terms,
    )
    assert read_symbolic_corpus(output).manifest == new_manifest
    assert not any(path.name.endswith(corpus_module.STAGE_SUFFIX) for path in generations.iterdir())


@pytest.mark.parametrize("failure_point", ["before", "after"])
def test_pointer_switch_interruption_never_selects_incomplete_generation(
    tmp_path,
    monkeypatch,
    vocab,
    terms,
    failure_point,
):
    output = tmp_path / "output"
    old_manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )
    original_replace = corpus_module._atomic_replace

    def interrupted_replace(directory_fd, temporary_name, artifact_name):
        assert artifact_name == corpus_module.CURRENT_SELECTOR
        if failure_point == "after":
            original_replace(directory_fd, temporary_name, artifact_name)
        raise OSError(f"injected {failure_point} selector interruption")

    monkeypatch.setattr(corpus_module, "_atomic_replace", interrupted_replace)

    with pytest.raises(OSError, match=f"injected {failure_point} selector interruption"):
        export_symbolic_corpus(ROOT / "data/examples/sequence.yaml", output, vocab, terms)

    snapshot = read_symbolic_corpus(output)
    if failure_point == "before":
        assert snapshot.manifest == old_manifest
    else:
        assert snapshot.manifest["source_files"] == ["sequence.yaml"]
    assert hashlib.sha256(snapshot.corpus_bytes).hexdigest() == snapshot.manifest["corpus_sha256"]
    assert not list(output.glob(f"{corpus_module.CURRENT_TEMP_PREFIX}*.tmp"))


def test_stale_owned_stage_and_selector_are_cleaned_on_next_export(tmp_path, vocab, terms):
    output = tmp_path / "output"
    manifest = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )
    generations = output / corpus_module.GENERATIONS_DIRECTORY
    stage = generations / f"{corpus_module.GENERATION_STAGE_PREFIX}{'a' * 32}.stage"
    stage.mkdir()
    (stage / "corpus.jsonl").write_bytes(b"incomplete")
    stage.chmod(0o500)
    stale_selector = output / f"{corpus_module.CURRENT_TEMP_PREFIX}{'b' * 32}.tmp"
    stale_selector.symlink_to(os.readlink(output / corpus_module.CURRENT_SELECTOR))

    repeated = export_symbolic_corpus(
        ROOT / "data/examples/installation.yaml",
        output,
        vocab,
        terms,
    )

    assert repeated == manifest
    assert not stage.exists()
    assert not stale_selector.exists()
    _assert_current_pair(output, manifest)


def test_corpus_export_reports_missing_posix_lock_support(tmp_path, monkeypatch, vocab, terms):
    original_import_module = corpus_module.importlib.import_module

    def import_without_fcntl(name, package=None):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
        return original_import_module(name, package)

    monkeypatch.setattr(corpus_module.importlib, "import_module", import_without_fcntl)

    with pytest.raises(ValueError, match="requires POSIX fcntl file locking"):
        export_symbolic_corpus(
            ROOT / "data/examples/installation.yaml",
            tmp_path / "output",
            vocab,
            terms,
        )


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

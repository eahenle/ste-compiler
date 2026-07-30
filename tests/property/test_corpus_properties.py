import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ste_compiler.training import (
    CorpusSelectionV1,
    read_training_release,
    verify_demonstration_corpus,
)
from ste_compiler.training.release import EXPECTED_RELEASE_FILES

ROOT = Path(__file__).parents[2]
RELEASE = ROOT / "datasets/demonstration-corpus-2"
MUTABLE_RELEASE_FILES = tuple(sorted(EXPECTED_RELEASE_FILES))


def _selection() -> CorpusSelectionV1:
    manifest_bytes = (RELEASE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    return CorpusSelectionV1(
        dataset_version="demonstration-corpus-2",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        train_sha256=artifacts["train.jsonl"]["sha256"],
        validation_sha256=artifacts["validation.jsonl"]["sha256"],
    )


@st.composite
def release_mutations(draw: st.DrawFn) -> tuple[str, str, int]:
    artifact = draw(st.sampled_from(MUTABLE_RELEASE_FILES))
    data = (RELEASE / artifact).read_bytes()
    operation = draw(st.sampled_from(["tamper", "truncate"]))
    position = draw(st.integers(min_value=0, max_value=len(data) - 1))
    return artifact, operation, position


@settings(max_examples=30, deadline=None)
@given(split=st.sampled_from(["train", "validation", "test", "adversarial"]))
def test_corpus_v2_snapshot_identity_is_stable(split: str) -> None:
    selection = _selection()
    first = read_training_release(RELEASE, selection)
    second = read_training_release(RELEASE, selection)

    assert first == second
    assert first.manifest_sha256 == selection.manifest_sha256
    records = getattr(first, split)
    assert records
    assert len({record.record_id for record in records}) == len(records)
    assert verify_demonstration_corpus(RELEASE)["dataset_version"] == selection.dataset_version


@settings(max_examples=30, deadline=None)
@given(mutation=release_mutations())
def test_corpus_v2_rejects_tampered_or_truncated_release_entries(
    mutation: tuple[str, str, int],
) -> None:
    artifact, operation, position = mutation
    with tempfile.TemporaryDirectory(prefix="ste-compiler-property-corpus-") as temporary:
        release = Path(temporary) / "release"
        shutil.copytree(RELEASE, release)
        path = release / artifact
        data = path.read_bytes()
        if operation == "tamper":
            changed = bytes([data[position] ^ 0x01])
            path.write_bytes(data[:position] + changed + data[position + 1 :])
        else:
            path.write_bytes(data[:position])

        with pytest.raises(ValueError):
            read_training_release(release, _selection())


EXTRA_NAMES = st.from_regex(r"[a-z][a-z0-9_-]{0,15}\.(?:json|txt|bin)", fullmatch=True).filter(
    lambda name: name not in EXPECTED_RELEASE_FILES
)


@settings(max_examples=25, deadline=None)
@given(extra_name=EXTRA_NAMES, payload=st.binary(max_size=128))
def test_corpus_v2_rejects_extra_release_entries(extra_name: str, payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="ste-compiler-property-corpus-") as temporary:
        release = Path(temporary) / "release"
        shutil.copytree(RELEASE, release)
        (release / extra_name).write_bytes(payload)

        with pytest.raises(ValueError, match="file set is invalid.*unexpected"):
            read_training_release(release, _selection())

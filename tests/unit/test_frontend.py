import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ste_compiler.frontend import LLMFrontend, ReplayIRProvider

ROOT = Path(__file__).parents[2]
EXAMPLE_ROOT = ROOT / "data/end_to_end"


def _proposal() -> dict[str, object]:
    raw = yaml.safe_load((EXAMPLE_ROOT / "hydraulic_warning.ir.yaml").read_text())
    assert isinstance(raw, dict)
    proposal = raw["ir"]
    assert isinstance(proposal, dict)
    return proposal


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def test_replay_frontend_verifies_source_and_overrides_proposed_identity():
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    proposal = _proposal()
    metadata = proposal.setdefault("metadata", {})
    assert isinstance(metadata, dict)
    metadata["frontend"] = "untrusted-claim"
    metadata["frontend_version"] = "untrusted-version"

    document = LLMFrontend(ReplayIRProvider(proposal, _source_digest(source))).parse(
        source,
        source_id="hydraulic_warning.txt",
    )

    assert document.id == "hydraulic_warning_source"
    assert document.metadata.frontend == "offline-replay"
    assert document.metadata.frontend_version == LLMFrontend.version


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_id": "other.txt"}, "expected 'hydraulic_warning.txt'"),
        ({"source_id": " "}, "String should match pattern"),
        ({"end": 10_000}, "exceeds source length"),
        ({"quote": "Stop something else."}, "quote does not match"),
    ],
)
def test_replay_frontend_rejects_unverifiable_provenance(change, message):
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    proposal = _proposal()
    span = proposal["sections"][0]["statements"][0]["source_spans"][0]
    span.update(change)

    with pytest.raises(ValueError, match=message):
        LLMFrontend(ReplayIRProvider(proposal, _source_digest(source)), retries=0).parse(
            source,
            source_id="hydraulic_warning.txt",
        )


def test_replay_frontend_retries_after_provenance_feedback():
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    valid = _proposal()
    invalid = _proposal()
    invalid["sections"][0]["statements"][0]["source_spans"][0]["quote"] = "wrong"

    class Provider:
        model_id = "repairing-provider"

        def __init__(self):
            self.feedback: list[str | None] = []

        def extract_ir(self, source, schema, feedback):
            del source, schema
            self.feedback.append(feedback)
            return invalid if feedback is None else valid

    provider = Provider()
    document = LLMFrontend(provider, retries=1).parse(
        source,
        source_id="hydraulic_warning.txt",
    )

    assert document.id == "hydraulic_warning_source"
    assert provider.feedback[0] is None
    assert "quote does not match" in provider.feedback[1]


def test_replay_provider_rejects_unsupported_or_non_object_fixtures(tmp_path):
    unsupported = tmp_path / "proposal.txt"
    unsupported.write_text("{}")
    with pytest.raises(ValueError, match="unsupported replay IR file type"):
        ReplayIRProvider.from_path(unsupported)

    sequence = tmp_path / "proposal.yaml"
    sequence.write_text("- not\n- an\n- object\n")
    with pytest.raises(ValueError, match="string-keyed object"):
        ReplayIRProvider.from_path(sequence)

    missing_envelope = tmp_path / "missing-envelope.yaml"
    missing_envelope.write_text("ir: {}\n")
    with pytest.raises(ValueError, match="must contain exactly"):
        ReplayIRProvider.from_path(missing_envelope)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("sections: [unterminated")
    with pytest.raises(ValueError, match="invalid replay IR fixture"):
        ReplayIRProvider.from_path(malformed)


def test_replay_provider_loads_json_fixture(tmp_path):
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    fixture = tmp_path / "proposal.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_version": "replay-ir-v1",
                "source_sha256": _source_digest(source),
                "ir": _proposal(),
            }
        )
    )

    assert ReplayIRProvider.from_path(fixture).extract_ir(source, {}, None) == _proposal()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "replay-ir-v2", "schema_version must be"),
        ("ir", [], "ir must contain a string-keyed object"),
        ("source_sha256", 1, "source_sha256 must be a string"),
    ],
)
def test_replay_provider_rejects_invalid_envelope_values(tmp_path, field, value, message):
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    payload = {
        "schema_version": "replay-ir-v1",
        "source_sha256": _source_digest(source),
        "ir": _proposal(),
    }
    payload[field] = value
    fixture = tmp_path / "proposal.yaml"
    fixture.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=message):
        ReplayIRProvider.from_path(fixture)


def test_replay_provider_rejects_invalid_source_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ReplayIRProvider({}, "not-a-digest")


def test_replay_provider_binds_proposal_to_the_complete_source():
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    provider = ReplayIRProvider(_proposal(), _source_digest(source))

    with pytest.raises(ValueError, match="source SHA-256 does not match"):
        provider.extract_ir(
            source + "\nDisconnect the pump.",
            {},
            None,
        )


def test_llm_frontend_rejects_blank_provider_identity():
    class Provider:
        model_id = " "

        def extract_ir(self, source, schema, feedback):
            raise AssertionError

    with pytest.raises(ValueError, match="model_id must be nonblank"):
        LLMFrontend(Provider())


@pytest.mark.parametrize("retries", [-1, True, 1.0])
def test_llm_frontend_rejects_invalid_retry_limits(retries: object) -> None:
    class Provider:
        model_id = "test-provider"

        def extract_ir(self, source, schema, feedback):
            raise AssertionError

    with pytest.raises(ValueError, match="retries must be a non-negative integer"):
        LLMFrontend(Provider(), retries=retries)  # type: ignore[arg-type]


def test_llm_frontend_rejects_blank_source_quote():
    source = (EXAMPLE_ROOT / "hydraulic_warning.txt").read_text()
    proposal = _proposal()
    proposal["sections"][0]["statements"][0]["source_spans"][0]["quote"] = " "

    with pytest.raises(ValueError, match="nonblank source quotes"):
        LLMFrontend(ReplayIRProvider(proposal, _source_digest(source)), retries=0).parse(source)

from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ste_compiler.frontend import LLMFrontend

SOURCE_SENTINEL = "SOURCE_SENTINEL: internal maintenance procedure"
CREDENTIAL_SENTINEL = "CREDENTIAL_SENTINEL"

MALFORMED_OUTPUTS = st.sampled_from(
    [
        None,
        [],
        {},
        {"sections": []},
        {"id": "proposal", "sections": [{"id": "empty", "kind": "procedure", "statements": []}]},
        {"id": "proposal", "sections": "not-a-list"},
    ]
)


class MalformedProvider:
    model_id = "malformed-provider"

    def __init__(self, output: object):
        self.output = output
        self.credential = CREDENTIAL_SENTINEL
        self.feedback: list[str | None] = []

    def extract_ir(
        self,
        source: str,
        schema: dict[str, object],
        feedback: str | None,
    ) -> dict[str, object]:
        del source, schema
        self.feedback.append(feedback)
        return cast(dict[str, object], self.output)


@settings(max_examples=30, deadline=None)
@given(output=MALFORMED_OUTPUTS, retries=st.integers(min_value=0, max_value=3))
def test_malformed_provider_outputs_exhaust_only_the_bounded_validation_attempts(
    output: object,
    retries: int,
) -> None:
    provider = MalformedProvider(output)

    with pytest.raises((ValidationError, ValueError)) as raised:
        LLMFrontend(provider, retries=retries).parse(
            SOURCE_SENTINEL,
            source_id="internal.txt",
        )

    assert len(provider.feedback) == retries + 1
    assert provider.feedback[0] is None
    assert all(
        feedback is not None and feedback.startswith("Schema/provenance validation failed:")
        for feedback in provider.feedback[1:]
    )
    exposed = "\n".join(
        [
            str(raised.value),
            *(feedback or "" for feedback in provider.feedback),
        ]
    )
    assert SOURCE_SENTINEL not in exposed
    assert CREDENTIAL_SENTINEL not in exposed


TRANSPORT_ERRORS = st.sampled_from([TimeoutError, ConnectionError])


@settings(max_examples=10, deadline=None)
@given(error_type=TRANSPORT_ERRORS, configured_retries=st.integers(min_value=0, max_value=5))
def test_transport_failures_propagate_once_without_frontend_added_sensitive_context(
    error_type: type[TimeoutError] | type[ConnectionError],
    configured_retries: int,
) -> None:
    provider_error = error_type("provider unavailable")

    class UnavailableProvider:
        model_id = "unavailable-provider"

        def __init__(self) -> None:
            self.credential = CREDENTIAL_SENTINEL
            self.calls = 0

        def extract_ir(
            self,
            source: str,
            schema: dict[str, object],
            feedback: str | None,
        ) -> dict[str, object]:
            del source, schema, feedback
            self.calls += 1
            raise provider_error

    provider = UnavailableProvider()
    with pytest.raises(error_type) as raised:
        LLMFrontend(provider, retries=configured_retries).parse(
            SOURCE_SENTINEL,
            source_id="internal.txt",
        )

    assert raised.value is provider_error
    assert provider.calls == 1
    assert SOURCE_SENTINEL not in str(raised.value)
    assert CREDENTIAL_SENTINEL not in str(raised.value)

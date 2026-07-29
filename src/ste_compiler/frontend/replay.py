"""Offline replay of a stored IR proposal through the live frontend boundary."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import yaml

REPLAY_FIXTURE_SCHEMA_VERSION = "replay-ir-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReplayIRProvider:
    """Return one stored proposal without representing replay as model extraction."""

    model_id = "offline-replay"
    version = "0.1.0"

    def __init__(self, proposal: dict[str, object], source_sha256: str):
        if SHA256_PATTERN.fullmatch(source_sha256) is None:
            raise ValueError("replay IR source_sha256 must be a lowercase SHA-256 digest")
        self._proposal = deepcopy(proposal)
        self._source_sha256 = source_sha256

    @classmethod
    def from_path(cls, path: Path) -> "ReplayIRProvider":
        suffix = path.suffix.casefold()
        try:
            if suffix == ".json":
                raw: object = json.loads(path.read_text(encoding="utf-8"))
            elif suffix in {".yaml", ".yml"}:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                raise ValueError(f"unsupported replay IR file type: {path.suffix or '<none>'}")
        except (json.JSONDecodeError, yaml.YAMLError) as error:
            raise ValueError(f"invalid replay IR fixture: {error}") from error
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("replay IR fixture must contain a string-keyed object")
        if set(raw) != {"schema_version", "source_sha256", "ir"}:
            raise ValueError(
                "replay IR fixture must contain exactly schema_version, source_sha256, and ir"
            )
        if raw["schema_version"] != REPLAY_FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"replay IR fixture schema_version must be {REPLAY_FIXTURE_SCHEMA_VERSION!r}"
            )
        proposal = raw["ir"]
        if not isinstance(proposal, dict) or not all(isinstance(key, str) for key in proposal):
            raise ValueError("replay IR fixture ir must contain a string-keyed object")
        source_sha256 = raw["source_sha256"]
        try:
            if not isinstance(source_sha256, str):
                raise TypeError("source_sha256 must be a string")
            return cls(proposal, source_sha256)
        except TypeError as error:
            raise ValueError(f"invalid replay IR fixture: {error}") from error

    def extract_ir(
        self,
        source: str,
        schema: dict[str, object],
        feedback: str | None,
    ) -> dict[str, object]:
        del schema, feedback
        actual_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual_sha256 != self._source_sha256:
            raise ValueError(
                "replay IR source SHA-256 does not match the fixture; "
                "regenerate and re-review the gold IR"
            )
        return deepcopy(self._proposal)

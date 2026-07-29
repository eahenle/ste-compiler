"""Offline replay of a stored IR proposal through the live frontend boundary."""

import json
from copy import deepcopy
from pathlib import Path

import yaml


class ReplayIRProvider:
    """Return one stored proposal without representing replay as model extraction."""

    model_id = "offline-replay"
    version = "0.1.0"

    def __init__(self, proposal: dict[str, object]):
        self._proposal = deepcopy(proposal)

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
        return cls(raw)

    def extract_ir(
        self,
        source: str,
        schema: dict[str, object],
        feedback: str | None,
    ) -> dict[str, object]:
        del source, schema, feedback
        return deepcopy(self._proposal)

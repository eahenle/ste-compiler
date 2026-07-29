import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_citation_metadata_matches_package_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == project["name"]
    assert citation["version"] == project["version"]
    assert citation["license"] == "MIT"
    assert citation["repository-code"] == project["urls"]["Homepage"]


def test_required_open_source_policy_files_are_present_and_complete():
    required = [
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "SECURITY.md",
        "docs/release-policy.md",
        "src/ste_compiler/py.typed",
    ]

    for path in required:
        assert (ROOT / path).is_file(), path
    assert "TODO" not in (ROOT / "SECURITY.md").read_text()


def test_reproducible_lock_and_decoder_quick_start_are_shipped():
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    readme = (ROOT / "README.md").read_text()

    assert (ROOT / "uv.lock").is_file()
    assert "uv.lock" not in ignored
    assert "'.[dev,neural]'" in readme
    assert 'DECODER_SMOKE_ROOT="$(mktemp -d ' in readme
    assert 'DECODER_SMOKE_MODEL="$DECODER_SMOKE_ROOT/model"' in readme
    assert 'DECODER_SMOKE_RUN="$DECODER_SMOKE_ROOT/run"' in readme
    assert 'MODEL_SNAPSHOT_MANIFEST_SHA256="$(' in readme
    assert '"$MODEL_SNAPSHOT_MANIFEST_SHA256"' in readme

import re
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
        "docs/release-build-provenance.md",
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


def _workflow(name: str) -> dict[str, object]:
    loaded = yaml.load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(loaded, dict)
    return loaded


def _workflow_action_references(workflow: dict[str, object]) -> list[str]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    references: list[str] = []
    for job in jobs.values():
        assert isinstance(job, dict)
        steps = job["steps"]
        assert isinstance(steps, list)
        for step in steps:
            assert isinstance(step, dict)
            reference = step.get("uses")
            if reference is not None:
                assert isinstance(reference, str)
                references.append(reference)
    return references


def test_compatibility_workflow_covers_platform_and_dependency_profiles():
    workflow = _workflow("compatibility.yml")
    triggers = workflow["on"]
    jobs = workflow["jobs"]
    assert isinstance(triggers, dict)
    assert set(triggers) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert isinstance(jobs, dict)
    assert set(jobs) == {"portable-platform", "dependency-resolution"}

    platform = jobs["portable-platform"]
    resolution = jobs["dependency-resolution"]
    assert isinstance(platform, dict)
    assert isinstance(resolution, dict)
    assert platform["timeout-minutes"] == "15"
    assert resolution["timeout-minutes"] == "30"
    assert platform["strategy"]["matrix"]["runner"] == [
        "ubuntu-24.04",
        "macos-14",
        "windows-2022",
    ]
    assert resolution["strategy"]["matrix"]["resolution"] == [
        "lowest-direct",
        "highest",
    ]
    resolution_commands = "\n".join(
        str(step.get("run", "")) for step in resolution["steps"] if isinstance(step, dict)
    )
    assert "git archive HEAD" in resolution_commands
    assert 'rm "$RESOLUTION_PROJECT/uv.lock"' in resolution_commands
    assert 'pytest -q "$GITHUB_WORKSPACE/tests"' in resolution_commands


def test_declared_dependency_floors_match_the_verified_python_312_profiles():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["dependencies"] == [
        "pydantic>=2.10,<3",
        "PyYAML>=6.0.1,<7",
        "typer>=0.26,<1",
    ]
    assert "hypothesis>=6.88.2" in project["optional-dependencies"]["dev"]
    assert "huggingface-hub>=1.3,<2" in project["optional-dependencies"]["neural"]
    assert "transformers>=5,<6" in project["optional-dependencies"]["neural"]
    assert "safetensors>=0.4.3,<1" in project["optional-dependencies"]["neural"]
    assert "tokenizers>=0.22,<1" in project["optional-dependencies"]["neural"]
    assert "torch>=2.4,<3" in project["optional-dependencies"]["neural"]
    assert "torch>=2.4,<3" in project["optional-dependencies"]["encoder-training"]
    assert metadata["tool"]["uv"]["required-environments"] == [
        "sys_platform == 'linux' and platform_machine == 'x86_64'",
    ]


def test_scheduled_workflow_verifies_installed_examples_and_checked_in_artifacts():
    workflow = _workflow("scheduled-verification.yml")
    triggers = workflow["on"]
    jobs = workflow["jobs"]
    assert isinstance(triggers, dict)
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "17 9 * * 1"}]
    assert workflow["permissions"] == {"contents": "read"}
    assert isinstance(jobs, dict)
    assert set(jobs) == {"installed-artifact-verification"}
    job = jobs["installed-artifact-verification"]
    assert isinstance(job, dict)
    assert job["timeout-minutes"] == "30"
    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict)
    )
    assert "scripts/ci/distribution_smoke.py" in commands
    assert "tests/integration/test_executable_examples.py" in commands
    assert "tests/unit/test_benchmark_evidence.py" in commands
    assert "tests/unit/test_reference_release.py" in commands
    assert "tests/unit/test_release.py" in commands


def test_all_compatibility_and_scheduled_actions_are_commit_pinned():
    references = [
        *_workflow_action_references(_workflow("compatibility.yml")),
        *_workflow_action_references(_workflow("scheduled-verification.yml")),
    ]
    assert references
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference) for reference in references)

from pathlib import Path

import pytest

from ste_compiler.terminology import TerminologyRegistry, Vocabulary

ROOT = Path(__file__).parents[1]


@pytest.fixture
def vocab():
    return Vocabulary.load(ROOT / "data/demo_vocabulary.yaml")


@pytest.fixture
def terms():
    return TerminologyRegistry.load(ROOT / "data/demo_terminology.yaml")

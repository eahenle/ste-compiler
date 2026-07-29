from pathlib import Path

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import factory
from ste_compiler.realizer.config import (
    DecoderOnlyLoRARealizerConfigV1,
    DeterministicRealizerConfigV1,
    EncoderDecoderRealizerConfigV1,
    realizer_config_sha256,
)
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.realizer.decoder_protocol import DECODER_PROMPT_PROFILE
from ste_compiler.realizer.deterministic import DeterministicRealizer
from ste_compiler.training.config import ArtifactIdentityV1

ROOT = Path(__file__).parents[2]
MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"
ADAPTER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


def _identity(repo_id: str, revision: str = MODEL_REVISION) -> ArtifactIdentityV1:
    return ArtifactIdentityV1(repo_id=repo_id, revision=revision)


def _plan_generator(document, vocab, terms, *, model_id, **revisions):
    expected = DeterministicRealizer().realize(document, vocab, terms)
    plan = SymbolicLexicalizer(vocab, terms).symbolize(expected.text)

    class Generator:
        def __init__(self):
            self.model_id = model_id
            for name, value in revisions.items():
                setattr(self, name, value)

        def generate_symbols(self, serialized_ir, allowed_symbols):
            del serialized_ir
            assert allowed_symbols == frozenset(plan.split())
            return plan

    return Generator()


def test_deterministic_factory_adds_config_identity_without_changing_default_metadata(
    vocab,
    terms,
):
    config = DeterministicRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="deterministic",
    )
    document = load_document(ROOT / "data/examples/negative.yaml")
    baseline = DeterministicRealizer().realize(document, vocab, terms)

    result = factory.build_realizer(config).realize(document, vocab, terms)

    assert result.text == baseline.text
    assert result.mappings == baseline.mappings
    assert "artifact_mode" not in baseline.metadata
    assert "realizer_config_sha256" not in baseline.metadata
    assert result.metadata == {
        **baseline.metadata,
        "realizer_config_sha256": realizer_config_sha256(config),
        "artifact_mode": "offline-cache-only",
    }


def test_encoder_decoder_factory_maps_fields_and_forces_cache_only(
    monkeypatch,
    vocab,
    terms,
):
    document = load_document(ROOT / "data/examples/negative.yaml")
    constructed = []
    fake_generator = _plan_generator(
        document,
        vocab,
        terms,
        model_id=f"org/encoder@{MODEL_REVISION}",
        model_revision=MODEL_REVISION,
    )

    def construct(runtime_config):
        constructed.append(runtime_config)
        return fake_generator

    monkeypatch.setattr(factory, "TransformersEncoderDecoderSymbolGenerator", construct)
    config = EncoderDecoderRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="encoder-decoder",
        checkpoint=_identity("org/encoder"),
        max_source_tokens=321,
        max_new_tokens=123,
        num_beams=2,
    )

    realizer = factory.build_realizer(config)

    assert len(constructed) == 1
    runtime_config = constructed[0]
    assert runtime_config.model_id == "org/encoder"
    assert runtime_config.revision == MODEL_REVISION
    assert runtime_config.max_source_tokens == 321
    assert runtime_config.max_new_tokens == 123
    assert runtime_config.num_beams == 2
    assert runtime_config.local_files_only is True

    result = realizer.realize(document, vocab, terms)
    assert result.metadata["artifact_mode"] == "offline-cache-only"
    assert result.metadata["realizer_config_sha256"] == realizer_config_sha256(config)
    assert result.metadata["model_revision"] == MODEL_REVISION


def test_decoder_factory_is_lazy_maps_fields_and_forces_cache_only(
    monkeypatch,
    vocab,
    terms,
):
    document = load_document(ROOT / "data/examples/negative.yaml")
    constructed = []
    fake_generator = _plan_generator(
        document,
        vocab,
        terms,
        model_id=(f"org/decoder@{MODEL_REVISION}+peft:org/decoder-adapter@{ADAPTER_REVISION}"),
        base_model_revision=MODEL_REVISION,
        adapter_revision=ADAPTER_REVISION,
    )

    def construct(runtime_config):
        constructed.append(runtime_config)
        return fake_generator

    monkeypatch.setattr(factory, "DecoderOnlyLoRASymbolGenerator", construct)
    config = DecoderOnlyLoRARealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="decoder-only-lora",
        base_model=_identity("org/decoder"),
        adapter=_identity("org/decoder-adapter", ADAPTER_REVISION),
        prompt_profile=DECODER_PROMPT_PROFILE,
        max_new_tokens=234,
        max_symbols=56,
    )

    realizer = factory.build_realizer(config)

    assert not constructed
    result = realizer.realize(document, vocab, terms)
    assert len(constructed) == 1
    runtime_config = constructed[0]
    assert runtime_config.base_model_id == "org/decoder"
    assert runtime_config.base_model_revision == MODEL_REVISION
    assert runtime_config.adapter_id == "org/decoder-adapter"
    assert runtime_config.adapter_revision == ADAPTER_REVISION
    assert runtime_config.max_new_tokens == 234
    assert runtime_config.max_symbols == 56
    assert runtime_config.local_files_only is True
    assert result.metadata["artifact_mode"] == "offline-cache-only"
    assert result.metadata["realizer_config_sha256"] == realizer_config_sha256(config)
    assert result.metadata["base_model_revision"] == MODEL_REVISION
    assert result.metadata["adapter_revision"] == ADAPTER_REVISION

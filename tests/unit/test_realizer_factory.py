from pathlib import Path

import pytest

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import factory
from ste_compiler.realizer.config import (
    DecoderOnlyLoRALocalBundleRealizerConfigV1,
    DecoderOnlyLoRARealizerConfigV1,
    DeterministicRealizerConfigV1,
    EncoderDecoderLocalBundleRealizerConfigV1,
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


def test_deterministic_factory_rejects_local_artifact_locators(tmp_path):
    config = DeterministicRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="deterministic",
    )

    with pytest.raises(ValueError, match="does not accept local artifact locators"):
        factory.build_realizer(config, artifact_bundle=tmp_path / "bundle")
    with pytest.raises(ValueError, match="does not accept local artifact locators"):
        factory.build_realizer(config, model_snapshot=tmp_path / "snapshot")

    factory.build_realizer(config).prepare()


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


def test_hub_encoder_decoder_factory_rejects_local_locators(tmp_path):
    config = EncoderDecoderRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="encoder-decoder",
        checkpoint=_identity("org/encoder"),
    )

    with pytest.raises(ValueError, match="does not accept local locators"):
        factory.build_realizer(config, artifact_bundle=tmp_path / "bundle")
    with pytest.raises(ValueError, match="does not accept local locators"):
        factory.build_realizer(config, model_snapshot=tmp_path / "snapshot")


def test_encoder_local_bundle_factory_requires_and_maps_untrusted_locator(
    monkeypatch,
    tmp_path,
    vocab,
    terms,
):
    document = load_document(ROOT / "data/examples/negative.yaml")
    constructed = []
    digest = "a" * 64
    fake_generator = _plan_generator(
        document,
        vocab,
        terms,
        model_id=f"ste-artifact-bundle:sha256:{digest}",
        artifact_manifest_sha256=digest,
        run_manifest_sha256="b" * 64,
        artifact_intended_use="mechanics-smoke",
    )
    prepared = []
    fake_generator.prepare = lambda: prepared.append(True)

    def construct(runtime_config):
        constructed.append(runtime_config)
        return fake_generator

    monkeypatch.setattr(factory, "TransformersEncoderDecoderSymbolGenerator", construct)
    config = EncoderDecoderLocalBundleRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="encoder-decoder-local-bundle",
        artifact_manifest_sha256=digest,
        intended_use="mechanics-smoke",
        max_source_tokens=321,
        max_new_tokens=123,
        num_beams=2,
    )
    bundle = tmp_path / "untrusted-bundle"

    with pytest.raises(ValueError, match="requires --artifact-bundle"):
        factory.build_realizer(config)
    with pytest.raises(ValueError, match="does not accept --model-snapshot"):
        factory.build_realizer(
            config,
            artifact_bundle=bundle,
            model_snapshot=tmp_path / "snapshot",
        )

    realizer = factory.build_realizer(config, artifact_bundle=bundle)
    realizer.prepare()
    result = realizer.realize(
        document,
        vocab,
        terms,
    )

    runtime_config = constructed[0]
    assert runtime_config.artifact_bundle == bundle
    assert runtime_config.artifact_manifest_sha256 == digest
    assert runtime_config.intended_use == "mechanics-smoke"
    assert runtime_config.max_source_tokens == 321
    assert runtime_config.max_new_tokens == 123
    assert runtime_config.num_beams == 2
    assert prepared == [True]
    assert result.metadata["artifact_mode"] == "content-addressed-local-bundle"
    assert result.metadata["artifact_manifest_sha256"] == digest
    assert result.metadata["run_manifest_sha256"] == "b" * 64
    assert result.metadata["artifact_intended_use"] == "mechanics-smoke"


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


def test_hub_decoder_factory_rejects_local_locators_and_protocol_mismatch(tmp_path):
    config = DecoderOnlyLoRARealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="decoder-only-lora",
        base_model=_identity("org/decoder"),
        adapter=_identity("org/decoder-adapter", ADAPTER_REVISION),
        prompt_profile=DECODER_PROMPT_PROFILE,
    )

    with pytest.raises(ValueError, match="does not accept local locators"):
        factory.build_realizer(config, artifact_bundle=tmp_path / "bundle")
    with pytest.raises(ValueError, match="does not accept local locators"):
        factory.build_realizer(config, model_snapshot=tmp_path / "snapshot")

    mismatched = config.model_copy(update={"prompt_profile": "other-protocol"})
    with pytest.raises(ValueError, match="prompt profile does not match"):
        factory.build_realizer(mismatched)


def test_decoder_local_bundle_factory_requires_both_locators_and_loads_lazily(
    monkeypatch,
    tmp_path,
    vocab,
    terms,
):
    document = load_document(ROOT / "data/examples/negative.yaml")
    artifact_digest = "a" * 64
    snapshot_digest = "b" * 64
    run_digest = "c" * 64
    constructed = []
    fake_generator = _plan_generator(
        document,
        vocab,
        terms,
        model_id=(
            f"org/decoder@{MODEL_REVISION}"
            f"+peft-bundle:sha256:{artifact_digest}"
            f"+model-snapshot:sha256:{snapshot_digest}"
        ),
        base_model_revision=MODEL_REVISION,
        artifact_manifest_sha256=artifact_digest,
        run_manifest_sha256=run_digest,
        model_snapshot_manifest_sha256=snapshot_digest,
        artifact_intended_use="mechanics-smoke",
    )

    def load(runtime_config):
        constructed.append(runtime_config)
        return fake_generator

    monkeypatch.setattr(factory, "load_local_decoder_lora_generator", load)
    config = DecoderOnlyLoRALocalBundleRealizerConfigV1(
        schema_version="ste-realizer-config-v1",
        architecture="decoder-only-lora-local-bundle",
        artifact_manifest_sha256=artifact_digest,
        model_snapshot_manifest_sha256=snapshot_digest,
        base_model=_identity("org/decoder"),
        tokenizer=_identity("org/decoder"),
        intended_use="mechanics-smoke",
        prompt_profile=DECODER_PROMPT_PROFILE,
        max_new_tokens=234,
        max_symbols=56,
    )
    bundle = tmp_path / "adapter-bundle"
    snapshot = tmp_path / "base-snapshot"

    with pytest.raises(ValueError, match="requires --artifact-bundle and --model-snapshot"):
        factory.build_realizer(config, artifact_bundle=bundle)

    realizer = factory.build_realizer(
        config,
        artifact_bundle=bundle,
        model_snapshot=snapshot,
    )

    assert not constructed
    realizer.prepare()
    assert len(constructed) == 1
    result = realizer.realize(document, vocab, terms)
    assert len(constructed) == 1
    runtime_config = constructed[0]
    assert runtime_config.artifact_bundle == bundle
    assert runtime_config.model_snapshot == snapshot
    assert runtime_config.artifact_manifest_sha256 == artifact_digest
    assert runtime_config.model_snapshot_manifest_sha256 == snapshot_digest
    assert runtime_config.base_model == config.base_model
    assert runtime_config.tokenizer == config.tokenizer
    assert runtime_config.max_new_tokens == 234
    assert runtime_config.max_symbols == 56
    assert result.metadata["artifact_mode"] == "content-addressed-local-bundle"
    assert result.metadata["artifact_manifest_sha256"] == artifact_digest
    assert result.metadata["run_manifest_sha256"] == run_digest
    assert result.metadata["model_snapshot_manifest_sha256"] == snapshot_digest
    assert result.metadata["artifact_intended_use"] == "mechanics-smoke"

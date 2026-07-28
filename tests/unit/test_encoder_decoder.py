from pathlib import Path
from types import SimpleNamespace

import pytest

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import (
    DeterministicRealizer,
    EncoderDecoderConfig,
    InvalidSymbolGeneration,
    NeuralRealizer,
    TransformersEncoderDecoderSymbolGenerator,
)
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.realizer.encoder_decoder import (
    _SymbolTokenConstraint,
)
from ste_compiler.validators.semantic import SemanticValidator

ROOT = Path(__file__).parents[2]
MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = None

    def __call__(self, text, *, return_tensors, truncation, max_length):
        assert return_tensors == "pt"
        return {
            "input_ids": [self.encode(text, add_special_tokens=False)[:max_length]],
            "attention_mask": [[1] * min(len(text), max_length)],
        }

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) + 10 for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id - 10) for token_id in token_ids)


class CheckingModel:
    config = SimpleNamespace(decoder_start_token_id=0)

    def __init__(self, tokenizer, output):
        self.tokenizer = tokenizer
        self.output = output
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        constraint = kwargs["prefix_allowed_tokens_fn"]
        generated = [self.config.decoder_start_token_id]
        planned = self.tokenizer.encode(self.output, add_special_tokens=False)
        for token_id in planned:
            assert token_id in constraint(0, generated)
            generated.append(token_id)
        assert self.tokenizer.eos_token_id in constraint(0, generated)
        return [[*generated, self.tokenizer.eos_token_id]]


def test_encoder_decoder_generation_is_lazy_pinned_and_constrained():
    tokenizer = CharacterTokenizer()
    model = CheckingModel(tokenizer, "WORD_do WORD_not PERIOD")
    calls = []

    def load(config):
        calls.append(config)
        return tokenizer, model

    config = EncoderDecoderConfig(
        model_id="organization/small-seq2seq",
        revision=MODEL_REVISION,
        max_source_tokens=64,
        max_new_tokens=40,
    )
    generator = TransformersEncoderDecoderSymbolGenerator(config, component_loader=load)
    assert not calls

    plan = generator.generate_symbols(
        '{"id":"negative"}',
        frozenset({"WORD_do", "WORD_not", "PERIOD"}),
    )

    assert plan == "WORD_do WORD_not PERIOD"
    assert calls == [config]
    assert generator.model_id == f"organization/small-seq2seq@{MODEL_REVISION}"
    assert generator.model_revision == MODEL_REVISION
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["return_dict_in_generate"] is False
    assert model.kwargs["input_ids"]


def test_constraint_allows_eos_only_at_symbol_boundaries():
    tokenizer = CharacterTokenizer()
    constraint = _SymbolTokenConstraint(
        tokenizer,
        frozenset({"WORD_do", "PERIOD"}),
        decoder_start_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
    )
    partial = tokenizer.encode("WORD_", add_special_tokens=False)
    complete = tokenizer.encode("WORD_do", add_special_tokens=False)

    assert tokenizer.eos_token_id not in constraint(0, [0])
    assert tokenizer.eos_token_id not in constraint(0, [0, *partial])
    assert tokenizer.eos_token_id in constraint(0, [0, *complete])
    space = tokenizer.encode(" ", add_special_tokens=False)[0]
    assert space in constraint(0, [0, *complete])


def test_percent_encoded_symbols_remain_opaque_to_adapter():
    tokenizer = CharacterTokenizer()
    encoded_unit = "UNIT_N%20m"
    model = CheckingModel(tokenizer, f"NUMBER_20 {encoded_unit} PERIOD")
    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, model),
    )

    assert (
        generator.generate_symbols(
            "{}",
            frozenset({"NUMBER_20", encoded_unit, "PERIOD"}),
        )
        == f"NUMBER_20 {encoded_unit} PERIOD"
    )


def test_encoder_decoder_revision_is_auditable_through_neural_realizer(vocab, terms):
    document = load_document(ROOT / "data/examples/negative.yaml")
    expected = DeterministicRealizer().realize(document, vocab, terms)
    plan = SymbolicLexicalizer(vocab, terms).symbolize(expected.text)
    tokenizer = CharacterTokenizer()
    model = CheckingModel(tokenizer, plan)
    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(
            model_id="organization/small-seq2seq",
            revision=MODEL_REVISION,
        ),
        component_loader=lambda config: (tokenizer, model),
    )

    result = NeuralRealizer(generator).realize(document, vocab, terms)

    assert result.text == expected.text
    assert result.metadata["model_id"] == f"organization/small-seq2seq@{MODEL_REVISION}"
    assert result.metadata["model_revision"] == MODEL_REVISION
    assert not SemanticValidator().validate(document, result)


def test_adapter_rejects_output_when_model_ignores_constraint():
    tokenizer = CharacterTokenizer()

    class IgnoringModel:
        config = SimpleNamespace(decoder_start_token_id=0)

        def generate(self, **kwargs):
            del kwargs
            output = tokenizer.encode("WORD_invented PERIOD", add_special_tokens=False)
            return [[0, *output, tokenizer.eos_token_id]]

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, IgnoringModel()),
    )
    with pytest.raises(InvalidSymbolGeneration, match="outside the document allowlist"):
        generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"}))


def test_adapter_requires_explicit_termination():
    tokenizer = CharacterTokenizer()

    class UnterminatedModel:
        config = SimpleNamespace(decoder_start_token_id=0)

        def generate(self, **kwargs):
            del kwargs
            output = tokenizer.encode("WORD_do PERIOD", add_special_tokens=False)
            return [[0, *output]]

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, UnterminatedModel()),
    )
    with pytest.raises(InvalidSymbolGeneration, match="terminate with EOS"):
        generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"}))


def test_constraint_rejects_a_normalizing_tokenizer():
    class NormalizingTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **kwargs):
            return super().decode(token_ids, **kwargs).strip()

    with pytest.raises(ValueError, match="losslessly encode"):
        _SymbolTokenConstraint(
            NormalizingTokenizer(),
            frozenset({"WORD_do", "PERIOD"}),
            decoder_start_token_id=0,
            eos_token_id=CharacterTokenizer.eos_token_id,
        )


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "main",
        "MASTER",
        "latest",
        "HEAD",
        "develop",
        "stable",
        "refs/heads/main",
        "refs/tags/v1",
        "release-2026",
        "v1.0.0",
        "0123456789abcdef",
        "0123456789ABCDEF0123456789ABCDEF01234567",
        " refs/tags/v1 ",
    ],
)
def test_encoder_decoder_configuration_rejects_unpinned_revisions(revision):
    with pytest.raises(ValueError, match="revision"):
        EncoderDecoderConfig(model_id="test/model", revision=revision)


def test_adapter_rejects_non_padding_tokens_after_eos():
    tokenizer = CharacterTokenizer()

    class SuffixModel:
        config = SimpleNamespace(decoder_start_token_id=0)

        def generate(self, **kwargs):
            del kwargs
            output = tokenizer.encode("WORD_do PERIOD", add_special_tokens=False)
            suffix = tokenizer.encode(" WORD_invented", add_special_tokens=False)
            return [[0, *output, tokenizer.eos_token_id, *suffix]]

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, SuffixModel()),
    )

    with pytest.raises(InvalidSymbolGeneration, match="non-padding tokens after EOS"):
        generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"}))


def test_adapter_accepts_padding_after_eos():
    tokenizer = CharacterTokenizer()

    class PaddedModel:
        config = SimpleNamespace(decoder_start_token_id=0)

        def generate(self, **kwargs):
            del kwargs
            output = tokenizer.encode("WORD_do PERIOD", add_special_tokens=False)
            return [[0, *output, tokenizer.eos_token_id, tokenizer.pad_token_id]]

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, PaddedModel()),
    )

    assert generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"})) == "WORD_do PERIOD"


def test_adapter_handles_structured_transformers_output_defensively():
    tokenizer = CharacterTokenizer()

    class StructuredModel:
        config = SimpleNamespace(decoder_start_token_id=0)

        def generate(self, **kwargs):
            assert kwargs["return_dict_in_generate"] is False
            output = tokenizer.encode("WORD_do PERIOD", add_special_tokens=False)
            return SimpleNamespace(
                sequences=[[0, *output, tokenizer.eos_token_id]],
            )

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, StructuredModel()),
    )

    assert generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"})) == "WORD_do PERIOD"


@pytest.mark.parametrize(
    "generated",
    [
        [],
        [0, 1, 2],
        [[[0, 1, 2]]],
        [["0", "1", "2"]],
        [[True, 1, 2]],
    ],
)
def test_adapter_rejects_missing_or_non_integer_sequence_shapes(generated):
    with pytest.raises(InvalidSymbolGeneration, match="generated sequence|no generated sequence"):
        TransformersEncoderDecoderSymbolGenerator._first_sequence(generated)

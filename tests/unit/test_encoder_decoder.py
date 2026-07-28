from pathlib import Path
from types import SimpleNamespace

import pytest

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import (
    DeterministicRealizer,
    EncoderDecoderConfig,
    EncoderDecoderError,
    InvalidSymbolGeneration,
    NeuralRealizer,
    TransformersEncoderDecoderSymbolGenerator,
    encoder_decoder,
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
    assert "forced_decoder_ids" not in model.kwargs
    assert model.kwargs["input_ids"]


def test_encoder_decoder_loader_requires_one_safe_pinned_snapshot(monkeypatch, tmp_path):
    calls = []
    tokenizer = object()
    model = object()
    snapshot = tmp_path / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").touch()

    class Factory:
        def __init__(self, name, result):
            self.name = name
            self.result = result

        def from_pretrained(self, *args, **kwargs):
            calls.append((self.name, args, kwargs))
            if self.name == "snapshot":
                local_collision = tmp_path / "organization" / "small-seq2seq"
                local_collision.mkdir(parents=True)
            return self.result

    modules = {
        "transformers": SimpleNamespace(
            AutoTokenizer=Factory("tokenizer", tokenizer),
            AutoModelForSeq2SeqLM=Factory("model", model),
        ),
        "huggingface_hub": SimpleNamespace(
            snapshot_download=Factory("snapshot", str(snapshot)).from_pretrained
        ),
    }
    monkeypatch.setattr(
        encoder_decoder,
        "import_module",
        modules.__getitem__,
    )
    monkeypatch.chdir(tmp_path)
    config = EncoderDecoderConfig(
        model_id="organization/small-seq2seq",
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    assert TransformersEncoderDecoderSymbolGenerator._load_transformers_components(config) == (
        tokenizer,
        model,
    )
    assert calls == [
        (
            "snapshot",
            (),
            {
                "repo_id": "organization/small-seq2seq",
                "revision": MODEL_REVISION,
                "local_files_only": True,
                "allow_patterns": [
                    "*.json",
                    "*.merges",
                    "*.model",
                    "*.safetensors",
                    "*.spm",
                    "*.tiktoken",
                    "*.txt",
                    "*.vocab",
                ],
            },
        ),
        (
            "tokenizer",
            (str(snapshot),),
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        ),
        (
            "model",
            (str(snapshot),),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("snapshot_name", "artifacts", "message"),
    [
        ("different-revision", {"config.json", "model.safetensors"}, "configured commit"),
        (MODEL_REVISION, {"model.safetensors"}, "config.json"),
        (MODEL_REVISION, {"config.json"}, "safetensors weights"),
    ],
)
def test_encoder_decoder_rejects_unsafe_model_snapshot(
    tmp_path,
    snapshot_name,
    artifacts,
    message,
):
    snapshot = tmp_path / "snapshots" / snapshot_name
    snapshot.mkdir(parents=True)
    for artifact in artifacts:
        (snapshot / artifact).touch()
    hub = SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot))
    config = EncoderDecoderConfig(
        model_id="organization/small-seq2seq",
        revision=MODEL_REVISION,
    )

    with pytest.raises(EncoderDecoderError, match=message):
        TransformersEncoderDecoderSymbolGenerator._resolve_safe_model_snapshot(config, hub)


def test_encoder_decoder_wraps_snapshot_resolution_failure():
    def fail(**kwargs):
        raise OSError("cache failure")

    hub = SimpleNamespace(snapshot_download=fail)
    config = EncoderDecoderConfig(
        model_id="organization/small-seq2seq",
        revision=MODEL_REVISION,
    )

    with pytest.raises(EncoderDecoderError, match="could not be resolved") as caught:
        TransformersEncoderDecoderSymbolGenerator._resolve_safe_model_snapshot(config, hub)

    assert isinstance(caught.value.__cause__, OSError)


def test_encoder_decoder_overrides_inherited_generation_strategy():
    tokenizer = CharacterTokenizer()
    model = CheckingModel(tokenizer, "PERIOD")
    model.generation_config = SimpleNamespace(
        do_sample=True,
        num_beams=8,
        num_beam_groups=4,
        num_return_sequences=4,
        return_dict_in_generate=True,
        penalty_alpha=0.6,
        dola_layers="high",
        constraints=[object()],
        force_words_ids=[[123]],
        forced_decoder_ids=[[1, 123]],
        forced_bos_token_id=123,
        forced_eos_token_id=124,
        suppress_tokens=[125],
        begin_suppress_tokens=[126],
        bad_words_ids=[[127]],
        sequence_bias={(128,): 10.0},
        no_repeat_ngram_size=4,
        encoder_no_repeat_ngram_size=4,
        repetition_penalty=2.0,
        encoder_repetition_penalty=2.0,
        diversity_penalty=1.0,
        length_penalty=2.0,
        early_stopping=True,
        exponential_decay_length_penalty=(4, 1.5),
        renormalize_logits=True,
        remove_invalid_values=True,
        assistant_model=object(),
        prompt_lookup_num_tokens=8,
        min_length=256,
        min_new_tokens=128,
        max_time=0.001,
        stop_strings=["stop"],
        token_healing=True,
        watermarking_config=object(),
        guidance_scale=2.0,
    )
    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(
            model_id="organization/small-seq2seq",
            revision=MODEL_REVISION,
            num_beams=2,
        ),
        component_loader=lambda config: (tokenizer, model),
    )

    assert generator.generate_symbols("{}", frozenset({"PERIOD"})) == "PERIOD"
    assert {
        name: model.kwargs[name]
        for name in (
            "do_sample",
            "num_beams",
            "num_beam_groups",
            "num_return_sequences",
            "decoder_start_token_id",
            "return_dict_in_generate",
            "penalty_alpha",
            "dola_layers",
            "constraints",
            "force_words_ids",
            "forced_decoder_ids",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
            "bad_words_ids",
            "sequence_bias",
            "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size",
            "repetition_penalty",
            "encoder_repetition_penalty",
            "diversity_penalty",
            "length_penalty",
            "early_stopping",
            "exponential_decay_length_penalty",
            "renormalize_logits",
            "remove_invalid_values",
            "assistant_model",
            "prompt_lookup_num_tokens",
            "min_length",
            "min_new_tokens",
            "max_time",
            "stop_strings",
            "token_healing",
            "watermarking_config",
            "guidance_scale",
        )
    } == {
        "do_sample": False,
        "num_beams": 2,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "decoder_start_token_id": 0,
        "return_dict_in_generate": False,
        "penalty_alpha": None,
        "dola_layers": None,
        "constraints": None,
        "force_words_ids": None,
        "forced_decoder_ids": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "bad_words_ids": None,
        "sequence_bias": None,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "diversity_penalty": 0.0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "exponential_decay_length_penalty": None,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "assistant_model": None,
        "prompt_lookup_num_tokens": None,
        "min_length": 0,
        "min_new_tokens": 0,
        "max_time": None,
        "stop_strings": None,
        "token_healing": False,
        "watermarking_config": None,
        "guidance_scale": None,
    }


def test_encoder_decoder_configuration_accepts_hub_repository_id():
    config = EncoderDecoderConfig(model_id="organization/model", revision=MODEL_REVISION)

    assert config.model_id == "organization/model"


def test_encoder_decoder_configuration_rejects_existing_local_paths(tmp_path, monkeypatch):
    local_directory = tmp_path / "local-model"
    local_directory.mkdir()
    local_file = tmp_path / "model.bin"
    local_file.write_bytes(b"mutable")
    monkeypatch.chdir(tmp_path)

    for model_id in (
        str(local_directory),
        str(local_file),
        local_directory.name,
        local_file.name,
    ):
        with pytest.raises(EncoderDecoderError, match="local filesystem model paths"):
            EncoderDecoderConfig(model_id=model_id, revision=MODEL_REVISION)


@pytest.mark.parametrize(
    "model_id",
    [
        "/models/local",
        "./models/local",
        "../models/local",
        "~/models/local",
        "file:///models/local",
        r"C:\models\local",
        r"\\server\share\local",
        "models/nested/local",
    ],
)
def test_encoder_decoder_configuration_rejects_local_path_forms(model_id):
    with pytest.raises(EncoderDecoderError, match="Hugging Face Hub repository ID"):
        EncoderDecoderConfig(model_id=model_id, revision=MODEL_REVISION)


def test_local_model_created_after_configuration_is_rejected_before_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = EncoderDecoderConfig(model_id="organization/model", revision=MODEL_REVISION)
    calls = []

    def load(config):
        calls.append(config)
        raise AssertionError("loader must not run for a local model path")

    generator = TransformersEncoderDecoderSymbolGenerator(config, component_loader=load)
    (tmp_path / "organization/model").mkdir(parents=True)

    with pytest.raises(EncoderDecoderError, match="local filesystem model paths"):
        generator.generate_symbols("{}", frozenset({"PERIOD"}))
    assert not calls


def test_local_model_created_before_generator_is_rejected_before_identity_claim(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = EncoderDecoderConfig(model_id="organization/model", revision=MODEL_REVISION)
    (tmp_path / "organization/model").mkdir(parents=True)
    calls = []

    def load(config):
        calls.append(config)
        raise AssertionError("loader must not run for a local model path")

    with pytest.raises(EncoderDecoderError, match="local filesystem model paths"):
        TransformersEncoderDecoderSymbolGenerator(config, component_loader=load)
    assert not calls


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
    with pytest.raises(InvalidSymbolGeneration, match="outside the symbolic grammar"):
        generator.generate_symbols("{}", frozenset({"WORD_do", "PERIOD"}))


@pytest.mark.parametrize("special_token_id", [0, 2], ids=["PAD", "BOS"])
def test_adapter_rejects_special_tokens_before_eos_even_when_decode_hides_them(
    special_token_id,
):
    plan = "WORD_do PERIOD"

    class SpecialTokenHidingTokenizer(CharacterTokenizer):
        bos_token_id = 2

        def decode(self, token_ids, **kwargs):
            hidden = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            return super().decode(
                [token_id for token_id in token_ids if token_id not in hidden],
                **kwargs,
            )

    tokenizer = SpecialTokenHidingTokenizer()
    encoded_plan = tokenizer.encode(plan, add_special_tokens=False)
    injected = [*encoded_plan[:4], special_token_id, *encoded_plan[4:]]
    assert (
        tokenizer.decode(
            injected,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        == plan
    )

    class InjectingModel:
        config = SimpleNamespace(decoder_start_token_id=tokenizer.pad_token_id)

        def generate(self, **kwargs):
            del kwargs
            return [[tokenizer.pad_token_id, *injected, tokenizer.eos_token_id]]

    generator = TransformersEncoderDecoderSymbolGenerator(
        EncoderDecoderConfig(model_id="test/model", revision=MODEL_REVISION),
        component_loader=lambda config: (tokenizer, InjectingModel()),
    )

    with pytest.raises(
        InvalidSymbolGeneration,
        match="token path outside the symbolic grammar before EOS",
    ):
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


def test_adapter_rejects_multiple_sequences_before_selecting_first():
    valid_first = [0, 1, 2]
    malformed_second = ["not", "token", "ids"]

    class TensorLikeOutput:
        def tolist(self):
            return [valid_first, malformed_second]

    for generated in (
        [valid_first, malformed_second],
        SimpleNamespace(sequences=[valid_first, malformed_second]),
        TensorLikeOutput(),
        SimpleNamespace(sequences=TensorLikeOutput()),
    ):
        with pytest.raises(InvalidSymbolGeneration, match="multiple generated sequences"):
            TransformersEncoderDecoderSymbolGenerator._first_sequence(generated)


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

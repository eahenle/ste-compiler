from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import (
    DecoderOnlyLoRAConfig,
    DecoderOnlyLoRAError,
    DecoderOnlyLoRASymbolGenerator,
    DeterministicRealizer,
    NeuralRealizer,
    decoder_lora,
)
from ste_compiler.realizer.constrained import SymbolicLexicalizer
from ste_compiler.validators.semantic import SemanticValidator

BASE_REVISION = "0123456789abcdef0123456789abcdef01234567"
ADAPTER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = None

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids if token_id)

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens
        return {"input_ids": [[ord(character) for character in text]]}


class ConstrainedFakeModel:
    device = None

    def __init__(self, plan: str, *, terminate: bool = True):
        self.plan = plan
        self.terminate = terminate
        self.generate_arguments = None

    def generate(self, **kwargs):
        self.generate_arguments = kwargs
        prefix = list(kwargs["input_ids"][0])
        callback: Callable[[int, list[int]], list[int]] = kwargs["prefix_allowed_tokens_fn"]
        generated: list[int] = []
        for token_id in [ord(character) for character in self.plan]:
            assert token_id in callback(0, prefix + generated)
            generated.append(token_id)
        if self.terminate:
            assert 0 in callback(0, prefix + generated)
            generated.append(0)
        return [prefix + generated]


class UnconstrainedFakeModel(ConstrainedFakeModel):
    def generate(self, **kwargs):
        prefix = list(kwargs["input_ids"][0])
        generated = [ord(character) for character in self.plan]
        return [prefix + generated + [0]]


class MultipleSequenceFakeModel(ConstrainedFakeModel):
    def generate(self, **kwargs):
        prefix = list(kwargs["input_ids"][0])
        generated = [ord(character) for character in self.plan]
        sequence = prefix + generated + [0]
        return [sequence, sequence]


def _config(**updates):
    values = {
        "base_model_id": "org/compact-decoder",
        "base_model_revision": BASE_REVISION,
        "adapter_id": "org/ste-symbol-lora",
        "adapter_revision": ADAPTER_REVISION,
        "max_new_tokens": 1000,
    }
    values.update(updates)
    return DecoderOnlyLoRAConfig(**values)


def test_decoder_lora_generates_constrained_symbols_and_records_revisions(vocab, terms):
    document = load_document(Path("data/examples/negative.yaml"))
    expected = DeterministicRealizer().realize(document, vocab, terms)
    plan = SymbolicLexicalizer(vocab, terms).symbolize(expected.text)
    model = ConstrainedFakeModel(plan)
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=model,
    )

    result = NeuralRealizer(generator).realize(document, vocab, terms)

    assert result.text == expected.text
    assert result.metadata["model_id"] == (
        f"org/compact-decoder@{BASE_REVISION}+peft:org/ste-symbol-lora@{ADAPTER_REVISION}"
    )
    assert result.metadata["base_model_revision"] == BASE_REVISION
    assert result.metadata["adapter_revision"] == ADAPTER_REVISION
    assert generator.base_model_revision == BASE_REVISION
    assert generator.adapter_revision == ADAPTER_REVISION
    assert not SemanticValidator().validate(document, result)
    assert model.generate_arguments["do_sample"] is False
    assert model.generate_arguments["num_beams"] == 1
    assert model.generate_arguments["num_return_sequences"] == 1
    assert model.generate_arguments["return_dict_in_generate"] is False
    assert model.generate_arguments["eos_token_id"] == 0


def test_decoder_lora_grammar_rejects_an_out_of_plan_symbol():
    model = ConstrainedFakeModel("WORD_close")
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=model,
    )

    with pytest.raises(AssertionError):
        generator.generate_symbols(
            '{"id":"negative"}',
            frozenset({"WORD_do", "WORD_not", "WORD_open"}),
        )


def test_decoder_lora_postcondition_rejects_a_model_that_ignores_the_grammar():
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=UnconstrainedFakeModel("WORD_close"),
    )

    with pytest.raises(DecoderOnlyLoRAError, match="escaped the symbol allowlist"):
        generator.generate_symbols(
            '{"id":"negative"}',
            frozenset({"WORD_do", "WORD_not", "WORD_open"}),
        )


def test_decoder_lora_rejects_multiple_returned_sequences():
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=MultipleSequenceFakeModel("WORD_open"),
    )

    with pytest.raises(DecoderOnlyLoRAError, match="multiple token sequences"):
        generator.generate_symbols('{"id":"test"}', frozenset({"WORD_open"}))


def test_decoder_lora_accepts_exactly_one_batch_of_integer_token_ids():
    assert decoder_lora._integer_sequence([[1, 2, 3]], batched=True) == [1, 2, 3]


@pytest.mark.parametrize(
    "output",
    [
        [],
        [1],
        [[1], [2]],
        [[1.5, 2.0]],
        [["1", "2"]],
        [[True, False]],
        [[[[1, 2]]]],
    ],
)
def test_decoder_lora_rejects_invalid_output_token_shapes_without_raw_errors(output):
    with pytest.raises(DecoderOnlyLoRAError):
        decoder_lora._integer_sequence(output, batched=True)


def test_decoder_lora_requires_eos_termination():
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=ConstrainedFakeModel("WORD_open", terminate=False),
    )

    with pytest.raises(DecoderOnlyLoRAError, match="terminate with EOS"):
        generator.generate_symbols('{"id":"test"}', frozenset({"WORD_open"}))


def test_decoder_lora_overrides_inherited_generation_strategy():
    model = ConstrainedFakeModel("WORD_open")
    model.generation_config = SimpleNamespace(
        do_sample=True,
        num_beams=8,
        num_beam_groups=4,
        num_return_sequences=4,
        return_dict_in_generate=True,
        penalty_alpha=0.6,
        top_k=4,
        dola_layers="high",
        constraints=[object()],
        force_words_ids=[[123]],
        assistant_model=object(),
        prompt_lookup_num_tokens=8,
        min_length=256,
        min_new_tokens=128,
    )
    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=CharacterTokenizer(),
        model=model,
    )

    assert (
        generator.generate_symbols(
            '{"id":"test"}',
            frozenset({"WORD_open"}),
        )
        == "WORD_open"
    )
    assert {
        name: model.generate_arguments[name]
        for name in (
            "do_sample",
            "num_beams",
            "num_beam_groups",
            "num_return_sequences",
            "return_dict_in_generate",
            "penalty_alpha",
            "dola_layers",
            "constraints",
            "force_words_ids",
            "assistant_model",
            "prompt_lookup_num_tokens",
            "min_length",
            "min_new_tokens",
        )
    } == {
        "do_sample": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "penalty_alpha": None,
        "dola_layers": None,
        "constraints": None,
        "force_words_ids": None,
        "assistant_model": None,
        "prompt_lookup_num_tokens": None,
        "min_length": 0,
        "min_new_tokens": 0,
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"base_model_revision": ""}, "base_model_revision"),
        ({"adapter_revision": ""}, "adapter_revision"),
        ({"base_model_revision": "main"}, "commit digest"),
        ({"adapter_revision": "refs/heads/release"}, "commit digest"),
        ({"base_model_revision": "release-2026"}, "commit digest"),
        ({"adapter_revision": "v1.0.0"}, "commit digest"),
        ({"base_model_revision": "0123456789abcdef"}, "commit digest"),
        (
            {"adapter_revision": "89ABCDEF0123456789ABCDEF0123456789ABCDEF"},
            "commit digest",
        ),
        ({"max_symbols": 0}, "max_symbols"),
    ],
)
def test_decoder_lora_config_rejects_unpinned_or_invalid_values(update, message):
    with pytest.raises(ValueError, match=message):
        _config(**update)


@pytest.mark.parametrize("field", ["base_model_id", "adapter_id"])
def test_decoder_lora_config_rejects_local_artifact_paths(tmp_path, field):
    local_artifact = tmp_path / field
    local_artifact.mkdir()

    with pytest.raises(ValueError, match=f"{field} must be a Hub repository ID"):
        _config(**{field: str(local_artifact)})


@pytest.mark.parametrize("field", ["base_model_id", "adapter_id"])
def test_decoder_lora_config_rejects_unresolved_relative_paths(field):
    with pytest.raises(ValueError, match=f"{field} must be a Hub repository ID"):
        _config(**{field: f"./missing-{field}"})


def test_decoder_lora_runtime_rechecks_ids_that_become_local(monkeypatch, tmp_path):
    config = _config()
    monkeypatch.chdir(tmp_path)
    local_base = tmp_path / "org" / "compact-decoder"
    local_base.mkdir(parents=True)

    with pytest.raises(DecoderOnlyLoRAError, match="base_model_id resolved to a local path"):
        DecoderOnlyLoRASymbolGenerator(config)


def test_decoder_lora_requires_a_lossless_tokenizer():
    class NormalizingTokenizer(CharacterTokenizer):
        def decode(self, token_ids, **kwargs):
            return super().decode(token_ids, **kwargs).strip()

    generator = DecoderOnlyLoRASymbolGenerator(
        _config(),
        tokenizer=NormalizingTokenizer(),
        model=ConstrainedFakeModel("WORD_open"),
    )

    with pytest.raises(DecoderOnlyLoRAError, match="losslessly encode"):
        generator.generate_symbols('{"id":"test"}', frozenset({"WORD_open"}))


def test_decoder_lora_runtime_requires_safe_pinned_artifacts(monkeypatch, tmp_path):
    calls = []
    tokenizer = CharacterTokenizer()
    base_model = object()
    adapter_snapshot = tmp_path / "snapshots" / ADAPTER_REVISION
    adapter_snapshot.mkdir(parents=True)
    (adapter_snapshot / "adapter_model.safetensors").touch()
    (adapter_snapshot / "adapter_config.json").write_text("{}")
    peft_config = SimpleNamespace(
        peft_type="LORA",
        task_type="CAUSAL_LM",
        base_model_name_or_path="org/compact-decoder",
        revision=BASE_REVISION,
    )

    class Factory:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def from_pretrained(self, *args, **kwargs):
            calls.append((self.name, args, kwargs))
            return self.value

    class AdapterModel:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True

    adapter_model = AdapterModel()
    modules = {
        "transformers": SimpleNamespace(
            AutoTokenizer=Factory("tokenizer", tokenizer),
            AutoModelForCausalLM=Factory("base", base_model),
        ),
        "peft": SimpleNamespace(
            PeftConfig=Factory("adapter_config", peft_config),
            PeftModel=Factory("adapter", adapter_model),
        ),
        "huggingface_hub": SimpleNamespace(
            snapshot_download=Factory(
                "adapter_snapshot",
                str(adapter_snapshot),
            ).from_pretrained
        ),
    }
    monkeypatch.setattr(decoder_lora.importlib, "import_module", modules.__getitem__)

    generator = DecoderOnlyLoRASymbolGenerator(_config(local_files_only=True))

    assert generator.model_id.endswith(
        f"{BASE_REVISION}+peft:org/ste-symbol-lora@{ADAPTER_REVISION}"
    )
    assert calls == [
        (
            "adapter_snapshot",
            (),
            {
                "repo_id": "org/ste-symbol-lora",
                "revision": ADAPTER_REVISION,
                "local_files_only": True,
                "allow_patterns": [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                ],
            },
        ),
        (
            "adapter_config",
            (str(adapter_snapshot),),
            {"local_files_only": True},
        ),
        (
            "tokenizer",
            ("org/compact-decoder",),
            {
                "revision": BASE_REVISION,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        ),
        (
            "base",
            ("org/compact-decoder",),
            {
                "revision": BASE_REVISION,
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
            },
        ),
        (
            "adapter",
            (base_model, str(adapter_snapshot)),
            {
                "config": peft_config,
                "local_files_only": True,
                "is_trainable": False,
            },
        ),
    ]
    assert adapter_model.evaluated


@pytest.mark.parametrize(
    "missing",
    ["adapter_config.json", "adapter_model.safetensors"],
)
def test_decoder_lora_rejects_snapshot_without_required_safe_artifacts(tmp_path, missing):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for filename in {"adapter_config.json", "adapter_model.safetensors"} - {missing}:
        (snapshot / filename).touch()
    hub = SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot))

    with pytest.raises(DecoderOnlyLoRAError, match=missing):
        DecoderOnlyLoRASymbolGenerator._resolve_safe_adapter_snapshot(
            _config(),
            hub,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"peft_type": "IA3"}, "PEFT type LORA"),
        ({"task_type": "SEQ_2_SEQ_LM"}, "CAUSAL_LM"),
        ({"base_model_name_or_path": "org/different-model"}, "configured base model"),
        ({"revision": None}, "declare the configured base model revision"),
        ({"revision": "different-base-revision"}, "declare the configured base model revision"),
    ],
)
def test_decoder_lora_rejects_incompatible_adapter_configuration(update, message):
    values = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "org/compact-decoder",
        "revision": BASE_REVISION,
    }
    values.update(update)

    with pytest.raises(DecoderOnlyLoRAError, match=message):
        DecoderOnlyLoRASymbolGenerator._validate_adapter_config(
            _config(),
            SimpleNamespace(**values),
        )

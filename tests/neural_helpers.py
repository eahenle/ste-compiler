from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

FIXTURE_REPO_ID = "fixture/tiny-t5"
FIXTURE_REVISION = "1111111111111111111111111111111111111111"


def build_tiny_t5_snapshot(root: Path) -> Path:
    import torch
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing

    snapshot = root / FIXTURE_REVISION
    snapshot.mkdir(parents=True)
    special = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    vocab = {
        **special,
        **{token: index + len(special) for index, token in enumerate(sorted(ByteLevel.alphabet()))},
    }
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevelDecoder()
    backend.post_processor = TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B:1 </s>:1",
        special_tokens=[("</s>", 1)],
    )
    tokenizer_class: Any = getattr(transformers, "TokenizersBackend", None)
    if tokenizer_class is None:
        tokenizer_class = transformers.PreTrainedTokenizerFast
    tokenizer = tokenizer_class(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=8192,
    )
    tokenizer.save_pretrained(snapshot)

    torch.manual_seed(0)
    model_config = transformers.T5Config(
        vocab_size=len(vocab),
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = transformers.AutoModelForSeq2SeqLM.from_config(model_config)
    save_kwargs = (
        {"safe_serialization": True}
        if "safe_serialization" in inspect.signature(model.save_pretrained).parameters
        else {}
    )
    model.save_pretrained(snapshot, **save_kwargs)
    return snapshot

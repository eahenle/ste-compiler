# ADR 0002: Constrained encoder-decoder symbol generation

## Status

Accepted as an inference adapter design. No trained model or quality result is shipped.

## Decision

Use a small encoder-decoder Transformers model behind the provider-neutral
`SymbolGenerator` protocol. The immutable adapter configuration requires both a Hugging Face Hub
repository ID and a full lowercase 40-character commit digest. Local filesystem identifiers are
rejected before loading because a digest-shaped configuration value cannot make mutable local
contents immutable. Tags, branches, abbreviated hashes, and other mutable or ambiguous revision
labels are also rejected. The exact digest is retained separately in realization metadata as well
as in the revision-qualified model ID. Loading is lazy, disables remote model code, and remains
optional through the `neural` dependency extra.

The encoder receives canonical serialized IR. The decoder can emit only token paths that form:

```text
SYMBOL (SPACE SYMBOL)* EOS
```

For each request, `SYMBOL` is restricted to the deterministic reference plan's document-specific
allowlist. The adapter builds first-symbol and space-prefixed token tries from the selected
tokenizer, supplies the resulting prefix constraint to deterministic generation, requires explicit
EOS termination, rejects non-padding tokens after EOS, and checks the decoded symbols against the
allowlist again.

## Safety boundary

Tokenizer constraints reduce the model's output language; they do not establish semantic
correctness. Model IDs, revisions, scores, and other model metadata are untrusted audit context.
`SymbolicLexicalizer` remains the only path from symbols to controlled words, terms, units, and
quantities. Independent exact surface alignment and deterministic validators remain authoritative,
including for omissions and reordered sentences.

## Consequences and limits

- The tokenizer must represent every allowed symbol without EOS or an unknown token.
- Every first and space-prefixed symbolic form must round-trip through the tokenizer exactly;
  normalization is rejected before generation.
- Source truncation and the output-token cap are explicit configuration choices and can reduce
  quality; truncated, incomplete, or unterminated output is rejected.
- Inherited checkpoint generation modes are neutralized; beam count and output length come only
  from `EncoderDecoderConfig`, with one return sequence and no minimum-length or forced-token mode.
- Generation explicitly requests tensor output and also validates that any returned sequence is a
  one-dimensional series of integer token IDs.
- The adapter does not download or initialize a model until its first generation request. Tests use
  injected fakes and make no network calls.
- This slice ships no training recipe, checkpoint, benchmark, parameter-count claim, or consumer-GPU
  result. Those require a pinned corpus manifest and a separately reported experiment.

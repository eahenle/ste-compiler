# ADR 0002: Decoder-only LoRA symbolic generator

**Status:** accepted for an inference adapter, 2026-07-27

## Decision

Provide a concrete, optional decoder-only adapter based on Transformers causal language
models and PEFT LoRA adapters. A deployment configuration records and passes the base model
Hub ID and full lowercase 40-character commit digest plus the adapter Hub ID and its full commit
digest. Local paths, tags, branches, abbreviated hashes, and other mutable or ambiguous revision
labels are rejected. The revision-qualified model ID and both exact digests are retained in
realization metadata. Imports are lazy, remote model code is disabled, and callers can require
cached artifacts with `local_files_only`. Base weights must be available as safetensors. Before
PEFT loading, the adapter commit is resolved once to an exact local snapshot containing both
`adapter_config.json` and `adapter_model.safetensors`; configuration and weights are then loaded
only from that checked snapshot. Pickle-only base or adapter weights fail closed. The pinned
adapter configuration must identify LoRA for `CAUSAL_LM` and name the exact configured base model
and commit.

The prompt is a versioned canonical JSON envelope containing the serialized IR. Generation is
greedy and batch size one; inference explicitly neutralizes inherited sampling, beam,
contrastive, DoLa, constrained-beam, assisted, return-count, return-shape, and minimum-length
settings. It also clears inherited forced BOS/EOS IDs, token suppression and bad-word lists,
decoder and encoder no-repeat n-gram processors, time and stop-string criteria, and both
`min_length` and `min_new_tokens`. These overrides prevent inherited processors from masking a
grammar-permitted token or terminating before explicit EOS, including when the plan reaches
`max_symbols`. A token grammar constructs lossless tokenizer encodings for only the
document-specific allowed symbols. The model can continue with one of those encodings or terminate
with EOS; a post-generation check requires exactly one batch dimension containing only integer
token IDs, replays the raw pre-EOS token path through the same grammar, and independently rejects
multiple sequences, malformed token shapes, hidden special tokens, missing termination,
noncanonical spacing, too many symbols, and any escaped symbol.

## Trust boundary

The adapter is a proposal mechanism, not a semantic authority. Its prompt and output are
untrusted. LoRA changes model probabilities but cannot prove correctness. `NeuralRealizer`
still lexicalizes only allowlisted symbols, and the independent exact surface aligner grants
IR mappings only to position-preserving deterministic sentences. Existing lexical, structural,
and semantic validators remain authoritative.

The tokenizer must losslessly round-trip every allowed symbolic form. This explicitly avoids
assuming that a model token is an approved word. It also makes tokenizer incompatibility a
fail-closed error rather than silently weakening constraints.

Pinned revisions identify the artifacts used, but pinning does not make externally supplied
artifacts trustworthy. Local paths are deliberately unsupported because an arbitrary revision
label cannot identify their contents. The safetensors requirement and exact-snapshot reuse remove
pickle deserialization from this loader; operators must still authorize the configured
repositories and revisions.

## Consequences and unshipped work

The adapter supports inference wiring and offline fake-based tests. It does not ship trained
weights, a training loop, chosen public model revisions, benchmark results, or a claim that a
decoder-only architecture is better than the encoder-decoder alternative. Training and
evaluation must record the corpus manifest hash, seeds, hardware, dependency versions, base
and adapter revisions, and constrained versus unconstrained results.

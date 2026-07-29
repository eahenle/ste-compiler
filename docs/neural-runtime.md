# Typed offline neural runtime

`ste-compiler` selects deterministic, encoder-decoder, and decoder-only LoRA realization through a
strict, versioned realizer configuration. The configuration identifies a realization architecture
and its immutable model artifacts; it does not contain source text, IR, credentials, or an
instruction to download anything.

This is the first runtime-integration slice. It connects the existing constrained neural adapters
to the compiler CLI while keeping inference cache-only. It does not publish a reference model or
make a model-quality claim.

## Configuration contract

Realizer configurations use `ste-realizer-config-v1`. They are separate from
`ste-training-config-v1`: a training configuration identifies a corpus and optimizer run, while a
realizer configuration identifies artifacts and bounded decoding settings for inference.

The schema is a strict architecture-discriminated union:

- `deterministic` selects the credential-free reference realizer and has no model identity.
- `encoder-decoder` identifies one encoder-decoder checkpoint.
- `decoder-only-lora` identifies a causal base model and one compatible PEFT adapter.

Unknown fields are rejected. Neural artifact identities must be Hugging Face Hub repository IDs
plus full lowercase 40-character commit digests. Local paths, tags, branches, abbreviated hashes,
blank values, and invalid decoding bounds fail validation. The canonical configuration has a
SHA-256 identity that is included in compiler provenance.

Repository identity is also an authorization boundary. Pinning a commit makes the selected
revision immutable, but it does not decide whether an operator is permitted to use that repository.
Review the model and adapter licenses, origin, intended use, and repository owner before approving
a configuration.

The checked-in neural configurations use illustrative identities. They demonstrate the schema;
they do not select published project checkpoints, promise that an artifact is present in a local
cache, or provide evidence of model quality.

Validate a configuration without running inference:

```bash
ste-compiler validate-realizer-config path/to/realizer.yaml --json
```

## Cache-only inference

CLI inference is offline-only in this slice. The runtime forces local-file resolution for every
model, tokenizer, and adapter. It does not turn a cache miss into a network request. Missing
optional dependencies, absent revisions, incompatible tokenizers, unsafe weights, and invalid
adapter identities produce a controlled failure instead of falling back to a mutable or remote
artifact.

The existing neural loaders continue to:

- disable remote model code;
- require safetensors model and adapter weights;
- bind encoder-decoder checkpoints to one full commit;
- bind decoder adapters to both their full adapter commit and exact base-model revision; and
- retain the exact model and adapter identities in realization metadata.

An operator must prepare the approved immutable revisions in the Hugging Face cache before running
these commands. There is intentionally no implicit download mode hidden inside `compile`.

Install the optional runtime dependencies, then pass the reviewed configuration to an IR compile:

```bash
python -m pip install -e '.[neural]'
ste-compiler compile \
  data/examples/warning_pressure.yaml \
  --realizer-config path/to/realizer.yaml \
  --json
```

The same selection applies after the offline replay frontend has validated raw-source provenance:

```bash
ste-compiler compile-source \
  data/end_to_end/hydraulic_warning.txt \
  --ir-fixture data/end_to_end/hydraulic_warning.ir.yaml \
  --realizer-config path/to/realizer.yaml \
  --json
```

Omitting `--realizer-config` preserves the deterministic default.

## Neural output remains untrusted

A realizer configuration changes how a symbolic plan is proposed. It does not weaken the compiler
boundary. The neural realizer receives canonical IR and only the symbols present in the
deterministic reference plan. The generated plan must use the exact whitespace protocol, terminate
correctly, and remain inside the document-specific symbol allowlist.

The symbolic lexicalizer, independent surface alignment, lexical checks, structural checks, and
semantic validators still run after generation. Changed, missing, reordered, malformed, or
unsupported output is rejected. A configured model never writes production prose directly and
never bypasses validation.

## Relationship to training artifacts

The two smoke trainers prove offline mechanics, safe serialization, run-manifest provenance,
content-bound bundle preflight, and reload evaluation. Their generated local checkpoints are test
artifacts, not automatically authorized runtime inputs. Runtime selection deliberately continues
to reject arbitrary local model paths.

`ste-compiler preflight-artifact` verifies a trainer output against an externally retained
`artifact-manifest.json` SHA-256. It performs an exact private capture and architecture-specific
loadability checks without network access. This standalone preflight does not make the bundle a
runtime input and does not establish license authorization or model quality.

Promoting a training result to a runtime artifact requires a separate publication step that assigns
an immutable repository revision, records checksums and licensing, and produces a reviewed realizer
configuration. Direct content-addressed loading of local trainer outputs is not part of
`ste-realizer-config-v1` runtime integration yet.

## Explicit non-goals

This slice does not provide:

- a network-enabled fetch command;
- a content-digest artifact registry;
- direct inference from an unpublished local training-output directory;
- selected or published reference checkpoints;
- benchmark predictions, constrained/unconstrained comparisons, figures, or quality claims; or
- a live semantic frontend.

The next artifact slice can connect verified local bundles to a distinct content-addressed runtime
configuration without weakening the existing Hub-only identities. Explicit fetch, reference model
selection, artifact publication, and benchmark reproduction remain later release gates.

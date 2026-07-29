# Typed offline neural runtime

`ste-compiler` selects deterministic, encoder-decoder, and decoder-only LoRA realization through a
strict, versioned realizer configuration. The configuration identifies a realization architecture
and its immutable model artifacts; it does not contain source text, IR, credentials, or an
instruction to download anything.

The runtime supports immutable Hub revisions from an existing local cache and content-addressed
local trainer bundles. Neither mode implicitly downloads artifacts. This does not publish a
reference model or make a model-quality claim.

## Configuration contract

Realizer configurations use `ste-realizer-config-v1`. They are separate from
`ste-training-config-v1`: a training configuration identifies a corpus and optimizer run, while a
realizer configuration identifies artifacts and bounded decoding settings for inference.

The schema is a strict architecture-discriminated union:

- `deterministic` selects the credential-free reference realizer and has no model identity.
- `encoder-decoder` identifies one encoder-decoder checkpoint.
- `decoder-only-lora` identifies a causal base model and one compatible PEFT adapter.
- `encoder-decoder-local-bundle` identifies one complete encoder trainer bundle by manifest
  SHA-256 and `mechanics-smoke` intended use.
- `decoder-only-lora-local-bundle` identifies one decoder adapter bundle and its separately
  content-bound base snapshot, including exact base-model and tokenizer identities.

Unknown fields are rejected. Hub identities require repository IDs plus full lowercase
40-character commit digests. Local variants require lowercase 64-character manifest digests.
Filesystem paths are deliberately forbidden in the canonical configuration: `--artifact-bundle`
and, for decoder runs, `--model-snapshot` are untrusted operational locators supplied separately.
Tags, branches, abbreviated hashes, blank values, and invalid decoding bounds fail validation.
The portable canonical configuration has a SHA-256 identity that is included in compiler
provenance.

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

## Content-addressed local-bundle inference

Local-bundle inference first binds the caller's locator to the external digest, copies the exact
no-follow file inventory into a private materialization, and performs architecture-specific
validation there. Framework loaders see only that private path. There is no fallback to the
original directory, Hub resolution, remote model code, or pickle-capable weights.

For encoder-decoder bundles:

```bash
ste-compiler compile data/examples/warning_pressure.yaml \
  --realizer-config data/realizers/encoder-decoder-local-bundle-schema-example.yaml \
  --artifact-bundle path/to/encoder-training-output \
  --json
```

For decoder-only LoRA, both the adapter bundle and the exact base snapshot are required:

```bash
ste-compiler compile data/examples/warning_pressure.yaml \
  --realizer-config data/realizers/decoder-only-lora-local-bundle-schema-example.yaml \
  --artifact-bundle path/to/decoder-training-output \
  --model-snapshot path/to/base-snapshot \
  --json
```

The decoder loader cross-checks the adapter run manifest, prompt profile, base and tokenizer
identities, base-snapshot manifest digest, snapshot inventory, LoRA configuration, and paired LoRA
tensor structure before local-only Transformers and PEFT loading. Output provenance records the
portable realizer-config digest, artifact/run/snapshot digests, base revision, explicit
`mechanics-smoke` use, and `content-addressed-local-bundle` mode. It never records the caller's
machine-specific locator.

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
content-bound bundle preflight, reload evaluation, and content-addressed runtime loading. Their
generated local checkpoints remain test artifacts, not automatically authorized production
inputs. Arbitrary local paths without the external identities and exact cross-checks are rejected.

`ste-compiler preflight-artifact` verifies a trainer output against an externally retained
`artifact-manifest.json` SHA-256. It performs an exact private capture and architecture-specific
loadability checks without network access. This standalone preflight does not make the bundle a
runtime input and does not establish license authorization or model quality.

Loading proves that the supplied bytes match the reviewed content identities and are structurally
safe for the selected mechanics workflow. Promotion still requires a publication step that records
authorization, licensing, model/data cards, intended use, benchmark evidence, and reviewed external
digests.

## Explicit non-goals

This slice does not provide:

- a network-enabled fetch command;
- a content-digest artifact registry;
- selected or published reference checkpoints;
- benchmark predictions, constrained/unconstrained comparisons, figures, or quality claims; or
- a live semantic frontend.

Explicit fetch, reference model selection, artifact publication, benchmark reproduction, and
production-neutral large-model snapshot packaging remain later release gates.

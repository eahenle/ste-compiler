# Reproducible training

Training is being added in architecture-specific slices. The shared foundation is complete: strict,
versioned configurations and a race-resistant reader bind every run to one immutable corpus
release. The decoder-only track now includes a deterministic, two-step, offline CPU mechanics
smoke run over a generated tiny local causal model. It is not a selected public model, a published
checkpoint, a benchmark, or a model-quality claim. The encoder-decoder trainer and reference runs
remain future work.

## Configuration identity

Both architectures use `ste-training-config-v1`. Unknown fields are rejected. Every mutable input
that affects a run is represented by an immutable identity:

- `corpus` pins the dataset version, complete manifest, and train and validation artifacts.
- model and tokenizer identities are Hub repository IDs plus full 40-character commits.
- seeds, step counts, effective batching, optimizer values, token limits, and architecture-specific
  settings are part of the canonical configuration hash.

The package commit, clean-tree state, package-source tree digest, dependency versions, and
`uv.lock` digest are observed at run time and belong in the run manifest. The trainer rejects a
dirty source checkout or a checkout whose `src/ste_compiler` tree differs from the package code
that is executing. These values are deliberately not self-reported by the input configuration.

The decoder-only LoRA schema additionally requires one prompt profile, an exact LoRA module list,
and a tokenizer identity equal to the base-model identity. The encoder-decoder schema currently
supports full fine-tuning.

Validate and identify either packaged schema example:

```bash
ste-compiler validate-training-config \
  data/training/encoder-decoder-schema-example.yaml \
  --json
ste-compiler validate-training-config \
  data/training/decoder-only-lora-schema-example.yaml \
  --json
```

The model identities in those examples are intentionally illustrative. The decoder example's
synthetic identity is accepted only by the local fixture builder described below. A reference run
must instead select authorized repositories and exact Hub commit digests. Validation proves the
configuration shape and computes its identity; it does not download or authorize a model.

## Immutable corpus reader

`verify-training-release` validates a release before returning any records to a trainer:

```bash
ste-compiler verify-training-release \
  data/training/encoder-decoder-schema-example.yaml \
  datasets/demonstration-corpus-1 \
  --json
```

The reader pins the release directory, rejects symlinked, non-regular, and multiply linked entries,
requires the exact release file set, bounds every read, and detects files that change during a
read. It then checks:

1. the configured manifest, train, and validation SHA-256 values;
2. the complete checksum inventory and every manifest artifact size and digest;
3. strict record schemas, split membership, unique IDs, canonical IR, source hashes, symbolic-plan
   grammar, and sorted symbol and feature sets;
4. manifest split counts and runtime profiles; and
5. every deterministic text and symbolic target rebuilt from the released IR, vocabulary, and
   terminology.

The reader consumes a real release directory, not the mutable `current` selector used by the
general-purpose symbolic-corpus exporter. Its hardened directory-descriptor implementation is
POSIX-specific.

Python callers can use the same boundary:

```python
from pathlib import Path

from ste_compiler.training import load_training_config, read_training_release

config = load_training_config(Path("data/training/encoder-decoder-schema-example.yaml"))
release = read_training_release(
    Path("datasets/demonstration-corpus-1"),
    config.corpus,
)
print(len(release.train), len(release.validation))
```

`TrainingReleaseSnapshot` exposes immutable tuples for all four splits and a frozen union of the
allowed symbol inventory. Trainers may seed and shuffle their own training view; they must not
mutate or reinterpret the released ordering.

## Decoder-only LoRA smoke run

Install the optional neural dependencies. The versioned demonstration corpus is already checked in
and verified by each command. These commands create a tiny byte-level BPE tokenizer and
GPT-2-shaped causal model entirely from released local data, train exactly two AdamW optimizer
steps, atomically publish a safe adapter, reload it on a fresh base model, and evaluate the
validation split. `prepare-decoder-smoke-fixture --json` reports the required SHA-256 identity of
the generated model-snapshot manifest; pass that digest positionally after the snapshot path to
both training and evaluation:

```bash
python -m pip install -e '.[dev,neural]'
MODEL_SNAPSHOT_MANIFEST_SHA256="$(
  ste-compiler prepare-decoder-smoke-fixture \
    data/training/decoder-only-lora-schema-example.yaml \
    datasets/demonstration-corpus-1 \
    decoder-smoke-model \
    --json |
    python -c 'import json,sys; print(json.load(sys.stdin)["manifest_sha256"])'
)"
ste-compiler train-decoder-lora \
  data/training/decoder-only-lora-schema-example.yaml \
  datasets/demonstration-corpus-1 \
  decoder-smoke-model \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" \
  decoder-smoke-run \
  --source-checkout .
ste-compiler evaluate-decoder-lora \
  data/training/decoder-only-lora-schema-example.yaml \
  datasets/demonstration-corpus-1 \
  decoder-smoke-model \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" \
  decoder-smoke-run/adapter \
  --json
```

The versioned `decoder-only-symbol-plan-v1` protocol is shared by training and inference. Each
example is the canonical JSON prompt followed by segmented, losslessly round-tripped symbol
tokens. Prompt positions receive label `-100`; target positions and exactly one final EOS token
are supervised. Before training or evaluation starts, the tokenizer preflights every train,
validation, test, and adversarial example. Any example that exceeds the configured token limit
fails instead of truncating.

The output directory must not already exist. Publication uses a private sibling staging directory,
file and directory synchronization, and one atomic rename. The final artifact set is:

- `adapter/adapter_model.safetensors`, `adapter_config.json`, and a smoke-only model card;
- canonical `training-config.json` and `run-manifest.json`; and
- `checksums.sha256` covering every other output file.

The run manifest derives the package commit and dirty state from `--source-checkout`, hashes its
`uv.lock`, records installed dependency versions, corpus and model-snapshot identities, LoRA
parameter names and counts, deterministic sample order and losses, CPU/Python details, duration,
output hashes, and a shell-quoted evaluation command. Model and adapter loading is local-only and
safetensors-only. There is deliberately no checkpoint resume: optimizer state is not emitted
because the supported serialization must not introduce pickle-capable artifacts.

The fixture builder is a CI test tool. Its schema contract requires `max_steps: 2`; changing the
step count is rejected instead of silently changing the meaning of the smoke evidence. The fixture
is content-bound by its own externally supplied manifest digest and must not be substituted for a
pinned, authorized public base model in a reference experiment.

## Remaining training gates

The encoder-decoder architecture still needs its trainer and smoke coverage. Both architectures
still need documented reference configurations, explicitly selected public base-model repositories
and immutable revisions, and measured reference runs. Checkpoint resume remains blocked until
optimizer state can be represented without pickle-capable artifacts.

# Reproducible training

Training is being added in architecture-specific slices. The shared foundation is complete: strict,
versioned configurations and a race-resistant reader bind every run to one immutable corpus
release. Both architecture tracks now include deterministic offline CPU smoke trainers,
safetensors-only outputs, runtime-derived run manifests, reload evaluation, and installed-wheel
coverage. The checked-in configurations select only local test fixtures, not public reference
models, published checkpoints, benchmarks, or model-quality claims.

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
DECODER_SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ste-compiler-decoder-smoke.XXXXXX")"
DECODER_SMOKE_MODEL="$DECODER_SMOKE_ROOT/model"
DECODER_SMOKE_RUN="$DECODER_SMOKE_ROOT/run"
MODEL_SNAPSHOT_MANIFEST_SHA256="$(
  ste-compiler prepare-decoder-smoke-fixture \
    data/training/decoder-only-lora-schema-example.yaml \
    datasets/demonstration-corpus-1 \
    "$DECODER_SMOKE_MODEL" \
    --json |
    python -c 'import json,sys; print(json.load(sys.stdin)["manifest_sha256"])'
)"
ste-compiler train-decoder-lora \
  data/training/decoder-only-lora-schema-example.yaml \
  datasets/demonstration-corpus-1 \
  "$DECODER_SMOKE_MODEL" \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" \
  "$DECODER_SMOKE_RUN" \
  --source-checkout .
ste-compiler evaluate-decoder-lora \
  data/training/decoder-only-lora-schema-example.yaml \
  datasets/demonstration-corpus-1 \
  "$DECODER_SMOKE_MODEL" \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" \
  "$DECODER_SMOKE_RUN/adapter" \
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
`uv.lock`, records the complete installed-distribution inventory, corpus and model-snapshot
identities, LoRA parameter names and counts, deterministic sample order and losses, CPU/Python
details, duration, output hashes, and a shell-quoted evaluation command. Model and adapter loading
is local-only and safetensors-only. There is deliberately no checkpoint resume: optimizer state is
not emitted because the supported serialization must not introduce pickle-capable artifacts.

The fixture builder is a CI test tool. Its schema contract requires `max_steps: 2`; changing the
step count is rejected instead of silently changing the meaning of the smoke evidence. The fixture
is content-bound by its own externally supplied manifest digest and must not be substituted for a
pinned, authorized public base model in a reference experiment.

## Encoder-decoder smoke training

Install the smaller training-specific dependency set:

```bash
python -m pip install -e '.[encoder-training]'
```

The trainer takes an encoder-decoder configuration, an exact corpus release, and an empty output
path:

```bash
ste-compiler train-encoder-decoder \
  path/to/encoder-decoder.yaml \
  path/to/corpus-release \
  --output path/to/new-checkpoint \
  --source-root . \
  --dependency-lock uv.lock \
  --cache-dir path/to/prepared-hugging-face-cache \
  --json
```

The configured model and tokenizer identities must be identical and name one authorized Hugging
Face Hub repository at a full 40-character commit. Resolution is local-only: the command neither
selects a public model nor implicitly downloads one. Prepare the exact snapshot before entering an
offline environment. The trainer stable-reads every resolved cache file into a private immutable
materialization, records every file hash and size, and loads only that capture. A normal Hub cache
layout with snapshot symlinks to content-addressed blobs is supported. Tests create a tiny local
T5-style fixture rather than relying on a public checkpoint.

Before constructing a training batch, the command verifies the configured corpus identity and
checks every source and target in all four splits. Both sides must round-trip losslessly and may not
contain an unknown token. It rejects any source or target that exceeds its configured token limit,
the tokenizer capacity, or an exposed model position capacity; it never truncates training data.
Tokenizer vocabulary and special-token IDs must exactly fit the model input and output embeddings.
Every inventory symbol is tested both at the start of a target and after a literal space, with
exact tokenizer round trips and without unknown or embedded EOS tokens. Target construction
appends one explicit EOS token.

Training is a deterministic, CPU-only PyTorch loop with the configured seed, example order,
effective batch size, gradient accumulation, and exact optimizer-step count. The smoke
configuration uses full fine-tuning because it exercises the complete encoder-decoder save and
reload boundary with fewer moving parts than an adapter. It is a pipeline check, not evidence of
model quality.

The output directory is staged beside its final destination, synchronized, and renamed only after
training, saving, reload, and validation-loss evaluation succeed. An existing destination is never
overwritten. Only safetensors model weights are allowed; pickle-capable suffixes, symlinks,
non-regular files, and multiply linked files fail closed. `run-manifest.json` records canonical
configuration and corpus identities, the complete base/tokenizer snapshot inventory,
runtime-derived package provenance, the complete installed-distribution inventory, lock-file hash,
optimizer and loss history,
parameter counts, hardware, duration, peak memory, output hashes, and the evaluation command.
Training requires a clean Git checkout and proves that its `src/ste_compiler` Python tree matches
the package that is actually executing.

Reload and verify a completed checkpoint with the same configuration and release:

```bash
ste-compiler evaluate-encoder-decoder-checkpoint \
  path/to/encoder-decoder.yaml \
  path/to/corpus-release \
  path/to/checkpoint \
  --run-manifest-sha256 <sha256-reported-by-training> \
  --json
```

Retain the externally reported run-manifest digest with the checkpoint publication record.
Evaluation requires that digest, stable-reads the complete checkpoint through no-follow directory
handles into a private materialization, rechecks all internal configuration, corpus, training,
snapshot, package, metric, and output identities, then loads only the private safetensors capture.
It also requires the recomputed validation metrics to match the recorded metrics. It does not yet
produce benchmark predictions or a released prediction hash.

## Remaining training gates

Both tracks still need documented single-GPU reference configurations, published model artifacts,
evaluation predictions, and measured reference runs. Checkpoint resume remains blocked until
optimizer state can be represented without pickle-capable artifacts. Public reference runs
additionally require explicitly selected base-model repositories and immutable revisions.

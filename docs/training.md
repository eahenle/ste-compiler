# Reproducible training

Training is being added in architecture-specific slices. The shared foundation is complete: strict,
versioned configurations and a race-resistant reader bind every run to one immutable corpus
release. The encoder-decoder and decoder-only LoRA trainers, run manifests, and published model
artifacts are not included yet. The checked-in configurations are schema examples, not runnable
model selections, and this repository makes no model-quality claim.

## Configuration identity

Both architectures use `ste-training-config-v1`. Unknown fields are rejected. Every mutable input
that affects a run is represented by an immutable identity:

- `corpus` pins the dataset version, complete manifest, and train and validation artifacts.
- model and tokenizer identities are Hub repository IDs plus full 40-character commits.
- seeds, step counts, effective batching, optimizer values, token limits, and architecture-specific
  settings are part of the canonical configuration hash.

The package commit, dirty-tree state, dependency versions, and lock-file digest are observed at run
time and belong in the run manifest. They are deliberately not self-reported by the input
configuration.

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

The model identities in those examples are intentionally illustrative. Replace them with real,
authorized repositories and exact Hub commit digests before training. Validation proves the
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

config = load_training_config(
    Path("data/training/encoder-decoder-schema-example.yaml")
)
release = read_training_release(
    Path("datasets/demonstration-corpus-1"),
    config.corpus,
)
print(len(release.train), len(release.validation))
```

`TrainingReleaseSnapshot` exposes immutable tuples for all four splits and a frozen union of the
allowed symbol inventory. Trainers may seed and shuffle their own training view; they must not
mutate or reinterpret the released ordering.

## Remaining training gates

Each architecture still needs an offline, two-optimizer-step CPU smoke trainer, safetensors-only
atomic outputs, a canonical run manifest, reload and evaluation checks, installed-wheel coverage,
and a documented reference configuration. Checkpoint resume remains blocked until optimizer state
can be represented without pickle-capable artifacts. Public reference runs additionally require an
explicitly selected base-model repository and immutable revision.

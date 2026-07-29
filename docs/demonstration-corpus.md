# Demonstration corpus

`datasets/demonstration-corpus-1` is the first separately consumable dataset release for the
project. It contains 12 original, synthetic technical examples under the MIT license. It does not
contain ASD-STE100 text, rules, or vocabulary and does not claim training adequacy, certification,
or production quality.

The records are frozen into four purpose-specific splits:

| Split | Records | Intended use |
| --- | ---: | --- |
| `train` | 4 | Basic realization patterns |
| `validation` | 2 | Model and configuration selection |
| `test` | 3 | Held-out semantic compositions |
| `adversarial` | 3 | Ambiguity, Unicode, temporal, and reference boundaries |

## Reconstruct and verify

The construction input and terminology snapshot are packaged in the wheel. Reconstruct the release
without network access:

```bash
ste-compiler build-demonstration-corpus --output rebuilt-corpus
ste-compiler verify-demonstration-corpus rebuilt-corpus
```

`verify-demonstration-corpus` reloads the embedded vocabulary, terminology, and construction
snapshot, rebuilds the release in an isolated temporary directory, and requires every released byte
to match. The repository test suite also reconstructs the checked-in release and compares every
file byte for byte.

## Artifact contract

The release contains:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `adversarial.jsonl`
- `source-construction.json` with the recorded seed and original raw sources
- `vocabulary.json` and `terminology.json` resource snapshots
- `dataset-card.md`
- `license-inventory.json`
- `feature-coverage.json`
- `leakage-report.json`
- `manifest.json`
- `checksums.sha256`

Each JSONL record includes its raw source and SHA-256, materialized source spans, schema-valid IR,
canonical serialized IR, deterministic controlled text, exact symbolic plan, allowed-symbol set,
runtime profile, split, license ID, and machine-reported semantic features.

The constructor rejects empty splits, missing required feature coverage, duplicate record or source
IDs, normalized source duplicates across splits, and test/adversarial feature compositions that are
identical to a training composition. Construction metadata declares the dataset and resource
licenses, origins, versions, and canonical SHA-256 values; supplied records and resource snapshots
must match those declarations. The constructor runs every record through IR validation,
deterministic realization, symbolization, and the validation pipeline before writing any release.

## Scope

This is deliberately a small demonstration corpus. It is sufficient to exercise the complete data
contract and the project’s neural smoke paths, but it is not enough to establish model quality.
Phase 1 remains open for faithful causal-relation realization, benchmark-scale expansion,
exhaustive schema-negative examples, and a versioned release attachment suitable for independent
download.

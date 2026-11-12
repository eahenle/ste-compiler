# Demonstration corpus

The repository contains three separately consumable, byte-reproducible dataset releases.

- `datasets/demonstration-corpus-1` is the 12-record smoke sample and the CLI default.
- `datasets/demonstration-corpus-2` is the 24-record benchmark-contract demonstration with broader
  terminology and source-boundary coverage.
- `datasets/demonstration-corpus-3` is the 64-record synthetic contract benchmark with broader
  semantic composition coverage.

All three releases contain only original, synthetic technical examples under the MIT license.
The examples and vocabulary were authored independently of ASD-STE100. Training adequacy,
production readiness, model quality, and ASD-STE100 compliance require separate evidence.

Each release freezes its records into four purpose-specific splits.

| Split | Corpus 1 | Corpus 2 | Corpus 3 | Intended use |
| --- | ---: | ---: | ---: | --- |
| `train` | 4 | 12 | 32 | Basic realization patterns |
| `validation` | 2 | 4 | 8 | Model and configuration selection |
| `test` | 3 | 4 | 12 | Held-out semantic compositions |
| `adversarial` | 3 | 4 | 12 | Ambiguity, Unicode, temporal, causal, and reference boundaries |

Corpus 2 adds records that exercise canonical, alias, and deprecated terminology handling and
uppercase, colon, and tab boundaries. It also adds a held-out composition combining negation, a
condition, and a quantity. The normative contract is in
[demonstration-corpus-v2-spec.md](demonstration-corpus-v2-spec.md).

## Additive Corpus 3

Corpus 3 retains all 24 Corpus 2 records in their original splits. Its 40 new examples each have a
distinct semantic feature composition absent from Corpus 2. The Corpus 3 test and adversarial splits
exercise compositions held out from its training split. See
[the Corpus 3 specification](demonstration-corpus-v3-spec.md).

## Reconstruct and verify

The construction input and terminology snapshot are packaged in the wheel. Reconstruct the release
without network access:

```bash
ste-compiler build-demonstration-corpus --output rebuilt-corpus
ste-compiler verify-demonstration-corpus rebuilt-corpus
```

Select the packaged construction inputs for Corpus 2 or Corpus 3 with `--version`.

```bash
ste-compiler build-demonstration-corpus --version 2 --output rebuilt-corpus-2
ste-compiler verify-demonstration-corpus rebuilt-corpus-2

ste-compiler build-demonstration-corpus --version 3 --output rebuilt-corpus-3
ste-compiler verify-demonstration-corpus rebuilt-corpus-3
```

`verify-demonstration-corpus` reloads the embedded vocabulary, terminology, and construction
snapshot, rebuilds the release in an isolated temporary directory, and requires every released byte
to match. The repository test suite also compares the rebuilt and checked-in releases byte for
byte.

The frozen benchmark report uses Corpus 2. The non-publishing release workflow packages Corpus 2 as
`ste-compiler-<version>-dataset-demonstration-corpus-2.tar`. Its candidate manifest is bound to the
same release identity as the wheel, source distribution, fixture-report candidate, and outer
release manifest. Candidate verification reconstructs and validates the corpus before finalization;
the trusted workflow rebuilds the archive independently and requires byte identity. This candidate
remains verification evidence until an authorized release process publishes it.

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

These deliberately small demonstration corpora exercise the data contract. The checked-in neural
smoke examples use Corpus 1. Corpus 2 makes the benchmark shape and acceptance gates executable.
Phase 1 still requires benchmark-scale source expansion and an authorized downloadable GitHub
release attachment.

# Demonstration corpus

The repository contains two separately consumable, byte-reproducible dataset releases:

- `datasets/demonstration-corpus-1` is the 12-record wheel-bundled smoke sample.
- `datasets/demonstration-corpus-2` is the 24-record benchmark-contract demonstration with broader
  terminology and source-boundary coverage.

Both releases contain only original, synthetic technical examples under the MIT license. They do
not contain ASD-STE100 text, rules, or vocabulary and do not claim training adequacy,
certification, production quality, or statistically meaningful model-quality measurement.

The records are frozen into four purpose-specific splits:

| Split | Records | Intended use |
| --- | ---: | --- |
| `train` | 4 | Basic realization patterns |
| `validation` | 2 | Model and configuration selection |
| `test` | 3 | Held-out semantic compositions |
| `adversarial` | 3 | Ambiguity, Unicode, temporal, causal, and reference boundaries |

Corpus 2 freezes 12 train, 4 validation, 4 compositional-test, and 4 adversarial records. Its
additional records exercise canonical, alias, and deprecated terminology handling; uppercase,
colon, and tab boundaries; and a held-out negation-plus-condition-plus-quantity composition. The
normative contract is in [demonstration-corpus-v2-spec.md](demonstration-corpus-v2-spec.md).

## Reconstruct and verify

The construction input and terminology snapshot are packaged in the wheel. Reconstruct the release
without network access:

```bash
ste-compiler build-demonstration-corpus --output rebuilt-corpus
ste-compiler verify-demonstration-corpus rebuilt-corpus
```

Build corpus 2 by selecting its packaged construction inputs:

```bash
ste-compiler build-demonstration-corpus --version 2 --output rebuilt-corpus-2
ste-compiler verify-demonstration-corpus rebuilt-corpus-2
```

`verify-demonstration-corpus` reloads the embedded vocabulary, terminology, and construction
snapshot, rebuilds the release in an isolated temporary directory, and requires every released byte
to match. The repository test suite also reconstructs the checked-in release and compares every
file byte for byte.

The non-publishing release workflow packages Corpus V2 as
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

These are deliberately small demonstration corpora. They exercise the complete data contract and
the project’s neural smoke paths, but they are not enough to establish model quality. Corpus 2
makes the benchmark shape and acceptance gates executable. Its reproducible release candidate now
exists, but benchmark-scale source expansion and an authorized downloadable GitHub release
attachment remain before Phase 1 is complete.

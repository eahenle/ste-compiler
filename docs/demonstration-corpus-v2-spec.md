# Demonstration corpus 2 specification

`demonstration-corpus-2` is the first frozen release that exercises the project's benchmark
contract. It is a contract and integration benchmark, not a statistical model-quality benchmark.

## Identity and licensing

- Dataset version: `demonstration-corpus-2`
- Construction seed: `2718`
- Dataset license: MIT
- Source origin: original synthetic examples authored for `ste-compiler`
- Vocabulary: `demo-3`, with its canonical SHA-256 recorded in the construction input
- Terminology: `demonstration-corpus-2`, with its canonical SHA-256 recorded in the construction
  input

No source, rule, or vocabulary entry is copied from ASD-STE100. Each released record repeats its
source license and SHA-256. `license-inventory.json` records every source and resource identity.

## Frozen split profile

| Split | Exact frozen count | Purpose |
| --- | ---: | --- |
| `train` | 12 | Basic and combined realization patterns |
| `validation` | 4 | Configuration selection without test exposure |
| `test` | 4 | Held-out semantic compositions |
| `adversarial` | 4 | Ambiguity, references, Unicode, whitespace, punctuation, and casing |

The constructor requires exactly 24 records and the exact split counts above. It also rejects empty
splits, duplicate record or source IDs, normalized source duplicates across splits, and
test/adversarial feature compositions present in training. The hash-pinned training reader
independently enforces the same version profile.

## Required positive coverage

Corpus 2 inherits every corpus-1 semantic feature gate and additionally requires:

- `source.casing_upper`
- `source.punctuation_colon`
- `source.whitespace_tab`
- `terminology.alias_surface`
- `terminology.canonical_surface`
- `terminology.deprecated_reference`

The deprecated reference resolves through an explicit replacement term. The released record keeps
the original typed reference in its IR while deterministic realization uses the approved canonical
replacement.

`feature-coverage.json` is the machine-readable positive-coverage report. The regular test suite
also exercises invalid construction schema fields, resource identities, licenses, source spans,
required feature coverage, leakage, nonempty output handling, and release tampering. Schema-derived
property tests cover every document-reachable nested IR model and field through required-field
deletion, defaulted-field omission, unknown-field injection, constrained-value mutations, and
graph-invariant mutations.

## Reproducibility

Run:

```bash
ste-compiler build-demonstration-corpus --version 2 --output rebuilt-corpus-2
ste-compiler verify-demonstration-corpus rebuilt-corpus-2
```

The build runs without network access. It validates construction provenance, materializes source
spans, validates every IR, realizes controlled text, creates symbolic plans and allowed-symbol
sets, runs the validation pipeline, checks coverage and leakage, and writes checksummed artifacts.
Verification rebuilds in an isolated temporary directory and compares every released byte.

The checked-in `datasets/demonstration-corpus-2` directory is the canonical release payload from
which a downloadable archive can be created. Publication of that archive is intentionally outside
the constructor so repository tests do not depend on GitHub credentials or network access.

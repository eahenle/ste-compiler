# Benchmark evidence

The benchmark evidence pipeline turns raw, per-example observations into a hash-bound metrics file,
Markdown report, and report manifest. Every reported value is recomputed from the raw JSONL. The
pipeline rejects a changed benchmark specification, taxonomy, dataset release, system identity, or
prediction byte stream.

The checked-in `data/benchmark/v1` material is deliberately labeled
`deterministic_fixture_only`. Its four hand-authored observations exercise one successful workflow
and one failure at each of the frontend, realizer, and validator stages. The resulting numbers test
the reporting mechanics. They are not model-quality measurements, external benchmark results,
certification evidence, or evidence of ASD-STE100 compliance.

## Reproduce the fixture report

Run without network access or credentials:

```bash
ste-compiler benchmark-report \
  data/benchmark/v1/benchmark-spec.json \
  data/benchmark/v1/failure-taxonomy.json \
  data/benchmark/v1/prediction-manifest.json \
  data/benchmark/v1/predictions.jsonl \
  datasets/demonstration-corpus-2 \
  --output benchmark-report \
  --json
```

The output contains:

- `metrics.json`: recomputed counts, rates, Wilson 95% intervals, and input SHA-256 bindings
- `report.md`: the same metrics plus uncensored deterministic failure fixtures and claim limits
- `report-manifest.json`: hashes and byte lengths for both generated artifacts

Running the command twice with the same inputs produces identical bytes. The test suite compares a
fresh report with `data/benchmark/v1/expected-report`.

## Frozen inputs and cross-binding

`benchmark-spec.json` freezes:

- the benchmark ID, version, seed, evidence label, and claim scope;
- the exact corpus manifest SHA-256;
- ordered case IDs, splits, and source SHA-256 values;
- ordered system identities and their evidence kind;
- the failure-taxonomy SHA-256;
- the complete v1 metric inventory and confidence-interval method.

`prediction-manifest.json` repeats and binds the specification SHA-256, dataset identity, exact
system definitions, prediction filename, byte length, SHA-256, and record count. The report command
then fully validates the selected corpus release with the existing race-resistant training-release
reader and confirms each case's split and source hash.

Library callers can use the exported `recompute_metrics()` function only with a validated
`BenchmarkSpecV1` and typed prediction records. The function requires the complete, ordered
case-by-system Cartesian product with no missing, duplicate, or extra records. It also checks every
record's benchmark ID, dataset identity, case source SHA-256, system identity, evidence kind, and
per-case gold contract before computing any metric. Report generation uses the same validator.

An `external_measured` system must provide an immutable artifact-manifest SHA-256. A
`deterministic_fixture` system must not claim one, and a specification cannot mix the two evidence
kinds. Metrics are computed independently for every system.

The v1 report generator intentionally fails closed for `external_measured` specifications. Its raw
observation counts are sufficient to test report mechanics, but they are attestations rather than
values recomputed from released frontend proposals, symbolic plans, controlled text, and validator
artifacts. A future schema must bind those raw stage artifacts and an evaluator manifest before the
tool will label a report external measured evidence.

## Raw prediction observations

Each JSONL record contains:

- benchmark, dataset, source, case, and system identities;
- frontend schema, required-field, hallucination, ambiguity, and source-span counts;
- realizer plan, grammar, EOS, symbol-authorization, and constraint-rejection observations;
- validator disposition, expected disposition, and raw diagnostic codes;
- provenance and repeatability observations;
- the first failed stage, a taxonomy code, raw output, and explanatory notes.

Schema invariants enforce causal stage ordering. A frontend failure prevents realization and
validation. A realizer failure prevents validation. A validator failure is only possible after
both upstream stages succeed. This makes stage counts meaningful rather than inferring ownership
from an undifferentiated final error.

## Metrics

The v1 metric inventory covers pipeline success; frontend schema, field, hallucination, ambiguity,
and source-span behavior; realizer exact-plan, grammar, EOS, unauthorized-symbol, and constraint
behavior; validator acceptance, rejection, and false acceptance; provenance coverage; and
deterministic repeatability.

Rates store their raw numerator and denominator. Binomial rates include Wilson score 95% intervals.
Required-field F1 is deterministically derived from micro precision and recall and is explicitly
marked as a derived value without a binomial interval. A zero denominator produces `null` rather
than silently reporting zero.

The fixture does not report latency, throughput, peak memory, artifact size, human review,
constrained/unconstrained model ablations, or model-quality comparisons. Those require reviewed
external measured runs and suitable raw observation fields before they can be claimed.

## Failure taxonomy

The frozen taxonomy separates:

- frontend schema, required-field, and source-span failures;
- realizer unauthorized-symbol, incomplete-EOS, grammar, and standalone constraint-rejection
  failures;
- validator semantic, lexical, structural, and false-accept failures.

Prediction codes must exist in the hash-bound taxonomy and match the observed failure stage. Reports
include every failed fixture's raw output and notes.

## External measured runs

No external measured run is checked in, and the v1 generator rejects attempts to create one. A
future measured release must:

1. select and publish reviewed model/run artifacts with immutable manifests and licenses;
2. define a new schema that binds the evaluator and recomputable raw artifacts for every stage;
3. create a frozen benchmark specification whose systems are `external_measured`;
4. record raw predictions and resource observations without deleting failed examples;
5. bind prediction and dataset manifests exactly;
6. generate the report from those released predictions;
7. publish predictions, reports, hardware/dependency disclosure, and checksums together.

The fixture report must never be relabeled or cited as if those steps had occurred.

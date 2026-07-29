# Benchmark evidence report

> Deterministic fixture evidence only. These values test the evidence pipeline; they are not model-quality measurements, certification evidence, or external benchmark results.

Benchmark: `ste-compiler-pipeline-fixture-1`

Claim scope:

    Exercises benchmark cross-binding, metric recomputation, reporting, and stage-specific failure attribution with deterministic fixtures.

## System `failure-taxonomy-fixture-v1`

### Recomputed metrics

| Metric | Value | Numerator | Denominator | 95% interval |
| --- | ---: | ---: | ---: | --- |
| `complete_success_rate` | 0.250000 | 1 | 4 | [0.045587, 0.699358] |
| `frontend.schema_valid_rate` | 0.750000 | 3 | 4 | [0.300642, 0.954413] |
| `frontend.required_field_precision` | 1.000000 | 14 | 14 | [0.784689, 1.000000] |
| `frontend.required_field_recall` | 0.823529 | 14 | 17 | [0.589705, 0.938089] |
| `frontend.required_field_f1` | 0.903226 | n/a | n/a | n/a (derived) |
| `frontend.hallucinated_node_rate` | 0.000000 | 0 | 11 | [0.000000, 0.258833] |
| `frontend.ambiguity_preservation_rate` | 0.000000 | 0 | 1 | [0.000000, 0.793451] |
| `frontend.source_span_exact_rate` | 0.750000 | 3 | 4 | [0.300642, 0.954413] |
| `frontend.source_span_overlap_rate` | 0.750000 | 3 | 4 | [0.300642, 0.954413] |
| `realizer.exact_symbolic_plan_rate` | 0.333333 | 1 | 3 | [0.061492, 0.792340] |
| `realizer.grammar_valid_rate` | 0.666667 | 2 | 3 | [0.207660, 0.938508] |
| `realizer.eos_completion_rate` | 1.000000 | 3 | 3 | [0.438503, 1.000000] |
| `realizer.unauthorized_symbol_rate` | 0.032258 | 1 | 31 | [0.005717, 0.161941] |
| `realizer.constraint_rejection_rate` | 0.333333 | 1 | 3 | [0.061492, 0.792340] |
| `validator.accepted_rate` | 0.500000 | 1 | 2 | [0.094531, 0.905469] |
| `validator.rejected_rate` | 0.500000 | 1 | 2 | [0.094531, 0.905469] |
| `validator.false_accept_rate` | n/a | 0 | 0 | n/a |
| `provenance_coverage_rate` | 1.000000 | 4 | 4 | [0.510109, 1.000000] |
| `deterministic_repeatability_rate` | 1.000000 | 4 | 4 | [0.510109, 1.000000] |

### Failure taxonomy counts

| Stage | Count |
| --- | ---: |
| `frontend` | 1 |
| `none` | 1 |
| `realizer` | 1 |
| `validator` | 1 |

## Uncensored deterministic failure fixtures

### `adversarial_ambiguity` — `frontend.schema_invalid`

Stage: `frontend`

Raw output:

    {"sections":[

Notes:

    Hand-authored invalid-JSON fixture proves frontend failures stop downstream stages.

### `adversarial_unicode_term` — `realizer.unauthorized_symbol`

Stage: `realizer`

Raw output:

    PLAN_EXACT_WHITESPACE_V1 WORD_Install SPACE UNAUTHORIZED_symbol

Notes:

    Hand-authored out-of-set symbol fixture proves realization failures remain distinct from validation.

### `adversarial_tab_casing` — `validator.semantic_rejection`

Stage: `validator`

Raw output:

    Disconnect the power supply.

Notes:

    Hand-authored omission fixture reaches independent validation and is rejected there.

## External measured runs

No external measured runs are included. The v1 generator fails closed for measured evidence until a future schema binds recomputable raw stage artifacts and an evaluator manifest.

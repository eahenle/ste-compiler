# V1 end-to-end implementation plan

## Goal

V1 is an open-source, reproducible demonstration of the complete compiler concept:

```text
raw technical source
  -> schema-validated semantic IR with verified source spans
  -> deterministic or constrained-neural symbolic realization
  -> controlled text
  -> lexical, structural, and semantic validation
  -> auditable output or structured rejection
```

The demonstration remains STE-inspired. It does not reproduce ASD-STE100 rules or vocabulary and
does not claim certification or compliance.

## V1 definition of done

A new user can:

1. Install a released wheel from a public package index.
2. Run one offline command from checked-in raw source to validated controlled text.
3. Run the same gold IR through the deterministic, encoder-decoder, and decoder-only LoRA paths.
4. Download versioned dataset and model artifacts by immutable identity and checksum.
5. Rebuild the corpus from documented source-construction inputs and reproduce its manifest hash.
6. Re-run training or consume the published checkpoints with the exact recorded configuration.
7. Reproduce the checked-in benchmark predictions, metrics, and failure analysis.
8. Inspect provenance, model identity, dataset identity, validation results, and rejection reasons.
9. Execute all documented examples as tested programs.
10. Verify every shipped artifact's license, origin, integrity, and intended-use statement.

## Principles

- Deterministic validation is authoritative; neural components only propose typed IR or symbols.
- Offline, credential-free reproduction is a first-class workflow.
- Live providers are optional adapters and never the only way to exercise the pipeline.
- Training and benchmark artifacts are immutable, checksummed, licensed, and independently
  consumable.
- Dataset coverage is measured against semantic features and compositions, not only row count.
- Unsafe or ambiguous model artifacts fail closed.
- Published results include failures and rejection rates, not only accepted examples.

## Phase 0: Freeze the reference workflow

Status: complete. The offline replay frontend, exact provenance verification, packaged
`ste-compiler demo`, explicit `compile-source` workflow, and `compile-source-v1` output marker are
implemented. `ste-compiler schema compile-source` exposes the formal result schema. Three
MIT-licensed source/IR pairs cover a hazard workflow plus verbatim Corpus V2 multi-section and
reference/causal records.

### Deliverables

- `ste-compiler compile-source` command.
- Offline replay frontend that returns a stored IR proposal while exercising the same schema and
  provenance boundary used by a live frontend.
- Exact source-span verification: source identity, bounds, and quoted source text.
- Three checked-in raw-source examples with gold IR and expected validated output.
- Machine-readable output containing IR, controlled text, mappings, validation, and reproducibility
  metadata.
- A versioned JSON output schema for end-to-end results.

### Acceptance gates

- The workflow runs without network access or credentials.
- A versioned replay fixture binds to the complete source SHA-256; any byte change invalidates it
  and fails without a traceback.
- Frontend identity in output metadata comes from the configured frontend, not from its proposal.
- The installed wheel can execute the example outside the repository checkout.
- The deterministic result is stable across supported Python versions.

## Phase 1: Complete the demonstration corpus

Status: in progress. `demonstration-corpus-2` now provides a 24-record, original MIT-licensed
benchmark-contract demonstration with frozen 12/4/4/4 splits, raw source and gold IR,
deterministic text and symbolic plans, resource snapshots, expanded terminology and boundary
coverage, leakage checks, licensing, manifests, checksums, and byte-for-byte reconstruction.
Schema-derived property tests now delete every required field, omit every defaulted field, reject
unknown fields at every document-reachable nested model boundary, and exercise constrained fields
and graph invariants. Benchmark-scale source expansion and publication as a versioned downloadable
release remain.

### Deliverables

- Original or otherwise redistributable raw technical source documents.
- Gold IR, source spans, deterministic text, symbolic plans, and allowed-symbol sets.
- Deterministic construction/generation scripts with recorded seeds.
- Frozen train, validation, compositional-test, and adversarial-test splits.
- Immutable corpus generations produced by `export-symbolic-corpus`.
- Dataset card and machine-readable license/provenance inventory.
- Feature-coverage report and duplicate/leakage report.
- Small wheel-bundled sample plus a larger separately versioned release artifact.

### Required semantic coverage

- Section kinds: procedure, description, warning, caution, and note.
- Statement kinds: instruction and state assertion.
- Actors, direct and indirect objects, manner, purpose, and references.
- Conditions, exceptions, negation, quantities, comparators, tolerances, and units.
- Before/after and causal relations.
- Hazards and thresholds.
- Ambiguities and alternatives.
- Multi-section and multi-statement documents.
- Unicode terminology, punctuation, casing, whitespace, and boundary cases.
- Compositional holdouts such as negation plus quantity plus condition.

### Artifact contract

Each released dataset version includes:

- `dataset-card.md`
- `license-inventory.json`
- `source-construction.json`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `adversarial.jsonl`
- `feature-coverage.json`
- `manifest.json`
- SHA-256 checksums for every file

### Acceptance gates

- Every schema field and supported section/statement kind has positive and negative coverage.
- Split generation is deterministic.
- No document or normalized source duplicate crosses splits.
- Test compositions are not duplicated in training.
- Every record round-trips through schema validation, realization, symbolization, and validation.
- Corpus reconstruction produces the published manifest hash.

## Phase 2: Reproducible training

Status: in progress. Strict versioned configurations now cover both architectures, including full
artifact identities, seeds, batching, optimizer values, token limits, and architecture-specific
settings. A race-resistant release reader pins the manifest and train/validation hashes, validates
the complete release, and rebuilds every deterministic training target before exposing immutable
split records. Both tracks now include deterministic offline two-step CPU mechanics trainers,
full-corpus tokenizer and overflow preflight, atomic safetensors-only output, runtime-derived run
manifests, canonical content-bound bundle manifests, hardened standalone preflight, reload
evaluation, CLI, CI, and installed-wheel coverage. Safe resume state, public model selection,
measured quality-bearing reference runs, and publication of model bytes remain. A
decision-independent dual-architecture mechanics release now verifies both local loaders, records
canonical prediction or rejection hashes, requires exact license declarations, writes model
cards, and reproduces byte for byte without embedding weights.

### Shared deliverables

- Versioned training configuration schema.
- Deterministic dataset reader pinned to one corpus generation.
- Seed control and run manifest.
- Checkpoint, optimizer-state, resume, and evaluation hooks.
- Safetensors-only model output.
- CPU two-step smoke configuration for CI.
- Documented single-GPU reference configuration.
- Dependency constraints or lock files for experiment reproduction.

### Encoder-decoder track

- Training entry point for canonical serialized IR to exact symbolic plan.
- Pinned public base-model repository and full commit digest.
- Tokenizer compatibility preflight against the entire dataset symbol inventory.
- Full fine-tuning or parameter-efficient configuration with recorded rationale.
- Published checkpoint and model card.

### Decoder-only LoRA track

- Versioned canonical prompt/segmented-target construction shared by training and inference.
- Pinned causal base model and full commit digest.
- PEFT LoRA configuration with parameter count and target modules.
- Prompt masking and exact EOS training behavior.
- Published adapter safetensors, adapter configuration, and model card.

### Run manifest

Every run records:

- package commit
- corpus generation ID and file hashes
- base model and tokenizer identities
- adapter identity when applicable
- dependency versions
- seed
- hyperparameters
- parameter counts
- hardware
- duration
- peak memory
- output hashes
- evaluation command

### Acceptance gates

- Both smoke-training jobs complete in CI without network access after fixture preparation.
- A documented reference run completes on the stated hardware.
- Resuming a run preserves dataset identity and configuration.
- Published checkpoints load through the package's production safety boundaries.
- Repeating evaluation from a published checkpoint produces the released prediction hashes.

## Phase 3: Neural CLI integration

Status: in progress. Strict `ste-realizer-config-v1` files now select deterministic,
encoder-decoder, and decoder-only LoRA realization for `compile` and the offline replay
`compile-source` workflow. Neural CLI inference is cache-only and retains immutable Hub commit
identities in provenance. Standalone artifact preflight now verifies complete local trainer
outputs against externally retained digests. Additive local-bundle configurations now load both
trainer architectures from exact private captures; the decoder path separately binds its base
snapshot. The dual-architecture mechanics release now emits immutable local configurations,
model cards, exact prediction hashes, and reproducible verification commands. Explicit fetch,
public base selection, artifact hosting, and quality-bearing reference configurations remain.

### Deliverables

- Realizer selection for deterministic, encoder-decoder, and decoder-only LoRA modes.
- Typed model configuration files rather than a long collection of CLI flags.
- Offline/cache-only and explicit-download modes.
- Preflight commands for model identity, tokenizer compatibility, and artifact safety.
- Structured handling of unavailable dependencies, missing caches, and incompatible tokenizers.
- Provenance metadata including all model and adapter revisions.

### Acceptance gates

- The same IR runs through all three realization modes.
- Neural output never bypasses symbol allowlisting, lexicalization, alignment, or validators.
- Constrained inference rejects unauthorized, incomplete, malformed, or semantically unsupported
  output.
- Network access is never implicit in offline mode.

The initial selection slice does not make a model-quality claim. Checked-in neural identities are
illustrative schema examples until reference checkpoints, licenses, checksums, predictions, and
benchmark reports are published.

## Phase 4: End-to-end frontends

### Offline replay frontend

- Remains the stable, credential-free reference workflow.
- Uses gold IR fixtures but exercises schema, provenance, realization, and validation boundaries.
- Is clearly labeled as replay rather than extraction.

### Live structured frontend

- At least one optional provider adapter with structured-schema output.
- Provider/model identity and version recorded in metadata.
- Bounded retries with machine-readable feedback.
- Exact source-span verification after every proposal.
- Rate-limit, timeout, malformed response, and provider-unavailable handling.
- No provider-produced final prose.

### Acceptance gates

- Live and replay frontends produce the same gold IR on the reference examples or report explicit
  differences.
- Invalid spans, invented quotes, missing required nodes, and unresolved ambiguity fail closed.
- Provider credentials are optional and never required for package import or offline examples.

## Phase 5: Evaluation and evidence

Status: scaffolded. A frozen, hash-bound v1 benchmark specification, raw prediction schema,
prediction manifest, failure taxonomy, recomputable metrics with Wilson intervals, deterministic
Markdown/JSON reporting, and frontend/realizer/validator failure fixtures are implemented. The
checked-in evidence is explicitly deterministic fixture evidence only. External measured model
runs, constrained/unconstrained ablations, resource measurements, human review, and publishable
model-quality claims remain.

### Systems

- direct unconstrained prose baseline
- prompted controlled-prose baseline
- deterministic compiler
- untrained base model
- trained unconstrained encoder-decoder
- trained constrained encoder-decoder
- trained unconstrained decoder-only adapter
- trained constrained decoder-only adapter
- complete source-to-text workflows for replay and live frontends

### Metrics

Frontend:

- schema-valid rate
- required-field precision, recall, and F1
- hallucinated-node rate
- ambiguity preservation
- source-span exact and overlap scores

Realizer:

- exact symbolic-plan accuracy
- grammar-valid rate
- EOS completion rate
- unauthorized-symbol rate
- constraint rejection rate

Semantic and end-to-end:

- required-node coverage
- negation, quantity, condition, temporal, causal, and hazard preservation
- vocabulary and terminology compliance
- accepted, rejected, and false-accept rates
- provenance coverage
- deterministic repeatability
- latency, throughput, peak memory, and artifact size

Human review:

- clarity
- semantic fidelity
- usefulness of rejection diagnostics

### Deliverables

- Frozen benchmark specification.
- Raw per-example prediction records.
- Machine-readable metrics and confidence intervals.
- Markdown report.
- Constrained/unconstrained ablations.
- Error taxonomy and uncensored failure examples.
- Hardware and dependency disclosure.

### Acceptance gates

- Every reported number can be recomputed from released predictions.
- Benchmark code rejects mismatched dataset/model manifests.
- Results distinguish frontend, realization, and validator failures.
- Claims are limited to the released benchmark and remain explicitly non-certified.

## Phase 6: Examples and documentation

### Executable examples

1. Raw source to deterministic controlled text.
2. Offline replay source extraction.
3. Optional live-provider extraction.
4. Custom vocabulary and terminology.
5. Dataset construction and manifest inspection.
6. Encoder-decoder smoke training.
7. Decoder-only LoRA smoke training.
8. Inference with each released checkpoint.
9. Constrained versus unconstrained comparison.
10. Provenance and alignment inspection.
11. Expected validation rejection.
12. Offline cached operation.
13. Benchmark reproduction.

Each implemented example has expected output and is executed in CI. Scenarios 3, 8, and 9 remain
explicitly gated until their provider, released-checkpoint, and comparison-protocol inputs exist.

The machine-readable inventory in [`examples/manifest.yaml`](../examples/manifest.yaml) records the
current command, fixtures, expected output, CI owner, and any unresolved release gate for every
scenario. See the [executable example catalog](executable-examples.md) for the tested offline subset
and the distinction between mechanics coverage and unreleased model or benchmark evidence.

### Documentation

- Installation and quick start.
- End-to-end tutorial.
- CLI reference.
- Python API reference.
- IR schema guide.
- Vocabulary and terminology authoring guide.
- Dataset card and construction guide.
- Training and checkpoint guide.
- Evaluation and result-reproduction guide.
- Threat model and artifact trust policy.
- Troubleshooting and hardware guide.
- Limitations, licensing, and non-compliance statement.
- Contributor and release documentation.

## Phase 7: Test and release hardening

### CI matrix

- Python 3.12, 3.13, and 3.14.
- Minimum and current supported neural dependency sets.
- Wheel and sdist installation.
- Linux plus explicit macOS/Windows coverage or documented platform exclusions.
- Core offline tests on every change.
- Tiny neural training/inference tests on every change.
- Published artifact verification and full examples in scheduled CI.

### Quality gates

- Ruff formatting and lint.
- Strict mypy.
- Maintained line and branch coverage thresholds. The initial offline all-extras gate enforces 88%
  line and 76% branch coverage from exact coverage.py counts on Python 3.12; floor increases remain
  follow-up hardening.
- Property tests for schemas, symbols, and corpus invariants.
- Adversarial tests for paths, symlinks, artifact corruption, malformed model output, and provider
  failures.
- Dependency vulnerability and license checks.
- Reproducible build and artifact provenance checks.

Status: the schema, causal-graph, exact-symbolic-plan, demonstration-corpus V2 integrity, malformed
provider-output, and provider transport-failure slice is implemented in `tests/property/`.
`LLMFrontend` currently retries only schema/provenance failures; provider exceptions propagate
after one call, so transport retry and redaction policies remain responsibilities of a future live
provider adapter. Exact-ratio coverage thresholds, a hash-locked five-profile vulnerability and
license-policy matrix, the Linux/macOS/Windows portable distribution matrix, lowest-direct/current
all-extras resolution, and weekly checked-in artifact/example verification are implemented.
A least-privilege signed-tag/manual-dry-run provenance workflow also emits canonical checksums and
an SPDX SBOM. It now builds exact Corpus V2 dataset and pipeline-fixture report candidates, binds
them into the release manifest and checksums, and has its read-only trusted verifier rebuild,
re-verify, and byte-compare both archives using default-branch code over the exact release source.
The privileged job retains distribution provenance and SPDX attestations, adds candidate-only
build provenance, and has no checkout or shell execution. External published-model verification,
release-signer authorization, authorized GitHub release attachment, and trusted package-index
publishing remain closed Phase 7 gates.

### Open-source release deliverables

- Complete package metadata and project URLs.
- Typed-package marker.
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `CITATION.cff`
- Release and compatibility policy.
- Trusted package-index publishing.
- Signed release artifacts and build provenance.
- Versioned GitHub release containing the small reference dataset, reports, and checksums.
- External immutable hosting for larger datasets and model checkpoints.

### Final release gates

- A clean environment can install the published wheel and run the offline reference example.
- Every documented command is tested.
- Dataset, model, report, and package identities are mutually linked.
- Both model architectures have reproducible released results.
- CI and scheduled artifact verification are green.
- No unresolved high-severity security, licensing, or reproducibility issue remains.

## Sequencing and dependencies

1. Phase 0 fixes the contract all later work targets.
2. Phase 1 must precede meaningful training and benchmark claims.
3. Phase 2 produces artifacts required by Phase 3 and Phase 5.
4. Phase 3 and Phase 4 complete the runnable product path.
5. Phase 5 determines whether the concept is demonstrated empirically.
6. Phase 6 turns the implementation into a usable open-source project.
7. Phase 7 is the release gate, not a cleanup phase deferred until the end; its tests should be
   added continuously.

## Progress tracking

Each phase should be represented by one tracking issue with:

- scoped deliverables
- artifact schema/version changes
- acceptance gates copied from this document
- required tests
- documentation updates
- released hashes or URLs
- unresolved risks

No phase is complete because code exists; it is complete only when its acceptance gates are
automated or backed by a versioned, inspectable artifact.

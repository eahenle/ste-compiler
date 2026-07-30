# Implementation plan

This file records the original prototype milestones. The complete package, dataset, model,
evaluation, documentation, and release roadmap is in
[V1 end-to-end implementation plan](v1-implementation-plan.md).

1. Confirm the public ASD-STE100 edition and document licensing boundaries; use only an original, permissively licensed demonstration vocabulary.
2. Define the typed semantic IR, safe serialization, versioned terminology and vocabulary registries, and machine-readable diagnostics.
3. Build a deterministic sentence planner/realizer, symbolic constrained lexicalizer, and lexical, structural, and metadata-backed semantic validators.
4. Expose manual and provider-neutral LLM frontends plus CLI workflows for validation, realization, compilation, glossary checks, and evaluation.
5. Add five valid and two invalid examples, reproducible baseline data, JSON/Markdown evaluation reports, and comprehensive offline tests.
6. Document architecture, assumptions, security boundaries, neural/LoRA extension paths, limitations, and experiment design; run formatting, typing, linting, and tests.

Release hardening now includes a deterministic offline coverage gate, dependency vulnerability and
license policy, signed-tag/manual-dry-run release provenance, Linux/macOS/Windows
installed-distribution coverage, lowest-direct/current all-extras dependency resolution, and
weekly checked-in artifact/example verification. The Python 3.12 all-extras suite must maintain at
least 88% line and 76% branch coverage; see [`testing.md`](testing.md). External artifact
publication, release-signer authorization, and trusted package publication remain tracked in the
[V1 end-to-end implementation plan](v1-implementation-plan.md).

## Milestone two

1. Export canonical IR-to-symbol training records from the deterministic realizer.
2. Define the provider-neutral `SymbolGenerator` inference boundary and enforce a document-specific symbol allowlist.
3. Independently align lexicalized model output before granting IR mappings.
4. Export deterministic bulk JSONL with duplicate-ID checks and a SHA-256 reproducibility manifest.
5. Add a concrete pinned model adapter, synthetic source construction, constrained decoding, and offline evaluation as separate follow-up slices.

# Implementation plan

1. Confirm the public ASD-STE100 edition and document licensing boundaries; use only an original, permissively licensed demonstration vocabulary.
2. Define the typed semantic IR, safe serialization, versioned terminology and vocabulary registries, and machine-readable diagnostics.
3. Build a deterministic sentence planner/realizer, symbolic constrained lexicalizer, and lexical, structural, and metadata-backed semantic validators.
4. Expose manual and provider-neutral LLM frontends plus CLI workflows for validation, realization, compilation, glossary checks, and evaluation.
5. Add five valid and two invalid examples, reproducible baseline data, JSON/Markdown evaluation reports, and comprehensive offline tests.
6. Document architecture, assumptions, security boundaries, neural/LoRA extension paths, limitations, and experiment design; run formatting, typing, linting, and tests.

## Milestone two

1. Export canonical IR-to-symbol training records from the deterministic realizer.
2. Define the provider-neutral `SymbolGenerator` inference boundary and enforce a document-specific symbol allowlist.
3. Independently align lexicalized model output before granting IR mappings.
4. Add a concrete pinned model adapter, reproducible synthetic corpus construction, constrained decoding, and offline evaluation as separate follow-up slices.

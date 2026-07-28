# Architecture

The manual frontend safely loads YAML/JSON into strict Pydantic models. The optional provider-neutral LLM frontend requests a `Document`, validates it, requires quoted provenance spans, feeds validation errors back for bounded retries, and never returns surface prose.

The IR explicitly represents actors, actions, referents, conditions/exceptions, negation, before/after relations, quantities/units/tolerances, hazards, provenance, references, causal links, and unresolved ambiguity. Section kinds distinguish procedure, description, warning, caution, and note.

The deterministic realizer composes clauses and records the complete source-node feature snapshot for every sentence. Lexical validation classifies canonical terminology before individual words, then accepts general words, units, numbers, and punctuation. Structural checks are deliberately small STE-inspired heuristics. Semantic checks use mappings plus independent text checks for high-risk features. Metadata is not enough for an untrusted neural backend: it must create a mapping that an independent aligner verifies. The parser-free CLI aligner grants mappings only to position-preserving exact matches of deterministic controlled sentences; changed, omitted, reordered, and extra sentences remain unmapped and are rejected.

## Constrained neural design

1. **Encoder-decoder:** encode canonical JSON IR and decode a symbolic sentence plan. This fits transformation tasks and can be compact, but needs task-specific training.
2. **Decoder-only + LoRA:** place canonical IR before the symbolic plan. Existing small models and adapters are convenient, but prompt handling and unsupported continuations increase risk.

Both must emit from a grammar of symbolic word, term, unit, punctuation, newline, and number IDs. `plan-symbols` creates deterministic training targets. At inference, `NeuralRealizer` supplies only the symbols present in the document's deterministic reference plan; this prevents invented words and quantities without trusting model metadata. `SymbolicLexicalizer` rejects out-of-plan symbols and copies canonical terms and units. The independent surface aligner then grants IR mappings only to position-preserving exact sentences, so reordering or omission still fails semantic validation.

A prefix trie or character-level finite-state machine can constrain symbol syntax in a concrete model adapter. Naive BPE masking cannot reliably express word boundaries because approved strings and model tokens are not one-to-one. The deterministic validator remains authoritative; LoRA changes probabilities, not guarantees.

## Trust and reproducibility

Source, YAML, glossary data, and model output are untrusted. There is no template execution or shell interpolation. Version metadata travels with every realization. Tests make no network calls. Current limitations include heuristic POS/voice analysis, no full discourse planner, and metadata-assisted rather than parser-backed semantic alignment.

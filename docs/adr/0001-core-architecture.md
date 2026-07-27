# ADR 0001: Typed IR and constrained realization

**Status:** accepted, 2026-07-27

## Decision

Use strict Pydantic semantic models, an original versioned vocabulary/terminology layer, deterministic realization first, symbolic constrained generation as the neural boundary, and layered deterministic validation.

## Assumptions and consequences

* Unknown meaning is represented as `Ambiguity`; frontends must not invent facts.
* Source spans are offsets into identified source material; LLM-extracted claims also require quotes.
* Canonical term copying occurs after planning.
* Demonstration rules are illustrative and incomplete. Passing them is not ASD-STE100 compliance.
* Alignment metadata is sufficient for this deterministic milestone but not a sole trust mechanism for future models.
* Warnings and cautions retain typed hazards even while the initial surface grammar remains intentionally small.


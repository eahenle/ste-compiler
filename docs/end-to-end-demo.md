# Offline end-to-end reference demo

The reference demo exercises the complete compiler boundary without credentials, network access,
or an unshipped model:

```text
packaged raw source
  -> replayed gold IR proposal
  -> strict Document schema validation
  -> exact source-span verification
  -> deterministic realization
  -> lexical, structural, and semantic validation
  -> controlled text plus audit metadata
```

Replay is a reproducibility tool, not an extraction result. Output metadata identifies the
frontend as `offline-replay`; it never represents the gold fixture as model-produced IR.

## Run the packaged example

After installing the package:

```bash
ste-compiler demo
ste-compiler demo --json
```

The text result is:

```text
Warning: injury can occur when hydraulic pressure is more than 20 MPa.
If hydraulic pressure is more than 20 MPa, stop the hydraulic pressure.
accepted
```

The JSON result uses the version marker `compile-source-v1` and includes:

- source ID and SHA-256
- schema-validated IR
- controlled text
- sentence-to-IR mappings
- validation status and diagnostics
- frontend, realizer, vocabulary, terminology, and validator identities

Print the formal JSON Schema for this result:

```bash
ste-compiler schema compile-source
```

## Run the explicit source/fixture pairs

From a source checkout:

```bash
ste-compiler compile-source \
  data/end_to_end/hydraulic_warning.txt \
  --ir-fixture data/end_to_end/hydraulic_warning.ir.yaml \
  --json
ste-compiler compile-source \
  data/end_to_end/test_multisection_state.txt \
  --ir-fixture data/end_to_end/test_multisection_state.ir.yaml \
  --json
ste-compiler compile-source \
  data/end_to_end/adversarial_reference_sequence.txt \
  --ir-fixture data/end_to_end/adversarial_reference_sequence.ir.yaml \
  --json
```

| Pair | Demonstrated boundary | Source SHA-256 |
|---|---|---|
| `hydraulic_warning` | condition, hazard, threshold, and warning realization | `b4c9933386cfff902e7fbd087ff95031993ab4dcbfb870fb92ce722ba8579f37` |
| `test_multisection_state` | description/procedure sections, quantity state, string state, and manner | `7f5f570480a7ead8e59023ded56a0f4affc8d4e5c8214a0baf664f4f0444680e` |
| `adversarial_reference_sequence` | temporal sequence, reference metadata, causal relation, and cause/effect mappings | `76b3e3a5d7fee55ec600888ebc1a56376fae9ab1d6dc178140903713b17382bd` |

The multi-section and reference/causal pairs are copied verbatim from the MIT-licensed frozen
Corpus V2 records with the same IDs. Their tests require exact source bytes and IR equality to the
corpus release as well as exact controlled text, mappings, provenance, metadata, and validation.

Each versioned replay fixture includes the SHA-256 of the complete source file. Its `source_spans`
must identify the source file; every span must be within the raw source and its `quote` must exactly
equal the source substring at `[start:end]`. Changing any source byte—including text outside all
represented spans—without updating and re-reviewing the gold IR therefore fails closed:

```text
replay IR source SHA-256 does not match the fixture
```

Use `--source-id` when the logical source identity is intentionally different from the filename.

## Trust boundary

The replay provider binds its untrusted object to the complete source digest. `LLMFrontend` then
applies the `Document` schema, requires provenance for each statement, validates ambiguity spans,
verifies exact source identity and text, and overwrites any frontend identity claimed by the
proposal. Only then can realization and validation run.

An optional live structured provider remains behind the same boundary. These offline pairs remain
the stable regression and documentation path.

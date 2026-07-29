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

## Run an explicit source/fixture pair

From a source checkout:

```bash
ste-compiler compile-source \
  data/end_to_end/hydraulic_warning.txt \
  --ir-fixture data/end_to_end/hydraulic_warning.ir.yaml \
  --json
```

The fixture's `source_spans` must identify `hydraulic_warning.txt`. Every span must be within the
raw source and its `quote` must exactly equal the source substring at `[start:end]`. Changing the
source without updating and re-reviewing the gold IR therefore fails closed:

```text
source span 0:79 quote does not match the source
```

Use `--source-id` when the logical source identity is intentionally different from the filename.

## Trust boundary

The replay provider returns an untrusted object. `LLMFrontend` applies the `Document` schema,
requires provenance for each statement, validates ambiguity spans, verifies exact source identity
and text, and overwrites any frontend identity claimed by the proposal. Only then can realization
and validation run.

The next frontend milestone adds an optional live structured provider behind the same boundary.
The offline demo will remain the stable regression and documentation path.

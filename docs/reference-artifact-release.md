# Dual-architecture reference artifact releases

`build-reference-release` turns one encoder-decoder trainer bundle, one decoder-only LoRA trainer
bundle, its separately content-bound causal-model snapshot, and their common corpus into a small,
portable metadata release. It runs every frozen test and adversarial record through the production
content-addressed local loaders. The output contains immutable realizer configurations, canonical
prediction or rejection records, prediction hashes, model cards, checksums, and one externally
retainable release-manifest identity.

Model weights are deliberately not copied into this release. The trainer bundles and decoder base
snapshot remain separately hosted artifacts identified by their manifest SHA-256 values. This
keeps the repository and wheel small and makes a future hosting decision independent of the
release format.

## License authorization is explicit

The builder requires canonical `ste-reference-release-metadata-v1` JSON. It binds each exact base
repository and commit to an operator-reviewed origin, base license, and derived-artifact license.
It refuses a declaration whose identity differs from the trainer run manifest. It never guesses a
license from a repository ID.

[`synthetic-mechanics-metadata.json`](../data/reference-release/synthetic-mechanics-metadata.json)
is valid only for artifacts made from the repository's generated synthetic fixtures. Do not reuse
its MIT declaration for a public base model. A public reference run needs a new reviewed file that
names the selected immutable base commits and their actual licenses.

## Build

First create both trainer outputs by following the architecture commands in
[the training guide](training.md). Retain the reported encoder and decoder
`artifact-manifest.json` digests and the decoder fixture's model-snapshot manifest digest. Then
run:

```bash
ste-compiler build-reference-release \
  data/reference-release/synthetic-mechanics-metadata.json \
  datasets/demonstration-corpus-1 \
  path/to/encoder-bundle <encoder-artifact-manifest-sha256> \
  path/to/decoder-bundle <decoder-artifact-manifest-sha256> \
  path/to/decoder-model-snapshot <snapshot-manifest-sha256> \
  path/to/new-reference-release \
  --json
```

The output directory must not exist. The command:

1. privately captures and verifies each complete trainer bundle by its external digest;
2. requires both run manifests to bind the same exact corpus;
3. checks the authorization file against both base-model identities;
4. verifies the decoder snapshot and its cross-link from the LoRA run;
5. creates strict local-bundle runtime configurations;
6. runs the test and adversarial splits through both production local loaders;
7. records accepted outputs, validator rejections, or bounded constrained-generation errors;
8. writes model cards that state `mechanics-smoke` use and prohibit quality or compliance claims;
9. writes canonical checksums and a content-bound release manifest; and
10. reports the release-manifest SHA-256 that must be retained outside the directory.

The prediction files are evidence of exact mechanics behavior only. A rejection is a first-class
prediction record, not omitted data. Acceptance counts from the synthetic two-step trainers are
not benchmark results and must not be used as model-quality evidence.

## Verify and reproduce

Verify the release bytes without loading neural dependencies:

```bash
ste-compiler verify-reference-release \
  path/to/reference-release \
  <release-manifest-sha256> \
  --json
```

Regenerate every prediction through both exact local loaders and require byte identity:

```bash
ste-compiler verify-reference-release \
  path/to/reference-release \
  <release-manifest-sha256> \
  --regenerate \
  --corpus-release datasets/demonstration-corpus-1 \
  --encoder-bundle path/to/encoder-bundle \
  --decoder-bundle path/to/decoder-bundle \
  --decoder-model-snapshot path/to/decoder-model-snapshot \
  --json
```

The verifier opens the release directory without following symlinks, authenticates the manifest
against the external digest before trusting its inventory, bounds descriptor-relative reads,
requires single-link regular files, verifies exact hashes and sizes, and checks the canonical
checksum inventory. Regeneration rebuilds into a temporary directory and compares every byte.

## Output contract

- `release-manifest.json`: canonical dual-architecture identity and complete file inventory.
- `release-metadata.json`: exact reviewed origin and license declarations.
- `*-realizer.json`: portable local-bundle configurations with no machine-specific paths.
- `*-predictions.jsonl`: canonical frozen evaluation predictions and rejections.
- `*-model-card.md`: intended use, limitations, immutable identities, origin, and licenses.
- `checksums.sha256`: canonical checksums for every other non-manifest release file.
- `REPRODUCE.md`: digest-pinned command templates suitable for an attached release.

Print the formal manifest schema with:

```bash
ste-compiler schema reference-release
ste-compiler schema reference-release-metadata
ste-compiler schema reference-prediction
```

## Decisions still required for a public reference release

This workflow completes the decision-independent mechanics and provenance path. Publishing a
quality-bearing reference result still requires maintainers to choose:

1. one encoder-decoder base repository and full commit with a redistribution-compatible license;
2. one causal base repository and full commit with a redistribution-compatible license; and
3. an immutable hosting target for the two trainer bundles and decoder base snapshot.

After those choices, create a new authorization file, run the documented training and release
commands, attach the externally retained digests to the hosting record, and perform the separate
benchmark and failure-analysis phase. The mechanics release itself makes no quality claim.

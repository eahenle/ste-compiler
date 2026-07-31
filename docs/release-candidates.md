# Core release candidates

The core release-candidate builder packages the checked demonstration dataset and deterministic
fixture benchmark evidence without changing their bytes. It is an internal release primitive, not
a general-purpose tar utility and not a publication step.

Given a validated `ReleaseIdentity`, one build creates exactly:

- `ste-compiler-{version}-dataset-demonstration-corpus-2.tar`
- `ste-compiler-{version}-report-ste-compiler-pipeline-fixture-1.tar`

The archives are uncompressed POSIX USTAR. Every archive has one explicit top-level directory named
after the archive, without the `.tar` suffix. Files are ordered by their UTF-8 encoded POSIX paths.
Directory and file modes are respectively `0755` and `0644`; modification times equal the release
identity's `source_date_epoch`; numeric owners are zero; and owner and group names are empty.
Filesystem mtimes, owners, checkout locations, and temporary paths therefore do not affect bytes.

## Build contract

The source root is bound to the supplied identity before any candidate input is read:

- it is the real Git worktree root, without symlink ancestors;
- `HEAD`, the commit timestamp, the project version, and `CITATION.cff` match the identity;
- tracked and untracked status is clean; and
- every normative input is a mode-`100644` blob in the identity commit, its hardened working-tree
  read matches that blob byte-for-byte, and reconstruction uses the captured blob-matching bytes;
  and
- every normative directory remains inside the root without symlink components, with an exact
  expected inventory, while normative files are regular and have one hard link.

The dataset candidate is rebuilt from
`data/demonstration_corpus/v2/source-construction.json` and the reviewed resources. Both the rebuilt
directory and `datasets/demonstration-corpus-2` pass the corpus verifier, and every rebuilt byte must
equal its checked counterpart.

The report candidate includes the four files under `data/benchmark/v1` that define predictions and
their contract, plus a freshly generated `metrics.json`, `report.md`, and `report-manifest.json`.
Generation uses the freshly rebuilt dataset. Every generated report byte must equal
`data/benchmark/v1/expected-report`.

The destination must not exist, its path must not contain symbolic-link ancestors, and its existing
real parent must be outside the source worktree. Private stages are created through an open
descriptor for that parent, and the parent identity is checked before and after publication.
Publication uses an atomic platform no-replace operation, keeps the renamed stage descriptor open,
and requires the final output name to resolve to that exact directory before returning. A
concurrently created or substituted destination is preserved and never removed as cleanup for the
private stage. Once the atomic rename succeeds, exceptional concurrent path changes do not trigger
a partial name-based rollback: the completed staged directory remains intact rather than becoming
an empty destination that blocks a safe retry.

Library callers use:

```python
from scripts.release.release_candidates import build_candidate_directory

dataset_tar, report_tar = build_candidate_directory(source_root, identity, output)
```

The internal command consumes the canonical identity file produced by the release-ref validator:

```bash
python -m scripts.release.release_candidates build \
  --source-root . \
  --identity release-identity.json \
  --output release-candidates
```

## Identity graph

Each top-level directory contains a canonical `release-candidate.json` with:

- the complete outer `ReleaseIdentity`, including dry-run or tag mode;
- the archive name, artifact kind and ID, and fixed top-level directory;
- a sorted size and SHA-256 identity for every payload file;
- a `content_identity` naming the authoritative inner manifest, its schema, and its SHA-256; and
- dependencies.

For the dataset, the authoritative content manifest is `manifest.json` with schema
`demonstration-corpus-release-v1`, and dependencies are empty.

For the report, the authoritative content manifest is `report-manifest.json` with schema
`ste-benchmark-report-manifest-v1`. Its one dependency names the dataset archive and binds:

- the exact dataset archive SHA-256;
- the dataset candidate-manifest SHA-256; and
- the Corpus V2 `manifest.json` SHA-256.

This makes the report-to-dataset edge explicit at the outer archive, candidate-manifest, and corpus
release levels.

## Verification and hostile-input policy

Verify both siblings together:

```python
from scripts.release.release_candidates import verify_candidate_directory

dataset_tar, report_tar = verify_candidate_directory(path, identity)
```

or:

```bash
python -m scripts.release.release_candidates verify \
  --path release-candidates \
  --identity release-identity.json
```

Verification treats the directory, tar bytes, manifests, and payloads as untrusted. It does not call
`extract()` or `extractall()`. It bounds archive size, member size, and member count before reading
payload bytes or materializing an unbounded header inventory. Candidate files are opened once with
no-follow and nonblocking semantics, their descriptor identity is stable across the bounded read,
and the verified dataset digest is reused for the report dependency edge. The verifier accepts only
canonical regular-file and explicit-directory USTAR members; and rejects links, devices, FIFOs,
sparse or unknown types, GNU and PAX extensions, unsafe or duplicate paths, base-256 and
noncanonical headers, metadata changes, truncation, nonzero padding, trailing bytes, concatenated
archives, missing members, and extras.

After structural and hash verification, the verifier materializes only validated relative payload
paths in a private temporary directory. It reconstructs Corpus V2 again, regenerates benchmark
evidence against that corpus, requires byte identity, and validates the report's dataset
cross-link. Verification with a different dry-run/tag identity fails even when version and commit
otherwise match.

# Release build provenance

The `Release provenance` workflow prepares reviewable package artifacts without publishing them. It
has two modes:

- `workflow_dispatch` is a manual dry-run. It validates the selected commit, builds and independently
  verifies reproducible wheel and source distributions plus the exact dataset and fixture-report
  candidate archives, generates an SPDX JSON SBOM, writes canonical checksums and build metadata,
  and uploads a short-lived workflow artifact.
- A `push` of a `vMAJOR.MINOR.PATCH` tag follows the same build path, but only after the tag is an
  annotated SSH-signed tag for the exact checkout commit, its version equals both `pyproject.toml`
  and `CITATION.cff`, the tag points to the exact current reviewed default-branch commit, and Git
  accepts its signer through the allowlist fetched from that commit. The tag workflow has
  read-only repository access and can only upload a verification bundle. A separate `workflow_run`
  workflow, loaded from the default branch rather than the tag, independently verifies that bundle,
  source commit, tag signature, and default-branch signer policy. It then uses trusted scripts to
  reproduce the distributions from the reviewed source and rebuilds the candidates with trusted
  default-branch code reading the exact validated release source as data. It requires both sets of
  bytes to match the upstream build, creates a fresh SBOM and canonical evidence bundle, and only
  then lets GitHub create distribution provenance, distribution SPDX SBOM, and candidate-only
  build-provenance attestations.

The workflow does not create a tag, GitHub Release, package-index upload, model release, or dataset
release. The uploaded workflow artifact is verification evidence, not a public package release.

## Closed signed-tag gate

The signer allowlist at `.github/release/trusted-tag-signers` intentionally contains no key. The
read-only tag build fetches this policy from the current default branch for an early rejection; it
never trusts the copy in the tagged checkout. More importantly, the privileged `Release
attestation` workflow executes its validator and reads its policy from its own default-branch
checkout. It treats the triggering bundle and tagged source as untrusted inputs, verifies their
canonical inventory and checksums, requires the tagged commit to equal its default-branch commit,
and verifies the tag again. The exact commit equality makes GitHub's signed build-provenance
workflow SHA identical to the source commit that produced the distributions. Therefore signed-tag
workflow runs fail closed before attestation until the release owner explicitly reviews a release
signing identity and commits an SSH `allowed_signers` entry:

```text
release-identity@example.com namespaces="git" ssh-ed25519 AAAA...
```

This is a project decision, not a value for automation to invent. Adding a key authorizes that
identity to satisfy the repository's signed-tag build gate; it does not authorize package
publication. The workflow forces Git's SSH signature verifier to use only this file.

After signer authorization, a release tag must still:

1. use stable `vMAJOR.MINOR.PATCH` syntax without a moving branch, lightweight tag, or prerelease;
2. equal the project and citation version exactly;
3. point to the exact clean commit currently at the tip of the reviewed default branch;
4. carry a valid annotated SSH signature from the allowlist.

The strict validation implementation is
[`scripts/release/release_contract.py`](../scripts/release/release_contract.py). It writes a
`ste-release-build-identity-v1` record before the build. After SBOM generation, the same script
writes `release-build.json` and `SHA256SUMS`. Before finalization, the offline candidate builder
creates exactly:

- `ste-compiler-<version>-dataset-demonstration-corpus-2.tar`
- `ste-compiler-<version>-report-ste-compiler-pipeline-fixture-1.tar`

The candidate verifier binds both archives to that identity and verifies their report-to-corpus
cross-links. The release contract refuses missing, extra, renamed, symlinked, hard-linked,
malformed, identity-mismatched, cross-link-mismatched, or tampered candidates. The manifest
deterministically inventories both candidates, both distributions, and the SBOM; the checksum file
additionally binds that manifest without recursively attempting to checksum itself. Finalization
and verification keep no-follow descriptors open for every single-link regular subject, stream
each subject into one size-and-digest snapshot, and recheck both descriptor and path identity before
accepting the result. Candidate archives are size-bounded and captured as part of those same
snapshots; semantic verification runs against private materializations of the captured bytes rather
than reopening their original paths. Candidate validation, manifest construction, and
`SHA256SUMS` therefore cannot silently refer to different path states.

## Manual dry-run

Use GitHub's workflow UI or:

```bash
gh workflow run release-provenance.yml --ref <reviewed-branch-or-tag>
```

A manual workflow ref must be a branch or tag whose tip is the exact commit to rehearse; the
`gh workflow run --ref` option does not accept a raw commit SHA.

A manual run never claims a tag and skips the attestation job. After this workflow is present on the
default branch, its successful run triggers the read-only default-branch verifier, including the
cross-run download, strict bundle validation, trusted rebuild, byte comparison, SBOM generation,
and canonical finalization. It is suitable for checking action compatibility and inspecting the
prospective wheel, source distribution, dataset candidate, fixture-report candidate, SBOM,
manifest, and checksums. It is not evidence that the tag, signer, version, package-index
environment, or release notes were authorized.

The underlying distribution gate can also retain verified local artifacts in a new directory:

```bash
uv run --locked --extra dev \
  python scripts/ci/distribution_smoke.py --release-output ./verified-distributions
```

The output directory must not exist. The gate builds twice with the reviewed commit timestamp,
requires byte-identical distributions, rebuilds the wheel from the sdist, installs and executes it
outside the checkout under the network tripwire, and copies only the verified wheel and sdist.

## Workflow permissions and attestations

All actions are pinned to immutable commit SHAs. The build job inherits only `contents: read`, and
checkout does not persist credentials. SBOM generation disables dependency-snapshot, workflow
artifact, and release-asset side effects; the repository workflow uploads the complete evidence
bundle itself.

The tag-triggered workflow never receives an identity token or attestation permission. The
default-branch `workflow_run` first handles untrusted inputs and performs its rebuild in a
read-only job. A separate dependent job receives `id-token: write` and `attestations: write`; it can
only download the exact trusted artifact ID emitted by the verifier job and invoke pinned
attestation actions. Binding the handoff to the uploaded artifact ID also makes failed-job retries
reuse the evidence that actually passed verification. It has no checkout or shell step and does not
execute project code. GitHub uses those permissions to sign and store SLSA build provenance for
the distributions and candidates plus the distributions' SPDX SBOM attestation. Consumers can
verify a signed-tag candidate online with:

```bash
gh attestation verify \
  path/to/ste-compiler-<version>-dataset-demonstration-corpus-2.tar \
  --repo eahenle/ste-compiler
```

Distribution provenance and SPDX verification remain available with the same command forms
documented by GitHub; candidate provenance is a separate subject set and does not treat package
code as a candidate archive.

GitHub documents the attestation permissions, generation, and verification contract in
[Using artifact attestations to establish provenance for builds](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations).

## Trusted publishing remains a decision gate

There is deliberately no PyPI publisher in this repository. Before adding one, the release owner
must explicitly decide and review:

1. the public version and release commit;
2. the PyPI project and owner;
3. a protected GitHub environment and its required reviewers;
4. the exact PyPI trusted-publisher binding to this repository, workflow, and environment;
5. whether signed-tag provenance and SBOM attestations are mandatory publication inputs; and
6. the rollback and incident response for a compromised signer or publisher.

A future publisher should be a separate environment-protected job, use PyPI's short-lived OIDC
trusted publishing instead of a long-lived API token, download the exact verified artifact from the
same workflow, and refuse manual dry-runs. None of those future choices is implied or authorized by
the current build workflow.

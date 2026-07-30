# Release build provenance

The `Release provenance` workflow prepares reviewable package artifacts without publishing them. It
has two modes:

- `workflow_dispatch` is a manual dry-run. It validates the selected commit, builds and independently
  verifies reproducible wheel and source distributions, generates an SPDX JSON SBOM, writes canonical
  checksums and build metadata, and uploads a short-lived workflow artifact.
- A `push` of a `vMAJOR.MINOR.PATCH` tag follows the same build path, but only after the tag is an
  annotated SSH-signed tag for the exact checkout commit, its version equals both `pyproject.toml`
  and `CITATION.cff`, and Git accepts its signer through the reviewed repository allowlist. A
  successful signed-tag build also asks GitHub to create build-provenance and SPDX SBOM attestations
  for the wheel and source distribution.

The workflow does not create a tag, GitHub Release, package-index upload, model release, or dataset
release. The uploaded workflow artifact is verification evidence, not a public package release.

## Closed signed-tag gate

The signer allowlist at `.github/release/trusted-tag-signers` intentionally contains no key.
Therefore signed-tag workflow runs fail closed before building until the release owner explicitly
reviews a release signing identity and commits an SSH `allowed_signers` entry:

```text
release-identity@example.com namespaces="git" ssh-ed25519 AAAA...
```

This is a project decision, not a value for automation to invent. Adding a key authorizes that
identity to satisfy the repository's signed-tag build gate; it does not authorize package
publication. The workflow forces Git's SSH signature verifier to use only this file.

After signer authorization, a release tag must still:

1. use stable `vMAJOR.MINOR.PATCH` syntax without a moving branch, lightweight tag, or prerelease;
2. equal the project and citation version exactly;
3. point to the clean reviewed commit selected by the workflow; and
4. carry a valid annotated SSH signature from the allowlist.

The strict validation implementation is
[`scripts/release/release_contract.py`](../scripts/release/release_contract.py). It writes a
`ste-release-build-identity-v1` record before the build. After SBOM generation, the same script
writes `release-build.json` and `SHA256SUMS`. The manifest deterministically inventories the
distributions and SBOM; the checksum file additionally binds that manifest without recursively
attempting to checksum itself.

## Manual dry-run

Use GitHub's workflow UI or:

```bash
gh workflow run release-provenance.yml --ref <reviewed-branch-or-commit>
```

A manual run never claims a tag and skips the attestation job. It is suitable for checking action
compatibility and inspecting the prospective wheel, source distribution, SBOM, manifest, and
checksums. It is not evidence that the tag, signer, version, package-index environment, or release
notes were authorized.

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

Only the signed-tag attestation job receives `id-token: write` and `attestations: write`, plus
`contents: read`. GitHub uses those permissions to sign and store SLSA build-provenance and SPDX SBOM
attestations. Consumers can verify a signed-tag distribution online with:

```bash
gh attestation verify path/to/ste_compiler-<version>-py3-none-any.whl \
  --repo eahenle/ste-compiler

gh attestation verify path/to/ste_compiler-<version>-py3-none-any.whl \
  --repo eahenle/ste-compiler \
  --predicate-type https://spdx.dev/Document/v2.3
```

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

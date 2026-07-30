# Release and compatibility policy

`ste-compiler` uses Semantic Versioning. Before 1.0, minor releases can intentionally change public
Python APIs, CLI behavior, schemas, symbolic protocols, or artifact layouts. Such changes must be
called out in the changelog and use a new schema, protocol, dataset, or manifest version whenever
old and new artifacts could otherwise be confused.

Patch releases contain compatible bug, documentation, packaging, or security fixes. Released
dataset generations, model revisions, prediction files, and checksums are immutable. A correction
to released bytes produces a new artifact version instead of replacing the old identity.

## Supported environments

- Core package: Python 3.12, 3.13, and 3.14 on Linux. Python 3.12 installed-distribution and
  portable-catalog behavior is additionally exercised on macOS 14 and Windows Server 2022.
- Neural features: the explicitly locked dependency and hardware profile documented by the
  corresponding release. Current neural mechanics CI runs on Linux CPU; macOS and Windows neural
  training are not claimed as supported profiles.
- Hardened training-release and model-artifact bundle reads and publication: Linux POSIX
  filesystems with directory descriptors and no-follow support. The portable package imports and
  catalog run on Windows, but symbolic-corpus export and hardened artifact/report/reference-release
  publication remain explicit POSIX-only exclusions.

Optional providers and live network integrations are outside the credential-free core support
contract.

## Release gates

A public release requires:

1. formatting, lint, strict typing, tests, and installed-wheel smoke checks;
2. reproducible wheel and source-distribution builds, member inspection, and an outside-checkout
   offline execution smoke from a clean checkout;
3. an updated changelog and version-coherent citation metadata;
4. immutable checksums and provenance for included data, reports, and neural artifacts;
5. license and intended-use review; and
6. a clean, signed tag created from the reviewed release commit.

Package-index publication should use trusted publishing and attach build provenance. The current
[release build provenance workflow](release-build-provenance.md) deliberately stops before package
or GitHub Release publication: manual runs create only disposable verification artifacts, and the
signed-tag path is closed until a release signer is explicitly authorized. GitHub release assets
must include checksums. Model and dataset artifacts too large for the package must be linked by
immutable repository revision and digest. A model bundle publication must carry the externally
retained `artifact-manifest.json` SHA-256 in signed or otherwise reviewed release metadata; the
colocated manifest alone is not sufficient.

CI executes `scripts/ci/distribution_smoke.py` with the reviewed commit timestamp as
`SOURCE_DATE_EPOCH`. The gate builds wheel and source distributions twice, requires byte-identical
hashes, inspects required package and release members, installs the wheel outside the checkout with
network access blocked, rebuilds that wheel byte-for-byte from the generated source distribution,
runs the packaged demo, and reconstructs and verifies corpus version 2.

With `--release-output`, that same gate copies only the verified wheel and source distribution into
a new output directory. The release workflow then creates an SPDX JSON SBOM, a canonical
`ste-release-build-manifest-v1` inventory, and `SHA256SUMS`. GitHub build and SBOM attestations are
limited to a successfully verified signed-version-tag run; manual dry-runs do not receive release
attestations.

Compatibility CI repeats that installed-distribution boundary on Linux, macOS, and Windows and
tests both the lowest declared direct dependencies and the highest currently compatible
all-extras resolution. Weekly credential-free CI repeats installed-wheel catalog execution and
checked-in corpus, benchmark, and mechanics-release reconstruction. It does not contact an
external artifact host until the project publishes an immutable, licensed model locator and
digest.

## Deprecation

When practical, a public interface is deprecated for one minor release before removal. Security,
licensing, data-integrity, or unsafe-artifact defects can require immediate rejection without a
compatibility window.

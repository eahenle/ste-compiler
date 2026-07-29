# Release and compatibility policy

`ste-compiler` uses Semantic Versioning. Before 1.0, minor releases can intentionally change public
Python APIs, CLI behavior, schemas, symbolic protocols, or artifact layouts. Such changes must be
called out in the changelog and use a new schema, protocol, dataset, or manifest version whenever
old and new artifacts could otherwise be confused.

Patch releases contain compatible bug, documentation, packaging, or security fixes. Released
dataset generations, model revisions, prediction files, and checksums are immutable. A correction
to released bytes produces a new artifact version instead of replacing the old identity.

## Supported environments

- Core package: the Python versions listed in `pyproject.toml` and exercised by CI.
- Neural features: the explicitly locked dependency and hardware profile documented by the
  corresponding release.
- Hardened training-release directory reads: POSIX filesystems.

Optional providers and live network integrations are outside the credential-free core support
contract.

## Release gates

A public release requires:

1. formatting, lint, strict typing, tests, and installed-wheel smoke checks;
2. wheel and source-distribution inspection from a clean checkout;
3. an updated changelog and version-coherent citation metadata;
4. immutable checksums and provenance for included data, reports, and neural artifacts;
5. license and intended-use review; and
6. a clean, signed tag created from the reviewed release commit.

Package-index publication should use trusted publishing and attach build provenance. GitHub release
assets must include checksums. Model and dataset artifacts too large for the package must be linked
by immutable repository revision and digest.

## Deprecation

When practical, a public interface is deprecated for one minor release before removal. Security,
licensing, data-integrity, or unsafe-artifact defects can require immediate rejection without a
compatibility window.

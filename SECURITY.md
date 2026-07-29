# Security policy

## Supported versions

Security fixes are provided for the latest released minor version and the current `main` branch.
Pre-1.0 schemas and APIs can change, but confirmed vulnerabilities will receive a documented fix.

## Reporting a vulnerability

Do not open a public issue containing exploit details, sensitive source text, credentials, or
private model and dataset locations.

Use the repository's private GitHub security-advisory reporting flow when it is available:

<https://github.com/eahenle/ste-compiler/security/advisories/new>

If private reporting is unavailable, open a minimal public issue asking the maintainer to establish
private contact. Include no vulnerability details in that issue.

Please provide:

- the affected version or commit;
- the smallest reproducible example;
- expected and observed behavior;
- impact and realistic attack preconditions; and
- any proposed mitigation or disclosure deadline.

The maintainer will acknowledge a report when it is received, validate its scope, coordinate a fix
and release, and credit the reporter unless anonymity is requested. No guaranteed response time is
offered while the project is maintained by volunteers.

## Security boundaries

The project treats source documents, semantic-frontend output, neural symbols, configuration files,
dataset releases, and model artifacts as untrusted. Expected protections include strict schemas,
exact source provenance, allowlisted symbolic output, independent validation, immutable artifact
identities, offline model loading, and safetensors-only neural outputs.

This prototype is not certified for safety-critical use and does not claim ASD-STE100 compliance.
Passing its validators is not a substitute for technical review.

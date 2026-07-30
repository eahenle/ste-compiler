# Dependency vulnerability and license policy

The `Dependencies` CI matrix is the repository's fail-closed third-party dependency gate. It runs
on Python 3.12 and Ubuntu against five independently exported profiles:

- core runtime dependencies;
- the `dev` extra;
- the `neural` extra;
- the `encoder-training` extra; and
- the union of every optional extra.

Each profile comes directly from `uv.lock` as a fully pinned, hash-bearing requirements file.
`pip-audit==2.10.0` runs with dependency resolution disabled and queries the OSV vulnerability
service. OSV is used because the Linux lock selects PyTorch's `+cpu` local-version distribution,
which the PyPI advisory service cannot identify as a published PyPI release. The policy checker
requires the scanner exit status to agree with the JSON report, so a
missing, malformed, truncated, or contradictory report cannot be interpreted as clean.

The all-extras requirements are also installed by hash into a fresh target environment.
`pip-licenses==5.5.5` inventories that environment without including its own tool dependencies.
The installer declares both PyPI and the PyTorch CPU index needed for Linux's locked `torch` local
version. uv must consider matching versions across both indexes because `torch` also exists on
PyPI; this normally riskier multi-index strategy remains constrained here by exact versions and
mandatory artifact hashes exported from the reviewed lock.
The checker requires its canonical package/version set to equal the all-extras vulnerability
inventory exactly before it evaluates licenses. Both tools are exact direct dependencies in the
locked `dependency-audit` dependency group. GitHub Actions and `uv` are pinned by full commit or
exact version in the workflow.

Unlike the package tests, this dedicated job intentionally uses the network for current
vulnerability data and locked package downloads. It requires no credentials and does not exercise
live model or provider APIs.

## License policy

[`policy/dependency-audit-policy.json`](../policy/dependency-audit-policy.json) is authoritative.
The current allow list contains only the exact permissive or weak-copyleft expressions emitted for
the locked dependency union: MIT, BSD, Apache, ISC, MPL, PSF, and reviewed composite expressions.
GPL and AGPL markers are explicitly denied. An unknown expression, `UNKNOWN`, a package/version
inventory mismatch, or an unlisted expression fails closed.

The gate evaluates distribution metadata; it is not a legal opinion and does not replace source,
NOTICE, or model/dataset license review. When a lock update changes an expression, review the
upstream license files before updating the allow list.

## Suppressions and exceptions

There are no current vulnerability suppressions or license exceptions. A future entry must be
narrowly bound to:

- canonical package name;
- exact locked version;
- exact vulnerability ID or emitted license expression;
- a nonblank review rationale; and
- an ISO expiration date.

Expired entries fail. Entries that no longer match a report also fail, which prevents stale
blanket suppressions. Profile-specific vulnerability jobs ignore suppressions for packages outside
their own inventory; the `all` profile is the authoritative stale-suppression gate and requires
every configured suppression to match. A license exception can override a denial only for its
exact package/version/expression tuple. An exception for an already allowed expression is rejected.

Do not suppress a fixable vulnerability merely to restore CI. Prefer updating the lock. A
suppression is appropriate only when the advisory is demonstrably inapplicable or no safe version
exists, and its rationale must identify the compensating control and tracking issue.

## Local reproduction

The policy parser and adversarial fixtures are offline:

```bash
uv run --locked --extra dev pytest -q tests/unit/test_dependency_policy.py
```

The live audit requires network access. The CI commands are intentionally visible in
`.github/workflows/ci.yml`; use the same date and generated profile files when reproducing a
specific run. Scanner results vary as OSV changes even though dependency
inputs, tools, and policy are locked.

## Scope and limitations

The gate audits Python dependencies selected for Ubuntu and Python 3.12. Platform-specific
dependencies that are not selected there remain part of the planned macOS/Windows CI expansion.
The absence of a published advisory does not prove a package is safe. Package licenses do not
establish dataset, model, standards-content, or generated-artifact rights; those remain governed by
their separate provenance inventories and release gates.

# Testing, coverage, and adversarial boundaries

The default test suite is offline and deterministic. It covers the compiler core, command-line
interfaces, data-release reconstruction, executable examples, and the tiny local neural mechanics
fixtures. Tests are credential-free and must not download models, datasets, or tokenizer files
implicitly.

## Local gates

Run the same quality checks used by CI:

```bash
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev ruff check .
uv run --locked --extra dev mypy src
uv run --locked --extra dev pytest -q -m "not neural"
```

The dedicated Python 3.12 coverage job installs every optional dependency group and runs the
complete suite in offline mode:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --python 3.12 --all-extras \
  pytest -q --cov=ste_compiler --cov-branch --cov-report=json:coverage.json
uv run --locked --python 3.12 --all-extras \
  python scripts/ci/check_coverage.py coverage.json
```

The checked gate initially requires at least 88% line coverage and 76% branch coverage. The checker
derives both percentages from integer counts in coverage.py's JSON report and compares exact
ratios, so display rounding cannot turn a failing result into a pass. A report with missing,
negative, contradictory, or zero-denominator counts fails closed.

These floors are repository-wide regression barriers, not coverage targets for new code. Changes
should test their meaningful success, rejection, and failure paths even when the aggregate remains
above the floor. Raise the floors as coverage improves; do not lower them to accommodate an
untested change.

## CI ownership

- `Quality` checks formatting, lint, and strict typing.
- `Tests` runs the non-neural suite on Python 3.12, 3.13, and 3.14.
- `Coverage` runs every test and every extra on Python 3.12 with network-backed model access
  disabled, then enforces line and branch floors.
- `Dependencies` exports five hash-locked core/optional profiles, queries current vulnerability
  data, and reconciles an isolated all-extras license inventory against reviewed policy.
- Architecture-specific jobs retain focused offline neural smoke coverage.
- `Distribution smoke` proves reproducible wheel and source-distribution construction plus
  outside-checkout execution.

Coverage does not establish model quality, semantic correctness by itself, or suitability for a
production use case. Those claims require the repository's separate deterministic contracts and
versioned evaluation evidence.

Ordinary package tests remain network-free. The dedicated dependency matrix is the exception: it
uses the network without credentials to query OSV vulnerability data and download only
hash-locked distributions. See the
[dependency vulnerability and license policy](dependency-policy.md) for exact profiles,
fail-closed report checks, and suppression rules.

## Property and adversarial suite

The Phase 7 property suite can run independently:

```bash
uv run --locked --extra dev pytest -q tests/property
```

Hypothesis exercises bounded, deterministic invariants at the package's strictest public
boundaries. Each property explicitly enables Hypothesis's derandomized mode, which disables the
example database and derives a repeatable case sequence from the test definition:

- semantic IR JSON/YAML round trips, nested unknown-field rejection, and invalid causal-graph
  mutations;
- lossless exact-whitespace symbolic plans, document-specific allowlists, and mutated or
  unauthorized symbols;
- demonstration-corpus V2 snapshot identity plus tampered, truncated, and extra release entries;
  and
- malformed structured-provider proposals and provider transport failures.

These tests supplement the fixed regression fixtures. They do not replace corpus reconstruction,
installed-wheel, neural smoke, or distribution-reproducibility jobs.

## Provider failure contract

`LLMFrontend` retries only schema and source-provenance validation failures. With `retries=N`, it
makes at most `N + 1` proposal attempts and sends machine-readable validation feedback after the
first failure. Construction rejects booleans, non-integers, and negative retry limits.

Exceptions raised by the provider itself, including `ValueError`, `TimeoutError`, and
`ConnectionError`, are outside that retry boundary. The frontend makes one call and propagates the
same exception unchanged, regardless of its configured validation retries. A future live-provider
adapter can add a rate-limit, backoff, or transport retry policy, but the current provider-neutral
frontend does not invent one.

Because transport exceptions are propagated unchanged, provider adapters must emit sanitized
messages that contain no credentials or sensitive source text. The adversarial tests verify that
the frontend does not add source or provider credential state to sanitized transport failures.
They do not claim that the frontend can redact secrets already embedded in an exception created by
a provider.

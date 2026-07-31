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

If you use pre-commit locally, install and run it with:

```bash
uvx --from pre-commit pre-commit install
uvx --from pre-commit pre-commit run --all-files
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
- `Portable distribution` repeats the reproducible build, installed-wheel, network tripwire, and
  portable executable-catalog boundary on Ubuntu 24.04, macOS 14, and Windows Server 2022.
- `Dependency resolution` stages the commit into a disposable resolution-only project, deletes
  only that project's copy of the lock, resolves every extra with uv's `lowest-direct` and
  `highest` strategies, and runs the complete offline suite against each resulting environment
  while retaining the clean source checkout required by provenance tests. The committed lock
  remains the reproducible default; these two jobs test the declared lower bounds and current
  compatible releases independently. The uv project metadata requires an installable Linux x86_64
  resolution so `lowest-direct` selects PyTorch's compatible `+cpu` wheel instead of an
  unsuffixed wheel published only for another architecture; this resolver-platform contract
  preserves the tested `torch>=2.4,<3` dependency floor.
- `Scheduled artifact and example verification` runs every Monday at 09:17 UTC and on manual
  dispatch. It rebuilds and installs the distribution, runs the packaged portable catalog, and
  reconstructs the checked-in demonstration corpora, benchmark evidence, and mechanics reference
  releases.

## Platform scope

Linux is the full support gate: Python 3.12, 3.13, and 3.14 core tests, complete Python 3.12
all-extras coverage, both offline neural mechanics paths, hardened POSIX artifact publication and
reading, and distribution reproduction all run there.

macOS 14 and Windows Server 2022 run the Python 3.12 distribution and installed portable-catalog
gate. This proves package construction, installation outside the checkout, CLI and module entry
points, and packaged resources. macOS runs the default seven credential-free scenarios and eleven
commands, including the POSIX benchmark-reproduction scenario. Windows applies the manifest's
explicit `win32` override and runs the six portable scenarios and their nine commands. It does not
claim full neural training coverage on either platform. Windows also does not support the
explicitly POSIX-only symbolic-corpus export, hardened artifact-bundle publication, benchmark
report publication, or reference-release publication paths. Those exclusions are enforced by the
implementation and remain covered by Linux rejection tests. Hardened paths are not promoted to
cross-platform support merely because the portable catalog passes.

The scheduled job verifies artifacts committed to or bundled with this repository. No externally
published project model exists yet, so CI has no external model URL or digest to verify. Adding one
requires a reviewed immutable locator, checksum, license authorization, and intended-use metadata;
until then, external artifact verification remains an explicit release gate rather than a network
placeholder.

Coverage does not establish model quality, semantic correctness by itself, or suitability for a
production use case. Those claims require the repository's separate deterministic contracts and
versioned evaluation evidence.

The raw-source replay tests additionally bind the two Corpus V2-derived examples to their exact
MIT-licensed source bytes and IR records. They assert complete deterministic output IR and metadata,
controlled text, every mapping's node IDs and source-span provenance, accepted validation, and
failure after a one-byte source mutation. The wheel test executes both pairs from an installed
package outside the source checkout.

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

- semantic IR JSON/YAML round trips, schema-derived required-field deletion and defaulted-field
  omission at every document-reachable nested model boundary, direct `Measurement` coverage,
  unknown fields, invalid field constraints, and invalid causal-graph mutations;
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

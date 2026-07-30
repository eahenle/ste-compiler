# Executable example catalog

The authoritative Phase 6 inventory is
[`examples/manifest.yaml`](../examples/manifest.yaml). It maps all 13 scenarios in the
[V1 implementation plan](v1-implementation-plan.md#phase-6-examples-and-documentation) to:

- the exact command and checked-in fixtures;
- expected exit status, JSON fields, or generated artifacts;
- whether network access is forbidden or optional;
- the exact pytest node, owner module, and GitHub Actions job when pytest supplies the evidence; and
- an explicit gate when the repository does not yet contain the required provider, artifact, or
  benchmark evidence.

Run the credential-free catalog checks with:

```bash
uv run --locked --extra dev pytest -q tests/integration/test_executable_examples.py
```

The test normalizes its command working directory to the repository root, so the same absolute test
path also works when pytest starts in another directory. It blocks connection, datagram-send, and
standard DNS-resolution socket APIs while it runs every `portable-ci` and `posix-ci` command in the
test process. The test also parses every non-gated pytest command, verifies that the node still
exists in its declared `pytest_owner`, and checks that the declared `ci_job` selects that owner.
The normal CI test matrix discovers this test on Python 3.12, 3.13, and 3.14.

This Python tripwire is not an operating-system network sandbox. It does not automatically cover a
child process, a non-Python executable, or code that bypasses the `socket` module. The current
`portable-ci` and `posix-ci` commands use the in-process CLI runner or `runpy` and do not spawn
children. A future example that crosses that boundary must add an independently verified
child-process tripwire or run under an OS-level egress control before it can retain
`network: forbidden`. Neural mechanics examples retain their dedicated offline CI jobs because
they install PyTorch and Transformers and build temporary model fixtures; the normal test matrix
excludes tests marked `neural`.

## Installed distribution

The wheel intentionally ships the complete manifest, the custom-resource Python module, its three
resource files, and every installed-catalog fixture under `ste_compiler/`. The machine-readable
`distribution` block records both execution classes and the narrower `win32` override. From any
directory after installation, run:

```bash
python -m ste_compiler.examples.custom_resources
```

Run every portable installed example and verify its frozen expectations with:

```bash
python -m ste_compiler.examples.catalog_runner ./ste-example-output
```

The output argument must name a directory that does not yet exist. The runner creates it, gives each
scenario an isolated subdirectory, and requires unique positive IDs plus unique lowercase kebab-case
slugs. It resolves each scenario directory and proves that it remains below the output root before
creation. Portable options use separate value arguments; `--option=value` is rejected so a path
value cannot bypass package/output confinement. A shared command-dependency registry binds implicit
CLI, module, and pytest resources to each scenario's declared fixtures in both source and installed
catalog validation. Linux and macOS select the default `portable-ci` plus `posix-ci` execution
classes. Windows selects only `portable-ci`; benchmark reproduction scenario 13 remains
`posix-ci`. The runner prints a JSON summary only after all expected exits, recursively type-exact
JSON fields, and frozen report artifacts match. In particular, JSON booleans, integers, and
floating-point numbers are not interchangeable.

Code that wants to inspect the installed catalog can obtain it without assuming a checkout:

```python
from importlib.resources import files

catalog = files("ste_compiler.examples").joinpath("manifest.yaml")
print(catalog.read_text(encoding="utf-8"))
```

The distribution smoke builds and installs the wheel into a temporary target outside the source
checkout. It verifies the 13-entry catalog, requires every installed-catalog fixture below the
package root, executes the platform-selected catalog, and surfaces the catalog subprocess's stdout
and stderr on failure. It also builds a wheel from the sdist and guards the sdist's catalog,
resources, executable-example test, and broader development artifacts. The `existing-ci` and
`neural-ci` rows remain source-only because they invoke repository test targets and optional
development runtimes; that boundary is explicit in the manifest rather than implied by missing
wheel files.

## Current coverage

| # | Scenario | State | Executable evidence |
|---:|---|---|---|
| 1 | Raw source to deterministic controlled text | Tested | `ste-compiler demo --json` |
| 2 | Offline replay source extraction | Tested | Explicit `compile-source` source/IR pair |
| 3 | Optional live-provider extraction | Gated | No live structured provider or credential contract |
| 4 | Custom vocabulary and terminology | Tested | `python examples/custom_resources.py` |
| 5 | Dataset construction and manifest inspection | Tested | Corpus V2 build, manifest assertions, and byte verification |
| 6 | Encoder-decoder smoke training | Tested in neural CI | Declared training schema with tiny-fixture identity and runtime token-limit overrides |
| 7 | Decoder-only LoRA smoke training | Tested in neural CI | Deterministic two-step temporary fixture |
| 8 | Inference with each released checkpoint | Gated | No reviewed checkpoint release exists |
| 9 | Constrained versus unconstrained comparison | Gated | Requires released artifacts and an approved comparison protocol |
| 10 | Provenance and alignment inspection | Tested | Deterministic compile JSON with mapping and negation assertions |
| 11 | Expected validation rejection | Tested | Critical semantic-change and omitted-node diagnostics |
| 12 | Offline cached operation | Partial in existing CI | Both declared realizer YAML files are loaded and routed in local-files-only mode with test-double generators; no cached checkpoint is executed |
| 13 | Benchmark reproduction | Tested | Hash-bound deterministic fixture report with Wilson intervals |

“Tested” means mechanics or deterministic compiler behavior is executable. It does not turn the
temporary neural fixtures into released checkpoints or their outputs into model-quality evidence.
Scenario 6 reads `data/training/encoder-decoder-schema-example.yaml` and preserves its corpus,
optimizer, seed, step count, batching, and strategy. The smoke test replaces the model and tokenizer
identities with the generated tiny T5 fixture identity so no network or external checkpoint is
required. Because that fixture's byte-level tokenizer uses approximately one token per byte, the
test also widens only its runtime source and target token limits from `1024`/`256` to `8192`/`2048`;
the checked-in schema stays unchanged. Scenario 7 likewise loads its declared training schema and
corpus fixture. Scenario 12 loads both declared realizer YAML files directly; its generators remain
test doubles, which is why the scenario is partial rather than tested as materialized-cache
inference.
The scenario 13 command rebuilds demonstration corpus V2 and regenerates the frozen benchmark
fixture report from its hash-bound specification, taxonomy, prediction manifest, and raw JSONL. It
exercises confidence intervals, stage attribution, and report publication, but the observations
remain deliberately labeled `deterministic_fixture_only`. They are not model predictions or
model-quality evidence.

## Custom resource example

The custom-resource example deliberately uses small, original MIT-licensed files:

- `examples/resources/custom_vocabulary.yaml`;
- `examples/resources/custom_terminology.yaml`; and
- `examples/resources/custom_installation.yaml`.

It loads those files through the public Python API, realizes `Install the access hatch.`, validates
the text, and emits the resource versions, mappings, and validation report as JSON. This keeps custom
resource authoring explicit without adding source-checkout paths to the portable CLI configuration
contract.

## Remaining gates

No user decision is needed for the covered offline subset. Completing scenarios 3, 8, 9, and 12,
plus publishing measured benchmark evidence beyond scenario 13's fixture, requires new project
inputs:

1. select and implement a live structured provider plus its credential and fixture policy;
2. review and publish immutable encoder and decoder checkpoints with licenses and external digests,
   then exercise them through a materialized offline cache;
3. freeze the constrained/unconstrained comparison protocol; and
4. run a future measured benchmark schema against released artifacts and publish its raw stage
   artifacts, predictions, metrics, hardware disclosure, human review, and uncensored failures.

Until those inputs exist, the manifest leaves the scenarios gated or explicitly partial instead of
claiming that test doubles are cached checkpoints or inventing measured results.

# Contributing

Contributions are welcome when they preserve the project's auditable compiler boundary and
licensing constraints.

## Development setup

Requires Python 3.12 or newer and `uv`.

```bash
git clone https://github.com/eahenle/ste-compiler.git
cd ste-compiler
uv sync --extra dev
```

Run the quality and test gates directly:

```bash
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev pytest -q
```

Use the architecture-specific optional dependency set documented by a neural-training change.
Neural tests must run offline after their fixture preparation and must not silently download model
artifacts.

## Pull requests

- Keep each change focused and include regression tests.
- Update executable documentation when a public command, schema, or artifact contract changes.
- Record exact model, tokenizer, dataset, dependency, and output identities for neural work.
- Keep deterministic validation authoritative; model output is always an untrusted proposal.
- Report failures and limitations. Do not turn smoke-test completion into a model-quality claim.
- Run the supported Python matrix when a change can be version-sensitive.

## Data, vocabulary, and standards

Every contributed dataset, vocabulary entry, terminology resource, model, and derived artifact must
have a redistributable license and inspectable origin. Do not submit ASD-STE100 text, dictionary
content, or other material that the project is not authorized to redistribute. This project is
STE-inspired and must not be described as certified or compliant.

Dataset changes must preserve deterministic reconstruction, source-span grounding, frozen split
identity, leakage checks, manifests, and checksums. Changes to released artifacts require a new
version; do not overwrite an existing release identity.

## Reporting security issues

Do not disclose a vulnerability in a public issue. Follow [SECURITY.md](SECURITY.md).

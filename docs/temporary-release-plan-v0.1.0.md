# Temporary release plan for `v0.1.0` (ste-compiler)

This is a temporary execution plan to move from the current state to a tagged
`v0.1.0` release with minimal friction, while deferring any v1.0.0 work.

## Constraints from user

1. Do not require SSH key-based security on release tagging if it can be avoided.
2. Cut the release as `v0.1.0` and create the tag when possible.
3. Do not publish to a package registry.
4. Keep this as a temporary state; revisit before `v1.0.0`.
5. Use GitHub account `eahenle` for release operations.

## Preferred path (no SSH key required)

- Keep tag mode as an annotated tag for `v0.1.0` on the reviewed `main` commit.
- Run the local release contract check for the tag path.
- If tag validation can be made to pass without verifying a signer, use it and
  avoid any SSH key handling.
- Run/trigger the release provenance workflow to produce the reproducible
  artifacts and checksums for `v0.1.0`.
- Push only the annotated `v0.1.0` tag (no package-registry publish).

## Fallback path (if signing is required by tooling)

If tag validation still fails because SSH signing is enforced:

- Mint an SSH signing key locally.
- Add the matching public key to `.github/release/trusted-tag-signers`.
- Sign the `v0.1.0` annotated tag using that key.
- Push the tag.

Keep this as temporary behavior until we are ready for the `v1.0.0` release
decision point.

## Deferred until pre-v1.0 actions

- Any registry publishing (PyPI/GitHub Release asset publishing with package
  distribution expectations).
- Long-term release-signer policy changes or stricter release governance.
- Release policy updates for `v1.0.0`.

## Validation checklist

- `git status --short --branch` is clean and `main` points at the release
  commit.
- `gh auth status` shows `eahenle`.
- `v0.1.0` tag is created as an annotated tag and pushed.
- Release provenance checks pass.
- No dependency on package-registry publishing for this release.

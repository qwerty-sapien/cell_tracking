# Git large-file guard and chunked push

GitHub applies two different limits relevant to this repository:

- An individual Git blob larger than 100 MiB is rejected. A blob cannot be
  divided during `git push`; it must be removed from history or stored with Git
  LFS.
- A single push is limited to 2 GiB. A repository with only sub-100 MB files
  can still exceed this aggregate limit.

## One-time setup

```bash
scripts/install_git_hooks.sh
```

This sets the repository-local `core.hooksPath` and installs a `git safe-add`
alias. No global Git configuration is changed.

## Adding files

For immediate cleanup after staging, use the same arguments accepted by
`git add`:

```bash
git safe-add .
git safe-add path/to/files
```

Any staged file over 100,000,000 bytes is kept on disk, removed from the index,
and added to `.gitignore` using its exact repository-relative path.

Git has no `pre-add` hook. If ordinary `git add` is used, the repository's
`pre-commit` hook performs the same cleanup before the commit is created.

To inspect the index without changing it:

```bash
scripts/guard_large_files.sh --check
```

## Pushing this large branch

Inspect the operation first:

```bash
scripts/git_push_chunked.sh \
  --dry-run \
  --branch codex/trackastra-parity-diagnostic-v1
```

Then push:

```bash
scripts/git_push_chunked.sh \
  --branch codex/trackastra-parity-diagnostic-v1
```

The helper removes only oversized historical paths that are already ignored,
creates a timestamped local backup branch, seeds acceptable blobs through
temporary remote branches in 750 MB uncompressed batches, pushes the real
branch, and deletes the temporary remote branches after success.

If the final push fails, the temporary branches remain on the remote so a retry
does not need to upload those objects again. Local working files are never
deleted.

The branch remains unusually large even after the three rejected blobs are
removed. GitHub recommends keeping repositories below 1 GB and strongly
recommends staying below 5 GB, so moving reproducible binary artifacts to a
release, object storage, or Git LFS is still the better long-term layout.

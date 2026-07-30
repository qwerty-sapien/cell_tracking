#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a Git repository" >&2
  exit 2
}
cd "$repo_root"

chmod +x \
  .githooks/pre-commit \
  scripts/git-safe-add.sh \
  scripts/guard_large_files.sh \
  scripts/git_push_chunked.sh

git config --local core.hooksPath .githooks
git config --local alias.safe-add '!scripts/git-safe-add.sh'

echo "Installed repository hooks."
echo "Use 'git safe-add ...' for immediate cleanup after add."
echo "Ordinary 'git add ...' is checked and cleaned automatically at commit time."


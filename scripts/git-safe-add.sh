#!/usr/bin/env bash
#
# Usage: git safe-add <git-add arguments>
#        scripts/git-safe-add.sh <git-add arguments>

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a Git repository" >&2
  exit 2
}

git add "$@"
"$repo_root/scripts/guard_large_files.sh" --fix


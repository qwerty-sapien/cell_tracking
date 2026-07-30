#!/usr/bin/env bash
#
# Remove oversized staged files from Git while keeping them on disk.
# Exact repository-root paths are added to .gitignore so subsequent adds skip
# the files. This is used by both git-safe-add.sh and the pre-commit hook.

set -euo pipefail

MAX_BLOB_BYTES="${GIT_MAX_BLOB_BYTES:-100000000}"
FIX=0

usage() {
  cat <<'EOF'
Usage: scripts/guard_large_files.sh [--fix] [--check]

  --fix    Unstage/untrack oversized files and add exact paths to .gitignore.
  --check  Only report oversized staged files and exit non-zero (default).

The default limit is 100,000,000 bytes. Override it with GIT_MAX_BLOB_BYTES.
Files are never deleted from disk.
EOF
}

while (($#)); do
  case "$1" in
    --fix) FIX=1 ;;
    --check) FIX=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! "$MAX_BLOB_BYTES" =~ ^[0-9]+$ ]] || ((MAX_BLOB_BYTES < 1)); then
  echo "error: GIT_MAX_BLOB_BYTES must be a positive integer" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a Git repository" >&2
  exit 2
}
cd "$repo_root"

declare -a oversized_paths=()
declare -a oversized_sizes=()

while IFS= read -r -d '' path; do
  index_entry="$(git ls-files --stage -- "$path" | awk '$3 == 0 { print $2; exit }')"
  [[ -n "$index_entry" ]] || continue
  size="$(git cat-file -s "$index_entry")"
  if ((size > MAX_BLOB_BYTES)); then
    oversized_paths+=("$path")
    oversized_sizes+=("$size")
  fi
done < <(git diff --cached --name-only --diff-filter=AM -z)

if ((${#oversized_paths[@]} == 0)); then
  exit 0
fi

echo "Oversized staged Git blobs (limit: ${MAX_BLOB_BYTES} bytes):" >&2
for i in "${!oversized_paths[@]}"; do
  printf '  %12d  %s\n' "${oversized_sizes[$i]}" "${oversized_paths[$i]}" >&2
done

if ((FIX == 0)); then
  echo "Run scripts/guard_large_files.sh --fix to keep them locally and ignore them." >&2
  exit 1
fi

if [[ ! -e .gitignore ]]; then
  : > .gitignore
fi

if ! grep -Fqx '# Exact paths automatically excluded by the 100 MB Git guard' .gitignore; then
  if [[ -s .gitignore ]] && [[ "$(tail -c 1 .gitignore | wc -l | tr -d ' ')" == "0" ]]; then
    printf '\n' >> .gitignore
  fi
  printf '\n# Exact paths automatically excluded by the 100 MB Git guard\n' >> .gitignore
fi

ignore_pattern_for_path() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "error: cannot safely create a .gitignore entry for a path containing a newline" >&2
    return 1
  fi
  value="${value//\\/\\\\}"
  value="${value//\*/\\*}"
  value="${value//\?/\\?}"
  value="${value//\[/\\[}"
  value="${value//\]/\\]}"
  value="${value//\#/\\#}"
  value="${value//\!/\\!}"
  value="${value// /\\ }"
  printf '/%s' "$value"
}

for path in "${oversized_paths[@]}"; do
  pattern="$(ignore_pattern_for_path "$path")"
  if ! grep -Fqx "$pattern" .gitignore; then
    printf '%s\n' "$pattern" >> .gitignore
  fi
  git rm --cached --force --ignore-unmatch -- "$path" >/dev/null
  printf 'kept locally and ignored: %s\n' "$path" >&2
done

git add -- .gitignore


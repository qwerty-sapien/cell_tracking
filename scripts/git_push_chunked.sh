#!/usr/bin/env bash
#
# Push a large unpublished branch to GitHub in bounded blob batches.
#
# GitHub rejects any individual blob above 100 MiB and any single push above
# 2 GiB. This script:
#   1. removes currently ignored >100 MB paths from unpublished history,
#   2. pre-seeds acceptable blobs through temporary remote branches, and
#   3. pushes the real branch and removes the temporary branches.
#
# Local files are never deleted. A backup branch is created before any rewrite.

set -euo pipefail

REMOTE="origin"
BRANCH=""
MAX_BLOB_BYTES="${GIT_MAX_BLOB_BYTES:-100000000}"
CHUNK_BYTES="${GIT_PUSH_CHUNK_BYTES:-750000000}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/git_push_chunked.sh [options]

Options:
  --remote NAME       Remote to push (default: origin)
  --branch NAME       Local and remote branch (default: current branch)
  --dry-run           Inspect and print the planned rewrite/push only
  -h, --help          Show this help

Environment:
  GIT_MAX_BLOB_BYTES    Maximum allowed blob size (default: 100000000)
  GIT_PUSH_CHUNK_BYTES  Maximum uncompressed blob bytes per seed push
                        (default: 750000000)

The script may rewrite only commits not shared with the selected remote branch
(or remote default branch for a new branch). It creates backup/pre-clean-push/*
before rewriting and uses --force-with-lease when updating an existing branch.
EOF
}

while (($#)); do
  case "$1" in
    --remote)
      [[ $# -ge 2 ]] || { echo "error: --remote requires a value" >&2; exit 2; }
      REMOTE="$2"
      shift
      ;;
    --branch)
      [[ $# -ge 2 ]] || { echo "error: --branch requires a value" >&2; exit 2; }
      BRANCH="$2"
      shift
      ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for value_name in MAX_BLOB_BYTES CHUNK_BYTES; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || ((value < 1)); then
    echo "error: $value_name must be a positive integer" >&2
    exit 2
  fi
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a Git repository" >&2
  exit 2
}
cd "$repo_root"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "error: detached HEAD; pass --branch explicitly" >&2
    exit 2
  }
fi

if ! git check-ref-format --branch "$BRANCH" >/dev/null; then
  echo "error: invalid branch name: $BRANCH" >&2
  exit 2
fi
if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "error: local branch does not exist: $BRANCH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  if ((DRY_RUN)); then
    echo "warning: working tree is dirty; dry-run inspects committed history only" >&2
  else
    echo "error: the working tree must be clean before a history rewrite/push" >&2
    exit 1
  fi
fi
git remote get-url "$REMOTE" >/dev/null

echo "Fetching $REMOTE..."
if ((DRY_RUN == 0)); then
  git fetch "$REMOTE" --prune
fi

remote_ref="refs/remotes/$REMOTE/$BRANCH"
remote_exists=0
remote_tip=""
if git show-ref --verify --quiet "$remote_ref"; then
  remote_exists=1
  remote_tip="$(git rev-parse "$remote_ref")"
  base="$(git merge-base "refs/heads/$BRANCH" "$remote_ref")"
else
  default_remote_ref="$(git symbolic-ref --quiet "refs/remotes/$REMOTE/HEAD" || true)"
  if [[ -z "$default_remote_ref" ]]; then
    echo "error: cannot determine $REMOTE's default branch" >&2
    exit 1
  fi
  base="$(git merge-base "refs/heads/$BRANCH" "$default_remote_ref")"
fi

if [[ "$(git rev-parse "refs/heads/$BRANCH")" == "$base" ]]; then
  echo "Nothing to push."
  exit 0
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/git-push-chunked.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

object_inventory="$tmp_dir/objects.tsv"
large_paths="$tmp_dir/large-paths.z"

inventory_objects() {
  git rev-list --objects "$base..refs/heads/$BRANCH" |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    awk -v max="$MAX_BLOB_BYTES" -v inventory="$object_inventory" \
      -v large="$tmp_dir/large-lines.tsv" '
        $1 == "blob" && !seen[$2]++ {
          print $2 "\t" $3 > inventory
          if ($3 > max) {
            path = $0
            sub(/^[^ ]+ [^ ]+ [^ ]+ /, "", path)
            print $2 "\t" $3 "\t" path > large
          }
        }
      '
  touch "$object_inventory" "$tmp_dir/large-lines.tsv"
}

inventory_objects

large_count="$(wc -l < "$tmp_dir/large-lines.tsv" | tr -d ' ')"
if ((large_count > 0)); then
  echo "Oversized blobs found in unpublished history:"
  while IFS=$'\t' read -r oid size path; do
    printf '  %12d  %s  %s\n' "$size" "$oid" "$path"
    if ! git check-ignore -q -- "$path"; then
      echo "error: oversized historical path is not ignored: $path" >&2
      echo "Add an intentional ignore rule, untrack it, commit, then retry." >&2
      exit 1
    fi
    printf '%s\0' "$path" >> "$large_paths"
  done < "$tmp_dir/large-lines.tsv"

  backup_name="backup/pre-clean-push/${BRANCH//\//-}-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "History rewrite required; backup branch: $backup_name"
  if ((DRY_RUN)); then
    echo "Dry run: would remove $large_count ignored oversized path(s) from $base..$BRANCH."
    echo "Dry run: no history was rewritten and nothing was pushed."
    exit 0
  fi

  git branch "$backup_name" "refs/heads/$BRANCH"
  export CLEAN_PUSH_PATH_FILE="$large_paths"
  FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force \
    --index-filter 'xargs -0 git rm -r --cached --ignore-unmatch -- < "$CLEAN_PUSH_PATH_FILE"' \
    --prune-empty \
    "$base..refs/heads/$BRANCH"
  unset CLEAN_PUSH_PATH_FILE

  rm -f "$object_inventory" "$tmp_dir/large-lines.tsv"
  inventory_objects
  if [[ -s "$tmp_dir/large-lines.tsv" ]]; then
    echo "error: oversized blobs remain after rewrite; refusing to push" >&2
    exit 1
  fi
elif ((DRY_RUN)); then
  echo "Dry run: no oversized blobs require a history rewrite."
fi

blob_count="$(wc -l < "$object_inventory" | tr -d ' ')"
blob_bytes="$(awk -F '\t' '{ total += $2 } END { printf "%.0f", total }' "$object_inventory")"
echo "Unpublished unique blobs: $blob_count (${blob_bytes} uncompressed bytes)"

if ((DRY_RUN)); then
  chunks=$(( (blob_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES ))
  echo "Dry run: would use approximately $chunks seed push(es), then push $BRANCH."
  exit 0
fi

seed_prefix="__codex_blob_seed/${BRANCH//\//-}/$(git rev-parse --short=12 "refs/heads/$BRANCH")"
declare -a seed_remote_refs=()
batch_file="$tmp_dir/batch.tsv"
batch_bytes=0
batch_number=0

push_seed_batch() {
  local tree_input
  local entry_number
  local batch_oid
  local batch_size
  local tree_oid
  local seed_commit
  local remote_seed_ref

  [[ -s "$batch_file" ]] || return 0
  batch_number=$((batch_number + 1))
  tree_input="$tmp_dir/tree-input-$batch_number"
  : > "$tree_input"
  entry_number=0
  while IFS=$'\t' read -r batch_oid batch_size; do
    entry_number=$((entry_number + 1))
    printf '100644 blob %s\tblob-%08d\n' "$batch_oid" "$entry_number" >> "$tree_input"
  done < "$batch_file"

  tree_oid="$(git mktree < "$tree_input")"
  seed_commit="$(
    GIT_AUTHOR_NAME='Codex blob seed' \
    GIT_AUTHOR_EMAIL='blob-seed@invalid' \
    GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
    GIT_COMMITTER_NAME='Codex blob seed' \
    GIT_COMMITTER_EMAIL='blob-seed@invalid' \
    GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
      git commit-tree "$tree_oid" -p "$base" \
        -m "Temporary blob seed $batch_number for $BRANCH"
  )"
  remote_seed_ref="refs/heads/$seed_prefix/batch-$batch_number"
  echo "Pushing seed batch $batch_number ($batch_bytes uncompressed bytes)..."
  git push "$REMOTE" "$seed_commit:$remote_seed_ref"
  seed_remote_refs+=("$remote_seed_ref")
  : > "$batch_file"
  batch_bytes=0
}

: > "$batch_file"
while IFS=$'\t' read -r oid size; do
  if ((batch_bytes > 0 && batch_bytes + size > CHUNK_BYTES)); then
    push_seed_batch
  fi
  printf '%s\t%s\n' "$oid" "$size" >> "$batch_file"
  batch_bytes=$((batch_bytes + size))
done < "$object_inventory"
push_seed_batch

echo "Pushing real branch $BRANCH..."
if ((remote_exists)); then
  git push \
    --force-with-lease="refs/heads/$BRANCH:$remote_tip" \
    "$REMOTE" "refs/heads/$BRANCH:refs/heads/$BRANCH"
else
  git push --set-upstream "$REMOTE" "refs/heads/$BRANCH:refs/heads/$BRANCH"
fi

if ((${#seed_remote_refs[@]} > 0)); then
  echo "Removing temporary remote seed branches..."
  delete_refspecs=()
  for ref in "${seed_remote_refs[@]}"; do
    delete_refspecs+=(":$ref")
  done
  git push "$REMOTE" "${delete_refspecs[@]}"
fi

echo "Push complete: $REMOTE/$BRANCH"

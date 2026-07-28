#!/usr/bin/env python3
"""Assert every worktree change is inside the assigned branch subtree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


WORKTREE = Path("/private/tmp/g02_phase1_20260728/sa_repr")
PREFIX = "reports/g02/screening/tiered_surface/branches/sa_repr_multi/"
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKTREE / PREFIX / "PATH_ALLOWLIST_AUDIT.json",
    )
    args = parser.parse_args()
    output = args.output
    lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=WORKTREE,
        text=True,
    ).splitlines()
    paths = []
    violations = []
    for line in lines:
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw)
        if not raw.startswith(PREFIX):
            violations.append(raw)
    output_relative = str(output.relative_to(WORKTREE))
    if output_relative not in paths:
        paths.append(output_relative)
    result = {
        "schema": "g02.sa-repr-multi.path-allowlist-audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "worktree": str(WORKTREE),
        "allowed_prefix": PREFIX,
        "changed_path_count": len(paths),
        "changed_paths": paths,
        "violation_count": len(violations),
        "violations": violations,
        "protected_single_writer_paths_modified": [
            path
            for path in paths
            if path.endswith(
                (
                    "LIVE_STATE.md",
                    "LEDGER.md",
                    "tier_census.json",
                    "kaggle_budget.json",
                )
            )
        ],
    }
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "paths": len(paths)}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

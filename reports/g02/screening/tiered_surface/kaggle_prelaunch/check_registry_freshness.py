#!/usr/bin/env python3
"""Fail closed if a dataset or kernel slug named in LIVE_STATE.md has no registry entry.

The prelaunch guard (`kaggle_prelaunch_guard.py`) only protects a dataset that is
already listed in `KAGGLE_FAILURE_REGISTRY.json`'s ownership maps. This
script closes the process gap that lets that map go stale: it extracts every
`<owner>/<slug>` Kaggle dataset reference from LIVE_STATE.md and reports any that
`dataset_ownership` does not cover. It performs no network access and mutates
nothing; it is read-only evidence for a human/Codex decision to update the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
LIVE_STATE = ROOT / "reports/g02/screening/LIVE_STATE.md"
REGISTRY = ROOT / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"

# Kaggle dataset/kernel slugs look like `owner/dataset-name-v1`; restrict to the
# one owner actually used by this campaign to avoid matching unrelated text.
SLUG_PATTERN = re.compile(r"\bonebitbrain/[a-z0-9][a-z0-9-]*[a-z0-9]\b")


def main() -> int:
    live_state_text = LIVE_STATE.read_text(encoding="utf-8")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    dataset_ownership = registry.get("dataset_ownership", {})
    kernel_ownership = registry.get("kernel_ownership", {})
    ownership = {**dataset_ownership, **kernel_ownership}

    referenced = sorted(set(SLUG_PATTERN.findall(live_state_text)))
    covered = sorted(s for s in referenced if s in ownership)
    uncovered = sorted(s for s in referenced if s not in ownership)

    report = {
        "live_state_referenced_slugs": referenced,
        "registry_covered_slugs": covered,
        "registry_uncovered_slugs": uncovered,
        "registry_ownership_entries": sorted(ownership),
        "registry_dataset_entries": sorted(dataset_ownership),
        "registry_kernel_entries": sorted(kernel_ownership),
        "status": "PASS" if not uncovered else "FAIL_REGISTRY_STALE",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not uncovered else 1


if __name__ == "__main__":
    raise SystemExit(main())

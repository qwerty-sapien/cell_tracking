#!/usr/bin/env python3
"""Fail-closed validation for the Lane 4 park audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ALLOWED_CAUSES = {
    "EVIDENCE",
    "DEPENDENCY",
    "GOVERNANCE_CONTROL",
    "OWNER_POLICY",
    "DELEGATED_OWNER_OVERRIDE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    census = json.loads(args.census.read_text())
    result = json.loads(args.result.read_text())
    expected = {
        row["id"]
        for row in census["nodes"]
        if len(row.get("blocked_on", [])) > 0
    }
    rows = result["rows"]
    actual = [row["id"] for row in rows]
    assert len(actual) == len(set(actual)), "duplicate audit row"
    assert set(actual) == expected, {
        "missing": sorted(expected - set(actual)),
        "unexpected": sorted(set(actual) - expected),
    }
    assert result["source_census_sha256"] == sha256(args.census)
    assert result["blocked_rows_expected"] == len(expected)
    assert result["blocked_rows_audited"] == len(rows)
    assert result["edit_set_size"] == 0
    assert result["graph"]["executed"] is False
    assert result["graph"]["delta_adjusted_raw_node"] == [0.0, 0.0, 0.0]

    for row in rows:
        assert row["cause"] in ALLOWED_CAUSES, row
        assert row["tier_move_proposed"] is False, row
        if row["cause"] == "DEPENDENCY":
            falsifier = row["cheapest_own_merits_falsifier"].strip().lower()
            assert falsifier and falsifier != "none", row

    counts = Counter(row["cause"] for row in rows)
    assert dict(counts) == result["cause_counts"], (counts, result["cause_counts"])
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "rows": len(rows),
                "cause_counts": dict(counts),
                "tier_moves": 0,
                "edit_set_size": 0,
                "graph_delta_adjusted_raw_node": [0.0, 0.0, 0.0],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

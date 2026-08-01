#!/usr/bin/env python3
"""Independent terminal validation for the A7 zero-edit graph outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = Path(__file__).resolve().parents[6]


class ValidationError(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    require(WORKTREE_ROOT in run.parents, "run must be in lane worktree")
    require(WORKTREE_ROOT in output.parents, "output must be in lane worktree")
    require(not output.exists(), "validation output already exists")

    result_path = run / "RESULT.json"
    result = load(result_path)
    guard_path = run / "GUARD_RECOVERY_3.json"
    guard = load(guard_path)
    require(result["status"] == "KILL_44B6_CANNOT_BEAT_APNCC", "terminal status drift")
    require(result["edit_set"]["applied_edits"] == 0, "edit count is not zero")
    require(result["edit_set"]["threshold_pass_before_disjoint_budget_sum"] == 0, "threshold support is not zero")
    require(result["exact_44b6"]["candidate"] == result["exact_44b6"]["apncc"], "candidate/APNCC mismatch")
    require(
        result["exact_44b6"]["delta"]
        == {
            "adjusted_edge_jaccard": 0.0,
            "raw_edge_jaccard": 0.0,
            "node_recall": 0.0,
        },
        "exact delta is not zero",
    )
    require(result["exact_44b6"]["edge_counts"]["candidate"] == result["exact_44b6"]["edge_counts"]["apncc"], "edge counts changed")
    require(all(result["properties"].values()), "property sieve failed")
    require(not any(result["downstream"].values()), "downstream action was taken")
    records = result["recovery"]["candidate_fragment_records"]
    require(len(records) == 36, "candidate coverage is not 36 videos")
    identities = set()
    for row in records:
        identity = str(row["identity"])
        require(identity not in identities, f"duplicate identity: {identity}")
        identities.add(identity)
        require(row["byte_identical"] is True, f"non-identical fragment: {identity}")
        require(row["candidate_sha256"] == row["control_sha256"], f"fragment hash mismatch: {identity}")
        path = run / "candidate_fragments" / (identity.replace(":", "__", 1) + ".csv")
        require(sha256(path) == row["candidate_sha256"], f"candidate drift after result: {identity}")
    require(
        guard.get("status") == "PASS"
        and guard.get("stage") == "recover"
        and int(guard.get("returncode", -1)) == 0
        and int(guard.get("swap_growth_bytes", -1)) == 0,
        "terminal recovery guard did not pass cleanly",
    )

    validation = {
        "schema": "celltrack-alpha.a7-phase-tracklet-terminal-validation.v1",
        "status": "PASS",
        "result_sha256": sha256(result_path),
        "guard_sha256": sha256(guard_path),
        "candidate_fragments_verified": 36,
        "candidate_control_byte_identity": True,
        "edit_set_size": 0,
        "exact_delta": result["exact_44b6"]["delta"],
        "property_sieve": result["properties"],
        "downstream_actions": result["downstream"],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

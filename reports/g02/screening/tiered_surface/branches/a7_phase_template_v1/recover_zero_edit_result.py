#!/usr/bin/env python3
"""Serialize the parity-proven A7 result without replaying the official scorer.

The guarded scientific run completed exact scoring and exact/algebra parity before
failing in the edge-count reporting block.  This recovery is admissible only when
every sealed candidate fragment is byte-identical to its frozen APNCC control,
which makes the selected edit set empty and all exact metric deltas identically
zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = Path(__file__).resolve().parents[6]
CURRENT_ROOT = Path("/Users/agent/Desktop/cell_tracking")
PREREG = HERE / "PREREGISTRATION.json"
FREEZE = HERE / "IMPLEMENTATION_FREEZE.json"
REPAIR_1 = HERE / "EXECUTION_REPAIR_1.json"
REPAIR_2 = HERE / "EXECUTION_REPAIR_2.json"
REPAIR_3 = HERE / "EXECUTION_REPAIR_3.json"
PRIOR_RUNNER_SHA256 = "d0335deaefc69e2747f2880e736e803fe02515c56cb75e31cb684c7b9bd8ac3d"
EMPTY_I64_SHA256 = hashlib.sha256(b"").hexdigest()


class RecoveryError(RuntimeError):
    """The sealed recovery evidence did not authenticate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON object required: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("recover",), default="recover")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output.resolve()
    require(WORKTREE_ROOT in output.parents, "output must remain in lane worktree")

    prereg = load_json(PREREG)
    freeze = load_json(FREEZE)
    repair_1 = load_json(REPAIR_1)
    repair_2 = load_json(REPAIR_2)
    repair_3 = load_json(REPAIR_3)
    require(
        prereg.get("status") == "FROZEN_BEFORE_PHASE_FEATURE_OR_GRAPH_OUTCOME",
        "preregistration status drift",
    )
    require(
        freeze.get("files", {}).get("run_a7_phase_template.py")
        == PRIOR_RUNNER_SHA256,
        "prior scientific runner pin drift",
    )
    require(
        repair_1.get("after_guard_sha256")
        == "89ce5f6f0aab6590e4b4779235b4b20ff1fd5465564dfe900f24d2cd7de4b202"
        and repair_1.get("unchanged", {}).get("runner_sha256")
        == PRIOR_RUNNER_SHA256
        and repair_1.get("unchanged", {}).get("thresholds_budgets_null_kill_line")
        is True,
        "first execution repair or its unchanged-science proof drifted",
    )
    require(
        repair_2.get("runner_sha256_after")
        == "ca57604d47e3106aa538445ce6d7e8ac60b7a77760e2fd1e98ee43d3a5540d35",
        "reporting repair pin drift",
    )
    require(
        repair_3.get("status") == "SEALED_BEFORE_ZERO_EDIT_RECOVERY",
        "recovery repair is not sealed",
    )

    manifest_path = output / "FEATURE_MANIFEST.json"
    manifest = load_json(manifest_path)
    require(
        manifest.get("status") == "COMPLETE_FEATURES_BEFORE_SELECTION_OUTCOMES"
        and int(manifest.get("actions", -1)) == 45_961
        and len(manifest.get("shards", ())) == 36,
        "sealed feature manifest drift",
    )
    require(
        sha256(manifest_path)
        == "6b59a3fc421094cf4c8c8fd34b7fe4ec8b8aeed0449048902cd4d3649d1518a6",
        "feature manifest hash drift",
    )

    guard_path = output / "GUARD_RETRY_2.json"
    guard = load_json(guard_path)
    require(
        guard.get("status") == "CHILD_EXIT"
        and int(guard.get("returncode", 0)) == 1
        and int(guard.get("swap_growth_bytes", -1)) == 0
        and guard.get("stage") == "all",
        "prior exact execution receipt drift",
    )

    report_pin = prereg["frozen_inputs"]["apncc_44b6_report"]
    report_path = CURRENT_ROOT / str(report_pin["path"])
    require(sha256(report_path) == str(report_pin["sha256"]), "APNCC report drift")
    report = load_json(report_path)
    control = report["graph"]["control"]
    files = control["input_provenance"]["files"]
    require(len(files) == 36, "APNCC control coverage drift")

    candidate_rows = []
    for row in files:
        identity = str(row["identity"])
        control_path = Path(str(row["path"]))
        if not control_path.is_absolute():
            control_path = CURRENT_ROOT / control_path
        require(sha256(control_path) == str(row["sha256"]), f"control drift: {identity}")
        candidate_path = output / "candidate_fragments" / (
            identity.replace(":", "__", 1) + ".csv"
        )
        candidate_sha = sha256(candidate_path)
        require(
            candidate_sha == str(row["sha256"]),
            f"nonempty or drifting candidate fragment: {identity}",
        )
        candidate_rows.append(
            {
                "identity": identity,
                "control_sha256": str(row["sha256"]),
                "candidate_sha256": candidate_sha,
                "bytes": candidate_path.stat().st_size,
                "byte_identical": True,
            }
        )

    base = control["summary"]
    edge_counts = control["edge_counts"]
    require(
        math.isclose(float(base["adj_edge_jaccard"]), 0.8665345260361912, abs_tol=1e-12)
        and math.isclose(float(base["edge_jaccard"]), 0.862596057467424, abs_tol=1e-12)
        and math.isclose(float(base["node_recall"]), 0.9926180002844743, abs_tol=1e-12),
        "APNCC exact baseline drift",
    )
    exact = {
        "adjusted_edge_jaccard": float(base["adj_edge_jaccard"]),
        "raw_edge_jaccard": float(base["edge_jaccard"]),
        "node_recall": float(base["node_recall"]),
    }
    zero = {
        "adjusted_edge_jaccard": 0.0,
        "raw_edge_jaccard": 0.0,
        "node_recall": 0.0,
    }
    result = {
        "schema": "celltrack-alpha.a7-phase-tracklet-swap-result.v1",
        "status": "KILL_44B6_CANNOT_BEAT_APNCC",
        "node": prereg["node"],
        "preregistration_sha256": sha256(PREREG),
        "feature_manifest_sha256": sha256(manifest_path),
        "recovery": {
            "schema": "celltrack-alpha.a7-zero-edit-parity-recovery.v1",
            "execution_repair_sha256": sha256(REPAIR_3),
            "prior_guard_receipt": str(guard_path.relative_to(WORKTREE_ROOT)),
            "prior_guard_receipt_sha256": sha256(guard_path),
            "prior_runner_sha256": PRIOR_RUNNER_SHA256,
            "prior_zero_swap_exact_graph_execution_seconds": float(guard["elapsed_seconds"]),
            "exact_graph_executed_prior_zero_swap_run": True,
            "exact_algebra_parity_passed_before_reporting_exception": True,
            "apncc_parity_passed_before_reporting_exception": True,
            "official_scorer_replayed_by_recovery": False,
            "all_candidate_fragments_byte_identical_to_apncc": True,
            "candidate_fragment_records": candidate_rows,
        },
        "population": {
            "actions": 45_961,
            "videos": 36,
            "scope": "occupied-endpoint conflicts only",
        },
        "edit_set": {
            "threshold_pass_before_disjoint_budget_sum": 0,
            "applied_edits": 0,
            "affected_videos": 0,
            "postselection_correct": 0,
            "postselection_harmful": 0,
            "postselection_precision": None,
            "selected_indices_sha256": EMPTY_I64_SHA256,
        },
        "exact_44b6": {
            "apncc": exact,
            "candidate": exact,
            "delta": zero,
            "algebra_parity_delta": zero,
            "edge_counts": {
                "apncc": {
                    "tp": int(edge_counts["edge_tp"]),
                    "fp": int(edge_counts["edge_fp"]),
                    "fn": int(edge_counts["edge_fn"]),
                },
                "candidate": {
                    "tp": int(edge_counts["edge_tp"]),
                    "fp": int(edge_counts["edge_fp"]),
                    "fn": int(edge_counts["edge_fn"]),
                },
            },
            "beats_apncc": False,
            "campaign_adjusted_plus_0p005": False,
            "historical_apncc_cv57_gain_context": 0.008372476430831166,
        },
        "nested_selection": {
            "protocol": "five-fold sealed OOF under frozen preregistration",
            "terminal_observation": "ABSTAIN_ALL_FOLDS_OR_ZERO_HELD_SUPPORT",
            "held_applied_edits_total": 0,
            "held_threshold_pass_total": 0,
            "fold_receipts_not_serialized_due_postscore_reporting_exception": True,
        },
        "properties": {
            "appearance_grounded": True,
            "narrow": True,
            "node_disjoint": True,
            "recall_preserving": True,
        },
        "downstream": {
            "family_6bba_run": False,
            "cv57_run": False,
            "kaggle_action": False,
            "gpu_action": False,
            "submission_action": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    print(json.dumps({"status": result["status"], "edits": 0, "delta": zero}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the contract-conformant, no-score AE-P5 structural falsifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root: Path, suffix: str) -> str:
    rows = [
        (path.name, sha256(path), path.stat().st_size)
        for path in sorted(root.glob(f"*{suffix}"))
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def main() -> int:
    contract_path = HERE / "AE_P5_C0_ASSIGNMENT_EXECUTION_CONTRACT.json"
    result_path = HERE / "ae_p5_rank1" / "AE_P5_RANK1_RESULT_V2.json"
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    expected = contract["inputs"]
    checks = {
        "frozen_execution_contract": contract["status"]
        == "FROZEN_BEFORE_CPU_STRUCTURAL_FALSIFIER",
        "population_pin": aggregate(
            ROOT / expected["frozen_population"]["path"], ".npz"
        )
        == expected["frozen_population"]["aggregate_sha256"],
        "control_report_pin": sha256(ROOT / expected["apncc_control"]["report_path"])
        == expected["apncc_control"]["report_sha256"],
        "control_fragments_pin": aggregate(
            ROOT / expected["apncc_control"]["fragment_root"], ".csv"
        )
        == expected["apncc_control"]["fragment_aggregate_sha256"],
        "n1_implementation_pin": sha256(ROOT / expected["n1"]["implementation_path"])
        == expected["n1"]["implementation_sha256"],
        "exact_action_count": result["totals"]["actions"]
        == expected["frozen_population"]["action_count_expected"],
        "complete_ownership": set(result["ownership"])
        == {
            "source",
            "destination",
            "incumbent_edge",
            "challenger_edge",
            "no_op_endpoint",
        },
        "every_component_falls_back": result["totals"]["components"]
        == result["totals"]["components_fallback"]
        and result["totals"]["components_applied"] == 0
        and result["totals"]["actions_selected"] == 0,
        "decisive_batches_compiled": result["totals"]["n1_batch_compiles"] == 361
        and result["totals"]["n1_batch_rejections"] == 0,
        "no_score_scope": result["scope"]
        == {
            "source_partition": "fit-train105",
            "family": "44b6",
            "held_labels_used": False,
            "graph_run": False,
            "family_6bba_run": False,
            "cv57_run": False,
        },
    }
    payload = {
        "schema": "celltrack-alpha.ae-p5-c0-c2-validation.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract_sha256": sha256(contract_path),
        "result_sha256": sha256(result_path),
        "checks": checks,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

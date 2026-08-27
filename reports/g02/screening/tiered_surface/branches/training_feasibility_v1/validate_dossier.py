#!/usr/bin/env python3
"""Deterministically validate the training-feasibility decision dossier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "reports/g02/screening").is_dir())
EVIDENCE = HERE / "EVIDENCE.json"
DOSSIER = HERE / "DECISION_DOSSIER.md"
PEFT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1/"
    "run_v1/kaggle_output/PEFT_RESULT.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
peft = json.loads(PEFT.read_text(encoding="utf-8"))
dossier = DOSSIER.read_text(encoding="utf-8")

assert evidence["status"] == "HUMAN_DECISION_REQUIRED_NO_PLAN_AMENDMENT"
assert sha256(PEFT) == evidence["peft"]["result_sha256"]
assert evidence["peft"]["result_sha256"] == (
    "d9f982177e64a60d4a0e76a68d5e45598999c0ad0757cd1f46c992eb812c600b"
)

trajectory = [row["auc"] for row in peft["capacity_probe"]["trajectory"]]
assert trajectory == evidence["peft"]["auc_trajectory"]
assert evidence["peft"]["plateau_epochs"] == [6, 7, 8]
assert close(evidence["peft"]["plateau_gain_epoch_6_to_8"], trajectory[7] - trajectory[5])
assert evidence["peft"]["selected_outer_epochs"] == [
    row["selected_epoch"] for row in peft["selection"]["outer"]
] == [1, 2, 2, 2, 2]
assert close(
    evidence["peft"]["graph_selected_auc"],
    peft["capacity_probe"]["selected_auc"],
)
for short, source in (
    ("adjusted", "adjusted_edge_jaccard"),
    ("raw", "raw_edge_jaccard"),
    ("node", "node_recall"),
):
    assert close(
        evidence["peft"]["graph_delta"][short],
        peft["graph"]["aggregate"]["delta"][source],
    )

actual_seconds = peft["execution"]["elapsed_seconds"]
assert close(evidence["peft"]["runtime"]["actual_seconds"], actual_seconds)
assert close(evidence["peft"]["runtime"]["actual_h"], actual_seconds / 3600)
assert close(
    evidence["peft"]["runtime"]["projection_over_actual_factor"],
    4.8 / (actual_seconds / 3600),
)

methods = evidence["approach_auc"]
assert [row["name"] for row in methods] == [
    "incumbent",
    "C1 conflict-set attention",
    "PEFT epoch-8 descriptive",
    "DINOv2 appearance",
    "C2 assignment evidence",
]
assert [round(row["auc"], 3) for row in methods] == [0.502, 0.502, 0.635, 0.670, 0.670]

historical = evidence["historical_compute"]
assert close(
    historical["implied_h_per_epoch"],
    historical["organizer_temporalunet3d_full_horizon_h"]
    / historical["organizer_full_horizon_epochs"],
)
assert close(historical["three_epoch_useful_checkpoint_h"], 478 / 300 * 3)
assert close(historical["seven_epoch_projection_h"], 478 / 300 * 7)

trackastra, raw3d = evidence["variants"]
assert trackastra["id"] == "TRACKASTRA-FINETUNE-MVE"
cost = trackastra["cost_model"]
low = 1.25 * (
    2 * 0.25 * cost["measured_frozen_full_fit105_pass_h"] * 2.5
    + cost["full_44b6_evaluation_h"]
)
high = 1.25 * (
    2 * 0.25 * cost["measured_frozen_full_fit105_pass_h"] * 4.0
    + cost["full_44b6_evaluation_h"]
)
assert close(cost["mve_t4_h_range"][0], low)
assert close(cost["mve_t4_h_range"][1], high)
assert raw3d["id"] == "RAW3D-NOLINK-MVE"
assert raw3d["cost_model"]["matched_calibration_cap_t4_h"] == 0.25
assert raw3d["cost_model"]["useful_checkpoint_t4_h_range_after_calibration"] == [2.0, 6.0]

assert evidence["decision"]["option_count"] == 2
assert evidence["decision"]["recommended_option"] == 1
assert evidence["decision"]["recommended_node"] == "RAW3D-NOLINK-MVE"
option_headers = re.findall(r"^## Option [12] — .+$", dossier, flags=re.MULTILINE)
assert len(option_headers) == 2
assert option_headers[0].startswith("## Option 1")
assert option_headers[1].startswith("## Option 2")
assert not re.search(r"^## Option [3-9]", dossier, flags=re.MULTILINE)

required_text = (
    "118.6×",
    "0.634749",
    "1/2/2/2/2",
    "+0.000133827/+0.000133160/0",
    "4.05–5.96 T4-h",
    "2–6 T4-h",
    "0.9255–0.9422",
    "Recommend Option 1",
    "HALT. NO TRAINING, GPU KERNEL, OR PLAN AMENDMENT WAS MADE.",
)
for text in required_text:
    assert text in dossier, text

for source in evidence["sources"]:
    assert source["primary"] is True
    assert source["url"] in dossier
    assert source["transfer_scope"]

assert evidence["scope"] == {
    "analysis_only": True,
    "training_performed": False,
    "gpu_kernel_pushed": False,
    "competition_submission_performed": False,
    "dead_register_changed": False,
    "human_locked_plan_changed": False,
}

print(
    json.dumps(
        {
            "schema": "celltrack-alpha.training-feasibility-validation.v1",
            "status": "PASS",
            "checks": 25,
            "option_count": len(option_headers),
            "recommended_option": evidence["decision"]["recommended_option"],
            "peft_result_sha256": sha256(PEFT),
            "dossier_sha256": sha256(DOSSIER),
            "evidence_sha256": sha256(EVIDENCE),
        },
        indent=2,
        sort_keys=True,
    )
)

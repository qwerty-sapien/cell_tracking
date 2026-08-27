#!/usr/bin/env python3
"""Compile and score the frozen CRW-HPO1000 winner exactly once on held-fit 44b6."""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ALPHA = ROOT / "dag_prompts/2026-08-10-action-vocabulary-alpha"
CURRENT_LMT = ROOT / "dag_prompts/2026-08-12-lmt-resume"
for location in (HERE, ALPHA, CURRENT_LMT, ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import a1r2_crw_supplier as a1r2  # noqa: E402
import a1r3_crw_supplier as a1r3  # noqa: E402
import a1r4_lmt_runtime as lmt_runtime  # noqa: E402
import current_lmt  # noqa: E402


CONTRACT = HERE / "WINNER_SCORE_CONTRACT.json"
FREEZE = HERE / "REMOTE_OUTPUT_FREEZE.json"
WINNER = HERE / "remote_output_v1/crw_hpo1000_v1/WINNER.json"
SEMANTIC = HERE / "remote_output_v1/crw_hpo1000_v1/SEMANTIC_RECEIPT.json"
DISPERSION = HERE / "remote_output_v1/crw_hpo1000_v1/DISPERSION_SUMMARY.json"
CARRIER = HERE / "input_dataset/crw_hpo1000_carrier.npz"
CARRIER_MANIFEST = HERE / "input_dataset/CARRIER_MANIFEST.json"
PREREG = HERE / "PREREGISTRATION.json"
BASE_CONTRACT = ALPHA / "a1_supplier_contract.json"
CLAIM = HERE / "WINNER_GRAPH_SCORE_CLAIM.json"
OUTPUT = HERE / "LOCAL_WINNER_GRAPH_SCORE.json"


class ScoreError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ScoreError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ScoreError(f"refusing to overwrite one-shot artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical(dict(value)))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class CompositeGuard:
    def __init__(self, monitor: a1r2.LMTResourceMonitor, workspace: current_lmt.WorkspaceGrowthGuard) -> None:
        self.monitor = monitor
        self.workspace = workspace
        self.maximum_combined_rss_gib = current_lmt.current_lmt_values()["maximum_combined_rss_gib"]

    def check(self) -> None:
        self.monitor.check()
        self.workspace.check()
        if self.monitor.combined_rss_gib() >= self.maximum_combined_rss_gib:
            raise ScoreError("current_lmt_maximum_combined_rss_abort")


def verify_preconditions() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if CLAIM.exists() or OUTPUT.exists():
        raise ScoreError("one-score namespace has already been consumed")
    contract = load_json(CONTRACT)
    freeze = load_json(FREEZE)
    prereg = load_json(PREREG)
    if (
        contract.get("schema") != "celltrack-alpha.crw-hpo1000-winner-score-contract.v1"
        or contract.get("status") != "FROZEN_BEFORE_REMOTE_OUTCOME_RETRIEVAL"
        or contract.get("score_calls") != 1
        or contract.get("cv57_access") is not False
        or contract.get("six_bba_access") is not False
        or contract.get("competition_submission") is not False
        or contract.get("post_outcome_retuning") is not False
    ):
        raise ScoreError("winner-score contract drifted")
    bindings = contract.get("bindings", {})
    expected_paths = {
        "score_script": Path(__file__).resolve(),
        "freeze_script": HERE / "freeze_remote_output.py",
        "preregistration": PREREG,
        "base_supplier_contract": BASE_CONTRACT,
        "carrier_manifest": CARRIER_MANIFEST,
        "carrier": CARRIER,
        "lmt": ROOT / "LMT.md",
    }
    if set(bindings) != set(expected_paths):
        raise ScoreError("winner-score binding names drifted")
    for name, path in expected_paths.items():
        record = bindings[name]
        if (ROOT / str(record.get("path", ""))).resolve() != path or sha256(path) != record.get("sha256"):
            raise ScoreError(f"winner-score binding drifted: {name}")
    if (
        freeze.get("schema") != "celltrack-alpha.crw-hpo1000-remote-output-freeze.v1"
        or freeze.get("status") != "FROZEN_BEFORE_LOCAL_GRAPH_SCORE"
        or freeze.get("kernel_version") != 1
        or freeze.get("scientific_boundaries", {}).get("official_graph_score_calls_remaining") != 1
    ):
        raise ScoreError("remote output freeze is absent or inapplicable")
    artifact_root = ROOT / freeze["artifact_root"]
    for name, record in freeze["files"].items():
        path = artifact_root / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ScoreError(f"frozen remote output drifted: {name}")
    if prereg.get("status") != "FROZEN_BEFORE_NEW_CARRIER_SUPERVISION_MATERIALIZATION":
        raise ScoreError("preregistration status drifted")
    current_lmt.load_lmt()
    return contract, freeze, prereg


def winner_and_attribution(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, int], np.ndarray]:
    winner = load_json(WINNER)
    semantic = load_json(SEMANTIC)
    selected_ids = winner.get("selected_candidate_ids")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != freeze["semantic"]["selected_transactions"]
        or winner.get("selected_candidate_ids_sha256") != freeze["semantic"]["selected_candidate_ids_sha256"]
        or semantic.get("winner_trial_id") != freeze["semantic"]["winner_trial_id"]
        or semantic.get("winner_tuple_id") != freeze["semantic"]["winner_tuple_id"]
    ):
        raise ScoreError("winner drifted after remote-output freeze")
    with np.load(CARRIER, allow_pickle=False) as payload:
        carrier_ids = np.asarray(payload["candidate_id"])
        attribution = np.asarray(payload["attribution"], dtype=np.int64)
    decoded = np.asarray([value.decode("ascii") for value in carrier_ids], dtype="U64")
    index_by_id = {value: index for index, value in enumerate(decoded.tolist())}
    if len(index_by_id) != 92905 or any(value not in index_by_id for value in selected_ids):
        raise ScoreError("winner candidate membership is not exact")
    selected_indices = np.asarray([index_by_id[value] for value in selected_ids], dtype=np.int64)
    selected_attr = attribution[selected_indices]
    if np.any(selected_attr[:, 0] != selected_attr[:, 1] + selected_attr[:, 4] - selected_attr[:, 2] - selected_attr[:, 3]):
        raise ScoreError("carrier attribution conservation drifted")
    totals = selected_attr.sum(axis=0)
    receipt = {
        "net": int(totals[0]),
        "tp_added": int(totals[1]),
        "fp_added": int(totals[2]),
        "tp_removed": int(totals[3]),
        "fp_removed": int(totals[4]),
    }
    if receipt["net"] != int(winner["winner"]["objective_net"]):
        raise ScoreError("winner objective does not equal selected carrier net attribution")
    return winner, selected_ids, receipt, decoded


def verdict(delta: Mapping[str, Any], selected: int, net: int, gate: Mapping[str, Any]) -> tuple[str, str]:
    values = {name: float(delta[name]) for name in ("adjusted", "raw", "node")}
    if any(not math.isfinite(value) for value in values.values()):
        raise ScoreError("non-finite exact graph delta")
    if values["adjusted"] <= 0 or values["raw"] < float(gate["held_fit_exact_graph_raw_delta_min"]) or values["node"] < float(gate["held_fit_exact_graph_node_delta_min"]):
        return "N", "DEAD"
    if values["adjusted"] >= float(gate["held_fit_exact_graph_adjusted_delta_min"]) and selected >= int(gate["selected_transactions_min"]) and net > 0:
        return "Y", "GO"
    return "N", "PARK"


def run() -> dict[str, Any]:
    contract, freeze, prereg = verify_preconditions()
    winner, selected_ids, attribution, carrier_ids = winner_and_attribution(freeze)
    base_contract = load_json(BASE_CONTRACT)
    a1r3.a1r.base.verify_static_inputs(base_contract)
    policy = a1r2.load_lmt_policy(ROOT / "LMT.md")
    telemetry = a1r2.Telemetry(cpu_limit_seconds=None, rss_limit_gib=None)
    monitor = a1r2.LMTResourceMonitor(policy, telemetry)
    workspace = current_lmt.WorkspaceGrowthGuard(ROOT)
    guard = CompositeGuard(monitor, workspace)
    layout = a1r3.LayoutTelemetry(stage="crw_hpo1000_winner_score_not_started")
    scratch = Path(tempfile.mkdtemp(prefix="crw-hpo1000-winner-score-"))
    scratch.rmdir()
    original_rss = lmt_runtime.install()
    bank: a1r3.CompactCandidateBank | None = None
    result: dict[str, Any] | None = None
    try:
        monitor.calibrate()
        monitor.start_background()
        workspace.start()
        identities, baselines, bank, _features, feature_sha, _folds, requests = a1r3.compact_prepare_bank(
            base_contract, guard, scratch, layout
        )
        if (
            len(identities) != 36
            or len(bank) != 92905
            or requests != 45961
            or feature_sha != prereg["bindings"]["component_inventory"]["feature_membership_sha256"]
            or any(bank.candidate_id_at(index) != carrier_ids[index] for index in range(len(bank)))
        ):
            raise ScoreError("reconstructed graph-compiler population drifted from the carrier")
        index_by_id = {bank.candidate_id_at(index): index for index in range(len(bank))}
        selected_indices = [index_by_id[value] for value in selected_ids]
        owners: dict[tuple[str, tuple[Any, ...]], str] = {}
        for index in selected_indices:
            item = bank[index]
            for slot in item.slots:
                key = (item.identity, tuple(slot))
                if key in owners:
                    raise ScoreError(f"winner contains a slot conflict: {owners[key]} / {item.candidate_id}")
                owners[key] = item.candidate_id
        candidate_graphs, graph_receipts = a1r3.a1r.base.build_selected_graphs(
            identities, baselines, bank, selected_indices
        )
        if len(graph_receipts) != 36 or sum(row["selected"] for row in graph_receipts) != len(selected_ids):
            raise ScoreError("compiled graph receipt coverage drifted")
        claim = {
            "schema": "celltrack-alpha.crw-hpo1000-winner-graph-score-claim.v1",
            "status": "CLAIMED_BEFORE_OFFICIAL_SCORER",
            "score_calls_authorized": 1,
            "score_calls_consumed_after_claim": 1,
            "winner_sha256": sha256(WINNER),
            "selected_candidate_ids_sha256": winner["selected_candidate_ids_sha256"],
            "remote_output_freeze_sha256": sha256(FREEZE),
            "score_contract_sha256": sha256(CONTRACT),
            "score_script_sha256": sha256(Path(__file__).resolve()),
            "lmt_sha256": sha256(ROOT / "LMT.md"),
            "cv57_access": False,
            "six_bba_access": False,
            "competition_submission": False,
        }
        atomic_json(CLAIM, claim)
        telemetry.scorer_calls_started += 1
        score = a1r3.a1r.base.score_once(identities, baselines, candidate_graphs, base_contract, guard)
        telemetry.scorer_calls_completed += 1
        telemetry.scientific_metrics_materialized = True
        yn, status = verdict(score["delta"], len(selected_ids), attribution["net"], prereg["promotion_gate"])
        guard.check()
        result = {
            "schema": "celltrack-alpha.crw-hpo1000-local-winner-graph-score.v1",
            "status": "COMPLETE",
            "node": "CRW-HPO1000-KAGGLE-v1",
            "kernel_id": "onebitbrain/g02-crw-hpo1000-v1",
            "kernel_version": 1,
            "winner": {
                "trial_id": winner["winner"]["trial_id"],
                "tuple_id": winner["winner"]["tuple_id"],
                "selected_transactions": len(selected_ids),
                "selected_candidate_ids_sha256": winner["selected_candidate_ids_sha256"],
            },
            "selection_attribution": attribution,
            "graph_receipts": graph_receipts,
            "score": score,
            "verdict": {
                "yn": yn,
                "status": status,
                "gate": prereg["promotion_gate"],
                "may_open_next_held_out_gate": yn == "Y" and status == "GO",
                "competition_submission_authorized": False,
            },
            "dispersion_sha256": sha256(DISPERSION),
            "bindings": {
                "remote_output_freeze_sha256": sha256(FREEZE),
                "score_claim_sha256": sha256(CLAIM),
                "score_contract_sha256": sha256(CONTRACT),
                "score_script_sha256": sha256(Path(__file__).resolve()),
                "lmt_sha256": sha256(ROOT / "LMT.md"),
            },
            "scientific_boundaries": {
                "official_graph_score_calls": 1,
                "cv57_accessed": False,
                "six_bba_accessed": False,
                "competition_submission": False,
                "post_outcome_retuning": False,
                "leakage": False,
                "graphs_valid": True,
            },
            "resources": {
                "rss": monitor.snapshot(),
                "workspace": workspace.snapshot(),
                "layout": layout.snapshot(),
            },
        }
        atomic_json(OUTPUT, result)
        return result
    finally:
        monitor.stop()
        workspace.stop()
        if bank is not None:
            bank.close()
        gc.collect()
        if scratch.exists():
            shutil.rmtree(scratch)
        lmt_runtime.restore(original_rss)


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "score": result["score"]["delta"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

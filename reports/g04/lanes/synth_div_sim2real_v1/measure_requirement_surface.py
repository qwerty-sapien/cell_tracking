#!/usr/bin/env python3
"""Exact post-outcome measurement of the frozen synthetic division outputs.

This is deliberately an audit, not a refit, threshold sweep, or graph action.
It authenticates the sealed score bank before opening the real archive, then
separates parent-event detection from conditional daughter-pair selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for entry in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(entry))
import evaluate_synth_div_sim2real as ev  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
import zarr  # noqa: E402
from zarr.storage import ZipStore  # noqa: E402


def sha256(path: Path) -> str:
    return ev.sha256_file(path)


def gt_dividers(gt: dict) -> tuple[set[int], dict[int, set[int]]]:
    children: dict[int, set[int]] = defaultdict(set)
    for parent, child in np.asarray(gt["edges"], dtype=np.int64).reshape(-1, 2):
        children[int(parent)].add(int(child))
    return {parent for parent, values in children.items() if len(values) == 2}, children


def main(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    # This is the firewall boundary: full frozen-score authentication precedes
    # every GT archive read.
    manifest, records = ev.load_frozen_scores(args.manifest)
    event_scores: list[np.ndarray] = []
    event_labels: list[np.ndarray] = []
    conditional_pair_scores: list[np.ndarray] = []
    conditional_pair_labels: list[np.ndarray] = []
    unreachable = Counter()
    totals = Counter()
    with ZipStore(args.archive, mode="r") as store:
        for record in records:
            arrays = ev._load_candidate_npz(ev._candidate_path(args.manifest, manifest, record.identity), record.graph)
            gt = read_geff_arrays(store, "train", record.dataset)
            group = zarr.open_group(store=store, path=f"train/{record.dataset}.zarr", mode="r")
            scale = tuple(float(x) for x in parse_scale(dict(group.attrs)))
            pred_to_gt = ev._official_node_matching(record.graph, gt, scale)
            inverse = {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()}
            dividers, children = gt_dividers(gt)
            pair_labels, _all_gt, reached = ev.exact_gt_fork_labels(record.graph, arrays, gt, scale)
            starts = np.r_[0, np.flatnonzero(arrays.parent_id[1:] != arrays.parent_id[:-1]) + 1]
            ends = np.r_[starts[1:], len(arrays)]
            for start, end in zip(starts, ends, strict=True):
                parent = int(arrays.parent_id[start])
                gt_parent = pred_to_gt.get(parent)
                block = slice(int(start), int(end))
                probs = arrays.event_probability[block]
                if not np.allclose(probs, probs[0], rtol=0.0, atol=0.0):
                    raise RuntimeError("event score differs within one parent")
                event_scores.append(np.asarray([probs[0]], dtype=np.float64))
                event_positive = gt_parent in dividers
                event_labels.append(np.asarray([event_positive], dtype=np.uint8))
                if event_positive:
                    conditional_pair_scores.append(arrays.pair_probability[block])
                    conditional_pair_labels.append(pair_labels[block].astype(np.uint8))
            for gt_parent in dividers:
                totals["gt_divisions"] += 1
                gt_children = children[gt_parent]
                parent = inverse.get(gt_parent)
                daughters = [inverse.get(child) for child in gt_children]
                if parent is None or any(child is None for child in daughters):
                    unreachable["absent_from_candidate_universe_detection"] += 1
                    continue
                positions = np.flatnonzero(arrays.parent_id == parent)
                if positions.size == 0:
                    unreachable["absent_from_candidate_universe_parent"] += 1
                    continue
                offered = set(arrays.daughter_a_id[positions]) | set(arrays.daughter_b_id[positions])
                if not set(daughters).issubset(offered):
                    unreachable["outside_nearest8"] += 1
                    continue
                key = {int(daughters[0]), int(daughters[1])}
                found = any({int(arrays.daughter_a_id[i]), int(arrays.daughter_b_id[i])} == key for i in positions)
                if not found:
                    unreachable["absent_from_pair_universe"] += 1
            totals["reachable_exact_forks"] += reached
    es = np.concatenate(event_scores)
    el = np.concatenate(event_labels)
    ps = np.concatenate(conditional_pair_scores)
    pl = np.concatenate(conditional_pair_labels)
    event = ev.binary_ranking_metrics(es, el)
    pair = ev.binary_ranking_metrics(ps, pl)
    deployment = json.loads(args.evaluation.read_text())
    n = int(deployment["ranking"]["candidate_count"])
    p = int(deployment["ranking"]["positive_count"])
    total_gt = int(deployment["real_divisions"]["total"])
    # A +0.005 combined-score floor needs division Jaccard >= .05.  At the
    # smallest integral TP count, 5 TP and <=17 FP are necessary (22 actions).
    material_tp, material_fp = 5, 17
    k = material_tp + material_fp
    baseline_rate = total_gt / n
    precision = material_tp / k
    # Worst AP compatible with five positives first appearing at ranks 18..22,
    # divided by the real division deployment rate: a conservative AP-lift
    # floor, not another AUROC threshold.
    ap_floor = sum((i + 1) / (material_fp + i + 1) for i in range(material_tp)) / total_gt
    payload = {
        "schema": "celltrack-alpha.synth-div-requirement-surface.v1",
        "status": "MEASUREMENT_COMPLETE_NO_REFIT",
        "inputs": {"manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "evaluation": str(args.evaluation), "evaluation_sha256": sha256(args.evaluation)},
        "truth_firewall": {"sealed_scores_authenticated_before_gt_open": True, "model_refit": False, "threshold_search": False, "graph_integration": False},
        "deployment_requirement": {
            "combined_materiality_floor": 0.005,
            "required_division_jaccard": 0.05,
            "minimum_integer_operating_point": {"tp": material_tp, "fp_max": material_fp, "k": k, "precision_at_k_min": precision},
            "deployment_pair_base_rate": baseline_rate,
            "minimum_ap": ap_floor,
            "minimum_ap_lift": ap_floor / baseline_rate,
            "auroc_equivalent": "not_identifiable: AUROC can range from near 0.077 to 1.0 for a ranking with five positives in the first 22; it cannot certify FPR <= 17 / 80,013,717.",
            "required_point": {"tpr": material_tp / p, "fpr_max": material_fp / (n - p)},
        },
        "existing_pair_deployment": {**deployment["ranking"], "ap_lift_vs_83_of_n": float(deployment["ranking"]["pr_auc_average_precision"]) / baseline_rate},
        "event_head": event,
        "pair_head_conditional_on_true_dividing_parent": pair,
        "unreachable_divisions": {"total": total_gt - totals["reachable_exact_forks"], "by_cause": dict(unreachable), "r10": "not_applicable: the frozen synthetic generator has no r10 cutoff; it admits deterministic nearest-8 next-frame candidates."},
    }
    if sum(unreachable.values()) != payload["unreachable_divisions"]["total"]:
        raise RuntimeError("unreachable division causes do not conserve the exact evaluation total")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "event": event, "pair": pair, "unreachable": payload["unreachable_divisions"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=root / "fit117_scores_v3/SCORES_MANIFEST.json")
    parser.add_argument("--evaluation", type=Path, default=root / "REAL_EVALUATION_V3.json")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    parser.add_argument("--output", type=Path, default=root / "REQUIREMENT_SURFACE_V1.json")
    main(parser.parse_args())

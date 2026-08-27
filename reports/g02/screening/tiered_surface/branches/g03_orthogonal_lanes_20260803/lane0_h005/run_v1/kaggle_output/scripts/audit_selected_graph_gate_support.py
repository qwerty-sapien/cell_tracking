#!/usr/bin/env python3
"""Audit selected-graph gate support on the full fit-validation candidate pool."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _load_gate,
    _proposals,
)
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    _cache_identity,
    _load_base_scorer,
)
from prepare_selected_graph_conflict_gate import _selected_edges  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-gate-support-audit.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_GATE = ROOT / "reports/g02/selected_graph_conflict_gate_v1/gate.pt"
DEFAULT_GATE_REPORT = ROOT / "reports/g02/selected_graph_conflict_gate_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_conflict_support_v1.json"
SPARSE_RATE = 0.005
MIN_VIDEO_BUDGET = 16


class SupportAuditError(RuntimeError):
    """The fit-validation support audit contract was violated."""


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise SupportAuditError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SupportAuditError(f"refusing to overwrite output: {args.output}")
    prepare = _load(args.prepare)
    fit = _load(args.gate_report)
    scope = prepare.get("scope")
    if (
        prepare.get("schema") != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("validation_video_ids", [])) != 12
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
        or fit.get("scope", {}).get("held_out_used") is not False
    ):
        raise SupportAuditError("inputs are not the exact fit-validation scope")
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact, args.edge_auxiliary, args.support_root, "cpu"
    )
    gate, artifact_threshold = _load_gate(args.gate)
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    videos: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, identity in enumerate(validation_ids, start=1):
        slug = identity.replace(":", "__", 1)
        cache = bh_model.load_sparse_cache(
            args.cache / f"{slug}.bhc",
            expected_identity=_cache_identity(args.cache / f"{slug}.json"),
        )
        selected, graph_report = _selected_edges(cache, scorer)
        proposals, counts = _proposals(cache, scorer, selected, gate, threshold=0.0)
        confidence = np.asarray(
            [float(row["confidence"]) for row in proposals], dtype=np.float64
        )
        budget = max(MIN_VIDEO_BUDGET, int(math.ceil(SPARSE_RATE * len(selected))))
        videos.append(
            {
                "identity": identity,
                "selected_raw_edges": len(selected),
                "sparse_budget": budget,
                "nonconflicting_proposals": len(proposals),
                "confidence_quantiles": {
                    str(quantile): (
                        float(np.quantile(confidence, quantile))
                        if len(confidence)
                        else None
                    )
                    for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
                },
                "graph": graph_report,
                "proposal_counts": counts,
                "_proposal_confidences": confidence,
            }
        )
        print(
            f"support audit {index:02d}/12 {identity} "
            f"pool={len(proposals)} budget={budget}",
            flush=True,
        )
    threshold_rows: list[dict[str, Any]] = []
    for labeled in fit["threshold"]["sweep"]:
        threshold = float(labeled["threshold"])
        per_video = []
        for video in videos:
            confidence = np.asarray(video["_proposal_confidences"])
            accepted = int(np.sum(confidence >= threshold))
            budget = int(video["sparse_budget"])
            per_video.append(
                {
                    "identity": str(video["identity"]),
                    "accepted": accepted,
                    "budget": budget,
                    "within_budget": accepted <= budget,
                    "capped_accepted": min(accepted, budget),
                    "cap_removed": max(accepted - budget, 0),
                }
            )
        threshold_rows.append(
            {
                "threshold": threshold,
                "labeled_precision": float(labeled["precision"]),
                "labeled_accepted": int(labeled["accepted"]),
                "labeled_correct_rescues": int(labeled["correct_rescues"]),
                "labeled_harmful_swaps": int(labeled["harmful_swaps"]),
                "labeled_conservative_utility": int(labeled["conservative_utility"]),
                "all_videos_within_budget": all(
                    bool(row["within_budget"]) for row in per_video
                ),
                "max_budget_ratio": max(
                    row["accepted"] / max(row["budget"], 1) for row in per_video
                ),
                "per_video": per_video,
            }
        )
    admissible = [
        row
        for row in threshold_rows
        if row["labeled_accepted"] >= 10
        and row["labeled_precision"] >= 0.95
        and row["all_videos_within_budget"]
    ]
    selected_threshold = (
        max(
            admissible,
            key=lambda row: (
                row["labeled_conservative_utility"],
                row["labeled_correct_rescues"],
                -row["labeled_harmful_swaps"],
                row["threshold"],
            ),
        )["threshold"]
        if admissible
        else None
    )
    budgeted_admissible = [
        row
        for row in threshold_rows
        if row["labeled_accepted"] >= 10 and row["labeled_precision"] >= 0.95
    ]
    budgeted_selected_threshold = (
        max(float(row["threshold"]) for row in budgeted_admissible)
        if budgeted_admissible
        else None
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "validation_video_ids": list(validation_ids),
            "fit_validation_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "sparse_contract": {
            "rate_of_selected_edges": SPARSE_RATE,
            "minimum_video_budget": MIN_VIDEO_BUDGET,
            "rule": "every validation video must fit its budget before CV57 replay",
        },
        "artifact_threshold": artifact_threshold,
        "selected_threshold": selected_threshold,
        "support_passed": selected_threshold is not None,
        "budgeted_policy": {
            "support_passed": budgeted_selected_threshold is not None,
            "selected_threshold": budgeted_selected_threshold,
            "selection_rule": (
                "highest threshold retaining >=10 labeled proposals at >=0.95 "
                "precision; confidence-ordered node-disjoint proposals capped "
                "per video at the registered sparse budget"
            ),
            "natural_support_required": False,
            "fail_closed_cap_required": True,
        },
        "videos": [
            {name: value for name, value in video.items() if not name.startswith("_")}
            for video in videos
        ],
        "thresholds": threshold_rows,
        "provenance": {
            "prepare": str(args.prepare),
            "gate": str(args.gate),
            "gate_report": str(args.gate_report),
            "edge_artifact": edge_record,
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    value.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "support_passed": report["support_passed"],
                "selected_threshold": report["selected_threshold"],
            },
            sort_keys=True,
        )
    )

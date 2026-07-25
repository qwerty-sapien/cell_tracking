#!/usr/bin/env python3
"""Validate a geometry-prefiltered 3-D NCC graph gate on fit-validation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.relative_geometry_gate import relative_features  # noqa: E402
from audit_selected_graph_appearance import _sha256, ncc  # noqa: E402
from audit_selected_graph_appearance_support import (  # noqa: E402
    _detections,
    _features,
    _load_json,
)
from evaluate_selected_graph_conflict_gate import _load_gate  # noqa: E402
from p4_edge_attention import DEFAULT_CACHE, DEFAULT_ZIP, _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-appearance-combined-audit.v1"
DEFAULT_LABELED = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_SUPPORT = ROOT / "reports/g02/selected_graph_appearance_support_v1.json"
DEFAULT_GATE = ROOT / "reports/g02/selected_graph_background_sparse_w050_v1/gate.pt"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_appearance_combined_v1.json"


class CombinedAuditError(RuntimeError):
    """The combined fit-validation gate contract was violated."""


def _video_proposals(
    value: Mapping[str, Any],
    array: zarr.Array,
    cache: bh_model.SparseInferenceCache,
    gate: torch.nn.Module,
    *,
    geometry_threshold: float,
    ncc_threshold: float,
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels = np.asarray(value["labels"], dtype=np.int8)
    challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
    selected = np.asarray(value["selected_pairs"], dtype=np.int64)
    count = len(labels)
    relative = relative_features(
        np.asarray(value["challenger_features"], dtype=np.float32),
        np.asarray(value["selected_features"], dtype=np.float32),
        np.asarray(value["challenger_geometry_um"], dtype=np.float32),
        np.asarray(value["selected_geometry_um"], dtype=np.float32),
        challenger_logit=np.asarray(value["challenger_logits"], dtype=np.float32),
        winner_logit=np.asarray(value["selected_logits"], dtype=np.float32),
        dummy_logit=0.0,
        candidate_count=np.full(count, 2, dtype=np.float32),
    )
    with torch.no_grad():
        geometry_probabilities = torch.sigmoid(
            gate(torch.from_numpy(relative))
        ).numpy()
    detection_by_id, _detection_by_frame = _detections(cache)
    needed = set(challengers.ravel().tolist()) | set(selected.ravel().tolist())
    features = _features(array, needed, detection_by_id)
    margins = np.asarray(
        [
            ncc(features[int(challenger[0])], features[int(challenger[1])])
            - ncc(features[int(winner[0])], features[int(winner[1])])
            for challenger, winner in zip(challengers, selected, strict=True)
        ],
        dtype=np.float64,
    )
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, pair in enumerate(challengers.tolist()):
        grouped[(int(pair[0]), int(pair[1]))].append(index)
    raw: list[dict[str, Any]] = []
    for challenger in sorted(grouped):
        indices = np.asarray(grouped[challenger], dtype=np.int64)
        proposal_labels = {int(labels[index]) for index in indices}
        if len(proposal_labels) != 1:
            raise CombinedAuditError("one challenger has inconsistent labels")
        geometry_confidence = float(np.min(geometry_probabilities[indices]))
        ncc_confidence = float(np.min(margins[indices]))
        if (
            geometry_confidence < geometry_threshold
            or ncc_confidence < ncc_threshold
        ):
            continue
        conflicts = {
            tuple(int(value) for value in selected[index])
            for index in indices.tolist()
        }
        raw.append(
            {
                "challenger": challenger,
                "conflicts": sorted(conflicts),
                "label": proposal_labels.pop(),
                "geometry_confidence": geometry_confidence,
                "ncc_confidence": ncc_confidence,
            }
        )
    accepted: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for proposal in sorted(
        raw,
        key=lambda row: (
            -float(row["ncc_confidence"]),
            -float(row["geometry_confidence"]),
            row["challenger"],
        ),
    ):
        affected = set(proposal["challenger"])
        for conflict in proposal["conflicts"]:
            affected.update(conflict)
        if occupied.intersection(affected):
            continue
        occupied.update(affected)
        accepted.append(proposal)
        if len(accepted) >= budget:
            break
    return accepted, {
        "rows": count,
        "raw_threshold_proposals": len(raw),
        "accepted": len(accepted),
        "correct_rescues": sum(int(row["label"]) == 1 for row in accepted),
        "harmful_swaps": sum(int(row["label"]) == 0 for row in accepted),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CombinedAuditError(f"refusing to overwrite output: {args.output}")
    labeled = _load_json(args.labeled)
    support = _load_json(args.appearance_support)
    scope = labeled.get("scope")
    if (
        labeled.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or support.get("schema")
        != "celltrack-alpha.selected-graph-appearance-support.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or support.get("scope", {}).get("validation_video_ids")
        != scope.get("validation_video_ids")
    ):
        raise CombinedAuditError("combined inputs do not share exact fit-only scope")
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    by_id = {
        str(row["identity"]): {
            "path": ROOT / str(row["path"]),
            "selected_raw_edges": int(row["graph"]["raw_selected_edges"]),
        }
        for row in labeled["videos"]
        if str(row["identity"]) in validation_ids
    }
    if set(by_id) != set(validation_ids):
        raise CombinedAuditError("validation artifacts are incomplete")
    gate, artifact_threshold = _load_gate(args.geometry_gate)
    geometry_threshold = (
        artifact_threshold
        if args.geometry_threshold is None
        else float(args.geometry_threshold)
    )
    ncc_threshold = float(support["threshold"]["support_adjusted"])
    videos: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(validation_ids, start=1):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            value = torch.load(
                by_id[identity]["path"], map_location="cpu", weights_only=False
            )
            budget = max(
                args.minimum_video_budget,
                int(
                    math.ceil(
                        args.max_proposal_rate
                        * by_id[identity]["selected_raw_edges"]
                    )
                ),
            )
            accepted, counts = _video_proposals(
                value,
                array,
                cache,
                gate,
                geometry_threshold=geometry_threshold,
                ncc_threshold=ncc_threshold,
                budget=budget,
            )
            videos.append(
                {
                    "identity": identity,
                    "budget": budget,
                    **counts,
                    "proposals": accepted,
                }
            )
            print(
                f"combined appearance {index:02d}/12 {identity} "
                f"accepted={counts['accepted']}",
                flush=True,
            )
    accepted = sum(int(row["accepted"]) for row in videos)
    correct = sum(int(row["correct_rescues"]) for row in videos)
    harmful = sum(int(row["harmful_swaps"]) for row in videos)
    precision = float(correct / accepted) if accepted else 0.0
    support_passed = (
        accepted >= args.minimum_validation_accepted
        and precision >= args.minimum_precision
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "configuration": {
            "geometry_gate": str(args.geometry_gate),
            "geometry_artifact_threshold": artifact_threshold,
            "geometry_threshold": geometry_threshold,
            "ncc_threshold": ncc_threshold,
            "max_proposal_rate": args.max_proposal_rate,
            "minimum_video_budget": args.minimum_video_budget,
            "ordering": "NCC margin, geometry probability, challenger identity",
            "node_disjoint": True,
        },
        "validation": {
            "accepted": accepted,
            "correct_rescues": correct,
            "harmful_swaps": harmful,
            "precision": precision,
            "conservative_utility": correct - 2 * harmful,
            "videos": videos,
        },
        "support_passed": support_passed,
        "contract": {
            "minimum_validation_accepted": args.minimum_validation_accepted,
            "minimum_precision": args.minimum_precision,
            "full_pool_natural_support": False,
            "runtime_cap_required": True,
        },
        "provenance": {
            "labeled": str(args.labeled),
            "labeled_sha256": _sha256(args.labeled),
            "appearance_support": str(args.appearance_support),
            "appearance_support_sha256": _sha256(args.appearance_support),
            "geometry_gate_sha256": _sha256(args.geometry_gate),
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
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    value.add_argument("--appearance-support", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--geometry-gate", type=Path, default=DEFAULT_GATE)
    value.add_argument("--geometry-threshold", type=float)
    value.add_argument("--max-proposal-rate", type=float, default=0.005)
    value.add_argument("--minimum-video-budget", type=int, default=16)
    value.add_argument("--minimum-validation-accepted", type=int, default=10)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "support_passed": report["support_passed"],
                "validation": {
                    key: report["validation"][key]
                    for key in (
                        "accepted",
                        "correct_rescues",
                        "harmful_swaps",
                        "precision",
                        "conservative_utility",
                    )
                },
            },
            sort_keys=True,
        )
    )

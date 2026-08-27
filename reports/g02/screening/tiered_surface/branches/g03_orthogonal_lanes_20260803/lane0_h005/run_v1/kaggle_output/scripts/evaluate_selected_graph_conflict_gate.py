#!/usr/bin/env python3
"""Evaluate a frozen selected-graph conflict gate on the registered top-10."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
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
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.relative_geometry_gate import (  # noqa: E402
    RelativeGeometryGate,
    relative_features,
)
from evaluate_relative_geometry_swap_gate import REGISTERED_TOP10  # noqa: E402
from p4_edge_attention_cv57 import _solve  # noqa: E402
from prepare_selected_graph_conflict_gate import _selected_edges  # noqa: E402
from audit_selected_graph_appearance import (  # noqa: E402
    extract_patch,
    ncc,
    unit_vector,
)


SCHEMA = "celltrack-alpha.selected-graph-conflict-screen.v1"
DEFAULT_GATE = ROOT / "reports/g02/selected_graph_conflict_gate_v1/gate.pt"
DEFAULT_GATE_REPORT = ROOT / "reports/g02/selected_graph_conflict_gate_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_conflict_screen10_v1"
DEFAULT_APPEARANCE_REPORT = (
    ROOT / "reports/g02/selected_graph_appearance_combined_v1.json"
)
PROMOTED_PROBABILITY = 1.0 - 1.0e-6
DEMOTED_PROBABILITY = 1.0e-6


class SelectedGraphEvalError(RuntimeError):
    """The selected-graph affected-slice contract was violated."""


def _load_gate(path: Path) -> tuple[RelativeGeometryGate, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema")
        not in {
            "celltrack-alpha.selected-graph-conflict-gate.v1",
            "celltrack-alpha.selected-graph-conflict-gate.v2",
        }
    ):
        raise SelectedGraphEvalError("selected-graph gate artifact is invalid")
    model = RelativeGeometryGate(hidden=int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, float(payload["threshold"])


def _greedy_nonconflicting(
    proposals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain high-confidence proposals with disjoint affected graph nodes."""

    accepted: list[dict[str, Any]] = []
    occupied_nodes: set[int] = set()
    for raw in sorted(
        proposals,
        key=lambda row: (
            -float(row["confidence"]),
            tuple(int(value) for value in row["challenger"]),
        ),
    ):
        challenger = tuple(int(value) for value in raw["challenger"])
        conflicts = [
            tuple(int(value) for value in pair) for pair in raw["conflicts"]
        ]
        affected = {challenger[0], challenger[1]}
        for source, target in conflicts:
            affected.add(source)
            affected.add(target)
        if occupied_nodes.intersection(affected):
            continue
        occupied_nodes.update(affected)
        accepted.append(
            {
                "challenger": list(challenger),
                "conflicts": [list(pair) for pair in conflicts],
                "confidence": float(raw["confidence"]),
                "row_probabilities": [
                    float(value) for value in raw["row_probabilities"]
                ],
                **(
                    {
                        "appearance_margins": [
                            float(value)
                            for value in raw["appearance_margins"]
                        ],
                        "geometry_confidence": float(
                            raw["geometry_confidence"]
                        ),
                    }
                    if "appearance_margins" in raw
                    else {}
                ),
            }
        )
    return accepted


def _cap_proposals(
    proposals: Sequence[Mapping[str, Any]], maximum: int | None
) -> list[dict[str, Any]]:
    """Apply a fail-closed per-video cap after confidence-ordered greediness."""

    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise SelectedGraphEvalError("proposal cap must be a positive integer")
    rows = [dict(row) for row in proposals]
    return rows if maximum is None else rows[:maximum]


def _proposal_feature_groups(
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
    selected: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build every selected-edge conflict proposal before classification."""

    selected_out = {source: target for source, target in selected}
    selected_in = {target: source for source, target in selected}
    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            detections[int(node_id)] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    scale = np.asarray(scorer.config.scale_zyx, dtype=np.float64)
    feature_by_pair: dict[tuple[int, int], np.ndarray] = {}
    logit_by_pair: dict[tuple[int, int], float] = {}
    geometry_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for context, source, target in zip(
        cache.contexts, cache.detections[:-1], cache.detections[1:], strict=True
    ):
        previous_velocity: dict[int, np.ndarray | None] = {}
        for node_id, coordinate in zip(
            source.node_ids, source.coords_zyx, strict=True
        ):
            source_id = int(node_id)
            parent_id = selected_in.get(source_id)
            parent = detections.get(parent_id) if parent_id is not None else None
            if parent is None or parent[0] != int(source.frame_index) - 1:
                previous_velocity[source_id] = None
            else:
                previous_velocity[source_id] = (
                    np.asarray(coordinate, dtype=np.float64) - parent[1]
                ) * scale
        candidates, checkpoint, features = scorer.candidate_features(
            context,
            source,
            target,
            previous_velocity_um=previous_velocity,
        )
        with torch.no_grad():
            logits = checkpoint + scorer.calibrator(features)
        feature_rows = features.detach().cpu().numpy()
        logit_rows = logits.detach().cpu().numpy()
        for row, (source_id, target_id) in enumerate(
            zip(candidates.source_ids, candidates.target_ids, strict=True)
        ):
            pair = (int(source_id), int(target_id))
            if pair in feature_by_pair:
                raise SelectedGraphEvalError("candidate pair repeats across contexts")
            feature_by_pair[pair] = np.asarray(feature_rows[row], dtype=np.float32)
            logit_by_pair[pair] = float(logit_rows[row])
            source_record = detections[pair[0]]
            target_record = detections[pair[1]]
            geometry_by_pair[pair] = np.concatenate(
                (source_record[1] * scale, target_record[1] * scale)
            ).astype(np.float32, copy=False)
    groups: list[dict[str, Any]] = []
    compared_rows = 0
    for challenger in sorted(set(feature_by_pair) - selected):
        conflicts: list[tuple[int, int]] = []
        source_target = selected_out.get(challenger[0])
        if source_target is not None and source_target != challenger[1]:
            conflict = (challenger[0], source_target)
            if conflict in feature_by_pair:
                conflicts.append(conflict)
        target_source = selected_in.get(challenger[1])
        if target_source is not None and target_source != challenger[0]:
            conflict = (target_source, challenger[1])
            if conflict in feature_by_pair and conflict not in conflicts:
                conflicts.append(conflict)
        if not conflicts:
            continue
        count = len(conflicts)
        relative = relative_features(
            np.repeat(feature_by_pair[challenger][None], count, axis=0),
            np.stack([feature_by_pair[pair] for pair in conflicts]),
            np.repeat(geometry_by_pair[challenger][None], count, axis=0),
            np.stack([geometry_by_pair[pair] for pair in conflicts]),
            challenger_logit=np.full(
                count, logit_by_pair[challenger], dtype=np.float32
            ),
            winner_logit=np.asarray(
                [logit_by_pair[pair] for pair in conflicts], dtype=np.float32
            ),
            dummy_logit=0.0,
            candidate_count=np.full(count, 2, dtype=np.float32),
        )
        compared_rows += count
        groups.append(
            {
                "challenger": list(challenger),
                "conflicts": [list(pair) for pair in conflicts],
                "relative_features": relative,
            }
        )
    return groups, {
        "candidate_pairs": len(feature_by_pair),
        "selected_raw_edges": len(selected),
        "compared_rows": compared_rows,
        "raw_proposals": len(groups),
    }


def _proposals(
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
    selected: set[tuple[int, int]],
    gate: RelativeGeometryGate,
    *,
    threshold: float,
    maximum: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups, counts = _proposal_feature_groups(cache, scorer, selected)
    raw: list[dict[str, Any]] = []
    for group in groups:
        relative = np.asarray(group["relative_features"], dtype=np.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(gate(torch.from_numpy(relative))).numpy()
        confidence = float(np.min(probabilities))
        if confidence >= threshold:
            raw.append(
                {
                    "challenger": list(group["challenger"]),
                    "conflicts": [list(pair) for pair in group["conflicts"]],
                    "confidence": confidence,
                    "row_probabilities": probabilities.tolist(),
                }
            )
    accepted_before_cap = _greedy_nonconflicting(raw)
    accepted = _cap_proposals(accepted_before_cap, maximum)
    return accepted, {
        **counts,
        "threshold_proposals": len(raw),
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": maximum is not None,
        "cap_removed": len(accepted_before_cap) - len(accepted),
        "accepted_source_only": sum(len(row["conflicts"]) == 1 for row in accepted),
        "accepted_two_endpoint": sum(len(row["conflicts"]) == 2 for row in accepted),
    }


def _appearance_proposals(
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
    selected: set[tuple[int, int]],
    gate: RelativeGeometryGate,
    image: Any,
    *,
    geometry_threshold: float,
    appearance_threshold: float,
    maximum: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the frozen geometry prefilter, then full-3D NCC comparison."""

    groups, counts = _proposal_feature_groups(cache, scorer, selected)
    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            detections[int(node_id)] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    filtered_by_frame: dict[int, list[dict[str, Any]]] = {}
    geometry_count = 0
    for group in groups:
        relative = np.asarray(group["relative_features"], dtype=np.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(gate(torch.from_numpy(relative))).numpy()
        geometry_confidence = float(np.min(probabilities))
        if geometry_confidence < geometry_threshold:
            continue
        geometry_count += 1
        challenger = tuple(int(value) for value in group["challenger"])
        source_t = detections[challenger[0]][0]
        if detections[challenger[1]][0] != source_t + 1:
            raise SelectedGraphEvalError("appearance challenger is not adjacent")
        filtered_by_frame.setdefault(source_t, []).append(
            {
                "challenger": challenger,
                "conflicts": [
                    tuple(int(value) for value in pair)
                    for pair in group["conflicts"]
                ],
                "geometry_confidence": geometry_confidence,
                "row_probabilities": probabilities,
            }
        )

    raw: list[dict[str, Any]] = []
    extracted_nodes = 0
    for source_t in sorted(filtered_by_frame):
        frame_groups = filtered_by_frame[source_t]
        needed_by_time: dict[int, set[int]] = {
            source_t: set(),
            source_t + 1: set(),
        }
        for group in frame_groups:
            for pair in (group["challenger"], *group["conflicts"]):
                for node_id in pair:
                    time_index = detections[node_id][0]
                    if time_index not in needed_by_time:
                        raise SelectedGraphEvalError(
                            "appearance conflict escaped its frame pair"
                        )
                    needed_by_time[time_index].add(node_id)
        vectors: dict[int, np.ndarray] = {}
        for time_index in (source_t, source_t + 1):
            frame = np.asarray(image[time_index])
            for node_id in sorted(needed_by_time[time_index]):
                vectors[node_id] = unit_vector(
                    extract_patch(frame, detections[node_id][1])
                )
        extracted_nodes += len(vectors)
        pair_ncc: dict[tuple[int, int], float] = {}

        def score(pair: tuple[int, int]) -> float:
            if pair not in pair_ncc:
                pair_ncc[pair] = ncc(vectors[pair[0]], vectors[pair[1]])
            return pair_ncc[pair]

        for group in frame_groups:
            challenger_score = score(group["challenger"])
            margins = [
                challenger_score - score(pair) for pair in group["conflicts"]
            ]
            confidence = float(min(margins))
            if confidence < appearance_threshold:
                continue
            raw.append(
                {
                    "challenger": list(group["challenger"]),
                    "conflicts": [list(pair) for pair in group["conflicts"]],
                    "confidence": confidence,
                    "appearance_margins": margins,
                    "geometry_confidence": group["geometry_confidence"],
                    "row_probabilities": group["row_probabilities"].tolist(),
                }
            )
    accepted_before_cap = _greedy_nonconflicting(raw)
    accepted = _cap_proposals(accepted_before_cap, maximum)
    return accepted, {
        **counts,
        "geometry_threshold_proposals": geometry_count,
        "appearance_threshold_proposals": len(raw),
        "appearance_extracted_node_visits": extracted_nodes,
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": maximum is not None,
        "cap_removed": len(accepted_before_cap) - len(accepted),
        "accepted_source_only": sum(
            len(row["conflicts"]) == 1 for row in accepted
        ),
        "accepted_two_endpoint": sum(
            len(row["conflicts"]) == 2 for row in accepted
        ),
    }


def _apply(
    probabilities: Mapping[tuple[int, int], float],
    no_link: Mapping[int, float],
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    updated = dict(probabilities)
    updated_no_link = dict(no_link)
    for proposal in proposals:
        challenger = tuple(int(value) for value in proposal["challenger"])
        if challenger not in updated:
            raise SelectedGraphEvalError("accepted challenger is absent from probabilities")
        updated[challenger] = PROMOTED_PROBABILITY
        updated_no_link[challenger[0]] = DEMOTED_PROBABILITY
        for pair in proposal["conflicts"]:
            conflict = tuple(int(value) for value in pair)
            if conflict not in updated:
                raise SelectedGraphEvalError("selected conflict is absent from probabilities")
            updated[conflict] = DEMOTED_PROBABILITY
    return updated, updated_no_link


def _realization(fragment: Path, proposals: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    with fragment.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        edges = {
            (int(row["source_id"]), int(row["target_id"]))
            for row in rows
            if row["row_type"] == "edge"
        }
    challengers = {
        tuple(int(value) for value in proposal["challenger"])
        for proposal in proposals
    }
    conflicts = {
        tuple(int(value) for value in pair)
        for proposal in proposals
        for pair in proposal["conflicts"]
    }
    return {
        "proposed_challengers": len(challengers),
        "realized_challengers": len(challengers.intersection(edges)),
        "proposed_conflicts": len(conflicts),
        "retained_conflicts": len(conflicts.intersection(edges)),
    }


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: int(sum(float(row[name]) for row in rows))
        for name in ("edge_tp", "edge_fp", "edge_fn")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SelectedGraphEvalError(f"refusing to overwrite output: {args.output}")
    if (
        args.threshold_override is not None
        and not 0.0 <= args.threshold_override <= 1.0
    ):
        raise SelectedGraphEvalError("threshold override must be within [0, 1]")
    if args.max_proposal_rate is not None and not 0.0 < args.max_proposal_rate <= 1.0:
        raise SelectedGraphEvalError("maximum proposal rate must be within (0, 1]")
    if args.minimum_video_budget < 1:
        raise SelectedGraphEvalError("minimum video budget must be positive")
    gate_report = json.loads(args.gate_report.read_text(encoding="ascii"))
    if (
        gate_report.get("scope", {}).get("held_out_used") is not False
        or gate_report.get("scope", {}).get("audit_used") is not False
        or gate_report.get("scope", {}).get("test_used") is not False
    ):
        raise SelectedGraphEvalError("gate artifact is not fit-only")
    appearance_report: Mapping[str, Any] | None = None
    appearance_threshold: float | None = None
    if args.appearance_report is not None:
        appearance_report = json.loads(
            args.appearance_report.read_text(encoding="ascii")
        )
        appearance_scope = appearance_report.get("scope", {})
        if (
            appearance_report.get("schema")
            != "celltrack-alpha.selected-graph-appearance-combined-audit.v1"
            or appearance_report.get("support_passed") is not True
            or appearance_report.get("contract", {}).get(
                "runtime_cap_required"
            )
            is not True
            or any(
                appearance_scope.get(key) is not False
                for key in ("held_out_used", "audit_used", "test_used")
            )
        ):
            raise SelectedGraphEvalError(
                "appearance support report is not fit-only and admissible"
            )
        appearance_threshold = float(
            appearance_report["configuration"]["ncc_threshold"]
        )
    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=scoreall.archive_train_video_ids(args.zip),
    )
    if args.selection == "top10":
        selected_ids = tuple(
            identity for identity in cv57 if identity in set(REGISTERED_TOP10)
        )
        if set(selected_ids) != set(REGISTERED_TOP10):
            raise SelectedGraphEvalError("registered top-10 identity mismatch")
        selection_role = "preregistered-worst-residual-top10"
    else:
        selected_ids = tuple(cv57)
        if len(selected_ids) != 57:
            raise SelectedGraphEvalError("full confirm is not exact CV57")
        selection_role = "authorized-full-cv57-confirm"
    baseline_names = {
        identity.replace(":", "__", 1) + ".csv" for identity in selected_ids
    }
    baseline_manifest = scoreall.validate_fragment_set(
        [
            path
            for path in scoreall.resolve_fragment_paths(str(args.baseline_fragments))
            if path.name in baseline_names
        ],
        selected_ids,
    )
    cache_paths = {
        path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected_ids).issubset(cache_paths) or not set(selected_ids).issubset(
        meta_paths
    ):
        raise SelectedGraphEvalError("CV57 cache coverage is incomplete")
    base_model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise SelectedGraphEvalError("frozen edge scorer did not restore")
    gate, artifact_threshold = _load_gate(args.gate)
    threshold = (
        float(args.threshold_override)
        if args.threshold_override is not None
        else artifact_threshold
    )
    if appearance_report is not None and not math.isclose(
        threshold,
        float(appearance_report["configuration"]["geometry_threshold"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SelectedGraphEvalError(
            "geometry threshold differs from the combined support report"
        )
    if appearance_report is not None and args.max_proposal_rate is None:
        raise SelectedGraphEvalError(
            "appearance screen requires the registered fail-closed cap"
        )
    args.output.mkdir(parents=True)
    fragments = args.output / "fragments"
    fragments.mkdir()
    videos: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected_ids, start=1):
            dataset = identity.split(":", 1)[1]
            meta = sweep._load(meta_paths[identity], f"cache metadata for {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=sweep._cache_identity(meta)
            )
            probabilities, no_link, _association = run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=bh_edge
            )
            selected, baseline_graph = _selected_edges(cache, scorer)
            proposal_cap = (
                max(
                    args.minimum_video_budget,
                    int(math.ceil(args.max_proposal_rate * len(selected))),
                )
                if args.max_proposal_rate is not None
                else None
            )
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            if appearance_threshold is None:
                proposals, proposal_counts = _proposals(
                    cache,
                    scorer,
                    selected,
                    gate,
                    threshold=threshold,
                    maximum=proposal_cap,
                )
            else:
                proposals, proposal_counts = _appearance_proposals(
                    cache,
                    scorer,
                    selected,
                    gate,
                    image,
                    geometry_threshold=threshold,
                    appearance_threshold=appearance_threshold,
                    maximum=proposal_cap,
                )
            modified, modified_no_link = _apply(probabilities, no_link, proposals)
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            solved = _solve(
                identity=identity,
                dataset=dataset,
                shape=tuple(int(value) for value in image.shape),
                graph_nodes=graph_nodes,
                probabilities=modified,
                no_link=modified_no_link,
                refiner=sweep.MemoizedImageRefiner(image),
                fragment_dir=fragments,
            )
            realization = _realization(Path(solved["fragment"]), proposals)
            videos.append(
                {
                    "identity": identity,
                    "baseline_graph": baseline_graph,
                    "proposal_counts": proposal_counts,
                    "proposals": proposals,
                    "realization": realization,
                    **solved,
                }
            )
            for name, value in {**proposal_counts, **realization}.items():
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    totals[name] = totals.get(name, 0) + int(value)
            print(
                f"selected graph screen {index:02d}/{len(selected_ids)} {identity} "
                f"accepted={len(proposals)} realized={realization['realized_challengers']}",
                flush=True,
            )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(fragments)), selected_ids
        )
        baseline_rows, baseline_provenance, baseline_coverage = scoreall.score_fragment_input(
            store, baseline_manifest
        )
        candidate_rows, candidate_provenance, candidate_coverage = scoreall.score_fragment_input(
            store, candidate_manifest
        )
    baseline_by_id = {row["identity"]: row for row in baseline_rows}
    candidate_by_id = {row["identity"]: row for row in candidate_rows}
    ordered_baseline = [baseline_by_id[identity] for identity in selected_ids]
    ordered_candidate = [candidate_by_id[identity] for identity in selected_ids]
    baseline_summary = scoreall.official_summary(ordered_baseline)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    delta = float(candidate_summary["adj_edge_jaccard"]) - float(
        baseline_summary["adj_edge_jaccard"]
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "role": selection_role,
            "video_count": len(selected_ids),
            "video_ids": list(selected_ids),
        },
        "configuration": {
            "threshold": threshold,
            "artifact_threshold": artifact_threshold,
            "threshold_overridden": args.threshold_override is not None,
            "max_proposal_rate": args.max_proposal_rate,
            "minimum_video_budget": (
                args.minimum_video_budget
                if args.max_proposal_rate is not None
                else None
            ),
            "promoted_probability": PROMOTED_PROBABILITY,
            "demoted_probability": DEMOTED_PROBABILITY,
            "motion_direction": "backward",
            "motion_error_weight": 0.25,
            "division_enabled": False,
            "gnn": False,
            "appearance_mode": appearance_threshold is not None,
            "appearance_threshold": appearance_threshold,
            "appearance_patch_shape": (
                [16, 32, 32] if appearance_threshold is not None else None
            ),
        },
        "baseline": {
            "summary": baseline_summary,
            "edge_counts": _edge_counts(ordered_baseline),
            "coverage": baseline_coverage,
            "input_provenance": baseline_provenance,
        },
        "candidate": {
            "summary": candidate_summary,
            "edge_counts": _edge_counts(ordered_candidate),
            "coverage": candidate_coverage,
            "input_provenance": candidate_provenance,
            "totals": totals,
            "videos": videos,
        },
        "decision": {
            "delta_adjusted_edge_jaccard": delta,
            "proxy_gate": 0.005,
            "status": "CONFIRM" if delta >= 0.005 else "PARK",
            "full_cv57_authorized": delta >= 0.005,
        },
        "provenance": {
            "gate": str(args.gate),
            "gate_sha256": sweep._sha256(args.gate),
            "gate_report": str(args.gate_report),
            "gate_report_sha256": sweep._sha256(args.gate_report),
            "appearance_report": (
                str(args.appearance_report)
                if args.appearance_report is not None
                else None
            ),
            "appearance_report_sha256": (
                sweep._sha256(args.appearance_report)
                if args.appearance_report is not None
                else None
            ),
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=sweep.DEFAULT_CACHE)
    value.add_argument("--baseline-fragments", type=Path, default=sweep.DEFAULT_BASELINE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    value.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    value.add_argument("--appearance-report", type=Path)
    value.add_argument(
        "--selection", choices=("top10", "full57"), default="top10"
    )
    value.add_argument("--threshold-override", type=float)
    value.add_argument("--max-proposal-rate", type=float)
    value.add_argument("--minimum-video-budget", type=int, default=16)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": report["status"], "decision": report["decision"]}))

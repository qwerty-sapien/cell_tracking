#!/usr/bin/env python3
"""Validate a held-fit surrogate-to-graph gate on the parked temporal ranker."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
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
from celltrack_alpha import (  # noqa: E402
    appearance_swap,
    bh_edge,
    bh_graph,
    bh_model,
    temporal_rank_swap,
)
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _edge_counts,
    _realization,
)
from evaluate_selected_graph_direct_swap import _solve_direct  # noqa: E402
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    _fit_model,
    _fold,
    _video_data,
)
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _metrics,
    _model_scores,
    _select,
)


SCHEMA = "celltrack-alpha.temporal-rank-oof-graph-gate.v1"
CONTRACT_SCHEMA = "celltrack-alpha.temporal-rank-oof-graph-gate-contract.v1"
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/temporal_rank_oof_graph_gate_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "9e73a87c7427288d23d32715327eb0296a48b9dcf5f82e738fd2e100273b6e70"
)
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_OUTPUT = ROOT / "reports/g02/temporal_rank_oof_graph_gate_v1"
FAMILY = "6bba"


class TemporalOOFGraphGateError(RuntimeError):
    """The held-fit graph-gate contract or artifact identity was violated."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporalOOFGraphGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TemporalOOFGraphGateError(f"{label} must be a JSON object")
    return value


def _root_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TemporalOOFGraphGateError(f"{label} path is missing")
    path = ROOT / value
    if not path.is_file():
        raise TemporalOOFGraphGateError(f"{label} is missing: {path}")
    return path


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and validate the immutable diagnostic contract."""

    if len(expected_sha256) != 64 or sweep._sha256(path) != expected_sha256:
        raise TemporalOOFGraphGateError("graph-gate contract SHA-256 mismatch")
    value = _load_json(path, "graph-gate contract")
    scope = value.get("scope")
    candidate = value.get("candidate")
    control = value.get("control")
    data = value.get("data")
    graph_gate = value.get("graph_gate")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or value.get("purpose")
        != "retrospective-surrogate-to-graph-gate-validation"
        or not isinstance(scope, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(control, Mapping)
        or not isinstance(data, Mapping)
        or not isinstance(graph_gate, Mapping)
    ):
        raise TemporalOOFGraphGateError("graph-gate contract is malformed")
    if (
        scope.get("family") != FAMILY
        or scope.get("source_partition") != "fit-train105"
        or int(scope.get("fold_count", 0)) != 5
        or scope.get("validation12_access") is not False
        or scope.get("cv57_access") is not False
        or scope.get("promotion_authorized") is not False
        or any(
            scope.get(name) is not False
            for name in ("kaggle_action", "gpu_action", "runpod_action")
        )
    ):
        raise TemporalOOFGraphGateError("graph-gate scope is not fail-closed")
    configuration = candidate.get("configuration")
    threshold = candidate.get("threshold_selection")
    prerequisite = candidate.get("proposal_prerequisite")
    if (
        not isinstance(configuration, Mapping)
        or dict(configuration)
        != {
            "name": "linear-bg005",
            "hidden": 0,
            "background_weight": 0.05,
            "epochs": 40,
            "seed_base": 1207,
            "configuration_index": 0,
            "family_index": 1,
        }
        or candidate.get("fold_assignment")
        != "sha256(video identity)[0:8] modulo 5"
        or not isinstance(threshold, Mapping)
        or threshold.get("fit_partition_only") is not True
        or int(threshold.get("minimum_fit_accepted", 0)) != 25
        or float(threshold.get("minimum_fit_precision", 0.0)) != 0.95
        or not isinstance(prerequisite, Mapping)
        or int(prerequisite.get("minimum_oof_accepted", 0)) != 25
        or float(prerequisite.get("minimum_oof_precision", 0.0)) != 0.95
    ):
        raise TemporalOOFGraphGateError("candidate reproduction contract drifted")
    if (
        float(graph_gate.get("minimum_adjusted_edge_jaccard_delta", 0.0))
        != 0.005
        or graph_gate.get("require_nonnegative_raw_edge_jaccard_delta")
        is not True
        or float(graph_gate.get("minimum_node_recall_delta", -1.0)) != 0.0
    ):
        raise TemporalOOFGraphGateError("graph promotion contract drifted")
    return value


def held_fold_partitions(
    identities: Sequence[str], fold_count: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return deterministic fit/held partitions and assert one held use each."""

    ordered = tuple(str(value) for value in identities)
    if len(ordered) != len(set(ordered)) or not ordered:
        raise TemporalOOFGraphGateError("eligible identities are empty or duplicated")
    result = []
    observed: list[str] = []
    for fold in range(fold_count):
        fit_ids = tuple(
            identity
            for identity in ordered
            if _fold(identity, fold_count) != fold
        )
        held_ids = tuple(
            identity
            for identity in ordered
            if _fold(identity, fold_count) == fold
        )
        if not fit_ids or not held_ids or set(fit_ids).intersection(held_ids):
            raise TemporalOOFGraphGateError(f"invalid partition for fold {fold}")
        result.append((fit_ids, held_ids))
        observed.extend(held_ids)
    if sorted(observed) != sorted(ordered):
        raise TemporalOOFGraphGateError(
            "each eligible identity must be held exactly once"
        )
    return result


def validate_fit_scope(
    source_scope: Mapping[str, Any],
    prepare_scope: Mapping[str, Any],
    cv57: set[str],
) -> tuple[str, ...]:
    """Validate the sealed 6bba train105 diagnostic scope."""

    for name, scope in (("source", source_scope), ("prepare", prepare_scope)):
        if (
            scope.get("fit_only") is not True
            or scope.get("audit_used") is not False
            or scope.get("held_out_used") is not False
            or scope.get("test_used") is not False
        ):
            raise TemporalOOFGraphGateError(f"{name} scope is not fit-only")
    source_train = tuple(str(value) for value in source_scope["train_video_ids"])
    prepare_train = tuple(str(value) for value in prepare_scope["train_video_ids"])
    source_validation = set(
        str(value) for value in source_scope["validation_video_ids"]
    )
    prepare_validation = set(
        str(value) for value in prepare_scope["validation_video_ids"]
    )
    if (
        set(source_train) != set(prepare_train)
        or source_validation != prepare_validation
        or len(source_train) != 105
        or len(source_validation) != 12
        or set(source_train).intersection(source_validation)
    ):
        raise TemporalOOFGraphGateError("source and prepared 105/12 scopes disagree")
    eligible = tuple(
        identity
        for identity in source_train
        if identity.startswith(f"train:{FAMILY}_")
    )
    if len(eligible) != 69:
        raise TemporalOOFGraphGateError(
            f"expected 69 eligible {FAMILY} videos, found {len(eligible)}"
        )
    forbidden = set(eligible).intersection(source_validation | cv57)
    if forbidden:
        raise TemporalOOFGraphGateError(
            f"eligible fit scope overlaps sealed videos: {sorted(forbidden)}"
        )
    return eligible


def graph_gate_decision(
    control_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered material graph-improvement rule."""

    adjusted_delta = float(candidate_summary["adj_edge_jaccard"]) - float(
        control_summary["adj_edge_jaccard"]
    )
    raw_delta = float(candidate_summary["edge_jaccard"]) - float(
        control_summary["edge_jaccard"]
    )
    recall_delta = float(candidate_summary["node_recall"]) - float(
        control_summary["node_recall"]
    )
    passed = (
        adjusted_delta
        >= float(gate["minimum_adjusted_edge_jaccard_delta"])
        and (
            not bool(gate["require_nonnegative_raw_edge_jaccard_delta"])
            or raw_delta >= 0.0
        )
        and recall_delta >= float(gate["minimum_node_recall_delta"])
    )
    return {
        "adjusted_edge_jaccard_delta": adjusted_delta,
        "raw_edge_jaccard_delta": raw_delta,
        "node_recall_delta": recall_delta,
        "counterfactual_gate_status": "PASS-GRAPH" if passed else "PARK",
        "promotion_authorized": False,
    }


def _metric_subset(value: Mapping[str, Any]) -> dict[str, float | int]:
    keys = (
        "accepted_labeled",
        "correct_rescues",
        "harmful_swaps",
        "background_accepted",
        "conservative_utility",
        "precision",
    )
    return {name: value[name] for name in keys}


def _assert_metric_reproduction(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name, observed_value in _metric_subset(observed).items():
        expected_value = expected[name]
        if isinstance(observed_value, float) or isinstance(expected_value, float):
            if not math.isclose(
                float(observed_value),
                float(expected_value),
                rel_tol=1.0e-8,
                abs_tol=1.0e-8,
            ):
                raise TemporalOOFGraphGateError(
                    f"{label} metric {name} did not reproduce"
                )
        elif int(observed_value) != int(expected_value):
            raise TemporalOOFGraphGateError(
                f"{label} metric {name} did not reproduce"
            )


def _cap(contract: Mapping[str, Any], selected_edges: int) -> int:
    specification = contract["proposal_cap"]
    return max(
        int(specification["minimum_per_video"]),
        int(
            math.ceil(
                float(specification["selected_edge_fraction"]) * selected_edges
            )
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the guarded retrospective held-fit graph diagnostic."""

    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise TemporalOOFGraphGateError(
            "run must be launched by the local training guard"
        )
    if args.output.exists():
        raise TemporalOOFGraphGateError(
            f"refusing to overwrite output: {args.output}"
        )
    contract = load_contract(args.contract, args.contract_sha256)
    candidate_contract = contract["candidate"]
    data_contract = contract["data"]

    source_path = _root_path(
        candidate_contract["source_report"], "source proposal report"
    )
    prepare_path = _root_path(
        data_contract["prepare_report"], "prepared temporal data"
    )
    selection_path = _root_path(
        data_contract["selection_report"], "fit selection report"
    )
    runtime_contract_path = _root_path(
        contract["control"]["runtime_contract"], "APNCC runtime contract"
    )
    for path, expected, label in (
        (
            source_path,
            candidate_contract["source_report_sha256"],
            "source proposal report",
        ),
        (
            prepare_path,
            data_contract["prepare_report_sha256"],
            "prepared temporal data",
        ),
        (
            selection_path,
            data_contract["selection_report_sha256"],
            "fit selection report",
        ),
    ):
        if sweep._sha256(path) != expected:
            raise TemporalOOFGraphGateError(f"{label} SHA-256 mismatch")

    source = _load_json(source_path, "source proposal report")
    prepare = _load_json(prepare_path, "prepared temporal data")
    selection = _load_json(selection_path, "fit selection report")
    if (
        source.get("schema")
        != "celltrack-alpha.selected-graph-temporal-family-rank-gate.v1"
        or prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or selection.get("schema")
        != "celltrack-alpha.selected-graph-appearance-audit.v1"
    ):
        raise TemporalOOFGraphGateError("input report schema mismatch")
    selected = source.get("selected")
    if (
        not isinstance(selected, Mapping)
        or dict(selected.get("configuration", {}))
        != {
            "name": "linear-bg005",
            "hidden": 0,
            "background_weight": 0.05,
        }
    ):
        raise TemporalOOFGraphGateError("source selected configuration drifted")

    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = set(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.zip),
        )
    )
    eligible = validate_fit_scope(source["scope"], prepare["scope"], cv57)
    if set(selection["scope"]["train_video_ids"]) != set(
        source["scope"]["train_video_ids"]
    ):
        raise TemporalOOFGraphGateError("fit selection train105 identity drifted")

    rows_by_identity = {
        str(row["identity"]): row
        for row in prepare["videos"]
        if str(row["identity"]) in set(eligible)
    }
    if set(rows_by_identity) != set(eligible):
        raise TemporalOOFGraphGateError("prepared eligible coverage is incomplete")
    videos_by_identity = {
        identity: _video_data(rows_by_identity[identity])
        for identity in eligible
    }
    folds = held_fold_partitions(eligible, int(contract["scope"]["fold_count"]))
    source_folds = {
        int(row["fold"]): row
        for row in selected["families"][FAMILY]["folds"]
    }
    if set(source_folds) != set(range(len(folds))):
        raise TemporalOOFGraphGateError("source OOF fold coverage is incomplete")

    configuration = candidate_contract["configuration"]
    threshold_contract = candidate_contract["threshold_selection"]
    fold_models: dict[
        int, tuple[torch.nn.Module, np.ndarray, np.ndarray, float]
    ] = {}
    fold_records = []
    args.output.mkdir(parents=True, exist_ok=False)
    artifact_dir = args.output / "fold_models"
    control_dir = args.output / "control_fragments"
    candidate_dir = args.output / "candidate_fragments"
    artifact_dir.mkdir()
    control_dir.mkdir()
    candidate_dir.mkdir()
    started = time.perf_counter()

    for fold, (fit_ids, held_ids) in enumerate(folds):
        fit_videos = [videos_by_identity[value] for value in fit_ids]
        held_videos = [videos_by_identity[value] for value in held_ids]
        seed = (
            int(configuration["seed_base"])
            + 1000 * int(configuration["configuration_index"])
            + 100 * int(configuration["family_index"])
            + fold
        )
        model, mean, scale, optimization = _fit_model(
            fit_videos,
            hidden=int(configuration["hidden"]),
            background_weight=float(configuration["background_weight"]),
            epochs=int(configuration["epochs"]),
            seed=seed,
        )
        threshold, fit_metrics = _select(
            _model_scores(model, mean, scale, fit_videos),
            minimum_accepted=int(threshold_contract["minimum_fit_accepted"]),
            minimum_precision=float(threshold_contract["minimum_fit_precision"]),
        )
        held_metrics = _metrics(
            _model_scores(model, mean, scale, held_videos), threshold
        )
        expected = source_folds[fold]
        if not math.isclose(
            float(threshold),
            float(expected["threshold"]),
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        ):
            raise TemporalOOFGraphGateError(
                f"fold {fold} threshold did not reproduce: "
                f"observed={threshold:.17g}, "
                f"expected={float(expected['threshold']):.17g}"
            )
        _assert_metric_reproduction(
            fit_metrics, expected["fit"], label=f"fold {fold} fit"
        )
        _assert_metric_reproduction(
            held_metrics, expected["held"], label=f"fold {fold} held"
        )
        artifact = artifact_dir / f"gate_{FAMILY}_fold{fold}.pt"
        torch.save(
            {
                "schema": SCHEMA,
                "family": FAMILY,
                "fold": fold,
                "fit_ids": list(fit_ids),
                "held_ids": list(held_ids),
                "feature_count": 48,
                "hidden": int(configuration["hidden"]),
                "mean": mean,
                "scale": scale,
                "threshold": threshold,
                "state_dict": model.state_dict(),
                "seed": seed,
            },
            artifact,
        )
        fold_models[fold] = (model, mean, scale, threshold)
        fold_records.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_video_count": len(fit_ids),
                "held_video_count": len(held_ids),
                "fit_ids": list(fit_ids),
                "held_ids": list(held_ids),
                "threshold": threshold,
                "fit": _metric_subset(fit_metrics),
                "held": _metric_subset(held_metrics),
                "optimization": optimization,
                "artifact": str(artifact),
                "artifact_sha256": sweep._sha256(artifact),
            }
        )
        print(
            f"OOF model {fold + 1:02d}/{len(folds)} "
            f"held={len(held_ids)} threshold={threshold:.8f}",
            flush=True,
        )

    proposal_oof = selected["families"][FAMILY]["oof"]
    prerequisite = candidate_contract["proposal_prerequisite"]
    proposal_prerequisite_passed = (
        int(proposal_oof["accepted_labeled"])
        >= int(prerequisite["minimum_oof_accepted"])
        and float(proposal_oof["precision"])
        >= float(prerequisite["minimum_oof_precision"])
    )
    if not proposal_prerequisite_passed:
        raise TemporalOOFGraphGateError(
            "source proposal prerequisite did not pass"
        )

    apncc_contract, geometry_gate = appearance_swap.load_contract(
        runtime_contract_path,
        expected_sha256=contract["control"]["runtime_contract_sha256"],
        root=ROOT,
    )
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(eligible).issubset(cache_paths) or not set(eligible).issubset(
        meta_paths
    ):
        raise TemporalOOFGraphGateError("fit cache coverage is incomplete")
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
        raise TemporalOOFGraphGateError("frozen edge scorer did not restore")

    graph_records = []
    identical_fragments = 0
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(eligible, start=1):
            fold = _fold(identity, len(folds))
            model, mean, scale, temporal_threshold = fold_models[fold]
            dataset = identity.split(":", 1)[1]
            meta = sweep._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            probabilities, no_link, association = run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=bh_edge
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            adjacent_edges, adjacent_stats = bh_graph.build_adjacent_edges(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                topology_mode="legacy",
            )
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            refiner = sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = bh_graph.close_one_frame_gaps(
                graph_nodes,
                adjacent_edges,
                image_refine_callback=refiner,
            )
            _baseline_nodes, baseline_edges, _baseline_post = (
                bh_graph.postprocess_graph(gap_nodes, gap_edges)
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            baseline_selected = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids
                and int(edge.target_id) in raw_ids
            }
            expected_selected = int(
                rows_by_identity[identity]["graph"]["raw_selected_edges"]
            )
            if len(baseline_selected) != expected_selected:
                raise TemporalOOFGraphGateError(
                    f"selected graph identity drifted for {identity}"
                )
            proposal_cap = _cap(apncc_contract, len(baseline_selected))
            control_proposals, control_counts = appearance_swap.propose_swaps(
                cache,
                scorer,
                baseline_selected,
                geometry_gate,
                image,
                geometry_threshold=float(
                    apncc_contract["geometry_gate"]["threshold"]
                ),
                appearance_threshold=float(
                    apncc_contract["appearance"]["threshold"]
                ),
                maximum=proposal_cap,
            )
            candidate_proposals, candidate_counts = (
                temporal_rank_swap.propose_swaps(
                    cache,
                    scorer,
                    baseline_selected,
                    geometry_gate,
                    model,
                    mean,
                    scale,
                    image,
                    geometry_threshold=float(
                        apncc_contract["geometry_gate"]["threshold"]
                    ),
                    temporal_threshold=temporal_threshold,
                    maximum=proposal_cap,
                )
            )
            shape = tuple(int(value) for value in image.shape)
            control_solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=control_proposals,
                probabilities=probabilities,
                fragment_dir=control_dir,
            )
            candidate_solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=candidate_proposals,
                probabilities=probabilities,
                fragment_dir=candidate_dir,
            )
            control_realization = _realization(
                Path(control_solved["fragment"]), control_proposals
            )
            candidate_realization = _realization(
                Path(candidate_solved["fragment"]), candidate_proposals
            )
            if (
                control_realization["retained_conflicts"] != 0
                or candidate_realization["retained_conflicts"] != 0
            ):
                raise TemporalOOFGraphGateError(
                    f"direct swap realization failed for {identity}"
                )
            control_hash = sweep._sha256(Path(control_solved["fragment"]))
            candidate_hash = sweep._sha256(Path(candidate_solved["fragment"]))
            fragment_equal = control_hash == candidate_hash
            identical_fragments += int(fragment_equal)
            graph_records.append(
                {
                    "identity": identity,
                    "fold": fold,
                    "fit_contains_identity": False,
                    "threshold_selected_without_identity": True,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_cap": proposal_cap,
                    "control_proposals": len(control_proposals),
                    "candidate_proposals": len(candidate_proposals),
                    "control_proposal_counts": control_counts,
                    "candidate_proposal_counts": candidate_counts,
                    "control_realization": control_realization,
                    "candidate_realization": candidate_realization,
                    "control_fragment_sha256": control_hash,
                    "candidate_fragment_sha256": candidate_hash,
                    "fragments_exactly_equal": fragment_equal,
                }
            )
            print(
                f"OOF graph {index:02d}/{len(eligible)} {identity} "
                f"fold={fold} apncc={len(control_proposals)} "
                f"temporal={len(candidate_proposals)} "
                f"equal={fragment_equal}",
                flush=True,
            )

        control_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(control_dir)), eligible
        )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(candidate_dir)), eligible
        )
        control_rows, control_provenance, control_coverage = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )

    control_by_id = {row["identity"]: row for row in control_rows}
    candidate_by_id = {row["identity"]: row for row in candidate_rows}
    ordered_control = [control_by_id[value] for value in eligible]
    ordered_candidate = [candidate_by_id[value] for value in eligible]
    control_summary = scoreall.official_summary(ordered_control)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    decision = graph_gate_decision(
        control_summary, candidate_summary, contract["graph_gate"]
    )
    decision["diagnostic_success"] = (
        decision["counterfactual_gate_status"] == "PARK"
    )
    decision["interpretation"] = (
        "gate correctly screened the known CV57 regression"
        if decision["diagnostic_success"]
        else (
            "fit graph gate passed despite known CV57 regression; "
            "mechanism transfer remains unresolved"
        )
    )
    per_fold = []
    for fold, (_, held_ids) in enumerate(folds):
        control_fold = [control_by_id[value] for value in held_ids]
        candidate_fold = [candidate_by_id[value] for value in held_ids]
        fold_control_summary = scoreall.official_summary(control_fold)
        fold_candidate_summary = scoreall.official_summary(candidate_fold)
        per_fold.append(
            {
                "fold": fold,
                "video_count": len(held_ids),
                "control_summary": fold_control_summary,
                "candidate_summary": fold_candidate_summary,
                "decision": graph_gate_decision(
                    fold_control_summary,
                    fold_candidate_summary,
                    contract["graph_gate"],
                ),
            }
        )

    result = {
        "schema": SCHEMA,
        "status": "PARK",
        "scope": {
            "family": FAMILY,
            "source_partition": "fit-train105",
            "video_count": len(eligible),
            "video_ids": list(eligible),
            "held_predictions_per_video": 1,
            "validation12_access": False,
            "cv57_access": False,
            "promotion_authorized": False,
        },
        "proposal_prerequisite": {
            "source_oof": proposal_oof,
            "contract": prerequisite,
            "passed": proposal_prerequisite_passed,
        },
        "folds": fold_records,
        "graph": {
            "control": {
                "summary": control_summary,
                "edge_counts": _edge_counts(ordered_control),
                "coverage": control_coverage,
                "input_provenance": control_provenance,
            },
            "candidate": {
                "summary": candidate_summary,
                "edge_counts": _edge_counts(ordered_candidate),
                "coverage": candidate_coverage,
                "input_provenance": candidate_provenance,
            },
            "observation_identity": {
                "exactly_equal_fragment_count": identical_fragments,
                "observation_distinct_fragment_count": (
                    len(eligible) - identical_fragments
                ),
                "all_observations_identical": identical_fragments == len(eligible),
            },
            "per_fold": per_fold,
            "videos": graph_records,
        },
        "decision": decision,
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "source_report": str(source_path),
            "source_report_sha256": sweep._sha256(source_path),
            "prepare_report": str(prepare_path),
            "prepare_report_sha256": sweep._sha256(prepare_path),
            "selection_report": str(selection_path),
            "selection_report_sha256": sweep._sha256(selection_path),
            "runtime_contract": str(runtime_contract_path),
            "runtime_contract_sha256": apncc_contract["contract_sha256"],
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument(
        "--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY
    )
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report["decision"], sort_keys=True))

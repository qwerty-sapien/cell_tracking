#!/usr/bin/env python3
"""Run the frozen C1 conflict-set head on one held-fit family graph gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
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

import audit_c1_hoct_lite as c1_audit  # noqa: E402
import evaluate_h1_zero_wrfeat_graph_gate as h1  # noqa: E402
import evaluate_trackastra_linear_oof_graph_gate as engine  # noqa: E402
import run_bh  # noqa: E402
import scoreall  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.c1_conflict_set_attention import (  # noqa: E402
    ComponentExample,
    FittedConflictHead,
    _padded_batch,
    component_examples,
    fit_conflict_head,
    fold_index,
    nested_partitions,
)
from celltrack_alpha.trackastra_linear import (  # noqa: E402
    TrackastraLinearError,
    aggregate_precision,
    challenger_labels,
    precision_metrics,
    propose_swaps,
    select_threshold,
)
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _edge_counts,
    _realization,
)
from evaluate_selected_graph_direct_swap import _solve_direct  # noqa: E402
from evaluate_temporal_rank_oof_graph_gate import _cap  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402


SCHEMA = "celltrack-alpha.c1-hoct-lite-held-fit-graph-gate.v1"
PREREGISTRATION = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "c1_ta_hoct_lite/outputs/c1_n1_preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "d4a3651a2e3d429f432d90d7387478c18545f11612f481b9e43928536e26f5e9"
)
N2_REPORT = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "c1_ta_hoct_lite/outputs/c1_n2_build_validation.json"
)
U0_CONTRACT = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "u0_t4canon/results/u0_t4canon_contract_draft.json"
)
U0_CONTRACT_SHA256 = (
    "08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1"
)
U0_ATTESTATION = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "u0_t4canon/results/u0_t4canon_exact_replay_attestation.json"
)
H1_CONTRACT = (
    ROOT
    / "reports/g02/screening/"
    "h1_zero_wrfeat_graph_gate_contract_parity_repair.json"
)
APNCC_44B6_REPORT = (
    ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json"
)
APNCC_44B6_FRAGMENTS = APNCC_44B6_REPORT.parent / "control_fragments"
OUTPUTS = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "c1_ta_hoct_lite/outputs"
)


class C1GraphGateError(RuntimeError):
    """The frozen C1 graph gate failed closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise C1GraphGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise C1GraphGateError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("self_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _head_hash(head: FittedConflictHead) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(head.mean, dtype="<f4").tobytes())
    digest.update(np.asarray(head.scale, dtype="<f4").tobytes())
    digest.update(np.asarray(head.positive_weight, dtype="<f8").tobytes())
    for name, value in sorted(head.model.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _authenticate_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _load_json(PREREGISTRATION, "C1 N1 preregistration")
    if (
        prereg.get("status") != "locked"
        or prereg.get("self_sha256") != PREREGISTRATION_SHA256
        or _canonical_hash(prereg) != PREREGISTRATION_SHA256
        or prereg.get("budget", {}).get("candidate_configurations") != 1
        or prereg.get("budget", {}).get("seeds") != 1
        or prereg.get("budget", {}).get("gpu_action") is not False
        or prereg.get("graph_gates", {}).get(
            "cv57_selection_allowed"
        ) is not False
    ):
        raise C1GraphGateError("C1 N1 preregistration identity drifted")
    if _sha256(U0_CONTRACT) != U0_CONTRACT_SHA256:
        raise C1GraphGateError("locked U0-T4CANON identity drifted")
    u0 = _load_json(U0_ATTESTATION, "U0 exact replay attestation")
    u0_44b6 = u0.get("calibration_decisions", {}).get("44b6", {})
    if (
        u0.get("contract", {}).get("sha256") != U0_CONTRACT_SHA256
        or u0_44b6.get("admissible_threshold_folds") != 0
        or u0_44b6.get("no_admissible_threshold_folds") != 5
        or u0_44b6.get("action")
        != "fail-closed no-op; no threshold weakened"
    ):
        raise C1GraphGateError("U0 44b6 C0 no-op attestation drifted")
    n2 = _load_json(N2_REPORT, "C1 N2 validation")
    if (
        n2.get("status") != "PASS"
        or n2.get("verdict") != "GO"
        or n2.get("contract", {}).get("canonical_sha256")
        != PREREGISTRATION_SHA256
        or n2.get("manifest_replay", {}).get(
            "all_payload_hashes_verified"
        )
        is not True
        or n2.get("c1_grouped_oof_dry", {}).get("byte_identical")
        is not True
    ):
        raise C1GraphGateError("C1 N2 validation identity drifted")
    return prereg, n2


def _labels(identity: str, payload: Mapping[str, Any]) -> np.ndarray:
    path = (
        ROOT
        / "reports/g02/selected_graph_temporal_background_prepare_v1/examples"
        / f"{identity.replace(':', '__', 1)}.pt"
    )
    prepared = torch.load(path, map_location="cpu", weights_only=False)
    known = challenger_labels(
        np.asarray(prepared["labeled_challenger_pairs"]),
        np.asarray(prepared["labeled_labels"]),
    )
    challengers = np.asarray(payload["challengers"], dtype=np.int64)
    return np.asarray(
        [
            known.get((int(pair[0]), int(pair[1])), -1)
            for pair in challengers
        ],
        dtype=np.float32,
    )


def _score_payload(
    fitted: FittedConflictHead,
    identity: str,
    payload: Mapping[str, Any],
    *,
    batch_size: int = 512,
) -> np.ndarray:
    examples = component_examples(identity, payload)
    result = np.empty(len(np.asarray(payload["challengers"])), dtype=np.float32)
    fitted.model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            features, _labels_t, valid, _known = _padded_batch(
                batch, fitted.mean, fitted.scale
            )
            logits, no_link = fitted.model(features, valid)
            scores = (logits - no_link[:, None]).cpu().numpy()
            for row, example in enumerate(batch):
                result[example.proposal_indices] = scores[
                    row, : len(example.proposal_indices)
                ]
    if not np.isfinite(result).all():
        raise C1GraphGateError(f"non-finite C1 score: {identity}")
    return result


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (
        positive * negative
    )


def _fit_heads(
    payloads: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, np.ndarray],
    output: Path,
) -> tuple[
    dict[int, tuple[FittedConflictHead, float | None]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    identities = tuple(sorted(payloads))
    partitions = nested_partitions(identities)
    heads = {}
    records = []
    held_metric_rows = []
    held_score_chunks = []
    held_label_chunks = []
    model_dir = output / "fold_models"
    model_dir.mkdir()
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(
        partitions
    ):
        fit_examples = [
            example
            for identity in fit_ids
            for example in component_examples(
                identity, payloads[identity], labels[identity]
            )
        ]
        fitted = fit_conflict_head(
            fit_examples,
            seed=20260727,
            epochs=12,
            component_batch_size=64,
        )
        calibration_scores = []
        calibration_labels = []
        held_scores = []
        held_labels = []
        for identity in calibration_ids:
            mask = labels[identity] >= 0
            if np.any(mask):
                calibration_scores.append(
                    _score_payload(fitted, identity, payloads[identity])[mask]
                )
                calibration_labels.append(labels[identity][mask].astype(np.int8))
        for identity in held_ids:
            mask = labels[identity] >= 0
            if np.any(mask):
                values = _score_payload(fitted, identity, payloads[identity])[mask]
                held_scores.append(values)
                held_labels.append(labels[identity][mask].astype(np.int8))
                if identity.startswith("train:44b6_"):
                    held_score_chunks.append(values)
                    held_label_chunks.append(labels[identity][mask].astype(np.int8))
        cal_scores = np.concatenate(calibration_scores)
        cal_labels = np.concatenate(calibration_labels)
        fold_held_scores = np.concatenate(held_scores)
        fold_held_labels = np.concatenate(held_labels)
        threshold: float | None
        try:
            threshold, calibration_metrics = select_threshold(
                cal_scores,
                cal_labels,
                minimum_accepted=100,
                minimum_precision=0.95,
            )
            threshold_status = "ADMISSIBLE"
            held_metrics = precision_metrics(
                fold_held_scores, fold_held_labels, threshold
            )
        except (TrackastraLinearError, ValueError):
            threshold = None
            threshold_status = "NO_ADMISSIBLE_THRESHOLD_FAIL_CLOSED_NO_OP"
            calibration_metrics = {
                "eligible_labeled": len(cal_labels),
                "accepted_labeled": 0,
                "correct_rescues": 0,
                "harmful_swaps": 0,
                "precision": 0.0,
                "threshold": None,
            }
            held_metrics = {
                "eligible_labeled": len(fold_held_labels),
                "accepted_labeled": 0,
                "correct_rescues": 0,
                "harmful_swaps": 0,
                "precision": 0.0,
                "threshold": None,
            }
        head_sha256 = _head_hash(fitted)
        artifact = model_dir / f"fold{held_fold}.pt"
        torch.save(
            {
                "state_dict": fitted.model.state_dict(),
                "mean": fitted.mean,
                "scale": fitted.scale,
                "positive_weight": fitted.positive_weight,
                "threshold": threshold,
                "head_sha256": head_sha256,
            },
            artifact,
        )
        record = {
            "held_fold": held_fold,
            "calibration_fold": (held_fold + 1) % 5,
            "fit_ids": list(fit_ids),
            "calibration_ids": list(calibration_ids),
            "held_ids": list(held_ids),
            "fit_component_count": len(fit_examples),
            "head_sha256": head_sha256,
            "artifact": str(artifact.relative_to(ROOT)),
            "artifact_sha256": _sha256(artifact),
            "threshold": threshold,
            "threshold_status": threshold_status,
            "calibration": calibration_metrics,
            "held": held_metrics,
        }
        records.append(record)
        held_metric_rows.append(held_metrics)
        heads[held_fold] = (fitted, threshold)
        print(
            f"C1 fold {held_fold + 1}/5 {threshold_status} "
            f"labels={len(fold_held_labels)}",
            flush=True,
        )
    family_scores = np.concatenate(held_score_chunks)
    family_labels = np.concatenate(held_label_chunks)
    diagnostics = {
        "held_oof_fail_closed": aggregate_precision(held_metric_rows),
        "held_44b6_all_scores": {
            "eligible_labeled": len(family_labels),
            "positive_labeled": int(family_labels.sum()),
            "negative_labeled": int(len(family_labels) - family_labels.sum()),
            "roc_auc": _auc(family_scores, family_labels),
            "gating": False,
        },
        "proposal_metrics_gate": False,
    }
    return heads, records, diagnostics


def _authenticate_apncc(
    eligible: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    report = _load_json(APNCC_44B6_REPORT, "cached APNCC 44b6 graph report")
    videos = report.get("graph", {}).get("videos", [])
    hashes = {
        str(row["identity"]): str(row["control_fragment_sha256"])
        for row in videos
    }
    if (
        report.get("scope", {}).get("family") != "44b6"
        or report.get("scope", {}).get("validation12_access") is not False
        or report.get("scope", {}).get("cv57_access") is not False
        or set(hashes) != set(eligible)
    ):
        raise C1GraphGateError("cached APNCC 44b6 scope drifted")
    for identity, expected in hashes.items():
        path = APNCC_44B6_FRAGMENTS / f"{identity.replace(':', '__', 1)}.csv"
        if not path.is_file() or _sha256(path) != expected:
            raise C1GraphGateError(
                f"cached APNCC fragment identity drifted: {identity}"
            )
    return APNCC_44B6_FRAGMENTS, report


def _score_family(
    *,
    family: str,
    heads: Mapping[int, tuple[FittedConflictHead, float | None]],
    payloads: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    args: argparse.Namespace,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_json(
        ROOT / contract["data"]["fit_manifest"], "fit105 manifest"
    )
    eligible = tuple(
        str(value)
        for value in manifest["identities"]
        if str(value).startswith(f"train:{family}_")
    )
    expected = int(contract["scope"]["family_counts"][family])
    if len(eligible) != expected:
        raise C1GraphGateError(f"{family} identity coverage drifted")
    if family != "44b6":
        raise C1GraphGateError("this frozen invocation is 44b6-only")
    control_dir, control_source = _authenticate_apncc(eligible)
    candidate_dir = output / "candidate_fragments"
    candidate_dir.mkdir()

    runtime_path = ROOT / contract["control"]["runtime_contract"]
    apncc_contract, _geometry_gate = engine.appearance_swap.load_contract(
        runtime_path,
        expected_sha256=contract["control"]["runtime_contract_sha256"],
        root=ROOT,
    )
    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise C1GraphGateError("edge scorer did not restore")
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    graph_records = []
    identical = 0
    with ZipStore(args.raw_archive, mode="r") as store:
        for index, identity in enumerate(eligible, start=1):
            fitted, threshold = heads[fold_index(identity)]
            meta = sweep._load(meta_paths[identity], f"cache metadata {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            selected, _unused, _no_link, _graph = _baseline(cache, scorer)
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
            dataset = identity.split(":", 1)[1]
            image = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )["0"]
            refiner = sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = bh_graph.close_one_frame_gaps(
                graph_nodes,
                adjacent_edges,
                image_refine_callback=refiner,
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            _baseline_nodes, baseline_edges, _post = bh_graph.postprocess_graph(
                gap_nodes, gap_edges
            )
            reconstructed = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids
                and int(edge.target_id) in raw_ids
            }
            if reconstructed != selected:
                raise C1GraphGateError(
                    f"selected graph reconstruction drifted: {identity}"
                )
            cap = _cap(apncc_contract, len(selected))
            scores = _score_payload(fitted, identity, payloads[identity])
            if threshold is None:
                proposals = []
                counts = {
                    "full_pool_proposals": len(scores),
                    "threshold_proposals": 0,
                    "accepted_before_cap": 0,
                    "accepted_proposals": 0,
                    "proposal_cap": cap,
                    "cap_active": True,
                    "cap_removed": 0,
                    "fail_closed_no_admissible_threshold": True,
                }
            else:
                proposals, counts = propose_swaps(
                    payloads[identity],
                    scores,
                    threshold=threshold,
                    maximum=cap,
                )
                counts["fail_closed_no_admissible_threshold"] = False
            shape = tuple(int(value) for value in image.shape)
            solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=proposals,
                probabilities=probabilities,
                fragment_dir=candidate_dir,
            )
            realization = _realization(Path(solved["fragment"]), proposals)
            if realization["retained_conflicts"] != 0:
                raise C1GraphGateError(
                    f"C1 swap realization failed: {identity}"
                )
            candidate_hash = _sha256(Path(solved["fragment"]))
            control_hash = _sha256(
                control_dir / f"{identity.replace(':', '__', 1)}.csv"
            )
            equal = candidate_hash == control_hash
            identical += int(equal)
            graph_records.append(
                {
                    "identity": identity,
                    "held_fold": fold_index(identity),
                    "fit_contains_identity": False,
                    "calibration_contains_identity": False,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_cap": cap,
                    "candidate_proposals": len(proposals),
                    "candidate_proposal_counts": counts,
                    "candidate_realization": realization,
                    "control_fragment_sha256": control_hash,
                    "candidate_fragment_sha256": candidate_hash,
                    "fragments_exactly_equal": equal,
                }
            )
            print(
                f"C1 graph {index:02d}/{len(eligible)} {identity} "
                f"fold={fold_index(identity)} proposals={len(proposals)} "
                f"equal={equal}",
                flush=True,
            )
        control_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(control_dir)), eligible
        )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(candidate_dir)), eligible
        )
        control_rows_1, control_provenance_1, control_coverage_1 = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        control_rows_2, control_provenance_2, control_coverage_2 = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )
    by_control_1 = {row["identity"]: row for row in control_rows_1}
    by_control_2 = {row["identity"]: row for row in control_rows_2}
    by_candidate = {row["identity"]: row for row in candidate_rows}
    ordered_control_1 = [by_control_1[value] for value in eligible]
    ordered_control_2 = [by_control_2[value] for value in eligible]
    ordered_candidate = [by_candidate[value] for value in eligible]
    control_summary_1 = scoreall.official_summary(ordered_control_1)
    control_summary_2 = scoreall.official_summary(ordered_control_2)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    repeat_differences = [
        abs(
            float(by_control_1[identity]["adj_edge_jaccard"])
            - float(by_control_2[identity]["adj_edge_jaccard"])
        )
        for identity in eligible
    ]
    epsilon_null = max(1.0e-12, 2.0 * max(repeat_differences, default=0.0))
    apncc = {
        "summary": control_summary_1,
        "edge_counts": _edge_counts(ordered_control_1),
        "coverage": control_coverage_1,
        "input_provenance": control_provenance_1,
        "cached_source_report": str(APNCC_44B6_REPORT.relative_to(ROOT)),
        "cached_source_report_sha256": _sha256(APNCC_44B6_REPORT),
        "cached_source_summary": control_source["graph"]["control"]["summary"],
    }
    c0 = {
        "summary": control_summary_2,
        "edge_counts": _edge_counts(ordered_control_2),
        "coverage": control_coverage_2,
        "input_provenance": control_provenance_2,
        "u0_fail_closed_no_op": True,
        "exactly_equal_to_apncc": (
            control_provenance_1["aggregate_sha256"]
            == control_provenance_2["aggregate_sha256"]
        ),
    }
    candidate = {
        "summary": candidate_summary,
        "edge_counts": _edge_counts(ordered_candidate),
        "coverage": candidate_coverage,
        "input_provenance": candidate_provenance,
    }
    null = {
        "maximum_absolute_per_video_adjusted_score_difference": max(
            repeat_differences, default=0.0
        ),
        "epsilon_null": epsilon_null,
        "repeat_1_provenance": control_provenance_1,
        "repeat_2_provenance": control_provenance_2,
        "exact_identical_reconstruction": (
            control_provenance_1["aggregate_sha256"]
            == control_provenance_2["aggregate_sha256"]
        ),
    }
    observation = {
        "exactly_equal_fragment_count": identical,
        "observation_distinct_fragment_count": len(eligible) - identical,
        "observation_distinct": identical < len(eligible),
    }
    return (
        {
            "apncc": apncc,
            "c0": c0,
            "c1": candidate,
            "null": null,
            "observation_identity": observation,
            "videos": graph_records,
        },
        {
            "family": family,
            "eligible": list(eligible),
            "candidate_fragment_dir": str(candidate_dir.relative_to(ROOT)),
            "control_fragment_dir": str(control_dir.relative_to(ROOT)),
        },
        {
            "edge_scorer": edge_record,
            "control_repeat_exact": null["exact_identical_reconstruction"],
        },
    )


def _decision(graph: Mapping[str, Any], family: str) -> dict[str, Any]:
    apncc = graph["apncc"]["summary"]
    c0 = graph["c0"]["summary"]
    c1 = graph["c1"]["summary"]
    adjusted = float(c1["adj_edge_jaccard"]) - float(
        apncc["adj_edge_jaccard"]
    )
    raw = float(c1["edge_jaccard"]) - float(apncc["edge_jaccard"])
    recall = float(c1["node_recall"]) - float(apncc["node_recall"])
    alpha = float(c1["adj_edge_jaccard"]) - float(
        c0["adj_edge_jaccard"]
    )
    epsilon = float(graph["null"]["epsilon_null"])
    alpha_floor = max(0.001, epsilon)
    distinct = bool(graph["observation_identity"]["observation_distinct"])
    passed = (
        adjusted >= 0.005
        and raw >= 0.0
        and recall >= 0.0
        and distinct
        and (family != "44b6" or alpha >= alpha_floor)
    )
    if family == "44b6" and adjusted <= 0.0:
        verdict = "DEAD"
        reason = "adjusted edge-Jaccard delta is at or below zero"
    elif passed:
        verdict = "GO"
        reason = "every preregistered graph and alpha-attribution gate passed"
    else:
        verdict = "PARK"
        reason = "positive candidate did not clear every preregistered gate"
    return {
        "verdict": verdict,
        "reason": reason,
        "adjusted_edge_jaccard_delta_vs_apncc": adjusted,
        "raw_edge_jaccard_delta_vs_apncc": raw,
        "node_recall_delta_vs_apncc": recall,
        "adjusted_edge_jaccard_delta_vs_c0": alpha,
        "epsilon_null": epsilon,
        "alpha_margin_minimum": alpha_floor,
        "observation_distinct": distinct,
        "gates": {
            "adjusted_vs_apncc_ge_0_005": adjusted >= 0.005,
            "raw_vs_apncc_ge_0": raw >= 0.0,
            "node_recall_vs_apncc_ge_0": recall >= 0.0,
            "adjusted_vs_c0_ge_alpha_floor": alpha >= alpha_floor,
            "observation_distinct": distinct,
        },
        "promotion_authorized": False,
        "cv57_authorized": False,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    graph = result["graph"]
    lines = [
        "# C1 N3 — 44b6 graph falsification",
        "",
        f"**Verdict: {decision['verdict']}** — {decision['reason']}.",
        "",
        "## Exact graph vector",
        "",
        "| Comparison | Adjusted Δ | Raw Δ | Node-recall Δ |",
        "|---|---:|---:|---:|",
        (
            "| C1 − APNCC | "
            f"{decision['adjusted_edge_jaccard_delta_vs_apncc']:+.12f} | "
            f"{decision['raw_edge_jaccard_delta_vs_apncc']:+.12f} | "
            f"{decision['node_recall_delta_vs_apncc']:+.12f} |"
        ),
        (
            "| C1 − C0 | "
            f"{decision['adjusted_edge_jaccard_delta_vs_c0']:+.12f} | "
            "— | — |"
        ),
        "",
        (
            f"`epsilon_null={decision['epsilon_null']:.12g}`; "
            f"alpha floor `{decision['alpha_margin_minimum']:.12g}`."
        ),
        "",
        "## Counts",
        "",
        "| Arm | TP | FP | FN | adjusted | raw | node recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("apncc", "APNCC"), ("c0", "C0"), ("c1", "C1")):
        counts = graph[key]["edge_counts"]
        summary = graph[key]["summary"]
        lines.append(
            f"| {label} | {counts['edge_tp']} | {counts['edge_fp']} | "
            f"{counts['edge_fn']} | {summary['adj_edge_jaccard']:.12f} | "
            f"{summary['edge_jaccard']:.12f} | "
            f"{summary['node_recall']:.12f} |"
        )
    diagnostics = result["proposal_diagnostics"]
    held = diagnostics["held_oof_fail_closed"]
    lines.extend(
        [
            "",
            "## Proposal diagnostics (non-gating)",
            "",
            (
                f"Fail-closed held accepted `{held['accepted_labeled']}` / "
                f"`{held['eligible_labeled']}`; precision "
                f"`{held['precision']:.6f}`. Full held 44b6 score AUC: "
                f"`{diagnostics['held_44b6_all_scores']['roc_auc']}`."
            ),
            "",
            "## Integrity",
            "",
            (
                f"- APNCC and C0 were scored from the same authenticated fragments; "
                f"exact-repeat null: `{graph['null']['exact_identical_reconstruction']}`."
            ),
            (
                f"- Observation-distinct C1 fragments: "
                f"`{graph['observation_identity']['observation_distinct_fragment_count']}` "
                f"of `{result['scope']['video_count']}`."
            ),
            "- One configuration, one fixed seed, CPU only; no adaptive retry.",
            "- No validation12, CV57, GPU, Kaggle, RunPod, paid compute, or submission access.",
            "",
            "## Proposed LEDGER row",
            "",
            (
                "`C1-TA-HOCT-LITE-44B6 | SCREEN | >=+0.005 and >=+0.001 "
                "over beta | exact score vector | identity/observation class | "
                f"0 external usage | {decision['verdict']} | {decision['reason']}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
        or os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
    ):
        raise C1GraphGateError("runner requires the zero-swap local guard")
    if args.family != "44b6":
        raise C1GraphGateError("N3 must execute 44b6 first")
    if args.output.exists():
        raise C1GraphGateError(f"refusing to overwrite: {args.output}")
    if args.raw_archive.stat().st_size != 87393127165:
        raise C1GraphGateError("raw archive identity drifted")
    started = time.perf_counter()
    prereg, n2 = _authenticate_contracts()
    contract_hash = _sha256(args.graph_contract)
    contract = engine.load_h1_contract(args.graph_contract, contract_hash)
    parity = h1.verify_parity_repair(
        contract,
        edge_artifact=args.edge_artifact,
        edge_auxiliary=args.edge_auxiliary,
    )
    fit_inputs = h1.verify_fit_inputs(contract, cache_root=args.cache)
    args.output.mkdir(parents=True)
    payloads, _sparse = __import__(
        "run_c1_hoct_lite_dry"
    ).load_inputs()
    labels = {
        identity: _labels(identity, payload)
        for identity, payload in payloads.items()
    }
    heads, folds, diagnostics = _fit_heads(
        payloads, labels, args.output
    )
    graph, scope, scorer = _score_family(
        family=args.family,
        heads=heads,
        payloads=payloads,
        contract=contract,
        args=args,
        output=args.output,
    )
    decision = _decision(graph, args.family)
    result = {
        "schema": SCHEMA,
        "node": "C1-TA-HOCT-LITE-N3",
        "status": decision["verdict"],
        "scope": {
            "family": args.family,
            "source_partition": "fit-train105",
            "video_count": len(scope["eligible"]),
            "video_ids": scope["eligible"],
            "validation12_access": False,
            "cv57_access": False,
            "holdout_access": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
            "promotion_authorized": False,
        },
        "contracts": {
            "c1_preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "canonical_sha256": PREREGISTRATION_SHA256,
            },
            "u0_t4canon": {
                "path": str(U0_CONTRACT.relative_to(ROOT)),
                "sha256": U0_CONTRACT_SHA256,
                "locked": True,
            },
            "graph_runtime": {
                "path": str(args.graph_contract.relative_to(ROOT)),
                "sha256": contract_hash,
            },
            "n2_validation": {
                "path": str(N2_REPORT.relative_to(ROOT)),
                "sha256": _sha256(N2_REPORT),
                "verdict": n2["verdict"],
            },
        },
        "folds": folds,
        "proposal_diagnostics": diagnostics,
        "graph": graph,
        "decision": decision,
        "attestation": {
            "parity_repair": parity,
            "fit_inputs": fit_inputs,
            "scorer": scorer,
            "candidate_configuration_count": prereg["budget"][
                "candidate_configurations"
            ],
            "seed_count": prereg["budget"]["seeds"],
            "adaptive_retry": False,
            "threshold_weakened": False,
            "c0_44b6_fail_closed_no_op": True,
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "maximum_peak_rss_gib": 4.0,
            "maximum_wall_hours": 3.0,
            "elapsed_seconds": time.perf_counter() - started,
            "external_usage": 0,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    final_json = OUTPUTS / "c1_n3_44b6_graph.json"
    final_md = OUTPUTS / "c1_n3_44b6_graph.md"
    if final_json.exists() or final_md.exists():
        raise C1GraphGateError("refusing to overwrite final N3 artifact")
    run_bh.atomic_create_json(final_json, result)
    final_md.write_text(_markdown(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("44b6",), default="44b6")
    parser.add_argument("--graph-contract", type=Path, default=H1_CONTRACT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports/g02/screening/top50_parallel_dags/runs"
            / "20260727_top50_six_01/c1_ta_hoct_lite/n3_44b6_work"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "graph_contract",
        "cache",
        "raw_archive",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

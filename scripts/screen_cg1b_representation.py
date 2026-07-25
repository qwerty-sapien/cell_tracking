#!/usr/bin/env python3
"""CG1b full-pool abstention re-screen on existing frozen representations.

Stage 1 only: no model fitting, no feature-cache creation, no graph replay.  The
script streams the exact fit-train105 r10/k8 proposal pool one video at a time,
uses the banked geometry prefilter and APNCC/context features, and measures
known-correct challenger retention against exact GT-labeled background counts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
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
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import (  # noqa: E402
    appearance_swap,
    bh_edge,
    bh_model,
    point_context_swap,
)
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from fit_selected_graph_temporal_background_gate import _fold  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402


SCHEMA = "celltrack-alpha.cg1b-representation-rescreen.v1"
FAMILIES = ("44b6", "6bba")
FORBIDDEN_PATH_TOKENS = ("cv57", "holdout")
GO_RETENTION = 0.20
PARTIAL_RETENTION = 0.15
PARTIAL_LIFT = 0.02
CG1_BASELINES = {"44b6": 0.11879895561357702, "6bba": 0.11059137988640161}

DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_CG0 = ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.json"
DEFAULT_CG1 = ROOT / "reports/g02/screening/candgen/cg1_abstention_screen_v1.json"
DEFAULT_APPEARANCE_CONTRACT = ROOT / "reports/g02/appearance_direct_swap_contract_v1.json"
DEFAULT_EXISTING_BASE = ROOT / "reports/g02/trackastra_apncc_base_cache_v2/videos"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/screening/candgen/cg1b_representation_rescreen_v1.json"
)

REPRESENTATIONS = (
    "geometry_confidence",
    "v45_logit_margin",
    "apncc_margin",
    "apncc_geometry_rank",
    "apncc_motion_rank",
    "apncc_geometry_motion_v45_rank",
)
MECHANISMS = ("hard_per_video_topk", "family_conformal_threshold", "pu_family_threshold")


class CG1bError(RuntimeError):
    """The CG1b full-pool screen contract failed."""


def assert_safe_fit_only_path(path: Path) -> None:
    lowered = str(path).lower()
    token = next((value for value in FORBIDDEN_PATH_TOKENS if value in lowered), None)
    if token is not None:
        raise CG1bError(
            f"forbidden {token!r} artifact reference rejected before read: {path}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    assert_safe_fit_only_path(path)
    if not path.is_file():
        raise CG1bError(f"missing required input: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise CG1bError(f"expected JSON object: {path}")
    return value


def _family(identity: str) -> str:
    family = identity.split(":", 1)[1].split("_", 1)[0]
    if family not in FAMILIES:
        raise CG1bError(f"unexpected family identity: {identity}")
    return family


def _strict_scope(report: Mapping[str, Any], source: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope = report.get("scope")
    if not isinstance(scope, Mapping):
        raise CG1bError(f"missing scope: {source}")
    train = tuple(str(value) for value in scope.get("train_video_ids", ()))
    validation = tuple(str(value) for value in scope.get("validation_video_ids", ()))
    if (
        scope.get("fit_only") is not True
        or len(train) != 105
        or len(validation) != 12
        or set(train) & set(validation)
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
        or any(not identity.startswith("train:") for identity in (*train, *validation))
    ):
        raise CG1bError(f"not the exact fit-only 105/12 scope: {source}")
    return train, validation


def _known_correct_pairs(row: Mapping[str, Any]) -> set[tuple[int, int]]:
    path = ROOT / str(row["path"])
    assert_safe_fit_only_path(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    pairs = np.asarray(value["labeled_challenger_pairs"], dtype=np.int64)
    labels = np.asarray(value["labeled_labels"], dtype=np.int8)
    grouped: dict[tuple[int, int], set[int]] = {}
    for pair, label in zip(pairs, labels, strict=True):
        grouped.setdefault((int(pair[0]), int(pair[1])), set()).add(int(label))
    if any(len(observed) != 1 for observed in grouped.values()):
        raise CG1bError(f"mixed known labels in {row['identity']}")
    return {pair for pair, observed in grouped.items() if next(iter(observed)) == 1}


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values.copy()
    return rankdata(values, method="average") / len(values)


def _flatten_scores(
    *,
    challengers: np.ndarray,
    row_features: np.ndarray,
    offsets: np.ndarray,
    geometry: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(challengers)
    if (
        challengers.shape != (count, 2)
        or offsets.shape != (count + 1,)
        or int(offsets[-1]) != len(row_features)
        or row_features.shape[1:] != (48,)
        or geometry.shape != (count,)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise CG1bError("flattened APNCC proposal arrays disagree")
    starts = offsets[:-1]
    apncc = np.minimum.reduceat(row_features[:, 44], starts).astype(np.float64)
    motion = np.minimum.reduceat(row_features[:, 46], starts).astype(np.float64)
    v45 = np.minimum.reduceat(row_features[:, 39], starts).astype(np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    ranks = {
        "apncc": _rank01(apncc),
        "motion": _rank01(motion),
        "v45": _rank01(v45),
        "geometry": _rank01(geometry),
    }
    return {
        "geometry_confidence": geometry,
        "v45_logit_margin": v45,
        "apncc_margin": apncc,
        "apncc_geometry_rank": (ranks["apncc"] + ranks["geometry"]) / 2.0,
        "apncc_motion_rank": (ranks["apncc"] + ranks["motion"]) / 2.0,
        "apncc_geometry_motion_v45_rank": (
            ranks["apncc"] + ranks["geometry"] + ranks["motion"] + ranks["v45"]
        )
        / 4.0,
    }


def _from_windows(
    windows: Sequence[point_context_swap.RuntimeProposalWindow],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    challengers: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    offsets = [0]
    geometry: list[float] = []
    for window in windows:
        window.validate()
        for challenger, row_features, confidence in zip(
            window.challengers,
            window.row_features,
            window.geometry_confidences,
            strict=True,
        ):
            challengers.append(tuple(int(value) for value in challenger))
            rows.append(np.asarray(row_features, dtype=np.float32))
            offsets.append(offsets[-1] + len(row_features))
            geometry.append(float(confidence))
    challenger_array = np.asarray(challengers, dtype=np.int64).reshape(-1, 2)
    row_array = (
        np.concatenate(rows, axis=0)
        if rows
        else np.empty((0, 48), dtype=np.float32)
    )
    return challenger_array, _flatten_scores(
        challengers=challenger_array,
        row_features=row_array,
        offsets=np.asarray(offsets, dtype=np.int64),
        geometry=np.asarray(geometry, dtype=np.float64),
    )


def _from_existing_base(path: Path, identity: str) -> tuple[np.ndarray, dict[str, np.ndarray], Mapping[str, Any]]:
    assert_safe_fit_only_path(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "celltrack-alpha.trackastra-apncc-base-video.v2"
        or value.get("identity") != identity
    ):
        raise CG1bError(f"existing APNCC base identity drifted: {identity}")
    challengers = np.asarray(value["challengers"], dtype=np.int64)
    return (
        challengers,
        _flatten_scores(
            challengers=challengers,
            row_features=np.asarray(value["row_features"], dtype=np.float32),
            offsets=np.asarray(value["proposal_offsets"], dtype=np.int64),
            geometry=np.asarray(value["geometry_confidences"], dtype=np.float64),
        ),
        value["pool_counts"],
    )


def _label_proposals(
    challengers: np.ndarray,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
) -> np.ndarray:
    labels = np.zeros(len(challengers), dtype=np.int8)
    for index, (source, target) in enumerate(challengers):
        gt_source = detection_to_gt.get(int(source))
        gt_target = detection_to_gt.get(int(target))
        labels[index] = int(
            gt_source is not None
            and gt_target is not None
            and (int(gt_source), int(gt_target)) in gt_edges
        )
    return labels


def _video_record(
    *,
    identity: str,
    prepare_row: Mapping[str, Any],
    reachability_row: Mapping[str, Any],
    cache_root: Path,
    scorer: Any,
    geometry_gate: torch.nn.Module,
    geometry_threshold: float,
    store: ZipStore,
    existing_base: Path,
) -> dict[str, Any]:
    slug = identity.replace(":", "__", 1)
    meta_path = cache_root / f"{slug}.json"
    cache_path = cache_root / f"{slug}.bhc"
    for path in (meta_path, cache_path):
        assert_safe_fit_only_path(path)
    meta = _load_json(meta_path)
    cache = bh_model.load_sparse_cache(
        cache_path, expected_identity=sweep._cache_identity(meta)
    )
    selected, _probabilities, _no_link, graph = _baseline(cache, scorer)
    if int(graph["raw_selected_edges"]) != int(prepare_row["graph"]["raw_selected_edges"]):
        raise CG1bError(f"selected graph identity drifted: {identity}")

    base_path = existing_base / f"{slug}.pt"
    if base_path.is_file():
        challengers, scores, counts = _from_existing_base(base_path, identity)
        source = "existing-apncc-base-v2"
    else:
        dataset = identity.split(":", 1)[1]
        image = zarr.open_group(
            store=store, path=f"train/{dataset}.zarr", mode="r"
        )["0"]
        windows, counts = point_context_swap.build_full_pool_windows(
            cache,
            scorer,
            selected,
            geometry_gate,
            image,
            identity=identity,
            family=int(_family(identity) == "6bba"),
            geometry_threshold=geometry_threshold,
        )
        challengers, scores = _from_windows(windows)
        source = "streamed-existing-representation"
        del windows, image

    metrics = reachability_row["metrics"]
    pool_size = int(metrics["candidate_pool"]["r10_k08"])
    correct_pool = int(metrics["policy_reachable"]["r10_k08"])
    raw_background = pool_size - correct_pool
    if int(counts["candidate_pairs"]) != pool_size:
        raise CG1bError(
            f"full r10/k8 candidate count drifted for {identity}: "
            f"{counts['candidate_pairs']} != {pool_size}"
        )

    dataset = identity.split(":", 1)[1]
    gt = read_geff_arrays(store, "train", dataset)
    detection_to_gt, gt_edges_raw = match_detections_to_gt(cache.detections, gt)
    labels = _label_proposals(
        challengers,
        detection_to_gt,
        {
            (int(source_gt), int(target_gt))
            for source_gt, target_gt in gt_edges_raw
        },
    )
    known_correct = _known_correct_pairs(prepare_row)
    known_mask = np.asarray(
        [
            (int(source), int(target)) in known_correct
            for source, target in challengers
        ],
        dtype=bool,
    )
    if np.any(known_mask & (labels != 1)):
        raise CG1bError(f"known-correct labels disagree with exact GT: {identity}")
    budget = math.floor(
        float(prepare_row["cg1b_acceptance_rate"]) * raw_background
    )
    result = {
        "identity": identity,
        "family": _family(identity),
        "fold": _fold(identity, 5),
        "source": source,
        "candidate_pairs": pool_size,
        "raw_background": raw_background,
        "raw_proposals": int(counts["raw_proposals"]),
        "geometry_proposals": len(challengers),
        "actual_correct_geometry_proposals": int(labels.sum()),
        "known_correct_total": len(known_correct),
        "known_correct_geometry_present": int(known_mask.sum()),
        "budget": budget,
        "challengers": challengers,
        "labels": labels,
        "known_mask": known_mask,
        "scores": scores,
    }
    del cache
    gc.collect()
    return result


def _topk_video(video: Mapping[str, Any], representation: str) -> np.ndarray:
    scores = np.asarray(video["scores"][representation], dtype=np.float64)
    challengers = np.asarray(video["challengers"], dtype=np.int64)
    order = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            int(challengers[index, 0]),
            int(challengers[index, 1]),
        ),
    )
    accepted = np.zeros(len(scores), dtype=bool)
    accepted[np.asarray(order[: int(video["budget"])], dtype=np.int64)] = True
    return accepted


def _calibration_threshold(
    videos: Sequence[Mapping[str, Any]], representation: str
) -> float:
    cutoffs = []
    for video in videos:
        background = np.asarray(video["scores"][representation])[
            np.asarray(video["labels"]) == 0
        ]
        budget = int(video["budget"])
        if len(background) <= budget:
            cutoffs.append(-math.inf)
            continue
        ordered = np.sort(background)[::-1]
        cutoffs.append(float(ordered[budget]))
    return max(cutoffs, default=math.inf)


def _pu_transform(
    calibration: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    representation: str,
    *,
    bins: int = 64,
) -> tuple[list[np.ndarray], float]:
    positive = np.concatenate(
        [
            np.asarray(video["scores"][representation])[
                np.asarray(video["known_mask"])
            ]
            for video in calibration
        ]
    )
    unlabeled = np.concatenate(
        [np.asarray(video["scores"][representation]) for video in calibration]
    )
    if not len(positive) or not len(unlabeled):
        return [np.zeros(len(video["labels"])) for video in target], 0.0
    edges = np.unique(
        np.quantile(
            np.concatenate((positive, unlabeled)),
            np.linspace(0.0, 1.0, bins + 1),
        )
    )
    if len(edges) < 3:
        return [np.zeros(len(video["labels"])) for video in target], 0.0
    edges[0], edges[-1] = -math.inf, math.inf
    positive_count, _ = np.histogram(positive, bins=edges)
    unlabeled_count, _ = np.histogram(unlabeled, bins=edges)
    p_density = (positive_count + 0.5) / (
        len(positive) + 0.5 * len(positive_count)
    )
    u_density = (unlabeled_count + 0.5) / (
        len(unlabeled) + 0.5 * len(unlabeled_count)
    )
    prior = float(np.clip(np.min(u_density / p_density), 1.0e-8, 1.0))
    posterior = np.clip(prior * p_density / u_density, 0.0, 1.0)
    converted = []
    for video in target:
        values = np.asarray(video["scores"][representation])
        index = np.searchsorted(edges[1:-1], values, side="right")
        converted.append(posterior[index])
    return converted, prior


def _summarize_decisions(
    videos: Sequence[Mapping[str, Any]],
    decisions: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(videos) != len(decisions):
        raise CG1bError("decision/video count mismatch")
    rows = []
    for video, accepted in zip(videos, decisions, strict=True):
        accepted = np.asarray(accepted, dtype=bool)
        labels = np.asarray(video["labels"])
        known = np.asarray(video["known_mask"])
        fp = int(np.sum(accepted & (labels == 0)))
        rows.append(
            {
                "identity": str(video["identity"]),
                "family": str(video["family"]),
                "budget": int(video["budget"]),
                "background_fp": fp,
                "known_correct_kept": int(np.sum(accepted & known)),
                "known_correct_total": int(video["known_correct_total"]),
                "budget_pass": fp <= int(video["budget"]),
            }
        )
    result = {"videos": rows, "families": {}}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        kept = sum(row["known_correct_kept"] for row in selected)
        total = sum(row["known_correct_total"] for row in selected)
        fp = sum(row["background_fp"] for row in selected)
        budget = sum(row["budget"] for row in selected)
        result["families"][family] = {
            "video_count": len(selected),
            "known_correct_kept": kept,
            "known_correct_total": total,
            "correct_retention": kept / max(total, 1),
            "background_fp": fp,
            "background_budget": budget,
            "all_videos_budget_pass": all(row["budget_pass"] for row in selected),
        }
    return result


def _screen_family(
    videos: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, Any]:
    family_videos = [video for video in videos if video["family"] == family]
    if not family_videos:
        return {}
    result: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        topk = _summarize_decisions(
            family_videos,
            [_topk_video(video, representation) for video in family_videos],
        )
        result[f"{representation}__hard_per_video_topk"] = {
            "representation": representation,
            "mechanism": "hard_per_video_topk",
            **topk["families"][family],
        }

        conformal_folds = []
        pu_folds = []
        for fold in range(5):
            calibration = [
                video for video in family_videos if int(video["fold"]) != fold
            ]
            held = [video for video in family_videos if int(video["fold"]) == fold]
            threshold = _calibration_threshold(calibration, representation)
            conformal_decisions = [
                np.asarray(video["scores"][representation]) > threshold
                for video in held
            ]
            conformal_folds.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "summary": _summarize_decisions(held, conformal_decisions),
                }
            )

            calibration_pu, prior = _pu_transform(
                calibration, calibration, representation
            )
            held_pu, _ = _pu_transform(calibration, held, representation)
            calibration_wrapped = [
                {**video, "scores": {"pu": scores}}
                for video, scores in zip(calibration, calibration_pu, strict=True)
            ]
            threshold_pu = _calibration_threshold(calibration_wrapped, "pu")
            pu_decisions = [scores > threshold_pu for scores in held_pu]
            pu_folds.append(
                {
                    "fold": fold,
                    "threshold": threshold_pu,
                    "class_prior": prior,
                    "summary": _summarize_decisions(held, pu_decisions),
                }
            )

        for mechanism, folds in (
            ("family_conformal_threshold", conformal_folds),
            ("pu_family_threshold", pu_folds),
        ):
            video_rows = [
                row
                for fold in folds
                for row in fold["summary"]["videos"]
            ]
            kept = sum(row["known_correct_kept"] for row in video_rows)
            total = sum(row["known_correct_total"] for row in video_rows)
            fp = sum(row["background_fp"] for row in video_rows)
            budget = sum(row["budget"] for row in video_rows)
            result[f"{representation}__{mechanism}"] = {
                "representation": representation,
                "mechanism": mechanism,
                "video_count": len(video_rows),
                "known_correct_kept": kept,
                "known_correct_total": total,
                "correct_retention": kept / max(total, 1),
                "background_fp": fp,
                "background_budget": budget,
                "all_videos_budget_pass": all(
                    row["budget_pass"] for row in video_rows
                ),
                "folds": folds,
            }
    return result


def _best_candidate(results: Mapping[str, Mapping[str, Any]], family: str) -> tuple[str, Mapping[str, Any]]:
    eligible = [
        (name, row)
        for name, row in results.items()
        if bool(row.get("all_videos_budget_pass"))
    ]
    if not eligible:
        return "", {}
    return max(
        eligible,
        key=lambda item: (
            float(item[1]["correct_retention"]),
            -int(item[1]["background_fp"]),
            item[0],
        ),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(child) for child in value]
    return value


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    for path in (
        args.prepare,
        args.reachability,
        args.cg0,
        args.cg1,
        args.appearance_contract,
        args.cache,
        args.zip,
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        args.existing_base,
        args.output,
    ):
        assert_safe_fit_only_path(path)
    prepare = _load_json(args.prepare)
    reachability = _load_json(args.reachability)
    cg0 = _load_json(args.cg0)
    cg1 = _load_json(args.cg1)
    train_ids, validation_ids = _strict_scope(prepare, args.prepare)
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or reachability.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or cg0.get("verdict") != "GO"
        or cg1.get("verdict") != "PARK"
    ):
        raise CG1bError("upstream CG0/CG1/fit artifacts do not reconcile")

    current = next(
        row
        for row in cg0["policy_ladder"]
        if row["policy"] == "current_r10_k08"
    )
    acceptance_rates = {
        family: float(
            current["families"][family]["background_budget_at_goal"][
                "maximum_background_acceptance_fraction"
            ]
        )
        for family in FAMILIES
    }
    fit117_budgets = {
        family: int(
            current["families"][family]["background_budget_at_goal"][
                "maximum_background_false_positive_edges"
            ]
        )
        for family in (*FAMILIES, "overall")
    }
    prepare_rows = {
        str(row["identity"]): {**row, "cg1b_acceptance_rate": acceptance_rates[_family(str(row["identity"]))]}
        for row in prepare["videos"]
    }
    reachability_rows = {
        str(row["identity"]): row for row in reachability["videos"]
    }
    if set(prepare_rows) != set(reachability_rows):
        raise CG1bError("prepare/reachability video coverage differs")

    appearance_contract, geometry_gate = appearance_swap.load_contract(
        args.appearance_contract,
        expected_sha256=_sha256(args.appearance_contract),
        root=ROOT,
    )
    base_model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        _sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None or int(scorer.config.neighbor_cap) != 8:
        raise CG1bError("frozen r10/cap-8 scorer did not restore")
    geometry_threshold = float(appearance_contract["geometry_gate"]["threshold"])

    videos: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    processed_families: list[str] = []
    early_44b6_park = False
    with ZipStore(args.zip, mode="r") as store:
        for family in FAMILIES:
            identities = [
                identity for identity in train_ids if _family(identity) == family
            ]
            for ordinal, identity in enumerate(identities, start=1):
                video = _video_record(
                    identity=identity,
                    prepare_row=prepare_rows[identity],
                    reachability_row=reachability_rows[identity],
                    cache_root=args.cache,
                    scorer=scorer,
                    geometry_gate=geometry_gate,
                    geometry_threshold=geometry_threshold,
                    store=store,
                    existing_base=args.existing_base,
                )
                videos.append(video)
                print(
                    f"CG1b {family} {ordinal:03d}/{len(identities)} {identity}: "
                    f"geometry={video['geometry_proposals']} "
                    f"known={video['known_correct_geometry_present']}/"
                    f"{video['known_correct_total']} budget={video['budget']}",
                    flush=True,
                )
            results[family] = _screen_family(videos, family)
            processed_families.append(family)
            best_name, best = _best_candidate(results[family], family)
            print(
                f"CG1b {family} best={best_name} "
                f"retention={float(best.get('correct_retention', 0.0)):.6f}",
                flush=True,
            )
            if family == "44b6":
                retention = float(best.get("correct_retention", 0.0))
                if not (
                    retention >= PARTIAL_RETENTION
                    and retention >= CG1_BASELINES["44b6"] + PARTIAL_LIFT
                ):
                    early_44b6_park = True
                    break

    # Strip large in-memory arrays before constructing the report.
    video_reports = [
        {
            key: value
            for key, value in video.items()
            if key not in {"challengers", "labels", "known_mask", "scores"}
        }
        for video in videos
    ]
    best_by_family = {}
    for family in processed_families:
        name, row = _best_candidate(results[family], family)
        best_by_family[family] = {"candidate": name, **dict(row)}

    if early_44b6_park:
        verdict = "PARK"
        reason = (
            "44b6 failed the preregistered PARTIAL floor, so hardest-family "
            "falsification halted before 6bba."
        )
    elif set(processed_families) == set(FAMILIES):
        shared_candidates = set(results["44b6"]) & set(results["6bba"])
        go_candidates = [
            name
            for name in shared_candidates
            if all(
                bool(results[family][name]["all_videos_budget_pass"])
                and float(results[family][name]["correct_retention"]) >= GO_RETENTION
                for family in FAMILIES
            )
        ]
        partial_candidates = [
            name
            for name in shared_candidates
            if all(
                bool(results[family][name]["all_videos_budget_pass"])
                and float(results[family][name]["correct_retention"])
                >= PARTIAL_RETENTION
                and float(results[family][name]["correct_retention"])
                >= CG1_BASELINES[family] + PARTIAL_LIFT
                for family in FAMILIES
            )
        ]
        if go_candidates:
            verdict = "GO"
            reason = f"shared candidate passes both families: {sorted(go_candidates)[0]}"
        elif partial_candidates:
            verdict = "PARTIAL"
            reason = (
                "at least one shared candidate materially lifts both families "
                "but remains below the 20% GO floor"
            )
        else:
            verdict = "PARK"
            reason = "no shared candidate reaches the preregistered PARTIAL floor"
    else:
        raise CG1bError("family execution ended without a valid verdict")

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "verdict": verdict,
        "stage": 1,
        "scope": {
            "fit_only": True,
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "validation_used": False,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "preregistration": {
            "go_retention_both_families": GO_RETENTION,
            "partial_retention_both_families": PARTIAL_RETENTION,
            "partial_minimum_lift_each_family": PARTIAL_LIFT,
            "cg1_baselines": CG1_BASELINES,
            "family_order": list(FAMILIES),
            "fit117_background_budgets": fit117_budgets,
            "per_video_budget": (
                "floor(CG0 family acceptance rate * exact video raw background)"
            ),
            "early_stop": (
                "halt after 44b6 if its best exactly budget-compliant candidate "
                "misses the PARTIAL floor"
            ),
        },
        "full_pool_certification": {
            "candidate_policy": "exact r10/k8",
            "sampled_background": False,
            "implicit_rejection": (
                "all candidates outside selected-graph conflict plus frozen "
                "geometry-prefilter support receive no finite proposal score"
            ),
            "actual_gt_background_counted": True,
            "per_video_budget_enforced": True,
        },
        "representations": list(REPRESENTATIONS),
        "mechanisms": list(MECHANISMS),
        "processed_families": processed_families,
        "results": results,
        "best_by_family": best_by_family,
        "videos": video_reports,
        "decision": {
            "reason": reason,
            "cg2_authorized": verdict == "GO",
            "stage2_authorized": verdict == "PARTIAL",
            "escalation_required": verdict == "PARK",
        },
        "provenance": {
            "prepare": str(args.prepare.relative_to(ROOT)),
            "prepare_sha256": _sha256(args.prepare),
            "reachability": str(args.reachability.relative_to(ROOT)),
            "reachability_sha256": _sha256(args.reachability),
            "cg0": str(args.cg0.relative_to(ROOT)),
            "cg0_sha256": _sha256(args.cg0),
            "cg1": str(args.cg1.relative_to(ROOT)),
            "cg1_sha256": _sha256(args.cg1),
            "appearance_contract": str(args.appearance_contract.relative_to(ROOT)),
            "appearance_contract_sha256": _sha256(args.appearance_contract),
            "edge_artifact": str(args.edge_artifact.relative_to(ROOT)),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
            "existing_base_reused_without_writes": True,
        },
        "execution": {
            "device": "cpu",
            "model_training": False,
            "new_feature_cache_created": False,
            "graph_replay": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    return _json_safe(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    parser.add_argument("--cg0", type=Path, default=DEFAULT_CG0)
    parser.add_argument("--cg1", type=Path, default=DEFAULT_CG1)
    parser.add_argument(
        "--appearance-contract", type=Path, default=DEFAULT_APPEARANCE_CONTRACT
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--existing-base", type=Path, default=DEFAULT_EXISTING_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise CG1bError(f"refusing to overwrite output: {args.output}")
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": report["verdict"],
                "processed_families": report["processed_families"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

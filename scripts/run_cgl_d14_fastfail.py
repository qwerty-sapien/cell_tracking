#!/usr/bin/env python3
"""CGL: fit-only frozen-core compact-head tail-ranking fast-fail.

The released TemporalUNet3D core is never executed or trained here: its
hash-pinned 32-D node embeddings are read from the exact V45 sparse caches.
Five deterministic video folds train only a 4,385-parameter CPU head with a
same-video budget-tail pairwise ranking objective.  Every held candidate is
scored; no sampled-background estimate is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
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


SCHEMA = "celltrack-alpha.cgl-d14-fastfail.v1"
FAMILIES = ("44b6", "6bba")
FORBIDDEN_PATH_TOKENS = ("cv57", "holdout")
FEATURE_DIM = 135
HIDDEN_DIM = 32
HEAD_PARAMETERS = 4_385
FOLD_COUNT = 5
SEED = 250725
LEARNING_RATE = 1.0e-4
TRAIN_STEPS_PER_PHASE = 300
BATCH_VIDEOS_PER_FAMILY = 8
PAIRS_PER_VIDEO = 128
HARD_NEGATIVE_MULTIPLIER = 64
MINIMUM_HARD_NEGATIVES = 1024
PAIRWISE_MARGIN = 0.25
GO_44B6 = 0.15
NONCOLLAPSE_6BBA = 0.10
PARK_44B6 = 0.05

DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_CG0 = ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.json"
DEFAULT_CG1B = (
    ROOT / "reports/g02/screening/candgen/cg1b_representation_rescreen_v1.json"
)
DEFAULT_CANDIDATE = ROOT / "reports/g02/model_candidate_cgl_tail_rank_head_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/screening/candgen/cgl_d14_fastfail_v1.json"
)
DEFAULT_HEADS = (
    ROOT / "reports/g02/screening/candgen/cgl_d14_fastfail_v1_heads.pt"
)


class CGLFastFailError(RuntimeError):
    """The frozen-core tail-ranking fast-fail contract failed."""


def assert_safe_fit_only_path(path: Path) -> None:
    lowered = str(path).lower()
    token = next(
        (value for value in FORBIDDEN_PATH_TOKENS if value in lowered), None
    )
    if token is not None:
        raise CGLFastFailError(
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
        raise CGLFastFailError(f"missing required input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CGLFastFailError(f"expected JSON object: {path}")
    return value


def _family(identity: str) -> str:
    family = identity.split(":", 1)[1].split("_", 1)[0]
    if family not in FAMILIES:
        raise CGLFastFailError(f"unexpected family identity: {identity}")
    return family


def _strict_scope(
    report: Mapping[str, Any], source: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope = report.get("scope")
    if not isinstance(scope, Mapping):
        raise CGLFastFailError(f"missing scope: {source}")
    train = tuple(str(value) for value in scope.get("train_video_ids", ()))
    validation = tuple(
        str(value) for value in scope.get("validation_video_ids", ())
    )
    if (
        scope.get("fit_only") is not True
        or len(train) != 105
        or len(validation) != 12
        or set(train) & set(validation)
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
        or any(
            not identity.startswith("train:")
            for identity in (*train, *validation)
        )
    ):
        raise CGLFastFailError(f"not the exact fit-only 105/12 scope: {source}")
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
        raise CGLFastFailError(f"mixed known labels in {row['identity']}")
    return {
        pair for pair, observed in grouped.items() if next(iter(observed)) == 1
    }


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, np.float32(1.0e-6))


def compact_pair_features(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    displacement_um: np.ndarray,
    distance_um: np.ndarray,
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    checkpoint_logits: np.ndarray,
) -> np.ndarray:
    source = _unit_rows(source_features)
    target = _unit_rows(target_features)
    displacement = np.asarray(displacement_um, dtype=np.float32) / 10.0
    distance = np.asarray(distance_um, dtype=np.float32).reshape(-1, 1) / 10.0
    source_score = np.asarray(source_scores, dtype=np.float32).reshape(-1, 1)
    target_score = np.asarray(target_scores, dtype=np.float32).reshape(-1, 1)
    checkpoint = np.clip(
        np.asarray(checkpoint_logits, dtype=np.float32).reshape(-1, 1),
        -10.0,
        10.0,
    ) / 10.0
    result = np.concatenate(
        (
            source,
            target,
            target - source,
            target * source,
            displacement,
            distance,
            source_score,
            target_score,
            checkpoint,
        ),
        axis=1,
    )
    if result.shape[1] != FEATURE_DIM or not np.isfinite(result).all():
        raise CGLFastFailError("compact frozen-core features are invalid")
    return np.asarray(result, dtype=np.float16)


def _label_pairs(
    pairs: np.ndarray,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
) -> np.ndarray:
    labels = np.zeros(len(pairs), dtype=np.int8)
    for index, (source, target) in enumerate(pairs):
        gt_source = detection_to_gt.get(int(source))
        gt_target = detection_to_gt.get(int(target))
        labels[index] = int(
            gt_source is not None
            and gt_target is not None
            and (int(gt_source), int(gt_target)) in gt_edges
        )
    return labels


def _one_to_one_gt_edges(
    gt_edges: Sequence[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Match CG0: only one-to-one GT associations are realizable positives."""

    edges = [(int(source), int(target)) for source, target in gt_edges]
    outdegree = Counter(source for source, _target in edges)
    indegree = Counter(target for _source, target in edges)
    return {
        (source, target)
        for source, target in edges
        if outdegree[source] == 1 and indegree[target] == 1
    }


@dataclass
class VideoRows:
    identity: str
    family: str
    fold: int
    features: np.ndarray
    labels: np.ndarray
    known_mask: np.ndarray
    base_scores: np.ndarray
    candidate_pairs: int
    raw_background: int
    budget: int
    known_correct_total: int

    def validate(self) -> None:
        count = self.candidate_pairs
        if (
            self.family not in FAMILIES
            or not 0 <= self.fold < FOLD_COUNT
            or self.features.shape != (count, FEATURE_DIM)
            or self.features.dtype != np.float16
            or self.labels.shape != (count,)
            or self.known_mask.shape != (count,)
            or self.base_scores.shape != (count,)
            or self.raw_background != int(np.sum(self.labels == 0))
            or self.known_correct_total != int(np.sum(self.known_mask))
            or np.any(self.known_mask & (self.labels != 1))
            or self.budget < 0
        ):
            raise CGLFastFailError(f"video row contract failed: {self.identity}")


def _video_rows(
    *,
    identity: str,
    prepare_row: Mapping[str, Any],
    reachability_row: Mapping[str, Any],
    acceptance_rate: float,
    cache_root: Path,
    scorer: Any,
    store: ZipStore,
) -> VideoRows:
    slug = identity.replace(":", "__", 1)
    meta_path = cache_root / f"{slug}.json"
    cache_path = cache_root / f"{slug}.bhc"
    meta = _load_json(meta_path)
    cache = bh_model.load_sparse_cache(
        cache_path, expected_identity=sweep._cache_identity(meta)
    )
    dataset = identity.split(":", 1)[1]
    gt = read_geff_arrays(store, "train", dataset)
    detection_to_gt, gt_edges_raw = match_detections_to_gt(cache.detections, gt)
    gt_edges = _one_to_one_gt_edges(gt_edges_raw)
    known = _known_correct_pairs(prepare_row)

    feature_parts: list[np.ndarray] = []
    pair_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    for context, source, target in zip(
        cache.contexts, cache.detections[:-1], cache.detections[1:], strict=True
    ):
        candidates, checkpoint_logits, _residual = scorer.candidate_features(
            context, source, target
        )
        if (
            not np.array_equal(context.source_ids, source.node_ids)
            or not np.array_equal(context.target_ids, target.node_ids)
        ):
            raise CGLFastFailError(
                f"pair-context node ordering drifted: {identity}"
            )
        source_index = np.asarray(candidates.source_index, dtype=np.int64)
        target_index = np.asarray(candidates.target_index, dtype=np.int64)
        feature_parts.append(
            compact_pair_features(
                source_features=context.source_feat[source_index],
                target_features=context.target_feat[target_index],
                displacement_um=candidates.displacement_um,
                distance_um=candidates.distance_um,
                source_scores=source.scores[source_index],
                target_scores=target.scores[target_index],
                checkpoint_logits=checkpoint_logits.detach().cpu().numpy(),
            )
        )
        pair_parts.append(
            np.stack((candidates.source_ids, candidates.target_ids), axis=1)
        )
        base_parts.append(
            checkpoint_logits.detach().cpu().numpy().astype(np.float32)
        )
    features = np.concatenate(feature_parts, axis=0)
    pairs = np.concatenate(pair_parts, axis=0).astype(np.int64, copy=False)
    base_scores = np.concatenate(base_parts).astype(np.float32, copy=False)
    labels = _label_pairs(pairs, detection_to_gt, gt_edges)
    known_mask = np.asarray(
        [
            (int(source), int(target)) in known
            for source, target in pairs
        ],
        dtype=bool,
    )

    metrics = reachability_row["metrics"]
    expected_pool = int(metrics["candidate_pool"]["r10_k08"])
    expected_correct = int(metrics["policy_reachable"]["r10_k08"])
    expected_background = expected_pool - expected_correct
    if (
        len(features) != expected_pool
        or int(labels.sum()) != expected_correct
        or int(np.sum(labels == 0)) != expected_background
        or int(known_mask.sum()) != len(known)
    ):
        raise CGLFastFailError(
            f"exact r10/k8 pool or GT labels drifted for {identity}: "
            f"{len(features)}/{int(labels.sum())}/{int(known_mask.sum())} versus "
            f"{expected_pool}/{expected_correct}/{len(known)}"
        )
    result = VideoRows(
        identity=identity,
        family=_family(identity),
        fold=_fold(identity, FOLD_COUNT),
        features=features,
        labels=labels,
        known_mask=known_mask,
        base_scores=base_scores,
        candidate_pairs=expected_pool,
        raw_background=expected_background,
        budget=math.floor(acceptance_rate * expected_background),
        known_correct_total=len(known),
    )
    result.validate()
    del cache, pairs, gt
    gc.collect()
    return result


class TailRankHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        if sum(parameter.numel() for parameter in self.parameters()) != HEAD_PARAMETERS:
            raise CGLFastFailError("head parameter identity drifted")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _score(model: TailRankHead, features: np.ndarray) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), 32768):
            batch = torch.from_numpy(
                features[start : start + 32768].astype(np.float32)
            )
            parts.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _hard_negative_indices(
    scores: np.ndarray, labels: np.ndarray, budget: int
) -> np.ndarray:
    negatives = np.flatnonzero(labels == 0)
    keep = min(
        len(negatives),
        max(MINIMUM_HARD_NEGATIVES, HARD_NEGATIVE_MULTIPLIER * budget),
    )
    if keep == 0:
        return negatives
    order = np.lexsort((negatives, -scores[negatives]))
    return negatives[order[:keep]]


def _head_sha256(model: TailRankHead) -> str:
    digest = hashlib.sha256(
        f"{FEATURE_DIM}:{HIDDEN_DIM}:{HEAD_PARAMETERS}".encode("ascii")
    )
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        array = value.detach().cpu().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _train_phase(
    *,
    model: TailRankHead,
    optimizer: torch.optim.Optimizer,
    training: Sequence[VideoRows],
    hard_negatives: Mapping[str, np.ndarray],
    generator: np.random.Generator,
    steps: int,
) -> dict[str, Any]:
    eligible = {
        family: [
            video
            for video in training
            if video.family == family
            and np.any(video.labels == 1)
            and len(hard_negatives[video.identity])
        ]
        for family in FAMILIES
    }
    if any(not eligible[family] for family in FAMILIES):
        raise CGLFastFailError("a family has no trainable video")
    losses = []
    model.train()
    for _step in range(steps):
        positive_parts = []
        negative_parts = []
        for family in FAMILIES:
            chosen = generator.choice(
                len(eligible[family]),
                size=BATCH_VIDEOS_PER_FAMILY,
                replace=True,
            )
            for video_index in chosen:
                video = eligible[family][int(video_index)]
                positives = np.flatnonzero(video.labels == 1)
                negatives = hard_negatives[video.identity]
                positive_index = generator.choice(
                    positives, size=PAIRS_PER_VIDEO, replace=True
                )
                negative_index = generator.choice(
                    negatives, size=PAIRS_PER_VIDEO, replace=True
                )
                positive_parts.append(video.features[positive_index])
                negative_parts.append(video.features[negative_index])
        positive = torch.from_numpy(
            np.concatenate(positive_parts).astype(np.float32)
        )
        negative = torch.from_numpy(
            np.concatenate(negative_parts).astype(np.float32)
        )
        optimizer.zero_grad(set_to_none=True)
        positive_score = model(positive)
        negative_score = model(negative)
        pairwise = F.softplus(
            negative_score - positive_score + PAIRWISE_MARGIN
        ).mean()
        distributional = F.softplus(
            negative_score.mean() - positive_score.mean() + PAIRWISE_MARGIN
        )
        loss = pairwise + 0.25 * distributional
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
    }


def _train_fold(
    training: Sequence[VideoRows], fold: int
) -> tuple[TailRankHead, dict[str, Any]]:
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    random.seed(SEED + fold)
    generator = np.random.default_rng(SEED + fold)
    model = TailRankHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4
    )
    initial_hard = {
        video.identity: _hard_negative_indices(
            video.base_scores, video.labels, video.budget
        )
        for video in training
    }
    phase1 = _train_phase(
        model=model,
        optimizer=optimizer,
        training=training,
        hard_negatives=initial_hard,
        generator=generator,
        steps=TRAIN_STEPS_PER_PHASE,
    )
    refreshed = {}
    for video in training:
        scores = _score(model, video.features)
        refreshed[video.identity] = _hard_negative_indices(
            scores, video.labels, video.budget
        )
    phase2 = _train_phase(
        model=model,
        optimizer=optimizer,
        training=training,
        hard_negatives=refreshed,
        generator=generator,
        steps=TRAIN_STEPS_PER_PHASE,
    )
    return model, {
        "fold": fold,
        "learning_rate": LEARNING_RATE,
        "pairwise_margin": PAIRWISE_MARGIN,
        "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
        "minimum_hard_negatives": MINIMUM_HARD_NEGATIVES,
        "training_video_count": len(training),
        "phase1": phase1,
        "phase2": phase2,
        "head_sha256": _head_sha256(model),
    }


def _ranking_decision(
    scores: np.ndarray, labels: np.ndarray, budget: int
) -> np.ndarray:
    """Metric-only ordering cut after the budget-th held background item."""

    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, -scores))
    accepted = np.zeros(len(scores), dtype=bool)
    background = 0
    for index in order:
        if labels[index] == 0:
            if background >= budget:
                break
            background += 1
        accepted[index] = True
    return accepted


def _hard_topk_decision(scores: np.ndarray, budget: int) -> np.ndarray:
    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, -scores))
    accepted = np.zeros(len(scores), dtype=bool)
    accepted[order[:budget]] = True
    return accepted


def _video_metrics(
    video: VideoRows, scores: np.ndarray, decision: np.ndarray
) -> dict[str, Any]:
    fp = int(np.sum(decision & (video.labels == 0)))
    return {
        "identity": video.identity,
        "family": video.family,
        "fold": video.fold,
        "candidate_pairs": video.candidate_pairs,
        "raw_background": video.raw_background,
        "background_budget": video.budget,
        "background_fp": fp,
        "budget_pass": fp <= video.budget,
        "known_correct_kept": int(np.sum(decision & video.known_mask)),
        "known_correct_total": video.known_correct_total,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        kept = sum(int(row["known_correct_kept"]) for row in selected)
        total = sum(int(row["known_correct_total"]) for row in selected)
        result[family] = {
            "video_count": len(selected),
            "candidate_pairs": sum(
                int(row["candidate_pairs"]) for row in selected
            ),
            "raw_background": sum(
                int(row["raw_background"]) for row in selected
            ),
            "background_fp": sum(
                int(row["background_fp"]) for row in selected
            ),
            "background_budget": sum(
                int(row["background_budget"]) for row in selected
            ),
            "all_videos_budget_pass": all(
                bool(row["budget_pass"]) for row in selected
            ),
            "known_correct_kept": kept,
            "known_correct_total": total,
            "correct_retention": kept / max(total, 1),
        }
    return result


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    for path in (
        args.prepare,
        args.reachability,
        args.cg0,
        args.cg1b,
        args.candidate,
        args.cache,
        args.zip,
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        args.output,
        args.heads,
    ):
        assert_safe_fit_only_path(path)
    prepare = _load_json(args.prepare)
    reachability = _load_json(args.reachability)
    cg0 = _load_json(args.cg0)
    cg1b = _load_json(args.cg1b)
    candidate = _load_json(args.candidate)
    train_ids, validation_ids = _strict_scope(prepare, args.prepare)
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or reachability.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or cg0.get("verdict") != "GO"
        or cg1b.get("verdict") != "PARK"
        or candidate.get("candidate_id")
        != "cgl-temporalunet-tail-rank-head-v1"
        or candidate.get("decision") != "PASS"
    ):
        raise CGLFastFailError("upstream artifacts or candidate gate drifted")

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
    prepare_rows = {
        str(row["identity"]): row for row in prepare["videos"]
    }
    reachability_rows = {
        str(row["identity"]): row for row in reachability["videos"]
    }
    if set(prepare_rows) != set(reachability_rows):
        raise CGLFastFailError("prepare/reachability video coverage differs")

    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        _sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise CGLFastFailError("frozen released-core scorer did not restore")
    del base_model

    videos = []
    with ZipStore(args.zip, mode="r") as store:
        for ordinal, identity in enumerate(train_ids, start=1):
            video = _video_rows(
                identity=identity,
                prepare_row=prepare_rows[identity],
                reachability_row=reachability_rows[identity],
                acceptance_rate=acceptance_rates[_family(identity)],
                cache_root=args.cache,
                scorer=scorer,
                store=store,
            )
            videos.append(video)
            print(
                f"CGL features {ordinal:03d}/{len(train_ids)} {identity}: "
                f"pool={video.candidate_pairs} correct={int(video.labels.sum())} "
                f"known={video.known_correct_total} budget={video.budget}",
                flush=True,
            )

    held_ranking_rows = []
    held_topk_rows = []
    fold_reports = []
    head_states = {}
    for fold in range(FOLD_COUNT):
        training = [video for video in videos if video.fold != fold]
        held = [video for video in videos if video.fold == fold]
        model, optimization = _train_fold(training, fold)
        head_states[f"fold{fold}"] = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        for video in held:
            scores = _score(model, video.features)
            held_ranking_rows.append(
                _video_metrics(
                    video,
                    scores,
                    _ranking_decision(scores, video.labels, video.budget),
                )
            )
            held_topk_rows.append(
                _video_metrics(
                    video,
                    scores,
                    _hard_topk_decision(scores, video.budget),
                )
            )
        fold_reports.append(optimization)
        print(
            f"CGL fold {fold} head={optimization['head_sha256'][:12]} "
            f"loss={optimization['phase2']['final_loss']:.6f}",
            flush=True,
        )

    ranking = _aggregate(held_ranking_rows)
    hard_topk = _aggregate(held_topk_rows)
    r44 = float(ranking["44b6"]["correct_retention"])
    r6 = float(ranking["6bba"]["correct_retention"])
    if (
        r44 >= GO_44B6
        and r6 >= NONCOLLAPSE_6BBA
        and all(
            bool(ranking[family]["all_videos_budget_pass"])
            for family in FAMILIES
        )
    ):
        verdict = "GO"
        reason = "44b6 passed 15% and 6bba passed the 10% non-collapse floor"
    elif r44 < PARK_44B6:
        verdict = "DEAD"
        reason = "44b6 remained below the preregistered 5% material-lift floor"
    else:
        verdict = "PARK"
        reason = (
            "the learned head materially lifted 44b6 but missed GO, or 6bba "
            "fell below the preregistered non-collapse floor"
        )

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "verdict": verdict,
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
            "go_44b6_retention": GO_44B6,
            "noncollapse_6bba_retention": NONCOLLAPSE_6BBA,
            "park_44b6_retention": PARK_44B6,
            "fold_count": FOLD_COUNT,
            "seed": SEED,
            "primary_metric": (
                "OOF known-correct retention before the exact per-video "
                "background FP budget is exhausted"
            ),
            "secondary_deployable_lower_bound": (
                "OOF hard per-video top-k using k equal to the FP budget"
            ),
        },
        "head": {
            "candidate_id": candidate["candidate_id"],
            "frozen_core": "released TemporalUNet3D V45 32-D node embeddings",
            "frozen_core_weight_sha256": bundle.hashes.weight_sha256,
            "architecture": f"Linear({FEATURE_DIM},{HIDDEN_DIM})-GELU-Linear({HIDDEN_DIM},1)",
            "parameter_count": HEAD_PARAMETERS,
            "core_trainable": False,
            "head_trainable": True,
            "fold_head_sha256": [
                row["head_sha256"] for row in fold_reports
            ],
        },
        "objective": {
            "name": "per-video budget-tail pairwise distributional ranking",
            "generic_bce_used": False,
            "same_video_pairs": True,
            "hard_negative_refresh": True,
            "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
            "minimum_hard_negatives_per_video": MINIMUM_HARD_NEGATIVES,
            "pairwise_margin": PAIRWISE_MARGIN,
            "learning_rate": LEARNING_RATE,
            "steps_per_phase": TRAIN_STEPS_PER_PHASE,
            "phases": 2,
        },
        "full_pool_certification": {
            "candidate_policy": "exact current r10/k8",
            "sampled_background": False,
            "all_held_candidates_scored": True,
            "exact_gt_background_counted": True,
            "per_video_budget_enforced": True,
        },
        "results": {
            "ranking_at_fp_budget": ranking,
            "hard_topk_count_safe": hard_topk,
            "videos_ranking_at_fp_budget": held_ranking_rows,
            "videos_hard_topk": held_topk_rows,
            "fold_optimization": fold_reports,
        },
        "decision": {
            "reason": reason,
            "t03_analytically_authorized": verdict == "GO",
            "t03_gpu_human_gate_required": verdict == "GO",
            "option_d_escalation_required": verdict in {"PARK", "DEAD"},
        },
        "provenance": {
            "candidate_manifest": str(args.candidate.relative_to(ROOT)),
            "candidate_manifest_sha256": _sha256(args.candidate),
            "prepare": str(args.prepare.relative_to(ROOT)),
            "prepare_sha256": _sha256(args.prepare),
            "reachability": str(args.reachability.relative_to(ROOT)),
            "reachability_sha256": _sha256(args.reachability),
            "cg0": str(args.cg0.relative_to(ROOT)),
            "cg0_sha256": _sha256(args.cg0),
            "cg1b": str(args.cg1b.relative_to(ROOT)),
            "cg1b_sha256": _sha256(args.cg1b),
            "edge_artifact": str(args.edge_artifact.relative_to(ROOT)),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
            "edge_record": edge_record,
        },
        "execution": {
            "device": "cpu",
            "frozen_core_feature_extraction": False,
            "existing_frozen_core_cache_reused": True,
            "new_feature_cache_created": False,
            "model_training": True,
            "trained_parameters": HEAD_PARAMETERS,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "graph_replay": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    heads = {
        "schema": "celltrack-alpha.cgl-tail-rank-heads.v1",
        "candidate_id": candidate["candidate_id"],
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "fold_states": head_states,
    }
    return report, heads


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    parser.add_argument("--cg0", type=Path, default=DEFAULT_CG0)
    parser.add_argument("--cg1b", type=Path, default=DEFAULT_CG1B)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.heads = args.heads.resolve()
    for name in ("output", "heads"):
        try:
            getattr(args, name).relative_to(ROOT)
        except ValueError as exc:
            raise CGLFastFailError(
                f"{name} artifact must remain inside the workspace: "
                f"{getattr(args, name)}"
            ) from exc
    if args.output.exists() or args.heads.exists():
        raise CGLFastFailError("refusing to overwrite CGL output artifacts")
    report, heads = analyze(args)
    args.heads.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads, args.heads)
    report["head"]["artifact"] = str(args.heads.relative_to(ROOT))
    report["head"]["artifact_sha256"] = _sha256(args.heads)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "heads": str(args.heads),
                "verdict": report["verdict"],
                "44b6_retention": report["results"]["ranking_at_fp_budget"][
                    "44b6"
                ]["correct_retention"],
                "6bba_retention": report["results"]["ranking_at_fp_budget"][
                    "6bba"
                ]["correct_retention"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

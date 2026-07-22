#!/usr/bin/env python3
"""Fit the compact edge residual only on frozen alpha conflict decisions.

The released detector/head/readout and graph machinery are inference-only.
Supervision consists exclusively of identity-supported, cap-8-reachable true
continuation misses paired against the actual division-disabled alpha edge that
consumed the same source or target.  No random negatives or dense pair object
are constructed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import edgefit
import run_bh
from celltrack_alpha import bh_graph
from celltrack_alpha.bh_edge import (
    ExposureMetadata,
    SparseEdgeConfig,
    SparseEdgeScorer,
    load_edge_artifact,
    save_edge_artifact,
)
from celltrack_alpha.bh_model import (
    ArtifactBundle,
    CacheIdentity,
    VideoInference,
    load_sparse_cache,
    validate_artifact_bundle,
)
from celltrack_alpha.edge_data import (
    ConflictPairExamples,
    EdgeDataError,
    SOURCE_CONFLICT,
    TARGET_CONFLICT,
    conflict_examples_from_video,
    match_detections_to_gt,
)
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.temporal_pretrain import (
    TemporalPretrainError,
    load_alpha_pseudo_link_artifact,
)
from celltrack_alpha.v45_fast import (
    load_fast_readout,
)


SCHEMA = "celltrack-alpha.conflict-contrastive-edgefit.v1"
OBJECTIVE = "identity-supported-reachable-alpha-miss-vs-selected-conflict-softplus"


class ConflictFitError(ValueError):
    """A conflict-only residual fit violated its immutable contract."""


class _ReleasedPairCheckpoint(nn.Module):
    """Only the released tensors consumed by sparse candidate scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 128)
        self.norm_in = nn.LayerNorm(128)
        self.norm_out = nn.LayerNorm(128)
        self.pair_mlp = nn.Sequential(
            nn.Linear(259, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )


def load_released_pair_checkpoint(bundle: ArtifactBundle) -> _ReleasedPairCheckpoint:
    """Strictly project the hash-pinned released state to its sparse readout."""

    if not isinstance(bundle, ArtifactBundle):
        raise ConflictFitError("released pair checkpoint requires a validated artifact bundle")
    try:
        state = torch.load(bundle.weight_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConflictFitError("released checkpoint weights-only load failed") from exc
    if not isinstance(state, Mapping):
        raise ConflictFitError("released checkpoint state is not a tensor mapping")
    prefixes = ("transformer.proj.", "transformer.norm_in.", "transformer.norm_out.", "transformer.pair_mlp.")
    projected = {
        name.removeprefix("transformer."): value
        for name, value in state.items()
        if isinstance(name, str) and name.startswith(prefixes)
    }
    checkpoint = _ReleasedPairCheckpoint()
    try:
        incompatible = checkpoint.load_state_dict(projected, strict=True)
    except RuntimeError as exc:
        raise ConflictFitError("released sparse pair tensor schema mismatch") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ConflictFitError("released sparse pair checkpoint is incomplete")
    if any(
        value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item())
        for value in checkpoint.state_dict().values()
    ):
        raise ConflictFitError("released sparse pair checkpoint contains invalid tensors")
    checkpoint.eval()
    return checkpoint


def select_parent_sources(
    parent_sources: Sequence[Mapping[str, str]], requested: Sequence[str]
) -> tuple[dict[str, str], ...]:
    """Resolve an explicit subset without permitting a foreign source."""

    wanted = tuple(str(item).strip() for item in requested)
    if len(wanted) < 2 or any(not item for item in wanted) or len(set(wanted)) != len(wanted):
        raise ConflictFitError("source-videos must name at least two unique parent fit videos")
    by_id = {str(row["video_id"]): dict(row) for row in parent_sources}
    normalized = tuple(item if item.startswith("train:") else f"train:{item}" for item in wanted)
    if not set(normalized).issubset(by_id):
        raise ConflictFitError("requested source is not declared by the parent fit checkpoint")
    return tuple(by_id[video_id] for video_id in normalized)


def select_registry_sources(
    registry: Mapping[str, Any], requested: Sequence[str]
) -> tuple[dict[str, str], ...]:
    """Resolve explicit train video/group pairs from the immutable registry."""

    wanted = tuple(str(item).strip() for item in requested)
    if len(wanted) < 2 or any(not item for item in wanted) or len(set(wanted)) != len(wanted):
        raise ConflictFitError("source-videos must name at least two unique registry videos")
    rows = registry.get("records")
    if not isinstance(rows, list):
        raise ConflictFitError("immutable registry has no source records")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("split") != "train":
            continue
        video_id, group_id, stem = row.get("video_id"), row.get("group_id"), row.get("dataset")
        if (
            not isinstance(video_id, str)
            or not isinstance(group_id, str)
            or not isinstance(stem, str)
            or video_id != f"train:{stem}"
            or video_id in by_id
        ):
            raise ConflictFitError("immutable registry train source identity is invalid")
        by_id[video_id] = {"video_id": video_id, "group_id": group_id, "stem": stem}
    normalized = tuple(item if item.startswith("train:") else f"train:{item}" for item in wanted)
    if not set(normalized).issubset(by_id):
        raise ConflictFitError("requested source is not a declared competition-train registry video")
    return tuple(by_id[video_id] for video_id in normalized)


def _combined_topology_sha256(parts: Sequence[ConflictPairExamples]) -> str:
    value = [part.candidate_topology_sha256 for part in parts]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def combine_conflicts(parts: Sequence[ConflictPairExamples]) -> ConflictPairExamples:
    """Concatenate bounded per-video COO conflict rows."""

    if not parts:
        raise ConflictFitError("no conflict example parts were supplied")
    nonempty = [part for part in parts if len(part)]

    def rows(name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if not nonempty:
            return np.empty(shape, dtype=dtype)
        return np.concatenate([np.asarray(getattr(part, name)) for part in nonempty])

    count = sum(len(part) for part in nonempty)
    return ConflictPairExamples(
        true_features=rows("true_features", (0, 16), np.float32),
        competitor_features=rows("competitor_features", (0, 16), np.float32),
        true_checkpoint_logits=rows("true_checkpoint_logits", (0,), np.float32),
        competitor_checkpoint_logits=rows("competitor_checkpoint_logits", (0,), np.float32),
        conflict_kind=rows("conflict_kind", (0,), np.int8),
        true_pairs=rows("true_pairs", (0, 2), np.int64).reshape(count, 2),
        competitor_pairs=rows("competitor_pairs", (0, 2), np.int64).reshape(count, 2),
        candidate_count=sum(part.candidate_count for part in parts),
        reachable_miss_count=sum(part.reachable_miss_count for part in parts),
        source_conflict_count=sum(part.source_conflict_count for part in parts),
        target_conflict_count=sum(part.target_conflict_count for part in parts),
        skipped_division_truth=sum(part.skipped_division_truth for part in parts),
        skipped_without_candidate_conflict=sum(
            part.skipped_without_candidate_conflict for part in parts
        ),
        candidate_topology_sha256=_combined_topology_sha256(parts),
    )


ConflictTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def conflict_logits(
    scorer: SparseEdgeScorer, examples: ConflictTensors, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    true_features, competitor_features, true_base, competitor_base = examples
    true = true_base[rows] + scorer.calibrator(true_features[rows])
    competitor = competitor_base[rows] + scorer.calibrator(competitor_features[rows])
    if true.shape != competitor.shape or not torch.isfinite(true).all() or not torch.isfinite(competitor).all():
        raise ConflictFitError("conflict logits are invalid")
    return true, competitor


def conflict_loss(
    scorer: SparseEdgeScorer, examples: ConflictTensors, rows: torch.Tensor
) -> torch.Tensor:
    """Parameter-free pairwise logistic loss with true edge ranked first."""

    true, competitor = conflict_logits(scorer, examples, rows)
    return F.softplus(-(true - competitor)).mean()


def ranking_metrics(
    scorer: SparseEdgeScorer, examples: ConflictTensors, batch_size: int
) -> dict[str, float | int]:
    count = int(examples[0].shape[0])
    if count == 0:
        return {"pairs": 0, "loss": float("inf"), "strict_accuracy": 0.0, "mean_margin": float("-inf")}
    device = scorer.calibrator.output.weight.device
    losses: list[float] = []
    margins: list[np.ndarray] = []
    scorer.train(False)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            rows = torch.arange(start, min(start + batch_size, count), device=device)
            true, competitor = conflict_logits(scorer, examples, rows)
            losses.append(float(F.softplus(-(true - competitor)).mean().item()))
            margins.append((true - competitor).detach().cpu().numpy())
    margin = np.concatenate(margins)
    return {
        "pairs": count,
        "loss": float(np.mean(losses)),
        "strict_accuracy": float(np.mean(margin > 0.0)),
        "mean_margin": float(np.mean(margin)),
    }


def fit_conflicts(
    scorer: SparseEdgeScorer,
    train: ConflictPairExamples,
    validation: ConflictPairExamples,
    *,
    seed: int,
    epochs: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Optimize only the compact residual and retain best fit-only validation."""

    if len(train) < 2 or len(validation) < 1:
        raise ConflictFitError("conflict train/validation examples are insufficient")
    if any(type(value) is not int or value <= 0 for value in (epochs, max_steps, batch_size)):
        raise ConflictFitError("epochs, max-steps, and batch-size must be positive")
    if (
        not np.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not np.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ConflictFitError("optimizer values are invalid")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = scorer.calibrator.output.weight.device
    train_tensors = train.as_tensors(device)
    validation_tensors = validation.as_tensors(device)
    optimizer = torch.optim.AdamW(
        scorer.calibrator.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    initial_train = ranking_metrics(scorer, train_tensors, batch_size)
    initial_validation = ranking_metrics(scorer, validation_tensors, batch_size)
    best_loss = float(initial_validation["loss"])
    best = {name: value.detach().cpu().clone() for name, value in scorer.calibrator.state_dict().items()}
    history: list[dict[str, Any]] = []
    steps = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        scorer.train(True)
        order = torch.randperm(len(train), device=device)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            if steps >= max_steps:
                break
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = conflict_loss(scorer, train_tensors, rows)
            if not torch.isfinite(loss):
                raise ConflictFitError("conflict loss became nonfinite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                scorer.calibrator.parameters(), max_norm=5.0
            )
            if not torch.isfinite(gradient_norm):
                raise ConflictFitError("conflict gradient became nonfinite")
            optimizer.step()
            losses.append(float(loss.item()))
            steps += 1
        validation_metrics = ranking_metrics(scorer, validation_tensors, batch_size)
        improved = float(validation_metrics["loss"]) < best_loss - 1.0e-7
        history.append(
            {
                "epoch": epoch,
                "steps": steps,
                "train_loss": float(np.mean(losses)) if losses else None,
                "validation": validation_metrics,
                "improved": improved,
            }
        )
        if improved:
            best_loss = float(validation_metrics["loss"])
            best = {
                name: value.detach().cpu().clone()
                for name, value in scorer.calibrator.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if steps >= max_steps or stale >= 8:
            break
    scorer.calibrator.load_state_dict(best, strict=True)
    scorer.train(False)
    final_train = ranking_metrics(scorer, train_tensors, batch_size)
    final_validation = ranking_metrics(scorer, validation_tensors, batch_size)
    mechanism_passed = (
        steps > 0
        and np.isfinite(float(final_train["loss"]))
        and np.isfinite(float(final_validation["loss"]))
        and float(final_train["loss"]) < float(initial_train["loss"])
        and float(final_validation["loss"]) < float(initial_validation["loss"])
    )
    return {
        "objective": OBJECTIVE,
        "steps": steps,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "best_validation_loss": best_loss,
        "history": history,
        "mechanism_passed": mechanism_passed,
    }


def _selected_continuations(video: Any) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Replay the frozen graph with division disabled and retain raw-node edges."""

    graph_nodes = run_bh.detections_to_graph_nodes(video.detections, bh_graph.GraphNode)
    graph = bh_graph.calibrate_v45_graph(
        graph_nodes,
        topology_mode="legacy",
        division_enabled=False,
    )
    raw_ids = {
        int(node_id)
        for detection in video.detections
        for node_id in detection.node_ids
    }
    selected = {
        (int(edge.source_id), int(edge.target_id))
        for edge in graph.edges
        if int(edge.source_id) in raw_ids and int(edge.target_id) in raw_ids
    }
    outdegree: dict[int, int] = {}
    indegree: dict[int, int] = {}
    for source_id, target_id in selected:
        outdegree[source_id] = outdegree.get(source_id, 0) + 1
        indegree[target_id] = indegree.get(target_id, 0) + 1
    if max(outdegree.values(), default=0) > 1 or max(indegree.values(), default=0) > 1:
        raise ConflictFitError("division-disabled alpha replay produced a branch or merge")
    if graph.stats.get("division_added") != 0 or graph.provenance.get("division_enabled") is not False:
        raise ConflictFitError("division-disabled alpha replay attestation failed")
    return selected, {
        "input_nodes": int(graph.stats["input_node_count"]),
        "output_nodes": int(graph.stats["output_node_count"]),
        "output_edges": int(graph.stats["output_edge_count"]),
        "raw_selected_continuations": len(selected),
        "division_added": int(graph.stats["division_added"]),
        "full_frame_dense_matrix": bool(graph.provenance["full_frame_dense_matrix"]),
        "topology_mode": str(graph.provenance["topology_mode"]),
        "image_refinement": str(graph.provenance["image_refinement"]),
    }


def _load_alpha_cache(
    artifact_root: Path,
    source: Mapping[str, str],
    exact_identity_sha256: str,
    expected_cache_identity: CacheIdentity,
) -> tuple[VideoInference, dict[str, Any]]:
    """Restore one fully attested immutable released-detector cache."""

    try:
        artifact = load_alpha_pseudo_link_artifact(
            artifact_root,
            group_id=str(source["group_id"]),
            video_id=str(source["video_id"]),
            stem=str(source["stem"]),
            expected_exact_identity_sha256=exact_identity_sha256,
            min_migration_um=0.1,
            min_link_confidence=0.05,
            max_links_per_window=512,
            max_divisions_per_window=16,
        )
    except TemporalPretrainError as exc:
        raise ConflictFitError(f"immutable alpha cache rejected: {source['video_id']}: {exc}") from exc
    slug = f"train__{source['stem']}"
    meta_path = artifact_root / "cache" / f"{slug}.json"
    status_path = artifact_root / "status" / f"{slug}.json"
    meta = edgefit._load_json(
        meta_path,
        expected_sha256=edgefit._sha_file(meta_path),
        label=f"{source['video_id']} alpha cache metadata",
    )
    status = edgefit._load_json(
        status_path,
        expected_sha256=edgefit._sha_file(status_path),
        label=f"{source['video_id']} alpha status",
    )
    model_identity = meta.get("contract", {}).get("model")
    if model_identity != asdict(expected_cache_identity):
        raise ConflictFitError("alpha cache does not match the loaded released detector identity")
    gate = status.get("run_contract", {}).get("graph_config", {}).get("candidate_gate")
    if gate != {
        "algorithm": "physical-cKDTree",
        "neighbor_cap": 8,
        "radius_um": 10.0,
        "scale_zyx": [1.625, 0.40625, 0.40625],
    }:
        raise ConflictFitError("alpha cache graph identity is not the frozen 10um cap-8 KD-tree")
    cache_path = artifact_root / "cache" / f"{slug}.bhc"
    cache = load_sparse_cache(cache_path, expected_identity=expected_cache_identity)
    if cache.edges:
        raise ConflictFitError("alpha cache unexpectedly persists learned edge rows")
    video = VideoInference(
        cache.detections,
        cache.contexts,
        pair_batch_size=cache.identity.pair_batch_size,
        view_batch_size=cache.identity.view_batch_size,
    )
    provenance = {
        **artifact.provenance(),
        "meta_sha256": edgefit._sha_file(meta_path),
        "cache_identity": asdict(cache.identity),
        "candidate_gate": gate,
    }
    return video, provenance


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-videos", nargs="+", required=True)
    parser.add_argument(
        "--alpha-artifact-root",
        type=Path,
        default=ROOT / "kaggle_kernel/bh9o7/bh9out",
    )
    parser.add_argument("--fold", choices=(0, 1, 2), type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--support-root", type=Path, default=ROOT / "external/Tracking_Support")
    parser.add_argument("--checkpoint", type=Path, default=edgefit.DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=edgefit.DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--metrics", type=Path, default=edgefit.DEFAULT_METRICS)
    parser.add_argument("--metrics-sha256", default=edgefit.DEFAULT_METRICS_SHA256)
    parser.add_argument("--registry", type=Path, default=edgefit.DEFAULT_REGISTRY)
    parser.add_argument("--split", type=Path, default=edgefit.DEFAULT_SPLIT)
    parser.add_argument("--provenance", type=Path, default=edgefit.DEFAULT_PROVENANCE)
    parser.add_argument("--val-groups", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise ConflictFitError("out-dir must not already exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ConflictFitError("CUDA was requested but is unavailable")
    return args


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ConflictFitError(f"refusing to overwrite output: {path}")
    run_bh.atomic_create_json(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    split_sha = edgefit._sha_file(args.split)
    provenance_sha = edgefit._sha_file(args.provenance)
    split = edgefit._load_json(args.split, expected_sha256=split_sha, label="split")
    registry, registry_records, registry_sha = edgefit.load_edge_registry(args.registry)
    provenance = edgefit._load_json(
        args.provenance, expected_sha256=provenance_sha, label="provenance"
    )
    if (
        provenance.get("grouped_split_sha256") != split_sha
        or provenance.get("related_manifest_hashes", {}).get("dataset_registry_v1.json")
        != registry_sha
    ):
        raise ConflictFitError("fit provenance does not bind the requested split and registry")
    metrics = edgefit._load_json(
        args.metrics, expected_sha256=args.metrics_sha256, label="parent metrics"
    )
    parent_identity = edgefit.verify_parent_manifest_binding(
        metrics,
        registry_sha256=registry_sha,
        split_sha256=split_sha,
        provenance_sha256=provenance_sha,
    )
    bundle = validate_artifact_bundle(args.support_root)
    expected_core_state = metrics.get("core_state_sha256")
    auxiliary, ident = load_fast_readout(
        args.checkpoint,
        bundle,
        expected_sha256=args.checkpoint_sha256,
        target_sha256=TARGET_CONTRACT_SHA256,
        expected_core_state_sha256=str(expected_core_state),
    )
    if metrics.get("checkpoint_identity") != ident.as_dict():
        raise ConflictFitError("parent metrics and compact checkpoint identities differ")
    all_parent_sources = edgefit._verify_parent(
        metrics, args.checkpoint_sha256, ident, args.fold
    )
    selected_sources = select_registry_sources(registry, args.source_videos)
    if registry.get("artifact") == edgefit.REGISTRY_PROJECTION_ARTIFACT:
        if registry.get("parent_projection_sha256") != parent_identity.get("record_projection_sha256"):
            raise ConflictFitError("registry projection is not bound to the parent checkpoint")
        if set(row["video_id"] for row in selected_sources) != {
            str(record.video_id) for record in registry_records
        }:
            raise ConflictFitError("a projected registry cannot be silently subset for this fit")
    scope = edgefit.build_scope(
        registry_records,
        registry,
        split,
        selected_sources,
        fold=args.fold,
        split_sha256=split_sha,
        registry_sha256=registry_sha,
        provenance_sha256=provenance_sha,
        val_groups=args.val_groups,
    )
    expected_cache_identity = CacheIdentity.from_artifact(
        bundle, pair_batch_size=1, view_batch_size=1
    )
    pair_checkpoint = load_released_pair_checkpoint(bundle).to(args.device)
    scorer = SparseEdgeScorer(
        pair_checkpoint,
        config=SparseEdgeConfig(radius_um=10.0, neighbor_cap=8, division_enabled=False),
        pretrained_weight_sha256=bundle.hashes.weight_sha256,
        auxiliary=auxiliary,
    )
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "objective": OBJECTIVE,
        "parent": {
            "checkpoint_sha256": args.checkpoint_sha256,
            "metrics_sha256": args.metrics_sha256,
            "identity": ident.as_dict(),
            "declared_fit_sources": list(all_parent_sources),
        },
        "scope": {
            "fold": scope.fold,
            "train_video_ids": list(scope.train_ids),
            "validation_video_ids": list(scope.val_ids),
            "train_group_ids": list(scope.train_groups),
            "validation_group_ids": list(scope.val_groups),
            "split_sha256": scope.split_sha256,
            "registry_sha256": scope.registry_sha256,
            "provenance_sha256": scope.provenance_sha256,
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "model": {
            "trainable": "compact-residual-only",
            "trainable_parameter_count": scorer.trainable_parameter_count,
            "frozen_detector": True,
            "detector_execution": "immutable-released-exact-sequential-D4-cache",
            "frozen_node_generation": True,
            "candidate_graph": "physical-cKDTree-10um-cap8-COO",
            "constrained_solver": "locked-v45-legacy-sparse",
            "division_enabled": False,
            "dense_nxn_association": False,
            "architecture_sha256": scorer.architecture_sha256,
            "config": scorer.config.as_dict(),
        },
        "optimization": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "validation_group_count": args.val_groups,
            "device": args.device,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.dry_run:
        _atomic_json(args.out_dir / "report.json", base)
        print(json.dumps(base, ensure_ascii=True, sort_keys=True))
        return 0

    jobs = {job.identity: job for job in run_bh.discover_jobs(args.input, scope.records)}
    examples_by_id: dict[str, ConflictPairExamples] = {}
    video_reports: list[dict[str, Any]] = []
    frozen_before = scorer.frozen_subset_sha256
    calibrator_before = {
        name: value.detach().cpu().clone()
        for name, value in scorer.calibrator.state_dict().items()
    }
    started = time.perf_counter()
    for video_id in (*scope.train_ids, *scope.val_ids):
        job = jobs.get(video_id)
        if job is None:
            raise ConflictFitError(f"fit source was not discovered: {video_id}")
        handle = run_bh.open_video(job)
        try:
            scale = tuple(float(value) for value in run_bh.read_scale_attrs(handle.attrs))
            if not np.allclose(scale, scorer.config.scale_zyx, rtol=0.0, atol=1.0e-12):
                raise ConflictFitError("video physical scale differs from the residual candidate contract")
            source = next(row for row in selected_sources if row["video_id"] == video_id)
            cache_video, cache_provenance = _load_alpha_cache(
                args.alpha_artifact_root,
                source,
                str(job.record.exact_identity_sha256),
                expected_cache_identity,
            )
            gt = edgefit._read_gt(job, handle)
            matches, gt_edges = match_detections_to_gt(
                cache_video.detections, gt, scale_zyx=scale
            )
            selected_edges, graph_report = _selected_continuations(cache_video)
            examples = conflict_examples_from_video(
                scorer, cache_video, matches, gt_edges, selected_edges
            )
            examples_by_id[video_id] = examples
            video_reports.append(
                {
                    "video_id": video_id,
                    "group_id": next(
                        row["group_id"]
                        for row in selected_sources
                        if row["video_id"] == video_id
                    ),
                    "exact_identity_sha256": str(job.record.exact_identity_sha256),
                    "detections": len(matches),
                    "matched_detections": sum(value is not None for value in matches.values()),
                    "candidate_rows": examples.candidate_count,
                    "candidate_topology_sha256": examples.candidate_topology_sha256,
                    "reachable_misses": examples.reachable_miss_count,
                    "conflict_pairs": len(examples),
                    "source_conflicts": examples.source_conflict_count,
                    "target_conflicts": examples.target_conflict_count,
                    "skipped_division_truth": examples.skipped_division_truth,
                    "skipped_without_candidate_conflict": examples.skipped_without_candidate_conflict,
                    "graph": graph_report,
                    "alpha_cache": cache_provenance,
                }
            )
        except (EdgeDataError, edgefit.EdgeFitError) as exc:
            raise ConflictFitError(str(exc)) from exc
        finally:
            handle.close()
    train = combine_conflicts([examples_by_id[video_id] for video_id in scope.train_ids])
    validation = combine_conflicts(
        [examples_by_id[video_id] for video_id in scope.val_ids]
    )
    result = fit_conflicts(
        scorer,
        train,
        validation,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    frozen_after = scorer.frozen_subset_sha256
    if frozen_after != frozen_before:
        raise ConflictFitError("frozen released scorer state drifted")
    changed = sorted(
        name
        for name, value in scorer.calibrator.state_dict().items()
        if not torch.equal(value.detach().cpu(), calibrator_before[name])
    )
    if not changed or any(not name.startswith(("input.", "output.", "dummy_residual")) for name in changed):
        raise ConflictFitError("compact residual update scope is invalid")
    exposure = ExposureMetadata(
        fold=args.fold,
        source_groups=tuple(sorted((*scope.train_groups, *scope.val_groups))),
        source_video_ids=tuple(sorted((*scope.train_ids, *scope.val_ids))),
        split_manifest_sha256=scope.split_sha256,
        provenance_manifest_sha256=scope.provenance_sha256,
        audit_exposed=False,
        test_exposed=False,
    )
    artifact_path = args.out_dir / "edge.json"
    artifact_identity = save_edge_artifact(artifact_path, scorer, exposure)
    restored, restored_identity = load_edge_artifact(
        artifact_path,
        scorer,
        expected_pretrained_weight_sha256=bundle.hashes.weight_sha256,
        auxiliary=auxiliary,
    )
    restore_passed = (
        restored_identity == artifact_identity
        and restored.frozen_subset_sha256 == scorer.frozen_subset_sha256
        and all(
            torch.equal(restored.calibrator.state_dict()[name], value)
            for name, value in scorer.calibrator.state_dict().items()
        )
    )
    if not restore_passed:
        raise ConflictFitError("edge artifact checkpoint restore did not preserve residual state")
    report = {
        **base,
        "status": "completed",
        "mechanism_passed": bool(result["mechanism_passed"] and restore_passed),
        "artifact": {
            "path": str(artifact_path),
            "sha256": edgefit._sha_file(artifact_path),
            "identity": asdict(artifact_identity),
            "restore_passed": restore_passed,
        },
        "training": result,
        "integrity": {
            "frozen_subset_sha256_before": frozen_before,
            "frozen_subset_sha256_after": frozen_after,
            "changed_residual_tensors": changed,
            "candidate_topology_frozen": True,
            "division_enabled": False,
            "dense_nxn_association": False,
        },
        "examples": {
            "train": {
                "videos": len(scope.train_ids),
                "pairs": len(train),
                "reachable_misses": train.reachable_miss_count,
                "source_conflicts": train.source_conflict_count,
                "target_conflicts": train.target_conflict_count,
                "candidate_topology_sha256": train.candidate_topology_sha256,
            },
            "validation": {
                "videos": len(scope.val_ids),
                "pairs": len(validation),
                "reachable_misses": validation.reachable_miss_count,
                "source_conflicts": validation.source_conflict_count,
                "target_conflicts": validation.target_conflict_count,
                "candidate_topology_sha256": validation.candidate_topology_sha256,
            },
            "videos": video_reports,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.out_dir / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "mechanism_passed": report["mechanism_passed"],
                "artifact": report["artifact"],
                "training": report["training"],
                "examples": report["examples"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

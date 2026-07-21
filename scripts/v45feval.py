#!/usr/bin/env python3
"""Evaluate a provenance-bound one-pass V45 checkpoint on held-out videos.

This is deliberately separate from ``run_bh.py``.  The latter owns the
eight-view released V45 cache contract, whereas this evaluator owns a
one-view compact-head execution identity and never reads or writes that cache.
It can retain the established sparse graph solver as an explicitly labelled
rule-based fallback, or load a hash-pinned, fit-only sparse edge calibrator
against the exact checkpoint-bound motion readout.  The explicit ``a0`` mode
instead reuses the released checkpoint's frozen attention only inside bounded
components of the same physical sparse candidate gate; it performs no fit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_bh
from celltrack_alpha import bh_graph
from celltrack_alpha.bh_edge import (
    A0CheckpointScorer,
    A0PairProbabilities,
    EdgeArtifactIdentity,
    ExposureMetadata,
    SparseEdgeConfig,
    SparseEdgeScorer,
    gate_sparse_candidates,
    learned_edge_probability_map,
    load_edge_artifact,
)
from celltrack_alpha.bh_model import load_v45_model, v45_config_sha
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.v45_fast import FastIdent, FrozenV45Fast, fast_readout, infer_fast, load_fast


SCHEMA = "celltrack-alpha.v45-fast-eval.v2"
DEFAULT_CHECKPOINT = ROOT / "reports/g02/v45f4.pt"
DEFAULT_METRICS = ROOT / "reports/g02/v45f4met.json"
DEFAULT_CHECKPOINT_SHA256 = "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473"
DEFAULT_METRICS_SHA256 = "1d67a099377dde39c04985982be1d427b3b4d3c7abcb6b4d0fa07d79c944b4b5"
DEFAULT_PROVENANCE = ROOT / "reports/g02/fit_provenance_v1.json"


class V45FastEval(ValueError):
    """Raised when a compact checkpoint evaluation is not provenance safe."""


@dataclass(frozen=True)
class EvalScope:
    """Immutable, whole-video held-out selection for one trained fold."""

    fold: int
    records: tuple[Any, ...]
    audit_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    group_by_id: Mapping[str, str]
    split_sha256: str
    registry_sha256: str


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _load_json(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_file() or _sha_file(path) != expected_sha256:
        raise V45FastEval(f"{label} SHA-256 does not match the pinned value")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V45FastEval(f"{label} is not readable canonical JSON") from exc
    if not isinstance(value, Mapping):
        raise V45FastEval(f"{label} must be a JSON object")
    return value


def registry_groups(registry: Mapping[str, Any], records: Sequence[Any]) -> dict[str, str]:
    """Restore the immutable grouping field omitted by ``RegistryRecord``."""

    if registry.get("artifact") != "dataset_registry_v1":
        raise V45FastEval("registry is not the frozen dataset registry")
    rows = registry.get("records")
    if not isinstance(rows, list):
        raise V45FastEval("registry does not contain record mappings")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise V45FastEval("registry contains a non-object record")
        identity, group = row.get("video_id"), row.get("group_id")
        if not isinstance(identity, str) or not identity or not isinstance(group, str) or not group:
            raise V45FastEval("registry row lacks a stable video/group identity")
        if identity in result:
            raise V45FastEval("registry contains duplicate video identities")
        result[identity] = group
    identities = {str(record.video_id) for record in records}
    if identities != set(result):
        raise V45FastEval("runtime registry records disagree with group registry")
    return result


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise V45FastEval(f"{label} must be a nonempty string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise V45FastEval(f"{label} contains duplicate identities")
    return result


def select_heldout(
    records: Sequence[Any],
    split: Mapping[str, Any],
    *,
    fold: int,
    registry_sha256: str,
    split_sha256: str,
    group_by_id: Mapping[str, str] | None = None,
    requested: Sequence[str] | None = None,
    limit: int | None = None,
) -> EvalScope:
    """Resolve only whole held-out train videos and reject audit/fit leakage."""

    if type(fold) is not int or fold not in {0, 1, 2}:
        raise V45FastEval("fold must be one of 0, 1, or 2")
    if split.get("artifact") != "grouped_3fold_audit20_v1":
        raise V45FastEval("split is not the frozen grouped audit manifest")
    if split.get("dataset_registry_sha256") != registry_sha256:
        raise V45FastEval("split does not bind the exact registry")
    folds = split.get("folds")
    audit = split.get("audit")
    if not isinstance(folds, Mapping) or not isinstance(audit, Mapping):
        raise V45FastEval("split folds or audit object is invalid")
    fold_data = folds.get(str(fold))
    if not isinstance(fold_data, Mapping):
        raise V45FastEval("requested fold is absent from frozen split")
    held_ids = _strings(fold_data.get("held_out_video_ids"), label="held_out_video_ids")
    held_groups = _strings(fold_data.get("held_out_group_ids"), label="held_out_group_ids")
    fit_groups = _strings(fold_data.get("fit_group_ids"), label="fit_group_ids")
    audit_ids = _strings(audit.get("video_ids"), label="audit.video_ids")
    audit_groups = _strings(audit.get("group_ids"), label="audit.group_ids")
    if set(held_ids) & set(audit_ids) or set(held_groups) & set(audit_groups):
        raise V45FastEval("frozen held-out selection overlaps untouched audit")
    if set(held_groups) & set(fit_groups):
        raise V45FastEval("frozen fold has held-out/fit group overlap")
    by_id = {str(record.video_id): record for record in records}
    if len(by_id) != len(records):
        raise V45FastEval("registry records do not have unique video identities")
    missing = sorted(set(held_ids) - set(by_id))
    if missing:
        raise V45FastEval(f"held-out identities are absent from registry: {missing[:3]}")
    selected_ids = held_ids
    if requested is not None:
        chosen = tuple(str(item).strip() for item in requested)
        if not chosen or any(not item for item in chosen) or len(set(chosen)) != len(chosen):
            raise V45FastEval("requested datasets must be nonempty and unique")
        dataset_to_id = {str(record.dataset): str(record.video_id) for record in by_id.values()}
        normalized = tuple(item if item.startswith("train:") else dataset_to_id.get(item, "") for item in chosen)
        if any(not item for item in normalized) or not set(normalized).issubset(held_ids):
            raise V45FastEval("requested datasets must be held-out identities for this fold")
        selected_ids = tuple(sorted(normalized))
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise V45FastEval("limit must be a positive integer")
        selected_ids = tuple(sorted(selected_ids))[:limit]
    chosen_records = tuple(by_id[item] for item in selected_ids)
    if group_by_id is None:
        group_by_id = {str(record.video_id): str(getattr(record, "group_id", "")) for record in records}
    if set(group_by_id) != set(by_id) or any(not isinstance(group, str) or not group for group in group_by_id.values()):
        raise V45FastEval("group index does not cover the runtime registry")
    for record in chosen_records:
        group = group_by_id[str(record.video_id)]
        if str(record.split) != "train" or group not in held_groups:
            raise V45FastEval("held-out selection contains a nontrain or foreign group")
        if str(record.video_id) in audit_ids or group in audit_groups:
            raise V45FastEval("untouched audit leaked into evaluation selection")
    return EvalScope(
        fold=fold,
        records=tuple(sorted(chosen_records, key=lambda item: str(item.video_id))),
        audit_ids=tuple(sorted(audit_ids)),
        held_out_ids=tuple(sorted(held_ids)),
        group_by_id=dict(group_by_id),
        split_sha256=split_sha256,
        registry_sha256=registry_sha256,
    )


def verify_training(metrics: Mapping[str, Any], checkpoint_sha256: str, ident: FastIdent, *, fold: int) -> tuple[dict[str, Any], ...]:
    """Bind the measured fit-only run to the restored compact checkpoint."""

    if metrics.get("status") != "completed" or metrics.get("checkpoint_sha256") != checkpoint_sha256:
        raise V45FastEval("training metrics do not attest the completed checkpoint")
    if metrics.get("completed_steps") != ident.steps or ident.division_enabled:
        raise V45FastEval("checkpoint step/division identity disagrees with training metrics")
    train_id = metrics.get("identity")
    if not isinstance(train_id, Mapping) or train_id.get("fold") != fold:
        raise V45FastEval("checkpoint does not belong to the requested evaluation fold")
    if train_id.get("target_contract_sha256") != ident.target_contract_sha256:
        raise V45FastEval("training target contract disagrees with checkpoint")
    if hashlib.sha256(_canonical(dict(train_id))).hexdigest() != ident.train_sha256:
        raise V45FastEval("training identity hash disagrees with checkpoint")
    selected = train_id.get("selected_sources")
    if not isinstance(selected, list) or not selected:
        raise V45FastEval("training metrics lack selected fit-only sources")
    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for row in selected:
        if not isinstance(row, Mapping) or set(row) != {"group_id", "stem", "video_id"}:
            raise V45FastEval("training source schema is invalid")
        source = {key: str(row[key]) for key in ("group_id", "stem", "video_id")}
        if not all(source.values()) or source["video_id"] != f"train:{source['stem']}":
            raise V45FastEval("training source identity is invalid")
        if source["video_id"] in seen_ids or source["group_id"] in seen_groups:
            raise V45FastEval("training sources are not group unique")
        seen_ids.add(source["video_id"])
        seen_groups.add(source["group_id"])
        sources.append(source)
    return tuple(sources)


def verify_parent_manifests(
    metrics: Mapping[str, Any], *, registry_sha256: str, split_sha256: str, provenance_sha256: str
) -> Mapping[str, Any]:
    """Bind restored parent metrics to the immutable evaluator manifests."""
    identity = metrics.get("identity")
    if not isinstance(identity, Mapping):
        raise V45FastEval("training metrics do not contain a training identity")
    expected = {
        "registry_sha256": registry_sha256,
        "split_sha256": split_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise V45FastEval("training identity does not bind the evaluator manifests")
    return identity


def verify_edge_artifact_scope(
    identity: EdgeArtifactIdentity,
    sources: Sequence[Mapping[str, Any]],
    scope: EvalScope,
    split: Mapping[str, Any],
    *,
    provenance_sha256: str,
) -> None:
    """Require a trained residual to expose only registry-bound fold-fit sources."""
    exposure = identity.exposure
    if exposure.fold != scope.fold:
        raise V45FastEval("edge artifact belongs to another fold")
    if exposure.audit_exposed or exposure.test_exposed:
        raise V45FastEval("edge artifact was exposed to audit or test data")
    if not exposure.source_video_ids or not exposure.source_groups:
        raise V45FastEval("edge artifact does not declare fit-only source exposure")
    if exposure.split_manifest_sha256 != scope.split_sha256 or exposure.provenance_manifest_sha256 != provenance_sha256:
        raise V45FastEval("edge artifact exposure does not bind evaluator manifests")
    # The compact parent and residual may use different subsets of the same
    # immutable fit fold. Verify the parent pairs independently, then validate
    # every residual exposure pair directly against the frozen registry/split.
    if any(
        scope.group_by_id.get(str(item["video_id"])) != str(item["group_id"])
        for item in sources
    ):
        raise V45FastEval("edge artifact parent source groups disagree with the frozen registry")
    exposed_registry_groups = {
        scope.group_by_id.get(video_id) for video_id in exposure.source_video_ids
    }
    if None in exposed_registry_groups or exposed_registry_groups != set(exposure.source_groups):
        raise V45FastEval("edge artifact exposed video/group pairs disagree with the frozen registry")
    folds = split.get("folds")
    audit = split.get("audit")
    if not isinstance(folds, Mapping) or not isinstance(audit, Mapping):
        raise V45FastEval("frozen split structure is invalid for edge artifact validation")
    fold_data = folds.get(str(scope.fold))
    if not isinstance(fold_data, Mapping):
        raise V45FastEval("frozen split lacks edge artifact fold")
    fit_groups = set(_strings(fold_data.get("fit_group_ids"), label="fit_group_ids"))
    audit_groups = set(_strings(audit.get("group_ids"), label="audit.group_ids"))
    held_groups = {scope.group_by_id[video_id] for video_id in scope.held_out_ids}
    if (
        not set(exposure.source_groups).issubset(fit_groups)
        or set(exposure.source_groups) & (held_groups | audit_groups)
        or set(exposure.source_video_ids) & (set(scope.held_out_ids) | set(scope.audit_ids))
    ):
        raise V45FastEval("edge artifact source exposure is not fit-only for the requested fold")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise V45FastEval(f"refusing to overwrite existing report: {path}")
    run_bh.atomic_create_json(path, value)


def sparse_probs(
    video: Any, scorer: SparseEdgeScorer
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[str, int]]:
    """Score only KD-tree-gated adjacent pairs from one fast-video record."""

    if not isinstance(scorer, SparseEdgeScorer):
        raise V45FastEval("sparse association requires a SparseEdgeScorer")
    detections = video.detections
    contexts = video.contexts
    by_frame = {int(detection.frame_index): detection for detection in detections}
    if len(by_frame) != len(detections):
        raise V45FastEval("fast video detections do not have unique frame identities")
    if not contexts:
        raise V45FastEval("fast video contains no sparse association contexts")
    for context in contexts:
        if context.source_frame not in by_frame or context.target_frame not in by_frame:
            raise V45FastEval("fast video context references a missing detection frame")
    pairs = tuple(
        scorer.score_pair(context, by_frame[context.source_frame], by_frame[context.target_frame])
        for context in contexts
    )
    probabilities = learned_edge_probability_map(pairs)
    no_link: dict[int, float] = {}
    for pair in pairs:
        for source_id, probability in zip(
            pair.no_link_source_ids, pair.no_link_probabilities, strict=True
        ):
            source = int(source_id)
            if source in no_link:
                raise V45FastEval("duplicate sparse no-link probability across frame pairs")
            no_link[source] = float(probability)
    candidate_count = sum(int(pair.stats.candidate_count) for pair in pairs)
    if candidate_count != len(probabilities):
        raise V45FastEval("sparse association candidate mapping is incomplete")
    max_peak = max((int(pair.stats.peak_pair_chunk) for pair in pairs), default=0)
    return probabilities, no_link, {
        "pair_count": len(pairs),
        "candidate_count": candidate_count,
        "probability_count": len(probabilities),
        "no_link_probability_count": len(no_link),
        "peak_pair_chunk": max_peak,
        "neighbor_cap": scorer.config.neighbor_cap,
    }


def frozen_sparse_scorer(model: Any, bundle: Any) -> SparseEdgeScorer:
    """Expose the released candidate scorer without fitting its residual head."""

    scorer = SparseEdgeScorer.from_loaded_model(model, bundle)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    scorer.eval()
    if scorer.trainable_parameter_count != 0:
        raise V45FastEval("frozen sparse association unexpectedly has trainable parameters")
    return scorer


def freeze_fitted_scorer_for_evaluation(scorer: SparseEdgeScorer) -> int:
    """Freeze a restored fit artifact so evaluation builds no autograd graph."""

    if not isinstance(scorer, SparseEdgeScorer):
        raise V45FastEval("fit-only evaluator requires a SparseEdgeScorer")
    fitted_parameter_count = scorer.trainable_parameter_count
    if fitted_parameter_count <= 0:
        raise V45FastEval("restored fit-only scorer exposes no fitted parameters")
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    scorer.eval()
    if scorer.trainable_parameter_count != 0:
        raise V45FastEval("fit-only scorer remained trainable during evaluation")
    return fitted_parameter_count


def a0_probs(video: Any, scorer: A0CheckpointScorer) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Run frozen checkpoint attention only within cap-8 KD-tree components."""

    if not isinstance(scorer, A0CheckpointScorer):
        raise V45FastEval("A0 association requires an A0CheckpointScorer")
    detections = video.detections
    contexts = video.contexts
    by_frame = {int(detection.frame_index): detection for detection in detections}
    if len(by_frame) != len(detections) or not contexts:
        raise V45FastEval("A0 association requires complete fast-video contexts")
    gate_config = SparseEdgeConfig(
        radius_um=run_bh.A0_GATE_RADIUS_UM,
        neighbor_cap=run_bh.A0_GATE_NEIGHBOR_CAP,
    )
    if scorer.config.neighbor_cap != gate_config.neighbor_cap:
        raise V45FastEval("A0 scorer and sparse candidate gate neighbor caps disagree")
    probabilities: dict[tuple[int, int], float] = {}
    pair_stats: list[dict[str, Any]] = []
    for context in contexts:
        source = by_frame.get(int(context.source_frame))
        target = by_frame.get(int(context.target_frame))
        if source is None or target is None:
            raise V45FastEval("A0 context references a missing detection frame")
        candidates = gate_sparse_candidates(source, target, config=gate_config)
        scored = scorer.score_pair(context, source, target, candidates=candidates)
        if not isinstance(scored, A0PairProbabilities) or len(scored) != len(candidates):
            raise V45FastEval("A0 scorer did not preserve the original sparse candidate rows")
        for source_id, target_id, probability in zip(
            scored.source_ids, scored.target_ids, scored.probabilities, strict=True
        ):
            pair = (int(source_id), int(target_id))
            value = float(probability)
            if pair in probabilities or not 0.0 <= value <= 1.0:
                raise V45FastEval("A0 candidate probabilities are duplicate or invalid")
            probabilities[pair] = value
        pair_stats.append(asdict(scored.stats))
    if not pair_stats or sum(int(item["candidate_count"]) for item in pair_stats) != len(probabilities):
        raise V45FastEval("A0 association candidate accounting is incomplete")
    return probabilities, {
        "candidate_gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": gate_config.radius_um,
            "neighbor_cap": gate_config.neighbor_cap,
            "scale_zyx": list(gate_config.scale_zyx),
        },
        "pair_count": len(pair_stats),
        "component_count": sum(int(item["component_count"]) for item in pair_stats),
        "candidate_count": len(probabilities),
        "probability_count": len(probabilities),
        "normalization": "target-wise-source-softmax-no-dummy",
        "attention_component_count": sum(int(item["attention_component_count"]) for item in pair_stats),
        "fallback_component_count": sum(int(item["fallback_component_count"]) for item in pair_stats),
        "node_budget_fallback_component_count": sum(
            int(item["node_budget_fallback_component_count"]) for item in pair_stats
        ),
        "attention_budget_fallback_component_count": sum(
            int(item["attention_budget_fallback_component_count"]) for item in pair_stats
        ),
        "unsupported_fallback_component_count": sum(
            int(item["unsupported_fallback_component_count"]) for item in pair_stats
        ),
        "attention_candidate_count": sum(int(item["attention_candidate_count"]) for item in pair_stats),
        "fallback_candidate_count": sum(int(item["fallback_candidate_count"]) for item in pair_stats),
        "peak_attention_cells": max(int(item["peak_attention_cells"]) for item in pair_stats),
        "peak_pair_chunk": max(int(item["peak_pair_chunk"]) for item in pair_stats),
        "neighbor_cap": gate_config.neighbor_cap,
        "full_frame_dense_matrix": False,
    }


def evaluate_one(
    job: Any,
    *,
    model: FrozenV45Fast,
    expected_execution_sha256: str,
    device: str,
    out_dir: Path,
    scorer: SparseEdgeScorer | A0CheckpointScorer | None,
    association_mode: str,
    topology_mode: str,
    division_mode: str,
) -> dict[str, Any]:
    """Run fresh one-view inference with a declared sparse association mode."""

    handle = run_bh.open_video(job)
    started = time.perf_counter()
    try:
        q_low, q_high = run_bh.read_quantile_attrs(handle.attrs)
        scale = run_bh.read_scale_attrs(handle.attrs)
        frames = run_bh.read_all_frames(handle.array, int(job.record.image_shape[0]), None)
        fast_video = infer_fast(model, frames, q_low=q_low, q_high=q_high, device=device)
        if fast_video.execution_sha256 != expected_execution_sha256:
            raise V45FastEval("fast inference execution identity changed during evaluation")
        video = fast_video.video
        if video.d4_views != (0,) or video.view_batch_size != 1 or video.pair_batch_size != 1:
            raise V45FastEval("fast evaluator received non-one-pass inference provenance")
        graph_nodes = run_bh.detections_to_graph_nodes(video.detections, bh_graph.GraphNode)
        refiner = run_bh.LazyImageRefiner(handle.array, bh_graph.intensity_centroid_refine, deadline_epoch=None)
        association: dict[str, Any]
        if association_mode == "fallback":
            if scorer is not None:
                raise V45FastEval("fallback association must not receive a scorer")
            probabilities = None
            no_link_probabilities = None
            association = {
                "mode": "rule-v45-sparse-fallback",
                "learned_association_active": False,
            }
        elif association_mode in {"frozen", "sparse"}:
            if not isinstance(scorer, SparseEdgeScorer):
                raise V45FastEval("sparse association requires a SparseEdgeScorer")
            if association_mode == "frozen" and scorer.trainable_parameter_count != 0:
                raise V45FastEval("frozen sparse association has trainable parameters")
            probabilities, no_link_probabilities, association_stats = sparse_probs(video, scorer)
            association = {
                "mode": (
                    "frozen-v45-sparse-identity"
                    if association_mode == "frozen"
                    else f"fit-only-v45-sparse-{topology_mode}"
                ),
                "learned_association_active": True,
                "topology_mode": topology_mode,
                "division_enabled": scorer.config.division_enabled,
                "architecture": scorer.architecture,
                "association_training": "none" if association_mode == "frozen" else "fit-only-artifact",
                "trainable_parameter_count": scorer.trainable_parameter_count,
                **association_stats,
            }
        elif association_mode == "a0":
            if not isinstance(scorer, A0CheckpointScorer) or topology_mode != "legacy":
                raise V45FastEval("A0 association requires the frozen scorer and legacy topology")
            probabilities, association_stats = a0_probs(video, scorer)
            no_link_probabilities = None
            association = {
                "mode": "frozen-v45-component-local-attention",
                "learned_association_active": True,
                "topology_mode": "legacy",
                "architecture": scorer.architecture,
                "frozen_parameter_count": scorer.frozen_parameter_count,
                "trainable_parameter_count": scorer.trainable_parameter_count,
                **association_stats,
            }
        else:
            raise V45FastEval("unsupported association mode")
        artifact_division_enabled = (
            scorer.config.division_enabled
            if isinstance(scorer, SparseEdgeScorer)
            else True
        )
        division_enabled = True if division_mode == "inherited" else artifact_division_enabled
        association["artifact_division_enabled"] = artifact_division_enabled
        association["effective_division_enabled"] = division_enabled
        association["division_mode"] = division_mode
        graph = bh_graph.calibrate_v45_graph(
            graph_nodes,
            learned_edge_probabilities=probabilities,
            source_no_link_probabilities=no_link_probabilities,
            image_refine_callback=refiner,
            topology_mode=topology_mode,
            division_enabled=division_enabled,
        )
        if graph.provenance.get("full_frame_dense_matrix") is not False:
            raise V45FastEval("fallback graph did not attest sparse execution")
        graph = run_bh.project_graph_result_to_image_support(
            job.record.dataset,
            graph,
            job.record.image_shape,
            scale_zyx=scale,
            max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
        )
        rows = run_bh.graph_rows(job.record.dataset, graph, job.record.image_shape)
        for index, row in enumerate(rows):
            row["id"] = index
        run_bh.validate_submission_rows(rows, [job.record])
        fragment = out_dir / "frags" / f"{job.slug}.csv"
        run_bh.write_rows_atomic(fragment, rows)
        return {
            "identity": job.identity,
            "dataset": job.record.dataset,
            "fragment": str(fragment),
            "fragment_sha256": _sha_file(fragment),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "rows": len(rows),
            "seconds": time.perf_counter() - started,
            "cache": "disabled-distinct-fast-execution-identity",
            "inference": {
                "schema": fast_video.schema,
                "execution_sha256": fast_video.execution_sha256,
                "d4_views": list(video.d4_views),
                "pair_batch_size": video.pair_batch_size,
                "view_batch_size": video.view_batch_size,
            },
            "graph": {
                **association,
                "full_frame_dense_matrix": False,
                "stats": dict(graph.stats),
                "image_support_projection": dict(graph.provenance["image_support_projection"]),
                "lazy_refinement": refiner.report(),
            },
        }
    finally:
        handle.close()


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="competition ZIP or extracted dataset root")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--support-root", type=Path, default=ROOT / "external" / "Tracking_Support")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--metrics-sha256", default=DEFAULT_METRICS_SHA256)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--registry", type=Path, default=ROOT / "reports/g02/dataset_registry_v1.json")
    parser.add_argument("--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json")
    parser.add_argument("--assoc", choices=("fallback", "frozen", "sparse", "a0"), default="fallback")
    parser.add_argument("--topology", choices=("legacy", "scored_partial"), default="legacy")
    parser.add_argument(
        "--division-mode",
        choices=("artifact", "inherited"),
        default="artifact",
        help="Use the artifact flag or keep the inherited V45 geometric division branch active.",
    )
    parser.add_argument("--edge-artifact", type=Path)
    parser.add_argument("--edge-artifact-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.datasets is not None:
        raise V45FastEval("--limit and --datasets are mutually exclusive")
    has_edge_artifact = args.edge_artifact is not None or args.edge_artifact_sha256 is not None
    if args.assoc == "sparse" and (args.edge_artifact is None or args.edge_artifact_sha256 is None):
        raise V45FastEval("sparse association requires a pinned edge artifact and SHA-256")
    if args.assoc != "sparse" and has_edge_artifact:
        raise V45FastEval("edge artifact options require sparse association")
    if args.assoc != "sparse" and args.topology != "legacy":
        raise V45FastEval("scored_partial topology requires fit-only sparse association")
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise V45FastEval("CUDA was requested but is unavailable")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if args.out_dir.exists():
        raise V45FastEval("out-dir must not already exist")
    registry_sha256 = _sha_file(args.registry)
    split_sha256 = _sha_file(args.split)
    provenance_sha256 = _sha_file(args.provenance)
    split = _load_json(args.split, expected_sha256=split_sha256, label="split")
    registry = _load_json(args.registry, expected_sha256=registry_sha256, label="registry")
    provenance = _load_json(args.provenance, expected_sha256=provenance_sha256, label="provenance")
    if (
        provenance.get("artifact") != "fit_provenance_v1"
        or provenance.get("grouped_split_sha256") != split_sha256
        or provenance.get("dataset_registry_sha256") != registry_sha256
    ):
        raise V45FastEval("fit provenance does not bind the evaluator split and registry")
    edge_artifact_sha256 = None
    if args.edge_artifact is not None:
        edge_artifact_sha256 = _sha_file(args.edge_artifact)
        if edge_artifact_sha256 != args.edge_artifact_sha256:
            raise V45FastEval("edge artifact SHA-256 does not match the pinned value")
    records = run_bh.load_registry(args.registry)
    group_by_id = registry_groups(registry, records)
    scope = select_heldout(
        records,
        split,
        fold=args.fold,
        registry_sha256=registry_sha256,
        split_sha256=split_sha256,
        group_by_id=group_by_id,
        requested=args.datasets,
        limit=args.limit,
    )
    metrics = _load_json(args.metrics, expected_sha256=args.metrics_sha256, label="training metrics")
    verify_parent_manifests(
        metrics,
        registry_sha256=registry_sha256,
        split_sha256=split_sha256,
        provenance_sha256=provenance_sha256,
    )
    import torch

    core, bundle = load_v45_model(args.support_root, device=args.device)
    model = FrozenV45Fast(core).to(args.device).eval()
    if model.accounting()["trainable"] != FrozenV45Fast.head_params():
        raise V45FastEval("fast model trainable parameter count drifted")
    ident = load_fast(
        args.checkpoint,
        model,
        bundle,
        expected_sha256=args.checkpoint_sha256,
        target_sha256=TARGET_CONTRACT_SHA256,
    )
    sources = verify_training(metrics, args.checkpoint_sha256, ident, fold=args.fold)
    scorer: SparseEdgeScorer | A0CheckpointScorer | None = None
    edge_identity = None
    edge_fit_parameter_count = None
    if args.assoc == "sparse":
        auxiliary = fast_readout(
            model,
            bundle,
            ident,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        scorer, edge_identity = load_edge_artifact(
            args.edge_artifact,
            model.core,
            expected_pretrained_weight_sha256=bundle.hashes.weight_sha256,
            auxiliary=auxiliary,
        )
        verify_edge_artifact_scope(
            edge_identity,
            sources,
            scope,
            split,
            provenance_sha256=provenance_sha256,
        )
        edge_fit_parameter_count = freeze_fitted_scorer_for_evaluation(scorer)
    elif args.assoc == "frozen":
        scorer = frozen_sparse_scorer(model.core, bundle)
    elif args.assoc == "a0":
        scorer = A0CheckpointScorer.from_loaded_model(model.core, bundle)
        if scorer.trainable_parameter_count != 0:
            raise V45FastEval("A0 association must remain fully frozen")
    source_ids = {item["video_id"] for item in sources}
    source_groups = {item["group_id"] for item in sources}
    if any(
        str(record.video_id) in source_ids or scope.group_by_id[str(record.video_id)] in source_groups
        for record in scope.records
    ):
        raise V45FastEval("fit-only sources leaked into requested held-out evaluation")
    fast_video_config_sha256 = v45_config_sha(pair_batch_size=1, view_batch_size=1, d4_views=(0,))
    execution_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema": "celltrack-alpha.v45-fast-infer.v1",
                "video_config_sha256": fast_video_config_sha256,
                "core_state_sha256": ident.core_state_sha256,
                "fast_state_sha256": ident.state_sha256,
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    base_report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "scope": {
            "fold": scope.fold,
            "selected_video_ids": [str(record.video_id) for record in scope.records],
            "held_out_video_count": len(scope.held_out_ids),
            "audit_video_count": len(scope.audit_ids),
            "audit_used": False,
            "fit_used": False,
            "test_used": False,
            "split_sha256": scope.split_sha256,
            "registry_sha256": scope.registry_sha256,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": args.checkpoint_sha256,
            "identity": ident.as_dict(),
            "metrics_path": str(args.metrics),
            "metrics_sha256": args.metrics_sha256,
            "fit_sources": list(sources),
        },
        "execution": {
            "device": args.device,
            "model": "frozen-v45-one-pass-compact-head",
            "d4_views": [0],
            "base_v45_cache": "disabled",
            "learned_association_active": scorer is not None,
            "graph": (
                f"fit-only-v45-sparse-{args.topology}"
                if args.assoc == "sparse"
                else "frozen-v45-sparse-identity"
                if args.assoc == "frozen"
                else "frozen-v45-component-local-attention"
                if args.assoc == "a0"
                else "rule-v45-sparse-fallback"
            ),
            "association": (
                f"fit-only-sparse-{args.topology}"
                if args.assoc == "sparse"
                else "frozen-sparse-identity"
                if args.assoc == "frozen"
                else "frozen-component-local-attention"
                if args.assoc == "a0"
                else "none"
            ),
            "topology_mode": args.topology,
            "division_mode": args.division_mode,
            "effective_division_enabled": (
                True
                if args.division_mode == "inherited"
                else scorer.config.division_enabled
                if isinstance(scorer, SparseEdgeScorer)
                else True
            ),
            "edge_artifact": None if edge_identity is None else {
                "path": str(args.edge_artifact),
                "sha256": edge_artifact_sha256,
                "identity": asdict(edge_identity),
                "fitted_parameter_count": edge_fit_parameter_count,
                "runtime_trainable_parameter_count": scorer.trainable_parameter_count,
            },
            "association_training": "fit-only-artifact" if args.assoc == "sparse" else "none",
            "frozen_sparse_checkpoint": None if args.assoc != "frozen" else {
                "pretrained_weight_sha256": scorer.pretrained_weight_sha256,
                "frozen_subset_sha256": scorer.frozen_subset_sha256,
                "trainable_parameter_count": scorer.trainable_parameter_count,
                "factory": "frozen_sparse_scorer",
            },
            "a0_checkpoint": None if not isinstance(scorer, A0CheckpointScorer) else {
                "weight_sha256": scorer.pretrained_weight_sha256,
                "architecture_sha256": scorer.architecture_sha256,
                "frozen_subset_sha256": scorer.frozen_subset_sha256,
                "trainable_parameter_count": scorer.trainable_parameter_count,
                "factory": "A0CheckpointScorer.from_loaded_model",
            },
            "dense_nxn_association": False,
            "target_contract_sha256": TARGET_CONTRACT_SHA256,
            "video_config_sha256": fast_video_config_sha256,
            "expected_execution_sha256": execution_sha256,
        },
    }
    if args.dry_run:
        _atomic_json(args.out_dir / "report.json", base_report)
        print(json.dumps(base_report, ensure_ascii=True, sort_keys=True))
        return 0
    jobs = run_bh.discover_jobs(args.input, scope.records)
    results = [
        evaluate_one(
            job,
            model=model,
            expected_execution_sha256=execution_sha256,
            device=args.device,
            out_dir=args.out_dir,
            scorer=scorer,
            association_mode=args.assoc,
            topology_mode=args.topology,
            division_mode=args.division_mode,
        )
        for job in jobs
    ]
    elapsed = sum(float(item["seconds"]) for item in results)
    score_command: list[str] | None = [
        sys.executable,
        str(ROOT / "scripts/scoreall.py"),
        "--zip",
        str(args.input),
        "--split",
        str(args.split),
        "--subset",
        f"fold{scope.fold}",
        "--frags",
        str(args.out_dir / "frags"),
        "--output",
        str(args.out_dir / "scores.json"),
    ]
    selected_ids = tuple(str(record.video_id) for record in scope.records)
    if len(selected_ids) < len(scope.held_out_ids):
        if selected_ids == scope.held_out_ids[: len(selected_ids)]:
            score_command.extend(("--max-datasets", str(len(selected_ids))))
        else:
            score_command = None
    report = {
        **base_report,
        "status": "completed",
        "results": results,
        "summary": {
            "videos": len(results),
            "nodes": sum(int(item["nodes"]) for item in results),
            "edges": sum(int(item["edges"]) for item in results),
            "seconds": elapsed,
            "seconds_per_video": elapsed / max(1, len(results)),
            "full_fold": len(results) == len(scope.held_out_ids),
            "score_command": score_command,
        },
    }
    _atomic_json(args.out_dir / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

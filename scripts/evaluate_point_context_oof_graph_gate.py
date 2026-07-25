#!/usr/bin/env python3
"""Run the nested 6bba point-context full-pool held-graph screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
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
    point_context_swap,
)
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from celltrack_alpha.point_context import (  # noqa: E402
    PointContextTransformer,
    ProposalWindow,
    collate_windows,
    parental_softmax_loss,
)
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _edge_counts,
    _realization,
)
from evaluate_selected_graph_direct_swap import _solve_direct  # noqa: E402
from evaluate_temporal_rank_oof_graph_gate import (  # noqa: E402
    _cap,
    graph_gate_decision,
    validate_fit_scope,
)
from fit_compact_point_context_pilot import (  # noqa: E402
    build_windows,
    score_windows,
)
from fit_selected_graph_temporal_background_gate import _fold  # noqa: E402
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _metrics,
    _select,
)


SCHEMA = "celltrack-alpha.point-context-oof-graph-gate.v1"
CONTRACT_SCHEMA = "celltrack-alpha.point-context-oof-graph-gate-contract.v1"
FOLD_CHECKPOINT_SCHEMA = (
    "celltrack-alpha.point-context-oof-fold-checkpoint.v1"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/point_context_oof_graph_gate_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "a374a9e04488e28c911385c7af7873a10f45dced630844af816a39e97ba46f53"
)
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_OUTPUT = ROOT / "reports/g02/point_context_oof_graph_gate_v1"
FAMILY = "6bba"
FOLD_CODE_PATHS = (
    "scripts/evaluate_point_context_oof_graph_gate.py",
    "scripts/fit_compact_point_context_pilot.py",
    "src/celltrack_alpha/bh_model.py",
    "src/celltrack_alpha/point_context.py",
    "src/celltrack_alpha/point_context_swap.py",
)


class PointContextOOFGraphGateError(RuntimeError):
    """The prospective nested point-context graph screen failed closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PointContextOOFGraphGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PointContextOOFGraphGateError(f"{label} must be a JSON object")
    return value


def _root_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PointContextOOFGraphGateError(f"{label} path is missing")
    path = ROOT / value
    if not path.is_file():
        raise PointContextOOFGraphGateError(f"{label} is missing: {path}")
    return path


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate the exact prospective design before any fitting."""

    if len(expected_sha256) != 64 or sweep._sha256(path) != expected_sha256:
        raise PointContextOOFGraphGateError("contract SHA-256 mismatch")
    value = _load_json(path, "contract")
    scope = value.get("scope")
    candidate = value.get("candidate")
    data = value.get("data")
    control = value.get("control")
    graph_gate = value.get("graph_gate")
    execution = value.get("execution")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or value.get("purpose")
        != "prospective-6bba-point-context-full-pool-graph-screen"
        or not all(
            isinstance(section, Mapping)
            for section in (
                scope,
                candidate,
                data,
                control,
                graph_gate,
                execution,
            )
        )
    ):
        raise PointContextOOFGraphGateError("contract is malformed")
    if (
        scope.get("family") != FAMILY
        or int(scope.get("eligible_video_count", 0)) != 69
        or int(scope.get("outer_fold_count", 0)) != 5
        or scope.get("outer_fold_assignment")
        != "sha256(video identity)[0:8] modulo 5"
        or scope.get("calibration_fold")
        != "(outer_held_fold + 1) modulo 5"
        or any(
            scope.get(name) is not False
            for name in (
                "validation12_access",
                "cv57_access",
                "promotion_authorized",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
    ):
        raise PointContextOOFGraphGateError("scope is not fail-closed")
    if dict(execution) != {
        "fold_checkpoint_schema": FOLD_CHECKPOINT_SCHEMA,
        "checkpoint_unit": (
            "one complete fit plus calibration plus held-proposal outer fold"
        ),
        "publication": (
            "same-filesystem atomic directory rename after model and "
            "evidence fsync"
        ),
        "resume_policy": (
            "accept only complete checkpoints matching contract, "
            "implementation, partition, model, artifact, and metric "
            "identities"
        ),
        "unpublished_stage_policy": (
            "ignore as non-authoritative and recompute the fold"
        ),
        "graph_resume": False,
    }:
        raise PointContextOOFGraphGateError(
            "fold checkpoint execution contract drifted"
        )
    architecture = candidate.get("architecture")
    optimization = candidate.get("optimization")
    threshold = candidate.get("threshold_selection")
    prerequisite = candidate.get("proposal_prerequisite")
    pool = candidate.get("deployment_pool")
    if (
        dict(architecture or {})
        != {
            "feature_count": 48,
            "width": 96,
            "heads": 4,
            "layers": 2,
            "dropout": 0.1,
            "parental_softmax": True,
            "explicit_no_parent": True,
        }
        or dict(optimization or {})
        != {
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "background_weight": 0.05,
            "batch_size_windows": 64,
            "max_steps": 500,
            "seed_base": 240824,
        }
        or threshold.get("calibration_partition_only") is not True
        or int(threshold.get("minimum_calibration_accepted", 0)) != 25
        or float(threshold.get("minimum_calibration_precision", 0.0)) != 0.95
        or int(prerequisite.get("minimum_held_accepted", 0)) != 25
        or float(prerequisite.get("minimum_held_precision", 0.0)) != 0.95
        or pool.get("sampled_background_used_for_inference") is not False
        or pool.get("retain_challenger_and_all_conflict_ids") is not True
    ):
        raise PointContextOOFGraphGateError("candidate contract drifted")
    if (
        float(graph_gate.get("minimum_adjusted_edge_jaccard_delta", 0.0))
        != 0.005
        or graph_gate.get("require_nonnegative_raw_edge_jaccard_delta")
        is not True
        or float(graph_gate.get("minimum_node_recall_delta", -1.0)) != 0.0
        or graph_gate.get("require_observation_distinct") is not True
    ):
        raise PointContextOOFGraphGateError("graph gate drifted")
    return value


def nested_partitions(
    identities: Sequence[str], fold_count: int
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    """Return disjoint fit/calibration/held partitions for every outer fold."""

    ordered = tuple(str(value) for value in identities)
    if (
        fold_count < 3
        or not ordered
        or len(ordered) != len(set(ordered))
    ):
        raise PointContextOOFGraphGateError(
            "nested partition identities are invalid"
        )
    held_once: list[str] = []
    result = []
    for held_fold in range(fold_count):
        calibration_fold = (held_fold + 1) % fold_count
        fit_ids = tuple(
            identity
            for identity in ordered
            if _fold(identity, fold_count)
            not in (held_fold, calibration_fold)
        )
        calibration_ids = tuple(
            identity
            for identity in ordered
            if _fold(identity, fold_count) == calibration_fold
        )
        held_ids = tuple(
            identity
            for identity in ordered
            if _fold(identity, fold_count) == held_fold
        )
        sets = (set(fit_ids), set(calibration_ids), set(held_ids))
        if (
            not all(sets)
            or sets[0] & sets[1]
            or sets[0] & sets[2]
            or sets[1] & sets[2]
            or set().union(*sets) != set(ordered)
        ):
            raise PointContextOOFGraphGateError(
                f"invalid nested partition for held fold {held_fold}"
            )
        held_once.extend(held_ids)
        result.append((fit_ids, calibration_ids, held_ids))
    if sorted(held_once) != sorted(ordered):
        raise PointContextOOFGraphGateError(
            "each eligible identity must be held exactly once"
        )
    return result


def _windows_for(
    windows: Sequence[ProposalWindow], identities: set[str]
) -> list[ProposalWindow]:
    result = [window for window in windows if window.identity in identities]
    observed = {window.identity for window in result}
    if observed != identities:
        missing = sorted(identities - observed)
        raise PointContextOOFGraphGateError(
            f"window coverage is incomplete: {missing}"
        )
    return result


def _load_partition_windows(
    prepare: Mapping[str, Any],
    *,
    cache_root: Path,
    identities: Sequence[str],
) -> tuple[list[ProposalWindow], dict[str, int]]:
    """Load only one nested partition so held payloads are never co-resident."""

    expected = set(str(value) for value in identities)
    windows, selected_edges, _ = build_windows(
        prepare,
        cache_root=cache_root,
        allowed_identities=expected,
    )
    observed = {window.identity for window in windows}
    if (
        observed != expected
        or set(selected_edges) != expected
        or any(window.family != 1 for window in windows)
    ):
        raise PointContextOOFGraphGateError(
            "partition window coverage or family identity drifted"
        )
    return windows, selected_edges


def _train(
    windows: Sequence[ProposalWindow],
    configuration: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[PointContextTransformer, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one deterministic outer-fold model using fit identities only."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features = np.concatenate([window.features for window in windows]).astype(
        np.float32, copy=False
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-5] = 1.0
    model = PointContextTransformer(
        feature_count=int(configuration["feature_count"]),
        width=int(configuration["width"]),
        heads=int(configuration["heads"]),
        layers=int(configuration["layers"]),
        dropout=float(configuration["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    totals: dict[str, int] = defaultdict(int)
    step = 0
    maximum = int(configuration["max_steps"])
    batch_size = int(configuration["batch_size_windows"])
    while step < maximum:
        order = torch.randperm(len(windows), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            chosen = [
                windows[index] for index in order[start : start + batch_size]
            ]
            batch = collate_windows(
                chosen,
                mean=mean,
                scale=scale,
                device=torch.device("cpu"),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            edge_logits, hidden = model(
                batch["features"],
                batch["families"],
                batch["padding_mask"],
            )
            loss, groups = parental_softmax_loss(
                model,
                edge_logits,
                hidden,
                batch["labels"],
                batch["target_ids"],
                batch["padding_mask"],
                background_weight=float(configuration["background_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            for key, count in groups.items():
                totals[key] += int(count)
            step += 1
            if step >= maximum:
                break
    model.eval()
    return model, mean, scale, {
        "steps": step,
        "final_loss": losses[-1],
        "mean_last_50_loss": float(np.mean(losses[-50:])),
        "parental_groups_seen": dict(totals),
    }


def _metric_subset(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value[name]
        for name in (
            "accepted_labeled",
            "correct_rescues",
            "harmful_swaps",
            "background_accepted",
            "conservative_utility",
            "precision",
            "total_selected",
        )
    }


def aggregate_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate already-thresholded fold metrics without mixing thresholds."""

    integer_keys = (
        "accepted_labeled",
        "correct_rescues",
        "harmful_swaps",
        "background_accepted",
        "conservative_utility",
        "total_selected",
    )
    result = {
        key: sum(int(value[key]) for value in values) for key in integer_keys
    }
    accepted = result["accepted_labeled"]
    result["precision"] = (
        result["correct_rescues"] / accepted if accepted else 0.0
    )
    return result


def fold_code_identity() -> dict[str, str]:
    """Bind resumable folds to the exact implementation that produced them."""

    result = {}
    for relative in FOLD_CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise PointContextOOFGraphGateError(
                f"fold implementation file is missing: {relative}"
            )
        result[relative] = sweep._sha256(path)
    return result


def _fold_paths(model_dir: Path, held_fold: int) -> tuple[Path, Path]:
    stem = f"point_context_{FAMILY}_fold{held_fold}"
    fold_dir = model_dir / stem
    return fold_dir / "model.pt", fold_dir / "checkpoint.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_fold_directory(model_dir: Path, fold_count: int) -> None:
    """Reject unknown authoritative entries while ignoring unpublished stages."""

    expected = {
        f"point_context_{FAMILY}_fold{held_fold}"
        for held_fold in range(fold_count)
    }
    for entry in model_dir.iterdir():
        if entry.name in expected:
            if not entry.is_dir():
                raise PointContextOOFGraphGateError(
                    f"fold checkpoint path is not a directory: {entry}"
                )
            continue
        if (
            entry.is_dir()
            and entry.name.startswith(f".point_context_{FAMILY}_fold")
            and entry.name.endswith(".stage")
        ):
            continue
        raise PointContextOOFGraphGateError(
            f"unexpected fold checkpoint entry: {entry}"
        )


def _model_spec(architecture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_count": int(architecture["feature_count"]),
        "width": int(architecture["width"]),
        "heads": int(architecture["heads"]),
        "layers": int(architecture["layers"]),
        "dropout": float(architecture["dropout"]),
    }


def write_fold_checkpoint(
    *,
    model_dir: Path,
    held_fold: int,
    model: PointContextTransformer,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    seed: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    held_ids: Sequence[str],
    architecture: Mapping[str, Any],
    fit_record: Mapping[str, Any],
    calibration_metrics: Mapping[str, Any],
    held_metrics: Mapping[str, Any],
    contract_sha256: str,
    code_identity: Mapping[str, str],
    fold_count: int,
) -> dict[str, Any]:
    """Atomically publish one immutable fold artifact and its evidence record."""

    artifact, checkpoint = _fold_paths(model_dir, held_fold)
    fold_dir = artifact.parent
    if fold_dir.exists():
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} checkpoint already exists"
        )
    spec = _model_spec(architecture)
    calibration_record = _metric_subset(calibration_metrics)
    held_record = _metric_subset(held_metrics)
    optimization_record = dict(fit_record)
    artifact_payload = {
        "schema": SCHEMA,
        "family": FAMILY,
        "held_fold": held_fold,
        "fit_ids": list(fit_ids),
        "calibration_ids": list(calibration_ids),
        "held_ids": list(held_ids),
        **spec,
        "mean": np.asarray(mean, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "threshold": float(threshold),
        "state_dict": model.state_dict(),
        "seed": int(seed),
        "contract_sha256": contract_sha256,
        "code_identity": dict(code_identity),
        "calibration": calibration_record,
        "held": held_record,
        "optimization": optimization_record,
    }
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{fold_dir.name}.", suffix=".stage", dir=model_dir
        )
    )
    stage_artifact = stage / artifact.name
    stage_checkpoint = stage / checkpoint.name
    try:
        buffer = io.BytesIO()
        torch.save(artifact_payload, buffer)
        run_bh.atomic_create_bytes(stage_artifact, buffer.getvalue())
        fold_record = {
            "held_fold": held_fold,
            "calibration_fold": (held_fold + 1) % fold_count,
            "seed": int(seed),
            "fit_ids": list(fit_ids),
            "calibration_ids": list(calibration_ids),
            "held_ids": list(held_ids),
            "fit_video_count": len(fit_ids),
            "calibration_video_count": len(calibration_ids),
            "held_video_count": len(held_ids),
            "threshold": float(threshold),
            "calibration": calibration_record,
            "held": held_record,
            "optimization": optimization_record,
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sweep._sha256(stage_artifact),
        }
        record = {
            "schema": FOLD_CHECKPOINT_SCHEMA,
            "contract_sha256": contract_sha256,
            "code_identity": dict(code_identity),
            "fold": fold_record,
        }
        run_bh.atomic_create_json(stage_checkpoint, record)
        _fsync_directory(stage)
        os.rename(stage, fold_dir)
        _fsync_directory(model_dir)
    finally:
        if stage.exists():
            for path in (stage_artifact, stage_checkpoint):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            try:
                stage.rmdir()
            except FileNotFoundError:
                pass
    return fold_record


def load_fold_checkpoint(
    *,
    model_dir: Path,
    held_fold: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    held_ids: Sequence[str],
    architecture: Mapping[str, Any],
    seed: int,
    contract_sha256: str,
    code_identity: Mapping[str, str],
    fold_count: int,
) -> (
    tuple[
        PointContextTransformer,
        np.ndarray,
        np.ndarray,
        float,
        dict[str, Any],
    ]
    | None
):
    """Restore one complete fold or fail closed on any partial/drifted state."""

    artifact, checkpoint = _fold_paths(model_dir, held_fold)
    if not artifact.exists() and not checkpoint.exists():
        return None
    if not artifact.is_file() or not checkpoint.is_file():
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} checkpoint is partial"
        )
    record = _load_json(checkpoint, f"fold {held_fold} checkpoint")
    if (
        set(record) != {
            "schema",
            "contract_sha256",
            "code_identity",
            "fold",
        }
        or record["schema"] != FOLD_CHECKPOINT_SCHEMA
        or record["contract_sha256"] != contract_sha256
        or record["code_identity"] != dict(code_identity)
        or not isinstance(record["fold"], Mapping)
    ):
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} checkpoint identity drifted"
        )
    fold_record = dict(record["fold"])
    expected_record_keys = {
        "held_fold",
        "calibration_fold",
        "seed",
        "fit_ids",
        "calibration_ids",
        "held_ids",
        "fit_video_count",
        "calibration_video_count",
        "held_video_count",
        "threshold",
        "calibration",
        "held",
        "optimization",
        "artifact",
        "artifact_sha256",
    }
    if (
        set(fold_record) != expected_record_keys
        or fold_record["held_fold"] != held_fold
        or fold_record["calibration_fold"]
        != (held_fold + 1) % fold_count
        or fold_record["seed"] != seed
        or fold_record["fit_ids"] != list(fit_ids)
        or fold_record["calibration_ids"] != list(calibration_ids)
        or fold_record["held_ids"] != list(held_ids)
        or fold_record["fit_video_count"] != len(fit_ids)
        or fold_record["calibration_video_count"] != len(calibration_ids)
        or fold_record["held_video_count"] != len(held_ids)
        or fold_record["artifact"] != str(artifact.resolve())
        or fold_record["artifact_sha256"] != sweep._sha256(artifact)
        or not isinstance(fold_record["calibration"], Mapping)
        or not isinstance(fold_record["held"], Mapping)
        or not isinstance(fold_record["optimization"], Mapping)
    ):
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} evidence drifted"
        )
    try:
        payload = torch.load(
            artifact, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} artifact cannot be restored"
        ) from exc
    spec = _model_spec(architecture)
    expected_payload_keys = {
        "schema",
        "family",
        "held_fold",
        "fit_ids",
        "calibration_ids",
        "held_ids",
        *spec,
        "mean",
        "scale",
        "threshold",
        "state_dict",
        "seed",
        "contract_sha256",
        "code_identity",
        "calibration",
        "held",
        "optimization",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_payload_keys
        or payload["schema"] != SCHEMA
        or payload["family"] != FAMILY
        or payload["held_fold"] != held_fold
        or payload["fit_ids"] != list(fit_ids)
        or payload["calibration_ids"] != list(calibration_ids)
        or payload["held_ids"] != list(held_ids)
        or payload["seed"] != seed
        or payload["contract_sha256"] != contract_sha256
        or payload["code_identity"] != dict(code_identity)
        or payload["calibration"] != fold_record["calibration"]
        or payload["held"] != fold_record["held"]
        or payload["optimization"] != fold_record["optimization"]
        or any(payload[name] != value for name, value in spec.items())
    ):
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} artifact identity drifted"
        )
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    threshold = float(payload["threshold"])
    if (
        mean.shape != (spec["feature_count"],)
        or scale.shape != (spec["feature_count"],)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or not math.isfinite(threshold)
        or threshold != float(fold_record["threshold"])
    ):
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} normalization or threshold is invalid"
        )
    model = PointContextTransformer(**spec)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise PointContextOOFGraphGateError(
            f"fold {held_fold} state dict is invalid"
        ) from exc
    model.eval()
    return model, mean, scale, threshold, fold_record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Fit nested models and evaluate each exactly once on a held full graph."""

    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise PointContextOOFGraphGateError(
            "run must be launched by the local training guard"
        )
    if args.output.exists() and not args.resume:
        raise PointContextOOFGraphGateError(
            f"refusing to overwrite output: {args.output}"
        )
    if args.resume and not args.output.is_dir():
        raise PointContextOOFGraphGateError(
            f"resume output is missing: {args.output}"
        )
    contract = load_contract(args.contract, args.contract_sha256)
    candidate_contract = contract["candidate"]
    data_contract = contract["data"]
    manifest_path = _root_file(candidate_contract["manifest"], "candidate")
    pilot_path = _root_file(candidate_contract["pilot_report"], "pilot report")
    prepare_path = _root_file(data_contract["prepare_report"], "prepare report")
    audit_path = _root_file(data_contract["data_audit"], "data audit")
    runtime_path = _root_file(
        contract["control"]["runtime_contract"], "APNCC runtime contract"
    )
    for path, expected, label in (
        (manifest_path, candidate_contract["manifest_sha256"], "candidate"),
        (pilot_path, candidate_contract["pilot_report_sha256"], "pilot report"),
        (prepare_path, data_contract["prepare_report_sha256"], "prepare report"),
        (audit_path, data_contract["data_audit_sha256"], "data audit"),
        (
            runtime_path,
            contract["control"]["runtime_contract_sha256"],
            "APNCC runtime contract",
        ),
    ):
        if sweep._sha256(path) != expected:
            raise PointContextOOFGraphGateError(f"{label} SHA-256 mismatch")
    candidate_gate = validate_candidate(
        load_json_object(manifest_path), load_json_object(args.policy)
    )
    prepare = _load_json(prepare_path, "prepare report")
    audit = _load_json(audit_path, "data audit")
    pilot = _load_json(pilot_path, "pilot report")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or audit.get("schema")
        != "celltrack-alpha.compact-point-context-data-audit.v2"
        or audit.get("status") != "PASS"
        or audit.get("scope", {}).get("cv57_overlap") != []
        or pilot.get("schema")
        != "celltrack-alpha.compact-point-context-pilot.v1"
        or pilot.get("candidate", {})
        .get("validation_families", {})
        .get(FAMILY, {})
        .get("correct_rescues")
        != 182
    ):
        raise PointContextOOFGraphGateError("candidate evidence identity drifted")

    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = set(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.zip),
        )
    )
    eligible = validate_fit_scope(
        prepare["scope"], prepare["scope"], cv57
    )
    partitions = nested_partitions(
        eligible, int(contract["scope"]["outer_fold_count"])
    )
    model_dir = args.output / "fold_models"
    control_dir = args.output / "control_fragments"
    candidate_dir = args.output / "candidate_fragments"
    if args.resume:
        if (
            not model_dir.is_dir()
            or not control_dir.is_dir()
            or not candidate_dir.is_dir()
            or (args.output / "report.json").exists()
            or any(control_dir.iterdir())
            or any(candidate_dir.iterdir())
        ):
            raise PointContextOOFGraphGateError(
                "resume output has completed or partial graph state"
            )
        validate_fold_directory(model_dir, len(partitions))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        model_dir.mkdir()
        control_dir.mkdir()
        candidate_dir.mkdir()
    started = time.perf_counter()
    architecture = candidate_contract["architecture"]
    optimization = candidate_contract["optimization"]
    training_configuration = {**architecture, **optimization}
    threshold_contract = candidate_contract["threshold_selection"]
    fold_models: dict[
        int, tuple[PointContextTransformer, np.ndarray, np.ndarray, float]
    ] = {}
    fold_records = []
    held_fold_metrics = []
    code_identity = fold_code_identity()
    resumed_folds = 0

    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(partitions):
        seed = int(optimization["seed_base"]) + held_fold
        restored = load_fold_checkpoint(
            model_dir=model_dir,
            held_fold=held_fold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            held_ids=held_ids,
            architecture=architecture,
            seed=seed,
            contract_sha256=args.contract_sha256,
            code_identity=code_identity,
            fold_count=len(partitions),
        )
        if restored is not None:
            model, mean, scale, threshold, fold_record = restored
            fold_models[held_fold] = (model, mean, scale, threshold)
            fold_records.append(fold_record)
            held_fold_metrics.append(dict(fold_record["held"]))
            resumed_folds += 1
            print(
                f"point-context fold {held_fold + 1:02d}/"
                f"{len(partitions)} resumed sha256="
                f"{fold_record['artifact_sha256']}",
                flush=True,
            )
            continue
        fit_windows, _fit_selected_edges = _load_partition_windows(
            prepare, cache_root=args.cache, identities=fit_ids
        )
        model, mean, scale, fit_record = _train(
            fit_windows, training_configuration, seed=seed
        )
        del fit_windows, _fit_selected_edges
        gc.collect()
        calibration_windows, calibration_selected_edges = (
            _load_partition_windows(
                prepare,
                cache_root=args.cache,
                identities=calibration_ids,
            )
        )
        calibration_scored = score_windows(
            model,
            calibration_windows,
            selected_edges=calibration_selected_edges,
            mean=mean,
            scale=scale,
            device=torch.device("cpu"),
            batch_size=int(optimization["batch_size_windows"]),
        )
        threshold, calibration_metrics = _select(
            calibration_scored,
            minimum_accepted=int(
                threshold_contract["minimum_calibration_accepted"]
            ),
            minimum_precision=float(
                threshold_contract["minimum_calibration_precision"]
            ),
        )
        del calibration_windows, calibration_selected_edges, calibration_scored
        gc.collect()
        held_windows, held_selected_edges = _load_partition_windows(
            prepare, cache_root=args.cache, identities=held_ids
        )
        held_scored = score_windows(
            model,
            held_windows,
            selected_edges=held_selected_edges,
            mean=mean,
            scale=scale,
            device=torch.device("cpu"),
            batch_size=int(optimization["batch_size_windows"]),
        )
        held_metrics = _metrics(held_scored, threshold)
        del held_windows, held_selected_edges, held_scored
        gc.collect()
        held_fold_metrics.append(held_metrics)
        fold_models[held_fold] = (model, mean, scale, threshold)
        fold_records.append(
            write_fold_checkpoint(
                model_dir=model_dir,
                held_fold=held_fold,
                model=model,
                mean=mean,
                scale=scale,
                threshold=threshold,
                seed=seed,
                fit_ids=fit_ids,
                calibration_ids=calibration_ids,
                held_ids=held_ids,
                architecture=architecture,
                fit_record=fit_record,
                calibration_metrics=calibration_metrics,
                held_metrics=held_metrics,
                contract_sha256=args.contract_sha256,
                code_identity=code_identity,
                fold_count=len(partitions),
            )
        )
        print(
            f"point-context fold {held_fold + 1:02d}/{len(partitions)} "
            f"fit={len(fit_ids)} cal={len(calibration_ids)} "
            f"held={len(held_ids)} threshold={threshold:.8f} "
            f"held_precision={float(held_metrics['precision']):.6f}",
            flush=True,
        )

    held_oof = aggregate_metrics(held_fold_metrics)
    prerequisite = candidate_contract["proposal_prerequisite"]
    prerequisite_passed = (
        int(held_oof["accepted_labeled"])
        >= int(prerequisite["minimum_held_accepted"])
        and float(held_oof["precision"])
        >= float(prerequisite["minimum_held_precision"])
    )
    if not prerequisite_passed:
        result = {
            "schema": SCHEMA,
            "status": "PARK",
            "scope": {
                "family": FAMILY,
                "source_partition": "fit-train105",
                "video_count": len(eligible),
                "validation12_access": False,
                "cv57_access": False,
                "promotion_authorized": False,
            },
            "proposal_prerequisite": {
                "held_oof": held_oof,
                "contract": prerequisite,
                "passed": False,
            },
            "folds": fold_records,
            "graph": None,
            "decision": {
                "status": "PARK",
                "reason": "held proposal precision/support prerequisite failed",
                "promotion_authorized": False,
            },
            "provenance": {
                "contract": str(args.contract),
                "contract_sha256": args.contract_sha256,
                "candidate_gate": candidate_gate,
            },
            "execution": {
                "device": "cpu",
                "guard_required": True,
                "fold_checkpoint_schema": FOLD_CHECKPOINT_SCHEMA,
                "fold_code_identity": code_identity,
                "resumed_fold_count": resumed_folds,
                "kaggle_action": False,
                "gpu_action": False,
                "runpod_action": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        run_bh.atomic_create_json(args.output / "report.json", result)
        return result

    apncc_contract, geometry_gate = appearance_swap.load_contract(
        runtime_path,
        expected_sha256=contract["control"]["runtime_contract_sha256"],
        root=ROOT,
    )
    rows_by_identity = {
        str(row["identity"]): row
        for row in prepare["videos"]
        if str(row["identity"]) in set(eligible)
    }
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if (
        set(rows_by_identity) != set(eligible)
        or not set(eligible).issubset(cache_paths)
        or not set(eligible).issubset(meta_paths)
    ):
        raise PointContextOOFGraphGateError("fit graph coverage is incomplete")
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
        raise PointContextOOFGraphGateError("frozen edge scorer did not restore")

    graph_records = []
    identical_fragments = 0
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(eligible, start=1):
            held_fold = _fold(identity, len(partitions))
            model, mean, scale, threshold = fold_models[held_fold]
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
                raise PointContextOOFGraphGateError(
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
            runtime_windows, pool_counts = (
                point_context_swap.build_full_pool_windows(
                    cache,
                    scorer,
                    baseline_selected,
                    geometry_gate,
                    image,
                    identity=identity,
                    family=1,
                    geometry_threshold=float(
                        apncc_contract["geometry_gate"]["threshold"]
                    ),
                )
            )
            candidate_proposals, candidate_counts = (
                point_context_swap.propose_swaps(
                    model,
                    runtime_windows,
                    mean=mean,
                    scale=scale,
                    threshold=threshold,
                    maximum=proposal_cap,
                    batch_size=int(optimization["batch_size_windows"]),
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
                raise PointContextOOFGraphGateError(
                    f"direct swap realization failed for {identity}"
                )
            control_hash = sweep._sha256(Path(control_solved["fragment"]))
            candidate_hash = sweep._sha256(
                Path(candidate_solved["fragment"])
            )
            equal = control_hash == candidate_hash
            identical_fragments += int(equal)
            graph_records.append(
                {
                    "identity": identity,
                    "held_fold": held_fold,
                    "fit_contains_identity": False,
                    "calibration_contains_identity": False,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_cap": proposal_cap,
                    "control_proposals": len(control_proposals),
                    "candidate_proposals": len(candidate_proposals),
                    "control_proposal_counts": control_counts,
                    "candidate_pool_counts": pool_counts,
                    "candidate_score_counts": candidate_counts,
                    "control_realization": control_realization,
                    "candidate_realization": candidate_realization,
                    "control_fragment_sha256": control_hash,
                    "candidate_fragment_sha256": candidate_hash,
                    "fragments_exactly_equal": equal,
                }
            )
            print(
                f"point-context graph {index:02d}/{len(eligible)} {identity} "
                f"fold={held_fold} apncc={len(control_proposals)} "
                f"candidate={len(candidate_proposals)} equal={equal}",
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
    observation_distinct = identical_fragments < len(eligible)
    graph_passed = (
        decision["counterfactual_gate_status"] == "PASS-GRAPH"
        and observation_distinct
    )
    decision.update(
        {
            "status": "PASS-GRAPH" if graph_passed else "PARK",
            "proposal_prerequisite_passed": True,
            "observation_distinct": observation_distinct,
            "promotion_authorized": False,
            "next_action": (
                "validation12-only-confirmation"
                if graph_passed
                else "PARK-no-validation12-or-CV57"
            ),
        }
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS-GRAPH" if graph_passed else "PARK",
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
            "held_oof": held_oof,
            "contract": prerequisite,
            "passed": True,
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
                "all_observations_identical": not observation_distinct,
            },
            "videos": graph_records,
        },
        "decision": decision,
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "candidate": str(manifest_path),
            "candidate_sha256": sweep._sha256(manifest_path),
            "candidate_gate": candidate_gate,
            "pilot_report": str(pilot_path),
            "pilot_report_sha256": sweep._sha256(pilot_path),
            "prepare_report": str(prepare_path),
            "prepare_report_sha256": sweep._sha256(prepare_path),
            "data_audit": str(audit_path),
            "data_audit_sha256": sweep._sha256(audit_path),
            "runtime_contract": str(runtime_path),
            "runtime_contract_sha256": apncc_contract["contract_sha256"],
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "fold_checkpoint_schema": FOLD_CHECKPOINT_SCHEMA,
            "fold_code_identity": code_identity,
            "resumed_fold_count": resumed_folds,
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
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
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
    value.add_argument(
        "--resume",
        action="store_true",
        help="resume only complete hash-bound folds from an existing output",
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report["decision"], sort_keys=True))

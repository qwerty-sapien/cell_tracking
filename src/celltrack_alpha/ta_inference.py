"""Fold-safe full-video inference helpers for ``TemporalAdapterModel``.

This module is deliberately small.  It validates the paired ``train_ta``
artifacts before image access, preserves the child-owned backward-motion
contract during sliding inference, emits only sparse alpha-style association
candidates, and writes fragments consumable by the frozen official scorer.
It never reads GEFF, audit, or test data itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    write_submission,
)


GATE_SCHEMA = "celltrack-alpha.temporal-adapter-heldout-gate.v1"
PRETRAIN_OBJECTIVE = "alpha-sparse-migration-ranking.v1"
COMPARISON_IDENTITY_FIELDS = (
    "fold",
    "scenario",
    "split_sha256",
    "provenance_sha256",
    "registry_sha256",
    "weights_sha256",
    "config_sha256",
    "seed",
    "fraction",
    "lr",
    "unfreeze",
    "detector_policy",
    "center_weight",
    "teacher_weight",
    "selected_sources",
)


class GateError(ValueError):
    """The inference gate cannot proceed without violating its provenance contract."""


@dataclass(frozen=True)
class TrainOutputAttestation:
    label: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    metrics_path: str
    metrics_sha256: str
    fold: int
    completed_steps: int
    source_groups: tuple[str, ...]
    source_video_ids: tuple[str, ...]
    temporal_pretrain_sha256: str | None
    identity: Mapping[str, Any]

    def json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "metrics_path": self.metrics_path,
            "metrics_sha256": self.metrics_sha256,
            "fold": self.fold,
            "completed_steps": self.completed_steps,
            "source_groups": list(self.source_groups),
            "source_video_ids": list(self.source_video_ids),
            "temporal_pretrain_sha256": self.temporal_pretrain_sha256,
        }


@dataclass(frozen=True)
class PretrainAttestation:
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    metrics_path: str
    metrics_sha256: str
    source_groups: tuple[str, ...]
    source_video_ids: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "metrics_path": self.metrics_path,
            "metrics_sha256": self.metrics_sha256,
            "source_groups": list(self.source_groups),
            "source_video_ids": list(self.source_video_ids),
        }


@dataclass(frozen=True)
class GateAttestation:
    fold: int
    held_out_dataset: str
    held_out_video_id: str
    held_out_group_id: str
    held_out_exact_identity_sha256: str
    held_out_image_shape: tuple[int, int, int, int]
    split_path: str
    split_sha256: str
    provenance_path: str
    provenance_sha256: str
    registry_path: str
    registry_sha256: str
    baseline: TrainOutputAttestation
    candidate: TrainOutputAttestation
    pretrain: PretrainAttestation
    audit_disjoint: bool = True
    test_used: bool = False

    def json(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "fold": self.fold,
            "held_out_dataset": self.held_out_dataset,
            "held_out_video_id": self.held_out_video_id,
            "held_out_group_id": self.held_out_group_id,
            "held_out_exact_identity_sha256": self.held_out_exact_identity_sha256,
            "held_out_image_shape": list(self.held_out_image_shape),
            "manifests": {
                "split": {"path": self.split_path, "sha256": self.split_sha256},
                "provenance": {"path": self.provenance_path, "sha256": self.provenance_sha256},
                "registry": {"path": self.registry_path, "sha256": self.registry_sha256},
            },
            "baseline": self.baseline.json(),
            "candidate": self.candidate.json(),
            "pretrain": self.pretrain.json(),
            "scope_proof": {
                "fold_fit_sources_only": True,
                "held_out_video_count": 1,
                "audit_disjoint": self.audit_disjoint,
                "audit_used": False,
                "test_used": self.test_used,
                "geff_read_during_inference": False,
            },
        }


@dataclass(frozen=True)
class HybridAttestation:
    """Fold-fit proof for alpha-node inference with one migration adapter.

    Unlike :class:`GateAttestation`, this contract has no learned baseline:
    frozen alpha is the baseline and the attested model may contribute only
    child-owned motion evidence to its existing sparse parent candidates.
    """

    fold: int
    held_out_dataset: str
    held_out_video_id: str
    held_out_group_id: str
    held_out_exact_identity_sha256: str
    held_out_image_shape: tuple[int, int, int, int]
    split_path: str
    split_sha256: str
    provenance_path: str
    provenance_sha256: str
    registry_path: str
    registry_sha256: str
    candidate: TrainOutputAttestation
    pretrain: PretrainAttestation
    audit_disjoint: bool = True
    test_used: bool = False

    def json(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "kind": "alpha-node-migration-motion-hybrid-attestation",
            "fold": self.fold,
            "held_out_dataset": self.held_out_dataset,
            "held_out_video_id": self.held_out_video_id,
            "held_out_group_id": self.held_out_group_id,
            "held_out_exact_identity_sha256": self.held_out_exact_identity_sha256,
            "held_out_image_shape": list(self.held_out_image_shape),
            "manifests": {
                "split": {"path": self.split_path, "sha256": self.split_sha256},
                "provenance": {"path": self.provenance_path, "sha256": self.provenance_sha256},
                "registry": {"path": self.registry_path, "sha256": self.registry_sha256},
            },
            "candidate": self.candidate.json(),
            "pretrain": self.pretrain.json(),
            "reranker_policy": {
                "alpha_nodes_frozen": True,
                "adapter_outputs_used": [
                    "child_backward_dz_um",
                    "child_backward_dy_um",
                    "child_backward_dx_um",
                    "child_log_variance",
                ],
                "adapter_center_logits_used": False,
                "learned_or_calibrated_rerank_weights": 0,
                "held_out_target_calibration_used": False,
            },
            "scope_proof": {
                "fold_fit_sources_only": True,
                "held_out_video_count": 1,
                "audit_disjoint": self.audit_disjoint,
                "audit_used": False,
                "test_used": self.test_used,
                "geff_read_during_inference": False,
            },
        }


@dataclass(frozen=True)
class FrameOutput:
    """One streaming frame output with child-owned association channels."""

    frame_index: int
    center_logit: np.ndarray
    association: np.ndarray
    association_valid: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise GateError(f"missing required JSON artifact: {source}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                GateError(f"nonfinite JSON constant {item} in {source}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid JSON artifact: {source}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON artifact must contain one object: {source}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GateError(f"{label} is not a SHA-256 digest")
    return value


def _fold_row(split: Mapping[str, Any], fold: int) -> Mapping[str, Any]:
    if isinstance(fold, bool) or fold not in (0, 1, 2):
        raise GateError("fold must be 0, 1, or 2")
    folds = split.get("folds")
    row: Any = folds.get(str(fold)) if isinstance(folds, Mapping) else None
    if row is None and isinstance(folds, list) and len(folds) == 3:
        row = folds[fold]
    if not isinstance(row, Mapping):
        raise GateError(f"split has no valid fold-{fold} record")
    return row


def _train_ids(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise GateError(f"{label} must be a list")
    result: list[str] = []
    for value in values:
        item = str(value)
        if not item.startswith("train:") or not item[6:] or ":" in item[6:]:
            raise GateError(f"{label} contains a non-train identity")
        result.append(item)
    if len(result) != len(set(result)):
        raise GateError(f"{label} contains duplicate identities")
    return tuple(result)


def select_held_out_video(split: Mapping[str, Any], fold: int, dataset: str) -> str:
    """Return one exact fold-held-out train identity and reject audit/test scope."""

    requested = str(dataset)
    if requested.startswith("test:"):
        raise GateError("held-out evidence must use a train video, never test")
    if requested.startswith("train:"):
        requested_id = requested
        requested = requested[6:]
    elif ":" in requested or not requested:
        raise GateError("held-out dataset must be one canonical train identity")
    else:
        requested_id = f"train:{requested}"
    held = set(_train_ids(_fold_row(split, fold).get("held_out_video_ids"), "held-out scope"))
    audit_obj = split.get("audit")
    if not isinstance(audit_obj, Mapping):
        raise GateError("split audit scope is missing")
    audit = set(_train_ids(audit_obj.get("video_ids"), "audit scope"))
    if held & audit:
        raise GateError("held-out and audit scopes overlap")
    if requested_id not in held or requested_id in audit:
        raise GateError(f"{requested_id} is not a fold-{fold} held-out video")
    return requested_id


def _registry_records(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = registry.get("records")
    if not isinstance(records, list):
        raise GateError("registry records are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise GateError("registry contains a non-object record")
        video_id = str(value.get("video_id"))
        if video_id in result:
            raise GateError(f"registry duplicates {video_id}")
        result[video_id] = value
    return result


def _source_scope(
    records: Any,
    *,
    label: str,
    fold_row: Mapping[str, Any],
    provenance_fold: Mapping[str, Any],
    registry_by_video: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(records, list) or not records:
        raise GateError(f"{label} has no selected fit sources")
    fit_groups = set(str(value) for value in fold_row.get("fit_group_ids", ()))
    fit_videos = set(_train_ids(fold_row.get("fit_video_ids"), "fold fit scope"))
    fit_sources = provenance_fold.get("fit_sources")
    calibration = (
        set(str(value) for value in fit_sources.get("calibration", ()))
        if isinstance(fit_sources, Mapping)
        else set()
    )
    groups: list[str] = []
    videos: list[str] = []
    for value in records:
        if not isinstance(value, Mapping):
            raise GateError(f"{label} fit source record is invalid")
        group = str(value.get("group_id"))
        video = str(value.get("video_id"))
        stem = str(value.get("stem"))
        registry_record = registry_by_video.get(video)
        if (
            group not in fit_groups
            or group not in calibration
            or video not in fit_videos
            or video != f"train:{stem}"
            or registry_record is None
            or str(registry_record.get("group_id")) != group
            or str(registry_record.get("dataset")) != stem
            or str(registry_record.get("split")) != "train"
        ):
            raise GateError(f"{label} fit source crosses or mismatches the fold-fit scope")
        groups.append(group)
        videos.append(video)
    if len(groups) != len(set(groups)) or len(videos) != len(set(videos)):
        raise GateError(f"{label} fit source identities are duplicated")
    return tuple(groups), tuple(videos)


def _attest_train_output(
    label: str,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    *,
    manifest_hashes: Mapping[str, str],
    fold: int,
    fold_row: Mapping[str, Any],
    provenance_fold: Mapping[str, Any],
    registry_by_video: Mapping[str, Mapping[str, Any]],
) -> TrainOutputAttestation:
    checkpoint = Path(checkpoint_path)
    metrics_source = Path(metrics_path)
    if not checkpoint.is_file():
        raise GateError(f"{label} checkpoint is missing")
    metrics = _strict_json(metrics_source)
    checkpoint_hash = sha256_file(checkpoint)
    if _sha(metrics.get("checkpoint_sha256"), f"{label} checkpoint_sha256") != checkpoint_hash:
        raise GateError(f"{label} checkpoint hash disagrees with train_ta metrics")
    identity = metrics.get("identity")
    if not isinstance(identity, Mapping):
        raise GateError(f"{label} train_ta identity is missing")
    if identity.get("fold") != fold or identity.get("scenario") != "universal":
        raise GateError(f"{label} fold or scenario identity is invalid")
    for name, expected in manifest_hashes.items():
        if identity.get(name) != expected:
            raise GateError(f"{label} {name} does not match the supplied manifest")
    completed = metrics.get("completed_steps")
    execution = metrics.get("execution")
    if (
        metrics.get("status") != "completed"
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed <= 0
        or not isinstance(execution, Mapping)
        or execution.get("max_steps") != completed
    ):
        raise GateError(f"{label} train_ta output is not a completed bounded run")
    groups, videos = _source_scope(
        identity.get("selected_sources"),
        label=label,
        fold_row=fold_row,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    if tuple(metrics.get("source_groups", ())) != groups or tuple(metrics.get("source_video_ids", ())) != videos:
        raise GateError(f"{label} metrics and identity fit source lists disagree")
    temporal_hash = identity.get("temporal_pretrain_sha256")
    if temporal_hash is not None:
        temporal_hash = _sha(temporal_hash, f"{label} temporal_pretrain_sha256")
    return TrainOutputAttestation(
        label=label,
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_bytes=checkpoint.stat().st_size,
        metrics_path=str(metrics_source.resolve()),
        metrics_sha256=sha256_file(metrics_source),
        fold=fold,
        completed_steps=completed,
        source_groups=groups,
        source_video_ids=videos,
        temporal_pretrain_sha256=temporal_hash,
        identity=dict(identity),
    )


def _attest_pretrain(
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    *,
    manifest_hashes: Mapping[str, str],
    fold: int,
    fold_row: Mapping[str, Any],
    provenance_fold: Mapping[str, Any],
    registry_by_video: Mapping[str, Mapping[str, Any]],
) -> PretrainAttestation:
    checkpoint = Path(checkpoint_path)
    metrics_source = Path(metrics_path)
    if not checkpoint.is_file():
        raise GateError("temporal pretrain checkpoint is missing")
    metrics = _strict_json(metrics_source)
    checkpoint_hash = sha256_file(checkpoint)
    checkpoint_row = metrics.get("checkpoint")
    identity = metrics.get("identity")
    if (
        not isinstance(checkpoint_row, Mapping)
        or checkpoint_row.get("sha256") != checkpoint_hash
        or not isinstance(identity, Mapping)
        or identity.get("fold") != fold
        or identity.get("objective") != PRETRAIN_OBJECTIVE
    ):
        raise GateError("temporal pretrain identity or checkpoint hash is invalid")
    for name, expected in manifest_hashes.items():
        if identity.get(name) != expected:
            raise GateError(f"temporal pretrain {name} does not match the supplied manifest")
    groups, videos = _source_scope(
        identity.get("selected_sources"),
        label="temporal pretrain",
        fold_row=fold_row,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    policy = identity.get("input_policy")
    required_policy = {
        "competition_train_images_only": True,
        "alpha_image_derived_pseudo_links_used": True,
        "labels_used": False,
        "tracks_used": False,
        "held_out_used": False,
        "audit_used": False,
        "test_used": False,
    }
    if not isinstance(policy, Mapping) or any(policy.get(key) is not value for key, value in required_policy.items()):
        raise GateError("temporal pretrain input policy does not prove image-only fold-fit scope")
    return PretrainAttestation(
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_bytes=checkpoint.stat().st_size,
        metrics_path=str(metrics_source.resolve()),
        metrics_sha256=sha256_file(metrics_source),
        source_groups=groups,
        source_video_ids=videos,
    )


def attest_gate_pair(
    *,
    baseline_checkpoint: str | Path,
    baseline_metrics: str | Path,
    candidate_checkpoint: str | Path,
    candidate_metrics: str | Path,
    pretrain_checkpoint: str | Path,
    pretrain_metrics: str | Path,
    split_manifest: str | Path,
    provenance_manifest: str | Path,
    registry_manifest: str | Path,
    fold: int,
    held_out_dataset: str,
) -> GateAttestation:
    """Attest an equal-budget release-vs-migration pair before image access."""

    split_path = Path(split_manifest)
    provenance_path = Path(provenance_manifest)
    registry_path = Path(registry_manifest)
    split = _strict_json(split_path)
    provenance = _strict_json(provenance_path)
    registry = _strict_json(registry_path)
    hashes = {
        "split_sha256": sha256_file(split_path),
        "provenance_sha256": sha256_file(provenance_path),
        "registry_sha256": sha256_file(registry_path),
    }
    held_out_video = select_held_out_video(split, fold, held_out_dataset)
    fold_record = _fold_row(split, fold)
    provenance_folds = provenance.get("folds")
    provenance_fold = provenance_folds.get(str(fold)) if isinstance(provenance_folds, Mapping) else None
    if not isinstance(provenance_fold, Mapping):
        raise GateError(f"provenance has no valid fold-{fold} record")
    registry_by_video = _registry_records(registry)
    held_record = registry_by_video.get(held_out_video)
    if (
        held_record is None
        or str(held_record.get("split")) != "train"
        or str(held_record.get("dataset")) != held_out_video[6:]
    ):
        raise GateError("held-out video does not resolve exactly in the registry")
    compact = held_record.get("compact_metadata")
    shape = compact.get("image_shape") if isinstance(compact, Mapping) else None
    if not isinstance(shape, list) or len(shape) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
        raise GateError("held-out registry image shape is invalid")
    exact_identity = _sha(held_record.get("exact_identity_sha256"), "held-out exact identity")

    baseline = _attest_train_output(
        "released_baseline",
        baseline_checkpoint,
        baseline_metrics,
        manifest_hashes=hashes,
        fold=fold,
        fold_row=fold_record,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    candidate = _attest_train_output(
        "migration_pretrained_adapter",
        candidate_checkpoint,
        candidate_metrics,
        manifest_hashes=hashes,
        fold=fold,
        fold_row=fold_record,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    pretrain = _attest_pretrain(
        pretrain_checkpoint,
        pretrain_metrics,
        manifest_hashes=hashes,
        fold=fold,
        fold_row=fold_record,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    if baseline.temporal_pretrain_sha256 is not None:
        raise GateError("released baseline unexpectedly has a temporal-pretrain parent")
    if candidate.temporal_pretrain_sha256 != pretrain.checkpoint_sha256:
        raise GateError("migration candidate does not bind the supplied temporal-pretrain checkpoint")
    if baseline.checkpoint_sha256 == candidate.checkpoint_sha256:
        raise GateError("baseline and migration candidate checkpoints are identical")
    if baseline.completed_steps != candidate.completed_steps:
        raise GateError("paired supervised runs do not have equal optimizer-step budgets")
    for field in COMPARISON_IDENTITY_FIELDS:
        if baseline.identity.get(field) != candidate.identity.get(field):
            raise GateError(f"paired supervised identity differs at {field}")
    audit_obj = split.get("audit")
    audit = set(_train_ids(audit_obj.get("video_ids"), "audit scope")) if isinstance(audit_obj, Mapping) else set()
    all_fit_sources = set(baseline.source_video_ids) | set(candidate.source_video_ids) | set(pretrain.source_video_ids)
    if held_out_video in all_fit_sources or audit.intersection(all_fit_sources | {held_out_video}):
        raise GateError("held-out, fit-source, and audit scopes are not disjoint")
    return GateAttestation(
        fold=fold,
        held_out_dataset=held_out_video[6:],
        held_out_video_id=held_out_video,
        held_out_group_id=str(held_record.get("group_id")),
        held_out_exact_identity_sha256=exact_identity,
        held_out_image_shape=tuple(int(value) for value in shape),
        split_path=str(split_path.resolve()),
        split_sha256=hashes["split_sha256"],
        provenance_path=str(provenance_path.resolve()),
        provenance_sha256=hashes["provenance_sha256"],
        registry_path=str(registry_path.resolve()),
        registry_sha256=hashes["registry_sha256"],
        baseline=baseline,
        candidate=candidate,
        pretrain=pretrain,
    )


def attest_hybrid_checkpoint(
    *,
    candidate_checkpoint: str | Path,
    candidate_metrics: str | Path,
    pretrain_checkpoint: str | Path,
    pretrain_metrics: str | Path,
    split_manifest: str | Path,
    provenance_manifest: str | Path,
    registry_manifest: str | Path,
    fold: int,
    held_out_dataset: str,
) -> HybridAttestation:
    """Attest one migration-pretrained adapter for a frozen-alpha rerank.

    All identity and split checks complete before callers may open held-out
    image voxels.  The target is never admitted to either training source set,
    and no calibration artifact or target-derived weight is accepted.
    """

    split_path = Path(split_manifest)
    provenance_path = Path(provenance_manifest)
    registry_path = Path(registry_manifest)
    split = _strict_json(split_path)
    provenance = _strict_json(provenance_path)
    registry = _strict_json(registry_path)
    hashes = {
        "split_sha256": sha256_file(split_path),
        "provenance_sha256": sha256_file(provenance_path),
        "registry_sha256": sha256_file(registry_path),
    }
    held_out_video = select_held_out_video(split, fold, held_out_dataset)
    fold_record = _fold_row(split, fold)
    provenance_folds = provenance.get("folds")
    provenance_fold = provenance_folds.get(str(fold)) if isinstance(provenance_folds, Mapping) else None
    if not isinstance(provenance_fold, Mapping):
        raise GateError(f"provenance has no valid fold-{fold} record")
    registry_by_video = _registry_records(registry)
    held_record = registry_by_video.get(held_out_video)
    if (
        held_record is None
        or str(held_record.get("split")) != "train"
        or str(held_record.get("dataset")) != held_out_video[6:]
    ):
        raise GateError("held-out video does not resolve exactly in the registry")
    compact = held_record.get("compact_metadata")
    shape = compact.get("image_shape") if isinstance(compact, Mapping) else None
    if not isinstance(shape, list) or len(shape) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise GateError("held-out registry image shape is invalid")
    exact_identity = _sha(held_record.get("exact_identity_sha256"), "held-out exact identity")
    candidate = _attest_train_output(
        "migration_pretrained_adapter",
        candidate_checkpoint,
        candidate_metrics,
        manifest_hashes=hashes,
        fold=fold,
        fold_row=fold_record,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    pretrain = _attest_pretrain(
        pretrain_checkpoint,
        pretrain_metrics,
        manifest_hashes=hashes,
        fold=fold,
        fold_row=fold_record,
        provenance_fold=provenance_fold,
        registry_by_video=registry_by_video,
    )
    if candidate.temporal_pretrain_sha256 != pretrain.checkpoint_sha256:
        raise GateError("migration candidate does not bind the supplied temporal-pretrain checkpoint")
    audit_obj = split.get("audit")
    audit = set(_train_ids(audit_obj.get("video_ids"), "audit scope")) if isinstance(audit_obj, Mapping) else set()
    fit_sources = set(candidate.source_video_ids) | set(pretrain.source_video_ids)
    if held_out_video in fit_sources or audit.intersection(fit_sources | {held_out_video}):
        raise GateError("held-out, fit-source, and audit scopes are not disjoint")
    return HybridAttestation(
        fold=fold,
        held_out_dataset=held_out_video[6:],
        held_out_video_id=held_out_video,
        held_out_group_id=str(held_record.get("group_id")),
        held_out_exact_identity_sha256=exact_identity,
        held_out_image_shape=tuple(int(value) for value in shape),
        split_path=str(split_path.resolve()),
        split_sha256=hashes["split_sha256"],
        provenance_path=str(provenance_path.resolve()),
        provenance_sha256=hashes["provenance_sha256"],
        registry_path=str(registry_path.resolve()),
        registry_sha256=hashes["registry_sha256"],
        candidate=candidate,
        pretrain=pretrain,
    )


def _pair_output(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 5 or array.shape[0] != 2 or array.shape[1] != 6 or min(array.shape) <= 0:
        raise GateError("adapter pair output must have shape (2,6,Z,Y,X)")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise GateError("adapter pair output must be finite floating point")
    return np.ascontiguousarray(array, dtype=np.float32)


def iter_sliding_frame_outputs(pair_outputs: Iterable[Any]) -> Iterator[FrameOutput]:
    """Stream frame outputs while keeping backward motion owned by the child.

    Interior detection logits are averaged across their two temporal contexts.
    Displacement and log-variance for frame ``t`` come only from the target
    output of window ``(t-1,t)``; the unsupervised source-side value is never
    mixed into association.
    """

    iterator = iter(pair_outputs)
    try:
        first = _pair_output(next(iterator))
    except StopIteration as exc:
        raise GateError("full-video inference requires at least one adjacent pair") from exc
    spatial = tuple(int(value) for value in first.shape[-3:])
    yield FrameOutput(0, first[0, 0].copy(), np.zeros((4, *spatial), dtype=np.float32), False)
    previous_target = first[1]
    frame_index = 1
    for raw in iterator:
        current = _pair_output(raw)
        if tuple(current.shape[-3:]) != spatial:
            raise GateError("adapter sliding outputs changed spatial shape")
        center = (previous_target[0] + current[0, 0]) * np.float32(0.5)
        yield FrameOutput(frame_index, center, previous_target[1:5].copy(), True)
        previous_target = current[1]
        frame_index += 1
    yield FrameOutput(frame_index, previous_target[0].copy(), previous_target[1:5].copy(), True)


def sparse_alpha_association(
    dataset: str,
    raw_nodes: Sequence[Mapping[str, Any]],
    scale_zyx: Sequence[float],
    *,
    edge_radius_um: float = 8.0,
    neighbor_cap: int = 4,
    edge_threshold: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, int]], dict[str, Any]]:
    """Associate adapter detections through bounded sparse motion candidates.

    Each child first queries around its learned backward location.  If that
    query yields no accepted candidate, the existing local geometric alpha
    route is retained as a per-child fallback.  Both routes use ``cKDTree``
    queries and the same total neighbor cap; no framewise ``N x N`` tensor is
    allocated.
    """

    from scipy.spatial import cKDTree

    from .solver import select_edges, validate_lineage_graph
    from .types import CandidateEdge, CandidateNode, RouteConfig

    if not isinstance(dataset, str) or not dataset or ":" in dataset:
        raise GateError("dataset must be one unprefixed train stem")
    if len(scale_zyx) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in scale_zyx):
        raise GateError("scale_zyx must contain three positive finite values")
    if not math.isfinite(edge_radius_um) or edge_radius_um <= 0:
        raise GateError("edge radius must be positive and finite")
    if isinstance(neighbor_cap, bool) or not isinstance(neighbor_cap, int) or neighbor_cap <= 0:
        raise GateError("neighbor cap must be a positive integer")
    if not math.isfinite(edge_threshold) or not 0 <= edge_threshold <= 1:
        raise GateError("edge threshold must be in [0,1]")
    scale = np.asarray(scale_zyx, dtype=np.float64)
    nodes: list[Any] = []
    raw_by_id: dict[int, Mapping[str, Any]] = {}
    for row in raw_nodes:
        try:
            node_id = int(row["node_id"])
            t = int(row["t"])
            coords = tuple(int(row[axis]) for axis in ("z", "y", "x"))
            score = float(row["score"])
            log_variance = float(row["log_variance"])
            backward = np.asarray(row["backward_um"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise GateError("raw adapter node schema is invalid") from exc
        if (
            node_id <= 0
            or node_id in raw_by_id
            or t < 0
            or any(value < 0 for value in coords)
            or not math.isfinite(score)
            or not math.isfinite(log_variance)
            or backward.shape != (3,)
            or not np.isfinite(backward).all()
        ):
            raise GateError("raw adapter node contains an invalid value")
        nodes.append(
            CandidateNode(
                dataset=dataset,
                node_id=node_id,
                t=t,
                z=coords[0],
                y=coords[1],
                x=coords[2],
                score=score,
                sources=("temporal_adapter",),
                radius_um=0.0,
                uncertainty=float(math.exp(max(-10.0, min(10.0, log_variance)))),
            )
        )
        raw_by_id[node_id] = row
    if not nodes:
        raise GateError("adapter inference emitted no nodes")
    times = sorted({int(node.t) for node in nodes})
    by_time = {time: [node for node in nodes if int(node.t) == time] for time in times}
    candidates: list[Any] = []
    fallback_pairs: set[tuple[int, int]] = set()
    peak_candidates = 0
    target_count = 0
    for source_time, target_time in zip(times[:-1], times[1:], strict=True):
        if target_time != source_time + 1:
            continue
        sources = by_time[source_time]
        targets = by_time[target_time]
        if not sources or not targets:
            continue
        source_physical = np.asarray([node.coord_zyx for node in sources], dtype=np.float64) * scale
        tree = cKDTree(source_physical)
        for target in targets:
            target_count += 1
            target_physical = np.asarray(target.coord_zyx, dtype=np.float64) * scale
            backward = np.asarray(raw_by_id[int(target.node_id)]["backward_um"], dtype=np.float64)

            def ranked(center: np.ndarray) -> list[tuple[float, float, int]]:
                values: list[tuple[float, float, int]] = []
                for index in tree.query_ball_point(center, r=edge_radius_um):
                    actual = float(np.linalg.norm(source_physical[int(index)] - target_physical))
                    if actual > edge_radius_um:
                        continue
                    residual = float(np.linalg.norm(source_physical[int(index)] - center))
                    values.append((residual, actual, int(index)))
                values.sort(key=lambda item: (item[0], item[1], int(sources[item[2]].node_id)))
                return values

            learned_center = target_physical + backward
            route = "learned_backward"
            rows = ranked(learned_center)
            accepted = [row for row in rows if 1.0 - row[0] / edge_radius_um >= edge_threshold]
            if not accepted:
                route = "alpha_geometric_fallback"
                learned_center = target_physical
                rows = ranked(target_physical)
                accepted = [row for row in rows if 1.0 - row[0] / edge_radius_um >= edge_threshold]
            accepted = accepted[:neighbor_cap]
            peak_candidates = max(peak_candidates, len(accepted))
            for residual, actual, index in accepted:
                probability = max(0.0, min(1.0, 1.0 - residual / edge_radius_um))
                source_id = int(sources[index].node_id)
                pair = (source_id, int(target.node_id))
                if route == "alpha_geometric_fallback":
                    fallback_pairs.add(pair)
                candidates.append(
                    CandidateEdge(
                        source_id=source_id,
                        target_id=int(target.node_id),
                        dt=1,
                        distance_um=actual,
                        cost=1.0 - probability,
                        score=probability,
                        features={
                            "motion_residual_um": residual,
                            "temporal_adapter_motion_score": probability,
                            "alpha_fallback": 1.0 if route == "alpha_geometric_fallback" else 0.0,
                        },
                    )
                )
    ordered = [by_time[time] for time in times]
    config = RouteConfig(
        label="temporal_adapter_sparse_alpha",
        detector_sources=("temporal_adapter",),
        threshold_quantiles=(),
        min_distance_um=0.0,
        fusion_radius_um=0.0,
        max_points_per_t=max((len(value) for value in ordered), default=1),
        max_link_um=edge_radius_um,
        max_neighbors=neighbor_cap,
        edge_cost_threshold=1.0 - edge_threshold,
        birth_cost=0.5,
        death_cost=0.5,
        node_score_weight=0.0,
        density_cost_weight=0.0,
        appearance_cost_weight=0.0,
        gap_motion_cost_weight=0.0,
        enable_divisions=False,
    )
    selected, forks = select_edges(ordered, candidates, tuple(float(value) for value in scale), config)
    graph = validate_lineage_graph(nodes, selected, allow_divisions=False)
    if forks or int(graph["num_divisions"]) != 0:
        raise GateError("edge-only sparse alpha association emitted a division")
    prediction_nodes = [
        {
            "dataset": dataset,
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": int(node.z),
            "y": int(node.y),
            "x": int(node.x),
        }
        for node in nodes
    ]
    prediction_edges = [edge.to_submission_edge() for edge in selected]
    selected_pairs = {(int(row["source_id"]), int(row["target_id"])) for row in prediction_edges}
    report = {
        **graph,
        "candidate_edges": len(candidates),
        "candidate_targets": target_count,
        "fallback_candidate_edges": len(fallback_pairs),
        "fallback_selected_edges": len(selected_pairs & fallback_pairs),
        "peak_candidates_per_target": peak_candidates,
        "neighbor_cap": neighbor_cap,
        "dense_nxn": False,
        "association": "child-backward-sparse-cKDTree-with-alpha-geometric-fallback",
    }
    return prediction_nodes, prediction_edges, report


def write_fragment(
    path: str | Path,
    dataset: str,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> str:
    """Write one validated train fragment for the unchanged frozen scorer."""

    destination = Path(path)
    if destination.exists():
        raise GateError(f"refusing to overwrite fragment: {destination}")
    try:
        graph = SubmissionGraph(
            dataset=dataset,
            nodes={
                int(row["node_id"]): SubmissionNode(
                    node_id=int(row["node_id"]),
                    t=int(row["t"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                )
                for row in nodes
            },
            edges=[SubmissionEdge(int(row["source_id"]), int(row["target_id"])) for row in edges],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise GateError("fragment graph rows are invalid") from exc
    if len(graph.nodes) != len(nodes) or any(str(row.get("dataset")) != dataset for row in nodes):
        raise GateError("fragment node identity is duplicated or mismatched")
    write_submission({dataset: graph}, destination)
    return sha256_file(destination)


def run_deterministic_smoke() -> dict[str, Any]:
    """Exercise sparse fallback and the official parity fixture without real data."""

    from .scorer_parity import run_scorer_parity

    raw = [
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0, "score": 0.9,
         "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
        {"node_id": 2, "t": 1, "z": 0, "y": 0, "x": 1, "score": 0.9,
         "backward_um": [0.0, 0.0, 99.0], "log_variance": 0.0},
    ]
    nodes, edges, association = sparse_alpha_association(
        "tiny", raw, (1.0, 1.0, 1.0), edge_radius_um=2.0, neighbor_cap=1, edge_threshold=0.2
    )
    parity = run_scorer_parity()
    payload: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "scope": "tiny_synthetic_no_competition_data",
        "official_parity_pass": bool(parity.get("overall_pass")),
        "association": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "candidate_edges": int(association["candidate_edges"]),
            "fallback_candidate_edges": int(association["fallback_candidate_edges"]),
            "fallback_preserved": int(association["fallback_selected_edges"]) == 1,
            "dense_nxn": bool(association["dense_nxn"]),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    payload["deterministic_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


__all__ = [
    "FrameOutput",
    "GATE_SCHEMA",
    "GateAttestation",
    "GateError",
    "HybridAttestation",
    "PretrainAttestation",
    "TrainOutputAttestation",
    "attest_gate_pair",
    "attest_hybrid_checkpoint",
    "iter_sliding_frame_outputs",
    "run_deterministic_smoke",
    "select_held_out_video",
    "sha256_file",
    "sparse_alpha_association",
    "write_fragment",
]

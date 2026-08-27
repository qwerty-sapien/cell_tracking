#!/usr/bin/env python3
"""Fit a compact sparse edge calibrator from fit-only GEFF labels.

The released V45 core, compact detector/motion head, and checkpoint pair scorer
remain frozen.  This script builds KD-tree-gated candidate rows once per whole
fit video, then trains only the 578-parameter residual calibrator.  Evaluation
and audit groups are rejected before image or label access.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_bh
from celltrack_alpha.bh_edge import ExposureMetadata, SparseEdgeConfig, SparseEdgeScorer, save_edge_artifact
from celltrack_alpha.bh_model import CacheIdentity, VideoInference, load_sparse_cache, load_v45_model
from celltrack_alpha.edge_data import EdgeDataError, EdgeExamples, examples_from_video, match_detections_to_gt
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.temporal_pretrain import TemporalPretrainError, load_alpha_pseudo_link_artifact
from celltrack_alpha.v45_fast import FastIdent, FrozenV45Fast, fast_readout, infer_fast, load_fast
from celltrack_baseline.rule_based import read_geff_arrays


SCHEMA = "celltrack-alpha.edgefit.v1"
DEFAULT_CHECKPOINT = ROOT / "reports/g02/v45f4.pt"
DEFAULT_METRICS = ROOT / "reports/g02/v45f4met.json"
DEFAULT_CHECKPOINT_SHA256 = "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473"
DEFAULT_METRICS_SHA256 = "1d67a099377dde39c04985982be1d427b3b4d3c7abcb6b4d0fa07d79c944b4b5"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
DEFAULT_PROVENANCE = ROOT / "reports/g02/fit_provenance_v1.json"
REGISTRY_ARTIFACT = "dataset_registry_v1"
REGISTRY_PROJECTION_ARTIFACT = "celltrack-alpha.edgefit-registry-projection.v1"


class EdgeFitError(ValueError):
    """Raised when an edge-calibrator fit would violate provenance or bounds."""


@dataclass(frozen=True)
class FitScope:
    fold: int
    records: tuple[Any, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    split_sha256: str
    registry_sha256: str
    provenance_sha256: str


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
        raise EdgeFitError(f"{label} SHA-256 does not match the pinned value")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EdgeFitError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise EdgeFitError(f"{label} must be an object")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise EdgeFitError(f"{label} must be a nonempty string list")
    result = tuple(sorted(value))
    if len(set(result)) != len(result):
        raise EdgeFitError(f"{label} contains duplicates")
    return result


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_edge_registry(path: Path) -> tuple[Mapping[str, Any], tuple[Any, ...], str]:
    """Load either the complete frozen registry or a hash-bound fit-only projection."""
    file_sha256 = _sha_file(path)
    registry = _load_json(path, expected_sha256=file_sha256, label="registry")
    artifact = registry.get("artifact")
    if artifact == REGISTRY_ARTIFACT:
        canonical_sha256 = file_sha256
    elif artifact == REGISTRY_PROJECTION_ARTIFACT:
        canonical_sha256 = registry.get("source_registry_sha256")
        if set(registry) != {"artifact", "source_registry_sha256", "parent_projection_sha256", "records"}:
            raise EdgeFitError("registry projection schema is not exact")
        if not _valid_sha256(canonical_sha256) or not _valid_sha256(registry.get("parent_projection_sha256")):
            raise EdgeFitError("registry projection does not bind valid source identities")
    else:
        raise EdgeFitError("registry is not the frozen registry or an edgefit projection")
    rows = registry.get("records")
    if not isinstance(rows, list) or not rows:
        raise EdgeFitError("registry has no records")
    if artifact == REGISTRY_PROJECTION_ARTIFACT:
        expected_record = {"split", "dataset", "video_id", "exact_identity_sha256", "compact_metadata", "group_id"}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != expected_record:
                raise EdgeFitError("registry projection record schema is not exact")
            metadata = row.get("compact_metadata")
            if not isinstance(metadata, Mapping) or set(metadata) != {"image_shape"}:
                raise EdgeFitError("registry projection image metadata schema is not exact")
    try:
        records = tuple(run_bh.RegistryRecord.from_mapping(row) for row in rows if isinstance(row, Mapping))
    except run_bh.DiscoveryError as exc:
        raise EdgeFitError("registry record is not a valid image identity") from exc
    if len(records) != len(rows) or len({record.video_id for record in records}) != len(records):
        raise EdgeFitError("registry runtime records are invalid or duplicated")
    return registry, records, str(canonical_sha256)


def verify_parent_manifest_binding(
    metrics: Mapping[str, Any], *, registry_sha256: str, split_sha256: str, provenance_sha256: str
) -> Mapping[str, Any]:
    """Reject a parent checkpoint whose training manifests differ from edgefit inputs."""
    identity = metrics.get("identity")
    if not isinstance(identity, Mapping):
        raise EdgeFitError("parent metrics do not contain a training identity")
    expected = {
        "registry_sha256": registry_sha256,
        "split_sha256": split_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise EdgeFitError("parent training identity does not bind the requested manifests")
    return identity


def _registry_groups(registry: Mapping[str, Any]) -> dict[str, str]:
    if registry.get("artifact") not in {REGISTRY_ARTIFACT, REGISTRY_PROJECTION_ARTIFACT} or not isinstance(registry.get("records"), list):
        raise EdgeFitError("registry is not the frozen dataset registry")
    result: dict[str, str] = {}
    for row in registry["records"]:
        if not isinstance(row, Mapping):
            raise EdgeFitError("registry contains a non-object record")
        video_id, group_id = row.get("video_id"), row.get("group_id")
        if not isinstance(video_id, str) or not isinstance(group_id, str) or not video_id or not group_id or video_id in result:
            raise EdgeFitError("registry video/group identity is invalid")
        result[video_id] = group_id
    return result


def _verify_parent(metrics: Mapping[str, Any], checkpoint_sha256: str, ident: FastIdent, fold: int) -> tuple[dict[str, str], ...]:
    if metrics.get("status") != "completed" or metrics.get("checkpoint_sha256") != checkpoint_sha256:
        raise EdgeFitError("compact parent metrics do not attest the loaded checkpoint")
    if ident.division_enabled or metrics.get("completed_steps") != ident.steps:
        raise EdgeFitError("compact parent division or step identity is invalid")
    identity = metrics.get("identity")
    if not isinstance(identity, Mapping) or identity.get("fold") != fold:
        raise EdgeFitError("compact parent belongs to another fold")
    if identity.get("target_contract_sha256") != ident.target_contract_sha256:
        raise EdgeFitError("compact parent target contract is invalid")
    if hashlib.sha256(_canonical(dict(identity))).hexdigest() != ident.train_sha256:
        raise EdgeFitError("compact parent training identity hash differs")
    sources = identity.get("selected_sources")
    if not isinstance(sources, list) or not sources:
        raise EdgeFitError("compact parent has no declared sources")
    result: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    seen_videos: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != {"group_id", "stem", "video_id"}:
            raise EdgeFitError("compact parent source schema is invalid")
        item = {key: str(row[key]) for key in ("group_id", "stem", "video_id")}
        if not all(item.values()) or item["video_id"] != f"train:{item['stem']}":
            raise EdgeFitError("compact parent source identity is invalid")
        if item["group_id"] in seen_groups or item["video_id"] in seen_videos:
            raise EdgeFitError("compact parent sources are not group unique")
        seen_groups.add(item["group_id"])
        seen_videos.add(item["video_id"])
        result.append(item)
    return tuple(result)


def select_registry_sources(
    registry: Mapping[str, Any], requested: Sequence[str]
) -> tuple[dict[str, str], ...]:
    """Resolve explicit train video/group pairs from the immutable registry."""

    wanted = tuple(str(item).strip() for item in requested)
    if len(wanted) < 2 or any(not item for item in wanted) or len(set(wanted)) != len(wanted):
        raise EdgeFitError("source manifest must name at least two unique videos")
    rows = registry.get("records")
    if not isinstance(rows, list):
        raise EdgeFitError("immutable registry has no source records")
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
            raise EdgeFitError("immutable registry train source identity is invalid")
        by_id[video_id] = {"video_id": video_id, "group_id": group_id, "stem": stem}
    normalized = tuple(item if item.startswith("train:") else f"train:{item}" for item in wanted)
    if not set(normalized).issubset(by_id):
        raise EdgeFitError("source manifest contains a video outside the train registry")
    return tuple(by_id[video_id] for video_id in normalized)


def load_source_manifest(
    path: Path, registry: Mapping[str, Any], *, fold: int
) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    """Load an attested fit-only cache manifest without expanding its data boundary."""

    sha256 = _sha_file(path)
    manifest = _load_json(path, expected_sha256=sha256, label="source manifest")
    boundary = manifest.get("data_boundary")
    video_ids = manifest.get("video_ids")
    if (
        manifest.get("schema") != "celltrack-alpha.fit-cache-corpus.v1"
        or manifest.get("status") != "validated"
        or manifest.get("fold") != fold
        or not isinstance(boundary, Mapping)
        or boundary.get("train_fit_only") is not True
        or any(
            boundary.get(key) is not False
            for key in (
                "held_out_used",
                "audit_used",
                "test_used",
                "competition_submission_performed",
            )
        )
        or not isinstance(video_ids, list)
        or manifest.get("video_count") != len(video_ids)
    ):
        raise EdgeFitError("source manifest is not an exact completed fit-only cache corpus")
    sources = select_registry_sources(registry, video_ids)
    return sources, {
        "path": str(path),
        "sha256": sha256,
        "video_count": len(sources),
        "data_boundary": dict(boundary),
    }


def build_scope(
    records: Sequence[Any],
    registry: Mapping[str, Any],
    split: Mapping[str, Any],
    parent_sources: Sequence[Mapping[str, str]],
    *,
    fold: int,
    split_sha256: str,
    registry_sha256: str,
    provenance_sha256: str,
    val_groups: int,
) -> FitScope:
    if type(fold) is not int or fold not in {0, 1, 2}:
        raise EdgeFitError("fold must be 0, 1, or 2")
    if type(val_groups) is not int or val_groups < 1:
        raise EdgeFitError("val_groups must be a positive integer")
    if split.get("artifact") != "grouped_3fold_audit20_v1" or split.get("dataset_registry_sha256") != registry_sha256:
        raise EdgeFitError("split does not bind the frozen registry")
    folds, audit = split.get("folds"), split.get("audit")
    if not isinstance(folds, Mapping) or not isinstance(audit, Mapping) or not isinstance(folds.get(str(fold)), Mapping):
        raise EdgeFitError("split does not contain the requested fold")
    fold_data = folds[str(fold)]
    fit_groups = set(_strings(fold_data.get("fit_group_ids"), "fit_group_ids"))
    held_groups = set(_strings(fold_data.get("held_out_group_ids"), "held_out_group_ids"))
    audit_groups = set(_strings(audit.get("group_ids"), "audit.group_ids"))
    if fit_groups & held_groups or fit_groups & audit_groups or held_groups & audit_groups:
        raise EdgeFitError("frozen split groups overlap")
    group_by_id = _registry_groups(registry)
    records_by_id = {str(record.video_id): record for record in records}
    if len(records_by_id) != len(records):
        raise EdgeFitError("runtime registry contains duplicate video IDs")
    source_map = {str(row["video_id"]): str(row["group_id"]) for row in parent_sources}
    source_ids = tuple(sorted(source_map))
    source_groups = tuple(sorted(source_map.values()))
    if len(set(source_ids)) != len(source_ids) or len(set(source_groups)) != len(source_groups):
        raise EdgeFitError("parent sources are not unique")
    if set(source_ids) - set(records_by_id) or set(source_ids) - set(group_by_id):
        raise EdgeFitError("parent source is missing from the runtime registry")
    if registry.get("artifact") == REGISTRY_PROJECTION_ARTIFACT and set(group_by_id) != set(source_ids):
        raise EdgeFitError("edgefit registry projection must contain exactly the parent sources")
    if any(group_by_id[video_id] != source_map[video_id] for video_id in source_ids):
        raise EdgeFitError("parent source group disagrees with frozen registry")
    if not set(source_groups).issubset(fit_groups) or set(source_groups) & (held_groups | audit_groups):
        raise EdgeFitError("parent sources are not fit-only for the requested fold")
    if val_groups >= len(source_groups):
        raise EdgeFitError("association holdback must leave at least one training group")
    ranked = sorted(source_groups, key=lambda group: hashlib.sha256(f"edgefit-v1:{fold}:{group}".encode("ascii")).hexdigest())
    validation = tuple(sorted(ranked[:val_groups]))
    training = tuple(sorted(set(source_groups) - set(validation)))
    group_to_id = {group: video_id for video_id, group in source_map.items()}
    return FitScope(
        fold=fold,
        records=tuple(records_by_id[video_id] for video_id in source_ids),
        train_ids=tuple(group_to_id[group] for group in training),
        val_ids=tuple(group_to_id[group] for group in validation),
        train_groups=training,
        val_groups=validation,
        split_sha256=split_sha256,
        registry_sha256=registry_sha256,
        provenance_sha256=provenance_sha256,
    )


def _read_directory_gt(root: Path, record: Any) -> dict[str, Any]:
    import zarr

    base = root / str(record.split) / f"{record.dataset}.geff"
    group = zarr.open_group(str(base), mode="r", zarr_format=3)
    nodes = {
        "node_id": np.asarray(group["nodes/ids"], dtype=np.int64),
        "t": np.asarray(group["nodes/props/t/values"], dtype=np.int64),
        "z": np.asarray(group["nodes/props/z/values"], dtype=np.int64),
        "y": np.asarray(group["nodes/props/y/values"], dtype=np.int64),
        "x": np.asarray(group["nodes/props/x/values"], dtype=np.int64),
    }
    edges = np.asarray(group["edges/ids"], dtype=np.int64)
    attrs = dict(group.attrs)
    estimated = attrs.get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")
    return {"attrs": attrs, "nodes": nodes, "edges": edges, "estimated_number_of_nodes": estimated}


def _read_gt(job: Any, handle: Any) -> dict[str, Any]:
    if job.input_kind == "zip":
        if handle.store is None:
            raise EdgeFitError("ZIP label reader has no backing store")
        return read_geff_arrays(handle.store, str(job.record.split), str(job.record.dataset))
    if job.input_kind == "directory":
        return _read_directory_gt(Path(job.input_path), job.record)
    raise EdgeFitError("unsupported local label source")


def _load_alpha_cache(
    artifact_root: Path,
    source: Mapping[str, str],
    exact_identity_sha256: str,
    expected_cache_identity: CacheIdentity,
) -> tuple[VideoInference, dict[str, Any]]:
    """Restore one immutable released-detector cache with full attestations."""

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
        raise EdgeFitError(f"immutable alpha cache rejected: {source['video_id']}: {exc}") from exc
    slug = f"train__{source['stem']}"
    meta_path = artifact_root / "cache" / f"{slug}.json"
    status_path = artifact_root / "status" / f"{slug}.json"
    meta = _load_json(
        meta_path,
        expected_sha256=_sha_file(meta_path),
        label=f"{source['video_id']} alpha cache metadata",
    )
    status = _load_json(
        status_path,
        expected_sha256=_sha_file(status_path),
        label=f"{source['video_id']} alpha status",
    )
    if meta.get("contract", {}).get("model") != asdict(expected_cache_identity):
        raise EdgeFitError("alpha cache does not match the loaded released detector identity")
    gate = status.get("run_contract", {}).get("graph_config", {}).get("candidate_gate")
    if gate != {
        "algorithm": "physical-cKDTree",
        "neighbor_cap": 8,
        "radius_um": 10.0,
        "scale_zyx": [1.625, 0.40625, 0.40625],
    }:
        raise EdgeFitError("alpha cache graph identity is not the frozen 10um cap-8 KD-tree")
    cache_path = artifact_root / "cache" / f"{slug}.bhc"
    cache = load_sparse_cache(cache_path, expected_identity=expected_cache_identity)
    if cache.edges:
        raise EdgeFitError("alpha cache unexpectedly persists learned edge rows")
    video = VideoInference(
        cache.detections,
        cache.contexts,
        pair_batch_size=cache.identity.pair_batch_size,
        view_batch_size=cache.identity.view_batch_size,
    )
    return video, {
        **artifact.provenance(),
        "meta_sha256": _sha_file(meta_path),
        "cache_identity": asdict(cache.identity),
        "candidate_gate": gate,
    }


def _combine(parts: Sequence[EdgeExamples]) -> EdgeExamples:
    if not parts:
        raise EdgeFitError("no edge examples were collected")
    nonempty = [part for part in parts if len(part.target_index)]
    width = parts[0].features.shape[1]
    if any(part.features.shape[1] != width for part in parts):
        raise EdgeFitError("edge example candidate widths disagree")
    totals = {
        name: sum(int(getattr(part, name)) for part in parts)
        for name in ("source_count", "candidate_count", "positive_count", "skipped_ambiguous_sources", "skipped_unknown_sources")
    }
    if not nonempty:
        return EdgeExamples(
            np.empty((0, width, 16), dtype=np.float32),
            np.empty((0, width), dtype=np.float32),
            np.empty((0, width), dtype=bool),
            np.empty((0,), dtype=np.int64),
            **totals,
        )
    return EdgeExamples(
        np.concatenate([part.features for part in nonempty]),
        np.concatenate([part.checkpoint_logits for part in nonempty]),
        np.concatenate([part.candidate_mask for part in nonempty]),
        np.concatenate([part.target_index for part in nonempty]),
        **totals,
    )


TensorExamples = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _loss(scorer: SparseEdgeScorer, examples: TensorExamples, rows: torch.Tensor) -> torch.Tensor:
    features, base, mask, targets = examples
    features = features[rows]
    base = base[rows]
    mask = mask[rows]
    targets = targets[rows]
    batch, width, feature_count = features.shape
    if feature_count != 16 or width != scorer.config.neighbor_cap:
        raise EdgeFitError("edge example shape disagrees with scorer config")
    residual = scorer.calibrator(features.reshape(batch * width, feature_count)).reshape(batch, width)
    logits = (base + residual).masked_fill(~mask, float("-inf"))
    dummy = scorer.calibrator.dummy_residual + float(scorer.config.dummy_logit)
    full_logits = torch.cat((logits, dummy.expand(batch, 1)), dim=1)
    return F.cross_entropy(full_logits, targets)


def _mean_loss(scorer: SparseEdgeScorer, examples: TensorExamples, batch_size: int) -> float:
    total = int(examples[3].shape[0])
    if total == 0:
        return float("inf")
    scorer.train(False)
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            rows = torch.arange(start, min(start + batch_size, total), device=scorer.calibrator.output.weight.device)
            values.append(float(_loss(scorer, examples, rows).item()))
    return float(np.mean(values))


def _fit(scorer: SparseEdgeScorer, train: EdgeExamples, validation: EdgeExamples, *, seed: int, epochs: int, max_steps: int, batch_size: int, lr: float, weight_decay: float) -> dict[str, Any]:
    if len(train.target_index) < 2 or len(validation.target_index) < 1:
        raise EdgeFitError("association train/validation examples are insufficient")
    if any(type(value) is not int or value <= 0 for value in (epochs, max_steps, batch_size)):
        raise EdgeFitError("epochs, max_steps, and batch_size must be positive")
    if not np.isfinite(lr) or lr <= 0.0 or not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise EdgeFitError("optimizer values are invalid")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.AdamW(scorer.calibrator.parameters(), lr=lr, weight_decay=weight_decay)
    device = scorer.calibrator.output.weight.device
    train_tensor = train.as_tensors(device)
    validation_tensor = validation.as_tensors(device)
    best = {name: value.detach().cpu().clone() for name, value in scorer.calibrator.state_dict().items()}
    best_loss = _mean_loss(scorer, validation_tensor, batch_size)
    history = [{"epoch": 0, "train_loss": None, "validation_loss": best_loss, "improved": True}]
    steps = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        scorer.train(True)
        order = torch.randperm(len(train.target_index), device=device)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            if steps >= max_steps:
                break
            rows = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(scorer, train_tensor, rows)
            if not torch.isfinite(loss):
                raise EdgeFitError("association loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.calibrator.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            steps += 1
        validation_loss = _mean_loss(scorer, validation_tensor, batch_size)
        improved = validation_loss < best_loss - 1.0e-6
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "validation_loss": validation_loss,
            "improved": improved,
        })
        if improved:
            best_loss = validation_loss
            best = {name: value.detach().cpu().clone() for name, value in scorer.calibrator.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if steps >= max_steps or stale >= 5:
            break
    scorer.calibrator.load_state_dict(best, strict=True)
    scorer.train(False)
    return {"steps": steps, "best_validation_loss": best_loss, "epochs": history}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise EdgeFitError(f"refusing to overwrite output: {path}")
    run_bh.atomic_create_json(path, value)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", choices=(0, 1, 2), type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--support-root", type=Path, default=ROOT / "external/Tracking_Support")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--metrics-sha256", default=DEFAULT_METRICS_SHA256)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Completed fit-only cache-corpus manifest used to expand beyond parent fit sources.",
    )
    parser.add_argument(
        "--alpha-artifact-root",
        type=Path,
        help="Immutable released-detector cache root paired with --source-manifest.",
    )
    parser.add_argument("--val-groups", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise EdgeFitError("out-dir must not already exist")
    if (args.source_manifest is None) != (args.alpha_artifact_root is None):
        raise EdgeFitError("--source-manifest and --alpha-artifact-root must be supplied together")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise EdgeFitError("CUDA was requested but is unavailable")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    split_sha = _sha_file(args.split)
    provenance_sha = _sha_file(args.provenance)
    split = _load_json(args.split, expected_sha256=split_sha, label="split")
    registry, registry_records, registry_sha = load_edge_registry(args.registry)
    provenance = _load_json(args.provenance, expected_sha256=provenance_sha, label="provenance")
    if provenance.get("grouped_split_sha256") != split_sha or provenance.get("related_manifest_hashes", {}).get("dataset_registry_v1.json") != registry_sha:
        raise EdgeFitError("fit provenance does not bind the requested split and registry")
    metrics = _load_json(args.metrics, expected_sha256=args.metrics_sha256, label="parent metrics")
    parent_identity = verify_parent_manifest_binding(
        metrics, registry_sha256=registry_sha, split_sha256=split_sha, provenance_sha256=provenance_sha
    )
    core, bundle = load_v45_model(args.support_root, device=args.device)
    model = FrozenV45Fast(core).to(args.device).eval()
    ident = load_fast(args.checkpoint, model, bundle, expected_sha256=args.checkpoint_sha256, target_sha256=TARGET_CONTRACT_SHA256)
    parent_sources = _verify_parent(metrics, args.checkpoint_sha256, ident, args.fold)
    source_manifest = None
    fit_sources = parent_sources
    if args.source_manifest is not None:
        fit_sources, source_manifest = load_source_manifest(
            args.source_manifest, registry, fold=args.fold
        )
    if registry.get("artifact") == REGISTRY_PROJECTION_ARTIFACT:
        if registry.get("parent_projection_sha256") != parent_identity.get("record_projection_sha256"):
            raise EdgeFitError("edgefit registry projection is not bound to the parent training projection")
    scope = build_scope(
        registry_records,
        registry,
        split,
        fit_sources,
        fold=args.fold,
        split_sha256=split_sha,
        registry_sha256=registry_sha,
        provenance_sha256=provenance_sha,
        val_groups=args.val_groups,
    )
    auxiliary = fast_readout(model, bundle, ident, checkpoint_sha256=args.checkpoint_sha256)
    scorer = SparseEdgeScorer(
        model.core,
        config=SparseEdgeConfig(),
        pretrained_weight_sha256=bundle.hashes.weight_sha256,
        auxiliary=auxiliary,
    )
    base = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "parent": {"checkpoint_sha256": args.checkpoint_sha256, "identity": ident.as_dict(), "metrics_sha256": args.metrics_sha256},
        "source_manifest": source_manifest,
        "scope": {
            "fold": scope.fold,
            "train_video_ids": list(scope.train_ids),
            "validation_video_ids": list(scope.val_ids),
            "train_group_ids": list(scope.train_groups),
            "validation_group_ids": list(scope.val_groups),
            "split_sha256": scope.split_sha256,
            "registry_sha256": scope.registry_sha256,
            "provenance_sha256": scope.provenance_sha256,
            "audit_used": False,
            "held_out_used": False,
            "test_used": False,
        },
        "model": {
            "association": "sparse-kdtree-released-pair-plus-compact-motion",
            "detector_execution": (
                "immutable-released-exact-sequential-D4-cache"
                if args.alpha_artifact_root is not None
                else "fresh-one-pass-fast-inference"
            ),
            "trainable_parameter_count": scorer.trainable_parameter_count,
            "division_enabled": False,
            "dense_nxn_association": False,
            "config": scorer.config.as_dict(),
            "architecture_sha256": scorer.architecture_sha256,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.dry_run:
        _atomic_json(args.out_dir / "report.json", base)
        print(json.dumps(base, ensure_ascii=True, sort_keys=True))
        return 0
    jobs = {job.identity: job for job in run_bh.discover_jobs(args.input, scope.records)}
    source_by_id = {str(row["video_id"]): row for row in fit_sources}
    expected_cache_identity = (
        CacheIdentity.from_artifact(bundle, pair_batch_size=1, view_batch_size=1)
        if args.alpha_artifact_root is not None
        else None
    )
    examples_by_id: dict[str, EdgeExamples] = {}
    video_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for video_id in (*scope.train_ids, *scope.val_ids):
        job = jobs.get(video_id)
        if job is None:
            raise EdgeFitError(f"fit source was not discovered: {video_id}")
        handle = run_bh.open_video(job)
        try:
            q_low, q_high = run_bh.read_quantile_attrs(handle.attrs)
            cache_provenance = None
            if args.alpha_artifact_root is not None:
                if expected_cache_identity is None:
                    raise EdgeFitError("cached execution lacks its released-detector identity")
                fast_video, cache_provenance = _load_alpha_cache(
                    args.alpha_artifact_root,
                    source_by_id[video_id],
                    str(job.record.exact_identity_sha256),
                    expected_cache_identity,
                )
            else:
                frames = run_bh.read_all_frames(handle.array, int(job.record.image_shape[0]), None)
                fast_video = infer_fast(model, frames, q_low=q_low, q_high=q_high, device=args.device).video
            gt = _read_gt(job, handle)
            matches, gt_edges = match_detections_to_gt(fast_video.detections, gt)
            examples = examples_from_video(scorer, fast_video, matches, gt_edges)
            examples_by_id[video_id] = examples
            video_reports.append({
                "video_id": video_id,
                "group_id": str(source_by_id[video_id]["group_id"]),
                "matched_detections": sum(value is not None for value in matches.values()),
                "detections": len(matches),
                "examples": len(examples.target_index),
                "candidate_rows": examples.candidate_count,
                "skipped_ambiguous_sources": examples.skipped_ambiguous_sources,
                "skipped_unknown_sources": examples.skipped_unknown_sources,
                "alpha_cache": cache_provenance,
            })
        finally:
            handle.close()
    train = _combine([examples_by_id[video_id] for video_id in scope.train_ids])
    validation = _combine([examples_by_id[video_id] for video_id in scope.val_ids])
    result = _fit(
        scorer,
        train,
        validation,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    exposure = ExposureMetadata(
        fold=args.fold,
        source_groups=tuple(sorted((*scope.train_groups, *scope.val_groups))),
        source_video_ids=tuple(sorted((*scope.train_ids, *scope.val_ids))),
        split_manifest_sha256=scope.split_sha256,
        provenance_manifest_sha256=scope.provenance_sha256,
        audit_exposed=False,
        test_exposed=False,
    )
    artifact = args.out_dir / "edge.json"
    artifact_identity = save_edge_artifact(artifact, scorer, exposure)
    report = {
        **base,
        "status": "completed",
        "artifact": {"path": str(artifact), "sha256": _sha_file(artifact), "identity": asdict(artifact_identity)},
        "training": result,
        "examples": {
            "train": {"groups": len(scope.train_groups), "examples": len(train.target_index), "candidates": train.candidate_count},
            "validation": {"groups": len(scope.val_groups), "examples": len(validation.target_index), "candidates": validation.candidate_count},
            "videos": video_reports,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.out_dir / "report.json", report)
    print(json.dumps({"artifact": report["artifact"], "training": report["training"], "examples": report["examples"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

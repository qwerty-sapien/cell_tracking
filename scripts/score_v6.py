#!/usr/bin/env python3
"""Run attested V6 inference and official scoring on one fold-safe train scope."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OFFICIAL = ROOT / "external" / "kaggle-cell-tracking-competition" / "src"
V6_SHA256 = "712646e9e263da018239dc46e927905708d31405c0b1774d20a260d85f480987"
V6_BYTES = 6_375_639
PARENT_SHA256 = "347915de9c33883cb2ee69832a8e4552c88b1ec692d0fbfe956422467d3d4235"
PARENT_BYTES = 8_357_783
PARENT_CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
MET_SHA256 = "fb9c8cf60fc5bff69ddb2500bc2e6206b46cb83af2c4281aedb7e80f8b58bc68"
SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_CKPT = ROOT / "reports/g02/t03-v6.pt"
DEFAULT_MET = ROOT / "reports/g02/met-v6.json"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_PRED = ROOT / "reports/g02/v6-pred.json"
DEFAULT_SCORE = ROOT / "reports/g02/v6-score.json"
DEFAULT_PARENT = Path("/private/tmp/ckpt-v1/edge_predictor_best.pth")
DEFAULT_PARENT_CONFIG = Path("/private/tmp/ckpt-v1/config.json")
DEFAULT_TORCH_PY = Path("/usr/local/bin/python3.13")
DEFAULT_TORCH_PATH = Path("/private/tmp/torchlib")
DOWN_SAMPLE = (1, 4, 4)
F0_RUNG_2 = ("44b6_24264f12", "6bba_0c7fa718")
ARTIFACT_KEYS = frozenset({"schema_version", "artifact", "model_id", "checkpoint", "metrics", "fold", "split_sha256"})
CHECKPOINT_KEYS = frozenset({"file", "sha256", "bytes"})
METRICS_KEYS = frozenset({"file", "sha256"})


class V6Error(ValueError):
    """Raised before an unsafe scope read or incompatible V6 restore."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise V6Error(f"identity mismatch for {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V6Error(f"invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise V6Error(f"expected an object in {path}")
    return value


def _safe_model_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,11}", value):
        raise V6Error("model_id must be a short lowercase artifact label")
    if value == "parent":
        raise V6Error("model_id is reserved")
    return value


def _artifact_file(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise V6Error(f"artifact {label} file must be a basename")
    return base / value


def checkpoint_artifact(path: Path | None) -> dict[str, Any]:
    """Load either the legacy V6 identity or a self-contained scored artifact."""

    if path is None:
        metrics = _exact_json(DEFAULT_MET, MET_SHA256)
        record = {
            "model_id": "v6",
            "checkpoint": DEFAULT_CKPT,
            "checkpoint_sha256": V6_SHA256,
            "checkpoint_bytes": V6_BYTES,
            "metrics": metrics,
            "metrics_sha256": MET_SHA256,
            "fold": 0,
            "artifact_path": None,
            "artifact_sha256": None,
        }
    else:
        if not path.is_file():
            raise V6Error("checkpoint artifact is missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V6Error("checkpoint artifact is not valid JSON") from exc
        if not isinstance(payload, Mapping) or set(payload) != ARTIFACT_KEYS:
            raise V6Error("checkpoint artifact schema drift")
        checkpoint_row, metrics_row = payload.get("checkpoint"), payload.get("metrics")
        if (
            payload.get("schema_version") != 1
            or payload.get("artifact") != "checkpoint-score-artifact-v1"
            or not isinstance(checkpoint_row, Mapping)
            or set(checkpoint_row) != CHECKPOINT_KEYS
            or not isinstance(metrics_row, Mapping)
            or set(metrics_row) != METRICS_KEYS
            or payload.get("split_sha256") != SPLIT_SHA256
            or isinstance(payload.get("fold"), bool)
            or payload.get("fold") not in (0, 1, 2)
        ):
            raise V6Error("checkpoint artifact values are invalid")
        model_id = _safe_model_id(payload.get("model_id"))
        checkpoint = _artifact_file(path.parent, checkpoint_row.get("file"), "checkpoint")
        metrics_path = _artifact_file(path.parent, metrics_row.get("file"), "metrics")
        checkpoint_sha = checkpoint_row.get("sha256")
        checkpoint_bytes = checkpoint_row.get("bytes")
        metrics_sha = metrics_row.get("sha256")
        if (
            not isinstance(checkpoint_sha, str)
            or not isinstance(checkpoint_bytes, int)
            or checkpoint_bytes <= 0
            or not isinstance(metrics_sha, str)
            or len(checkpoint_sha) != 64
            or len(metrics_sha) != 64
            or not checkpoint.is_file()
            or checkpoint.stat().st_size != checkpoint_bytes
            or sha256_file(checkpoint) != checkpoint_sha
        ):
            raise V6Error("checkpoint artifact checkpoint identity mismatch")
        metrics = _exact_json(metrics_path, metrics_sha)
        record = {
            "model_id": model_id,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "metrics": metrics,
            "metrics_sha256": metrics_sha,
            "fold": int(payload["fold"]),
            "artifact_path": path,
            "artifact_sha256": sha256_file(path),
        }
    identity = record["metrics"].get("identity")
    if (
        record["metrics"].get("checkpoint_sha256") != record["checkpoint_sha256"]
        or record["metrics"].get("status") != "completed"
        or not isinstance(identity, Mapping)
        or identity.get("fold") != record["fold"]
    ):
        raise V6Error("checkpoint metrics do not attest the artifact completion")
    return record


def selected_artifact(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact is not None and (args.checkpoint != DEFAULT_CKPT or args.metrics != DEFAULT_MET):
        raise V6Error("artifact mode does not accept checkpoint or metrics overrides")
    return checkpoint_artifact(args.artifact)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_json_safe(value), allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _train_stems(values: Sequence[Any], label: str) -> tuple[str, ...]:
    stems: list[str] = []
    for value in values:
        item = str(value)
        if not item.startswith("train:") or ":" in item[6:]:
            raise V6Error(f"{label} contains a non-train video identity")
        stems.append(item[6:])
    if len(stems) != len(set(stems)):
        raise V6Error(f"{label} contains duplicate video identities")
    return tuple(stems)


def fold_scope(
    split: Mapping[str, Any], metrics: Mapping[str, Any], scope: str, requested: Sequence[str]
) -> tuple[int, tuple[str, ...], dict[str, Any]]:
    """Validate checkpoint sources and return exact fit or held-out-CV membership."""

    identity = metrics.get("identity")
    if not isinstance(identity, Mapping):
        raise V6Error("metrics identity is missing")
    fold = identity.get("fold")
    if isinstance(fold, bool) or fold not in (0, 1, 2):
        raise V6Error("metrics fold is invalid")
    folds = split.get("folds")
    fold_row = folds.get(str(fold)) if isinstance(folds, Mapping) else None
    audit = split.get("audit")
    if not isinstance(fold_row, Mapping) or not isinstance(audit, Mapping):
        raise V6Error("split fold or audit schema is invalid")
    fit = set(_train_stems(fold_row.get("fit_video_ids", ()), "fit scope"))
    held = set(_train_stems(fold_row.get("held_out_video_ids", ()), "CV scope"))
    audit_stems = set(_train_stems(audit.get("video_ids", ()), "audit scope"))
    if fit & held or fit & audit_stems or held & audit_stems or len(audit_stems) != 20:
        raise V6Error("split scopes overlap or audit-20 membership drifted")

    selected = identity.get("selected_sources")
    if not isinstance(selected, list) or not selected:
        raise V6Error("checkpoint metrics have no fit-source identities")
    source_stems = {str(row.get("stem")) for row in selected if isinstance(row, Mapping)}
    source_video_ids = {str(row.get("video_id")) for row in selected if isinstance(row, Mapping)}
    if len(source_stems) != len(selected) or source_video_ids != {f"train:{stem}" for stem in source_stems}:
        raise V6Error("checkpoint metrics selected-source schema drift")
    if not source_stems <= fit or source_stems & held or source_stems & audit_stems:
        raise V6Error("checkpoint training sources cross a fold or audit boundary")

    allowed = fit if scope == "fit" else held
    datasets = tuple(str(value) for value in requested)
    if not datasets or len(datasets) != len(set(datasets)):
        raise V6Error("at least one unique dataset must be requested")
    invalid = sorted(set(datasets) - allowed)
    if invalid:
        raise V6Error(f"datasets are outside fold-{fold} {scope} scope: {invalid}")
    if set(datasets) & audit_stems:
        raise V6Error("audit-20 access is prohibited")
    proof = {
        "fold": int(fold),
        "scope": scope,
        "requested_video_ids": [f"train:{value}" for value in datasets],
        "checkpoint_fit_video_ids": sorted(source_video_ids),
        "checkpoint_sources_in_fit": True,
        "held_out_disjoint": True,
        "audit_20_disjoint": True,
        "test_used": False,
    }
    return int(fold), datasets, proof


def _first_gt_start(gt: Mapping[str, Any], total: int, length: int) -> int:
    times = sorted(set(int(value) for value in gt["nodes"]["t"]))
    if not times:
        return 0
    return max(0, min(times[0], max(0, total - length)))


def _crop_gt(gt: Mapping[str, Any], start: int, end: int, total: int) -> dict[str, Any]:
    import numpy as np

    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept = {int(value) for value in nodes["node_id"][mask]}
    edges = np.asarray(
        [pair for pair in gt["edges"] if int(pair[0]) in kept and int(pair[1]) in kept],
        dtype=np.int64,
    )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    try:
        estimated = float(gt.get("estimated_number_of_nodes")) * (end - start) / max(1, total)
    except (TypeError, ValueError):
        estimated = float("nan")
    return {
        "attrs": gt.get("attrs", {}),
        "nodes": {key: np.asarray(value)[mask] for key, value in nodes.items()},
        "edges": edges,
        "estimated_number_of_nodes": estimated,
    }


def _quantile(attrs: Mapping[str, Any], key: str) -> float:
    try:
        value = float(attrs["image_statistics"]["quantiles"][key])
    except (KeyError, TypeError, ValueError) as exc:
        raise V6Error(f"missing image quantile {key}") from exc
    if not math.isfinite(value):
        raise V6Error(f"nonfinite image quantile {key}")
    return value


def _load_adapter_module() -> ModuleType:
    package = ModuleType("celltrack_alpha")
    package.__path__ = [str(SRC / "celltrack_alpha")]  # type: ignore[attr-defined]
    sys.modules["celltrack_alpha"] = package
    name = "celltrack_alpha.t_adapter"
    spec = importlib.util.spec_from_file_location(name, SRC / "celltrack_alpha/t_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load t_adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pool_kernel(distance_um: float, voxel_um: Sequence[float]) -> tuple[int, int, int]:
    out: list[int] = []
    for scale in voxel_um:
        value = max(1, round(distance_um / float(scale)))
        out.append(value if value % 2 else value + 1)
    return tuple(out)  # type: ignore[return-value]


def _infer_job(job_path: Path) -> int:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D

    job = json.loads(job_path.read_text(encoding="utf-8"))
    model_id = _safe_model_id(job.get("model_id"))
    adapter = _load_adapter_module()
    v6_backbone = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
    v6_model, restored = adapter.load_trained_model(
        v6_backbone,
        job["checkpoint"],
        expected_sha256=job["checkpoint_sha256"],
        expected_size=job["checkpoint_bytes"],
        expected_fold=int(job["fold"]),
    )
    device = torch.device(str(job["device"]))
    v6_model = v6_model.to(device).eval()

    parent_backbone = None
    parent_detector = None
    parent_restore: dict[str, Any] | None = None
    if bool(job.get("compare_parent")):
        parent_backbone = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
        parent_detector, core = adapter.load_released_core(
            parent_backbone, job["parent"], job["parent_config"]
        )
        parent_backbone = parent_backbone.to(device).eval()
        parent_detector = parent_detector.to(device).eval()
        parent_restore = {
            "checkpoint_sha256": adapter.WEIGHTS_SHA256,
            "config_sha256": adapter.CONFIG_SHA256,
            "released_core": asdict(core),
            "released_transformer_discarded": True,
            "motion_head": "zero_motion",
        }

    def v6_forward(pair: torch.Tensor) -> torch.Tensor:
        return v6_model(pair, scenario="universal")[0]

    def parent_forward(pair: torch.Tensor) -> torch.Tensor:
        if parent_backbone is None or parent_detector is None:
            raise RuntimeError("parent model was not restored")
        features = parent_backbone(pair)
        _, steps, channels, z, y, x = features.shape
        centers = parent_detector(features.reshape(steps, channels, z, y, x))
        return torch.cat((centers, centers.new_zeros((steps, 5, z, y, x))), dim=1)

    def predict(forward: Any, model_name: str) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for item in job["datasets"]:
            frames = np.load(item["frames"], allow_pickle=False)
            if frames.ndim != 4 or frames.shape[0] < 2 or not np.isfinite(frames).all():
                raise V6Error("inference frames must be finite (T,Z,Y,X) with T>=2")
            count = int(frames.shape[0])
            sums: list[torch.Tensor | None] = [None] * count
            seen = [0] * count
            for start in range(count - 1):
                pair = torch.from_numpy(frames[start:start + 2]).unsqueeze(0).unsqueeze(2).to(device)
                output = forward(pair).detach().cpu()
                if tuple(output.shape[:2]) != (2, 6):
                    raise RuntimeError(f"{model_name} output schema drift")
                for offset in (0, 1):
                    current = output[offset]
                    sums[start + offset] = current.clone() if sums[start + offset] is None else sums[start + offset] + current
                    seen[start + offset] += 1
            averaged = [value / seen[index] for index, value in enumerate(sums) if value is not None]
            if len(averaged) != count or any(value < 1 for value in seen):
                raise RuntimeError("sliding inference failed to cover every frame")
            kernel = _pool_kernel(float(job["nms_um"]), item["model_voxel_um"])
            pad = tuple(value // 2 for value in kernel)
            nodes: list[dict[str, Any]] = []
            node_id = 1
            frame_counts: list[int] = []
            for offset, output in enumerate(averaged):
                logits = output[0:1].unsqueeze(0)
                pooled = functional.max_pool3d(logits, kernel, stride=1, padding=pad)
                peak = (logits == pooled) & (torch.sigmoid(logits) >= float(job["det_threshold"]))
                coords = torch.nonzero(peak[0, 0]).cpu().numpy()
                scores = torch.sigmoid(logits[0, 0][peak[0, 0]]).cpu().numpy()
                if len(scores) > int(job["max_points"]):
                    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores))[: int(job["max_points"])]
                    coords, scores = coords[order], scores[order]
                else:
                    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores))
                    coords, scores = coords[order], scores[order]
                frame_counts.append(int(len(coords)))
                for coord, score in zip(coords, scores, strict=True):
                    z, y, x = (int(value) for value in coord)
                    backward = [float(value) for value in output[1:4, z, y, x]]
                    log_variance = float(output[4, z, y, x])
                    division_probability = float(torch.sigmoid(output[5, z, y, x]))
                    if not all(math.isfinite(value) for value in (*backward, log_variance, division_probability)):
                        raise RuntimeError(f"nonfinite {model_name} node output")
                    downsample = item["downsample"]
                    nodes.append(
                        {
                            "node_id": node_id,
                            "t": int(item["start"]) + offset,
                            "z": z * int(downsample[0]),
                            "y": y * int(downsample[1]),
                            "x": x * int(downsample[2]),
                            "score": float(score),
                            "backward_um": backward,
                            "log_variance": log_variance,
                            "division_probability": division_probability,
                        }
                    )
                    node_id += 1
            datasets.append(
                {"dataset": item["dataset"], "model": model_name, "nodes": nodes, "frame_counts": frame_counts}
            )
        return datasets

    with torch.inference_mode():
        predictions = {model_id: predict(v6_forward, model_id)}
        restores: dict[str, Any] = {model_id: asdict(restored)}
        if parent_restore is not None:
            predictions["parent"] = predict(parent_forward, "parent")
            restores["parent"] = parent_restore
    payload = {"schema_version": 1, "restores": restores, "predictions": predictions}
    _atomic_json(Path(job["output"]), payload)
    return 0


def _link_graph(
    dataset: str,
    raw_nodes: Sequence[Mapping[str, Any]],
    scale_zyx: Sequence[float],
    edge_radius_um: float,
    neighbor_cap: int,
    edge_threshold: float,
    model_name: str = "v6",
) -> tuple[list[dict[str, Any]], list[dict[str, int]], dict[str, Any]]:
    import numpy as np
    from scipy.spatial import cKDTree

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from celltrack_alpha.solver import select_edges, validate_lineage_graph
    from celltrack_alpha.types import CandidateEdge, CandidateNode, RouteConfig

    scale = np.asarray(scale_zyx, dtype=np.float64)
    nodes = [
        CandidateNode(
            dataset=dataset,
            node_id=int(row["node_id"]),
            t=int(row["t"]),
            z=int(row["z"]),
            y=int(row["y"]),
            x=int(row["x"]),
            score=float(row["score"]),
            sources=(model_name,),
            radius_um=0.0,
            uncertainty=float(math.exp(max(-10.0, min(10.0, float(row["log_variance"]))))),
        )
        for row in raw_nodes
    ]
    times = sorted({node.t for node in nodes})
    nodes_by_time = {time: [node for node in nodes if node.t == time] for time in times}
    candidates: list[CandidateEdge] = []
    raw_by_id = {int(row["node_id"]): row for row in raw_nodes}
    for source_time, target_time in zip(times[:-1], times[1:], strict=True):
        if target_time != source_time + 1:
            continue
        source_nodes = nodes_by_time[source_time]
        target_nodes = nodes_by_time[target_time]
        if not source_nodes or not target_nodes:
            continue
        source_physical = np.asarray([node.coord_zyx for node in source_nodes], dtype=np.float64) * scale
        tree = cKDTree(source_physical)
        for target in target_nodes:
            target_physical = np.asarray(target.coord_zyx, dtype=np.float64) * scale
            backward = np.asarray(raw_by_id[target.node_id]["backward_um"], dtype=np.float64)
            predicted_parent = target_physical + backward
            nearby = tree.query_ball_point(predicted_parent, r=edge_radius_um)
            ranked: list[tuple[float, float, int]] = []
            for index in nearby:
                actual = float(np.linalg.norm(source_physical[int(index)] - target_physical))
                if actual > edge_radius_um:
                    continue
                residual = float(np.linalg.norm(source_physical[int(index)] - predicted_parent))
                ranked.append((residual, actual, int(index)))
            ranked.sort(key=lambda value: (value[0], value[1], source_nodes[value[2]].node_id))
            for residual, actual, index in ranked[:neighbor_cap]:
                score = max(0.0, 1.0 - residual / edge_radius_um)
                if score < edge_threshold:
                    continue
                candidates.append(
                    CandidateEdge(
                        source_id=source_nodes[index].node_id,
                        target_id=target.node_id,
                        dt=1,
                        distance_um=actual,
                        cost=1.0 - score,
                        score=score,
                        features={"motion_residual_um": residual, "v6_motion_score": score},
                    )
                )
    ordered_by_t = [nodes_by_time[time] for time in times]
    config = RouteConfig(
        label=f"{model_name}_edge",
        detector_sources=(model_name,),
        threshold_quantiles=(),
        min_distance_um=0.0,
        fusion_radius_um=0.0,
        max_points_per_t=max((len(value) for value in ordered_by_t), default=1),
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
    selected, forks = select_edges(ordered_by_t, candidates, tuple(float(value) for value in scale), config)
    graph = validate_lineage_graph(nodes, selected, allow_divisions=False)
    if forks or graph["num_divisions"]:
        raise RuntimeError("edge-only V6 inference unexpectedly emitted a division")
    prediction_nodes = [
        {"dataset": dataset, "node_id": node.node_id, "t": node.t, "z": node.z, "y": node.y, "x": node.x}
        for node in nodes
    ]
    prediction_edges = [edge.to_submission_edge() for edge in selected]
    return prediction_nodes, prediction_edges, {**graph, "candidate_edges": len(candidates), "dense_nxn": False}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, help="hashed checkpoint-score artifact descriptor")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT_CONFIG)
    parser.add_argument("--compare-parent", action="store_true")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_MET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--scope", choices=("fit", "cv"), default="cv")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--rung", choices=("f0-2",))
    parser.add_argument("--clip-start", type=int)
    parser.add_argument("--clip-length", type=int, default=2)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--nms-um", type=float, default=3.0)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--edge-radius-um", type=float, default=8.0)
    parser.add_argument("--neighbor-cap", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH_PY)
    parser.add_argument("--torch-path", type=Path, default=DEFAULT_TORCH_PATH)
    parser.add_argument("--pred-out", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    args = parser.parse_args(argv)
    if args.infer_job is not None:
        return args
    if args.rung is not None and args.dataset:
        raise V6Error("use either a frozen rung or explicit datasets, not both")
    if not 2 <= args.clip_length <= 8:
        raise V6Error("clip length must be in [2, 8]")
    if args.clip_start is not None and args.clip_start < 0:
        raise V6Error("clip start must be nonnegative")
    if not 0.2 <= args.det_threshold <= 0.999:
        raise V6Error("detection threshold is outside the locked range")
    if not 2.0 <= args.nms_um <= 5.0:
        raise V6Error("NMS radius is outside the locked range")
    if not 4.0 <= args.edge_radius_um <= 10.0:
        raise V6Error("edge radius is outside the locked range")
    if not 0.2 <= args.edge_threshold <= 0.8:
        raise V6Error("edge threshold is outside the locked range")
    if not 1 <= args.max_points <= 4096:
        raise V6Error("max points must be in [1, 4096]")
    for path in (args.pred_out, args.output):
        if len(path.name) >= 15:
            raise V6Error(f"output basename must be shorter than 15 characters: {path.name}")
    return args


def _orchestrate(args: argparse.Namespace) -> int:
    import numpy as np
    import zarr
    from zarr.storage import ZipStore

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from celltrack_alpha.official_eval import official_metric_row, official_summary
    from celltrack_baseline.rule_based import parse_scale, read_geff_arrays

    artifact = selected_artifact(args)
    checkpoint = Path(artifact["checkpoint"])
    checkpoint_sha = str(artifact["checkpoint_sha256"])
    checkpoint_bytes = int(artifact["checkpoint_bytes"])
    metrics = artifact["metrics"]
    metrics_sha = str(artifact["metrics_sha256"])
    model_id = str(artifact["model_id"])
    split = _exact_json(args.split, SPLIT_SHA256)
    requested = F0_RUNG_2 if args.rung == "f0-2" else args.dataset
    fold, datasets, scope_proof = fold_scope(split, metrics, args.scope, requested)
    if fold != artifact["fold"]:
        raise V6Error("artifact fold disagrees with split-scoped checkpoint metrics")
    if args.rung == "f0-2" and (args.scope != "cv" or fold != 0):
        raise V6Error("f0-2 is a fold-0 held-out CV rung")
    if not args.zip.is_file():
        raise V6Error("competition archive is missing")
    if not args.torch_python.is_file() or not args.torch_path.is_dir():
        raise V6Error("local Torch runtime is unavailable")
    if args.compare_parent:
        if (
            not args.parent.is_file()
            or args.parent.stat().st_size != PARENT_BYTES
            or sha256_file(args.parent) != PARENT_SHA256
            or not args.parent_config.is_file()
            or sha256_file(args.parent_config) != PARENT_CONFIG_SHA256
        ):
            raise V6Error("released parent checkpoint/config identity mismatch")

    gt_by_dataset: dict[str, dict[str, Any]] = {}
    attrs_by_dataset: dict[str, dict[str, Any]] = {}
    clip_by_dataset: dict[str, tuple[int, int, int]] = {}
    with tempfile.TemporaryDirectory(prefix="v6-") as temporary:
        temp = Path(temporary)
        job_items: list[dict[str, Any]] = []
        store = ZipStore(args.zip, mode="r")
        try:
            for index, dataset in enumerate(datasets):
                group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
                array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
                attrs = dict(group.attrs)
                gt = read_geff_arrays(store, "train", dataset)
                total = int(array.shape[0])
                start = args.clip_start if args.clip_start is not None else _first_gt_start(gt, total, args.clip_length)
                end = min(total, int(start) + args.clip_length)
                if end - int(start) < 2:
                    raise V6Error(f"dataset {dataset} cannot provide a two-frame clip")
                q_low, q_high = _quantile(attrs, "0.001"), _quantile(attrs, "0.999")
                if q_high <= q_low:
                    raise V6Error(f"dataset {dataset} has an invalid quantile range")
                raw = array[int(start):end, ::DOWN_SAMPLE[0], ::DOWN_SAMPLE[1], ::DOWN_SAMPLE[2]].astype(np.float32)
                frames = np.clip((raw - q_low) / (q_high - q_low), 0.0, 1.0)
                if not np.isfinite(frames).all():
                    raise V6Error(f"dataset {dataset} produced nonfinite frames")
                frame_path = temp / f"f{index}.npy"
                np.save(frame_path, frames, allow_pickle=False)
                scale = parse_scale(attrs)
                job_items.append(
                    {
                        "dataset": dataset,
                        "frames": str(frame_path),
                        "start": int(start),
                        "downsample": list(DOWN_SAMPLE),
                        "model_voxel_um": [float(value) * factor for value, factor in zip(scale, DOWN_SAMPLE, strict=True)],
                    }
                )
                gt_by_dataset[dataset] = _crop_gt(gt, int(start), end, total)
                attrs_by_dataset[dataset] = attrs
                clip_by_dataset[dataset] = (int(start), end, total)
        finally:
            store.close()

        worker_output = temp / "out.json"
        job_path = temp / "job.json"
        job = {
            "model_id": model_id,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "compare_parent": args.compare_parent,
            "parent": str(args.parent.resolve()) if args.compare_parent else None,
            "parent_config": str(args.parent_config.resolve()) if args.compare_parent else None,
            "fold": fold,
            "device": args.device,
            "det_threshold": args.det_threshold,
            "nms_um": args.nms_um,
            "max_points": args.max_points,
            "datasets": job_items,
            "output": str(worker_output),
        }
        job_path.write_bytes(_canonical_bytes(job))
        environment = dict(os.environ)
        prior = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(args.torch_path) if not prior else f"{args.torch_path}{os.pathsep}{prior}"
        completed = subprocess.run(
            [str(args.torch_python), "-B", str(Path(__file__).resolve()), "--infer-job", str(job_path)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"V6 Torch worker failed ({completed.returncode}): {completed.stderr[-2000:]}")
        inferred = json.loads(worker_output.read_text(encoding="utf-8"))

    restores = inferred.get("restores")
    restored = restores.get(model_id) if isinstance(restores, Mapping) else None
    if not isinstance(restored, Mapping) or restored.get("checkpoint_sha256") != checkpoint_sha or restored.get("fold") != fold:
        raise V6Error("Torch worker restore identity mismatch")
    model_names = ("parent", model_id) if args.compare_parent else (model_id,)
    predictions_by_model = inferred.get("predictions")
    if not isinstance(predictions_by_model, Mapping):
        raise V6Error("Torch worker prediction schema mismatch")
    if args.compare_parent:
        parent_restore = restores.get("parent") if isinstance(restores, Mapping) else None
        if (
            not isinstance(parent_restore, Mapping)
            or parent_restore.get("checkpoint_sha256") != PARENT_SHA256
            or not parent_restore.get("released_transformer_discarded")
        ):
            raise V6Error("Torch worker parent restore identity mismatch")

    graphs: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        raw_predictions = {
            str(item["dataset"]): item for item in predictions_by_model.get(model_name, [])
        }
        if set(raw_predictions) != set(datasets):
            raise V6Error(f"Torch worker {model_name} dataset coverage mismatch")
        for dataset in datasets:
            item = raw_predictions[dataset]
            scale = parse_scale(attrs_by_dataset[dataset])
            nodes, edges, graph = _link_graph(
                dataset,
                item["nodes"],
                scale,
                args.edge_radius_um,
                args.neighbor_cap,
                args.edge_threshold,
                model_name,
            )
            start, end, _ = clip_by_dataset[dataset]
            metric = official_metric_row(nodes, edges, gt_by_dataset[dataset], scale)
            graphs.append(
                {"model": model_name, "dataset": dataset, "clip_start": start, "clip_end": end,
                 "nodes": nodes, "edges": edges, "graph": graph}
            )
            score_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "fold": fold,
                    "scope": args.scope,
                    "clip_start": start,
                    "clip_end": end,
                    "gt_nodes": int(len(gt_by_dataset[dataset]["nodes"]["node_id"])),
                    "gt_edges": int(len(gt_by_dataset[dataset]["edges"])),
                    "num_pred_nodes": len(nodes),
                    "num_pred_edges": len(edges),
                    "graph": graph,
                    **metric,
                }
            )
    parameters = {
        "detection_threshold": args.det_threshold,
        "nms_um": args.nms_um,
        "max_points_per_frame": args.max_points,
        "edge_radius_um": args.edge_radius_um,
        "neighbor_cap": args.neighbor_cap,
        "edge_threshold": args.edge_threshold,
        "division_enabled": False,
        "downsample_zyx": list(DOWN_SAMPLE),
        "comparison_parent": args.compare_parent,
    }
    source_hashes = {
        "scripts/score_v6.py": sha256_file(Path(__file__).resolve()),
        "src/celltrack_alpha/t_adapter.py": sha256_file(SRC / "celltrack_alpha/t_adapter.py"),
    }
    prediction = {
        "schema_version": 1,
        "kind": "checkpoint_head_to_head_predictions" if args.compare_parent else "checkpoint_fold_safe_predictions",
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": PARENT_SHA256 if args.compare_parent else None,
        "metrics_sha256": metrics_sha,
        "artifact": (str(artifact["artifact_path"]) if artifact["artifact_path"] is not None else None),
        "artifact_sha256": artifact["artifact_sha256"],
        "split_sha256": SPLIT_SHA256,
        "restore": dict(restored),
        "restores": {name: dict(restores[name]) for name in model_names},
        "scope_proof": scope_proof,
        "rung": {"id": args.rung, "frozen": args.rung is not None, "datasets": list(datasets)},
        "source_hashes": source_hashes,
        "input_archive": {"path": str(args.zip), "bytes": args.zip.stat().st_size},
        "parameters": parameters,
        "graphs": graphs,
    }
    prediction_sha = _atomic_json(args.pred_out, prediction)
    summaries = {
        model_name: official_summary([row for row in score_rows if row["model"] == model_name])
        for model_name in model_names
    }
    summary = summaries[model_id]
    head_to_head: dict[str, Any] | None = None
    if args.compare_parent:
        parent_summary = summaries["parent"]
        delta: dict[str, float | None] = {}
        for key in ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall"):
            left, right = float(summary.get(key, float("nan"))), float(parent_summary.get(key, float("nan")))
            delta[key] = left - right if math.isfinite(left) and math.isfinite(right) else None
        head_to_head = {
            "comparable_local_rung": True,
            "same_datasets": True,
            "same_clip_windows": True,
            "same_postprocessing_parameters": True,
            "released_parent_transformer_used": False,
            "released_parent_motion": "zero_motion",
            f"{model_id}_minus_parent": delta,
        }
    full_cv = args.scope == "cv" and set(datasets) == {
        value[6:] for value in split["folds"][str(fold)]["held_out_video_ids"]
    }
    edge_gate = float(summary.get("adj_edge_jaccard", float("nan")))
    division_allowed = bool(full_cv and math.isfinite(edge_gate) and edge_gate >= 0.90)
    report = {
        "schema_version": 1,
        "kind": "checkpoint_head_to_head_official_score" if args.compare_parent else "checkpoint_fold_safe_official_score",
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": PARENT_SHA256 if args.compare_parent else None,
        "prediction": str(args.pred_out),
        "prediction_sha256": prediction_sha,
        "metrics_sha256": metrics_sha,
        "artifact": (str(artifact["artifact_path"]) if artifact["artifact_path"] is not None else None),
        "artifact_sha256": artifact["artifact_sha256"],
        "split_sha256": SPLIT_SHA256,
        "scope_proof": scope_proof,
        "rung": {"id": args.rung, "frozen": args.rung is not None, "datasets": list(datasets)},
        "source_hashes": source_hashes,
        "input_archive": {"path": str(args.zip), "bytes": args.zip.stat().st_size},
        "parameters": parameters,
        "summary": summary,
        "summaries": summaries,
        "head_to_head": head_to_head,
        "external_reference": {
            "public_kaggle_score": 0.903,
            "provenance": "coordinator_supplied_external_reference",
            "used_in_local_comparison": False,
        },
        "edge_only_gate": {
            "full_grouped_cv": full_cv,
            "adjusted_edge_jaccard": edge_gate if math.isfinite(edge_gate) else None,
            "threshold": 0.90,
            "division_activation_allowed": division_allowed,
            "division_emitted": False,
        },
        "rows": score_rows,
        "qualification": (
            "frozen_fold0_heldout_rung" if args.rung is not None
            else "diagnostic_subset" if not full_cv
            else "grouped_fold_cv"
        ),
        "audit_used": False,
        "test_used": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(_json_safe({"prediction": str(args.pred_out), "score": str(args.output), "summaries": summaries,
                                 "head_to_head": head_to_head}), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.infer_job is not None:
        return _infer_job(args.infer_job)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one full-video, fold-safe TemporalAdapterModel evidence comparison.

Inference is image-only and emits one fragment per paired ``train_ta`` output.
Only after both fragments are finalized are they scored against the held-out
GEFF through the unchanged ``scoreall`` frozen official-scoring functions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OFFICIAL = ROOT / "external/kaggle-cell-tracking-competition/src"
for value in (str(SRC), str(OFFICIAL), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from celltrack_alpha.ta_inference import (  # noqa: E402
    GATE_SCHEMA,
    GateAttestation,
    GateError,
    TrainOutputAttestation,
    attest_gate_pair,
    iter_sliding_frame_outputs,
    run_deterministic_smoke,
    sha256_file,
    sparse_alpha_association,
    write_fragment,
)


DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_PROVENANCE = ROOT / "reports/g02/fit_provenance_v1.json"
DEFAULT_REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
DOWN_SAMPLE = (1, 4, 4)
DETECTION_THRESHOLD = 0.97
NMS_UM = 3.0
MAX_POINTS_PER_FRAME = 512
EDGE_RADIUS_UM = 8.0
NEIGHBOR_CAP = 4
EDGE_THRESHOLD = 0.2
MODEL_LABELS = ("released_baseline", "migration_pretrained_adapter")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("ascii")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise GateError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON artifact is not an object: {path}")
    return value


def _quantile(attrs: Mapping[str, Any], wanted: str) -> float:
    try:
        value = float(attrs["image_statistics"]["quantiles"][wanted])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"image metadata lacks quantile {wanted}") from exc
    if not math.isfinite(value):
        raise GateError(f"image quantile {wanted} is nonfinite")
    return value


def _pool_kernel(distance_um: float, model_voxel_um: Sequence[float]) -> tuple[int, int, int]:
    result: list[int] = []
    for scale in model_voxel_um:
        value = max(1, round(distance_um / float(scale)))
        result.append(value if value % 2 else value + 1)
    return tuple(result)  # type: ignore[return-value]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float32)


def _detect_frame(
    frame: Any,
    *,
    node_id_start: int,
    model_voxel_um: Sequence[float],
    original_shape_zyx: Sequence[int],
) -> tuple[list[dict[str, Any]], int]:
    from scipy.ndimage import maximum_filter

    logits = np.asarray(frame.center_logit, dtype=np.float32)
    if logits.ndim != 3 or not np.isfinite(logits).all():
        raise GateError("frame center logits are invalid")
    kernel = _pool_kernel(NMS_UM, model_voxel_um)
    pooled = maximum_filter(logits, size=kernel, mode="constant", cval=-np.inf)
    probabilities = _sigmoid(logits)
    coords = np.argwhere((logits == pooled) & (probabilities >= DETECTION_THRESHOLD))
    scores = probabilities[tuple(coords.T)] if len(coords) else np.empty(0, dtype=np.float32)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores)) if len(coords) else np.empty(0, dtype=np.int64)
    order = order[:MAX_POINTS_PER_FRAME]
    coords, scores = coords[order], scores[order]
    nodes: list[dict[str, Any]] = []
    next_id = node_id_start
    for coord, score in zip(coords, scores, strict=True):
        z, y, x = (int(value) for value in coord)
        original = (z * DOWN_SAMPLE[0], y * DOWN_SAMPLE[1], x * DOWN_SAMPLE[2])
        if any(value < 0 or value >= bound for value, bound in zip(original, original_shape_zyx, strict=True)):
            raise GateError("downsampled detection maps outside the registered image")
        association = frame.association[:, z, y, x]
        if frame.association_valid:
            backward = [float(value) for value in association[:3]]
            log_variance = float(association[3])
        else:
            backward = [0.0, 0.0, 0.0]
            log_variance = 0.0
        if not all(math.isfinite(value) for value in (*backward, log_variance, float(score))):
            raise GateError("adapter detection contains a nonfinite association value")
        nodes.append(
            {
                "node_id": next_id,
                "t": int(frame.frame_index),
                "z": original[0],
                "y": original[1],
                "x": original[2],
                "score": float(score),
                "backward_um": backward,
                "log_variance": log_variance,
            }
        )
        next_id += 1
    return nodes, next_id


def _load_model(attestation: TrainOutputAttestation, device_name: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D

    from celltrack_alpha.t_adapter import load_trained_model

    backbone = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
    model, restored = load_trained_model(
        backbone,
        attestation.checkpoint_path,
        expected_sha256=attestation.checkpoint_sha256,
        expected_size=attestation.checkpoint_bytes,
        expected_fold=attestation.fold,
    )
    if tuple(restored.source_groups) != attestation.source_groups or tuple(restored.source_video_ids) != attestation.source_video_ids:
        raise GateError(f"{attestation.label} torch restore changed the attested fit-source scope")
    device = torch.device(device_name)
    model = model.to(device).eval()
    return model, {
        "checkpoint_sha256": restored.checkpoint_sha256,
        "fold": restored.fold,
        "scenario": restored.scenario,
        "completed_steps": restored.completed_steps,
        "source_groups": list(restored.source_groups),
        "source_video_ids": list(restored.source_video_ids),
        "model_tensor_count": restored.model_tensor_count,
        "model_numel": restored.model_numel,
        "device": str(device),
    }


def _infer_model(
    attestation: TrainOutputAttestation,
    *,
    array: Any,
    q_low: float,
    q_high: float,
    scale_zyx: Sequence[float],
    original_shape: Sequence[int],
    device_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    if q_high <= q_low:
        raise GateError("held-out image quantile range is invalid")
    model, restore = _load_model(attestation, device_name)
    device = torch.device(device_name)
    total_frames = int(original_shape[0])
    model_shape = (
        int(original_shape[1]),
        math.ceil(int(original_shape[2]) / DOWN_SAMPLE[1]),
        math.ceil(int(original_shape[3]) / DOWN_SAMPLE[2]),
    )

    def pairs() -> Iterator[np.ndarray]:
        with torch.inference_mode():
            for start in range(total_frames - 1):
                raw = array[
                    start : start + 2,
                    :: DOWN_SAMPLE[0],
                    :: DOWN_SAMPLE[1],
                    :: DOWN_SAMPLE[2],
                ].astype(np.float32)
                if tuple(raw.shape) != (2, *model_shape) or not np.isfinite(raw).all():
                    raise GateError("held-out two-frame image slab is missing or nonfinite")
                normalized = np.clip((raw - q_low) / (q_high - q_low), 0.0, 1.0)
                tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.float32)
                output = model(tensor, scenario="universal")[0]
                if tuple(output.shape) != (2, 6, *model_shape) or not bool(torch.isfinite(output).all().item()):
                    raise GateError("TemporalAdapterModel inference output schema drift")
                yield output.detach().cpu().numpy()
                if (start + 1) % 10 == 0 or start + 1 == total_frames - 1:
                    print(
                        json.dumps(
                            {"model": attestation.label, "completed_pairs": start + 1, "total_pairs": total_frames - 1},
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

    model_voxel_um = [float(value) * factor for value, factor in zip(scale_zyx, DOWN_SAMPLE, strict=True)]
    raw_nodes: list[dict[str, Any]] = []
    frame_counts: list[int] = []
    next_id = 1
    for frame in iter_sliding_frame_outputs(pairs()):
        detections, next_id = _detect_frame(
            frame,
            node_id_start=next_id,
            model_voxel_um=model_voxel_um,
            original_shape_zyx=original_shape[1:],
        )
        raw_nodes.extend(detections)
        frame_counts.append(len(detections))
    if len(frame_counts) != total_frames or not raw_nodes:
        raise GateError("full-video adapter inference did not cover every frame with any detections")
    del model
    gc.collect()
    report = {
        "restore": restore,
        "full_video": True,
        "frame_count": total_frames,
        "adjacent_pair_forwards": total_frames - 1,
        "prediction_node_count": len(raw_nodes),
        "frame_detection_counts": frame_counts,
        "min_frame_detections": min(frame_counts),
        "max_frame_detections": max(frame_counts),
        "preprocessing": {
            "downsample_zyx": list(DOWN_SAMPLE),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "clip": [0.0, 1.0],
        },
        "sliding_contract": {
            "center_logits": "mean-of-available-two-frame-contexts",
            "backward_displacement": "target-output-of-(t-1,t)-only",
            "log_variance": "target-output-of-(t-1,t)-only",
            "first_frame_association": "undefined-zero-sentinel-not-linked-backward",
        },
    }
    return raw_nodes, report


def _preflight_frozen_selection(gate: GateAttestation, archive: Path) -> tuple[Any, dict[str, str], tuple[str, ...]]:
    import scoreall

    split = _strict_json(Path(gate.split_path))
    scorer_hashes = scoreall.verify_scorer_sources(split)
    archive_ids = scoreall.archive_train_video_ids(archive)
    selection = scoreall.select_video_ids(
        split,
        f"fold{gate.fold}",
        archive_train_video_ids=archive_ids,
        max_datasets=1,
    )
    if selection.selected_video_ids != (gate.held_out_video_id,):
        raise GateError(
            "the one-sample evidence gate must use the frozen scorer's deterministic first held-out identity"
        )
    return selection, scorer_hashes, archive_ids


def _score_fragments(
    gate: GateAttestation,
    archive: Path,
    selection: Any,
    scorer_hashes: Mapping[str, str],
    fragments: Mapping[str, Path],
) -> dict[str, Any]:
    import scoreall
    from zarr.storage import ZipStore

    manifests = {
        label: scoreall.validate_fragment_set([path], selection.selected_video_ids)
        for label, path in fragments.items()
    }
    store = ZipStore(archive, mode="r")
    try:
        rows: dict[str, list[dict[str, Any]]] = {}
        input_provenance: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for label in MODEL_LABELS:
            model_rows, provenance, model_coverage = scoreall.score_fragment_input(store, manifests[label])
            rows[label] = model_rows
            input_provenance[label] = provenance
            coverage[label] = model_coverage
    finally:
        store.close()
    summaries = {label: scoreall.official_summary(rows[label]) for label in MODEL_LABELS}
    delta: dict[str, float | None] = {}
    for key in ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall"):
        candidate = float(summaries["migration_pretrained_adapter"].get(key, float("nan")))
        baseline = float(summaries["released_baseline"].get(key, float("nan")))
        delta[key] = candidate - baseline if math.isfinite(candidate) and math.isfinite(baseline) else None
    archive_hash, archive_bytes = scoreall.sha256_with_size(archive)
    return {
        "schema": GATE_SCHEMA,
        "kind": "single-heldout-full-video-temporal-adapter-evidence-gate",
        "status": "completed",
        "qualification": "one-fold0-heldout-video-only-not-full-fold-not-oof-not-kaggle",
        "selection": selection.as_dict(),
        "gate_provenance": gate.json(),
        "archive": {
            "path": str(archive.resolve()),
            "sha256": archive_hash,
            "bytes": archive_bytes,
        },
        "frozen_scorer_source_hashes": dict(scorer_hashes),
        "frozen_scorer_functions": [
            "scoreall.validate_fragment_set",
            "scoreall.score_fragment_input",
            "celltrack_alpha.official_eval.official_metric_row",
            "celltrack_alpha.official_eval.official_summary",
        ],
        "summaries": summaries,
        "candidate_minus_baseline": delta,
        "samples": rows,
        "input_provenance": input_provenance,
        "coverage": coverage,
        "evidence_boundary": {
            "held_out_video_count": 1,
            "full_video": True,
            "full_fold": False,
            "oof": False,
            "kaggle": False,
            "audit_used": False,
            "test_used": False,
            "promotion_claim_allowed": False,
            "alpha_fallback_preserved": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import zarr
    from zarr.storage import ZipStore

    if args.fold != 0:
        raise GateError("this bounded evidence run is locked to fold 0")
    if not args.zip.is_file():
        raise GateError("local competition archive is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise GateError("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = attest_gate_pair(
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_metrics=args.baseline_metrics,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_metrics=args.candidate_metrics,
        pretrain_checkpoint=args.pretrain_checkpoint,
        pretrain_metrics=args.pretrain_metrics,
        split_manifest=args.split_manifest,
        provenance_manifest=args.provenance_manifest,
        registry_manifest=args.registry_manifest,
        fold=args.fold,
        held_out_dataset=args.dataset,
    )
    selection, scorer_hashes, _archive_ids = _preflight_frozen_selection(gate, args.zip)
    _write_json_new(args.output_dir / "preflight.json", gate.json())

    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    from celltrack_baseline.rule_based import parse_scale

    store = ZipStore(args.zip, mode="r")
    try:
        group = zarr.open_group(store=store, path=f"train/{gate.held_out_dataset}.zarr", mode="r")
        array = zarr.open_array(store=store, path=f"train/{gate.held_out_dataset}.zarr/0", mode="r")
        observed_shape = tuple(int(value) for value in array.shape)
        if observed_shape != gate.held_out_image_shape:
            raise GateError("held-out archive image shape disagrees with the attested registry")
        attrs = dict(group.attrs)
        scale = parse_scale(attrs)
        q_low, q_high = _quantile(attrs, "0.001"), _quantile(attrs, "0.999")
        fragments: dict[str, Path] = {}
        inference_records: dict[str, Any] = {}
        for label, attestation in (
            ("released_baseline", gate.baseline),
            ("migration_pretrained_adapter", gate.candidate),
        ):
            started = time.perf_counter()
            raw_nodes, inference = _infer_model(
                attestation,
                array=array,
                q_low=q_low,
                q_high=q_high,
                scale_zyx=scale,
                original_shape=observed_shape,
                device_name=args.device,
            )
            nodes, edges, association = sparse_alpha_association(
                gate.held_out_dataset,
                raw_nodes,
                scale,
                edge_radius_um=EDGE_RADIUS_UM,
                neighbor_cap=NEIGHBOR_CAP,
                edge_threshold=EDGE_THRESHOLD,
            )
            fragment = args.output_dir / label / "frags" / f"train__{gate.held_out_dataset}.csv"
            fragment.parent.mkdir(parents=True, exist_ok=True)
            fragment_hash = write_fragment(fragment, gate.held_out_dataset, nodes, edges)
            fragments[label] = fragment
            inference_records[label] = {
                **inference,
                "association": association,
                "fragment": {
                    "path": str(fragment.resolve()),
                    "sha256": fragment_hash,
                    "bytes": fragment.stat().st_size,
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
    finally:
        store.close()

    inference_report = {
        "schema": GATE_SCHEMA,
        "kind": "paired-full-video-image-only-inference",
        "status": "completed",
        "gate_provenance": gate.json(),
        "source_hashes": {
            "scripts/score_ta_gate.py": sha256_file(Path(__file__).resolve()),
            "src/celltrack_alpha/ta_inference.py": sha256_file(SRC / "celltrack_alpha/ta_inference.py"),
            "src/celltrack_alpha/t_adapter.py": sha256_file(SRC / "celltrack_alpha/t_adapter.py"),
        },
        "parameters": {
            "detection_threshold": DETECTION_THRESHOLD,
            "nms_um": NMS_UM,
            "max_points_per_frame": MAX_POINTS_PER_FRAME,
            "edge_radius_um": EDGE_RADIUS_UM,
            "neighbor_cap": NEIGHBOR_CAP,
            "edge_threshold": EDGE_THRESHOLD,
            "division_enabled": False,
            "downsample_zyx": list(DOWN_SAMPLE),
        },
        "models": inference_records,
        "data_access": {
            "image_split": "train",
            "image_video_count": 1,
            "full_video": True,
            "geff_read": False,
            "held_out_labels_read": False,
            "audit_read": False,
            "test_read": False,
        },
    }
    inference_sha = _write_json_new(args.output_dir / "inference.json", inference_report)
    score_report = _score_fragments(gate, args.zip, selection, scorer_hashes, fragments)
    score_report["inference"] = {
        "path": str((args.output_dir / "inference.json").resolve()),
        "sha256": inference_sha,
    }
    _write_json_new(args.output_dir / "score.json", score_report)
    return score_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only the tiny deterministic no-data smoke")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--pretrain-checkpoint", type=Path)
    parser.add_argument("--pretrain-metrics", type=Path)
    parser.add_argument("--fold", type=int, choices=(0,), default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--provenance-manifest", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--registry-manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    if args.smoke:
        return args
    required = (
        "baseline_checkpoint",
        "baseline_metrics",
        "candidate_checkpoint",
        "candidate_metrics",
        "pretrain_checkpoint",
        "pretrain_metrics",
        "dataset",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise GateError(f"missing required evidence arguments: {missing}")
    if not 1 <= args.cpu_threads <= 32:
        raise GateError("--cpu-threads must be in [1,32]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.smoke:
            print(json.dumps(run_deterministic_smoke(), indent=2, sort_keys=True))
            return 0
        result = run(args)
    except GateError as exc:
        print(f"score_ta_gate: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str((args.output_dir / "score.json").resolve()),
                "qualification": result["qualification"],
                "summaries": result["summaries"],
                "candidate_minus_baseline": result["candidate_minus_baseline"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the organizer fit monitor with positive-edge metrics on tiny CPU slices.

The organizer training loop reports cell-wise accuracy over a sparse transition
matrix and evaluates on videos that are also members of fit117.  This script
keeps that fit-only boundary but makes the signal interpretable:

* it reports the all-negative accuracy implied by the exact official mask;
* it reports positive-edge precision/recall/F1, both on the official mask and
  on the complete detected-node matrix; and
* it records node recall and runtime for deterministic windows.

It never opens fold-0 holdout, audit, or test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/kaggle-cell-tracking-competition"
for location in (UPSTREAM / "src", UPSTREAM / "scripts", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import predict_unet_transformer as official_predict  # noqa: E402
import train_unet_transformer as official_train  # noqa: E402
import train_organizer_f0 as training_contract  # noqa: E402


SCHEMA = "celltrack-alpha.organizer-fit-monitor-audit.v1"
DEFAULT_TRAINING = (
    ROOT
    / "reports/g02/organizer_unet_transformer_f0/stage00_v3/organizer_f0"
)
DEFAULT_DATA = ROOT / "data/derived/t03_fold0_fit10_images_v1/train"
DEFAULT_OUTPUT = ROOT / "reports/g02/organizer_fit_monitor_audit_v1.json"
DEFAULT_VIDEO_IDS = ("train:6bba_d3da753b",)


class MonitorAuditError(RuntimeError):
    """The fit-monitor audit contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise MonitorAuditError(f"expected JSON object: {path}")
    return value


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def _summarise_edges(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_edges": tp + fp,
        "gt_edges": tp + fn,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
    }


def _confusion(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    prediction = prediction.bool()
    target = target.bool()
    return {
        "tp": int((prediction & target).sum().item()),
        "fp": int((prediction & ~target).sum().item()),
        "fn": int((~prediction & target).sum().item()),
        "tn": int((~prediction & ~target).sum().item()),
    }


def _summarise_confusion(row: Mapping[str, int]) -> dict[str, Any]:
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    tn = int(row["tn"])
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "positive_prevalence": _safe_ratio(tp + fn, total),
        "accuracy": _safe_ratio(tp + tn, total),
        "all_negative_accuracy": _safe_ratio(tn + fp, total),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
    }


def _add_confusion(total: dict[str, int], row: Mapping[str, int]) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        total[key] += int(row[key])


def _select_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 1:
        raise MonitorAuditError("monitor video produced no windows")
    count = min(count, length)
    if count == 1:
        return (length // 2,)
    return tuple(
        sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})
    )


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise MonitorAuditError(f"refusing to overwrite output: {args.output}")
    if args.max_videos < 1 or args.windows_per_video < 1:
        raise MonitorAuditError("video and window counts must be positive")
    if args.device != "cpu":
        raise MonitorAuditError("this audit is deliberately CPU-only")

    run = _load_object(args.training_dir / "run.json")
    provenance = _load_object(args.training_dir / "provenance.json")
    weights = args.training_dir / "edge_predictor_best.pth"
    scope = training_contract.frozen_scope()
    monitor_ids = tuple(str(value) for value in provenance.get("monitor_video_ids", ()))
    expected_monitor = training_contract.monitor_identities(scope["fit117"], len(monitor_ids))
    if (
        run.get("schema") != training_contract.SCHEMA
        or run.get("data_boundary", {}).get("fit_only") is not True
        or run.get("best_checkpoint_sha256") != _sha256(weights)
        or monitor_ids != expected_monitor
        or not set(monitor_ids).issubset(scope["fit117"])
    ):
        raise MonitorAuditError("organizer training or monitor provenance is invalid")

    requested_values = args.video_id or list(DEFAULT_VIDEO_IDS)
    requested_ids = tuple(
        value if value.startswith("train:") else f"train:{value}"
        for value in requested_values
    )
    if (
        not requested_ids
        or len(set(requested_ids)) != len(requested_ids)
        or not set(requested_ids).issubset(scope["fit117"])
    ):
        raise MonitorAuditError("audit identities must be unique members of fit117")
    selected_ids = requested_ids[: args.max_videos]
    model, window_size, downsample = official_predict.load_model(
        weights, torch.device("cpu")
    )
    if window_size != 2:
        raise MonitorAuditError(f"unsupported organizer window size: {window_size}")
    model.eval()

    official_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    full_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    inference_edge_total = {"tp": 0, "fp": 0, "fn": 0}
    gt_matched = 0
    gt_total = 0
    inference_gt_matched = 0
    inference_gt_total = 0
    video_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for video_index, identity in enumerate(selected_ids, start=1):
        stem = identity.split(":", 1)[1]
        path = args.data_dir / stem
        if not path.with_suffix(".zarr").is_dir() or not path.with_suffix(".geff").exists():
            raise MonitorAuditError(f"incomplete fit-only source: {identity}")
        meta, windows = official_train.load_dataset_windows(
            path,
            window_size=window_size,
            downsample=tuple(downsample),
        )
        indices = _select_indices(len(windows), args.windows_per_video)
        chosen = [windows[index] for index in indices]
        dataset = official_train.FrameWindowDataset([(meta, chosen)])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        video_official = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        video_full = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        video_inference_edges = {"tp": 0, "fp": 0, "fn": 0}
        video_gt_matched = 0
        video_gt_total = 0
        video_inference_gt_matched = 0
        video_inference_gt_total = 0
        video_started = time.perf_counter()

        for batch in loader:
            imgs = batch["imgs"].to(dtype=torch.float32)
            coords = batch["coords"]
            pos_feats = batch["pos_feats"]
            masks = batch["masks"]
            targets = batch["targets"]
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())
            ds_scale = batch["downsample"][0]
            _, frames = imgs.shape[:2]

            unet_out, det_logits = model.encode(imgs)
            frame_det = []
            for frame in range(frames):
                det_c, det_p, det_m, matches = official_train.detect_and_match(
                    det_logits[frame],
                    coords[:, frame],
                    masks[:, frame],
                    image_shape,
                    voxel_size=voxel_size,
                    pool_kernel_um=5.0,
                    frame_index=frame,
                    window_size=frames,
                )
                unet_feat = model._index_features(unet_out[:, frame], det_c, det_m)
                frame_det.append((det_c, det_p, det_m, matches, unet_feat))
                n_gt = int(masks[0, frame].sum().item())
                n_matched = int((matches[0] >= 0).sum().item())
                gt_total += n_gt
                gt_matched += n_matched
                video_gt_total += n_gt
                video_gt_matched += n_matched

            ns = frame_det[0][0].shape[1]
            nt = frame_det[1][0].shape[1]
            pair_target = official_train.build_matched_edge_targets(
                frame_det[0][3],
                frame_det[1][3],
                targets[:, 0],
                ns,
                nt,
            )
            logits = model.predict_edges(
                frame_det[0][4],
                frame_det[1][4],
                frame_det[0][0] * ds_scale,
                frame_det[1][0] * ds_scale,
                frame_det[0][1],
                frame_det[1][1],
                frame_det[0][2],
                frame_det[1][2],
            )
            ns_real = int(frame_det[0][2][0].sum().item())
            nt_real = int(frame_det[1][2][0].sum().item())
            target = pair_target[0, :ns_real, :nt_real]
            prediction = torch.softmax(logits[0, :ns_real, :nt_real], dim=0) > 0.5
            active_rows = target.sum(dim=1) > 0
            active_cols = target.sum(dim=0) > 0
            official_mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)

            official_confusion = _confusion(
                prediction[official_mask], target[official_mask]
            )
            full_confusion = _confusion(prediction, target)
            _add_confusion(official_total, official_confusion)
            _add_confusion(full_total, full_confusion)
            _add_confusion(video_official, official_confusion)
            _add_confusion(video_full, full_confusion)

            # Approximate the packaged inference contract without its four-pass
            # detection TTA.  A 0.99 sigmoid threshold is logit(0.99), and the
            # packaged graph uses a 3 um pool plus greedy 2-child/1-parent
            # constraints.  Count against every original GT edge so missed
            # detections remain false negatives.
            inference_frames = []
            for frame in range(frames):
                det_c, det_p, det_m, matches = official_train.detect_and_match(
                    det_logits[frame],
                    coords[:, frame],
                    masks[:, frame],
                    image_shape,
                    det_threshold=math.log(0.99 / 0.01),
                    voxel_size=voxel_size,
                    pool_kernel_um=3.0,
                    frame_index=frame,
                    window_size=frames,
                )
                unet_feat = model._index_features(unet_out[:, frame], det_c, det_m)
                inference_frames.append((det_c, det_p, det_m, matches, unet_feat))
                n_gt = int(masks[0, frame].sum().item())
                n_matched = int((matches[0] >= 0).sum().item())
                inference_gt_total += n_gt
                inference_gt_matched += n_matched
                video_inference_gt_total += n_gt
                video_inference_gt_matched += n_matched

            inference_ns = inference_frames[0][0].shape[1]
            inference_nt = inference_frames[1][0].shape[1]
            inference_logits = model.predict_edges(
                inference_frames[0][4],
                inference_frames[1][4],
                inference_frames[0][0] * ds_scale,
                inference_frames[1][0] * ds_scale,
                inference_frames[0][1],
                inference_frames[1][1],
                inference_frames[0][2],
                inference_frames[1][2],
            )
            inference_ns_real = int(inference_frames[0][2][0].sum().item())
            inference_nt_real = int(inference_frames[1][2][0].sum().item())
            inference_probs = torch.softmax(
                inference_logits[0, :inference_ns_real, :inference_nt_real],
                dim=0,
            )
            candidates = sorted(
                (
                    (float(inference_probs[source, target]), source, target)
                    for source, target in torch.nonzero(
                        inference_probs > 0.5, as_tuple=False
                    ).tolist()
                ),
                reverse=True,
            )
            children: dict[int, int] = {}
            parents: dict[int, int] = {}
            accepted: list[tuple[int, int]] = []
            for _, source, target in candidates:
                if children.get(source, 0) >= 2 or parents.get(target, 0) >= 1:
                    continue
                accepted.append((source, target))
                children[source] = children.get(source, 0) + 1
                parents[target] = parents.get(target, 0) + 1

            source_matches = inference_frames[0][3][0]
            target_matches = inference_frames[1][3][0]
            gt_transition = targets[0, 0]
            edge_tp = 0
            edge_fp = 0
            for source, target in accepted:
                gt_source = int(source_matches[source].item())
                gt_target = int(target_matches[target].item())
                if (
                    gt_source >= 0
                    and gt_target >= 0
                    and bool(gt_transition[gt_source, gt_target].item())
                ):
                    edge_tp += 1
                else:
                    edge_fp += 1
            edge_fn = int(gt_transition.sum().item()) - edge_tp
            for key, value in (("tp", edge_tp), ("fp", edge_fp), ("fn", edge_fn)):
                inference_edge_total[key] += value
                video_inference_edges[key] += value

        video_rows.append(
            {
                "identity": identity,
                "window_count_total": len(windows),
                "window_indices_audited": list(indices),
                "official_mask": _summarise_confusion(video_official),
                "full_detected_matrix": _summarise_confusion(video_full),
                "node_recall": _safe_ratio(video_gt_matched, video_gt_total),
                "gt_nodes_counted": video_gt_total,
                "inference_like_no_tta": {
                    "edge_outcomes": _summarise_edges(**video_inference_edges),
                    "node_recall": _safe_ratio(
                        video_inference_gt_matched, video_inference_gt_total
                    ),
                    "gt_nodes_counted": video_inference_gt_total,
                },
                "elapsed_seconds": time.perf_counter() - video_started,
            }
        )
        print(
            f"organizer monitor audit {video_index}/{len(selected_ids)} {identity} "
            f"elapsed={video_rows[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    report = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "video_ids": list(selected_ids),
            "windows_per_video": args.windows_per_video,
            "fit_only": True,
            "reported_monitor_video_ids": list(monitor_ids),
            "selected_are_reported_monitor": set(selected_ids).issubset(monitor_ids),
            "monitor_is_in_sample": True,
            "holdout_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "model": {
            "architecture": "TemporalUNet3D+SimpleNodeTransformer",
            "best_epoch": int(run["best_epoch"]),
            "training_epoch": int(run["epoch"]),
            "reported_monitor_accuracy": float(
                next(
                    row["monitor_acc"]
                    for row in json.loads(
                        (args.training_dir / "history.json").read_text(encoding="utf-8")
                    )
                    if row.get("epoch") == run["best_epoch"]
                )
            ),
            "reported_monitor_recall": float(
                next(
                    row["monitor_recall"]
                    for row in json.loads(
                        (args.training_dir / "history.json").read_text(encoding="utf-8")
                    )
                    if row.get("epoch") == run["best_epoch"]
                )
            ),
            "reported_monitor_product": float(run["best_fit_monitor_score"]),
            "checkpoint_sha256": _sha256(weights),
        },
        "metrics": {
            "official_mask": _summarise_confusion(official_total),
            "full_detected_matrix": _summarise_confusion(full_total),
            "node_recall": _safe_ratio(gt_matched, gt_total),
            "gt_nodes_counted": gt_total,
            "inference_like_no_tta": {
                "edge_outcomes": _summarise_edges(**inference_edge_total),
                "node_recall": _safe_ratio(
                    inference_gt_matched, inference_gt_total
                ),
                "gt_nodes_counted": inference_gt_total,
                "det_threshold": 0.99,
                "pool_kernel_um": 3.0,
                "edge_activation": "softmax_dim0",
                "edge_threshold": 0.5,
                "max_children_per_node": 2,
                "max_parents_per_node": 1,
            },
        },
        "videos": video_rows,
        "interpretation_contract": {
            "scorer_comparable": False,
            "monitor_accuracy_is_positive_edge_metric": False,
            "positive_edge_metrics_reported": True,
            "eligible_to_unlock_gpu": False,
        },
        "execution": {
            "device": "cpu",
            "external_gpu_action": False,
            "kaggle_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    if not math.isfinite(float(report["execution"]["elapsed_seconds"])):
        raise MonitorAuditError("nonfinite audit runtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.output)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    value.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--max-videos", type=int, default=1)
    value.add_argument("--windows-per-video", type=int, default=1)
    value.add_argument("--video-id", action="append")
    value.add_argument("--device", choices=("cpu",), default="cpu")
    return value


if __name__ == "__main__":
    result = audit(parser().parse_args())
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}))

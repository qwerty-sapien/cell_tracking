#!/usr/bin/env python3
"""End-to-end organizer checkpoint proxy on explicitly selected fit-only videos."""

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
from geff import GeffMetadata  # noqa: E402

import predict_unet_transformer as official_predict  # noqa: E402
from tracking_cellmot.io import open_dataset, save_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)
import train_organizer_f0 as training_contract  # noqa: E402


SCHEMA = "celltrack-alpha.organizer-f0-fit-proxy.v1"
DEFAULT_TRAINING = (
    ROOT
    / "reports/g02/organizer_unet_transformer_f0/stage00_v3/organizer_f0"
)
DEFAULT_DATA = ROOT / "data/derived/t03_fold0_fit10_images_v1/train"
DEFAULT_OUTPUT = ROOT / "reports/g02/organizer_f0_fit_proxy_v1"
DEFAULT_VIDEO = ("train:6bba_d3da753b",)


class OrganizerProxyError(RuntimeError):
    """The fit-only organizer proxy contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise OrganizerProxyError(f"expected JSON object: {path}")
    return value


def _estimated_nodes(path: Path) -> float:
    value = (GeffMetadata.read(path).extra or {}).get("estimated_number_of_nodes")
    return float(value) if value is not None else float("nan")


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise OrganizerProxyError(f"requested accelerator is unavailable: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise OrganizerProxyError(f"refusing to overwrite output: {args.output}")
    run_report = _load(args.training_dir / "run.json")
    provenance = _load(args.training_dir / "provenance.json")
    weights = args.training_dir / "edge_predictor_best.pth"
    scope = training_contract.frozen_scope()
    selected = tuple(
        value if value.startswith("train:") else f"train:{value}"
        for value in args.video_id
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(scope["fit117"])
    ):
        raise OrganizerProxyError("proxy identities must be unique members of fit117")
    if (
        run_report.get("schema") != training_contract.SCHEMA
        or run_report.get("fit_video_count") != 117
        or run_report.get("holdout_evaluated") is not False
        or run_report.get("data_boundary", {}).get("fit_only") is not True
        or provenance.get("data_boundary") != run_report.get("data_boundary")
        or run_report.get("best_checkpoint_sha256") != _sha256(weights)
    ):
        raise OrganizerProxyError("training checkpoint provenance is invalid")
    paths: list[Path] = []
    for identity in selected:
        stem = identity.split(":", 1)[1]
        path = args.data_dir / stem
        if not path.with_suffix(".zarr").is_dir() or not path.with_suffix(
            ".geff"
        ).exists():
            raise OrganizerProxyError(f"fit proxy source is incomplete: {identity}")
        paths.append(path)
    device = _device(args.device)
    model, window_size, downsample = official_predict.load_model(weights, device)
    config = official_predict.PredictConfig(
        det_threshold=0.99,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=False,
        max_parents_per_node=1,
        max_children_per_node=2,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions"
    predictions.mkdir()
    rows: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (identity, path) in enumerate(zip(selected, paths, strict=True), start=1):
        video_started = time.perf_counter()
        coords, edges = official_predict.predict_video(
            model,
            path,
            device,
            cfg=config,
            window_size=window_size,
            unet_batch_size=args.unet_batch_size,
            downsample=downsample,
        )
        graph = official_predict.build_graph(coords, edges)
        prediction = predictions / f"{identity.split(':', 1)[1]}.geff"
        save_graph(graph, prediction)
        dataset = open_dataset(path, normalize=False, require_tracks=True, load_image=False)
        result = compute_metric(
            graph, dataset.tracks, scale=dataset.scale, max_distance=7.0
        )
        recall = (
            node_recall(graph, dataset.tracks)
            if graph.num_nodes() and graph.num_edges()
            else 0.0
        )
        row = per_sample_metrics(
            result, _estimated_nodes(path.with_suffix(".geff")), recall
        )
        rows.append(row)
        per_video.append(
            {
                "identity": identity,
                **row,
                "predicted_nodes": int(graph.num_nodes()),
                "predicted_edges": int(graph.num_edges()),
                "prediction": str(prediction),
                "prediction_sha256": _sha256(prediction / "zarr.json"),
                "elapsed_seconds": time.perf_counter() - video_started,
            }
        )
        print(
            f"organizer fit proxy {index}/{len(selected)} {identity} "
            f"score={float(row['adj_edge_jaccard']):.9f}",
            flush=True,
        )
    summary = summarise(rows)
    if not math.isfinite(float(summary["score"])):
        raise OrganizerProxyError("fit proxy score is nonfinite")
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "video_ids": list(selected),
            "fit_only": True,
            "holdout_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "model": {
            "architecture": "TemporalUNet3D+SimpleNodeTransformer",
            "training_epoch": int(run_report["epoch"]),
            "best_epoch": int(run_report["best_epoch"]),
            "best_fit_monitor_score": float(run_report["best_fit_monitor_score"]),
            "checkpoint_sha256": _sha256(weights),
        },
        "prediction_config": {
            **vars(config),
            "window_size": window_size,
            "downsample": list(downsample),
        },
        "summary": summary,
        "videos": per_video,
        "execution": {
            "device": str(device),
            "gpu_external_action": False,
            "kaggle_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    temporary = args.output / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(args.output / "report.json")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    value.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--video-id", action="append", default=list(DEFAULT_VIDEO))
    value.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    value.add_argument("--unet-batch-size", type=int, default=1)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))

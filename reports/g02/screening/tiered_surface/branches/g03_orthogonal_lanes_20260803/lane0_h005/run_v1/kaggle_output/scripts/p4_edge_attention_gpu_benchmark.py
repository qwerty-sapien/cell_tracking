#!/usr/bin/env python3
"""Measure P4 wall-clock and CUDA-forward seconds per frame pair."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.edge_attention import EdgeResidualModel, attention_neighbours  # noqa: E402


SCHEMA = "celltrack-alpha.p4-edge-attention-gpu-benchmark.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/p4_edge_attention_prototype_v1/prepare.json"
DEFAULT_MODELS = ROOT / "reports/g02/p4_edge_attention_prototype_v1/models.pt"
DEFAULT_FRAMES = ROOT / "reports/g02/p4_edge_attention_prototype_v1/frames"
DEFAULT_OUTPUT = ROOT / "reports/g02/p4_edge_attention_gpu_benchmark_v1.json"
BUDGET_SECONDS_PER_PAIR = 3.6


class BenchmarkError(RuntimeError):
    """The actual-device benchmark contract was not satisfied."""


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def _load_models(path: Path, device: torch.device) -> dict[str, EdgeResidualModel]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "celltrack-alpha.p4-edge-attention-prototype.v1":
        raise BenchmarkError("P4 model artifact schema is invalid")
    hidden, heads = int(payload["hidden"]), int(payload["heads"])
    models = {
        "no_attention": EdgeResidualModel(attention=False, hidden=hidden, heads=heads).to(device),
        "attention": EdgeResidualModel(attention=True, hidden=hidden, heads=heads).to(device),
    }
    for name, model in models.items():
        model.load_state_dict(payload[name], strict=True)
        model.eval()
    return models


def _records(prepare_path: Path, frames_root: Path, max_pairs: int) -> list[dict[str, Any]]:
    prepare = json.loads(prepare_path.read_text(encoding="ascii"))
    if prepare.get("status") != "prepared" or prepare.get("scope", {}).get("held_out_used") is not False:
        raise BenchmarkError("fit-only P4 preparation report is invalid")
    records: list[dict[str, Any]] = []
    for video in prepare["videos"]:
        identity = str(video["identity"])
        path = frames_root / f"{identity.replace(':', '__', 1)}.pt"
        if not path.is_file():
            raise BenchmarkError(f"prepared frame file is missing: {path}")
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
        if max_pairs and len(records) >= max_pairs:
            return records[:max_pairs]
    if len(records) != int(prepare["totals"]["supervised_frame_pairs"]):
        raise BenchmarkError("prepared frame count disagrees with its report")
    return records


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise BenchmarkError("timing sample is empty or nonfinite")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
        "maximum": float(max(values)),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _one(
    record: dict[str, Any],
    models: dict[str, EdgeResidualModel],
    device: torch.device,
) -> tuple[dict[str, float], int, int]:
    edge_count = len(record["source_index"])
    wall: dict[str, float] = {}

    _synchronize(device)
    started = time.perf_counter()
    features = torch.as_tensor(record["features"], device=device)
    geometry = torch.as_tensor(record["geometry_um"], device=device)
    base = torch.as_tensor(record["base_logits"], device=device)
    empty_index = torch.empty((edge_count, 0), dtype=torch.long, device=device)
    empty_mask = torch.empty((edge_count, 0), dtype=torch.bool, device=device)
    with torch.no_grad():
        output = base + models["no_attention"](
            features, geometry, empty_index, empty_mask
        )
    _synchronize(device)
    if output.shape != (edge_count,):
        raise BenchmarkError("no-attention benchmark output shape is invalid")
    wall["no_attention"] = time.perf_counter() - started

    _synchronize(device)
    started = time.perf_counter()
    neighbours_np, mask_np = attention_neighbours(
        record["source_index"], record["target_index"], record["geometry_um"]
    )
    neighbours = torch.as_tensor(neighbours_np, device=device)
    mask = torch.as_tensor(mask_np, device=device)
    features = torch.as_tensor(record["features"], device=device)
    geometry = torch.as_tensor(record["geometry_um"], device=device)
    base = torch.as_tensor(record["base_logits"], device=device)
    with torch.no_grad():
        output = base + models["attention"](features, geometry, neighbours, mask)
    _synchronize(device)
    if output.shape != (edge_count,):
        raise BenchmarkError("attention benchmark output shape is invalid")
    wall["attention"] = time.perf_counter() - started
    return wall, edge_count, int(mask_np.sum())


def _cuda_forward(
    record: dict[str, Any],
    models: dict[str, EdgeResidualModel],
    device: torch.device,
) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    edge_count = len(record["source_index"])
    neighbours_np, mask_np = attention_neighbours(
        record["source_index"], record["target_index"], record["geometry_um"]
    )
    tensors = {
        "features": torch.as_tensor(record["features"], device=device),
        "geometry": torch.as_tensor(record["geometry_um"], device=device),
        "base": torch.as_tensor(record["base_logits"], device=device),
        "neighbours": torch.as_tensor(neighbours_np, device=device),
        "mask": torch.as_tensor(mask_np, device=device),
        "empty_index": torch.empty((edge_count, 0), dtype=torch.long, device=device),
        "empty_mask": torch.empty((edge_count, 0), dtype=torch.bool, device=device),
    }
    result: dict[str, float] = {}
    for name in ("no_attention", "attention"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            if name == "attention":
                _ = tensors["base"] + models[name](
                    tensors["features"],
                    tensors["geometry"],
                    tensors["neighbours"],
                    tensors["mask"],
                )
            else:
                _ = tensors["base"] + models[name](
                    tensors["features"],
                    tensors["geometry"],
                    tensors["empty_index"],
                    tensors["empty_mask"],
                )
        end.record()
        end.synchronize()
        result[name] = float(start.elapsed_time(end) / 1000.0)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BenchmarkError(f"refusing to overwrite benchmark report: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA benchmark requested but CUDA is unavailable")
    if args.device != "cuda" and not args.allow_cpu_smoke:
        raise BenchmarkError("a non-CUDA run is smoke-only; pass --allow-cpu-smoke explicitly")
    if args.repeats <= 0 or args.warmup_pairs < 0 or args.max_pairs < 0:
        raise BenchmarkError("repeats must be positive and pair limits must be nonnegative")
    device = torch.device(args.device)
    models = _load_models(args.models, device)
    records = _records(args.prepare_report, args.frames_root, args.max_pairs)
    if not records:
        raise BenchmarkError("no benchmark frame pairs were loaded")

    warmup = records[: min(args.warmup_pairs, len(records))]
    for record in warmup:
        _one(record, models, device)
        _cuda_forward(record, models, device)
    _synchronize(device)

    wall = {"no_attention": [], "attention": []}
    cuda = {"no_attention": [], "attention": []}
    candidate_edges = attention_cells = 0
    started = time.perf_counter()
    for repeat in range(args.repeats):
        for record in records:
            observed, edges, cells = _one(record, models, device)
            kernel = _cuda_forward(record, models, device)
            for name in wall:
                wall[name].append(observed[name])
                if kernel:
                    cuda[name].append(kernel[name])
            if repeat == 0:
                candidate_edges += edges
                attention_cells += cells
    elapsed = time.perf_counter() - started
    actual_gpu = device.type == "cuda"
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "actual_gpu_measurement": actual_gpu,
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device) if actual_gpu else platform.processor(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "corpus": {
            "frame_pairs": len(records),
            "repeats": args.repeats,
            "timed_pair_executions": len(records) * args.repeats,
            "candidate_edges": candidate_edges,
            "attention_cells": attention_cells,
            "fit_only": True,
            "held_out_used": False,
            "selection": "all strict confusable training contexts" if not args.max_pairs else "prefix smoke subset",
        },
        "wall_clock_seconds_per_pair": {
            name: _summary(values) for name, values in wall.items()
        },
        "cuda_forward_seconds_per_pair": (
            {name: _summary(values) for name, values in cuda.items()} if actual_gpu else None
        ),
        "budget": {
            "seconds_per_pair": BUDGET_SECONDS_PER_PAIR,
            "attention_mean_passes": (
                _summary(wall["attention"])["mean"] <= BUDGET_SECONDS_PER_PAIR
                if actual_gpu
                else None
            ),
        },
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare-report", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    value.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    value.add_argument("--allow-cpu-smoke", action="store_true")
    value.add_argument("--warmup-pairs", type=int, default=10)
    value.add_argument("--repeats", type=int, default=3)
    value.add_argument("--max-pairs", type=int, default=0)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report, sort_keys=True))

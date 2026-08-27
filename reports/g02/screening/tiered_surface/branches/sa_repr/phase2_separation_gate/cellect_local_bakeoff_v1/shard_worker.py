#!/usr/bin/env python3
"""Compute one frozen CELLECT request shard and checkpoint it atomically."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore

import constants


SCHEMA = "celltrack-alpha.cellect-local-shard.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, *, a1: np.ndarray, a0: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, a1=a1, a0=a0)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_feature(path: Path, expected_count: int) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"a1", "a0"}:
            raise RuntimeError("atomic feature shard field drift")
        a1 = loaded["a1"]
        a0 = loaded["a0"]
    if (
        a1.shape != (expected_count, 8)
        or a0.shape != (expected_count, 8)
        or a1.dtype != np.float32
        or a0.dtype != np.float32
        or not np.isfinite(a1).all()
        or not np.isfinite(a0).all()
    ):
        raise RuntimeError("atomic feature shard shape or value drift")


def infer_video(
    model: torch.nn.Module,
    device: torch.device,
    array: zarr.Array,
    request: dict[str, np.ndarray],
    source_module: Any,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = len(request["source_id"])
    source_embedding = np.empty((count, 64), dtype=np.float32)
    target_embedding = np.empty((count, 64), dtype=np.float32)
    source_written = np.zeros(count, dtype=bool)
    target_written = np.zeros(count, dtype=bool)
    source_groups = source_module.group_endpoint_indices(
        request["source_t"], request["source_zyx"], target_role=False
    )
    target_groups = source_module.group_endpoint_indices(
        request["target_t"], request["target_zyx"], target_role=True
    )
    empty = np.empty(0, dtype=np.int64)
    tile_jobs = [
        (
            key[0],
            key[1] * constants.TILE_ZYX[0],
            key[2] * constants.TILE_ZYX[1],
            key[3] * constants.TILE_ZYX[2],
            source_groups.get(key, empty),
            target_groups.get(key, empty),
        )
        for key in sorted(source_groups.keys() | target_groups.keys())
    ]
    if len(tile_jobs) != source_module.count_tile_jobs(request):
        raise RuntimeError("tile grouping drift")
    for offset in range(0, len(tile_jobs), batch):
        jobs = tile_jobs[offset : offset + batch]
        volumes = []
        for frame, z0, y0, x0, _src, _tgt in jobs:
            forward = np.asarray(
                array[
                    frame : frame + 2,
                    z0 : z0 + constants.TILE_ZYX[0],
                    y0 : y0 + constants.TILE_ZYX[1],
                    x0 : x0 + constants.TILE_ZYX[2],
                ],
                dtype=np.float32,
            )
            if forward.shape != (2,) + constants.TILE_ZYX:
                raise RuntimeError(
                    f"raw tile boundary drift at {(frame, z0, y0, x0)}: {forward.shape}"
                )
            volumes.extend((forward, forward[::-1].copy()))
        tensor = (
            torch.from_numpy(np.stack(volumes))
            .permute(0, 1, 3, 4, 2)
            .to(device)
        )
        with torch.inference_mode():
            embedding = (
                model(source_module.preprocess(tensor))
                .embedding.float()
                .cpu()
                .numpy()
            )
        for job_index, (_frame, z0, y0, x0, src, tgt) in enumerate(jobs):
            forward = embedding[2 * job_index]
            reverse = embedding[2 * job_index + 1]
            if len(src):
                point = request["source_zyx"][src] - np.asarray((z0, y0, x0))
                source_embedding[src] = forward[
                    :, point[:, 1], point[:, 2], point[:, 0]
                ].T
                source_written[src] = True
            if len(tgt):
                point = request["target_zyx"][tgt] - np.asarray((z0, y0, x0))
                target_embedding[tgt] = reverse[
                    :, point[:, 1], point[:, 2], point[:, 0]
                ].T
                target_written[tgt] = True
        del tensor, embedding, volumes
    if not source_written.all() or not target_written.all():
        raise RuntimeError("endpoint feature coverage is incomplete")
    return source_embedding, target_embedding, len(tile_jobs)


def run(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CELLTRACK_CELLECT_GUARD_ACTIVE") != "1":
        raise RuntimeError("binding zero-swap guard is absent")
    if constants.sha256(args.plan) != args.plan_sha256:
        raise RuntimeError("frozen shard plan SHA-256 drift")
    plan = json.loads(args.plan.read_text(encoding="ascii"))
    rows = plan.get("requests", [])
    if not 0 <= args.request_index < len(rows):
        raise RuntimeError("request index is outside the frozen plan")
    row = rows[args.request_index]
    request_path = Path(row["request_path"])
    if (
        constants.sha256(request_path) != row["request_sha256"]
        or request_path.stat().st_size != row["request_bytes"]
    ):
        raise RuntimeError("declared request shard identity mismatch")
    constants.verify_pins()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if args.threads < 1:
        raise RuntimeError("CPU thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    output = args.feature_root / row["action_class"] / row["output_name"]
    receipt = (
        args.checkpoint_root
        / row["action_class"]
        / f"{Path(row['output_name']).stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or receipt.exists():
        raise RuntimeError("final shard or checkpoint already exists")

    if str(constants.GATE) not in sys.path:
        sys.path.insert(0, str(constants.GATE))
    source_module = load_module(
        constants.SOURCE_FEATURIZER, "pinned_cellect_action_features"
    )
    model_module = load_module(constants.SAFE_LOADER, "pinned_cellect_safe_loader")
    device = torch.device(args.device)
    started = time.monotonic()
    model = model_module.load_published_backbone(
        constants.CHECKPOINT, device=device
    ).eval()
    with np.load(request_path, allow_pickle=False) as loaded:
        request = {key: loaded[key] for key in loaded.files}
    dataset = row["identity"].split(":", 1)[1]
    with ZipStore(constants.RAW_ARCHIVE, mode="r") as store:
        array = zarr.open_array(
            store=store, path=f"train/{dataset}.zarr/0", mode="r"
        )
        source, target, tile_jobs = infer_video(
            model, device, array, request, source_module, args.batch
        )
    a1 = source_module.action_features(source, target)
    a0_source = np.empty_like(source)
    a0_target = np.empty_like(target)
    for frame in np.unique(request["source_t"]):
        index = np.flatnonzero(request["source_t"] == frame)
        a0_source[index] = source_module.permutation(
            source[index],
            request["source_id"][index],
            row["identity"],
            int(frame),
        )
    for frame in np.unique(request["target_t"]):
        index = np.flatnonzero(request["target_t"] == frame)
        a0_target[index] = source_module.permutation(
            target[index],
            request["target_id"][index],
            row["identity"],
            int(frame),
        )
    a0 = source_module.action_features(a0_source, a0_target)
    if len(a1) != row["action_count"] or len(a0) != row["action_count"]:
        raise RuntimeError("action count disagrees with frozen plan")
    atomic_npz(output, a1=a1, a0=a0)
    validate_feature(output, row["action_count"])
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_ATOMIC",
        "completed_utc": now(),
        "ordinal": args.request_index,
        "identity": row["identity"],
        "action_class": row["action_class"],
        "count": row["action_count"],
        "request_path": row["request_path"],
        "request_sha256": row["request_sha256"],
        "output": output.name,
        "output_sha256": constants.sha256(output),
        "tile_jobs": tile_jobs,
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "batch": args.batch,
        "intraop_threads": args.threads,
        "interop_threads": 1,
        "checkpoint_bytes": constants.CHECKPOINT.stat().st_size,
        "checkpoint_sha256": constants.PINS[constants.CHECKPOINT],
        "safe_loader_sha256": constants.PINS[constants.SAFE_LOADER],
        "source_featurizer_sha256": constants.PINS[constants.SOURCE_FEATURIZER],
        "model_released_at_process_exit": True,
        "full_pool_cache": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "kaggle": False,
        "runpod": False,
    }
    atomic_json(receipt, result)
    del model, request, source, target, a0_source, a0_target, a1, a0
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--request-index", type=int, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--batch", type=int, default=constants.DEFAULT_BATCH)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute one frozen DINOv2 request shard and checkpoint it atomically."""

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


SCHEMA = "celltrack-alpha.dinov2-local-shard.v3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {path}")
    module = importlib.util.module_from_spec(spec)
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


def run(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CELLTRACK_DINO_GUARD_ACTIVE") != "1":
        raise RuntimeError("binding zero-swap guard is absent")
    if constants.sha256(args.plan) != args.plan_sha256:
        raise RuntimeError("frozen shard plan SHA-256 drift")
    plan = json.loads(args.plan.read_text(encoding="ascii"))
    rows = plan.get("requests", [])
    if not 0 <= args.request_index < len(rows):
        raise RuntimeError("request index is outside the frozen plan")
    row = rows[args.request_index]
    request_path = constants.ROOT / row["request_path"]
    if (
        constants.sha256(request_path) != row["request_sha256"]
        or request_path.stat().st_size != row["request_bytes"]
    ):
        raise RuntimeError("declared request shard identity mismatch")
    if constants.RAW_ARCHIVE.stat().st_size != plan["raw_archive_bytes"]:
        raise RuntimeError("raw archive byte-size drift")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    output = args.feature_root / row["action_class"] / request_path.name
    receipt = args.checkpoint_root / row["action_class"] / f"{request_path.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or receipt.exists():
        raise RuntimeError("final shard or checkpoint already exists")

    materializer = load_module(constants.MATERIALIZER, "pinned_materializer_v3")
    featurizer = load_module(constants.FEATURIZER, "pinned_featurizer_v3")
    request = materializer.load_request(request_path)
    unique_rows = np.unique(
        np.concatenate(
            (
                materializer.endpoint_rows(request, "source"),
                materializer.endpoint_rows(request, "target"),
            )
        ),
        axis=0,
    )
    device = torch.device(args.device)
    started = time.monotonic()
    model = materializer.load_model(constants.SOURCE, constants.CHECKPOINT, device)
    dataset = row["identity"].split(":", 1)[1]
    with ZipStore(constants.RAW_ARCHIVE, mode="r") as store:
        array = zarr.open_array(
            store=store, path=f"train/{dataset}.zarr/0", mode="r"
        )
        embeddings, quantiles = materializer.embed_video(
            model,
            device,
            array,
            unique_rows,
            args.crop_batch,
            args.forward_batch,
            started,
            0,
            len(unique_rows),
        )
    source_index = materializer.map_rows(
        unique_rows, materializer.endpoint_rows(request, "source")
    )
    target_index = materializer.map_rows(
        unique_rows, materializer.endpoint_rows(request, "target")
    )
    a1, a0 = materializer.permuted_features(
        featurizer,
        request,
        embeddings[source_index],
        embeddings[target_index],
        row["identity"],
    )
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
        "unique_endpoint_count": len(unique_rows),
        "normalization_q01": quantiles[0],
        "normalization_q99": quantiles[1],
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
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
    del model, request, unique_rows, embeddings, a1, a0
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
    parser.add_argument("--crop-batch", type=int, default=16)
    parser.add_argument("--forward-batch", type=int, default=16)
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

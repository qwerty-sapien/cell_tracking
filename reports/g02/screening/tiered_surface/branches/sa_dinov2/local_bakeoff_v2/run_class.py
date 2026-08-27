#!/usr/bin/env python3
"""Materialize one frozen DINOv2 separation class, releasing every video."""

from __future__ import annotations

import argparse
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


SCHEMA = "celltrack-alpha.dinov2-local-bakeoff-class.v2"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_plan(materializer: Any, action_class: str) -> tuple[list[tuple[str, Path]], int]:
    rows: list[tuple[str, Path]] = []
    endpoints = 0
    request_root = constants.GATE / "run_v1/feature_requests" / action_class
    for path in sorted(request_root.glob("*.npz")):
        identity = path.stem.replace("__", ":", 1)
        request = materializer.load_request(path)
        unique = np.unique(
            np.concatenate(
                (
                    materializer.endpoint_rows(request, "source"),
                    materializer.endpoint_rows(request, "target"),
                )
            ),
            axis=0,
        )
        endpoints += len(unique)
        rows.append((identity, path))
    return rows, endpoints


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if os.environ.get("CELLTRACK_DINO_GUARD_ACTIVE") != "1":
        raise RuntimeError("binding class subprocess guard is absent")
    frozen = constants.verify_pins()
    materializer = load_module(constants.MATERIALIZER, "pinned_dinov2_materializer")
    featurizer = load_module(constants.FEATURIZER, "pinned_dinov2_featurizer")
    manifest, expected_requests = materializer.authenticate_inputs(
        constants.RAW_ARCHIVE, constants.SOURCE, constants.CHECKPOINT
    )
    plan, endpoint_total = class_plan(materializer, args.action_class)
    expected_files = len(manifest["classes"][args.action_class]["videos"])
    if len(plan) != expected_files:
        raise RuntimeError("class request inventory drift")
    if sum(row["count"] for row in manifest["classes"][args.action_class]["videos"]) != constants.EXPECTED_ACTIONS[args.action_class]:
        raise RuntimeError("class action count drift")
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    started = time.monotonic()
    model = materializer.load_model(constants.SOURCE, constants.CHECKPOINT, device)
    args.output.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    completed = 0
    for ordinal, (identity, request_path) in enumerate(plan, start=1):
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
        dataset = identity.split(":", 1)[1]
        # One store/array/frame heap per video. Nothing image-sized crosses this block.
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
                completed,
                endpoint_total,
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
            identity,
        )
        destination = args.output / request_path.name
        np.savez_compressed(destination, a1=a1, a0=a0)
        files.append(
            {
                "identity": identity,
                "action_class": args.action_class,
                "count": len(a1),
                "request_sha256": expected_requests[(args.action_class, request_path.name)],
                "output": destination.name,
                "output_sha256": constants.sha256(destination),
            }
        )
        completed += len(unique_rows)
        del request, unique_rows, embeddings, source_index, target_index, a1, a0, array
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        print(
            f"{ordinal}/{len(plan)} {identity} endpoints={files[-1]['count']} "
            f"q01={quantiles[0]:.6f} q99={quantiles[1]:.6f}",
            flush=True,
        )
    if completed != endpoint_total or len(files) != expected_files:
        raise RuntimeError("terminal class inventory drift")
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "supplier_id": "F0-DINOV2-VITS14-XY-V1",
        "action_class": args.action_class,
        "amendment_sha256": constants.PINS[constants.AMENDMENT],
        "sealed_B0_sha256": constants.PINS[constants.SEALED_B0],
        "population_manifest_sha256": constants.PINS[constants.POPULATION_MANIFEST],
        "supplier_contract_sha256": constants.PINS[constants.STAGING_CONTRACT],
        "frozen_inputs": frozen,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "crop_batch": args.crop_batch,
        "forward_batch": args.forward_batch,
        "unique_endpoint_count": endpoint_total,
        "file_count": len(files),
        "action_count": constants.EXPECTED_ACTIONS[args.action_class],
        "files": files,
        "elapsed_seconds": time.monotonic() - started,
        "per_video_load_release": True,
        "full_pool_cache": False,
        "endpoint_cache_persisted": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
    }
    (args.output / "class_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": "COMPLETE", "class": args.action_class, "files": len(files)}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-class", choices=constants.CLASSES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--crop-batch", type=int, default=16)
    parser.add_argument("--forward-batch", type=int, default=16)
    args = parser.parse_args()
    if args.crop_batch <= 0 or args.forward_batch <= 0:
        raise SystemExit("batch sizes must be positive")
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
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

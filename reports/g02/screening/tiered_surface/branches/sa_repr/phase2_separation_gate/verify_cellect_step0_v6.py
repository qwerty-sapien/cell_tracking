#!/usr/bin/env python3
"""Run the exact CELLECT v6 step-0 contract without importing heavy dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_step0_preflight import run_cellect_step0

BUNDLE_MANIFEST_SHA256 = (
    "ae3bc9b2bbf5e7b4e544bcb44159a2fc0d6250b3178a81192f2d8e9f867fecd3"
)
CHECKPOINT_SHA256 = (
    "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
)
POPULATION_MANIFEST_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
REQUEST_MANIFEST_SHA256 = (
    "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
)
WHEEL_SHA256 = {
    "donfig-0.8.1.post1-py3-none-any.whl": (
        "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
    ),
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"
    ),
    "zarr-3.2.1-py3-none-any.whl": (
        "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_cellect_step0(
        args.input_root,
        args.work_root,
        expected_bundle_manifest_sha256=BUNDLE_MANIFEST_SHA256,
        expected_request_manifest_sha256=REQUEST_MANIFEST_SHA256,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_population_manifest_sha256=POPULATION_MANIFEST_SHA256,
        expected_wheels=WHEEL_SHA256,
    )
    args.receipt_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pure-stdlib remote mount attestation for the DINOv2 R5 CPU lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


DATASET_SOURCE = "onebitbrain/g02-dinov2-r5-cpu-input-v1"
DATASET_VERSION = 1
DATASET_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
)
EXPECTED_ARCHIVE_SHA256 = "580499a2e0f35965cc4a7b1a5716bde09671a3ea839cbea06f6160b24c22ca40"
EXPECTED_MANIFEST_SHA256 = "5801cab07ecd63d948efefa605b0a60b541bead6dfaafd06053d467d0e0c25bf"
EXPECTED_INVENTORY_SHA256 = "546cbf270a3e430ad5e35cccaad8778f33a11eeb149134c48e1721cc8dae1992"
EXPECTED_POPULATION_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
EXPECTED_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
EXPECTED_SOURCE_SHA256 = "e1b9403af2976443baeeab0ede0de0c53b63cf209c292bcdeccdc672674148c7"
EXPECTED_REQUEST_COUNT = 1391458


class Step0Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step0(input_root: Path = Path("/kaggle/input")) -> dict:
    started = time.perf_counter()
    dataset_root = (
        DATASET_ROOT
        if input_root == Path("/kaggle/input")
        else input_root / "datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
    )
    manifest_path = dataset_root / "dinov2_cpu_bundle_manifest.json"
    bundle_root = dataset_root / "dinov2_cpu_bundle"
    inventory_path = bundle_root / "dinov2_cpu_bundle_inventory.json"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise Step0Error("KFG-13 exact owner-qualified DINOv2 mount missing")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise Step0Error("DINOv2 input manifest SHA mismatch")
    if sha256(inventory_path) != EXPECTED_INVENTORY_SHA256:
        raise Step0Error("DINOv2 input inventory SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    records = {row["path"]: row for row in inventory.get("files", [])}
    if (
        len(records) != 166
        or manifest.get("archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256
        or manifest.get("checkpoint", {}).get("sha256") != EXPECTED_WEIGHT_SHA256
        or manifest.get("source", {}).get("sha256") != EXPECTED_SOURCE_SHA256
        or manifest.get("population_manifest_sha256") != EXPECTED_POPULATION_SHA256
        or manifest.get("request_count") != EXPECTED_REQUEST_COUNT
        or manifest.get("request_file_count") != 77
        or manifest.get("population_file_count") != 77
        or manifest.get("wheel_count") != 3
        or manifest.get("enable_gpu") is not False
    ):
        raise Step0Error("DINOv2 frozen mount contract drifted")
    request_paths = [
        path
        for path in records
        if "/feature_requests/" in f"/{path}" and path.endswith(".npz")
    ]
    population_files = [
        path
        for path in records
        if "/population/" in f"/{path}" and path.endswith(".npz")
    ]
    population_manifests = [
        path for path in records if path.endswith("/population_manifest.json")
    ]
    wheels = [path for path in records if path.endswith(".whl")]
    if (
        len(request_paths) != 77
        or len(population_files) != 77
        or len(population_manifests) != 1
        or len(wheels) != 3
    ):
        raise Step0Error("DINOv2 request/population/wheel inventory drifted")
    required_paths = (
        [manifest["checkpoint"]["path"], manifest["source"]["path"]]
        + population_manifests
        + wheels
    )
    for relative in required_paths:
        path = bundle_root / relative
        record = records[relative]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise Step0Error(f"required mounted file missing or wrong-sized: {relative}")
    population_path = bundle_root / population_manifests[0]
    if sha256(population_path) != EXPECTED_POPULATION_SHA256:
        raise Step0Error("population manifest SHA mismatch")
    competition_roots = (
        input_root
        / "competitions"
        / "biohub-cell-tracking-during-development",
        input_root / "biohub-cell-tracking-during-development",
    )
    competition_root = next(
        (path for path in competition_roots if (path / "train").is_dir()), None
    )
    if competition_root is None:
        raise Step0Error("competition train mount missing")
    elapsed = time.perf_counter() - started
    return {
        "schema": "celltrack-alpha.kaggle-remote-mount-attestation.v1",
        "status": "PASS",
        "dataset_source": DATASET_SOURCE,
        "dataset_version": DATASET_VERSION,
        "dataset_status": "ready",
        "dataset_root": str(dataset_root),
        "input_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "population_manifest_sha256": EXPECTED_POPULATION_SHA256,
        "checkpoint_sha256": EXPECTED_WEIGHT_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "request_count": EXPECTED_REQUEST_COUNT,
        "request_file_count": len(request_paths),
        "population_file_count": len(population_files),
        "wheel_count": len(wheels),
        "competition_root": str(competition_root),
        "step0_elapsed_seconds": elapsed,
        "step0_limit_seconds": 5.0,
        "enable_gpu": False,
        "heavy_work_started": False,
    }


STEP0 = run_step0()
if STEP0["step0_elapsed_seconds"] >= 5.0:
    raise SystemExit("remote mount step 0 exceeded five seconds")
Path("/kaggle/working/KAGGLE_REMOTE_MOUNT_ATTESTATION.json").write_text(
    json.dumps(STEP0, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(STEP0, sort_keys=True), flush=True)

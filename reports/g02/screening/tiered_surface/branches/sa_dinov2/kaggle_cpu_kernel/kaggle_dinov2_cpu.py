#!/usr/bin/env python3
"""Run frozen DINOv2 R5 XY extraction on a Kaggle CPU session."""

from __future__ import annotations

# Everything above run_step0 is standard library; no pip, NumPy, Torch, zarr,
# model load, or inference may precede the authenticated input block.
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace


EXPECTED_ARCHIVE_SHA256 = "580499a2e0f35965cc4a7b1a5716bde09671a3ea839cbea06f6160b24c22ca40"
EXPECTED_MANIFEST_SHA256 = "5801cab07ecd63d948efefa605b0a60b541bead6dfaafd06053d467d0e0c25bf"
EXPECTED_INVENTORY_SHA256 = "546cbf270a3e430ad5e35cccaad8778f33a11eeb149134c48e1721cc8dae1992"
EXPECTED_POPULATION_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
EXPECTED_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
EXPECTED_SOURCE_SHA256 = "e1b9403af2976443baeeab0ede0de0c53b63cf209c292bcdeccdc672674148c7"
EXPECTED_DATASET_SOURCE = "onebitbrain/g02-dinov2-r5-cpu-input-v1"
EXPECTED_DATASET_VERSION = 1
EXPECTED_DATASET_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
)


class Step0Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one_input(input_root: Path, pattern: str, label: str) -> Path:
    matches = sorted(input_root.glob(pattern))
    if len(matches) != 1:
        raise Step0Error(f"{label}: expected one input, found {len(matches)}")
    return matches[0]


def run_step0(input_root: Path = Path("/kaggle/input")) -> dict:
    started = time.perf_counter()
    dataset_root = (
        EXPECTED_DATASET_ROOT
        if input_root == Path("/kaggle/input")
        else input_root / "datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
    )
    manifest_path = dataset_root / "dinov2_cpu_bundle_manifest.json"
    if not manifest_path.is_file():
        raise Step0Error("exact DINOv2 private-dataset manifest missing")
    bundle_root = manifest_path.parent / "dinov2_cpu_bundle"
    inventory_path = bundle_root / "dinov2_cpu_bundle_inventory.json"
    if not bundle_root.is_dir() or not inventory_path.is_file():
        raise Step0Error("Kaggle-expanded DINOv2 bundle missing")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    archive = manifest.get("archive", {})
    checkpoint = manifest.get("checkpoint", {})
    source = manifest.get("source", {})
    if (
        sha256(manifest_path) != EXPECTED_MANIFEST_SHA256
        or sha256(inventory_path) != EXPECTED_INVENTORY_SHA256
        or manifest.get("status") != "FROZEN_BEFORE_KAGGLE_CPU_OUTCOMES"
        or manifest.get("enable_gpu") is not False
        or manifest.get("ram_gib") != 16
        or archive.get("sha256") != EXPECTED_ARCHIVE_SHA256
        or checkpoint.get("sha256") != EXPECTED_WEIGHT_SHA256
        or int(checkpoint.get("bytes", -1)) != 88_283_115
        or source.get("sha256") != EXPECTED_SOURCE_SHA256
        or source.get("revision")
        != "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        or source.get("tree")
        != "2a27257b79b0633b027a21014bc9360e3c1b3f43"
        or manifest.get("population_manifest_sha256")
        != EXPECTED_POPULATION_SHA256
        or int(manifest.get("request_count", 0)) != 1_391_458
        or int(manifest.get("request_file_count", 0)) != 77
        or int(manifest.get("population_file_count", 0)) != 77
        or int(manifest.get("wheel_count", 0)) != 3
        or int(manifest.get("unique_endpoint_count", 0)) != 1_417_863
    ):
        raise Step0Error("DINOv2 required-input contract drifted")
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    rows = inventory.get("files", [])
    if (
        not isinstance(rows, list)
        or len(rows) + 1 != int(archive.get("member_count", -1))
    ):
        raise Step0Error("DINOv2 inventory coverage drifted")
    records = {str(row.get("path", "")): row for row in rows}
    if len(records) != len(rows):
        raise Step0Error("DINOv2 inventory has duplicate paths")
    weight_path = bundle_root / str(checkpoint["path"])
    source_path = bundle_root / str(source["path"])
    population_paths = [
        path for path in records if path.endswith("population_manifest.json")
    ]
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
    wheel_paths = [path for path in records if path.endswith(".whl")]
    if (
        not weight_path.is_file()
        or weight_path.stat().st_size != 88_283_115
        or not source_path.is_file()
        or source_path.stat().st_size != int(source.get("bytes", -1))
        or records[str(checkpoint["path"])].get("sha256")
        != EXPECTED_WEIGHT_SHA256
        or records[str(source["path"])].get("sha256")
        != EXPECTED_SOURCE_SHA256
        or len(population_paths) != 1
        or sha256(bundle_root / population_paths[0]) != EXPECTED_POPULATION_SHA256
        or len(request_paths) != 77
        or len(population_files) != 77
        or len(wheel_paths) != 3
    ):
        raise Step0Error("DINOv2 expanded input set drifted")
    competition_candidates = (
        input_root / "competitions/biohub-cell-tracking-during-development",
        input_root / "biohub-cell-tracking-during-development",
    )
    competition = next(
        (path for path in competition_candidates if (path / "train").is_dir()),
        None,
    )
    required = manifest.get("required_train_video_ids", [])
    if competition is None or len(required) != 41:
        raise Step0Error("competition source or frozen DINOv2 coverage missing")
    return {
        "schema": "celltrack-alpha.dinov2-r5-kaggle-cpu-step0.v1",
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "bundle_root": str(bundle_root),
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "population_manifest_sha256": EXPECTED_POPULATION_SHA256,
        "request_count": int(manifest["request_count"]),
        "request_file_count": len(request_paths),
        "wheel_count": len(wheel_paths),
        "weight_path": str(weight_path),
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "source_tar_path": str(source_path),
        "source_tar_sha256": EXPECTED_SOURCE_SHA256,
        "required_video_count": len(required),
        "competition_root": str(competition),
        "enable_gpu": False,
        "dataset_source": EXPECTED_DATASET_SOURCE,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "dataset_root": str(dataset_root),
        "heavy_work_started": False,
    }


STARTED = time.time()
WORKING = Path("/kaggle/working")
STEP0 = run_step0()
if STEP0["elapsed_seconds"] >= 5.0:
    raise SystemExit("DINOv2 step 0 exceeded the five-second envelope")
(WORKING / "DINOV2_STEP0_PASS.json").write_text(
    json.dumps(STEP0, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)

# Heavy work begins only after step 0.
bundle_root = Path(STEP0["bundle_root"])
wheel_root = bundle_root / "external/Tracking_Support/wheels"
install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheel_root),
        "zarr==3.2.1",
    ],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(install.stdout, flush=True)
if install.returncode != 0:
    raise SystemExit("DINOv2 offline dependency installation failed")

source_parent = WORKING / "dinov2_source_checkout"
source_parent.mkdir()
with tarfile.open(STEP0["source_tar_path"], "r") as handle:
    handle.extractall(source_parent, filter="data")
source = source_parent / "dinov2_source"
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
tree = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
).strip()
if (
    revision != "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    or tree != "2a27257b79b0633b027a21014bc9360e3c1b3f43"
):
    raise SystemExit("DINOv2 extracted source identity drifted")

here = (
    bundle_root
    / "reports/g02/screening/tiered_surface/branches/sa_dinov2"
)
materializer_path = here / "materialize_dinov2_xy_separation.py"
spec = importlib.util.spec_from_file_location("dinov2_r5_cpu", materializer_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load frozen DINOv2 materializer")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

gate = (
    bundle_root
    / "reports/g02/screening/tiered_surface/branches/sa_repr/"
    "phase2_separation_gate"
)
module.ROOT = bundle_root
module.HERE = here
module.GATE = gate
module.POPULATION_ROOT = gate / "run_v1/population"
module.REQUEST_ROOT = gate / "run_v1/feature_requests"
module.POPULATION_MANIFEST = gate / "run_v1/population_manifest.json"
module.AMENDMENT = gate / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
module.SEALED_B0 = gate / "run_v3/B0_SEALED.json"
module.STAGING_CONTRACT = here / "R5_XY_BETA_STAGING_CONTRACT.json"
module.FEATURIZER_PATH = here / "dinov2_xy_action_featurizer.py"
module.MAX_SECONDS = int(11.5 * 3600)
module.SOFT_STOP_SECONDS = module.MAX_SECONDS - 300

from zarr.storage import LocalStore


class ExtractedStore:
    def __init__(self, _path, mode="r"):
        if mode != "r":
            raise module.MaterializationError("raw store must be read-only")
        self.store = LocalStore(Path(STEP0["competition_root"]), read_only=True)

    def __enter__(self):
        return self.store

    def __exit__(self, exc_type, exc, traceback):
        close = getattr(self.store, "close", None)
        if callable(close):
            close()


module.ZipStore = ExtractedStore


def authenticate_transport(_raw_archive, source_path, checkpoint_path):
    pins = (
        (module.AMENDMENT, module.AMENDMENT_SHA256),
        (module.SEALED_B0, module.SEALED_B0_SHA256),
        (module.POPULATION_MANIFEST, module.POPULATION_MANIFEST_SHA256),
        (module.STAGING_CONTRACT, module.STAGING_CONTRACT_SHA256),
        (module.FEATURIZER_PATH, module.FEATURIZER_SHA256),
    )
    for path, expected in pins:
        if module.sha256(path) != expected:
            raise module.MaterializationError(f"frozen input hash mismatch: {path}")
    observed_revision = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_revision != module.SOURCE_REVISION:
        raise module.MaterializationError("DINOv2 source revision drift")
    if (
        checkpoint_path.stat().st_size != module.WEIGHT_BYTES
        or module.sha256(checkpoint_path) != module.WEIGHT_SHA256
    ):
        raise module.MaterializationError("DINOv2 checkpoint identity mismatch")
    manifest = json.loads(module.POPULATION_MANIFEST.read_text())
    expected_requests = {}
    for action_class in module.CLASSES:
        for row in manifest["classes"][action_class]["videos"]:
            request = bundle_root / row["request_path"]
            population = bundle_root / row["population_path"]
            if module.sha256(request) != row["request_sha256"]:
                raise module.MaterializationError("feature request hash mismatch")
            if module.sha256(population) != row["population_sha256"]:
                raise module.MaterializationError("population hash mismatch")
            expected_requests[(action_class, request.name)] = row["request_sha256"]
    observed_requests = {
        (path.parent.name, path.name)
        for path in module.REQUEST_ROOT.glob("*/*.npz")
    }
    if observed_requests != set(expected_requests):
        raise module.MaterializationError("feature-request inventory mismatch")
    return manifest, expected_requests


module.authenticate_inputs = authenticate_transport
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONHASHSEED"] = "0"
output = WORKING / "dinov2_r5_features"
status = "COMPLETE"
error = None
try:
    result = module.run(
        SimpleNamespace(
            raw_archive=Path(STEP0["competition_root"]),
            source=source,
            checkpoint=Path(STEP0["weight_path"]),
            output=output,
            device="cpu",
            crop_batch=64,
            forward_batch=64,
        )
    )
except Exception as exc:
    status = "FAIL_CLOSED"
    error = {"type": type(exc).__name__, "message": str(exc)}
    result = None

elapsed = time.time() - STARTED
terminal = {
    "schema": "celltrack-alpha.dinov2-r5-kaggle-cpu-terminal.v1",
    "status": status,
    "error": error,
    "elapsed_seconds": elapsed,
    "cpu_hours": elapsed / 3600.0,
    "enable_gpu": False,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "ram_gib": 16,
    "soft_stop_hours": 11.5,
    "step0": STEP0,
    "result_manifest_present": (output / "manifest.json").is_file(),
    "result_status": result.get("status") if isinstance(result, dict) else None,
    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    "gpu_budget_debit_hours": 0.0,
    "submission_action": False,
    "runpod_action": False,
    "full_pool_cache": False,
}
(WORKING / "DINOV2_CPU_TERMINAL.json").write_text(
    json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
if status != "COMPLETE":
    raise SystemExit(f"DINOv2 CPU run failed closed: {error}")

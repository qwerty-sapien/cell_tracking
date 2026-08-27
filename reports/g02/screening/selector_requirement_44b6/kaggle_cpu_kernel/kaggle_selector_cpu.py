#!/usr/bin/env python3
"""Run frozen DAG-2 selector requirement on a Kaggle CPU session."""

from __future__ import annotations

# Step 0 is deliberately before NumPy, zarr, tracksdata, pip, or scoring imports.
import json
import gzip
import hashlib
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time


EXPECTED_MANIFEST_SHA256 = "73a15e27f69f0968d0bd5876775663b95c09d7b06df9d7fbc5c17b89540c2450"
EXPECTED_INVENTORY_SHA256 = "fc21fe717b9eed0460daefada4815a370c7d696e5492b96a376b2a5c1341fa79"
EXPECTED_POPULATION_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
PLAIN_POPULATION_PATH = (
    "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
    "generated/action_population.jsonl"
)
GZIP_POPULATION_PATH = f"{PLAIN_POPULATION_PATH}.gz"
EXPECTED_PLAIN_POPULATION_SHA256 = (
    "f033b5ac34f615cf05ed48ed73efd0f27dc8f4f931f75b7dbf75ee4fc3c414eb"
)
DATASET_SLUG = "g02-selector-requirement-cpu-inputs-v1"
BOOTSTRAP_STARTED = time.perf_counter()


class Step0Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_roots(input_root: Path) -> tuple[Path, ...]:
    candidates = (
        input_root,
        input_root / DATASET_SLUG,
        input_root / "datasets" / "onebitbrain" / DATASET_SLUG,
    )
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def _one_manifest(input_root: Path) -> Path:
    matches = sorted(
        {
            root / "selector_cpu_bundle_manifest.json"
            for root in _dataset_roots(input_root)
            if (root / "selector_cpu_bundle_manifest.json").is_file()
        }
    )
    if len(matches) != 1:
        raise Step0Error(
            f"selector manifest: expected one exact-mount input, found {len(matches)}"
        )
    return matches[0]


def _safe_member_path(bundle_root: Path, value: object) -> tuple[str, Path]:
    relative = str(value)
    member = Path(relative)
    if (
        not relative
        or not relative.isascii()
        or member.is_absolute()
        or ".." in member.parts
    ):
        raise Step0Error(f"unsafe selector member path: {relative!r}")
    return relative, bundle_root / member


def run_step0(
    input_root: Path = Path("/kaggle/input"),
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_inventory_sha256: str = EXPECTED_INVENTORY_SHA256,
    expected_population_sha256: str = EXPECTED_POPULATION_SHA256,
    expected_plain_population_sha256: str = EXPECTED_PLAIN_POPULATION_SHA256,
) -> dict:
    started = time.perf_counter()
    symbol_elapsed = started - BOOTSTRAP_STARTED
    if not callable(run_step0) or symbol_elapsed >= 5.0:
        raise Step0Error("inlined step-0 resolver did not resolve before five seconds")
    manifest_path = _one_manifest(input_root)
    inventory_candidates = sorted(
        {
            path
            for path in (
                manifest_path.parent
                / "selector_cpu_bundle"
                / "selector_cpu_bundle_inventory.json",
                manifest_path.parent / "selector_cpu_bundle_inventory.json",
            )
            if path.is_file()
        }
    )
    if len(inventory_candidates) != 1:
        raise Step0Error(
            "Kaggle-expanded selector bundle inventory expected one match, "
            f"found {len(inventory_candidates)}"
        )
    inventory_path = inventory_candidates[0]
    bundle_root = inventory_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    record = manifest.get("archive", {})
    if (
        sha256(manifest_path) != expected_manifest_sha256
        or sha256(inventory_path) != expected_inventory_sha256
        or manifest.get("schema")
        != "celltrack-alpha.selector-kaggle-cpu-bundle.v2"
        or manifest.get("status") != "FROZEN_BEFORE_KAGGLE_CPU_OUTCOMES"
        or manifest.get("enable_gpu") is not False
        or manifest.get("ram_gib") != 16
        or manifest.get("population_manifest_sha256")
        != expected_population_sha256
        or int(manifest.get("population_action_count", 0)) <= 0
        or int(manifest.get("wheel_count", 0)) <= 0
        or manifest.get("inventory_sha256") != expected_inventory_sha256
        or record.get("format") != "uncompressed_tar"
        or record.get("compression") != "none"
        or record.get("verification") != "member_sha256_only"
        or "sha256" in record
    ):
        raise Step0Error("selector bundle identity or required-input contract drifted")
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    rows = inventory.get("files", [])
    if (
        inventory.get("schema")
        != "celltrack-alpha.selector-cpu-bundle-inventory.v2"
        or inventory.get("verification") != "member_sha256_only"
        or not isinstance(rows, list)
        or len(rows) + 1 != int(record.get("member_count", -1))
    ):
        raise Step0Error("selector inventory coverage drifted")
    expected_files = {"selector_cpu_bundle_inventory.json"}
    population_plain = None
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise Step0Error("selector inventory row is not an object")
        relative, path = _safe_member_path(bundle_root, row.get("path", ""))
        expected_hash = row.get("sha256")
        expected_bytes = row.get("bytes")
        if (
            relative in seen
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not path.is_file()
            or path.stat().st_size != expected_bytes
        ):
            raise Step0Error(f"selector member drifted: {relative}")
        if sha256(path) != expected_hash:
            raise Step0Error(f"selector member SHA-256 mismatch: {relative}")
        seen.add(relative)
        expected_files.add(relative)
        if relative == PLAIN_POPULATION_PATH:
            if (
                expected_hash != expected_plain_population_sha256
                or row.get("source_path") != GZIP_POPULATION_PATH
                or row.get("transport") != "gzip_payload_as_plain_member"
            ):
                raise Step0Error("plain population transport contract drifted")
            population_plain = str(path)
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if (
        actual_files != expected_files
        or population_plain is None
        or sha256(
            bundle_root
            / "reports/g02/screening/tiered_surface/branches/sa_repr/"
            "phase2_separation_gate/run_v1/population_manifest.json"
        )
        != expected_population_sha256
        or not any(name.endswith(".whl") for name in actual_files)
        or not any(name.endswith("population_manifest.json") for name in actual_files)
    ):
        raise Step0Error("selector expanded-file set drifted")
    competition_candidates = (
        input_root / "competitions/biohub-cell-tracking-during-development",
        input_root / "biohub-cell-tracking-during-development",
    )
    competition = next(
        (path for path in competition_candidates if (path / "train").is_dir()),
        None,
    )
    if competition is None:
        raise Step0Error("competition train Zarr root missing")
    required = manifest.get("required_train_video_ids", [])
    if not required or any(
        not (
            competition / "train" / f"{str(identity).split(':', 1)[1]}.zarr"
        ).is_dir()
        for identity in required
    ):
        raise Step0Error("competition source lacks a frozen selector video")
    return {
        "schema": "celltrack-alpha.selector-kaggle-cpu-step0.v2",
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "step0_resolver": "inlined_single_code_file",
        "step0_symbol_resolved": True,
        "step0_symbol_elapsed_seconds": symbol_elapsed,
        "bundle_root": str(bundle_root),
        "archive_format": "uncompressed_tar",
        "container_roundtrip_verified": False,
        "member_verification": "sha256",
        "manifest_sha256": expected_manifest_sha256,
        "inventory_sha256": expected_inventory_sha256,
        "population_manifest_sha256": expected_population_sha256,
        "population_action_count": int(manifest["population_action_count"]),
        "wheel_count": int(manifest["wheel_count"]),
        "required_video_count": len(required),
        "competition_root": str(competition),
        "population_plain_source": population_plain,
        "member_count_verified": len(rows),
        "enable_gpu": False,
        "heavy_work_started": False,
    }


STEP0_SYMBOL_RECEIPT = {
    "schema": "celltrack-alpha.selector-step0-symbol.v1",
    "status": "PASS",
    "resolver": "inlined_single_code_file",
    "resolved": callable(run_step0),
    "elapsed_seconds": time.perf_counter() - BOOTSTRAP_STARTED,
}
if (
    not STEP0_SYMBOL_RECEIPT["resolved"]
    or STEP0_SYMBOL_RECEIPT["elapsed_seconds"] >= 5.0
):
    raise SystemExit("inlined step-0 resolver did not resolve before five seconds")
print(json.dumps(STEP0_SYMBOL_RECEIPT, sort_keys=True), flush=True)

STARTED = time.time()
WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "selector_cpu_run"
STEP0 = run_step0()
(WORKING / "SELECTOR_STEP0_PASS.json").write_text(
    json.dumps(STEP0, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)

# Heavy work begins only after step 0 has authenticated every required input.
REPO = WORKING / "selector_repo"
if REPO.exists() or RUN_ROOT.exists():
    raise SystemExit("refusing to reuse selector CPU working directories")
shutil.copytree(Path(STEP0["bundle_root"]), REPO)
population_plain = Path(STEP0["population_plain_source"])
relative_plain = population_plain.relative_to(Path(STEP0["bundle_root"]))
restored_gzip = REPO / f"{relative_plain}.gz"
with (REPO / relative_plain).open("rb") as source, restored_gzip.open("wb") as target:
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=target, compresslevel=9, mtime=0
    ) as compressed:
        shutil.copyfileobj(source, compressed, length=1 << 20)

wheel_root = REPO / "external/Tracking_Support/wheels"
install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheel_root),
        "tracksdata",
        "geff",
        "zarr==3.2.1",
    ],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(install.stdout, flush=True)
if install.returncode != 0:
    raise SystemExit("offline dependency installation failed")

environment = dict(os.environ)
environment.update(
    {
        "CELLTRACK_ROOT": str(REPO),
        "CELLTRACK_RAW_ROOT": STEP0["competition_root"],
        "CELLTRACK_GUARD_ACTIVE": "1",
        "CELLTRACK_HOLDOUT_ACCESS": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
)
HERE = REPO / "reports/g02/screening/selector_requirement_44b6"
RUN_ROOT.mkdir(parents=True)
classes = ("delete", "occupied_swap", "restore_topology")
stage_receipts = []
status = "COMPLETE"
reason = None

for action_class in classes:
    if time.time() - STARTED >= 11.5 * 3600:
        status = "FAIL_WALL_CLOCK_BEFORE_STAGE"
        reason = action_class
        break
    output = RUN_ROOT / action_class
    command = [
        sys.executable,
        str(HERE / "selector_requirement_class.py"),
        "--action-class",
        action_class,
        "--output",
        str(output),
    ]
    stage_started = time.time()
    process = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(process.stdout, flush=True)
    result = output / "CLASS_RESULT.json"
    stage_receipts.append(
        {
            "stage": action_class,
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - stage_started,
            "result_present": result.is_file(),
        }
    )
    if process.returncode != 0 or not result.is_file():
        status = "FAIL_CLASS"
        reason = action_class
        break

if status == "COMPLETE":
    aggregate = RUN_ROOT / "aggregate"
    command = [
        sys.executable,
        str(HERE / "selector_requirement_aggregate.py"),
        "--delete",
        str(RUN_ROOT / "delete"),
        "--occupied-swap",
        str(RUN_ROOT / "occupied_swap"),
        "--restore-topology",
        str(RUN_ROOT / "restore_topology"),
        "--output",
        str(aggregate),
    ]
    stage_started = time.time()
    process = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(process.stdout, flush=True)
    result = aggregate / "RESULT.json"
    stage_receipts.append(
        {
            "stage": "aggregate",
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - stage_started,
            "result_present": result.is_file(),
        }
    )
    if process.returncode != 0 or not result.is_file():
        status = "FAIL_AGGREGATE"
        reason = "aggregate"

terminal = {
    "schema": "celltrack-alpha.selector-requirement-kaggle-cpu-terminal.v1",
    "status": status,
    "reason": reason,
    "elapsed_seconds": time.time() - STARTED,
    "cpu_hours": (time.time() - STARTED) / 3600.0,
    "enable_gpu": False,
    "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
    "ram_gib": 16,
    "step0": STEP0,
    "stages": stage_receipts,
    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    "gpu_budget_debit_hours": 0.0,
    "submission_action": False,
    "runpod_action": False,
    "full_pool_cache": False,
}
(WORKING / "SELECTOR_CPU_TERMINAL.json").write_text(
    json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
if status != "COMPLETE":
    raise SystemExit(f"selector CPU run failed closed: {status}: {reason}")

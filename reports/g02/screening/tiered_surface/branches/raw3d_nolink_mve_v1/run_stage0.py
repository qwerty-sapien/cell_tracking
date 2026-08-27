#!/usr/bin/env python3
"""Run the preregistered RAW3D-NOLINK-MVE Stage-0 CPU falsifier once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
import zarr
from zarr.storage import ZipStore

import raw3d_features as feature_contract


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
GATE = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
DINO = ROOT / "reports/g02/screening/tiered_surface/branches/sa_dinov2"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
POPULATION_MANIFEST = GATE / "run_v1/population_manifest.json"
POPULATION_ROOT = GATE / "run_v1"
SEALED_B0 = GATE / "run_v3/B0_SEALED.json"
EVALUATOR = GATE / "evaluate_cellect_separation.py"
REQUIREMENT_SURFACE = (
    ROOT
    / "reports/g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json"
)
DINO_TERMINAL = DINO / "local_bakeoff_v3/TERMINAL_REPORT.json"
FEATURE_CODE = HERE / "raw3d_features.py"
RUN_CODE = HERE / "run_stage0.py"
PREREGISTRATION = HERE / "PREREGISTRATION.json"

PINS = {
    POPULATION_MANIFEST: "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    SEALED_B0: "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    EVALUATOR: "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
    REQUIREMENT_SURFACE: "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b",
}

RAW_ARCHIVE_BYTES = 87_393_127_165
RAW_ARCHIVE_BANKED_SHA256 = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
RAW_CENTRAL_DIRECTORY_SHA256 = "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77"
RAW_ENTRY_COUNT = 24_886
EXPECTED_VIDEO_COUNT = 41
EXPECTED_ACTIONS = 1_345_497
EXPECTED_BENEFICIAL = 1_118
EXPECTED_FOLD_ACTIONS = [244_465, 173_496, 311_859, 238_772, 376_905]
EXPECTED_FOLD_BENEFICIAL = [306, 99, 167, 195, 351]
STEP0_WARNING_SECONDS = 60.0
SESSION_TIMEBOX_SECONDS = 5_400.0
INCUMBENT_MEAN_AUC = 0.6704117434526917
ADVANCE_MARGIN_AUC = 0.010
ADVANCE_THRESHOLD_AUC = INCUMBENT_MEAN_AUC + ADVANCE_MARGIN_AUC
MEASURED_AUC_REQUIREMENT = 0.85
MEASURED_RETENTION_REQUIREMENT = 0.02039355992844365


class Stage0Error(RuntimeError):
    """Fail-closed Stage-0 contract violation."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def central_directory_identity(path: Path) -> tuple[str, int]:
    with zipfile.ZipFile(path) as archive:
        rows = [
            {
                "compressed_size": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "filename": info.filename,
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest(), len(rows)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Stage0Error(f"cannot load frozen module {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, *, a1: np.ndarray, a0: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, a1=a1, a0=a0)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def authenticate_inputs(preregistration_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    if sha256(PREREGISTRATION) != preregistration_sha256:
        raise Stage0Error("preregistration SHA-256 mismatch")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="ascii"))
    if (
        prereg.get("status") != "FROZEN_BEFORE_STAGE0_OUTCOME"
        or prereg.get("node_id") != "RAW3D-NOLINK-MVE"
    ):
        raise Stage0Error("preregistration is not the frozen Stage-0 authority")
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise Stage0Error(f"frozen input identity drift: {path}")
    implementation = prereg.get("implementation", {})
    for path in (FEATURE_CODE, RUN_CODE):
        row = implementation.get(path.name)
        if not isinstance(row, dict) or row.get("sha256") != sha256(path):
            raise Stage0Error(f"implementation pin drift: {path.name}")

    if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise Stage0Error("raw archive byte-size drift")
    directory_sha, entry_count = central_directory_identity(RAW_ARCHIVE)
    if (
        directory_sha != RAW_CENTRAL_DIRECTORY_SHA256
        or entry_count != RAW_ENTRY_COUNT
    ):
        raise Stage0Error("raw archive central-directory identity drift")

    manifest = json.loads(POPULATION_MANIFEST.read_text(encoding="ascii"))
    videos = manifest.get("classes", {}).get("delete", {}).get("videos", [])
    if len(videos) != EXPECTED_VIDEO_COUNT:
        raise Stage0Error("DELETE video inventory drift")
    action_count = 0
    beneficial_count = 0
    request_bytes = 0
    for row in videos:
        request = ROOT / row["request_path"]
        population = ROOT / row["population_path"]
        if not request.is_file() or not population.is_file():
            raise Stage0Error(f"declared input absent: {row['identity']}")
        if sha256(request) != row["request_sha256"]:
            raise Stage0Error(f"request SHA-256 drift: {row['identity']}")
        if sha256(population) != row["population_sha256"]:
            raise Stage0Error(f"population SHA-256 drift: {row['identity']}")
        action_count += int(row["count"])
        beneficial_count += int(row["beneficial"])
        request_bytes += request.stat().st_size
    if action_count != EXPECTED_ACTIONS or beneficial_count != EXPECTED_BENEFICIAL:
        raise Stage0Error("DELETE population count drift")

    sealed = json.loads(SEALED_B0.read_text(encoding="ascii"))
    b0 = sealed.get("results", {}).get("delete", {})
    fold_actions = [int(row["held_actions"]) for row in b0.get("folds", [])]
    fold_beneficial = [int(row["beneficial"]) for row in b0.get("folds", [])]
    if (
        b0.get("actions") != EXPECTED_ACTIONS
        or b0.get("beneficial") != EXPECTED_BENEFICIAL
        or fold_actions != EXPECTED_FOLD_ACTIONS
        or fold_beneficial != EXPECTED_FOLD_BENEFICIAL
    ):
        raise Stage0Error("sealed B0 identity or fold count drift")

    elapsed = time.monotonic() - started
    receipt = {
        "schema": "celltrack-alpha.raw3d-stage0-step0.v1",
        "status": "PASS",
        "completed_utc": now(),
        "elapsed_seconds": elapsed,
        "elapsed_warning_seconds": STEP0_WARNING_SECONDS,
        "elapsed_warning": elapsed > STEP0_WARNING_SECONDS,
        "elapsed_is_abort_condition": False,
        "hard_checks": {
            "all_declared_inputs_present": True,
            "all_declared_sha256_match": True,
            "request_count_positive": action_count > 0,
            "manifest_sha256_match": True,
            "sealed_B0_before_A1": True,
            "raw_archive_central_directory_match": True,
        },
        "declared": {
            "request_file_count": len(videos),
            "request_count": action_count,
            "request_bytes": request_bytes,
            "beneficial": beneficial_count,
            "raw_archive_bytes": RAW_ARCHIVE_BYTES,
            "raw_archive_banked_full_sha256": RAW_ARCHIVE_BANKED_SHA256,
            "raw_archive_central_directory_sha256": directory_sha,
            "raw_archive_entry_count": entry_count,
        },
        "preregistration_sha256": preregistration_sha256,
        "gpu": False,
        "training": False,
        "acquisition": False,
        "submission": False,
    }
    if not all(receipt["hard_checks"].values()):
        raise Stage0Error("Step 0 hard check failed")
    return manifest, receipt


def load_request(path: Path) -> dict[str, np.ndarray]:
    required = {
        "source_id",
        "target_id",
        "source_t",
        "target_t",
        "source_zyx",
        "target_zyx",
    }
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != required:
            raise Stage0Error(f"request field drift: {path}")
        request = {key: loaded[key] for key in loaded.files}
    count = len(request["source_id"])
    if any(len(request[key]) != count for key in required):
        raise Stage0Error(f"request alignment drift: {path}")
    return request


def endpoint_rows(request: dict[str, np.ndarray], role: str) -> np.ndarray:
    return np.column_stack(
        (
            request[f"{role}_t"].astype(np.int64),
            request[f"{role}_zyx"].astype(np.int64),
        )
    )


def structured_rows(rows: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(rows, dtype=np.int64)
    return contiguous.view(
        np.dtype([("t", "<i8"), ("z", "<i8"), ("y", "<i8"), ("x", "<i8")])
    ).reshape(-1)


def map_rows(unique_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    unique_structured = structured_rows(unique_rows)
    rows_structured = structured_rows(rows)
    indices = np.searchsorted(unique_structured, rows_structured)
    if np.any(indices >= len(unique_rows)) or not np.array_equal(
        unique_structured[indices], rows_structured
    ):
        raise Stage0Error("endpoint mapping incomplete")
    return indices


def permuted_action_features(
    request: dict[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    *,
    identity: str,
    preregistration_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    a1 = feature_contract.action_features(source, target)
    permuted_source = np.empty_like(source)
    permuted_target = np.empty_like(target)
    for role, values, output in (
        ("source", source, permuted_source),
        ("target", target, permuted_target),
    ):
        frames = request[f"{role}_t"]
        node_ids = request[f"{role}_id"]
        for frame in np.unique(frames):
            index = np.flatnonzero(frames == frame)
            output[index] = feature_contract.within_frame_permutation(
                values[index],
                node_ids[index],
                identity=identity,
                frame=int(frame),
                preregistration_sha256=preregistration_sha256,
            )
    a0 = feature_contract.action_features(permuted_source, permuted_target)
    return a1, a0


def materialize_video(
    store: ZipStore,
    row: dict[str, Any],
    feature_root: Path,
    preregistration_sha256: str,
    campaign_started: float,
) -> dict[str, Any]:
    if time.monotonic() - campaign_started >= SESSION_TIMEBOX_SECONDS:
        raise Stage0Error("Stage 0 exceeded the one-session time-box")
    started = time.monotonic()
    identity = row["identity"]
    request_path = ROOT / row["request_path"]
    request = load_request(request_path)
    source_rows = endpoint_rows(request, "source")
    target_rows = endpoint_rows(request, "target")
    unique_rows = np.unique(np.concatenate((source_rows, target_rows)), axis=0)
    dataset = identity.split(":", 1)[1]
    array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
    if tuple(array.shape[1:]) != (64, 256, 256) or str(array.dtype) != "uint16":
        raise Stage0Error(f"raw Zarr contract drift: {identity}: {array.shape}/{array.dtype}")

    endpoint = np.empty(
        (len(unique_rows), feature_contract.ENDPOINT_DIMENSION), dtype=np.float32
    )
    written = np.zeros(len(unique_rows), dtype=bool)
    frame_count = int(array.shape[0])
    for frame_index in np.unique(unique_rows[:, 0]):
        if time.monotonic() - campaign_started >= SESSION_TIMEBOX_SECONDS:
            raise Stage0Error("Stage 0 exceeded the one-session time-box")
        index = np.flatnonzero(unique_rows[:, 0] == frame_index)
        frame = np.asarray(array[int(frame_index)])
        endpoint[index] = feature_contract.endpoint_features(
            frame,
            unique_rows[index, 1:],
            time_fraction=float(frame_index) / max(frame_count - 1, 1),
        )
        written[index] = True
    if not written.all() or not np.isfinite(endpoint).all():
        raise Stage0Error(f"endpoint feature coverage incomplete: {identity}")

    source_index = map_rows(unique_rows, source_rows)
    target_index = map_rows(unique_rows, target_rows)
    a1, a0 = permuted_action_features(
        request,
        endpoint[source_index],
        endpoint[target_index],
        identity=identity,
        preregistration_sha256=preregistration_sha256,
    )
    if (
        a1.shape != (int(row["count"]), feature_contract.ACTION_DIMENSION)
        or a0.shape != a1.shape
    ):
        raise Stage0Error(f"action feature shape drift: {identity}")
    destination = feature_root / "delete" / request_path.name
    atomic_npz(destination, a1=a1, a0=a0)
    with np.load(destination, allow_pickle=False) as check:
        if (
            set(check.files) != {"a1", "a0"}
            or check["a1"].shape != a1.shape
            or check["a0"].shape != a0.shape
            or not np.isfinite(check["a1"]).all()
            or not np.isfinite(check["a0"]).all()
        ):
            raise Stage0Error(f"temporary feature shard failed validation: {identity}")
    receipt = {
        "identity": identity,
        "action_count": len(a1),
        "unique_endpoint_count": len(unique_rows),
        "frame_count": frame_count,
        "feature_sha256": sha256(destination),
        "feature_bytes": destination.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps({"stage0_video": receipt}, sort_keys=True), flush=True)
    return receipt


def summarize_result(
    class_result: dict[str, Any],
    *,
    preregistration_sha256: str,
    step0: dict[str, Any],
    feature_receipts: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    folds = sorted(class_result["folds"], key=lambda row: row["held_fold"])
    aucs = np.asarray([row["arms"]["A1"]["auc"] for row in folds], dtype=np.float64)
    retentions = np.asarray(
        [row["arms"]["A1"]["retention"] for row in folds], dtype=np.float64
    )
    a0_aucs = np.asarray([row["arms"]["A0"]["auc"] for row in folds], dtype=np.float64)
    selected_beneficial = sum(
        int(row["arms"]["A1"]["selected_beneficial"]) for row in folds
    )
    selected_harmful = sum(
        int(row["arms"]["A1"]["selected_harmful"]) for row in folds
    )
    mean_auc = float(aucs.mean())
    fold_floor_auc = float(aucs.min())
    mean_retention = float(retentions.mean())
    fold_floor_retention = float(retentions.min())
    full_population_retention = selected_beneficial / EXPECTED_BENEFICIAL
    advance_checks = {
        "mean_auc_at_least_incumbent_plus_margin": mean_auc >= ADVANCE_THRESHOLD_AUC,
        "fold_floor_auc_at_least_incumbent_plus_margin": (
            fold_floor_auc >= ADVANCE_THRESHOLD_AUC
        ),
    }
    advance = all(advance_checks.values())
    return {
        "schema": "celltrack-alpha.raw3d-nolink-mve-stage0-result.v1",
        "status": "COMPLETE",
        "completed_utc": now(),
        "node_id": "RAW3D-NOLINK-MVE",
        "stage": 0,
        "preregistration_sha256": preregistration_sha256,
        "execution_order": ["B0_SEALED", "RAW3D_A1", "PERMUTED_A0", "DECISION"],
        "population": {
            "action_class": "delete",
            "actions": EXPECTED_ACTIONS,
            "beneficial": EXPECTED_BENEFICIAL,
            "video_count": EXPECTED_VIDEO_COUNT,
            "fold_actions": EXPECTED_FOLD_ACTIONS,
            "fold_beneficial": EXPECTED_FOLD_BENEFICIAL,
        },
        "observed": {
            "A1_auc_by_fold": aucs.tolist(),
            "A1_auc_mean": mean_auc,
            "A1_auc_fold_floor": fold_floor_auc,
            "A1_retention_by_fold": retentions.tolist(),
            "A1_retention_mean": mean_retention,
            "A1_retention_fold_floor": fold_floor_retention,
            "A1_retention_full_population": full_population_retention,
            "A1_selected_beneficial": selected_beneficial,
            "A1_selected_harmful": selected_harmful,
            "A0_auc_by_fold": a0_aucs.tolist(),
            "A0_auc_mean": float(a0_aucs.mean()),
            "A1_minus_A0_auc_mean": mean_auc - float(a0_aucs.mean()),
        },
        "comparisons": {
            "measured_requirement": {
                "auc": MEASURED_AUC_REQUIREMENT,
                "retention": MEASURED_RETENTION_REQUIREMENT,
                "auc_pass": mean_auc >= MEASURED_AUC_REQUIREMENT,
                "retention_pass": mean_retention >= MEASURED_RETENTION_REQUIREMENT,
            },
            "incumbent_dinov2_xy": {
                "mean_auc": INCUMBENT_MEAN_AUC,
                "mean_retention": 0.0013071895424836603,
                "A1_auc_mean_delta": mean_auc - INCUMBENT_MEAN_AUC,
                "A1_retention_mean_delta": mean_retention - 0.0013071895424836603,
            },
        },
        "advance_rule": {
            "reference_auc": INCUMBENT_MEAN_AUC,
            "preregistered_margin": ADVANCE_MARGIN_AUC,
            "binding_mean_auc_threshold": ADVANCE_THRESHOLD_AUC,
            "binding_fold_floor_auc_threshold": ADVANCE_THRESHOLD_AUC,
            "checks": advance_checks,
            "pass": advance,
        },
        "verdict": (
            "ADVANCE_STAGE1_ASSET_PROVENANCE_GATE"
            if advance
            else "PARK_C_RAW3D_AND_CONSOLIDATE_CAMPAIGN"
        ),
        "downstream_authority": {
            "stage1_asset_provenance_open": advance,
            "stage2_representation_open": False,
            "stage3_training_open": False,
            "training_authority": False,
            "acquisition_authority": False,
        },
        "step0": step0,
        "feature_extraction": {
            "supplier_id": feature_contract.SUPPLIER_ID,
            "endpoint_dimension": feature_contract.ENDPOINT_DIMENSION,
            "action_dimension": feature_contract.ACTION_DIMENSION,
            "video_receipts": feature_receipts,
            "temporary_feature_shards_deleted_after_evaluation": True,
            "raw_voxel_or_crop_cache_written": False,
            "full_pool_cache": False,
        },
        "resource": {
            "device": "cpu",
            "gpu": False,
            "elapsed_seconds": time.monotonic() - started,
            "timebox_seconds": SESSION_TIMEBOX_SECONDS,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "hard_stops": {
            "submission": False,
            "training": False,
            "acquisition": False,
            "gpu": False,
            "runpod": False,
            "full_pool_cache": False,
            "plan_amendment": False,
            "dead_promotion": False,
        },
        "frozen_falsifier_result": class_result,
        "barrier": "HALT_AFTER_STAGE0_VERDICT",
    }


def run(preregistration_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise Stage0Error(f"refusing to overwrite {output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    manifest, step0 = authenticate_inputs(preregistration_sha256)
    print(json.dumps({"step0": step0}, sort_keys=True), flush=True)
    feature_receipts = []
    with tempfile.TemporaryDirectory(
        prefix="g02_raw3d_stage0_", dir="/private/tmp"
    ) as temporary:
        feature_root = Path(temporary) / "features"
        with ZipStore(RAW_ARCHIVE, mode="r") as store:
            for row in manifest["classes"]["delete"]["videos"]:
                feature_receipts.append(
                    materialize_video(
                        store,
                        row,
                        feature_root,
                        preregistration_sha256,
                        started,
                    )
                )
        if (
            len(feature_receipts) != EXPECTED_VIDEO_COUNT
            or sum(row["action_count"] for row in feature_receipts) != EXPECTED_ACTIONS
        ):
            raise Stage0Error("temporary feature inventory incomplete")
        evaluator = load_module(EVALUATOR, "frozen_separation_evaluator_raw3d")
        sealed = json.loads(SEALED_B0.read_text(encoding="ascii"))
        data = evaluator.load_class(POPULATION_ROOT, feature_root, "delete")
        class_result = evaluator.evaluate_class(
            data, "delete", sealed["results"]["delete"]
        )
        result = summarize_result(
            class_result,
            preregistration_sha256=preregistration_sha256,
            step0=step0,
            feature_receipts=feature_receipts,
            started=started,
        )
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "A1_auc_mean": result["observed"]["A1_auc_mean"],
                "A1_auc_fold_floor": result["observed"]["A1_auc_fold_floor"],
                "A1_retention_mean": result["observed"]["A1_retention_mean"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.preregistration_sha256, args.output)
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "celltrack-alpha.raw3d-stage0-error.v1",
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


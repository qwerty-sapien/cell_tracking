#!/usr/bin/env python3
"""Run 77 isolated CELLECT shards under one strict zero-swap baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

import constants


SCHEMA = "celltrack-alpha.cellect-isolated-feature-stage.v1"
HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
PLAN = RUN / "SHARD_PLAN.json"
FEATURES = RUN / "features"
CHECKPOINTS = RUN / "checkpoints"
GUARDS = RUN / "guards"


class FeatureStageError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def swap_used_bytes() -> int:
    text = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", text)
    swap = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if not page or not swap:
        raise FeatureStageError("cannot parse vm_stat swapouts")
    return int(page.group(1)) * int(swap.group(1))


def validate_preflight(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if (
        value.get("status") != "PASS_ZERO_SWAP_GROWTH"
        or float(value.get("swap_growth_mib", -1)) != 0
        or float(value.get("elapsed_seconds", 0)) < 45
        or time.time() - path.stat().st_mtime > 180
    ):
        raise FeatureStageError("fresh 45-second zero-swap preflight absent")
    return value


def count_tile_jobs(request: dict[str, np.ndarray]) -> int:
    source_keys = np.column_stack(
        (
            request["source_t"],
            request["source_zyx"][:, 0] // constants.TILE_ZYX[0],
            request["source_zyx"][:, 1] // constants.TILE_ZYX[1],
            request["source_zyx"][:, 2] // constants.TILE_ZYX[2],
        )
    )
    target_keys = np.column_stack(
        (
            request["target_t"] - 1,
            request["target_zyx"][:, 0] // constants.TILE_ZYX[0],
            request["target_zyx"][:, 1] // constants.TILE_ZYX[1],
            request["target_zyx"][:, 2] // constants.TILE_ZYX[2],
        )
    )
    return int(len(np.unique(np.concatenate((source_keys, target_keys)), axis=0)))


def build_plan() -> dict:
    frozen = constants.verify_pins()
    manifest = json.loads(constants.REQUEST_MANIFEST.read_text(encoding="ascii"))
    if (
        manifest.get("status") != "FROZEN_COMPLETE"
        or manifest.get("request_file_count") != 77
        or manifest.get("request_count") != 1_391_458
        or manifest.get("population_manifest_sha256")
        != constants.PINS[constants.POPULATION_MANIFEST]
    ):
        raise FeatureStageError("frozen request manifest drift")
    rows = []
    for ordinal, record in enumerate(manifest["requests"]):
        action_class = record["action_class"]
        request = constants.REQUEST_ROOT / record["path"]
        if action_class not in constants.CLASSES:
            raise FeatureStageError("unknown frozen action class")
        if (
            constants.sha256(request) != record["sha256"]
            or request.stat().st_size != record["bytes"]
        ):
            raise FeatureStageError(f"authenticated request drift: {request}")
        with np.load(request, allow_pickle=False) as loaded:
            request_arrays = {key: loaded[key] for key in loaded.files}
        if len(request_arrays["source_id"]) != record["action_count"]:
            raise FeatureStageError("request action count drift")
        rows.append(
            {
                "ordinal": ordinal,
                "action_class": action_class,
                "identity": record["identity"],
                "action_count": record["action_count"],
                "request_path": str(request),
                "request_bytes": record["bytes"],
                "request_sha256": record["sha256"],
                "output_name": record["identity"].replace(":", "__", 1) + ".npz",
                "tile_jobs": count_tile_jobs(request_arrays),
            }
        )
    for action_class in constants.CLASSES:
        selected = [row for row in rows if row["action_class"] == action_class]
        if (
            len(selected) != constants.EXPECTED_FILES[action_class]
            or sum(row["action_count"] for row in selected)
            != constants.EXPECTED_ACTIONS[action_class]
        ):
            raise FeatureStageError("per-class request population drift")
    if (
        len(rows) != 77
        or sum(row["action_count"] for row in rows) != 1_391_458
        or sum(row["tile_jobs"] for row in rows) != 217_595
    ):
        raise FeatureStageError("frozen 77-shard population or tile plan drift")
    return {
        "schema": "celltrack-alpha.cellect-shard-plan.v1",
        "status": "FROZEN_AUTHENTICATED",
        "frozen_inputs": frozen,
        "supplier_id": constants.SUPPLIER_ID,
        "supplier_sha256": constants.SUPPLIER_SHA256,
        "population_manifest_sha256": constants.PINS[
            constants.POPULATION_MANIFEST
        ],
        "sealed_B0_sha256": constants.PINS[constants.SEALED_B0],
        "amendment_sha256": constants.PINS[constants.AMENDMENT],
        "request_manifest_sha256": constants.PINS[constants.REQUEST_MANIFEST],
        "checkpoint_sha256": constants.PINS[constants.CHECKPOINT],
        "checkpoint_bytes": constants.CHECKPOINT.stat().st_size,
        "safe_loader_sha256": constants.PINS[constants.SAFE_LOADER],
        "source_featurizer_sha256": constants.PINS[
            constants.SOURCE_FEATURIZER
        ],
        "raw_archive_bytes": constants.RAW_ARCHIVE.stat().st_size,
        "request_file_count": len(rows),
        "request_count": sum(row["action_count"] for row in rows),
        "tile_jobs": sum(row["tile_jobs"] for row in rows),
        "requests": rows,
        "one_shard_per_process": True,
        "atomic_checkpointing": True,
        "two_stage_isolation": True,
        "full_pool_cache": False,
    }


def ensure_plan() -> tuple[dict, str]:
    candidate = build_plan()
    if PLAN.exists():
        observed = json.loads(PLAN.read_text(encoding="ascii"))
        if observed != candidate:
            raise FeatureStageError("existing shard plan disagrees with frozen inputs")
    else:
        atomic_json(PLAN, candidate)
    return candidate, constants.sha256(PLAN)


def validate_feature(path: Path, expected_count: int) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"a1", "a0"}:
            raise FeatureStageError("feature checkpoint field drift")
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
        raise FeatureStageError("feature checkpoint shape or value drift")


def validate_or_recover(row: dict) -> dict | None:
    output = FEATURES / row["action_class"] / row["output_name"]
    receipt = (
        CHECKPOINTS
        / row["action_class"]
        / f"{Path(row['output_name']).stem}.json"
    )
    if not output.exists() and not receipt.exists():
        return None
    if receipt.exists() and not output.exists():
        raise FeatureStageError(f"checkpoint exists without output: {receipt}")
    validate_feature(output, row["action_count"])
    if receipt.exists():
        value = json.loads(receipt.read_text(encoding="ascii"))
        if (
            value.get("status")
            not in ("COMPLETE_ATOMIC", "RECOVERED_ATOMIC_OUTPUT")
            or value.get("request_sha256") != row["request_sha256"]
            or value.get("output_sha256") != constants.sha256(output)
            or value.get("count") != row["action_count"]
            or value.get("tile_jobs") != row["tile_jobs"]
        ):
            raise FeatureStageError(f"checkpoint receipt drift: {receipt}")
        return value
    value = {
        "schema": "celltrack-alpha.cellect-local-shard.v1",
        "status": "RECOVERED_ATOMIC_OUTPUT",
        "completed_utc": now(),
        "ordinal": row["ordinal"],
        "identity": row["identity"],
        "action_class": row["action_class"],
        "count": row["action_count"],
        "request_path": row["request_path"],
        "request_sha256": row["request_sha256"],
        "output": output.name,
        "output_sha256": constants.sha256(output),
        "tile_jobs": row["tile_jobs"],
        "recovery_reason": (
            "process exited after atomic output rename but before receipt rename"
        ),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, value)
    return value


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def next_guard_path(row: dict) -> Path:
    base = (
        f"{row['ordinal']:03d}_{row['action_class']}_"
        f"{Path(row['output_name']).stem}"
    )
    attempt = 1
    while (GUARDS / f"{base}_attempt{attempt}.json").exists():
        attempt += 1
    return GUARDS / f"{base}_attempt{attempt}.json"


def guarded_shard(
    command: list[str], row: dict, *, global_swap_baseline: int
) -> dict:
    guard_path = next_guard_path(row)
    environment = dict(os.environ)
    environment["CELLTRACK_CELLECT_GUARD_ACTIVE"] = "1"
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=environment,
    )
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    lines = []
    peak_swap = global_swap_baseline
    forced = False
    while process.poll() is None:
        try:
            line = process.stdout.readline()
        except BlockingIOError:
            line = ""
        if line:
            lines.append(line)
            print(line, end="", flush=True)
        current = swap_used_bytes()
        peak_swap = max(peak_swap, current)
        if current > global_swap_baseline:
            forced = True
            terminate(process)
            break
        time.sleep(0.25)
    os.set_blocking(process.stdout.fileno(), True)
    remainder = process.stdout.read() or ""
    if remainder:
        lines.append(remainder)
        print(remainder, end="", flush=True)
    returncode = process.wait()
    ending_swap = swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    status = "PASS"
    if peak_swap > global_swap_baseline:
        status = "FAIL_SWAP_GROWTH"
    elif returncode != 0:
        status = "FAIL_COMMAND"
    result = {
        "schema": "celltrack-alpha.cellect-shard-resource-guard.v1",
        "status": status,
        "completed_utc": now(),
        "ordinal": row["ordinal"],
        "identity": row["identity"],
        "action_class": row["action_class"],
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "swap_baseline_mib": global_swap_baseline / 2**20,
        "swap_peak_mib": peak_swap / 2**20,
        "swap_end_mib": ending_swap / 2**20,
        "swap_growth_mib": (peak_swap - global_swap_baseline) / 2**20,
        "forced_stop_on_swap_growth": forced,
        "peak_child_rss_gib": (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
        ),
        "captured_output": "".join(lines),
    }
    atomic_json(guard_path, result)
    return result


def write_class_manifests(
    plan: dict, *, device: str, batch: int, threads: int
) -> dict[str, str]:
    result = {}
    for action_class in constants.CLASSES:
        rows = [
            row for row in plan["requests"] if row["action_class"] == action_class
        ]
        files = []
        for row in rows:
            checkpoint = validate_or_recover(row)
            if checkpoint is None:
                raise FeatureStageError(f"missing completed shard: {row['identity']}")
            files.append(
                {
                    "identity": row["identity"],
                    "action_class": action_class,
                    "count": row["action_count"],
                    "request_sha256": row["request_sha256"],
                    "output": row["output_name"],
                    "output_sha256": checkpoint["output_sha256"],
                    "tile_jobs": row["tile_jobs"],
                }
            )
        manifest = {
            "schema": "celltrack-alpha.cellect-local-bakeoff-class.v1",
            "status": "COMPLETE",
            "supplier_id": constants.SUPPLIER_ID,
            "supplier_sha256": constants.SUPPLIER_SHA256,
            "action_class": action_class,
            "amendment_sha256": constants.PINS[constants.AMENDMENT],
            "sealed_B0_sha256": constants.PINS[constants.SEALED_B0],
            "population_manifest_sha256": constants.PINS[
                constants.POPULATION_MANIFEST
            ],
            "checkpoint_sha256": constants.PINS[constants.CHECKPOINT],
            "safe_loader_sha256": constants.PINS[constants.SAFE_LOADER],
            "source_featurizer_sha256": constants.PINS[
                constants.SOURCE_FEATURIZER
            ],
            "device": device,
            "batch": batch,
            "intraop_threads": threads,
            "interop_threads": 1,
            "file_count": len(files),
            "action_count": sum(row["count"] for row in files),
            "tile_jobs": sum(row["tile_jobs"] for row in files),
            "files": files,
            "one_shard_per_process": True,
            "atomic_checkpointing": True,
            "two_stage_isolation": True,
            "full_pool_cache": False,
        }
        path = FEATURES / action_class / "class_manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise FeatureStageError(f"class manifest drift: {path}")
        else:
            atomic_json(path, manifest)
        result[action_class] = constants.sha256(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch", type=int, default=constants.DEFAULT_BATCH)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        preflight = validate_preflight(args.preflight)
        plan, plan_sha = ensure_plan()
        baseline = swap_used_bytes()
        printed_baseline = int(float(preflight["swap_end_mib"]) * 2**20)
        if abs(baseline - printed_baseline) > 16_384:
            raise FeatureStageError("swap baseline changed after idle preflight")
        completed = 0
        for row in plan["requests"]:
            if validate_or_recover(row) is not None:
                completed += 1
                print(
                    f"resume {completed}/77 {row['action_class']} "
                    f"{row['identity']}",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(HERE / "shard_worker.py"),
                "--plan",
                str(PLAN),
                "--plan-sha256",
                plan_sha,
                "--request-index",
                str(row["ordinal"]),
                "--feature-root",
                str(FEATURES),
                "--checkpoint-root",
                str(CHECKPOINTS),
                "--device",
                args.device,
                "--batch",
                str(args.batch),
                "--threads",
                str(args.threads),
            ]
            guard = guarded_shard(command, row, global_swap_baseline=baseline)
            if guard["status"] != "PASS":
                raise FeatureStageError(
                    f"shard {row['ordinal']} stopped under {guard['status']}; "
                    "rerun with a new 45-second preflight to resume"
                )
            if validate_or_recover(row) is None:
                raise FeatureStageError(
                    "guarded shard returned without atomic checkpoint"
                )
            completed += 1
            print(
                f"checkpoint {completed}/77 {row['action_class']} "
                f"{row['identity']}",
                flush=True,
            )
        manifests = write_class_manifests(
            plan, device=args.device, batch=args.batch, threads=args.threads
        )
        ending_swap = swap_used_bytes()
        if ending_swap != baseline:
            raise FeatureStageError("global zero-swap baseline changed at terminal")
        terminal = {
            "schema": SCHEMA,
            "status": "COMPLETE_77_ATOMIC_SHARDS_ZERO_SWAP",
            "completed_utc": now(),
            "preflight": {
                "path": str(args.preflight.resolve()),
                "sha256": constants.sha256(args.preflight),
            },
            "plan": {"path": str(PLAN), "sha256": plan_sha},
            "class_manifests": manifests,
            "request_file_count": 77,
            "request_count": 1_391_458,
            "tile_jobs": 217_595,
            "swap_baseline_mib": baseline / 2**20,
            "swap_end_mib": ending_swap / 2**20,
            "swap_growth_mib": 0.0,
            "device": args.device,
            "batch": args.batch,
            "intraop_threads": args.threads,
            "interop_threads": 1,
            "checkpoint_bytes": constants.CHECKPOINT.stat().st_size,
            "checkpoint_sha256": constants.PINS[constants.CHECKPOINT],
            "one_shard_per_process": True,
            "atomic_checkpointing": True,
            "two_stage_isolation": True,
            "full_pool_cache": False,
            "kaggle": False,
            "gpu_quota_hours": 0,
            "runpod": False,
        }
        atomic_json(RUN / "FEATURE_STAGE_TERMINAL.json", terminal)
        print(json.dumps(terminal, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "FAIL_CLOSED_RESUMABLE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

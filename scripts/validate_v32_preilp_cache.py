#!/usr/bin/env python3
"""Offline validation for sealed or producer-partial V32 replay caches.

This CLI never imports a scorer, opens ground truth, runs inference, or talks to
an external service.  ``sealed`` validates the complete durable contract.
``partial`` validates the sparse pre/post-ILP output emitted by the cache kernel
before a baseline submission fragment and top manifest are attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.v32_preilp_cache import (
    FRAME_PAIR_DTYPE,
    NODE_DTYPE,
    POST_EDGE_DTYPE,
    PRE_EDGE_DTYPE,
    V32CacheError,
    assert_label_free_open_paths,
    canonical_json_bytes,
    expected_runtime_split_sha256,
    make_final_edges,
    sha256_file,
    validate_cache,
    validate_dataset_arrays,
    validate_producer_bindings,
)


PARTIAL_SCHEMA = "celltrack-alpha.v32-preilp-export-partial.v1"
TERMINAL_INVENTORY_SCHEMA = "celltrack-alpha.v32-preilp-terminal-inventory.v1"
TERMINAL_RUN_SCHEMA = "celltrack-alpha.public-temporalunet-kaggle-run.v1"
_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _safe_dataset(dataset: str) -> str:
    if _DATASET.fullmatch(dataset) is None or ".." in dataset:
        raise V32CacheError(f"unsafe dataset name {dataset!r}")
    return dataset


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V32CacheError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V32CacheError(f"{path} must contain a JSON object")
    return value


def _load_array(path: Path, expected: np.dtype[Any]) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise V32CacheError(f"cannot load {path}: {exc}") from exc
    if value.ndim != 1 or value.dtype != expected or value.dtype.hasobject:
        raise V32CacheError(f"{path} dtype/shape drift")
    return value


def _validate_partial_receipt(
    path: Path, *, dataset: str, stage: str
) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != PARTIAL_SCHEMA or value.get("status") != "PASS_UNSEALED":
        raise V32CacheError(f"{path} schema/status drift")
    if value.get("dataset") != dataset or value.get("identity") != f"train:{dataset}":
        raise V32CacheError(f"{path} identity drift")
    if value.get("stage") != stage:
        raise V32CacheError(f"{path} stage drift")
    if value.get("label_free") is not True or value.get("evaluate") is not False:
        raise V32CacheError(f"{path} is not explicitly label-free")
    if value.get("m3_mounted") is not False:
        raise V32CacheError(f"{path} does not exclude M3")
    if not isinstance(value.get("frame_count"), int) or int(value["frame_count"]) <= 0:
        raise V32CacheError(f"{path} frame_count drift")
    source_zarr = value.get("source_zarr")
    if not isinstance(source_zarr, str):
        raise V32CacheError(f"{path} source_zarr drift")
    assert_label_free_open_paths([source_zarr])
    if Path(source_zarr).name != f"{dataset}.zarr":
        raise V32CacheError(f"{path} source_zarr identity drift")
    return value


def validate_partial_dataset(root: Path, dataset: str) -> dict[str, Any]:
    dataset = _safe_dataset(dataset)
    directory = root / "datasets" / dataset
    pre_receipt = _validate_partial_receipt(
        directory / "PRE_ILP_RECEIPT.json", dataset=dataset, stage="pre_ilp"
    )
    post_receipt = _validate_partial_receipt(
        directory / "POST_ILP_RECEIPT.json", dataset=dataset, stage="post_ilp"
    )
    if pre_receipt["frame_count"] != post_receipt["frame_count"]:
        raise V32CacheError(f"{dataset} pre/post frame count drift")
    if pre_receipt["source_zarr"] != post_receipt["source_zarr"]:
        raise V32CacheError(f"{dataset} pre/post source Zarr drift")
    files = {
        "pre_ilp_nodes": ("PRE_ILP_NODES.npy", NODE_DTYPE, pre_receipt, "nodes"),
        "pre_ilp_edges": ("PRE_ILP_EDGES.npy", PRE_EDGE_DTYPE, pre_receipt, "edges"),
        "frame_pairs": ("FRAME_PAIRS.npy", FRAME_PAIR_DTYPE, pre_receipt, "frame_pairs"),
        "post_ilp_nodes": ("POST_ILP_NODES.npy", NODE_DTYPE, post_receipt, "nodes"),
        "post_ilp_edges": ("POST_ILP_EDGES.npy", POST_EDGE_DTYPE, post_receipt, "edges"),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, (filename, dtype, receipt, receipt_key) in files.items():
        path = directory / filename
        record = receipt.get("files", {}).get(receipt_key)
        if not isinstance(record, dict) or record.get("path") != filename:
            raise V32CacheError(f"{dataset}/{key} receipt drift")
        if sha256_file(path) != record.get("sha256"):
            raise V32CacheError(f"{dataset}/{key} hash mismatch")
        arrays[key] = _load_array(path, dtype)
        if record.get("bytes") != path.stat().st_size:
            raise V32CacheError(f"{dataset}/{key} byte receipt drift")
        if record.get("rows") != len(arrays[key]):
            raise V32CacheError(f"{dataset}/{key} row receipt drift")
        if record.get("shape") != list(arrays[key].shape):
            raise V32CacheError(f"{dataset}/{key} shape receipt drift")
        if record.get("dtype") != json.loads(json.dumps(arrays[key].dtype.descr)):
            raise V32CacheError(f"{dataset}/{key} dtype receipt drift")
    # The complete validator already has the strongest root/post/final graph
    # invariants.  Reuse it with an identity final oracle solely for validating
    # the partial layers; this does not claim that post-ILP is final V32 output.
    post_nodes = np.asarray(arrays["post_ilp_nodes"])
    post_edges = arrays["post_ilp_edges"]
    synthetic_final_edges = make_final_edges(
        {
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
        }
        for row in post_edges
    )
    complete = {
        **arrays,
        "base_final_nodes": post_nodes.copy(),
        "base_final_edges": synthetic_final_edges,
    }
    counts = validate_dataset_arrays(
        complete, frame_count=int(pre_receipt["frame_count"])
    )
    return {
        "dataset": dataset,
        "identity": f"train:{dataset}",
        "status": "PASS_UNSEALED",
        "frame_count": int(pre_receipt["frame_count"]),
        "counts": {key: counts[key] for key in files},
        "scientific_use_open": False,
        "missing_before_seal": [
            "BASE_FINAL_NODES.npy",
            "BASE_FINAL_EDGES.npy",
            "BASE_SUBMISSION_FRAGMENT.csv",
            "DATASET_MANIFEST.json",
            "CACHE_MANIFEST.json",
        ],
    }


def validate_terminal_inventory(
    root: Path,
    terminal_receipt: Path,
    identities: Sequence[str],
    expected_producer: Mapping[str, Any],
) -> dict[str, Any]:
    producer = validate_producer_bindings(expected_producer)
    if not root.is_dir() or root.is_symlink():
        raise V32CacheError("terminal cache root is absent or a symlink")
    datasets_root = root / "datasets"
    if not datasets_root.is_dir() or datasets_root.is_symlink():
        raise V32CacheError("terminal datasets root is absent or a symlink")
    terminal = _load_json(terminal_receipt)
    datasets = [identity.split(":", 1)[1] for identity in identities]
    if (
        terminal.get("schema") != TERMINAL_RUN_SCHEMA
        or terminal.get("status") != "PASS"
        or terminal.get("variant") != "public_v32"
        or terminal.get("request_count") != len(datasets)
        or terminal.get("public_script_version_id") != 337501010
        or terminal.get("public_notebook_sha256") != producer["public_notebook_sha256"]
        or terminal.get("primary_weight_sha256") != producer["primary_weight_sha256"]
        or terminal.get("secondary_weight_sha256") != producer["secondary_weight_sha256"]
        or terminal.get("weight_refinement") is not False
        or terminal.get("substrate_only") is not True
        or terminal.get("weight_refinement_minimum_local_score") != 0.916
    ):
        raise V32CacheError("outer terminal run receipt contract drift")
    for record_name in ("submission", "run_stats", "run_log"):
        record = terminal.get(record_name)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None
        ):
            raise V32CacheError(f"outer terminal {record_name} record drift")
    if terminal["submission"]["sha256"] != producer["terminal_submission_sha256"]:
        raise V32CacheError("outer terminal submission differs from expected producer")
    submission_rows = terminal["submission"].get("rows")
    if (
        isinstance(submission_rows, bool)
        or not isinstance(submission_rows, int)
        or submission_rows < 0
    ):
        raise V32CacheError("outer terminal submission row count drift")
    for count_name in ("nodes", "edges"):
        count = terminal.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise V32CacheError(f"outer terminal {count_name} count drift")
    prediction_seconds = terminal.get("prediction_seconds")
    if (
        isinstance(prediction_seconds, bool)
        or not isinstance(prediction_seconds, (int, float))
        or not math.isfinite(float(prediction_seconds))
        or float(prediction_seconds) <= 0.0
    ):
        raise V32CacheError("outer terminal prediction time drift")
    if (
        isinstance(terminal.get("cuda_device_count"), bool)
        or not isinstance(terminal.get("cuda_device_count"), int)
        or terminal["cuda_device_count"] < 1
        or not isinstance(terminal.get("cuda_device_name"), str)
        or not terminal["cuda_device_name"]
    ):
        raise V32CacheError("outer terminal CUDA evidence drift")
    if terminal.get("selected_datasets") != datasets:
        raise V32CacheError("terminal selected-dataset order drift")
    runtime_sha256 = terminal.get("runtime_predict_script_sha256")
    if not isinstance(runtime_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None:
        raise V32CacheError("terminal runtime predictor hash is malformed")
    if runtime_sha256 != producer["runtime_predictor_sha256"]:
        raise V32CacheError("terminal runtime predictor differs from expected producer")
    inventory = terminal.get("v32_preilp_cache")
    if not isinstance(inventory, dict):
        raise V32CacheError("terminal cache inventory is absent")
    if (
        inventory.get("schema") != TERMINAL_INVENTORY_SCHEMA
        or inventory.get("status") != "PASS_UNSEALED"
        or inventory.get("scientific_use_open") is not False
        or inventory.get("identities") != list(identities)
    ):
        raise V32CacheError("terminal cache inventory contract drift")
    expected_names = {
        "PRE_ILP_NODES.npy", "PRE_ILP_EDGES.npy", "FRAME_PAIRS.npy",
        "PRE_ILP_RECEIPT.json", "POST_ILP_NODES.npy", "POST_ILP_EDGES.npy",
        "POST_ILP_RECEIPT.json",
    }
    observed_root_names = {path.name for path in root.iterdir()}
    if observed_root_names != {"datasets", "OPEN_AUDIT.jsonl"}:
        raise V32CacheError("terminal cache root file-set drift")
    terminal_dataset_entries = list(datasets_root.iterdir())
    if any(
        not path.is_dir() or path.is_symlink() for path in terminal_dataset_entries
    ) or sorted(path.name for path in terminal_dataset_entries) != sorted(datasets):
        raise V32CacheError("terminal cache dataset-directory inventory drift")
    open_audit_path = root / "OPEN_AUDIT.jsonl"
    if not open_audit_path.is_file() or open_audit_path.is_symlink():
        raise V32CacheError("terminal cache open audit is absent or non-regular")
    opened_paths: list[str] = []
    open_events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        open_audit_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V32CacheError(f"invalid runtime open audit line {line_number}") from exc
        if (
            not isinstance(event, dict)
            or event.get("schema") != "celltrack-alpha.v32-label-free-open-audit.v1"
            or event.get("event") != "input_open"
            or not isinstance(event.get("path"), str)
            or not isinstance(event.get("shard"), str)
        ):
            raise V32CacheError(f"runtime open audit contract drift at line {line_number}")
        kind = event.get("kind")
        if kind == "file":
            byte_count = event.get("bytes")
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(event.get("sha256", ""))) is None
            ):
                raise V32CacheError(
                    f"runtime file-open evidence drift at line {line_number}"
                )
        elif kind == "zarr_array":
            metadata = event.get("metadata")
            if not isinstance(metadata, list) or not metadata:
                raise V32CacheError(
                    f"runtime Zarr-open evidence drift at line {line_number}"
                )
            for record in metadata:
                if (
                    not isinstance(record, dict)
                    or record.get("name") not in {".zarray", ".zattrs", "zarr.json"}
                    or isinstance(record.get("bytes"), bool)
                    or not isinstance(record.get("bytes"), int)
                    or record["bytes"] < 0
                    or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                    is None
                ):
                    raise V32CacheError(
                        f"runtime Zarr metadata evidence drift at line {line_number}"
                    )
            expected_metadata_sha256 = hashlib.sha256(
                canonical_json_bytes(metadata)
            ).hexdigest()
            if event.get("metadata_sha256") != expected_metadata_sha256:
                raise V32CacheError(
                    f"runtime Zarr metadata aggregate drift at line {line_number}"
                )
        else:
            raise V32CacheError(f"runtime input kind drift at line {line_number}")
        opened_paths.append(event["path"])
        open_events.append(event)
    assert_label_free_open_paths(opened_paths)
    if not opened_paths:
        raise V32CacheError("runtime open audit is empty")
    normalized_opened = [path.replace("\\", "/") for path in opened_paths]
    raw_events: list[dict[str, Any]] = []
    for dataset in datasets:
        expected_suffix = f"/{dataset}.zarr/0"
        matching = [
            event for event, path in zip(open_events, normalized_opened)
            if path.endswith(expected_suffix)
        ]
        if len(matching) != 1 or matching[0].get("kind") != "zarr_array":
            raise V32CacheError(f"runtime open audit lacks raw Zarr access for {dataset}")
        raw_events.extend(matching)
    file_events = [event for event in open_events if event.get("kind") == "file"]
    weight_opens = [
        event for event in file_events
        if str(event["path"]).replace("\\", "/").endswith("/edge_predictor_best.pth")
    ]
    config_opens = [
        event for event in file_events
        if str(event["path"]).replace("\\", "/").endswith("/config.json")
    ]
    split_opens = [
        event for event in file_events
        if str(event["path"]).replace("\\", "/")
        in {"kaggle_test_splits_50ep.json", "./kaggle_test_splits_50ep.json"}
        or str(event["path"]).replace("\\", "/").endswith(
            "/kaggle_test_splits_50ep.json"
        )
    ]
    if len(file_events) != len(weight_opens) + len(config_opens) + len(split_opens):
        raise V32CacheError("runtime open audit contains an unexpected file input")
    worker_count = min(2, terminal["cuda_device_count"], len(datasets))
    expected_shards = (
        {"single"} if worker_count == 1
        else {f"{index}/{worker_count}" for index in range(worker_count)}
    )
    if {str(event["shard"]) for event in open_events} != expected_shards:
        raise V32CacheError("runtime open audit GPU-shard provenance drift")
    if (
        len(weight_opens) != 2 * worker_count
        or len(config_opens) != 2 * worker_count
        or len(split_opens) != worker_count
    ):
        raise V32CacheError("runtime open audit lacks exact primary/secondary/split inputs")
    expected_weight_hashes = sorted(
        [producer["primary_weight_sha256"], producer["secondary_weight_sha256"]]
    )
    expected_config_hashes = sorted(
        [producer["primary_config_sha256"], producer["secondary_config_sha256"]]
    )
    expected_split_sha256 = expected_runtime_split_sha256(identities)
    if producer["runtime_split_sha256"] != expected_split_sha256:
        raise V32CacheError("producer runtime split differs from expected identities")
    for shard in sorted(expected_shards):
        if sorted(
            str(event["sha256"]) for event in weight_opens if event["shard"] == shard
        ) != expected_weight_hashes:
            raise V32CacheError(f"runtime weight hashes drift on shard {shard}")
        if sorted(
            str(event["sha256"]) for event in config_opens if event["shard"] == shard
        ) != expected_config_hashes:
            raise V32CacheError(f"runtime config hashes drift on shard {shard}")
        matching_splits = [event for event in split_opens if event["shard"] == shard]
        if (
            len(matching_splits) != 1
            or matching_splits[0]["sha256"] != expected_split_sha256
        ):
            raise V32CacheError(f"runtime split hash drift on shard {shard}")
    actual_records: list[dict[str, Any]] = [
        {
            "path": "OPEN_AUDIT.jsonl",
            "bytes": open_audit_path.stat().st_size,
            "sha256": sha256_file(open_audit_path),
        }
    ]
    if sha256_file(open_audit_path) != producer["terminal_open_audit_sha256"]:
        raise V32CacheError("runtime open audit differs from expected producer")
    for dataset in sorted(datasets):
        directory = datasets_root / _safe_dataset(dataset)
        if not directory.is_dir() or directory.is_symlink():
            raise V32CacheError(
                f"terminal cache dataset directory is absent or a symlink: {dataset}"
            )
        observed_names = {
            path.name for path in directory.iterdir() if path.is_file()
        }
        if observed_names != expected_names:
            raise V32CacheError(f"terminal cache file-set drift for {dataset}")
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise V32CacheError(f"terminal cache contains a non-regular file: {path}")
            actual_records.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if inventory.get("files") != actual_records:
        raise V32CacheError("terminal cache file inventory differs from downloaded bytes")
    actual_inventory_sha256 = hashlib.sha256(canonical_json_bytes(actual_records)).hexdigest()
    if inventory.get("inventory_sha256") != actual_inventory_sha256:
        raise V32CacheError("terminal cache aggregate hash mismatch")
    terminal_receipt_sha256 = sha256_file(terminal_receipt)
    if terminal_receipt_sha256 != producer["terminal_receipt_sha256"]:
        raise V32CacheError("terminal receipt hash differs from expected producer")
    if actual_inventory_sha256 != producer["terminal_inventory_sha256"]:
        raise V32CacheError("terminal inventory hash differs from expected producer")
    return {
        "terminal_receipt_sha256": terminal_receipt_sha256,
        "runtime_predict_script_sha256": runtime_sha256,
        "terminal_submission_sha256": terminal["submission"]["sha256"],
        "inventory_sha256": actual_inventory_sha256,
        "file_count": len(actual_records),
        "opened_input_count": len(opened_paths),
    }


def validate_partial(
    root: Path,
    identities: Sequence[str],
    *,
    terminal_receipt: Path,
    expected_producer: Mapping[str, Any],
) -> dict[str, Any]:
    if not identities or len(set(identities)) != len(identities):
        raise V32CacheError("partial expected identities must be nonempty and unique")
    if not root.is_dir() or root.is_symlink():
        raise V32CacheError("partial cache root is absent or a symlink")
    datasets_root = root / "datasets"
    if not datasets_root.is_dir() or datasets_root.is_symlink():
        raise V32CacheError("partial datasets root is absent or a symlink")
    results = []
    for identity in identities:
        if not identity.startswith("train:"):
            raise V32CacheError(f"invalid expected identity {identity!r}")
        dataset = _safe_dataset(identity.split(":", 1)[1])
        directory = datasets_root / dataset
        if not directory.is_dir() or directory.is_symlink():
            raise V32CacheError(
                f"partial dataset directory is absent or a symlink: {dataset}"
            )
        results.append(validate_partial_dataset(root, dataset))
    dataset_entries = list(datasets_root.iterdir())
    if any(not path.is_dir() or path.is_symlink() for path in dataset_entries):
        raise V32CacheError("partial datasets root contains a non-directory or symlink")
    observed = sorted(f"train:{path.name}" for path in dataset_entries)
    if sorted(identities) != observed:
        raise V32CacheError("partial cache has missing or extra dataset directories")
    terminal = validate_terminal_inventory(
        root, terminal_receipt, identities, expected_producer
    )
    return {
        "status": "PASS_UNSEALED",
        "identity_count": len(identities),
        "identities": list(identities),
        "datasets": results,
        "terminal": terminal,
        "scientific_use_open": False,
    }


def _expected_identities(args: argparse.Namespace) -> list[str] | None:
    if args.identities_json is None:
        return None
    value = json.loads(args.identities_json.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("identities")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise V32CacheError("identities JSON must be a list or an object with an identities list")
    return list(value)


def _expected_producer(args: argparse.Namespace) -> dict[str, Any] | None:
    if getattr(args, "producer_json", None) is None:
        return None
    return _load_json(args.producer_json)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    for name in ("sealed", "partial"):
        child = subparsers.add_parser(name)
        child.add_argument("--cache-root", type=Path, required=True)
        child.add_argument("--identities-json", type=Path, required=True)
        child.add_argument("--producer-json", type=Path, required=True)
        if name == "partial":
            child.add_argument("--terminal-receipt", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    identities = _expected_identities(args)
    if identities is None:
        raise V32CacheError("validation requires --identities-json")
    if args.command == "sealed":
        producer = _expected_producer(args)
        if producer is None:
            raise V32CacheError("sealed validation requires --producer-json")
        result = validate_cache(
            args.cache_root,
            expected_identities=identities,
            expected_producer=producer,
        )
    else:
        producer = _expected_producer(args)
        if producer is None:
            raise V32CacheError("partial validation requires --producer-json")
        result = validate_partial(
            args.cache_root,
            identities,
            terminal_receipt=args.terminal_receipt,
            expected_producer=producer,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

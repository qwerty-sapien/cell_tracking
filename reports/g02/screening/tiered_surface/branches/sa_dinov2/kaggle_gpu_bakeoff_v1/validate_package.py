#!/usr/bin/env python3
"""Validate the generated DINOv2 Kaggle package against a mounted layout."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA = "celltrack-alpha.dinov2-kaggle-gpu-local-validation.v1"
ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
REFERENCE_FEATURIZER = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_dinov2/"
    "dinov2_xy_action_featurizer.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def first_parent_call(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parent_main"
    )
    statement = function.body[0]
    if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
        raise RuntimeError("parent_main first statement is not inventory assignment")
    return ast.unparse(statement.value.func)


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_root.resolve()
    kernel_path = package / "kernel/kaggle_dinov2_gpu.py"
    dataset = package / "dataset"
    build_receipt = json.loads((package / "BUILD_RECEIPT.json").read_text())
    kernel = load_module(kernel_path, "generated_dinov2_gpu_kernel")
    reference = load_module(REFERENCE_FEATURIZER, "reference_dinov2_featurizer")
    config = kernel.decode_config()

    if first_parent_call(kernel_path) != "dump_mount_inventory":
        raise RuntimeError("mount inventory is not the first parent step")
    top_imports = {
        alias.name.split(".", 1)[0]
        for node in ast.parse(kernel_path.read_text()).body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    if top_imports & {"numpy", "torch", "zarr"}:
        raise RuntimeError("heavy import precedes step 0")
    names = {path.name for path in dataset.iterdir()}
    if any("bundle_manifest" in name or name.endswith("_manifest.json") for name in names):
        raise RuntimeError("external bundle manifest leaked into dataset")

    with tempfile.TemporaryDirectory(prefix="g02_dino_mount_", dir="/private/tmp") as directory:
        mount = Path(directory) / "kaggle/input"
        private = mount / "opaque-private-dataset/expanded/layout"
        asset_paths = [
            path for path in sorted(dataset.iterdir()) if path.name != "dataset-metadata.json"
        ]
        for index, source in enumerate(asset_paths):
            hardlink(source, private / f"opaque-{index:03d}.payload")
        competition = mount / "competitions/biohub-cell-tracking-during-development"
        for identity in config["required_video_ids"]:
            dataset_id = identity.split(":", 1)[1]
            marker = competition / f"train/{dataset_id}.zarr/0/zarr.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}\n", encoding="ascii")

        output = io.StringIO()
        with redirect_stdout(output):
            rows, elapsed = kernel.dump_mount_inventory(mount)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        if events[0]["event"] != "STEP0_MOUNT_INVENTORY_BEGIN":
            raise RuntimeError("inventory begin is not first emitted event")
        if events[-1]["event"] != "STEP0_MOUNT_INVENTORY_END":
            raise RuntimeError("inventory end event missing")
        if sum(event["event"] == "STEP0_MOUNT_FILE" for event in events) != len(rows):
            raise RuntimeError("not every mounted file was printed")
        resolved = kernel.resolve_assets(rows, config)
        if len(resolved["requests"]) != 77 or len(resolved["wheels"]) != 3:
            raise RuntimeError("content-addressed mounted resolution incomplete")
        if Path(resolved["competition_root"]) != competition:
            raise RuntimeError("competition root resolution drift")

        duplicate = private / "duplicate-weight.payload"
        hardlink(Path(resolved["weight"]), duplicate)
        duplicate_row = {
            "path": str(duplicate),
            "bytes": duplicate.stat().st_size,
            "sha256": sha256(duplicate),
        }
        try:
            kernel.resolve_assets([*rows, duplicate_row], config)
        except kernel.RuntimeContractError:
            duplicate_fails_closed = True
        else:
            duplicate_fails_closed = False
        if not duplicate_fails_closed:
            raise RuntimeError("duplicate content hash did not fail closed")

        missing_hash = config["requests"][0]["sha256"]
        missing_rows = [row for row in rows if row["sha256"] != missing_hash]
        try:
            kernel.resolve_assets(missing_rows, config)
        except kernel.RuntimeContractError:
            missing_fails_closed = True
        else:
            missing_fails_closed = False
        if not missing_fails_closed:
            raise RuntimeError("missing request did not fail closed")

    source = np.arange(4 * 768, dtype=np.float32).reshape(4, 768) / 1000.0
    target = np.flip(source, axis=1).copy()
    observed_features = kernel.action_features(np, source, target)
    expected_features = reference.action_features(source, target)
    if not np.array_equal(observed_features, expected_features):
        raise RuntimeError("embedded action featurizer differs from frozen implementation")
    node_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
    kwargs = {
        "identity": "train:44b6_fixture",
        "frame": 7,
        "supplier_contract_sha256": config["pins"]["supplier_contract"],
    }
    observed_permutation = kernel.within_frame_permutation(
        np, source, node_ids, **kwargs
    )
    expected_permutation = reference.within_frame_permutation(
        source, node_ids, **kwargs
    )
    if not np.array_equal(observed_permutation, expected_permutation):
        raise RuntimeError("embedded A0 permutation differs from frozen implementation")

    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "generated_kernel_sha256": sha256(kernel_path),
        "build_receipt_sha256": sha256(package / "BUILD_RECEIPT.json"),
        "first_parent_call": "dump_mount_inventory",
        "full_recursive_file_events": len(rows),
        "mounted_layout": "OPAQUE_EXPANDED_LAYOUT",
        "request_hashes_resolved": len(resolved["requests"]),
        "wheel_hashes_resolved": len(resolved["wheels"]),
        "weight_hash_resolved": config["weight"]["sha256"],
        "source_hash_resolved": config["source"]["sha256"],
        "duplicate_hash_fails_closed": duplicate_fails_closed,
        "missing_hash_fails_closed": missing_fails_closed,
        "external_bundle_manifest_present": False,
        "heavy_import_before_step0": False,
        "frozen_featurizer_array_identical": True,
        "frozen_A0_permutation_array_identical": True,
        "partial_reused_rows": config["partial_reuse"]["reused_rows"],
        "action_counts": config["class_action_counts"],
        "class_unique_endpoint_counts": config["class_unique_endpoint_counts"],
        "dataset_file_count": build_receipt["dataset_file_count"],
        "inventory_fixture_elapsed_seconds": elapsed,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

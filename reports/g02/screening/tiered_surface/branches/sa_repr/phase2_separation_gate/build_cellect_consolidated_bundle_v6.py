#!/usr/bin/env python3
"""Build corrected CELLECT v6 from byte-exact v4 assets and frozen requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from build_cellect_asset_v4 import CHECKPOINT_SHA256, sha256

DATASET_ID = "onebitbrain/g02-cellect-r7-published-assets-v1"
POPULATION_MANIFEST_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
BUNDLE_MANIFEST_NAME = "consolidated_bundle_manifest.json"
REQUEST_MANIFEST_NAME = "action_request_manifest.json"
POPULATION_MANIFEST_NAME = "population_manifest.json"


class ConsolidatedBuildError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verified_record(path: Path, record: dict[str, object], label: str) -> None:
    if not path.is_file():
        raise ConsolidatedBuildError(f"missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
        raise ConsolidatedBuildError(f"{label} bytes or SHA-256 drift: {path}")


def _request_rows(
    population: dict[str, object],
    request_root: Path,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    request_count = 0
    classes = population.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ConsolidatedBuildError("population manifest has no action classes")
    for action_class, class_record in sorted(classes.items()):
        videos = class_record.get("videos") if isinstance(class_record, dict) else None
        if not isinstance(videos, list) or not videos:
            raise ConsolidatedBuildError(
                f"population class has no videos: {action_class}"
            )
        for video in videos:
            if not isinstance(video, dict):
                raise ConsolidatedBuildError("population video record is malformed")
            source = request_root / action_class / Path(
                str(video["request_path"])
            ).name
            expected_sha256 = str(video["request_sha256"])
            if not source.is_file() or sha256(source) != expected_sha256:
                raise ConsolidatedBuildError(
                    f"frozen request payload drift: {action_class}/{source.name}"
                )
            count = int(video["count"])
            if count <= 0:
                raise ConsolidatedBuildError(
                    f"request payload has nonpositive action count: {source}"
                )
            request_count += count
            rows.append(
                {
                    "action_class": action_class,
                    "action_count": count,
                    "bytes": source.stat().st_size,
                    "identity": str(video["identity"]),
                    "path": f"feature_request__{action_class}__{source.name}",
                    "sha256": expected_sha256,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise ConsolidatedBuildError("duplicate request payload path")
    return rows, request_count


def build_consolidated_bundle(
    v4_assets: Path,
    v4_manifest_path: Path,
    request_root: Path,
    population_manifest_path: Path,
    destination: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if destination.exists():
        raise ConsolidatedBuildError(f"destination already exists: {destination}")
    v4_manifest = json.loads(v4_manifest_path.read_text())
    if (
        v4_manifest.get("dataset", {}).get("id") != DATASET_ID
        or int(v4_manifest["dataset"]["target_version"]) != 4
    ):
        raise ConsolidatedBuildError("source asset manifest is not frozen dataset v4")
    if sha256(population_manifest_path) != POPULATION_MANIFEST_SHA256:
        raise ConsolidatedBuildError("population manifest SHA-256 drift")
    population = json.loads(population_manifest_path.read_text())
    if population.get("status") != "COMPLETE":
        raise ConsolidatedBuildError("population manifest is not complete")
    request_rows, request_count = _request_rows(population, request_root)

    request_manifest = {
        "schema": "celltrack-alpha.cellect-action-request-manifest.v1",
        "status": "FROZEN_COMPLETE",
        "population_manifest_path": POPULATION_MANIFEST_NAME,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "request_count": request_count,
        "request_file_count": len(request_rows),
        "classes": {
            action_class: {
                "request_count": sum(
                    int(row["action_count"])
                    for row in request_rows
                    if row["action_class"] == action_class
                ),
                "request_file_count": sum(
                    1 for row in request_rows if row["action_class"] == action_class
                ),
            }
            for action_class in sorted(
                {str(row["action_class"]) for row in request_rows}
            )
        },
        "requests": request_rows,
    }
    request_manifest_bytes = canonical_json(request_manifest)
    request_manifest_sha256 = hashlib.sha256(request_manifest_bytes).hexdigest()

    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists():
        raise ConsolidatedBuildError(f"partial destination already exists: {partial}")
    partial.mkdir(parents=True)

    files: list[dict[str, object]] = []
    for source_record in v4_manifest["files"]:
        record = dict(source_record)
        source = v4_assets / str(record["path"])
        _verified_record(source, record, "v4 payload")
        target = partial / str(record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if str(record["path"]).endswith(".whl"):
            record["role"] = "offline_wheel"
        elif str(record["path"]) == "cellect_model.py":
            record["role"] = "safe_loader"
        else:
            record["role"] = "published_asset"
        files.append(record)

    metadata = v4_assets / str(v4_manifest["control_metadata"]["path"])
    _verified_record(metadata, v4_manifest["control_metadata"], "dataset metadata")
    shutil.copyfile(metadata, partial / metadata.name)

    for request in request_rows:
        prefix = f"feature_request__{request['action_class']}__"
        request_name = str(request["path"])
        if not request_name.startswith(prefix):
            raise ConsolidatedBuildError("flattened request filename drift")
        source = request_root / str(request["action_class"]) / request_name.removeprefix(
            prefix
        )
        target = partial / str(request["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append(
            {
                "path": request["path"],
                "bytes": request["bytes"],
                "sha256": request["sha256"],
                "role": "feature_request",
                "action_class": request["action_class"],
                "action_count": request["action_count"],
                "identity": request["identity"],
            }
        )

    population_target = partial / POPULATION_MANIFEST_NAME
    shutil.copyfile(population_manifest_path, population_target)
    files.append(
        {
            "path": POPULATION_MANIFEST_NAME,
            "bytes": population_target.stat().st_size,
            "sha256": POPULATION_MANIFEST_SHA256,
            "role": "population_manifest",
        }
    )
    request_manifest_target = partial / REQUEST_MANIFEST_NAME
    request_manifest_target.write_bytes(request_manifest_bytes)
    files.append(
        {
            "path": REQUEST_MANIFEST_NAME,
            "bytes": len(request_manifest_bytes),
            "sha256": request_manifest_sha256,
            "role": "request_manifest",
            "request_count": request_count,
            "request_file_count": len(request_rows),
        }
    )
    files.sort(key=lambda row: str(row["path"]))

    bundle_manifest = {
        "schema": "celltrack-alpha.cellect-r7-kaggle-asset-manifest.v6",
        "status": "LOCAL_CONSOLIDATED_BYTE_EXACT",
        "dataset": {
            "id": DATASET_ID,
            "visibility": "private",
            "version_before": 5,
            "target_version": 6,
        },
        "derived_from": {
            "asset_manifest_path": str(v4_manifest_path),
            "asset_manifest_sha256": sha256(v4_manifest_path),
            "remote_v4_status": "BYTE_EXACT_ROUNDTRIP_PASS",
            "remote_v5_status": "INVALID_REQUEST_DIRECTORY_SKIPPED",
        },
        "bundle_manifest": {
            "path": BUNDLE_MANIFEST_NAME,
            "self_hash": "externally_pinned_by_kernel_and_roundtrip_receipt",
        },
        "checkpoint_transport": v4_manifest["checkpoint_transport"],
        "control_metadata": v4_manifest["control_metadata"],
        "coverage": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
            "request_manifest_sha256": request_manifest_sha256,
            "request_count": request_count,
            "request_file_count": len(request_rows),
            "offline_wheel_count": sum(
                1 for row in files if row.get("role") == "offline_wheel"
            ),
        },
        "files": files,
        "required_file_count": len(files),
        "required_total_bytes": sum(int(row["bytes"]) for row in files),
        "forbidden": v4_manifest["forbidden"],
    }
    bundle_manifest_bytes = canonical_json(bundle_manifest)
    (partial / BUNDLE_MANIFEST_NAME).write_bytes(bundle_manifest_bytes)
    partial.replace(destination)
    return bundle_manifest, request_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-assets", type=Path, required=True)
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--request-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--request-manifest-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle, requests = build_consolidated_bundle(
        args.v4_assets,
        args.v4_manifest,
        args.request_root,
        args.population_manifest,
        args.destination,
    )
    args.manifest_out.write_bytes(canonical_json(bundle))
    args.request_manifest_out.write_bytes(canonical_json(requests))
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "target_version": bundle["dataset"]["target_version"],
                "request_count": bundle["coverage"]["request_count"],
                "request_file_count": bundle["coverage"]["request_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

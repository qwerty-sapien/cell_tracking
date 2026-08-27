#!/usr/bin/env python3
"""Verify Kaggle's expanded DINOv2 CPU carrier against the frozen inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_OUTER_MEMBER_COUNT = 169
EXPECTED_TOP_MANIFEST_SHA256 = (
    "5801cab07ecd63d948efefa605b0a60b541bead6dfaafd06053d467d0e0c25bf"
)
EXPECTED_INVENTORY_SHA256 = (
    "546cbf270a3e430ad5e35cccaad8778f33a11eeb149134c48e1721cc8dae1992"
)
EXPECTED_EXPANDED_FILE_COUNT = 166
EXPECTED_ARCHIVE_SHA256 = (
    "580499a2e0f35965cc4a7b1a5716bde09671a3ea839cbea06f6160b24c22ca40"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
EXPECTED_SOURCE_SHA256 = (
    "e1b9403af2976443baeeab0ede0de0c53b63cf209c292bcdeccdc672674148c7"
)
EXPECTED_POPULATION_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
EXPECTED_REQUEST_COUNT = 1_391_458
EXPECTED_REQUEST_FILE_COUNT = 77
EXPECTED_ENDPOINT_COUNT = 1_417_863
EXPECTED_WHEEL_COUNT = 3


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    outer_sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    failures: list[str] = []

    with zipfile.ZipFile(args.archive) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(names) != EXPECTED_OUTER_MEMBER_COUNT:
            failures.append(
                f"outer member count {len(names)} != {EXPECTED_OUTER_MEMBER_COUNT}"
            )
        unsafe = [name for name in names if not safe_member(name)]
        if unsafe:
            failures.append(f"unsafe outer members: {unsafe[:3]}")

        top_manifest_name = "dinov2_cpu_bundle_manifest.json"
        inventory_name = "dinov2_cpu_bundle/dinov2_cpu_bundle_inventory.json"
        try:
            top_manifest_bytes = archive.read(top_manifest_name)
            inventory_bytes = archive.read(inventory_name)
        except KeyError as exc:
            failures.append(f"missing required member: {exc}")
            top_manifest_bytes = b"{}"
            inventory_bytes = b'{"files":[]}'

        top_manifest_sha = sha256_bytes(top_manifest_bytes)
        inventory_sha = sha256_bytes(inventory_bytes)
        if top_manifest_sha != EXPECTED_TOP_MANIFEST_SHA256:
            failures.append(f"top manifest SHA mismatch: {top_manifest_sha}")
        if inventory_sha != EXPECTED_INVENTORY_SHA256:
            failures.append(f"inventory SHA mismatch: {inventory_sha}")

        manifest = json.loads(top_manifest_bytes)
        inventory = json.loads(inventory_bytes)
        records = inventory.get("files", [])
        if len(records) != EXPECTED_EXPANDED_FILE_COUNT:
            failures.append(
                f"inventory record count {len(records)} != "
                f"{EXPECTED_EXPANDED_FILE_COUNT}"
            )

        expected_outer = {
            top_manifest_name,
            "dinov2_source.tar.bin",
            inventory_name,
        }
        verified_records = 0
        for record in records:
            relative = record["path"]
            member = f"dinov2_cpu_bundle/{relative}"
            expected_outer.add(member)
            try:
                payload = archive.read(member)
            except KeyError:
                failures.append(f"missing inventory member: {member}")
                continue
            if len(payload) != record["bytes"]:
                failures.append(
                    f"size mismatch: {relative}: {len(payload)} != {record['bytes']}"
                )
                continue
            digest = sha256_bytes(payload)
            if digest != record["sha256"]:
                failures.append(
                    f"SHA mismatch: {relative}: {digest} != {record['sha256']}"
                )
                continue
            verified_records += 1

        unexpected = sorted(set(names) - expected_outer)
        missing = sorted(expected_outer - set(names))
        if unexpected:
            failures.append(f"unexpected outer members: {unexpected[:3]}")
        if missing:
            failures.append(f"missing outer members: {missing[:3]}")

        try:
            top_source = archive.read("dinov2_source.tar.bin")
            inner_source = archive.read("dinov2_cpu_bundle/dinov2_source.tar.bin")
            if top_source != inner_source:
                failures.append("top-level and inventory source archives differ")
        except KeyError as exc:
            failures.append(f"missing source duplicate: {exc}")

    assertions = {
        "archive_sha256": manifest.get("archive", {}).get("sha256")
        == EXPECTED_ARCHIVE_SHA256,
        "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256")
        == EXPECTED_CHECKPOINT_SHA256,
        "source_sha256": manifest.get("source", {}).get("sha256")
        == EXPECTED_SOURCE_SHA256,
        "population_manifest_sha256": manifest.get("population_manifest_sha256")
        == EXPECTED_POPULATION_SHA256,
        "request_count_positive_and_exact": manifest.get("request_count")
        == EXPECTED_REQUEST_COUNT,
        "request_file_count": manifest.get("request_file_count")
        == EXPECTED_REQUEST_FILE_COUNT,
        "population_file_count": manifest.get("population_file_count")
        == EXPECTED_REQUEST_FILE_COUNT,
        "unique_endpoint_count": manifest.get("unique_endpoint_count")
        == EXPECTED_ENDPOINT_COUNT,
        "wheel_count": manifest.get("wheel_count") == EXPECTED_WHEEL_COUNT,
        "gpu_disabled": manifest.get("enable_gpu") is False,
    }
    for name, passed in assertions.items():
        if not passed:
            failures.append(f"manifest assertion failed: {name}")

    receipt = {
        "schema": "celltrack-alpha.dinov2-r5-kaggle-cpu-remote-pass.v1",
        "status": "PASS" if not failures else "FAIL",
        "archive": str(args.archive.resolve()),
        "outer_archive_bytes": args.archive.stat().st_size,
        "outer_archive_sha256": outer_sha,
        "kaggle_transform": "expanded dinov2_cpu_bundle.zip under dinov2_cpu_bundle/",
        "top_manifest_sha256": top_manifest_sha,
        "inventory_sha256": inventory_sha,
        "inventory_records": len(records),
        "verified_inventory_records": verified_records,
        "outer_file_count": len(names),
        "manifest_assertions": assertions,
        "elapsed_seconds": time.perf_counter() - started,
        "failures": failures,
        "kernel_launched": False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

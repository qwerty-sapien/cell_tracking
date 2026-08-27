#!/usr/bin/env python3
"""Verify a downloaded expanded-tar selector dataset member-by-member."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


VERSION = None
MANIFEST_SHA256 = "73a15e27f69f0968d0bd5876775663b95c09d7b06df9d7fbc5c17b89540c2450"
INVENTORY_SHA256 = "fc21fe717b9eed0460daefada4815a370c7d696e5492b96a376b2a5c1341fa79"
PLAIN_POPULATION_PATH = (
    "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
    "generated/action_population.jsonl"
)
PLAIN_SHA256 = "f033b5ac34f615cf05ed48ed73efd0f27dc8f4f931f75b7dbf75ee4fc3c414eb"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def one_suffix(names: set[str], suffix: str, label: str) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        raise SystemExit(f"{label}: expected one expanded member, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.manual_archive.resolve()
    if not archive.is_file():
        raise SystemExit("manual archive missing")
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        manifest_name = one_suffix(
            names, "selector_cpu_bundle_manifest.json", "selector manifest"
        )
        inventory_name = one_suffix(
            names,
            "selector_cpu_bundle/selector_cpu_bundle_inventory.json",
            "selector inventory",
        )
        manifest_bytes = handle.read(manifest_name)
        inventory_bytes = handle.read(inventory_name)
        if (
            digest(manifest_bytes) != MANIFEST_SHA256
            or digest(inventory_bytes) != INVENTORY_SHA256
        ):
            raise SystemExit("manual manifest or inventory drifted")
        manifest = json.loads(manifest_bytes)
        inventory = json.loads(inventory_bytes)
        expected = {
            manifest_name,
            inventory_name,
        }
        bundle_prefix = inventory_name.removesuffix(
            "selector_cpu_bundle_inventory.json"
        )
        verified = []
        for row in inventory["files"]:
            relative = str(row["path"])
            member = f"{bundle_prefix}{relative}"
            value = handle.read(member)
            if len(value) != int(row["bytes"]) or digest(value) != row["sha256"]:
                raise SystemExit(f"downloaded member SHA-256 mismatch: {relative}")
            if relative == PLAIN_POPULATION_PATH and (
                digest(value) != PLAIN_SHA256
                or row.get("source_path") != f"{PLAIN_POPULATION_PATH}.gz"
                or row.get("transport") != "gzip_payload_as_plain_member"
            ):
                raise SystemExit("plain population member contract drifted")
            verified.append(relative)
            expected.add(member)
        if names != expected:
            raise SystemExit("manual archive has missing or extra files")
        record = manifest.get("archive", {})
        if (
            manifest.get("schema")
            != "celltrack-alpha.selector-kaggle-cpu-bundle.v2"
            or record.get("format") != "uncompressed_tar"
            or record.get("compression") != "none"
            or record.get("verification") != "member_sha256_only"
            or "sha256" in record
            or manifest.get("inventory_sha256") != INVENTORY_SHA256
            or len(inventory["files"]) + 1 != manifest["archive"]["member_count"]
            or PLAIN_POPULATION_PATH not in verified
        ):
            raise SystemExit("manual archive contract drifted")
    receipt = {
        "schema": "celltrack-alpha.selector-kaggle-cpu-remote-members.v2",
        "status": "EXPANDED_MEMBER_SHA256_PASS",
        "dataset": "onebitbrain/g02-selector-requirement-cpu-inputs-v1",
        "version": VERSION,
        "manual_archive_path": str(archive),
        "manual_archive_bytes": archive.stat().st_size,
        "manual_archive_sha256": sha256(archive),
        "uploaded_format": "uncompressed_tar",
        "remote_materialization": "kaggle_expanded_directory",
        "container_roundtrip_verified": False,
        "verification": "member_sha256_only",
        "manifest_sha256": MANIFEST_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "inventory_file_count": len(inventory["files"]),
        "downloaded_file_count": len(names),
        "kernel_launched": False,
        "submission_action": False,
        "verified_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    output = (
        Path(__file__).resolve().parent
        / "SELECTOR_CPU_BUNDLE_REMOTE_MEMBER_PASS_V2.json"
    )
    output.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

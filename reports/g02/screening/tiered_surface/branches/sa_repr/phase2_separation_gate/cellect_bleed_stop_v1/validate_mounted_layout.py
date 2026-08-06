#!/usr/bin/env python3
"""Validate the self-contained CELLECT kernel on Kaggle-expanded mount shapes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from build_selfcontained import build_contract, sha256


def inventory(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[sha256(path)].append(path)
    return dict(result)


def one_content(
    by_hash: dict[str, list[Path]], digest: str, label: str
) -> Path:
    matches = by_hash.get(digest, [])
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one source object, found {len(matches)}")
    return matches[0]


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def run_step0(kernel: Path, mount: Path, work: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CELLECT_INPUT_ROOT_OVERRIDE": str(mount),
            "CELLECT_WORK_ROOT_OVERRIDE": str(work),
            "CELLECT_STEP0_ONLY": "1",
        }
    )
    return subprocess.run(
        [sys.executable, str(kernel)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def expanded_fixture(
    source_root: Path,
    mount_root: Path,
    contract: dict[str, object],
) -> dict[str, object]:
    source_inventory = inventory(source_root)
    records: list[dict[str, object]] = [
        contract["population"],
        contract["safe_loader"],
        *contract["wheels"],
        *contract["requests"],
    ]
    for ordinal, record in enumerate(records):
        source = one_content(
            source_inventory, str(record["sha256"]), f"payload {ordinal}"
        )
        target = (
            mount_root
            / f"expanded_{ordinal % 7:02d}"
            / f"opaque_{ordinal:04d}_{hashlib.sha256(str(ordinal).encode()).hexdigest()[:10]}"
        )
        link_or_copy(source, target)

    checkpoint = contract["checkpoint"]
    container = one_content(
        source_inventory,
        str(checkpoint["container_sha256"]),
        "checkpoint container",
    )
    payloads: list[bytes] = []
    with tarfile.open(container, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("checkpoint member unreadable")
            payload = handle.read()
            if hashlib.sha256(payload).hexdigest() == checkpoint["sha256"]:
                payloads.append(payload)
    if len(payloads) != 1:
        raise RuntimeError(f"expected one checkpoint member, found {len(payloads)}")
    checkpoint_target = mount_root / "expanded_checkpoint" / "opaque_member"
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_target.write_bytes(payloads[0])
    if checkpoint_target.stat().st_size != checkpoint["bytes"]:
        raise RuntimeError("expanded checkpoint size drift")

    hashes = inventory(mount_root)
    return {
        "file_count": sum(len(paths) for paths in hashes.values()),
        "unique_hash_count": len(hashes),
        "checkpoint_transport": "expanded_raw_member_opaque_name",
        "external_bundle_manifest_present": False,
        "external_request_manifest_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--source-kernel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-roundtrip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = json.loads(args.bundle_manifest.read_text())
    requests = json.loads(args.request_manifest.read_text())
    roundtrip = json.loads(args.checkpoint_roundtrip.read_text())
    contract = build_contract(bundle, requests, roundtrip)
    kernel_source = args.kernel.read_text()
    source_kernel_text = args.source_kernel.read_text()
    forbidden_literals = {
        "external_bundle_manifest_sha256": sha256(args.bundle_manifest),
        "external_request_manifest_sha256": sha256(args.request_manifest),
    }
    for label, literal in forbidden_literals.items():
        if literal in kernel_source:
            raise RuntimeError(f"generated kernel embeds forbidden {label}")
    if "BUNDLE_MANIFEST_SHA" in kernel_source or "REQUEST_MANIFEST_SHA" in kernel_source:
        raise RuntimeError("generated kernel retains external manifest constants")
    if kernel_source.index("STEP0 = (") > kernel_source.index("import numpy as np"):
        raise RuntimeError("step 0 is not before heavy imports")
    science_boundary = "import numpy as np"
    generated_science = kernel_source[kernel_source.index(science_boundary) :]
    source_science = source_kernel_text[source_kernel_text.index(science_boundary) :]
    generated_science = generated_science.replace(
        "ASSET_DATASET_VERSION = 6", "ASSET_DATASET_VERSION = <TRANSPORT_VERSION>"
    )
    source_science = source_science.replace(
        "ASSET_DATASET_VERSION = 4", "ASSET_DATASET_VERSION = <TRANSPORT_VERSION>"
    )
    if generated_science != source_science:
        raise RuntimeError("frozen post-step0 science code changed")
    science_sha256 = hashlib.sha256(generated_science.encode()).hexdigest()

    with tempfile.TemporaryDirectory(prefix="cellect-mounted-v1.") as temporary:
        root = Path(temporary)
        mount = root / "kaggle_input_expanded"
        fixture = expanded_fixture(args.source_root, mount, contract)

        positive = run_step0(args.kernel, mount, root / "work_positive")
        if positive.returncode != 0:
            raise RuntimeError(
                f"expanded mount preflight failed:\n{positive.stdout}\n{positive.stderr}"
            )
        if "PASS_SELF_CONTAINED_EXPANDED_MOUNT" not in positive.stdout:
            raise RuntimeError("expanded mount did not emit the required PASS token")

        request_zero = contract["requests"][0]
        duplicate = mount / "duplicate" / "opaque_duplicate"
        duplicate_source = one_content(
            inventory(mount), str(request_zero["sha256"]), "duplicate request"
        )
        link_or_copy(duplicate_source, duplicate)
        duplicate_result = run_step0(
            args.kernel, mount, root / "work_duplicate"
        )
        if duplicate_result.returncode == 0 or "found 2" not in duplicate_result.stderr:
            raise RuntimeError("duplicate request did not fail closed")
        duplicate.unlink()

        missing_source = one_content(
            inventory(mount), str(request_zero["sha256"]), "missing request"
        )
        held = root / "held_outside_mount"
        missing_source.replace(held)
        missing_result = run_step0(args.kernel, mount, root / "work_missing")
        if missing_result.returncode == 0 or "found 0" not in missing_result.stderr:
            raise RuntimeError("missing request did not fail closed")
        held.replace(missing_source)

        downloaded = run_step0(
            args.kernel, args.source_root, root / "work_downloaded"
        )
        if downloaded.returncode != 0:
            raise RuntimeError(
                f"downloaded layout compatibility failed:\n"
                f"{downloaded.stdout}\n{downloaded.stderr}"
            )

        receipt = {
            "schema": "celltrack-alpha.cellect-bleed-stop-mounted-validation.v1",
            "status": "PASS_SELF_CONTAINED_MOUNTED_LAYOUT",
            "kernel_sha256": sha256(args.kernel),
            "embedded_asset_contract_sha256": hashlib.sha256(
                json.dumps(
                    contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest(),
            "request_file_count": contract["request_file_count"],
            "request_count": contract["request_count"],
            "class_file_counts": {"delete": 41, "occupied_swap": 36},
            "class_action_counts": {
                "delete": 1345497,
                "occupied_swap": 45961,
            },
            "expanded_mount_fixture": fixture,
            "expanded_mount_positive": "PASS",
            "downloaded_layout_compatibility": "PASS",
            "missing_required_content": "FAIL_CLOSED",
            "duplicate_required_content": "FAIL_CLOSED",
            "heavy_imports_before_step0": False,
            "post_step0_science_code_unchanged": True,
            "post_step0_science_code_sha256": science_sha256,
            "external_bundle_manifest_required": False,
            "external_request_manifest_required": False,
            "kernel_launch": False,
            "gpu_authorization": False,
            "gpu_debit_hours": 0,
        }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

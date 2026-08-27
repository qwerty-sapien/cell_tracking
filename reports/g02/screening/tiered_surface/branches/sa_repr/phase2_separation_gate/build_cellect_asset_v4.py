#!/usr/bin/env python3
"""Build the byte-preserving CELLECT Kaggle asset v4 staging directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

CHECKPOINT_SHA256 = "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
CHECKPOINT_BYTES = 10_197_221
SAFE_MEMBER_NAME = "cellect_uext_x3rd_149_0_4p6540.pth"
CONTAINER_NAME = "cellect_uext_x3rd_149_0_4p6540.tar"
NON_CHECKPOINT_BYTES = 9_270_385
METADATA_SHA256 = "8607c64835e53cd9c75c0c7974649b419d31e081c81a00b74cc587954b336732"


class AssetBuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_container(container: Path) -> dict[str, object]:
    try:
        archive = tarfile.open(container, mode="r:")
    except tarfile.TarError as error:
        raise AssetBuildError(f"container is not an uncompressed tar: {error}") from error
    with archive:
        members = archive.getmembers()
        if len(members) != 1 or not members[0].isfile():
            raise AssetBuildError("container must have exactly one regular member")
        member = members[0]
        if member.name != SAFE_MEMBER_NAME or not member.name.isascii():
            raise AssetBuildError("container member name is not the frozen safe ASCII name")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise AssetBuildError("container member cannot be read")
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: extracted.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if size != CHECKPOINT_BYTES or digest.hexdigest() != CHECKPOINT_SHA256:
        raise AssetBuildError("container member does not match the authenticated checkpoint pin")
    return {
        "container_path": container.name,
        "container_bytes": container.stat().st_size,
        "container_sha256": sha256(container),
        "member": SAFE_MEMBER_NAME,
        "member_bytes": size,
        "member_sha256": digest.hexdigest(),
        "compression": "none",
    }


def build_asset(
    checkpoint: Path,
    source_assets: Path,
    destination: Path,
    v3_manifest: Path,
) -> dict[str, object]:
    if destination.exists():
        raise AssetBuildError(f"destination already exists: {destination}")
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise AssetBuildError("source checkpoint does not match the authenticated size and SHA-256")

    frozen_v3 = json.loads(v3_manifest.read_text())
    non_checkpoint = [
        item for item in frozen_v3["files"] if item["sha256"] != CHECKPOINT_SHA256
    ]
    if sum(int(item["bytes"]) for item in non_checkpoint) != NON_CHECKPOINT_BYTES:
        raise AssetBuildError("frozen non-checkpoint byte total drift")

    for item in non_checkpoint:
        source = source_assets / item["path"]
        if (
            not source.is_file()
            or source.stat().st_size != int(item["bytes"])
            or sha256(source) != item["sha256"]
        ):
            raise AssetBuildError(f"frozen source asset drift: {item['path']}")

    metadata = source_assets / "dataset-metadata.json"
    if not metadata.is_file() or sha256(metadata) != METADATA_SHA256:
        raise AssetBuildError("dataset-metadata.json is missing or drifted")
    metadata_value = json.loads(metadata.read_text())
    if metadata_value.get("id") != frozen_v3["dataset"]["id"]:
        raise AssetBuildError("dataset-metadata.json retargeted the dataset id")

    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists():
        raise AssetBuildError(f"partial destination already exists: {partial}")
    partial.mkdir(parents=True)
    copied: list[dict[str, object]] = []
    for item in non_checkpoint:
        shutil.copyfile(source_assets / item["path"], partial / item["path"])
        copied.append(dict(item))
    shutil.copyfile(metadata, partial / metadata.name)

    container = partial / CONTAINER_NAME
    info = tarfile.TarInfo(SAFE_MEMBER_NAME)
    info.size = CHECKPOINT_BYTES
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with tarfile.open(container, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
        with checkpoint.open("rb") as source:
            archive.addfile(info, source)
    roundtrip = verify_container(container)
    partial.replace(destination)

    container_record = {
        "path": CONTAINER_NAME,
        "bytes": roundtrip["container_bytes"],
        "sha256": roundtrip["container_sha256"],
        "format": "uncompressed_tar",
        "member": {
            "path": SAFE_MEMBER_NAME,
            "bytes": CHECKPOINT_BYTES,
            "sha256": CHECKPOINT_SHA256,
        },
    }
    files = sorted([*copied, container_record], key=lambda item: str(item["path"]))
    return {
        "schema": "celltrack-alpha.cellect-r7-kaggle-asset-manifest.v4",
        "status": "LOCAL_STAGING_BYTE_EXACT",
        "dataset": {
            "id": "onebitbrain/g02-cellect-r7-published-assets-v1",
            "visibility": "private",
            "version_before": 3,
            "target_version": 4,
        },
        "checkpoint_transport": {
            "reason": "Kaggle transformed the direct .pth payload in dataset v3",
            "original_direct_bytes": CHECKPOINT_BYTES,
            "remote_v3_inferred_bytes": 9_238_024,
            "v3_byte_deficit": 959_197,
            **roundtrip,
        },
        "control_metadata": {
            "path": metadata.name,
            "bytes": metadata.stat().st_size,
            "sha256": METADATA_SHA256,
            "dataset_id": metadata_value["id"],
        },
        "files": files,
        "required_file_count": len(files),
        "required_total_bytes": sum(int(item["bytes"]) for item in files),
        "non_checkpoint_bytes": NON_CHECKPOINT_BYTES,
        "forbidden": frozen_v3["forbidden"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-assets", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_asset(
        args.checkpoint,
        args.source_assets,
        args.destination,
        args.v3_manifest,
    )
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["checkpoint_transport"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

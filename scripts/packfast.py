#!/usr/bin/env python3
"""Build the deterministic, sub-1MB fast inference zipapp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "kaggle_kernel/gpuinfer.py"
DEFAULT_REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel/fastsrc.zip"
LIMIT = 1_000_000


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact_registry(path: Path) -> bytes:
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    records = []
    for row in source.get("records", []):
        chunks = sum(
            1
            for member in row.get("member_vector", [])
            if member.get("artifact") == "image" and "/c/" in str(member.get("member_path", ""))
        )
        records.append({
            "split": row["split"],
            "dataset": row["dataset"],
            "shape": row["compact_metadata"]["image_shape"],
            "chunks": chunks,
        })
    records.sort(key=lambda row: (row["split"], row["dataset"]))
    train = sum(row["split"] == "train" for row in records)
    test = sum(row["split"] == "test" for row in records)
    image_chunks = sum(int(row["chunks"]) for row in records)
    if len(records) != 203 or train != 199 or test != 4 or image_chunks != 20_300:
        raise ValueError("source registry does not have the expected 203-image/20,300-chunk coverage")
    compact = {
        "artifact": "fast_image_registry",
        "schema_version": 1,
        "source_registry_sha256": _sha256_bytes(source_bytes),
        "counts": {"videos": len(records), "train": train, "test": test, "image_chunks": image_chunks},
        "records": records,
    }
    return (json.dumps(compact, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _write_zip(output: Path, members: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.read_bytes()
    registry = _compact_registry(args.registry)
    members = {"__main__.py": source, "fastreg.json": registry}
    source_bytes = sum(len(value) for value in members.values())
    if source_bytes >= LIMIT:
        raise ValueError(f"source package exceeds {LIMIT} bytes: {source_bytes}")
    _write_zip(args.output, members)
    if args.output.stat().st_size >= LIMIT:
        raise ValueError(f"compressed package exceeds {LIMIT} bytes: {args.output.stat().st_size}")
    report = {
        "output": str(args.output),
        "compressed_bytes": args.output.stat().st_size,
        "source_bytes": source_bytes,
        "package_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "entrypoint_sha256": _sha256_bytes(source),
        "registry_sha256": _sha256_bytes(registry),
        "limit_bytes": LIMIT,
        "members": sorted(members),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

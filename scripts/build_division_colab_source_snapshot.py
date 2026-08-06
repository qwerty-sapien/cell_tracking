#!/usr/bin/env python3
"""Build or verify the immutable minimal source snapshot used by Colab Cell 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    from scripts.colab_division_harness import (
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from colab_division_harness import canonical_json_bytes, sha256_bytes, sha256_file


SCHEMA = "celltrack-alpha.colab-source-snapshot.v1"
MANIFEST = "SOURCE_MANIFEST.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "scripts/colab_division_harness.py",
    "scripts/preprocess_division_colab_data.py",
    "reports/g02/screening/tiered_surface/branches/division_temporal_colab_v1/"
    "TRAINING_OBJECTIVE_TEMPLATE.json",
    "reports/g02/screening/tiered_surface/branches/division_temporal_colab_v1/"
    "COLAB_REQUIREMENTS.txt",
)


class SnapshotError(RuntimeError):
    """The Colab source snapshot is incomplete, mutable, or unsafe."""


def _relative(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() == MANIFEST
    ):
        raise SnapshotError(f"unsafe source path: {value!r}")
    return path


def _inventory(root: Path, relative_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in sorted(relative_paths, key=lambda path: path.as_posix()):
        path = root / relative
        if not path.is_file():
            raise SnapshotError(f"source file is absent: {relative}")
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def verify_snapshot(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("cannot read source snapshot manifest") from exc
    rows = manifest.get("files")
    if manifest.get("schema") != SCHEMA or not isinstance(rows, list):
        raise SnapshotError("source snapshot schema is invalid")
    paths = [_relative(str(row.get("path", ""))) for row in rows]
    observed = _inventory(root, paths)
    if observed != rows or sha256_bytes(canonical_json_bytes(observed)) != manifest.get(
        "tree_sha256"
    ):
        raise SnapshotError("source snapshot file or tree digest drifted")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != {row["path"] for row in rows} | {MANIFEST}:
        raise SnapshotError("source snapshot has undeclared or missing files")
    return manifest


def build_snapshot(
    output: Path,
    *,
    extra_files: list[str],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    relative_paths = [_relative(value) for value in (*REQUIRED_FILES, *extra_files)]
    if len(set(relative_paths)) != len(relative_paths):
        raise SnapshotError("source snapshot contains duplicate paths")
    rows = _inventory(repository_root, relative_paths)
    manifest = {
        "schema": SCHEMA,
        "files": rows,
        "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }
    if output.exists():
        existing = verify_snapshot(output)
        if existing != manifest:
            raise SnapshotError("existing immutable source snapshot conflicts")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.partial-", dir=output.parent) as raw:
        staging = Path(raw) / output.name
        staging.mkdir()
        for relative in relative_paths:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repository_root / relative, destination)
        (staging / MANIFEST).write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, output)
    return verify_snapshot(output)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Additional repository-relative entrypoint or imported source file; repeat as needed.",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = verify_snapshot(args.output) if args.verify else build_snapshot(
        args.output, extra_files=args.file
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

#!/usr/bin/env python3
"""Build the G-01 reproducibility manifest without reading secrets or full data blobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_OUTPUT = Path("reports/general_model_reproducibility_manifest.json")

SECRET_NAMES = {".env", "gurobi.lic"}
SECRET_SUFFIXES = {".pem", ".key"}
EXCLUDED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TARGET_PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "pandas",
    "zarr",
    "scikit-image",
    "networkx",
    "traccuracy",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_relpath(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_secret_or_cache(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts if path.is_absolute() else path.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    name = path.name
    if name in SECRET_NAMES:
        return True
    if any(name.endswith(suffix) for suffix in SECRET_SUFFIXES):
        return True
    return False


def iter_files(patterns: Iterable[str], exclude: set[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = normalized_relpath(path)
            if rel in exclude or is_secret_or_cache(path):
                continue
            files[rel] = path
    return [files[key] for key in sorted(files)]


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": normalized_relpath(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def zip_central_directory_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    count = 0
    total_uncompressed = 0
    total_compressed = 0
    with zipfile.ZipFile(path, "r") as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            count += 1
            total_uncompressed += int(info.file_size)
            total_compressed += int(info.compress_size)
            row = "\t".join(
                [
                    info.filename,
                    str(info.CRC),
                    str(info.file_size),
                    str(info.compress_size),
                    str(info.date_time),
                ]
            )
            digest.update(row.encode("utf-8"))
            digest.update(b"\n")
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "zip_member_count": count,
        "zip_uncompressed_bytes": total_uncompressed,
        "zip_compressed_member_bytes": total_compressed,
        "zip_central_directory_sha256": digest.hexdigest(),
        "full_byte_hash": "skipped_low_compute_large_archive",
    }


def git_capture(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return {"available": False, "args": args, "error": str(exc)}
    return {
        "available": True,
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def external_checkout_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": normalized_relpath(path) if path.is_absolute() else str(path), "exists": False}
    return {
        "path": normalized_relpath(path),
        "exists": True,
        "head": git_capture(["rev-parse", "HEAD"], cwd=path),
        "status_short": git_capture(["status", "--short"], cwd=path),
        "key_files": [
            file_record(p)
            for p in iter_files(
                [
                    f"{normalized_relpath(path)}/metrics.md",
                    f"{normalized_relpath(path)}/pyproject.toml",
                    f"{normalized_relpath(path)}/**/*.py",
                ],
                exclude=set(),
            )
        ],
    }


def dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    packages: dict[str, str | None] = {}
    for name in TARGET_PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    versions["packages"] = packages
    return versions


def load_run_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    statuses: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "statuses": statuses,
        "commands": [
            {
                "created_at": row.get("created_at", ""),
                "node": row.get("node", ""),
                "command": row.get("command", ""),
                "dataset_scope": row.get("dataset_scope", ""),
                "status": row.get("status", ""),
                "side_effects": row.get("side_effects", ""),
            }
            for row in rows
        ],
    }


def validate_no_secret_paths(records: Iterable[dict[str, Any]]) -> None:
    offenders = [record["path"] for record in records if is_secret_or_cache(Path(record["path"]))]
    if offenders:
        raise RuntimeError(f"manifest would include forbidden secret/cache paths: {offenders}")


def build_manifest(output: Path, zip_path: Path) -> dict[str, Any]:
    output_abs = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    exclude = {normalized_relpath(output_abs)}
    source_patterns = [
        "src/**/*.py",
        "scripts/*.py",
        "tests/*.py",
        "docs/*.md",
        "doc_cell_tracking/*.md",
        ".goal-interrogator/*.md",
    ]
    report_patterns = [
        "reports/general_model_*",
    ]
    derived_patterns = [
        "data/derived/edge_scorer/*.csv",
    ]
    source_records = [file_record(path) for path in iter_files(source_patterns, exclude)]
    report_records = [file_record(path) for path in iter_files(report_patterns, exclude)]
    derived_records = [file_record(path) for path in iter_files(derived_patterns, exclude)]
    validate_no_secret_paths([*source_records, *report_records, *derived_records])
    manifest = {
        "kind": "general_model_reproducibility_manifest",
        "created_at_unix": time.time(),
        "root": str(ROOT),
        "secret_exclusion": {
            "excluded_names": sorted(SECRET_NAMES),
            "excluded_suffixes": sorted(SECRET_SUFFIXES),
            "excluded_dirs": sorted(EXCLUDED_DIR_NAMES),
            "policy": "Do not read or hash secrets; include only path-exclusion policy, never values.",
        },
        "git": {
            "root_head": git_capture(["rev-parse", "--verify", "HEAD"]),
            "root_status_short": git_capture(["status", "--short"]),
            "root_diff_name_status": git_capture(["diff", "--name-status"]),
            "root_cached_diff_name_status": git_capture(["diff", "--cached", "--name-status"]),
        },
        "dependencies": dependency_versions(),
        "data": {
            "kaggle_zip": zip_central_directory_identity(ROOT / zip_path),
            "split_manifest": file_record(ROOT / "reports/general_model_split_manifest.json")
            if (ROOT / "reports/general_model_split_manifest.json").exists()
            else None,
        },
        "external": {
            "official_metric_checkout": external_checkout_identity(ROOT / "external/kaggle-cell-tracking-competition"),
        },
        "run_log": load_run_log(ROOT / "reports/general_model_run_log.csv"),
        "files": {
            "source_and_docs": source_records,
            "reports": report_records,
            "derived_tables": derived_records,
        },
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.output, args.zip)
    output = ROOT / args.output if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "source_files": len(manifest["files"]["source_and_docs"]), "report_files": len(manifest["files"]["reports"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

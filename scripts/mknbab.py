#!/usr/bin/env python3
"""Build the compact notebook wrapper for the private V45 D4 A/B package."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts import mkab45
except ModuleNotFoundError:
    import mkab45  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "kaggle_kernel" / "ab45p2"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "ab45n2"
NOTEBOOK_NAME = "ab45.ipynb"
METADATA_NAME = "kernel-metadata.json"
SOURCE_LIMIT = 1_000_000


class NotebookError(RuntimeError):
    """The notebook wrapper cannot preserve the A/B package contract."""


def notebook_payload(package_dir: Path, *, mode: str) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    if mode not in {"batch", "compile"}:
        raise NotebookError("A/B mode must be batch or compile")
    expected = {
        "ab45.py",
        "ab45man.json",
        "bh45.py",
        "bh45att.json",
        "bh45man.json",
        "bh45src.zip",
        "kernel-metadata.json",
    }
    actual = {path.name for path in package_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise NotebookError(f"A/B package file set changed: expected={sorted(expected)}, actual={sorted(actual)}")
    files = {
        name: base64.b64encode((package_dir / name).read_bytes()).decode("ascii")
        for name in sorted(expected)
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = f'''import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/kaggle/working/ab45pkg")
FILES = json.loads({json.dumps(encoded)})
ROOT.mkdir(parents=True, exist_ok=True)
for name, value in FILES.items():
    target = ROOT / name
    target.write_bytes(base64.b64decode(value, validate=True))
result = subprocess.run([sys.executable, str(ROOT / "ab45.py"), "--mode", {json.dumps(mode)}], check=False)
if result.returncode != 0:
    raise RuntimeError(f"V45 D4 A/B failed with exit code {{result.returncode}}; inspect /kaggle/working/v45ab.json")
print("V45 D4 A/B completed")
'''
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _metadata(kernel_slug: str) -> bytes:
    payload = {
        "id": f"onebitbrain/{kernel_slug}",
        "title": kernel_slug,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


def build_notebook(
    package_dir: Path = DEFAULT_PACKAGE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    kernel_slug: str = mkab45.KERNEL_SLUG,
    mode: str = "batch",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {NOTEBOOK_NAME, METADATA_NAME}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in expected)
    if unexpected:
        raise NotebookError(f"refusing to mix A/B notebook with unexpected files: {unexpected}")
    notebook = json.dumps(notebook_payload(package_dir, mode=mode), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
    if len(notebook) >= SOURCE_LIMIT:
        raise NotebookError(f"notebook source must be <{SOURCE_LIMIT} bytes, got {len(notebook)}")
    files = {NOTEBOOK_NAME: notebook, METADATA_NAME: _metadata(kernel_slug)}
    for name, value in files.items():
        temporary = output_dir / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output_dir / name)
    return {
        "schema": "biohub-v45-ab-notebook.v1",
        "output_dir": str(output_dir),
        "notebook_bytes": len(notebook),
        "source_limit": SOURCE_LIMIT,
        "strictly_under_budget": len(notebook) < SOURCE_LIMIT,
        "kernel": f"onebitbrain/{kernel_slug}",
        "mode": mode,
        "files": [
            {"path": name, "bytes": len(value)}
            for name, value in sorted(files.items())
        ],
        "network_or_kaggle_actions": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kernel-slug", default=mkab45.KERNEL_SLUG)
    parser.add_argument("--mode", choices=("batch", "compile"), default="batch")
    args = parser.parse_args(argv)
    print(json.dumps(build_notebook(args.package, args.output, kernel_slug=args.kernel_slug, mode=args.mode), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

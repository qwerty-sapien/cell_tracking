#!/usr/bin/env python3
"""Build a compact Kaggle UI notebook around the hashed V45 package."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "kaggle_kernel" / "bh45pkg"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "bh45.ipynb"
PACKAGE_FILES = (
    "bh45.py",
    "bh45att.json",
    "bh45man.json",
    "bh45src.zip",
    "kernel-metadata.json",
)
SOURCE_LIMIT = 1_000_000


class NotebookPackageError(RuntimeError):
    """The UI notebook cannot preserve the hashed package contract."""


def notebook_payload(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    files: dict[str, str] = {}
    for name in PACKAGE_FILES:
        path = package_dir / name
        if not path.is_file():
            raise NotebookPackageError(f"package member is missing: {path}")
        files[name] = base64.b64encode(path.read_bytes()).decode("ascii")
    unexpected = sorted(path.name for path in package_dir.iterdir() if path.is_file() and path.name not in files)
    if unexpected:
        raise NotebookPackageError(f"unexpected package files: {unexpected}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = f'''import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/kaggle/working/bh45pkg")
FILES = json.loads({json.dumps(encoded)})
ROOT.mkdir(parents=True, exist_ok=True)
for name, value in FILES.items():
    destination = ROOT / name
    destination.write_bytes(base64.b64decode(value, validate=True))
result = subprocess.run([sys.executable, str(ROOT / "bh45.py")], check=False)
if result.returncode != 0:
    raise RuntimeError(f"V45 launcher failed with exit code {{result.returncode}}; inspect /kaggle/working/bhdiag.json")
print("V45 private control completed")
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


def build_notebook(package_dir: Path = DEFAULT_PACKAGE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = notebook_payload(package_dir)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
    if len(raw) >= SOURCE_LIMIT:
        raise NotebookPackageError(f"notebook source must be <{SOURCE_LIMIT} bytes, got {len(raw)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(raw)
    temporary.replace(output)
    return {"path": str(output), "bytes": len(raw), "strictly_under_budget": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build_notebook(args.package, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

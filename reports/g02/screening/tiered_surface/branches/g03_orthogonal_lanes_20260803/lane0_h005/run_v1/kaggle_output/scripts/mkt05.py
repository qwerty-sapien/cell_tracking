#!/usr/bin/env python3
"""Build the minimal private Kaggle upload directory for the attested T-05 notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle_kernel" / "t05.ipynb"
META = ROOT / "kaggle_kernel" / "t05.meta.json"
PACK = ROOT / "kaggle_kernel" / "t05.pack.json"
OUTPUT = ROOT / "kaggle_kernel" / "t05pkg"
KERNEL = "onebitbrain/biohub-t05d"


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain an object")
    return value


def metadata() -> dict:
    return {
        "code_file": "t05.ipynb",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "thibautgoldsborough/cellmot-baseline-artifacts",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "id": KERNEL,
        "is_private": True,
        "kernel_type": "notebook",
        "language": "python",
        "title": "biohub-t05d",
    }


def build() -> dict[Path, bytes]:
    package = read_json(PACK)
    source = read_json(META).get("source")
    notebook = NOTEBOOK.read_bytes()
    if not isinstance(source, dict) or source.get("sha256") != json.loads(notebook.decode("utf-8"))["metadata"].get("t05_script_sha256"):
        raise ValueError("T-05 notebook/source attestation mismatch")
    if package.get("notebook", {}).get("source_cell_sha256") != source.get("sha256"):
        raise ValueError("T-05 package/source attestation mismatch")
    return {
        OUTPUT / "t05.ipynb": notebook,
        OUTPUT / "kernel-metadata.json": (json.dumps(metadata(), ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii"),
        OUTPUT / "t05push.json": canonical({
            "artifact": "t05-push-v1",
            "kernel": KERNEL,
            "notebook": {"bytes": len(notebook), "sha256": sha256(notebook)},
            "private": True,
            "scored_submission": False,
            "training": True,
        }),
    }


def write() -> None:
    for path, data in build().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def check() -> None:
    expected = build()
    for path, data in expected.items():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"T-05 upload package drift: {path.relative_to(ROOT)}")
    package = read_json(OUTPUT / "kernel-metadata.json")
    if package != metadata() or package["id"] != KERNEL or not package["is_private"] or package["enable_internet"]:
        raise ValueError("T-05 upload metadata is unsafe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        check()
    else:
        write()
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

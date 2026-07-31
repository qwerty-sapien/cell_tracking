#!/usr/bin/env python3
"""Materialize immutable Kaggle packages for the G-03 Phase 1 proof.

The builder writes only into a fresh caller-selected directory.  It never replaces a
previous package: a retry must use a new explicit directory and a new preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "g03_phase1_a.py",
    "g03_phase1_b.py",
    "g03_resume_torch.py",
    "g03_checkpoint_resume.py",
)
CONFIG = {
    "seed": 20260731,
    "input_width": 256,
    "hidden_width": 512,
    "batch_size": 768,
    "learning_rate": 0.0003,
    "segment_minimum_seconds": 3600.0,
    "checkpoint_interval_seconds": 600.0,
    "amp_enabled": False,
    "dataloader_workers": 0,
    "device": "cuda",
}
HANDOFF_ID = "g03-phase1-a-to-b-v1"


class BuildError(RuntimeError):
    """The package cannot be materialized faithfully."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def source_hashes() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in SOURCES:
        path = ROOT / "scripts" / name
        if not path.is_file():
            raise BuildError(f"source is absent: {path}")
        values[name] = sha256(path)
    return values


def role_contract(role: str, hashes: dict[str, str]) -> dict[str, Any]:
    kernel_id = f"onebitbrain/g03-phase1-resume-{role.lower()}-v1"
    return {
        "schema": "celltrack-alpha.g03-phase1-contract.v1",
        "role": role,
        "kernel_id": kernel_id,
        "producer_kernel": "onebitbrain/g03-phase1-resume-a-v1",
        "handoff_id": HANDOFF_ID,
        "source_hashes": hashes,
        "config": CONFIG,
        "scientific_selection_authorized": False,
    }


def metadata(role: str) -> dict[str, Any]:
    dataset_sources = []
    if role == "B":
        dataset_sources = [
            "onebitbrain/g03-phase1-handoff-v1",
            "onebitbrain/trackastra-h1-fit105-cache-v1",
        ]
    return {
        "id": f"onebitbrain/g03-phase1-resume-{role.lower()}-v1",
        "title": f"g03-phase1-resume-{role.lower()}-v1",
        "code_file": f"g03_phase1_{role.lower()}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": dataset_sources,
    }


def materialize_role(root: Path, role: str, hashes: dict[str, str]) -> dict[str, Any]:
    directory = root / role.lower()
    directory.mkdir(parents=False)
    for name in SOURCES:
        shutil.copy2(ROOT / "scripts" / name, directory / name)
    (directory / "PHASE1_CONTRACT.json").write_bytes(canonical(role_contract(role, hashes)))
    (directory / "kernel-metadata.json").write_bytes(canonical(metadata(role)))
    observed = {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
    return {"path": str(directory), "sha256": observed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new empty package directory")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise BuildError(f"refusing to replace existing package output: {args.output}")
    if args.output.parent.exists() and not args.output.parent.is_dir():
        raise BuildError(f"package parent is not a directory: {args.output.parent}")
    hashes = source_hashes()
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "celltrack-alpha.g03-phase1-package-build.v1",
        "source_hashes": hashes,
        "source_bundle_sha256": hashlib.sha256(canonical(hashes)).hexdigest(),
        "roles": [materialize_role(args.output, role, hashes) for role in ("A", "B")],
    }
    (args.output / "BUILD_RECEIPT.json").write_bytes(canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

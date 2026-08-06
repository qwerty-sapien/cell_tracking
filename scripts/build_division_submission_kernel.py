#!/usr/bin/env python3
"""Build the hash-pinned, division-capable Lane 1 submission package.

The package is intentionally a post-graph emitter.  It has one missing input at
handoff time: a passing 44b6 D1 candidate receipt.  It does not contain a
division selector, fit any weights, score a held graph, or submit anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "kaggle_kernel" / "g02_division_lane1"
DEFAULT_MANIFEST = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division"
    / "LANE1_DIVISION_PACKAGE_MANIFEST.json"
)
FROZEN_D1 = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01"
    / "d1_persistent_fork"
)
FROZEN_N1_INTERFACE = (
    ROOT / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json"
)

PACKAGE_SOURCES = {
    "division_submission_kernel.py": ROOT / "scripts/division_submission_runtime.py",
    "frozen_d1_decoder.py": FROZEN_D1 / "d1_decoder.py",
    "frozen_d1_evidence.json": FROZEN_D1 / "N1_dry_validation.json",
    "frozen_n1_interface.json": FROZEN_N1_INTERFACE,
    "celltrack_alpha/action_compiler.py": ROOT / "src/celltrack_alpha/action_compiler.py",
    "celltrack_alpha/kaggle_calibration.py": ROOT / "src/celltrack_alpha/kaggle_calibration.py",
}

EXPECTED_SHA256 = {
    "frozen_d1_decoder.py": "e780a398a6586e0c6acd7fb89d54626e732b254992252db91e20f5385a4bbcf6",
    "frozen_d1_evidence.json": "f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9",
    "frozen_n1_interface.json": "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "celltrack_alpha/action_compiler.py": "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _copy_source(relative: str, source: Path, package: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"package source is missing: {source}")
    observed = sha256_file(source)
    expected = EXPECTED_SHA256.get(relative)
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"frozen input drifted for {relative}: expected {expected}, got {observed}"
        )
    destination = package / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = sha256_file(destination)
    if copied != observed:
        raise RuntimeError(f"copy hash mismatch for {relative}")
    return {"path": relative, "bytes": destination.stat().st_size, "sha256": copied}


def _write_kernel_entrypoint(package: Path) -> dict[str, Any]:
    source = """#!/usr/bin/env python3
\"\"\"Kaggle entrypoint for frozen D1 division emission.\"\"\"
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from division_submission_kernel import main


if __name__ == \"__main__\":
    base = os.environ.get(\"LANE1_BASE_SUBMISSION\")
    candidates = os.environ.get(\"LANE1_DIVISION_CANDIDATES\")
    if not base or not candidates:
        raise SystemExit(
            \"Set LANE1_BASE_SUBMISSION and LANE1_DIVISION_CANDIDATES before executing the division emitter\"
        )
    raise SystemExit(main([\"--base-submission\", base, \"--candidate-receipt\", candidates, \"--output\", \"/kaggle/working/submission.csv\"]))
"""
    path = package / "kernel.py"
    path.write_text(source, encoding="utf-8")
    return {"path": "kernel.py", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _write_runtime_package_init(package: Path) -> dict[str, Any]:
    """Write the intentionally minimal package initializer.

    The repository initializer imports the image-pipeline surface, which is not
    part of this post-graph emission package.  Keeping this namespace empty
    proves the emitted kernel needs only the stdlib graph contract.
    """

    path = package / "celltrack_alpha" / "__init__.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"""Minimal Lane 1 graph-emission namespace; no image pipeline is bundled."""\n',
        encoding="utf-8",
    )
    return {
        "path": "celltrack_alpha/__init__.py",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_metadata(package: Path) -> dict[str, Any]:
    metadata = {
        "id": "onebitbrain/g02-division-lane1-ready",
        "title": "g02-division-lane1-ready",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [],
    }
    path = package / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": "kernel-metadata.json", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_package(package: Path, manifest_path: Path) -> dict[str, Any]:
    package.mkdir(parents=True, exist_ok=True)
    copied = [_copy_source(relative, source, package) for relative, source in PACKAGE_SOURCES.items()]
    copied.append(_write_runtime_package_init(package))
    copied.append(_write_kernel_entrypoint(package))
    copied.append(_write_metadata(package))
    identity = {
        "schema": "celltrack-alpha.lane1-division-package.v1",
        "entrypoint": "kernel.py",
        "runtime_entrypoint": "division_submission_kernel.py",
        "files": sorted(copied, key=lambda item: str(item["path"])),
        "division_mode": {
            "enabled": True,
            "selector_included": False,
            "frozen_decoder": "D1_LATE_PERSISTENT_ONE_TO_ONE_FORK",
            "atomic_compiler": "N1_DIVISION_FORK",
            "candidate_gate": "44b6 combined>0, Jdiv>=0.30, nonnegative adjusted/raw/node, zero duplicates",
        },
        "scoring": {
            "official_scorer_fixture_evidence": "reused only; no fixture or held-graph scorer execution",
            "official_fixture_count": 14,
            "counterfactual_count": 51,
            "duplicate_survivors": 0,
        },
        "network_or_kaggle_actions": False,
        "submission_action": False,
    }
    manifest = {
        "schema": "celltrack-alpha.lane1-division-package-manifest.v1",
        "package_identity_sha256": canonical_sha256(identity),
        "identity": identity,
    }
    for destination in (package / "package_manifest.json", manifest_path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_package(args.package.resolve(), args.manifest.resolve())
    print(f"package_identity_sha256={manifest['package_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

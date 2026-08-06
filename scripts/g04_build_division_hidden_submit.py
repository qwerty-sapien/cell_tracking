#!/usr/bin/env python3
"""Build the hidden-rerun-capable G-04 Lane-A Kaggle package.

The scored APNCC monolithic kernel is embedded byte-for-byte.  After APNCC
generates a submission for the currently mounted (public or hidden) test set,
the frozen G-04 division postprocessor transforms that fresh output in place.
This builder performs no Kaggle or network action.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle_kernel/bhap-submit"
BASE_CODE = BASE_DIR / "bh45.py"
BASE_MANIFEST = BASE_DIR / "bh45man.json"
BASE_METADATA = BASE_DIR / "kernel-metadata.json"
POSTPROCESSOR = ROOT / "scripts/g04_division_hidden_postprocess.py"
RESULTS_ARCHIVE = ROOT / "results-bhap.zip"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel/g04divcap5-v1"
KERNEL_SLUG = "g04divcap5-v1"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
COMPETITION = "biohub-cell-tracking-during-development"
SUPPORT = "pilkwang/biohub-tracking-support-pack-50ep-v1"
BASE_CODE_SHA256 = "eafac752d0b67ec35e18bf6a6f2d03ecc1f9d1d27e0813959df00880cfd77db5"
BASE_PACKAGE_IDENTITY = "779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b"
BASE_PUBLIC_SUBMISSION_SHA256 = "6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3"
PROMOTED_PUBLIC_SHA256 = "da73c7540dc0f1052b1f6f0d1399a9a9867f24de9c8463180fa6eef2bb484509"
GLOBAL_DEADLINE_SECONDS = 360 * 60
SOURCE_BUDGET_BYTES = 1_000_000


class BuildError(RuntimeError):
    """The hidden-rerun package cannot be built without contract drift."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def metadata_bytes() -> bytes:
    return (
        json.dumps(
            {
                "code_file": "main.py",
                "competition_sources": [COMPETITION],
                "dataset_sources": [SUPPORT],
                "enable_gpu": True,
                "enable_internet": False,
                "id": KERNEL_REF,
                "is_private": True,
                "kernel_type": "script",
                "language": "python",
                "machine_shape": "NvidiaTeslaT4",
                "title": KERNEL_SLUG,
            },
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def wrapper_bytes(base_source: bytes, postprocessor: bytes) -> bytes:
    base_sha = sha256_bytes(base_source)
    post_sha = sha256_bytes(postprocessor)
    base64_source = base64.b64encode(base_source).decode("ascii")
    post64_source = base64.b64encode(postprocessor).decode("ascii")
    source = f'''#!/usr/bin/env python3
"""Run exact scored APNCC, then the frozen hidden-safe G-04 division transform."""

import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STARTED = time.monotonic()
GLOBAL_DEADLINE_SECONDS = {GLOBAL_DEADLINE_SECONDS}
BASE_SHA256 = {base_sha!r}
POSTPROCESS_SHA256 = {post_sha!r}
BASE_SOURCE = base64.b64decode({base64_source!r}, validate=True)
POSTPROCESS_SOURCE = base64.b64decode({post64_source!r}, validate=True)
WORK = Path("/kaggle/working")
PARENT = WORK / "g04_apncc_parent.csv"
FINAL = WORK / "submission.csv"
POSTPROCESS_PATH = WORK / ".g04_division_hidden_postprocess.py"
RECEIPT = WORK / "G04_DIVISION_POSTPROCESS_RECEIPT.json"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def competition_root():
    candidates = (
        Path("/kaggle/input/competitions/{COMPETITION}"),
        Path("/kaggle/input/{COMPETITION}"),
    )
    for candidate in candidates:
        if (candidate / "test").is_dir():
            return candidate
    raise RuntimeError("competition test input is missing from both known mounts")


if sha256(BASE_SOURCE) != BASE_SHA256 or sha256(POSTPROCESS_SOURCE) != POSTPROCESS_SHA256:
    raise RuntimeError("embedded source SHA-256 mismatch")

# Preserve the scored APNCC stage byte-for-byte.  Its inner launcher owns the
# T4x2 attestation, offline support validation, inference, and process deadline.
base_globals = {{
    "__name__": "__main__",
    "__file__": "/kaggle/working/.g04_exact_scored_bhap_submit.py",
    "__package__": None,
    "__cached__": None,
}}
try:
    exec(compile(BASE_SOURCE, base_globals["__file__"], "exec"), base_globals)
except SystemExit as exc:
    code = 0 if exc.code is None else int(exc.code)
    if code != 0:
        raise RuntimeError("exact APNCC stage failed with exit code %s" % code) from exc

remaining = GLOBAL_DEADLINE_SECONDS - (time.monotonic() - STARTED)
if remaining <= 0:
    raise RuntimeError("global 360-minute deadline expired before division postprocess")
if not FINAL.is_file() or PARENT.exists() or RECEIPT.exists():
    raise RuntimeError("fresh APNCC output or postprocess target state is invalid")
os.replace(FINAL, PARENT)
POSTPROCESS_PATH.write_bytes(POSTPROCESS_SOURCE)
if sha256(POSTPROCESS_PATH.read_bytes()) != POSTPROCESS_SHA256:
    raise RuntimeError("materialized postprocessor SHA-256 mismatch")

command = [
    sys.executable,
    str(POSTPROCESS_PATH),
    "--parent", str(PARENT),
    "--input-root", str(competition_root()),
    "--output", str(FINAL),
    "--receipt", str(RECEIPT),
    "--deadline-seconds", str(remaining),
]
process = subprocess.Popen(command, start_new_session=True)
try:
    returncode = process.wait(timeout=remaining)
except subprocess.TimeoutExpired as exc:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    raise RuntimeError("global 360-minute deadline terminated division postprocess") from exc
if returncode != 0:
    raise RuntimeError("division postprocess failed with exit code %s" % returncode)
if not FINAL.is_file() or not RECEIPT.is_file():
    raise RuntimeError("division postprocess returned without final output and receipt")
receipt = json.loads(RECEIPT.read_text(encoding="ascii"))
if (
    receipt.get("status") != "PASS"
    or receipt.get("dataset_discovery", {{}}).get("public_identity_pinned") is not False
    or receipt.get("dataset_discovery", {{}}).get("inventory_exact_match") is not True
    or receipt.get("transform", {{}}).get("labels_read") is not False
    or receipt.get("transform", {{}}).get("computed_raw_intensity_summaries") is not True
):
    raise RuntimeError("division postprocess receipt failed terminal predicates")
print(json.dumps({{
    "status": "PASS",
    "base_source_sha256": BASE_SHA256,
    "postprocess_source_sha256": POSTPROCESS_SHA256,
    "parent_sha256": receipt["parent"]["sha256"],
    "submission_sha256": receipt["output"]["sha256"],
    "datasets": receipt["dataset_discovery"]["datasets"],
    "selected_actions": receipt["transform"]["selected_actions"],
    "elapsed_seconds": time.monotonic() - STARTED,
}}, sort_keys=True))
'''
    return source.encode("ascii")


def validate_base() -> dict[str, Any]:
    for path in (BASE_CODE, BASE_MANIFEST, BASE_METADATA, POSTPROCESSOR, RESULTS_ARCHIVE):
        if not path.is_file():
            raise BuildError(f"required input is absent: {path}")
    if sha256_file(BASE_CODE) != BASE_CODE_SHA256:
        raise BuildError("scored APNCC monolithic source drifted")
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="ascii"))
    if manifest.get("identity_sha256") != BASE_PACKAGE_IDENTITY:
        raise BuildError("scored APNCC package identity drifted")
    metadata = json.loads(BASE_METADATA.read_text(encoding="ascii"))
    if metadata.get("id") != "onebitbrain/bhap-submit":
        raise BuildError("scored APNCC kernel identity drifted")
    with zipfile.ZipFile(RESULTS_ARCHIVE, "r") as archive:
        public_submission = archive.read("submission.csv")
    if sha256_bytes(public_submission) != BASE_PUBLIC_SUBMISSION_SHA256:
        raise BuildError("banked scored APNCC output drifted")
    return {
        "kernel_ref": "onebitbrain/bhap-submit",
        "script_version_id": 337554797,
        "submission_ref": 54942864,
        "package_identity_sha256": BASE_PACKAGE_IDENTITY,
        "monolithic_code_sha256": BASE_CODE_SHA256,
        "public_submission_sha256": BASE_PUBLIC_SUBMISSION_SHA256,
        "public_score": 0.881,
    }


def build_package(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    base = validate_base()
    output_dir = output_dir.resolve()
    if len(KERNEL_SLUG) > 15:
        raise BuildError("kernel slug exceeds the governed 15-character limit")
    base_source = BASE_CODE.read_bytes()
    postprocessor = POSTPROCESSOR.read_bytes()
    code = wrapper_bytes(base_source, postprocessor)
    metadata = metadata_bytes()
    if len(code) + len(metadata) >= SOURCE_BUDGET_BYTES:
        raise BuildError("kernel package exceeds the one-megabyte source budget")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {"main.py", "kernel-metadata.json"}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in expected_names)
    if unexpected:
        raise BuildError(f"refusing to mix package with unexpected files: {unexpected}")
    for name, value in (("main.py", code), ("kernel-metadata.json", metadata)):
        path = output_dir / name
        if path.exists() and path.read_bytes() != value:
            raise BuildError(f"refusing to overwrite drifted package member: {path}")
        path.write_bytes(value)
    identity = {
        "schema": "celltrack-alpha.g04-hidden-division-package.v1",
        "kernel_ref": KERNEL_REF,
        "network": "forbidden",
        "private": True,
        "competition_source": COMPETITION,
        "dataset_sources": [SUPPORT],
        "machine_shape": "NvidiaTeslaT4",
        "global_deadline_seconds": GLOBAL_DEADLINE_SECONDS,
        "source_budget_bytes": SOURCE_BUDGET_BYTES,
        "package_bytes": len(code) + len(metadata),
        "base": base,
        "postprocess": {
            "source": str(POSTPROCESSOR),
            "source_sha256": sha256_bytes(postprocessor),
            "policy": {
                "name": "unfiltered_geometry_cap5",
                "ranking": "geometry",
                "cap": 5,
                "appearance_candidate_cap": 8,
                "public_identity_pinned": False,
                "labels_read": False,
            },
        },
        "files": [
            {"path": "kernel-metadata.json", "bytes": len(metadata), "sha256": sha256_bytes(metadata)},
            {"path": "main.py", "bytes": len(code), "sha256": sha256_bytes(code)},
        ],
        "public_parity_target": {
            "parent_sha256": BASE_PUBLIC_SUBMISSION_SHA256,
            "candidate_sha256": PROMOTED_PUBLIC_SHA256,
        },
        "kaggle_actions": False,
    }
    return {
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_bytes(identity)),
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    report = build_package(args.output)
    if args.receipt is not None:
        path = args.receipt.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise BuildError(f"refusing to overwrite build receipt: {path}")
        path.write_bytes(canonical_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

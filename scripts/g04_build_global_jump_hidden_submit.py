#!/usr/bin/env python3
"""Build the hidden-rerun-capable APNCC+G1 Kaggle package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle_kernel/bhap-submit"
BASE_CODE = BASE_DIR / "bh45.py"
BASE_MANIFEST = BASE_DIR / "bh45man.json"
BASE_METADATA = BASE_DIR / "kernel-metadata.json"
POSTPROCESSOR = ROOT / "scripts/g04_global_jump_hidden_postprocess.py"
RESULTS_ARCHIVE = ROOT / "results-bhap.zip"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel/g1jump-submit"
KERNEL_SLUG = "g1jump-submit"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
COMPETITION = "biohub-cell-tracking-during-development"
SUPPORT = "pilkwang/biohub-tracking-support-pack-50ep-v1"
BASE_CODE_SHA256 = "eafac752d0b67ec35e18bf6a6f2d03ecc1f9d1d27e0813959df00880cfd77db5"
BASE_PACKAGE_IDENTITY = "779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b"
BASE_PUBLIC_SUBMISSION_SHA256 = "6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3"
PUBLIC_G1_SHA256 = "366a12d64e155d2722fccb579d76fb0c040a889f47f857ae089430f613c9a8c8"
GLOBAL_DEADLINE_SECONDS = 360 * 60
SOURCE_BUDGET_BYTES = 1_000_000


class BuildError(RuntimeError):
    """The package cannot be built without contract drift."""


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
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
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
"""Run exact scored APNCC, then frozen hidden-safe G1 registration."""

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
PARENT = WORK / "g1_apncc_parent.csv"
FINAL = WORK / "submission.csv"
POSTPROCESS_PATH = WORK / ".g04_global_jump_hidden_postprocess.py"
RECEIPT = WORK / "G04_GLOBAL_JUMP_RECEIPT.json"


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

base_globals = {{
    "__name__": "__main__",
    "__file__": "/kaggle/working/.g1_exact_scored_bhap_submit.py",
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
    raise RuntimeError("global 360-minute deadline expired before G1 postprocess")
if not FINAL.is_file() or PARENT.exists() or RECEIPT.exists():
    raise RuntimeError("fresh APNCC output or G1 target state is invalid")
diagnostic = json.loads((WORK / "bhdiag.json").read_text(encoding="ascii"))
support = Path(str(diagnostic.get("support_selected", "")))
source_root = WORK / "bh45src"
if not support.is_dir() or not source_root.is_dir():
    raise RuntimeError("authenticated support or materialized APNCC source is absent")
os.replace(FINAL, PARENT)
POSTPROCESS_PATH.write_bytes(POSTPROCESS_SOURCE)
if sha256(POSTPROCESS_PATH.read_bytes()) != POSTPROCESS_SHA256:
    raise RuntimeError("materialized G1 postprocessor SHA-256 mismatch")

command = [
    sys.executable,
    str(POSTPROCESS_PATH),
    "--parent", str(PARENT),
    "--input-root", str(competition_root()),
    "--source-root", str(source_root),
    "--support-root", str(support),
    "--work-root", str(WORK),
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
    raise RuntimeError("global 360-minute deadline terminated G1 postprocess") from exc
if returncode != 0:
    raise RuntimeError("G1 postprocess failed with exit code %s" % returncode)
if not FINAL.is_file() or not RECEIPT.is_file():
    raise RuntimeError("G1 postprocess returned without final output and receipt")
receipt = json.loads(RECEIPT.read_text(encoding="ascii"))
if (
    receipt.get("status") != "PASS"
    or receipt.get("dataset_discovery", {{}}).get("public_identity_pinned") is not False
    or receipt.get("dataset_discovery", {{}}).get("inventory_exact_match") is not True
    or receipt.get("transform", {{}}).get("labels_read") is not False
    or receipt.get("transform", {{}}).get("fresh_submission_generation") is not True
    or receipt.get("transform", {{}}).get("public_only_output_copy") is not False
):
    raise RuntimeError("G1 postprocess receipt failed terminal predicates")
print(json.dumps({{
    "status": "PASS",
    "base_source_sha256": BASE_SHA256,
    "postprocess_source_sha256": POSTPROCESS_SHA256,
    "parent_sha256": receipt["parent"]["sha256"],
    "submission_sha256": receipt["output"]["sha256"],
    "datasets": receipt["dataset_discovery"]["datasets"],
    "pool_breaking_transitions": receipt["transform"].get("pool_breaking_transitions", 0),
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
    parity = ROOT / "reports/g04/lanes/data_pathology_repair/global_jump_public_transfer_v1/PUBLIC_RAW_PARITY_RECEIPT.json"
    payload = json.loads(parity.read_text(encoding="ascii"))
    if (
        payload.get("status") != "PASS"
        or payload.get("parent", {}).get("sha256") != BASE_PUBLIC_SUBMISSION_SHA256
        or payload.get("output", {}).get("sha256") != PUBLIC_G1_SHA256
        or payload.get("dataset_discovery", {}).get("public_identity_pinned") is not False
        or payload.get("transform", {}).get("labels_read") is not False
    ):
        raise BuildError("public raw-data G1 parity receipt drifted")
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
    base_source = BASE_CODE.read_bytes()
    postprocessor = POSTPROCESSOR.read_bytes()
    code = wrapper_bytes(base_source, postprocessor)
    metadata = metadata_bytes()
    if len(KERNEL_SLUG) > 15 or len(code) + len(metadata) >= SOURCE_BUDGET_BYTES:
        raise BuildError("kernel slug/source budget contract failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {"main.py", "kernel-metadata.json"}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in expected_names)
    if unexpected:
        raise BuildError(f"refusing unexpected package files: {unexpected}")
    for name, value in (("main.py", code), ("kernel-metadata.json", metadata)):
        path = output_dir / name
        if path.exists() and path.read_bytes() != value:
            raise BuildError(f"refusing to overwrite drifted package member: {path}")
        path.write_bytes(value)
    identity = {
        "schema": "celltrack-alpha.g04-global-jump-hidden-package.v1",
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
            "public_parity_output_sha256": PUBLIC_G1_SHA256,
            "dynamic_test_discovery": True,
            "same_run_cache_required": True,
            "labels_read": False,
        },
        "files": [
            {"path": "kernel-metadata.json", "bytes": len(metadata), "sha256": sha256_bytes(metadata)},
            {"path": "main.py", "bytes": len(code), "sha256": sha256_bytes(code)},
        ],
        "kaggle_actions": False,
    }
    return {
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_bytes(identity)),
        "output_dir": str(output_dir),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
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

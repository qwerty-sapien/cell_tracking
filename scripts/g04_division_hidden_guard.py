#!/usr/bin/env python3
"""Fail-closed prelaunch and terminal guards for G-04 hidden transport v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "kaggle_kernel/g04divcap5-v1"
REPORT = ROOT / "reports/g04/lanes/division_permissive/hidden_transport_v1"
BUILD_RECEIPT = REPORT / "BUILD_RECEIPT.json"
PREREGISTRATION = REPORT / "PREREGISTRATION.json"
REGISTRY = REPORT / "TRANSPORT_REGISTRY.json"
RAW_PARITY = REPORT / "PUBLIC_RAW_PARITY_RECEIPT.json"
INVALIDATION = (
    ROOT
    / "reports/g04/lanes/division_permissive/code_transport_v1"
    / "TRANSPORT_INVALIDATION.json"
)
DEFAULT_PRELAUNCH = REPORT / "KERNEL_PRELAUNCH_RECEIPT.json"
DEFAULT_PREDICATE = REPORT / "CODE_SUBMISSION_PREDICATE.json"
KERNEL_ID = "onebitbrain/g04divcap5-v1"
COMPETITION = "biohub-cell-tracking-during-development"
CODE_SHA256 = "63f60d5c8c15c149239385221f4de54947c0e0beeef656f62a140f27b13f1bcf"
METADATA_SHA256 = "b28d10d483523fc0dc7c5947c828ff3a3348cdbef6740250229574f717240fd5"
PACKAGE_IDENTITY_SHA256 = "11256286686fae610e94295662fbe2c9dd730206d3dbf853d187559e11ba6eb2"
POSTPROCESS_SHA256 = "4b10b179fca2300a7e27e54021cd5f30d645c8b59062ff496250d96d0094d356"
PARENT_SHA256 = "6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3"
PARENT_BYTES = 13_712_542
PARENT_ROWS = 263_282
OUTPUT_SHA256 = "da73c7540dc0f1052b1f6f0d1399a9a9867f24de9c8463180fa6eef2bb484509"
OUTPUT_BYTES = 13_713_344
OUTPUT_ROWS = 263_297
OUTPUT_NODES = 134_409
OUTPUT_EDGES = 128_888
OUTPUT_DIVIDING = 15
PUBLIC_DATASETS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class GuardError(RuntimeError):
    """A required transport identity or predicate failed."""


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


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardError(f"unreadable JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardError(f"JSON value is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical_bytes(dict(value)))
    os.replace(temporary, path)


def _prelaunch_result() -> dict[str, Any]:
    failures: list[str] = []
    code_path = PACKAGE / "main.py"
    metadata_path = PACKAGE / "kernel-metadata.json"
    if sorted(path.name for path in PACKAGE.iterdir()) != ["kernel-metadata.json", "main.py"]:
        failures.append("package file inventory drift")
    if sha256_file(code_path) != CODE_SHA256:
        failures.append("kernel code SHA drift")
    if sha256_file(metadata_path) != METADATA_SHA256:
        failures.append("kernel metadata SHA drift")
    try:
        source = code_path.read_text(encoding="ascii")
        compile(source, str(code_path), "exec")
    except Exception as exc:
        source = ""
        failures.append(f"kernel code does not compile as ASCII Python: {exc}")
    for marker in (
        "BASE_SHA256 = 'eafac752d0b67ec35e18bf6a6f2d03ecc1f9d1d27e0813959df00880cfd77db5'",
        f"POSTPROCESS_SHA256 = '{POSTPROCESS_SHA256}'",
        "g04_apncc_parent.csv",
        "G04_DIVISION_POSTPROCESS_RECEIPT.json",
        "inventory_exact_match",
        "global 360-minute deadline",
    ):
        if marker not in source:
            failures.append(f"kernel source marker absent: {marker}")
    metadata = load_object(metadata_path)
    if metadata != {
        "code_file": "main.py",
        "competition_sources": [COMPETITION],
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "enable_gpu": True,
        "enable_internet": False,
        "id": KERNEL_ID,
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": "g04divcap5-v1",
    }:
        failures.append("kernel metadata contract drift")

    build = load_object(BUILD_RECEIPT)
    identity = build.get("identity")
    if not isinstance(identity, dict):
        failures.append("build identity absent")
    else:
        if build.get("identity_sha256") != PACKAGE_IDENTITY_SHA256:
            failures.append("build package identity drift")
        if identity.get("package_bytes") != 411_809:
            failures.append("build package byte count drift")
        if identity.get("kaggle_actions") is not False:
            failures.append("build receipt claims a Kaggle action")
        if identity.get("base", {}).get("package_identity_sha256") != (
            "779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b"
        ):
            failures.append("scored base package drift")
        if identity.get("postprocess", {}).get("source_sha256") != POSTPROCESS_SHA256:
            failures.append("postprocess source drift")

    prereg = load_object(PREREGISTRATION)
    if prereg.get("status") != "FROZEN_BEFORE_REMOTE_EXECUTION":
        failures.append("transport preregistration is not frozen")
    freeze = prereg.get("implementation_freeze", {})
    for key, path in (
        ("postprocessor_sha256", ROOT / "scripts/g04_division_hidden_postprocess.py"),
        ("builder_sha256", ROOT / "scripts/g04_build_division_hidden_submit.py"),
        ("test_sha256", ROOT / "tests/test_g04_division_hidden_submit.py"),
    ):
        if freeze.get(key) != sha256_file(path):
            failures.append(f"implementation freeze drift: {key}")

    parity = load_object(RAW_PARITY)
    if (
        parity.get("status") != "PASS"
        or parity.get("parent", {}).get("sha256") != PARENT_SHA256
        or parity.get("output", {}).get("sha256") != OUTPUT_SHA256
        or parity.get("output", {}).get("bytes") != OUTPUT_BYTES
        or parity.get("output", {}).get("rows") != OUTPUT_ROWS
        or parity.get("output", {}).get("dividing_nodes") != OUTPUT_DIVIDING
        or parity.get("dataset_discovery", {}).get("public_identity_pinned") is not False
        or parity.get("dataset_discovery", {}).get("inventory_exact_match") is not True
        or parity.get("transform", {}).get("computed_raw_intensity_summaries") is not True
        or parity.get("transform", {}).get("labels_read") is not False
        or parity.get("transform", {}).get("selected_actions") != 15
    ):
        failures.append("public raw-Zarr parity receipt drift")
    invalidation = load_object(INVALIDATION)
    if (
        invalidation.get("status") != "TERMINAL_INVALIDATED_NO_RETRY"
        or invalidation.get("supersedes", {}).get("eligible") is not False
    ):
        failures.append("old copy-transport invalidation drift")
    step_line = next(
        (index for index, line in enumerate(source.splitlines(), start=1) if line.startswith("if sha256(BASE_SOURCE)")),
        None,
    )
    passed = not failures
    return {
        "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
        "status": "PASS" if passed else "FAIL",
        "eligible": passed,
        "phase": "inline",
        "lane_owner": "G04 Lane A",
        "kernel_id": KERNEL_ID,
        "remote_preflight_kernel_id": None,
        "kernel_dir": str(PACKAGE.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "code_file": str(code_path.resolve()),
        "code_sha256": sha256_file(code_path),
        "contract_sha256": sha256_file(PREREGISTRATION),
        "registry_sha256": sha256_file(REGISTRY),
        "dataset_sources": metadata.get("dataset_sources"),
        "kernel_sources": metadata.get("kernel_sources", []),
        "exclusive_dataset_source": "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "expected_mount_root": "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
        "step0_line": step_line,
        "inline_boundary": "embedded base/postprocess SHA checks precede exact scored APNCC launcher; APNCC validates support/runtime and discovers mounted test metadata before inference",
        "package_identity_sha256": PACKAGE_IDENTITY_SHA256,
        "public_raw_parity_receipt_sha256": sha256_file(RAW_PARITY),
        "failures": failures,
    }


def prelaunch(output: Path = DEFAULT_PRELAUNCH) -> dict[str, Any]:
    result = _prelaunch_result()
    atomic_json(output.resolve(), result)
    return result


def validate_terminal(
    output_dir: Path,
    *,
    kernel_version: int,
    kernel_output_receipt: Path,
    prelaunch_receipt: Path = DEFAULT_PRELAUNCH,
    predicate_path: Path = DEFAULT_PREDICATE,
) -> dict[str, Any]:
    if isinstance(kernel_version, bool) or kernel_version <= 0:
        raise GuardError("kernel_version must be a positive integer")
    failures: list[str] = []
    current_prelaunch = _prelaunch_result()
    try:
        bound_prelaunch = load_object(prelaunch_receipt.resolve())
        if bound_prelaunch != current_prelaunch:
            failures.append("bound prelaunch receipt is stale or drifted")
    except GuardError as exc:
        failures.append(str(exc))
    output_dir = output_dir.resolve()
    kernel_output_receipt = kernel_output_receipt.resolve()
    try:
        download = load_object(kernel_output_receipt)
        argv = download.get("argv")
        expected_ref = f"{KERNEL_ID}/{kernel_version}"
        download_ok = (
            download.get("schema") == "celltrack-alpha.cmux-op-receipt.v1"
            and download.get("op") == "kernel_output"
            and download.get("mutating") is False
            and download.get("returncode") == 0
            and download.get("process_returncode") == 0
            and download.get("timed_out") is False
            and download.get("semantic_failure") is False
            and isinstance(argv, list)
            and len(argv) >= 6
            and argv[1:3] == ["kernels", "output"]
            and argv[3] == expected_ref
            and argv[4] == "-p"
            and Path(str(argv[5])).resolve() == output_dir
        )
        if not download_ok:
            failures.append("kernel output receipt is not bound to the exact version and path")
    except GuardError as exc:
        failures.append(str(exc))
    parent_path = output_dir / "g04_apncc_parent.csv"
    output_path = output_dir / "submission.csv"
    runtime_receipt_path = output_dir / "G04_DIVISION_POSTPROCESS_RECEIPT.json"
    for label, path, expected_sha, expected_bytes in (
        ("parent", parent_path, PARENT_SHA256, PARENT_BYTES),
        ("output", output_path, OUTPUT_SHA256, OUTPUT_BYTES),
    ):
        try:
            if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
                failures.append(f"downloaded {label} byte identity drift")
        except OSError as exc:
            failures.append(f"downloaded {label} absent: {exc}")
    try:
        runtime = load_object(runtime_receipt_path)
        if (
            runtime.get("schema") != "celltrack-alpha.g04-hidden-division-postprocess.v1"
            or runtime.get("status") != "PASS"
            or runtime.get("dataset_discovery", {}).get("datasets") != PUBLIC_DATASETS
            or runtime.get("dataset_discovery", {}).get("inventory_exact_match") is not True
            or runtime.get("dataset_discovery", {}).get("public_identity_pinned") is not False
            or runtime.get("parent", {}).get("sha256") != PARENT_SHA256
            or runtime.get("parent", {}).get("rows") != PARENT_ROWS
            or runtime.get("output", {}).get("sha256") != OUTPUT_SHA256
            or runtime.get("output", {}).get("bytes") != OUTPUT_BYTES
            or runtime.get("output", {}).get("rows") != OUTPUT_ROWS
            or runtime.get("output", {}).get("nodes") != OUTPUT_NODES
            or runtime.get("output", {}).get("edges") != OUTPUT_EDGES
            or runtime.get("output", {}).get("dividing_nodes") != OUTPUT_DIVIDING
            or runtime.get("transform", {}).get("selected_actions") != 15
            or runtime.get("transform", {}).get("computed_raw_intensity_summaries") is not True
            or runtime.get("transform", {}).get("labels_read") is not False
        ):
            failures.append("runtime postprocess receipt terminal predicate drift")
    except GuardError as exc:
        runtime = {}
        failures.append(str(exc))
    eligible = not failures
    predicate = {
        "schema": "celltrack-alpha.g04-hidden-code-submission-predicate.v1",
        "status": "PASS" if eligible else "FAIL",
        "eligible": eligible,
        "competition": COMPETITION,
        "kernel_id": KERNEL_ID,
        "kernel_version": kernel_version,
        "output_file": "submission.csv",
        "required_submission_flags": ["-k", "-v", "-f", "submission.csv"],
        "submission_mode": "code_output",
        "direct_file_upload": False,
        "hidden_rerun_compatible": True,
        "dynamic_test_discovery": True,
        "fresh_submission_generation": True,
        "public_only_output_copy": False,
        "revoked": False,
        "consumed": False,
        "output_sha256": OUTPUT_SHA256,
        "output_bytes": OUTPUT_BYTES,
        "output_data_rows": OUTPUT_ROWS,
        "downloaded_output_dir": str(output_dir),
        "kernel_output_receipt": str(kernel_output_receipt),
        "kernel_output_receipt_sha256": sha256_file(kernel_output_receipt) if kernel_output_receipt.is_file() else None,
        "prelaunch_receipt": str(prelaunch_receipt.resolve()),
        "prelaunch_receipt_sha256": sha256_file(prelaunch_receipt.resolve()) if prelaunch_receipt.is_file() else None,
        "runtime_receipt_sha256": sha256_file(runtime_receipt_path) if runtime_receipt_path.is_file() else None,
        "package_identity_sha256": PACKAGE_IDENTITY_SHA256,
        "failures": failures,
        "retry_policy": "one exact code submission only; no automatic or blind retry",
    }
    atomic_json(predicate_path.resolve(), predicate)
    return predicate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("prelaunch")
    pre.add_argument("--output", type=Path, default=DEFAULT_PRELAUNCH)
    terminal = subparsers.add_parser("validate-terminal")
    terminal.add_argument("--output-dir", type=Path, required=True)
    terminal.add_argument("--kernel-version", type=int, required=True)
    terminal.add_argument("--kernel-output-receipt", type=Path, required=True)
    terminal.add_argument("--prelaunch-receipt", type=Path, default=DEFAULT_PRELAUNCH)
    terminal.add_argument("--output", type=Path, default=DEFAULT_PREDICATE)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "prelaunch":
        result = prelaunch(args.output)
    else:
        result = validate_terminal(
            args.output_dir,
            kernel_version=args.kernel_version,
            kernel_output_receipt=args.kernel_output_receipt,
            prelaunch_receipt=args.prelaunch_receipt,
            predicate_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("eligible") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed prelaunch and terminal guards for the APNCC+G1 transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "kaggle_kernel/g1jump-submit"
REPORT = ROOT / "reports/g04/lanes/data_pathology_repair/global_jump_public_transfer_v1"
BUILD_RECEIPT = REPORT / "BUILD_RECEIPT.json"
PREREGISTRATION = REPORT / "PREREGISTRATION.json"
REGISTRY = REPORT / "TRANSPORT_REGISTRY.json"
RAW_PARITY = REPORT / "PUBLIC_RAW_PARITY_RECEIPT.json"
DEFAULT_PRELAUNCH = REPORT / "KERNEL_PRELAUNCH_RECEIPT.json"
DEFAULT_PREDICATE = REPORT / "CODE_SUBMISSION_PREDICATE.json"
KERNEL_ID = "onebitbrain/g1jump-submit"
COMPETITION = "biohub-cell-tracking-during-development"
CODE_SHA256 = "0252bc39f739737aa2cc30187219fa5f0d447611afde0f580929e4bdb7911d99"
METADATA_SHA256 = "e41a74ba75ef716be8dd4684bf3ad512eee8e48944b1c32e5959d84e79ff6400"
PACKAGE_IDENTITY_SHA256 = "95f40b681b875ebedfac9b7f1cd4c6cd458fe941c37a036c4f01cb448a2e3479"
POSTPROCESS_SHA256 = "9ee74040f10426942909c480c295a5fe4e7da4e5e245a9f4ee373b92fed81000"
PARENT_SHA256 = "6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3"
PARENT_BYTES = 13_712_542
PARENT_ROWS = 263_282
PARENT_NODES = 134_409
PARENT_EDGES = 128_873
OUTPUT_SHA256 = "366a12d64e155d2722fccb579d76fb0c040a889f47f857ae089430f613c9a8c8"
OUTPUT_BYTES = 13_448_137
OUTPUT_ROWS = 263_260
OUTPUT_NODES = 134_409
OUTPUT_EDGES = 128_851
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
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
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
    try:
        inventory = sorted(path.name for path in PACKAGE.iterdir())
    except OSError as exc:
        inventory = []
        failures.append(f"package inventory unreadable: {exc}")
    if inventory != ["kernel-metadata.json", "main.py"]:
        failures.append("package file inventory drift")
    for label, path, expected in (
        ("kernel code", code_path, CODE_SHA256),
        ("kernel metadata", metadata_path, METADATA_SHA256),
    ):
        try:
            if sha256_file(path) != expected:
                failures.append(f"{label} SHA drift")
        except OSError as exc:
            failures.append(f"{label} unreadable: {exc}")
    try:
        source = code_path.read_text(encoding="ascii")
        compile(source, str(code_path), "exec")
    except Exception as exc:
        source = ""
        failures.append(f"kernel code does not compile as ASCII Python: {exc}")
    for marker in (
        "BASE_SHA256 = 'eafac752d0b67ec35e18bf6a6f2d03ecc1f9d1d27e0813959df00880cfd77db5'",
        f"POSTPROCESS_SHA256 = '{POSTPROCESS_SHA256}'",
        "g1_apncc_parent.csv",
        "G04_GLOBAL_JUMP_RECEIPT.json",
        "inventory_exact_match",
        "global 360-minute deadline",
    ):
        if marker not in source:
            failures.append(f"kernel source marker absent: {marker}")

    try:
        metadata = load_object(metadata_path)
    except GuardError as exc:
        metadata = {}
        failures.append(str(exc))
    expected_metadata = {
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
        "title": "g1jump-submit",
    }
    if metadata != expected_metadata:
        failures.append("kernel metadata contract drift")

    try:
        build = load_object(BUILD_RECEIPT)
        identity = build.get("identity", {})
        if (
            build.get("identity_sha256") != PACKAGE_IDENTITY_SHA256
            or identity.get("package_bytes") != 406_614
            or identity.get("kaggle_actions") is not False
            or identity.get("base", {}).get("package_identity_sha256")
            != "779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b"
            or identity.get("postprocess", {}).get("source_sha256") != POSTPROCESS_SHA256
        ):
            failures.append("build identity drift")
    except GuardError as exc:
        failures.append(str(exc))

    try:
        prereg = load_object(PREREGISTRATION)
        freeze = prereg.get("implementation_freeze", {})
        if prereg.get("status") != "FROZEN_BEFORE_REMOTE_EXECUTION":
            failures.append("transport preregistration is not frozen")
        for key, path in (
            ("postprocessor_sha256", ROOT / "scripts/g04_global_jump_hidden_postprocess.py"),
            ("builder_sha256", ROOT / "scripts/g04_build_global_jump_hidden_submit.py"),
            ("submitter_sha256", ROOT / "scripts/g04_global_jump_hidden_submit.py"),
        ):
            if not path.is_file() or freeze.get(key) != sha256_file(path):
                failures.append(f"implementation freeze drift: {key}")
    except GuardError as exc:
        failures.append(str(exc))

    try:
        registry = load_object(REGISTRY)
        if (
            registry.get("status") != "ACTIVE_ONE_SHOT"
            or registry.get("kernel_id") != KERNEL_ID
            or registry.get("expected_initial_kernel_version") != 1
            or registry.get("revoked") is not False
            or registry.get("consumed") is not False
        ):
            failures.append("transport registry is not active and unused")
    except GuardError as exc:
        failures.append(str(exc))

    try:
        parity = load_object(RAW_PARITY)
        if (
            parity.get("status") != "PASS"
            or parity.get("parent", {}).get("sha256") != PARENT_SHA256
            or parity.get("parent", {}).get("bytes") != PARENT_BYTES
            or parity.get("parent", {}).get("rows") != PARENT_ROWS
            or parity.get("output", {}).get("sha256") != OUTPUT_SHA256
            or parity.get("output", {}).get("bytes") != OUTPUT_BYTES
            or parity.get("output", {}).get("rows") != OUTPUT_ROWS
            or parity.get("output", {}).get("nodes") != OUTPUT_NODES
            or parity.get("output", {}).get("edges") != OUTPUT_EDGES
            or parity.get("dataset_discovery", {}).get("datasets") != PUBLIC_DATASETS
            or parity.get("dataset_discovery", {}).get("public_identity_pinned") is not False
            or parity.get("dataset_discovery", {}).get("inventory_exact_match") is not True
            or parity.get("transform", {}).get("labels_read") is not False
            or parity.get("transform", {}).get("observation_distinct_from_parent") is not True
            or parity.get("transform", {}).get("pool_breaking_transitions") != 1
            or parity.get("transform", {}).get("fresh_submission_generation") is not True
            or parity.get("transform", {}).get("public_only_output_copy") is not False
        ):
            failures.append("public raw-Zarr parity receipt drift")
    except GuardError as exc:
        failures.append(str(exc))
    try:
        candidate = REPORT / "PUBLIC_G1_SUBMISSION.csv"
        if candidate.stat().st_size != OUTPUT_BYTES or sha256_file(candidate) != OUTPUT_SHA256:
            failures.append("local public parity output identity drift")
    except OSError as exc:
        failures.append(f"local public parity output absent: {exc}")

    step_line = next(
        (i for i, line in enumerate(source.splitlines(), 1) if line.startswith("if sha256(BASE_SOURCE)")),
        None,
    )
    passed = not failures
    return {
        "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
        "status": "PASS" if passed else "FAIL",
        "eligible": passed,
        "phase": "inline",
        "lane_owner": "G04 global-jump public transfer",
        "kernel_id": KERNEL_ID,
        "remote_preflight_kernel_id": None,
        "kernel_dir": str(PACKAGE.resolve()),
        "metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        "code_file": str(code_path.resolve()),
        "code_sha256": sha256_file(code_path) if code_path.is_file() else None,
        "contract_sha256": sha256_file(PREREGISTRATION) if PREREGISTRATION.is_file() else None,
        "registry_sha256": sha256_file(REGISTRY) if REGISTRY.is_file() else None,
        "expected_initial_kernel_version": 1,
        "dataset_sources": metadata.get("dataset_sources"),
        "kernel_sources": metadata.get("kernel_sources", []),
        "exclusive_dataset_source": "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "expected_mount_root": "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
        "step0_line": step_line,
        "inline_boundary": "embedded identities precede exact APNCC; G1 then discovers the mounted test inventory and consumes only same-run fragments/caches",
        "package_identity_sha256": PACKAGE_IDENTITY_SHA256,
        "public_raw_parity_receipt_sha256": sha256_file(RAW_PARITY) if RAW_PARITY.is_file() else None,
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
        if load_object(prelaunch_receipt.resolve()) != current_prelaunch:
            failures.append("bound prelaunch receipt is stale or drifted")
    except GuardError as exc:
        failures.append(str(exc))
    output_dir = output_dir.resolve()
    kernel_output_receipt = kernel_output_receipt.resolve()
    try:
        download = load_object(kernel_output_receipt)
        argv = download.get("argv")
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
            and argv[3] == f"{KERNEL_ID}/{kernel_version}"
            and argv[4] == "-p"
            and Path(str(argv[5])).resolve() == output_dir
        )
        if not download_ok:
            failures.append("kernel output receipt is not bound to the exact version and path")
    except GuardError as exc:
        failures.append(str(exc))

    parent_path = output_dir / "g1_apncc_parent.csv"
    output_path = output_dir / "submission.csv"
    runtime_path = output_dir / "G04_GLOBAL_JUMP_RECEIPT.json"
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
        runtime = load_object(runtime_path)
        if (
            runtime.get("schema") != "celltrack-alpha.g04-global-jump-hidden-postprocess.v1"
            or runtime.get("status") != "PASS"
            or runtime.get("dataset_discovery", {}).get("datasets") != PUBLIC_DATASETS
            or runtime.get("dataset_discovery", {}).get("inventory_exact_match") is not True
            or runtime.get("dataset_discovery", {}).get("public_identity_pinned") is not False
            or runtime.get("parent", {}).get("sha256") != PARENT_SHA256
            or runtime.get("parent", {}).get("bytes") != PARENT_BYTES
            or runtime.get("parent", {}).get("rows") != PARENT_ROWS
            or runtime.get("parent", {}).get("nodes") != PARENT_NODES
            or runtime.get("parent", {}).get("edges") != PARENT_EDGES
            or runtime.get("output", {}).get("sha256") != OUTPUT_SHA256
            or runtime.get("output", {}).get("bytes") != OUTPUT_BYTES
            or runtime.get("output", {}).get("rows") != OUTPUT_ROWS
            or runtime.get("output", {}).get("nodes") != OUTPUT_NODES
            or runtime.get("output", {}).get("edges") != OUTPUT_EDGES
            or runtime.get("mechanism", {}).get("candidate_radius_um") != 10.0
            or runtime.get("mechanism", {}).get("candidate_cap") != 8
            or runtime.get("transform", {}).get("pool_breaking_transitions") != 1
            or runtime.get("transform", {}).get("observation_distinct_from_parent") is not True
            or runtime.get("transform", {}).get("fresh_submission_generation") is not True
            or runtime.get("transform", {}).get("public_only_output_copy") is not False
            or runtime.get("transform", {}).get("labels_read") is not False
        ):
            failures.append("runtime global-jump receipt terminal predicate drift")
    except GuardError as exc:
        failures.append(str(exc))
    eligible = not failures
    predicate = {
        "schema": "celltrack-alpha.g04-global-jump-code-submission-predicate.v1",
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
        "runtime_receipt_sha256": sha256_file(runtime_path) if runtime_path.is_file() else None,
        "package_identity_sha256": PACKAGE_IDENTITY_SHA256,
        "failures": failures,
        "retry_policy": "one exact code submission only; no automatic or blind retry",
    }
    atomic_json(predicate_path.resolve(), predicate)
    return predicate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prelaunch")
    pre.add_argument("--output", type=Path, default=DEFAULT_PRELAUNCH)
    terminal = sub.add_parser("validate-terminal")
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

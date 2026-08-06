#!/usr/bin/env python3
"""Build and consume the one-shot exact-version V32 hidden submission predicate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable

import cmux_ops
import kaggle_submission_lifecycle as lifecycle
import public_temporalunet_baseline as baseline


COMPETITION = baseline.COMPETITION
KERNEL_ID = "onebitbrain/g02-v32-submit"
OUTPUT_FILE = "submission.csv"
PREDICATE_SCHEMA = "celltrack-alpha.public-temporalunet-hidden-code-submission-predicate.v1"


class HiddenSubmitError(cmux_ops.PredicateNotSatisfied):
    """The hidden output is not eligible for the exact-version submission."""


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HiddenSubmitError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HiddenSubmitError(f"{label} must be a JSON object")
    return value


def exact_status_complete(receipt: dict[str, Any], *, kernel_id: str) -> bool:
    argv = receipt.get("argv")
    return (
        receipt.get("schema") == "celltrack-alpha.cmux-op-receipt.v1"
        and receipt.get("op") == "kernel_status"
        and receipt.get("returncode") == 0
        and receipt.get("process_returncode") == 0
        and receipt.get("timed_out") is False
        and receipt.get("semantic_failure") is False
        and isinstance(argv, list)
        and len(argv) == 4
        and argv[1:3] == ["kernels", "status"]
        and argv[3] == kernel_id
        and "KernelWorkerStatus.COMPLETE" in str(receipt.get("stdout_scrubbed", ""))
    )


def exact_output_download(
    receipt: dict[str, Any], *, kernel_id: str, version: int, output_dir: Path
) -> bool:
    argv = receipt.get("argv")
    return (
        receipt.get("schema") == "celltrack-alpha.cmux-op-receipt.v1"
        and receipt.get("op") == "kernel_output"
        and receipt.get("returncode") == 0
        and receipt.get("process_returncode") == 0
        and receipt.get("timed_out") is False
        and receipt.get("semantic_failure") is False
        and isinstance(argv, list)
        and len(argv) >= 6
        and argv[1:3] == ["kernels", "output"]
        and argv[3] == f"{kernel_id}/{version}"
        and argv[4] == "-p"
        and Path(str(argv[5])).resolve() == output_dir.resolve()
    )


def revalidate_hidden(
    *, package_dir: Path, output_dir: Path, expected: dict[str, Any]
) -> None:
    try:
        recorded_package_dir = Path(
            str(expected["package_manifest"]["path"])
        ).parent
        recorded_output_dir = Path(str(expected["submission"]["path"])).parent
    except (KeyError, TypeError) as exc:
        raise HiddenSubmitError(
            "hidden terminal validation is missing its recorded package/output paths"
        ) from exc
    if (
        recorded_package_dir.resolve() != package_dir.resolve()
        or recorded_output_dir.resolve() != output_dir.resolve()
    ):
        raise HiddenSubmitError(
            "hidden terminal validation paths do not resolve to the predicate inputs"
        )
    with tempfile.TemporaryDirectory(prefix="v32-hidden-recheck-") as directory:
        result = baseline.validate_hidden_output(
            SimpleNamespace(
                # Preserve the receipt's original relative/absolute spelling so
                # an otherwise byte-identical revalidation does not fail merely
                # because predicate construction canonicalized its input dirs.
                package_dir=recorded_package_dir,
                output_dir=recorded_output_dir,
                output=Path(directory) / "validation.json",
            )
        )
    if result != expected:
        raise HiddenSubmitError("fresh hidden-output revalidation differs from its receipt")


def predicate_payload(
    *,
    version: int,
    package_dir: Path,
    output_dir: Path,
    terminal_validation: Path,
    terminal_status_receipt: Path,
    kernel_output_receipt: Path,
    prelaunch_receipt: Path,
) -> dict[str, Any]:
    if version != 1:
        raise HiddenSubmitError("the V32 hidden lifecycle requires exact Version 1")
    package_dir = package_dir.resolve()
    output_dir = output_dir.resolve()
    terminal_validation = terminal_validation.resolve()
    terminal_status_receipt = terminal_status_receipt.resolve()
    kernel_output_receipt = kernel_output_receipt.resolve()
    prelaunch_receipt = prelaunch_receipt.resolve()
    package_path = package_dir / "PACKAGE_MANIFEST.json"
    output_path = output_dir / OUTPUT_FILE
    package = load_object(package_path, "hidden package")
    validation = load_object(terminal_validation, "hidden terminal validation")
    status = load_object(terminal_status_receipt, "terminal status receipt")
    download = load_object(kernel_output_receipt, "exact-version output receipt")
    prelaunch = load_object(prelaunch_receipt, "inline prelaunch receipt")
    revalidate_hidden(package_dir=package_dir, output_dir=output_dir, expected=validation)
    failures: list[str] = []
    if (
        package.get("schema") != baseline.HIDDEN_PACKAGE_SCHEMA
        or package.get("kernel_id") != KERNEL_ID
        or package.get("expected_initial_kernel_version") != version
        or package.get("hidden_test_compatible") is not True
        or package.get("dynamic_test_discovery") is not True
        or package.get("public_identity_pinned") is not False
        or package.get("fresh_submission_generation") is not True
        or package.get("public_output_copied") is not False
    ):
        failures.append("hidden package identity or generation contract drift")
    if (
        prelaunch.get("schema") != "celltrack-alpha.kaggle-prelaunch-guard-result.v1"
        or prelaunch.get("status") != "PASS"
        or prelaunch.get("eligible") is not True
        or prelaunch.get("phase") != "inline"
        or prelaunch.get("kernel_id") != KERNEL_ID
        or prelaunch.get("code_sha256")
        != package.get("files", {}).get("public_temporalunet_hidden_kernel.py")
        or prelaunch.get("failures") != []
    ):
        failures.append("inline prelaunch receipt drift")
    if not exact_status_complete(status, kernel_id=KERNEL_ID):
        failures.append("kernel is not terminal COMPLETE on the exact slug")
    if not exact_output_download(
        download, kernel_id=KERNEL_ID, version=version, output_dir=output_dir
    ):
        failures.append("download receipt is not bound to exact Version 1 and output path")
    if (
        validation.get("status") != "PASS"
        or validation.get("eligible_for_submission_predicate") is not True
        or validation.get("kernel_id") != KERNEL_ID
        or validation.get("expected_kernel_version") != version
        or validation.get("dataset_discovery", {}).get("dynamic_test_discovery") is not True
        or validation.get("dataset_discovery", {}).get("public_identity_pinned") is not False
        or validation.get("dataset_discovery", {}).get("inventory_exact_match") is not True
        or validation.get("generation", {}).get("fresh_submission_generation") is not True
        or validation.get("generation", {}).get("public_output_copied") is not False
        or validation.get("logging", {}).get("status")
        != "PERSISTED_STRUCTURED_RUN_LOG_AUTHENTICATED"
        or validation.get("weight_refinement") is not False
        or validation.get("g1_applied") is not False
    ):
        failures.append("terminal hidden validation is ineligible")
    try:
        output_sha = lifecycle.sha256_file(output_path)
        output_bytes = output_path.stat().st_size
    except OSError as exc:
        failures.append(f"hidden submission output unavailable: {exc}")
        output_sha = None
        output_bytes = None
    if output_sha != validation.get("submission", {}).get("sha256"):
        failures.append("hidden submission output hash differs from terminal validation")
    local_gate = package.get("local_submission_gate", {})
    if (
        local_gate.get("eligible") is not True
        or float(local_gate.get("local_score", 0.0)) < 0.91
        or local_gate.get("public_output_copy_allowed") is not False
        or local_gate.get("g1_assumed_additive") is not False
    ):
        failures.append("local replication gate no longer permits submission")
    eligible = not failures
    return {
        "schema": PREDICATE_SCHEMA,
        "status": "PASS" if eligible else "FAIL",
        "eligible": eligible,
        "competition": COMPETITION,
        "kernel_id": KERNEL_ID,
        "kernel_version": version,
        "output_file": OUTPUT_FILE,
        "required_submission_flags": ["-k", "-v", "-f", OUTPUT_FILE],
        "submission_mode": "code_output",
        "direct_file_upload": False,
        "hidden_rerun_compatible": True,
        "dynamic_test_discovery": True,
        "fresh_submission_generation": True,
        "public_only_output_copy": False,
        "public_identity_pinned": False,
        "revoked": False,
        "consumed": False,
        "output_sha256": output_sha,
        "output_bytes": output_bytes,
        "output_data_rows": validation.get("submission", {}).get("rows"),
        "output_datasets": validation.get("graphs", {}).get("datasets"),
        "package_dir": str(package_dir),
        "package_manifest_sha256": lifecycle.sha256_file(package_path),
        "downloaded_output_dir": str(output_dir),
        "terminal_validation": str(terminal_validation),
        "terminal_validation_sha256": lifecycle.sha256_file(terminal_validation),
        "terminal_status_receipt": str(terminal_status_receipt),
        "terminal_status_receipt_sha256": lifecycle.sha256_file(terminal_status_receipt),
        "kernel_output_receipt": str(kernel_output_receipt),
        "kernel_output_receipt_sha256": lifecycle.sha256_file(kernel_output_receipt),
        "prelaunch_receipt": str(prelaunch_receipt),
        "prelaunch_receipt_sha256": lifecycle.sha256_file(prelaunch_receipt),
        "local_submission_gate_sha256": local_gate.get("gate_sha256"),
        "local_cv57_score": local_gate.get("local_score"),
        "weight_refinement": False,
        "g1_applied": False,
        "failures": failures,
        "retry_policy": "one exact code submission only; no retry",
    }


def validate_predicate(path: Path, *, version: int) -> dict[str, Any]:
    predicate = load_object(path.resolve(), "V32 hidden submission predicate")
    expected_static = {
        "schema": PREDICATE_SCHEMA,
        "status": "PASS",
        "eligible": True,
        "competition": COMPETITION,
        "kernel_id": KERNEL_ID,
        "kernel_version": version,
        "output_file": OUTPUT_FILE,
        "required_submission_flags": ["-k", "-v", "-f", OUTPUT_FILE],
        "submission_mode": "code_output",
        "direct_file_upload": False,
        "hidden_rerun_compatible": True,
        "dynamic_test_discovery": True,
        "fresh_submission_generation": True,
        "public_only_output_copy": False,
        "public_identity_pinned": False,
        "revoked": False,
        "consumed": False,
        "weight_refinement": False,
        "g1_applied": False,
        "failures": [],
    }
    drift = [key for key, expected in expected_static.items() if predicate.get(key) != expected]
    if drift:
        raise HiddenSubmitError(f"hidden submission predicate field drift: {drift}")
    regenerated = predicate_payload(
        version=version,
        package_dir=Path(str(predicate["package_dir"])),
        output_dir=Path(str(predicate["downloaded_output_dir"])),
        terminal_validation=Path(str(predicate["terminal_validation"])),
        terminal_status_receipt=Path(str(predicate["terminal_status_receipt"])),
        kernel_output_receipt=Path(str(predicate["kernel_output_receipt"])),
        prelaunch_receipt=Path(str(predicate["prelaunch_receipt"])),
    )
    if regenerated != predicate:
        raise HiddenSubmitError("hidden submission predicate differs from fresh regeneration")
    return {
        "predicate_path": str(path.resolve()),
        "predicate_sha256": lifecycle.sha256_file(path.resolve()),
        "kernel_id": KERNEL_ID,
        "kernel_version": version,
        "output_sha256": predicate["output_sha256"],
        "package_manifest_sha256": predicate["package_manifest_sha256"],
        "terminal_validation_sha256": predicate["terminal_validation_sha256"],
        "kernel_output_receipt_sha256": predicate["kernel_output_receipt_sha256"],
    }


def run_submission(
    *, version: int, message: str, predicate: Path, receipt_dir: Path,
    kaggle: str,
) -> dict[str, Any]:
    spec = lifecycle.CodeSubmissionSpec(
        competition=COMPETITION,
        kernel_id=KERNEL_ID,
        kernel_version=version,
        output_file=OUTPUT_FILE,
        message=message,
    )
    return lifecycle.run_code_submission_once(
        spec=spec,
        predicate_receipt=predicate,
        predicate_validator=validate_predicate,
        receipt_dir=receipt_dir,
        receipt_schema="celltrack-alpha.public-temporalunet-hidden-competition-submit.v1",
        node="public_temporalunet_v32_competition_code_submit",
        kaggle=kaggle,
        lock_path=cmux_ops.DEFAULT_LOCK,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-predicate")
    build.add_argument("--version", type=int, required=True)
    build.add_argument("--package-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--terminal-validation", type=Path, required=True)
    build.add_argument("--terminal-status-receipt", type=Path, required=True)
    build.add_argument("--kernel-output-receipt", type=Path, required=True)
    build.add_argument("--prelaunch-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--version", type=int, required=True)
    submit.add_argument("--message", required=True)
    submit.add_argument("--predicate", type=Path, required=True)
    submit.add_argument("--receipt-dir", type=Path, required=True)
    submit.add_argument("--kaggle", default="kaggle")
    submit.add_argument("--print-plan", action="store_true")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "build-predicate":
        payload = predicate_payload(
            version=args.version,
            package_dir=args.package_dir,
            output_dir=args.output_dir,
            terminal_validation=args.terminal_validation,
            terminal_status_receipt=args.terminal_status_receipt,
            kernel_output_receipt=args.kernel_output_receipt,
            prelaunch_receipt=args.prelaunch_receipt,
        )
        baseline.atomic_json(args.output.resolve(), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["eligible"] else 2
    spec = lifecycle.CodeSubmissionSpec(
        competition=COMPETITION,
        kernel_id=KERNEL_ID,
        kernel_version=args.version,
        output_file=OUTPUT_FILE,
        message=args.message,
    )
    if args.print_plan:
        print(json.dumps(spec.submit_argv(args.kaggle)))
        return 0
    receipt = run_submission(
        version=args.version,
        message=args.message,
        predicate=args.predicate,
        receipt_dir=args.receipt_dir,
        kaggle=args.kaggle,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

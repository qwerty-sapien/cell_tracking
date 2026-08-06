#!/usr/bin/env python3
"""One-shot, exact-version Kaggle code submission for APNCC+G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cmux_ops
import g04_global_jump_hidden_guard as guard
import kaggle_submission_lifecycle as lifecycle


COMPETITION = guard.COMPETITION
KERNEL_ID = guard.KERNEL_ID
OUTPUT_FILE = "submission.csv"


class SubmitPredicateError(cmux_ops.PredicateNotSatisfied):
    """The downloaded output is not bound to an eligible exact version."""


def command_plan(kaggle: str, *, version: int, message: str) -> list[str]:
    return lifecycle.CodeSubmissionSpec(
        competition=COMPETITION,
        kernel_id=KERNEL_ID,
        kernel_version=version,
        output_file=OUTPUT_FILE,
        message=message,
    ).submit_argv(kaggle)


def validate_predicate(path: Path, *, version: int) -> dict[str, Any]:
    try:
        predicate = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SubmitPredicateError(f"code-submission predicate unreadable: {exc}") from exc
    if not isinstance(predicate, dict):
        raise SubmitPredicateError("code-submission predicate must be a JSON object")
    expected = {
        "schema": "celltrack-alpha.g04-global-jump-code-submission-predicate.v1",
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
        "revoked": False,
        "consumed": False,
        "output_sha256": guard.OUTPUT_SHA256,
        "output_bytes": guard.OUTPUT_BYTES,
        "output_data_rows": guard.OUTPUT_ROWS,
        "package_identity_sha256": guard.PACKAGE_IDENTITY_SHA256,
        "failures": [],
    }
    drift = [key for key, value in expected.items() if predicate.get(key) != value]
    if drift:
        raise SubmitPredicateError(f"predicate field drift: {drift}")
    try:
        prelaunch = Path(str(predicate["prelaunch_receipt"])).resolve()
        downloaded = Path(str(predicate["downloaded_output_dir"])).resolve()
        output_receipt = Path(str(predicate["kernel_output_receipt"])).resolve()
    except Exception as exc:
        raise SubmitPredicateError(f"predicate paths are invalid: {exc}") from exc
    if lifecycle.sha256_file(prelaunch) != predicate.get("prelaunch_receipt_sha256"):
        raise SubmitPredicateError("bound prelaunch receipt SHA drift")
    if lifecycle.sha256_file(output_receipt) != predicate.get("kernel_output_receipt_sha256"):
        raise SubmitPredicateError("bound exact-version output receipt SHA drift")
    with tempfile.TemporaryDirectory(prefix="g1-terminal-recheck-") as temporary:
        rechecked = guard.validate_terminal(
            downloaded,
            kernel_version=version,
            kernel_output_receipt=output_receipt,
            prelaunch_receipt=prelaunch,
            predicate_path=Path(temporary) / "predicate.json",
        )
    if rechecked.get("eligible") is not True or rechecked != predicate:
        raise SubmitPredicateError("terminal predicate no longer matches the authenticated output")
    return {
        "predicate_path": str(path.resolve()),
        "predicate_sha256": lifecycle.sha256_file(path.resolve()),
        "prelaunch_receipt": str(prelaunch),
        "prelaunch_receipt_sha256": lifecycle.sha256_file(prelaunch),
        "kernel_output_receipt": str(output_receipt),
        "kernel_output_receipt_sha256": lifecycle.sha256_file(output_receipt),
        "downloaded_output_dir": str(downloaded),
        "output_sha256": guard.OUTPUT_SHA256,
        "kernel_id": KERNEL_ID,
        "kernel_version": version,
    }


def run(
    *, version: int, message: str, predicate_receipt: Path, receipt_dir: Path,
    kaggle: str = "kaggle", lock_path: Path = cmux_ops.DEFAULT_LOCK,
    timeout: float = 300.0,
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
        predicate_receipt=predicate_receipt,
        predicate_validator=validate_predicate,
        receipt_dir=receipt_dir,
        receipt_schema="celltrack-alpha.g04-global-jump-competition-code-submit-receipt.v1",
        node="g04_global_jump_competition_code_submit",
        kaggle=kaggle,
        lock_path=lock_path,
        timeout=timeout,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--predicate-receipt", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.print_plan:
        print(json.dumps(command_plan(args.kaggle, version=args.version, message=args.message)))
        return 0
    receipt = run(
        version=args.version,
        message=args.message,
        predicate_receipt=args.predicate_receipt,
        receipt_dir=args.receipt_dir,
        kaggle=args.kaggle,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

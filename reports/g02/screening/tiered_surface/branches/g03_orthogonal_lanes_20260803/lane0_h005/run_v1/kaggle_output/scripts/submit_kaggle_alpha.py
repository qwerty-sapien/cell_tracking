#!/usr/bin/env python3
"""Guarded Kaggle submission wrapper for the alpha Biohub candidate.

This script deliberately separates local readiness failures from Kaggle-side
permission failures. It does not try to accept rules or reuse browser state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
DEFAULT_CSV = ROOT / "reports/kaggle_g01/kernel_output/submission.csv"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_GATE = ROOT / "reports/kaggle_g01/pre_submit_gate_report.json"
DEFAULT_VALIDATION = ROOT / "reports/kaggle_g01/t04_pre_submit_validation_report.json"
DEFAULT_ATTEMPT = ROOT / "reports/kaggle_g01/t04_submission_attempt.json"
DEFAULT_KERNEL = "onebitbrain/biohub-celltrack-alpha"
DEFAULT_KERNEL_VERSION = "4"
DEFAULT_KERNEL_OUTPUT = "submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--attempt-report", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument(
        "--mode",
        choices=("file", "kernel"),
        default="file",
        help="Use direct CSV upload or Kaggle code-submission output.",
    )
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--kernel-version", default=DEFAULT_KERNEL_VERSION)
    parser.add_argument("--kernel-output", default=DEFAULT_KERNEL_OUTPUT)
    parser.add_argument(
        "--kaggle-python",
        type=Path,
        default=ROOT / ".venv/bin/python",
        help="Python executable with the Kaggle CLI installed.",
    )
    parser.add_argument("--message", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": shlex.join(command),
            "exit_status": int(completed.returncode),
            "duration_seconds": round(time.time() - started, 3),
            "output": completed.stdout,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": shlex.join(command),
            "exit_status": None,
            "duration_seconds": round(time.time() - started, 3),
            "output": exc.stdout or "",
            "timed_out": True,
        }


def kaggle_command(kaggle_python: Path, args: list[str]) -> list[str]:
    return [str(kaggle_python), "-m", "kaggle", *args]


def classify_submit_result(result: dict[str, Any]) -> str:
    output = str(result.get("output") or "")
    exit_status = result.get("exit_status")
    if exit_status == 0:
        return "submitted"
    if "403 Client Error: Forbidden" in output:
        if "CreateCodeSubmission" in output:
            return "blocked_kaggle_code_submission_permission_403"
        if "CreateSubmission" in output:
            return "blocked_kaggle_file_submission_permission_403"
        return "blocked_kaggle_submission_permission_403"
    if "Please accept the competition rules" in output or "must accept" in output.lower():
        return "blocked_competition_rules_not_accepted"
    if "No such file" in output or "does not exist" in output:
        return "blocked_missing_submission_file"
    return "failed_unknown"


def validate_csv(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_remote_submission_csv.py"),
        "--csv",
        str(args.csv),
        "--zip",
        str(args.zip),
        "--output",
        str(args.validation_output),
    ]
    result = run_command(command, timeout=180)
    report: dict[str, Any] | None = None
    if args.validation_output.exists():
        report = read_json(args.validation_output)
    return {"command_result": result, "report": report}


def gate_status(args: argparse.Namespace, csv_sha256: str) -> dict[str, Any]:
    if not args.gate.exists():
        return {"valid": False, "reason": f"pre-submit gate does not exist: {args.gate}"}
    gate = read_json(args.gate)
    gate_sha = gate.get("candidate_csv_sha256")
    return {
        "valid": bool(gate.get("ready_for_t04_scored_submission")) and gate_sha == csv_sha256,
        "path": str(args.gate),
        "status": gate.get("status"),
        "ready_for_t04_scored_submission": gate.get("ready_for_t04_scored_submission"),
        "candidate_csv": gate.get("candidate_csv"),
        "candidate_csv_sha256": gate_sha,
        "sha256_matches_current_csv": gate_sha == csv_sha256,
    }


def default_message(args: argparse.Namespace, csv_sha256: str) -> str:
    if args.message:
        return args.message
    try:
        gate = read_json(args.gate)
        package_identity = str(gate.get("checks", {}).get("package_identity", "unknown"))
    except Exception:
        package_identity = "unknown"
    package_short = package_identity[:7] if package_identity != "unknown" else "unknown"
    return f"G-01 T-04 alpha {args.mode} package {package_short} validated CSV {csv_sha256[:8]}"


def submission_history(args: argparse.Namespace) -> dict[str, Any]:
    command = kaggle_command(
        args.kaggle_python,
        [
            "competitions",
            "submissions",
            "-c",
            args.competition,
            "--format",
            "json",
        ],
    )
    result = run_command(command, timeout=120)
    parsed: Any = None
    output = str(result.get("output") or "").strip()
    if output.startswith("[") or output.startswith("{"):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
    return {"command_result": result, "parsed": parsed}


def submit(args: argparse.Namespace, message: str) -> dict[str, Any]:
    if args.mode == "file":
        command = kaggle_command(
            args.kaggle_python,
            [
                "competitions",
                "submit",
                "-c",
                args.competition,
                "-f",
                str(args.csv),
                "-m",
                message,
            ],
        )
    else:
        command = kaggle_command(
            args.kaggle_python,
            [
                "competitions",
                "submit",
                "-c",
                args.competition,
                "-k",
                args.kernel,
                "-v",
                str(args.kernel_version),
                "-f",
                args.kernel_output,
                "-m",
                message,
            ],
        )
    result = run_command(command, timeout=600)
    result["classification"] = classify_submit_result(result)
    return result


def main() -> int:
    args = parse_args()
    args.csv = args.csv.resolve()
    args.zip = args.zip.resolve()
    args.gate = args.gate.resolve()
    args.validation_output = args.validation_output.resolve()
    args.attempt_report = args.attempt_report.resolve()
    # Keep the virtualenv symlink path intact. Resolving it to the underlying
    # framework Python drops the venv site-packages and hides the Kaggle CLI.
    if not args.kaggle_python.is_absolute():
        args.kaggle_python = ROOT / args.kaggle_python

    errors: list[str] = []
    if not args.csv.exists():
        errors.append(f"candidate CSV does not exist: {args.csv}")
        csv_sha = None
    else:
        csv_sha = sha256_file(args.csv)
    if not args.kaggle_python.exists():
        errors.append(f"Kaggle Python executable does not exist: {args.kaggle_python}")

    validation = validate_csv(args) if csv_sha is not None else None
    if validation is not None:
        report = validation.get("report") or {}
        if not report.get("ready_for_scored_submission"):
            errors.append("CSV validator did not mark the candidate ready for scored submission")

    gate = gate_status(args, csv_sha) if csv_sha is not None else None
    if gate is not None and not gate["valid"]:
        errors.append("pre-submit gate is not PASS for the current candidate CSV")

    message = default_message(args, csv_sha or "missing")
    history_before = submission_history(args) if not errors and not args.dry_run else None
    submit_result = None
    history_after = None
    if not errors and not args.dry_run:
        submit_result = submit(args, message)
        history_after = submission_history(args)

    status = "dry_run_pass" if args.dry_run and not errors else "blocked_preflight"
    if submit_result is not None:
        status = str(submit_result["classification"])
    scored_submission_created = submit_result is not None and submit_result.get("exit_status") == 0

    attempt = {
        "schema_version": 1,
        "task_id": "T-04",
        "created_at_unix": time.time(),
        "status": status,
        "competition": args.competition,
        "mode": args.mode,
        "candidate_csv": str(args.csv),
        "candidate_csv_sha256": csv_sha,
        "candidate_csv_bytes": args.csv.stat().st_size if args.csv.exists() else None,
        "validation": validation,
        "pre_submit_gate": gate,
        "message": message,
        "dry_run": bool(args.dry_run),
        "history_before": history_before,
        "submit_result": submit_result,
        "history_after": history_after,
        "scored_submission_created": scored_submission_created,
        "d04_scored_submissions_used_delta": 1 if scored_submission_created else 0,
        "errors": errors,
        "next_required_action": (
            "Accept/join the competition for submissions in the logged-in Kaggle web UI, "
            "or otherwise grant this Kaggle API account submit permission, then rerun this script."
            if status.startswith("blocked_kaggle") or status == "blocked_competition_rules_not_accepted"
            else None
        ),
    }
    args.attempt_report.parent.mkdir(parents=True, exist_ok=True)
    args.attempt_report.write_text(json.dumps(attempt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(attempt, indent=2, sort_keys=True))
    return 0 if scored_submission_created or status == "dry_run_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

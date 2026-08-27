#!/usr/bin/env python3
"""Upload and verify a CELLECT asset version from the owner's authenticated cmux shell.

This file contains no kernel command.  Its mutating mode must be invoked only
through computer-use cmux after the saved Kaggle credential has been refreshed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from verify_cellect_asset_roundtrip import RoundtripError, verify_download

DATASET_ID = "onebitbrain/g02-cellect-r7-published-assets-v1"
VERSION_NOTES = (
    "v6 consolidated CELLECT input: byte-exact v4 assets plus flattened frozen "
    "requests and population manifest; universal content-addressed step 0"
)
POLL_SECONDS = 15
READY_TIMEOUT_SECONDS = 30 * 60


class CmuxRoundtripError(RuntimeError):
    pass


def command_plan(kaggle: str, staging: Path, download_root: Path) -> dict[str, list[str]]:
    """Return the complete bounded external command plan."""
    return {
        "upload": [
            kaggle,
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            VERSION_NOTES,
            "-r",
            "skip",
        ],
        "status": [
            kaggle,
            "datasets",
            "status",
            DATASET_ID,
            "--format",
            "json",
        ],
        "files": [
            kaggle,
            "datasets",
            "files",
            DATASET_ID,
            "--format",
            "json",
            "--page-size",
            "200",
        ],
        "download": [
            "python:kaggle.KaggleApi.dataset_download_files",
            DATASET_ID,
            str(download_root),
        ],
    }


def _run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no CLI diagnostic"
        raise CmuxRoundtripError(f"{label} failed with exit {result.returncode}: {detail}")
    return result


def _write_text(path: Path, value: str) -> None:
    path.write_text(value if value.endswith("\n") else value + "\n")


def _load_json_output(result: subprocess.CompletedProcess[str], label: str) -> object:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CmuxRoundtripError(f"{label} did not emit one JSON document") from error


def _download_via_authenticated_sdk(
    download_root: Path, target_version: int
) -> None:
    """Bypass the CLI's metadata preflight while retaining cmux OAuth auth."""
    # Import only inside the owner cmux execution path. Importing kaggle at
    # module load can attempt OAuth refresh from an unauthorised agent shell.
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        f"{DATASET_ID}/{target_version}",
        path=str(download_root),
        force=True,
        quiet=False,
        unzip=True,
        licenses=[],
    )


def _finish_ready(
    work_root: Path,
    manifest_path: Path,
    plan: dict[str, list[str]],
    target_version: int,
) -> dict[str, object]:
    """List if available, download via the authenticated SDK, and verify."""
    files_result = subprocess.run(
        plan["files"],
        text=True,
        capture_output=True,
        check=False,
    )
    files_path: Path | None = None
    files_error_path: Path | None = None
    if files_result.returncode == 0:
        _load_json_output(files_result, "dataset file listing")
        files_path = work_root / "files.json"
        _write_text(files_path, files_result.stdout)
    else:
        diagnostic = files_result.stderr.strip() or files_result.stdout.strip()
        if "403 Client Error" not in diagnostic or "ListDatasetFiles" not in diagnostic:
            raise CmuxRoundtripError(
                "dataset file listing failed with an unexpected diagnostic: "
                f"{diagnostic or 'no CLI diagnostic'}"
            )
        files_error_path = work_root / "files.error.txt"
        _write_text(files_error_path, diagnostic)

    download_root = work_root / "download"
    download_root.mkdir()
    _download_via_authenticated_sdk(download_root, target_version)
    download_receipt = work_root / "download.json"
    download_receipt.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "version": target_version,
                "method": "authenticated_kaggle_sdk_cmux",
                "success": True,
                "completed_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return verify_download(
        download_root,
        manifest_path,
        "authenticated_kaggle_cli_cmux",
        work_root / "status.json",
        files_path,
        files_error_path,
        download_receipt,
    )


def _verify_local_staging(staging: Path, manifest_path: Path) -> dict[str, object]:
    return verify_download(staging, manifest_path, "local_staging")


def execute(
    staging: Path,
    manifest_path: Path,
    work_root: Path,
    kaggle: str,
    ready_timeout_seconds: int = READY_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Create one next dataset version, wait, download, and verify exact bytes."""
    if work_root.exists():
        raise CmuxRoundtripError(f"work root must be fresh: {work_root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"]["id"] != DATASET_ID:
        raise CmuxRoundtripError("manifest dataset id drift")
    version_before = int(manifest["dataset"]["version_before"])
    target_version = int(manifest["dataset"]["target_version"])
    if target_version != version_before + 1 or target_version < 4:
        raise CmuxRoundtripError(
            f"unexpected version transition: {version_before} -> {target_version}"
        )
    local_receipt = _verify_local_staging(staging, manifest_path)

    work_root.mkdir(parents=True)
    download_root = work_root / "download"
    plan = command_plan(kaggle, staging, download_root)
    (work_root / "command_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (work_root / "local_staging_receipt.json").write_text(
        json.dumps(local_receipt, indent=2, sort_keys=True) + "\n"
    )

    status_before = _run(plan["status"], "pre-upload dataset status")
    status_before_value = _load_json_output(status_before, "pre-upload dataset status")
    if not (
        isinstance(status_before_value, dict)
        and status_before_value.get("status") == "ready"
        and int(status_before_value.get("current_version_number", -1))
        == version_before
    ):
        raise CmuxRoundtripError(
            "refusing upload because the private dataset is not ready at "
            f"frozen version {version_before}: {status_before_value!r}"
        )
    _write_text(work_root / "status_before.json", status_before.stdout)

    upload = _run(plan["upload"], "dataset version upload")
    _write_text(work_root / "upload.stdout.txt", upload.stdout)
    _write_text(work_root / "upload.stderr.txt", upload.stderr)

    deadline = time.monotonic() + ready_timeout_seconds
    status_value: object | None = None
    status_result: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        status_result = _run(plan["status"], "dataset status")
        _write_text(work_root / "status_latest.json", status_result.stdout)
        status_value = _load_json_output(status_result, "dataset status")
        if (
            isinstance(status_value, dict)
            and status_value.get("status") == "ready"
            and int(status_value.get("current_version_number", -1)) == target_version
        ):
            break
        time.sleep(POLL_SECONDS)
    else:
        raise CmuxRoundtripError(
            f"dataset did not become ready at version {target_version} within "
            f"{ready_timeout_seconds} seconds; last status={status_value!r}"
        )
    assert status_result is not None
    _write_text(work_root / "status.json", status_result.stdout)

    roundtrip = _finish_ready(work_root, manifest_path, plan, target_version)
    roundtrip["cmux_orchestration"] = {
        "dataset_id": DATASET_ID,
        "completed_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "command_plan_path": str(work_root / "command_plan.json"),
        "kernel_command_present": False,
        "submission_command_present": False,
    }
    (work_root / "roundtrip.json").write_text(
        json.dumps(roundtrip, indent=2, sort_keys=True) + "\n"
    )
    return roundtrip


def resume_ready(
    staging: Path,
    manifest_path: Path,
    work_root: Path,
    kaggle: str,
) -> dict[str, object]:
    """Verify an already-created target version without another mutation."""
    if work_root.exists():
        raise CmuxRoundtripError(f"work root must be fresh: {work_root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"]["id"] != DATASET_ID:
        raise CmuxRoundtripError("manifest dataset id drift")
    target_version = int(manifest["dataset"]["target_version"])
    if target_version < 4:
        raise CmuxRoundtripError(f"unexpected target version: {target_version}")
    local_receipt = _verify_local_staging(staging, manifest_path)
    work_root.mkdir(parents=True)
    plan = command_plan(kaggle, staging, work_root / "download")
    (work_root / "command_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (work_root / "local_staging_receipt.json").write_text(
        json.dumps(local_receipt, indent=2, sort_keys=True) + "\n"
    )
    status = _run(plan["status"], "dataset status")
    status_value = _load_json_output(status, "dataset status")
    if not (
        isinstance(status_value, dict)
        and status_value.get("status") == "ready"
        and int(status_value.get("current_version_number", -1)) == target_version
    ):
        raise CmuxRoundtripError(
            f"private dataset is not ready at target version {target_version}: "
            f"{status_value!r}"
        )
    _write_text(work_root / "status.json", status.stdout)
    roundtrip = _finish_ready(work_root, manifest_path, plan, target_version)
    roundtrip["cmux_orchestration"] = {
        "dataset_id": DATASET_ID,
        "completed_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "command_plan_path": str(work_root / "command_plan.json"),
        "resume_existing_version": target_version,
        "dataset_version_mutation": False,
        "kernel_command_present": False,
        "submission_command_present": False,
    }
    (work_root / "roundtrip.json").write_text(
        json.dumps(roundtrip, indent=2, sort_keys=True) + "\n"
    )
    return roundtrip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Mutate the private dataset; invoke only from authenticated computer-use cmux.",
    )
    parser.add_argument(
        "--resume-v4",
        action="store_true",
        help="Legacy alias: verify the manifest target without creating a version.",
    )
    parser.add_argument(
        "--resume-ready",
        action="store_true",
        help="Verify an already-ready manifest target without creating a version.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise CmuxRoundtripError("kaggle CLI is not on PATH")
    local_receipt = _verify_local_staging(args.staging, args.manifest)
    plan = command_plan(kaggle, args.staging, args.work_root / "download")
    resume = args.resume_v4 or args.resume_ready
    if args.execute and resume:
        raise CmuxRoundtripError("--execute and --resume-ready are mutually exclusive")
    if not args.execute and not resume:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_EXTERNAL_ACTION",
                    "local_staging_status": local_receipt["status"],
                    "command_plan": plan,
                    "kernel_command_present": False,
                    "submission_command_present": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        if resume:
            result = resume_ready(
                args.staging,
                args.manifest,
                args.work_root,
                kaggle,
            )
        else:
            result = execute(args.staging, args.manifest, args.work_root, kaggle)
    except (OSError, ValueError, RoundtripError) as error:
        raise CmuxRoundtripError(str(error)) from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

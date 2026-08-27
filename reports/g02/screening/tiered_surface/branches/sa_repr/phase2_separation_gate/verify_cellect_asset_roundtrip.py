#!/usr/bin/env python3
"""Verify a downloaded CELLECT asset dataset and its checkpoint tar byte-for-byte."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from build_cellect_asset_v4 import sha256, verify_container


class RoundtripError(RuntimeError):
    pass


def _load_cli_receipt(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RoundtripError(f"cannot read {label} receipt: {path}") from error


def _verify_remote_cli_receipts(
    status_path: Path,
    files_path: Path | None,
    files_error_path: Path | None,
    download_path: Path,
    manifest: dict[str, object],
    observed_files: dict[str, int],
) -> dict[str, object]:
    status = _load_cli_receipt(status_path, "dataset status")
    target_version = int(manifest["dataset"]["target_version"])
    if (
        not isinstance(status, dict)
        or status.get("status") != "ready"
        or int(status.get("current_version_number", -1)) != target_version
    ):
        raise RoundtripError("cmux dataset-status receipt is not ready at target version")

    files_receipt_sha256 = None
    files_error_receipt_sha256 = None
    if files_path is not None:
        raw_files = _load_cli_receipt(files_path, "dataset files")
        if isinstance(raw_files, dict):
            raw_files = raw_files.get("files", raw_files.get("datasetFiles"))
        if not isinstance(raw_files, list):
            raise RoundtripError("cmux dataset-files receipt is not a file list")
        listed: dict[str, int] = {}
        for row in raw_files:
            if not isinstance(row, dict):
                raise RoundtripError("cmux dataset-files receipt contains a malformed row")
            name = row.get("name", row.get("fileName"))
            size = row.get("totalBytes", row.get("bytes", row.get("size")))
            if not isinstance(name, str) or not isinstance(size, int):
                raise RoundtripError("cmux dataset-files receipt lacks exact name/byte fields")
            listed[name] = size
        if listed != observed_files:
            raise RoundtripError(
                "cmux dataset-files receipt disagrees with authenticated download: "
                f"listed={listed} downloaded={observed_files}"
            )
        files_receipt_sha256 = sha256(files_path)
        files_listing_status = "LIST_DATASET_FILES_PASS"
    else:
        if files_error_path is None:
            raise RoundtripError("remote mode requires a files receipt or exact 403 receipt")
        try:
            diagnostic = files_error_path.read_text()
        except (OSError, UnicodeError) as error:
            raise RoundtripError("cannot read dataset-files failure receipt") from error
        if "403 Client Error" not in diagnostic or "ListDatasetFiles" not in diagnostic:
            raise RoundtripError("dataset-files failure receipt is not the observed endpoint 403")
        files_error_receipt_sha256 = sha256(files_error_path)
        files_listing_status = "LIST_DATASET_FILES_ENDPOINT_403"

    download = _load_cli_receipt(download_path, "authenticated download")
    if not (
        isinstance(download, dict)
        and download.get("dataset_id") == manifest["dataset"]["id"]
        and download.get("version") == target_version
        and download.get("method") == "authenticated_kaggle_sdk_cmux"
        and download.get("success") is True
    ):
        raise RoundtripError("authenticated cmux download receipt drift")
    return {
        "dataset_status_receipt_sha256": sha256(status_path),
        "dataset_files_receipt_sha256": files_receipt_sha256,
        "dataset_files_error_receipt_sha256": files_error_receipt_sha256,
        "dataset_files_status": files_listing_status,
        "dataset_download_receipt_sha256": sha256(download_path),
        "dataset_download_method": download["method"],
        "status": "ready",
        "version": target_version,
        "file_count": len(observed_files),
    }


def verify_download(
    download_root: Path,
    manifest_path: Path,
    mode: str,
    status_receipt: Path | None = None,
    files_receipt: Path | None = None,
    files_error_receipt: Path | None = None,
    download_receipt: Path | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256(manifest_path)
    if mode not in {"local_staging", "authenticated_kaggle_cli_cmux"}:
        raise RoundtripError(f"unsupported verification mode: {mode}")
    expected = {str(item["path"]): item for item in manifest["files"]}
    bundle_manifest_control = manifest.get("bundle_manifest")
    bundle_manifest_name = None
    if bundle_manifest_control is not None:
        if not isinstance(bundle_manifest_control, dict):
            raise RoundtripError("bundle manifest control is malformed")
        bundle_manifest_name = str(bundle_manifest_control["path"])
        downloaded_manifest = download_root / bundle_manifest_name
        if (
            not downloaded_manifest.is_file()
            or downloaded_manifest.stat().st_size != manifest_path.stat().st_size
            or sha256(downloaded_manifest) != manifest_sha256
        ):
            raise RoundtripError("downloaded consolidated bundle manifest drift")
    metadata = manifest["control_metadata"]
    metadata_path = download_root / str(metadata["path"])
    if metadata_path.exists():
        if (
            metadata_path.stat().st_size != int(metadata["bytes"])
            or sha256(metadata_path) != metadata["sha256"]
        ):
            raise RoundtripError("dataset-metadata.json drifted")
    elif mode == "local_staging":
        raise RoundtripError("local staging is missing dataset-metadata.json")
    observed_names = {
        str(path.relative_to(download_root))
        for path in download_root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    transport = manifest["checkpoint_transport"]
    container_name = str(transport["container_path"])
    expanded_name = str(Path(container_name).with_suffix("") / str(transport["member"]))
    packed = container_name in observed_names
    expected_names = set(expected)
    if bundle_manifest_name is not None:
        expected_names.add(bundle_manifest_name)
    materialization = "uncompressed_tar"
    if not packed:
        expected_names.remove(container_name)
        expected_names.add(expanded_name)
        materialization = "kaggle_expanded_uncompressed_tar"
    if observed_names != expected_names:
        raise RoundtripError(
            f"downloaded file set drift: expected={sorted(expected_names)} observed={sorted(observed_names)}"
        )

    files: list[dict[str, object]] = []
    for name, record in sorted(expected.items()):
        if name == container_name and not packed:
            continue
        path = download_root / name
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(record["bytes"]) or digest != record["sha256"]:
            raise RoundtripError(
                f"downloaded file mismatch: {name} bytes={size} sha256={digest}"
            )
        files.append({"path": name, "bytes": size, "sha256": digest})

    if bundle_manifest_name is not None:
        files.append(
            {
                "path": bundle_manifest_name,
                "bytes": manifest_path.stat().st_size,
                "sha256": manifest_sha256,
                "role": "bundle_manifest",
            }
        )

    if packed:
        container = download_root / container_name
        checkpoint = verify_container(container)
        if (
            checkpoint["container_bytes"] != transport["container_bytes"]
            or checkpoint["container_sha256"] != transport["container_sha256"]
            or checkpoint["member"] != transport["member"]
            or checkpoint["member_bytes"] != transport["member_bytes"]
            or checkpoint["member_sha256"] != transport["member_sha256"]
        ):
            raise RoundtripError("downloaded checkpoint container receipt drift")
    else:
        member_path = download_root / expanded_name
        member_bytes = member_path.stat().st_size
        member_sha256 = sha256(member_path)
        if (
            member_bytes != transport["member_bytes"]
            or member_sha256 != transport["member_sha256"]
        ):
            raise RoundtripError("Kaggle-expanded checkpoint member drift")
        checkpoint = {
            "materialization": materialization,
            "uploaded_container_path": container_name,
            "uploaded_container_bytes": transport["container_bytes"],
            "uploaded_container_sha256": transport["container_sha256"],
            "member": transport["member"],
            "materialized_member_path": expanded_name,
            "member_bytes": member_bytes,
            "member_sha256": member_sha256,
            "compression": "none",
        }
        files.append(
            {
                "path": expanded_name,
                "bytes": member_bytes,
                "sha256": member_sha256,
            }
        )

    cli_receipts = None
    if mode == "authenticated_kaggle_cli_cmux":
        if status_receipt is None or download_receipt is None:
            raise RoundtripError("remote mode requires cmux status and download receipts")
        observed_files = {
            str(item["path"]): int(item["bytes"])
            for item in files
        }
        cli_receipts = _verify_remote_cli_receipts(
            status_receipt,
            files_receipt,
            files_error_receipt,
            download_receipt,
            manifest,
            observed_files,
        )
    elif any(
        value is not None
        for value in (
            status_receipt,
            files_receipt,
            files_error_receipt,
            download_receipt,
        )
    ):
        raise RoundtripError("local mode must not claim authenticated CLI receipts")

    return {
        "schema": "celltrack-alpha.cellect-r7-kaggle-roundtrip.v"
        + str(int(manifest["dataset"]["target_version"])),
        "status": (
            "BYTE_EXACT_ROUNDTRIP_PASS"
            if mode == "authenticated_kaggle_cli_cmux"
            else "LOCAL_BYTE_EXACT_PASS"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verification_context": mode,
        "dataset": manifest["dataset"],
        "bundle_manifest_sha256": (
            manifest_sha256 if bundle_manifest_name is not None else None
        ),
        "authenticated_cli_receipts": cli_receipts,
        "download_root": str(download_root),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "checkpoint_transport": checkpoint,
        "kernel_launched": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("local_staging", "authenticated_kaggle_cli_cmux"),
    )
    parser.add_argument("--status-receipt", type=Path)
    parser.add_argument("--files-receipt", type=Path)
    parser.add_argument("--files-error-receipt", type=Path)
    parser.add_argument("--download-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_download(
        args.download_root,
        args.manifest,
        args.mode,
        args.status_receipt,
        args.files_receipt,
        args.files_error_receipt,
        args.download_receipt,
    )
    args.receipt_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

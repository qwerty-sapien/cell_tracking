#!/usr/bin/env python3
"""Download-only owner-cmux retrieval and content-addressed Trackastra authentication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile


DATASET = "onebitbrain/trackastra-6bba-assets-v1"
MODEL_SHA = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
ARCHIVE_SHA = "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
MODEL_BYTES = 109_906_058
ARCHIVE_BYTES = 101_957_234


class RetrievalError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_member_sha256(archive: Path, member: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def content_matches(
    inventory: list[dict[str, object]], expected: str
) -> list[dict[str, object]]:
    return [record for record in inventory if record["sha256"] == expected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an already-downloaded root after a local verifier failure; no API call.",
    )
    args = parser.parse_args()
    work_root = args.work_root.resolve()
    download_root = work_root / "download"
    if args.verify_existing:
        if not download_root.is_dir():
            raise RetrievalError(f"existing download root is absent: {download_root}")
    else:
        if work_root.exists():
            raise RetrievalError(f"work root must be fresh: {work_root}")
        download_root.mkdir(parents=True)

        # Import only in the owner's authenticated cmux shell. The agent shell
        # must never refresh or inspect the owner's Kaggle credential.
        from kaggle import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            DATASET,
            path=str(download_root),
            force=True,
            quiet=False,
            unzip=True,
            licenses=[],
        )

    inventory: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in download_root.rglob("*") if candidate.is_file()):
        inventory.append(
            {
                "relative_path": path.relative_to(download_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    model_matches = content_matches(inventory, MODEL_SHA)
    archive_matches = content_matches(inventory, ARCHIVE_SHA)
    model = model_matches[0] if model_matches else None
    archive = archive_matches[0] if archive_matches else None
    status = (
        "BOTH_PINNED_ASSETS_AUTHENTICATED"
        if model_matches and archive_matches
        else "HASH_MISMATCH_STOP"
    )
    archive_member: dict[str, object] | None = None
    resolved: dict[str, str] = {}

    if model and archive:
        if model["bytes"] != MODEL_BYTES or archive["bytes"] != ARCHIVE_BYTES:
            raise RetrievalError("content match has impossible byte-size drift")
        archive_path = download_root / str(archive["relative_path"])
        with zipfile.ZipFile(archive_path) as bundle:
            model_members = [
                name for name in bundle.namelist()
                if not name.endswith("/") and Path(name).name == "model.pt"
            ]
        if len(model_members) != 1:
            raise RetrievalError(
                f"exact archive contains {len(model_members)} basename model.pt members"
            )
        member_sha, member_bytes = zip_member_sha256(archive_path, model_members[0])
        archive_member = {
            "member": model_members[0],
            "bytes": member_bytes,
            "sha256": member_sha,
            "matches_separate_model": member_sha == MODEL_SHA and member_bytes == MODEL_BYTES,
        }
        if not archive_member["matches_separate_model"]:
            raise RetrievalError("exact archive member differs from exact separate model")
        resolved_root = work_root / "resolved"
        resolved_root.mkdir(exist_ok=True)
        model_target = resolved_root / "model.pt"
        archive_target = resolved_root / "ctc.zip"
        if not model_target.exists():
            os.link(download_root / str(model["relative_path"]), model_target)
        if not archive_target.exists():
            os.link(archive_path, archive_target)
        resolved = {
            "model_pt": str(model_target),
            "ctc_zip": str(archive_target),
        }

    receipt = {
        "schema": "g02.sa-reverse-authentic.owner-download.v1",
        "status": status,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": DATASET,
        "method": "authenticated_kaggle_sdk_cmux",
        "verification_mode": (
            "EXISTING_DOWNLOAD_NO_EXTERNAL_ACTION"
            if args.verify_existing
            else "DOWNLOAD_AND_VERIFY"
        ),
        "external_actions": {
            "dataset_download_files_this_invocation": 0 if args.verify_existing else 1,
            "dataset_download_files_workflow_total": 1,
            "list_dataset_files": 0,
            "dataset_upload_or_mutation": 0,
            "kernel_launch": 0,
            "submission": 0,
        },
        "resolution": "CONTENT_SHA256_NOT_FILENAME",
        "required": {
            "model": {"sha256": MODEL_SHA, "bytes": MODEL_BYTES},
            "archive": {"sha256": ARCHIVE_SHA, "bytes": ARCHIVE_BYTES},
        },
        "matches": {
            "model": model_matches,
            "archive": archive_matches,
            "archive_member": archive_member,
        },
        "resolved": resolved,
        "downloaded_file_count": len(inventory),
        "downloaded_total_bytes": sum(int(record["bytes"]) for record in inventory),
        "inventory": inventory,
    }
    (work_root / "DOWNLOAD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": status,
        "receipt": str(work_root / "DOWNLOAD_RECEIPT.json"),
        "resolved": resolved,
    }, sort_keys=True))
    return 0 if status == "BOTH_PINNED_ASSETS_AUTHENTICATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reconcile one exact terminal Kaggle GPU reservation under the shared lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cmux_ops


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument(
        "--terminal-status",
        required=True,
        choices=("COMPLETE", "ERROR", "CANCELLED"),
    )
    parser.add_argument("--terminal-receipt", required=True, type=Path)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kaggle", default=".venv/bin/kaggle")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reconciliation: {args.output}")

    lock = cmux_ops.serial_lock(
        node="terminal_gpu_reservation_reconcile",
        target=args.kernel_id,
    ).acquire()
    try:
        quota_receipt = cmux_ops.run(
            "quota",
            kaggle=args.kaggle,
            receipt_dir=args.receipt_dir,
            _held_lock=lock,
        )
        if quota_receipt["returncode"] != 0:
            raise RuntimeError("authoritative terminal quota query failed")
        quota = cmux_ops.parse_authoritative_gpu_quota(
            json.loads(quota_receipt["stdout_scrubbed"])
        )
        event = cmux_ops.reconcile_gpu_reservation(
            args.kernel_id,
            terminal_status=args.terminal_status,
            quota=quota,
            quota_receipt=quota_receipt["receipt_path"],
            terminal_receipt=str(args.terminal_receipt),
            _held_lock=lock,
        )
    finally:
        lock.release()

    payload = {
        "schema": "celltrack-alpha.kernel-gpu-terminal-reconciliation.v1",
        "kernel_id": args.kernel_id,
        "terminal_status": args.terminal_status,
        "terminal_receipt": str(args.terminal_receipt),
        "terminal_receipt_sha256": sha256_file(args.terminal_receipt),
        "quota": quota,
        "quota_receipt": quota_receipt["receipt_path"],
        "event": event,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

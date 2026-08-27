#!/usr/bin/env python3
"""Release the exact terminal CRW-HPO1000 administrative reservation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cmux_ops  # noqa: E402


OUTPUT = HERE / "TERMINAL_RESERVATION_RECONCILIATION.json"
STATUS = HERE / "cmux_receipts/kernel_status_2026-08-12T141844575062Z.json"
QUOTA = HERE / "cmux_receipts/quota_2026-08-12T141845681788Z.json"
KERNEL_ID = "onebitbrain/g02-crw-hpo1000-v1"


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("refusing to overwrite immutable terminal reconciliation")
    quota_receipt = json.loads(QUOTA.read_text(encoding="utf-8"))
    quota = json.loads(quota_receipt["stdout_scrubbed"])
    lock = cmux_ops.serial_lock(
        node="CRW_HPO1000:terminal_reconcile",
        target=KERNEL_ID,
    ).acquire()
    try:
        event = cmux_ops.reconcile_gpu_reservation(
            KERNEL_ID,
            terminal_status="COMPLETE",
            quota=quota,
            quota_receipt=str(QUOTA),
            terminal_receipt=str(STATUS),
            _held_lock=lock,
        )
    finally:
        lock.release()
    ledger = json.loads(cmux_ops.DEFAULT_GPU_LEDGER.read_text(encoding="utf-8"))
    payload = {
        "schema": "celltrack-alpha.crw-hpo1000-terminal-reservation-reconciliation.v1",
        "status": "COMPLETE",
        "event": event,
        "active_reservations_after": ledger["active_reservations"],
        "latest_authoritative_quota": ledger["latest_authoritative_quota"],
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

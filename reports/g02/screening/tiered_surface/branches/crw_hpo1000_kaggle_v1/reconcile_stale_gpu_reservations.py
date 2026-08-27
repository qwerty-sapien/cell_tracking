#!/usr/bin/env python3
"""Release three terminal OCC administrative reservations under one serial lock."""

from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cmux_ops  # noqa: E402


OUTPUT = HERE / "STALE_GPU_RESERVATION_RECONCILIATION.json"
QUOTA_RECEIPT = HERE / "cmux_receipts/quota_2026-08-12T074841613932Z.json"
ROWS = [
    (
        "onebitbrain/g02-occ-sliced-w1-v1",
        "ERROR",
        HERE / "cmux_receipts/kernel_status_2026-08-12T074919098424Z.json",
    ),
    (
        "onebitbrain/g02-occ-sliced-w1-repair1-v1",
        "ERROR",
        HERE / "cmux_receipts/kernel_status_2026-08-12T074924494483Z.json",
    ),
    (
        "onebitbrain/g02-occ-sliced-w1-repair2-v1",
        "COMPLETE",
        HERE / "cmux_receipts/kernel_status_2026-08-12T074929740387Z.json",
    ),
]


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("refusing to overwrite immutable reconciliation receipt")
    quota_receipt = json.loads(QUOTA_RECEIPT.read_text(encoding="utf-8"))
    quota = json.loads(quota_receipt["stdout_scrubbed"])
    lock = cmux_ops.serial_lock(
        node="CRW_HPO1000:reconcile_stale_occ_reservations",
        target="shared_gpu_ledger",
    ).acquire()
    try:
        events = [
            cmux_ops.reconcile_gpu_reservation(
                kernel_id,
                terminal_status=status,
                quota=quota,
                quota_receipt=str(QUOTA_RECEIPT),
                terminal_receipt=str(receipt),
                _held_lock=lock,
            )
            for kernel_id, status, receipt in ROWS
        ]
    finally:
        lock.release()
    ledger = json.loads(cmux_ops.DEFAULT_GPU_LEDGER.read_text(encoding="utf-8"))
    payload = {
        "schema": "celltrack-alpha.crw-hpo1000-stale-reservation-reconciliation.v1",
        "status": "COMPLETE",
        "events": events,
        "active_reservations_after": ledger["active_reservations"],
        "latest_authoritative_quota": ledger["latest_authoritative_quota"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

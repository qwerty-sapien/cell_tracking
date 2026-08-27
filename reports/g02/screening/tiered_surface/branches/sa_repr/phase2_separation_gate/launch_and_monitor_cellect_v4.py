#!/usr/bin/env python3
"""One-shot cmux-only CELLECT v4 kernel launch, monitor, and output retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
RUN = HERE / "run_v4"
KAGGLE = ROOT / ".venv313/bin/kaggle"
KERNEL_ID = "onebitbrain/g02-cellect-r7-separation-v1"
KERNEL = HERE / "kaggle_cellect_action_features.py"
METADATA = HERE / "kaggle_kernel_metadata.json"
CANONICAL_METADATA = HERE / "kernel-metadata.json"
AMENDMENT = HERE / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
SEALED_B0 = HERE / "run_v3/B0_SEALED.json"
ASSET_ROUNDTRIP = HERE / "CELLECT_R7_ASSET_V4_REMOTE_ROUNDTRIP.json"
BUDGET = ROOT / "reports/g02/screening/tiered_surface/kaggle_budget.json"

PINS = {
    KERNEL: "1c8a6961cb72fdba47b530279239a793eb491f41bd7c7310e99a81ac49283e58",
    METADATA: "4d214527404284217358760d81a146516a3489bdd711c39bc240c98acda67264",
    CANONICAL_METADATA: "4d214527404284217358760d81a146516a3489bdd711c39bc240c98acda67264",
    AMENDMENT: "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    SEALED_B0: "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    ASSET_ROUNDTRIP: "4aa001066a3168ebf109a1afafd3b4a794f512c9052d630c6580bfc8de52ecd0",
}
AUTHORIZATION_ID = "CELLECT-R7-SEPARATION-V4"
HARD_LIMIT_SECONDS = 5 * 3600
POLL_SECONDS = 60


class LaunchError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: object) -> None:
    if path.exists():
        raise LaunchError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value if value.endswith("\n") else value + "\n")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def preflight() -> dict[str, object]:
    if RUN.exists():
        raise LaunchError(f"fresh run directory required: {RUN}")
    observed = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise LaunchError(f"frozen input hash mismatch: {path}: {actual}")
        observed[str(path.relative_to(ROOT))] = actual
    budget = json.loads(BUDGET.read_text())
    active = [
        row
        for row in budget["active_authorizations"]
        if row["id"] == AUTHORIZATION_ID
        and row["status"] == "AUTHORIZED_NOT_YET_LAUNCHED"
    ]
    if len(active) != 1 or len(budget["active_authorizations"]) != 1:
        raise LaunchError("serial active authorization is absent or ambiguous")
    if float(budget["remaining_kernel_h"]) != 27.4239:
        raise LaunchError("prelaunch weekly budget drift")
    if float(active[0]["projection_must_be_strictly_less_than_h"]) != 5.0:
        raise LaunchError("projection ceiling drift")
    if float(active[0]["hard_kill_elapsed_h"]) != 5.0:
        raise LaunchError("hard-kill ceiling drift")
    roundtrip = json.loads(ASSET_ROUNDTRIP.read_text())
    if (
        roundtrip.get("status") != "BYTE_EXACT_ROUNDTRIP_PASS"
        or roundtrip["dataset"].get("target_version") != 4
        or not roundtrip["checkpoint_transport"].get("byte_exact")
    ):
        raise LaunchError("asset v4 round-trip is not authenticated")
    metadata = json.loads(METADATA.read_text())
    if (
        metadata.get("id") != KERNEL_ID
        or metadata.get("code_file") != KERNEL.name
        or not metadata.get("enable_gpu")
        or metadata.get("enable_internet")
        or metadata.get("competition_sources")
        != ["biohub-cell-tracking-during-development"]
        or metadata.get("dataset_sources")
        != [
            "onebitbrain/g02-cellect-r7-action-requests-v1",
            "onebitbrain/g02-cellect-r7-published-assets-v1",
        ]
    ):
        raise LaunchError("kernel metadata scope drift")
    return {
        "schema": "celltrack-alpha.cellect-v4-prelaunch.v1",
        "status": "PASS_SERIAL_AUTHORIZED",
        "checked_utc": now(),
        "authorization": active[0],
        "budget_used_kernel_h": budget["used_kernel_h"],
        "budget_remaining_kernel_h": budget["remaining_kernel_h"],
        "pins": observed,
        "kernel_id": KERNEL_ID,
        "scope": "exact 44b6 separation-population endpoint features only",
        "planned_authenticated_commands": [
            [str(KAGGLE), "kernels", "push", "-p", str(HERE)],
            [str(KAGGLE), "kernels", "status", KERNEL_ID],
            [str(KAGGLE), "kernels", "logs", KERNEL_ID],
            [
                str(KAGGLE),
                "kernels",
                "output",
                KERNEL_ID,
                "-p",
                str(RUN / "kaggle_output"),
                "-o",
            ],
        ],
        "maximum_remote_runs": 1,
        "prior_local_cli_rejections": 1,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "submission": False,
        "runpod": False,
    }


def status_text(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout + "\n" + result.stderr).strip()


def main() -> int:
    try:
        receipt = preflight()
    except Exception as error:
        print(f"PRELAUNCH_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    RUN.mkdir(parents=True)
    write_once(RUN / "PRELAUNCH.json", receipt)
    launched_at = time.monotonic()
    push = run([str(KAGGLE), "kernels", "push", "-p", str(HERE)])
    write_once(RUN / "push.stdout.txt", push.stdout)
    write_once(RUN / "push.stderr.txt", push.stderr)
    write_once(
        RUN / "PUSH.json",
        {
            "schema": "celltrack-alpha.cellect-v4-push.v1",
            "started_utc": receipt["checked_utc"],
            "completed_utc": now(),
            "returncode": push.returncode,
            "remote_run_created": push.returncode == 0,
            "success": push.returncode == 0,
        },
    )
    if push.returncode != 0:
        write_once(
            RUN / "TERMINAL.json",
            {
                "schema": "celltrack-alpha.cellect-v4-terminal.v1",
                "status": "INVALID_PUSH_FAILED_NO_RETRY",
                "terminal_utc": now(),
                "elapsed_seconds": time.monotonic() - launched_at,
                "remote_run_created": false,
                "output_downloaded": False,
            },
        )
        return 3

    polls: list[dict[str, object]] = []
    terminal_status = None
    while time.monotonic() - launched_at < HARD_LIMIT_SECONDS:
        status = run([str(KAGGLE), "kernels", "status", KERNEL_ID])
        text = status_text(status)
        polls.append(
            {
                "observed_utc": now(),
                "elapsed_seconds": time.monotonic() - launched_at,
                "returncode": status.returncode,
                "text": text,
            }
        )
        write_once(
            RUN / f"status_{len(polls):04d}.json",
            polls[-1],
        )
        lowered = text.lower()
        if status.returncode == 0 and "complete" in lowered:
            terminal_status = "COMPLETE"
            break
        if any(word in lowered for word in ("error", "failed", "cancel")):
            terminal_status = "ERROR"
            break
        time.sleep(POLL_SECONDS)

    elapsed = time.monotonic() - launched_at
    if terminal_status is None:
        terminal_status = "HARD_LIMIT_REACHED"
    logs = run([str(KAGGLE), "kernels", "logs", KERNEL_ID])
    write_once(RUN / "kernel.log", logs.stdout + ("\n" + logs.stderr if logs.stderr else ""))
    output = None
    if terminal_status == "COMPLETE":
        output_root = RUN / "kaggle_output"
        output = run(
            [
                str(KAGGLE),
                "kernels",
                "output",
                KERNEL_ID,
                "-p",
                str(output_root),
                "-o",
            ]
        )
        write_once(RUN / "output.stdout.txt", output.stdout)
        write_once(RUN / "output.stderr.txt", output.stderr)
    terminal = {
        "schema": "celltrack-alpha.cellect-v4-terminal.v1",
        "status": terminal_status,
        "terminal_utc": now(),
        "elapsed_seconds": elapsed,
        "kernel_h_raw": elapsed / 3600.0,
        "remote_run_created": true,
        "poll_count": len(polls),
        "latest_status": polls[-1] if polls else None,
        "logs_returncode": logs.returncode,
        "output_downloaded": output is not None and output.returncode == 0,
        "output_returncode": output.returncode if output is not None else None,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "kernel_self_soft_stop_seconds": HARD_LIMIT_SECONDS - 300,
        "full_pool_cache": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
    }
    write_once(RUN / "TERMINAL.json", terminal)
    return 0 if terminal["status"] == "COMPLETE" and terminal["output_downloaded"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

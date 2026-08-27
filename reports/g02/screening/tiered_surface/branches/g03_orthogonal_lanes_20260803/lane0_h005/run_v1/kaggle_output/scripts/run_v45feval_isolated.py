#!/usr/bin/env python3
"""Run a full V45 held-out evaluation as resumable single-video processes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


class IsolatedEvalError(ValueError):
    """Raised when an isolated evaluation drifts from its dry-run contract."""


def _load(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IsolatedEvalError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise IsolatedEvalError(f"JSON artifact is not an object: {path}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IsolatedEvalError(f"{label} is not an object")
    return value


def _validate_single_report(
    path: Path, identity: str, expected_edge_sha256: str
) -> None:
    report = _load(path)
    scope = _mapping(report.get("scope"), label=f"{path} scope")
    execution = _mapping(report.get("execution"), label=f"{path} execution")
    edge = _mapping(execution.get("edge_artifact"), label=f"{path} edge artifact")
    results = report.get("results")
    if (
        report.get("status") != "completed"
        or scope.get("selected_video_ids") != [identity]
        or scope.get("held_out_video_count") != 58
        or any(scope.get(key) is not False for key in ("fit_used", "audit_used", "test_used"))
        or execution.get("division_mode") != "inherited"
        or execution.get("effective_division_enabled") is not True
        or edge.get("sha256") != expected_edge_sha256
        or not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], Mapping)
        or results[0].get("identity") != identity
    ):
        raise IsolatedEvalError(f"single-video report drifted from contract: {path}")


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dry-report", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--assembled-out-dir", type=Path, required=True)
    parser.add_argument("--edge-artifact", type=Path, required=True)
    parser.add_argument("--edge-artifact-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    dry = _load(args.dry_report)
    scope = _mapping(dry.get("scope"), label="dry-run scope")
    execution = _mapping(dry.get("execution"), label="dry-run execution")
    edge = _mapping(execution.get("edge_artifact"), label="dry-run edge artifact")
    identities = scope.get("selected_video_ids")
    if (
        dry.get("status") != "dry_run"
        or not isinstance(identities, list)
        or len(identities) != 58
        or len(set(identities)) != 58
        or scope.get("held_out_video_count") != 58
        or any(scope.get(key) is not False for key in ("fit_used", "audit_used", "test_used"))
        or execution.get("division_mode") != "inherited"
        or execution.get("effective_division_enabled") is not True
        or edge.get("sha256") != args.edge_artifact_sha256
    ):
        raise IsolatedEvalError("dry-run report is not the exact full held-out contract")
    if args.assembled_out_dir.exists():
        raise IsolatedEvalError("assembled-out-dir must not already exist")
    args.out_root.mkdir(parents=True, exist_ok=True)
    empty = args.out_root / "_preserved_empty"
    empty.mkdir(exist_ok=True)

    total = len(identities)
    for index, raw_identity in enumerate(identities, start=1):
        identity = str(raw_identity)
        dataset = identity.split(":", 1)[1]
        destination = args.out_root / identity.replace(":", "__", 1)
        report_path = destination / "report.json"
        if report_path.is_file():
            _validate_single_report(report_path, identity, args.edge_artifact_sha256)
            print(f"[{index}/{total}] validated existing {identity}", flush=True)
            continue
        if destination.exists():
            raise IsolatedEvalError(f"incomplete single-video directory requires diagnosis: {destination}")
        command = [
            sys.executable,
            str(ROOT / "scripts/v45feval.py"),
            "--input",
            str(args.input),
            "--out-dir",
            str(destination),
            "--fold",
            "0",
            "--datasets",
            dataset,
            "--device",
            args.device,
            "--assoc",
            "sparse",
            "--topology",
            "legacy",
            "--division-mode",
            "inherited",
            "--edge-artifact",
            str(args.edge_artifact),
            "--edge-artifact-sha256",
            args.edge_artifact_sha256,
        ]
        print(f"[{index}/{total}] starting {identity}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise IsolatedEvalError(
                f"single-video evaluator failed with exit {completed.returncode}: {identity}"
            )
        _validate_single_report(report_path, identity, args.edge_artifact_sha256)
        print(f"[{index}/{total}] completed {identity}", flush=True)

    assemble = [
        sys.executable,
        str(ROOT / "scripts/recover_v45feval.py"),
        "--input",
        str(args.input),
        "--split",
        str(args.split),
        "--registry",
        str(args.registry),
        "--dry-report",
        str(args.dry_report),
        "--interrupted-dir",
        str(empty),
        "--tail-root",
        str(args.out_root),
        "--out-dir",
        str(args.assembled_out_dir),
        "--expected-edge-sha256",
        args.edge_artifact_sha256,
    ]
    completed = subprocess.run(assemble, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise IsolatedEvalError(f"final assembly failed with exit {completed.returncode}")
    print(f"assembled {total} videos at {args.assembled_out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

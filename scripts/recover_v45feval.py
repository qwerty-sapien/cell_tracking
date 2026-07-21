#!/usr/bin/env python3
"""Assemble an attested V45 evaluation after an interrupted fragment run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_bh
from v45feval import _sha_file


SCHEMA = "celltrack-alpha.v45-fast-eval-recovery.v1"


class RecoveryError(ValueError):
    """Raised when interrupted and recovery artifacts cannot be combined safely."""


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise RecoveryError(f"JSON artifact is not an object: {path}")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} is not an object")
    return value


def _identity_from_fragment(path: Path) -> str:
    if path.suffix != ".csv" or "__" not in path.stem:
        raise RecoveryError(f"invalid fragment filename: {path}")
    return path.stem.replace("__", ":", 1)


def _fragment_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("*.csv")):
        identity = _identity_from_fragment(path)
        if identity in result:
            raise RecoveryError(f"duplicate fragment identity: {identity}")
        result[identity] = path
    return result


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dry-report", type=Path, required=True)
    parser.add_argument("--interrupted-dir", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-edge-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if args.out_dir.exists():
        raise RecoveryError("out-dir must not already exist")

    dry = _load_object(args.dry_report)
    if dry.get("status") != "dry_run":
        raise RecoveryError("contract report is not a completed dry run")
    scope = _mapping(dry.get("scope"), label="scope")
    execution = _mapping(dry.get("execution"), label="execution")
    checkpoint = _mapping(dry.get("checkpoint"), label="checkpoint")
    selected_value = scope.get("selected_video_ids")
    if (
        not isinstance(selected_value, list)
        or len(selected_value) != 58
        or len(set(selected_value)) != 58
        or scope.get("held_out_video_count") != 58
    ):
        raise RecoveryError("contract does not select exactly 58 unique held-out videos")
    selected = tuple(str(item) for item in selected_value)
    if any(scope.get(key) is not False for key in ("fit_used", "audit_used", "test_used")):
        raise RecoveryError("contract reports fit, audit, or test exposure")
    edge = _mapping(execution.get("edge_artifact"), label="execution.edge_artifact")
    if (
        execution.get("division_mode") != "inherited"
        or execution.get("effective_division_enabled") is not True
        or edge.get("sha256") != args.expected_edge_sha256
    ):
        raise RecoveryError("division or frozen edge-artifact contract drifted")
    edge_path = Path(str(edge.get("path", "")))
    if not edge_path.is_absolute():
        edge_path = ROOT / edge_path
    if not edge_path.is_file() or _sha_file(edge_path) != args.expected_edge_sha256:
        raise RecoveryError("runtime edge artifact does not match its contract hash")
    edge_identity = _mapping(edge.get("identity"), label="execution.edge_artifact.identity")
    exposure = _mapping(edge_identity.get("exposure"), label="edge exposure")
    exposed_ids = exposure.get("source_video_ids")
    if (
        exposure.get("audit_exposed") is not False
        or exposure.get("test_exposed") is not False
        or not isinstance(exposed_ids, list)
        or set(selected) & set(exposed_ids)
    ):
        raise RecoveryError("edge artifact exposure is not disjoint fit-only data")
    fit_sources = checkpoint.get("fit_sources")
    if not isinstance(fit_sources, list) or any(
        not isinstance(row, Mapping) or row.get("video_id") in selected for row in fit_sources
    ):
        raise RecoveryError("checkpoint fit sources overlap the held-out selection")

    interrupted = _fragment_map(args.interrupted_dir / "frags")
    if not args.interrupted_dir.is_dir() or not set(interrupted).issubset(selected):
        raise RecoveryError("interrupted fragment directory is absent or outside the held-out contract")

    tail_reports: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for report_path in sorted(args.tail_root.glob("*/report.json")):
        report = _load_object(report_path)
        tail_scope = _mapping(report.get("scope"), label=f"{report_path} scope")
        identities = tail_scope.get("selected_video_ids")
        if not isinstance(identities, list) or len(identities) != 1:
            raise RecoveryError(f"tail report is not single-video: {report_path}")
        identity = str(identities[0])
        if identity in tail_reports:
            raise RecoveryError(f"duplicate tail report identity: {identity}")
        tail_reports[identity] = (report_path, report)

    missing = set(selected) - set(interrupted)
    if set(tail_reports) != missing:
        raise RecoveryError("tail reports do not exactly cover interrupted-run omissions")

    fragments: dict[str, tuple[Path, str, Mapping[str, Any] | None]] = {
        identity: (path, "interrupted-run-preserved", None)
        for identity, path in interrupted.items()
    }
    for identity, (report_path, report) in tail_reports.items():
        tail_scope = _mapping(report.get("scope"), label=f"{report_path} scope")
        tail_execution = _mapping(report.get("execution"), label=f"{report_path} execution")
        if (
            report.get("status") != "completed"
            or tail_scope.get("held_out_video_count") != 58
            or any(tail_scope.get(key) is not False for key in ("fit_used", "audit_used", "test_used"))
            or tail_execution.get("division_mode") != "inherited"
            or tail_execution.get("effective_division_enabled") is not True
            or _mapping(tail_execution.get("edge_artifact"), label="tail edge artifact").get("sha256")
            != args.expected_edge_sha256
        ):
            raise RecoveryError(f"tail report contract drifted: {report_path}")
        results = report.get("results")
        if not isinstance(results, list) or len(results) != 1 or results[0].get("identity") != identity:
            raise RecoveryError(f"tail report result identity drifted: {report_path}")
        result = _mapping(results[0], label=f"{report_path} result")
        graph = _mapping(result.get("graph"), label=f"{report_path} graph")
        if graph.get("division_mode") != "inherited" or graph.get("effective_division_enabled") is not True:
            raise RecoveryError(f"tail result division contract drifted: {report_path}")
        fragment = report_path.parent / "frags" / f"{identity.replace(':', '__', 1)}.csv"
        if not fragment.is_file() or _sha_file(fragment) != result.get("fragment_sha256"):
            raise RecoveryError(f"tail fragment hash drifted: {fragment}")
        fragments[identity] = (
            fragment,
            "isolated-recovery" if interrupted else "isolated-evaluation",
            result,
        )

    if set(fragments) != set(selected):
        raise RecoveryError("combined fragment set does not exactly cover the held-out contract")
    records = {str(item.video_id): item for item in run_bh.load_registry(args.registry)}
    if not set(selected).issubset(records):
        raise RecoveryError("registry does not cover the held-out contract")

    validated: list[dict[str, Any]] = []
    for identity in selected:
        fragment, source, result = fragments[identity]
        rows = run_bh.read_rows(fragment)
        counts = run_bh.validate_submission_rows(rows, [records[identity]])
        validated.append(
            {
                "identity": identity,
                "dataset": identity.split(":", 1)[1],
                "source": source,
                "source_fragment": str(fragment),
                "fragment_sha256": _sha_file(fragment),
                "rows": int(counts["rows"]),
                "nodes": int(counts["nodes"]),
                "edges": int(counts["edges"]),
                "seconds": None if result is None else float(result["seconds"]),
            }
        )

    output_frags = args.out_dir / "frags"
    output_frags.mkdir(parents=True, exist_ok=False)
    for item in validated:
        source = Path(str(item["source_fragment"]))
        target = output_frags / f"{str(item['identity']).replace(':', '__', 1)}.csv"
        os.link(source, target)
        if _sha_file(target) != item["fragment_sha256"]:
            raise RecoveryError(f"linked output fragment hash drifted: {target}")

    score_command = [
        sys.executable,
        str(ROOT / "scripts/scoreall.py"),
        "--zip",
        str(args.input),
        "--split",
        str(args.split),
        "--subset",
        "fold0",
        "--frags",
        str(output_frags),
        "--output",
        str(args.out_dir / "scores.json"),
    ]
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": dict(scope),
        "checkpoint": dict(checkpoint),
        "execution": dict(execution),
        "recovery": {
            "classification": (
                "ungraceful-external-termination-no-python-exception"
                if interrupted
                else "isolated-resumable-batch"
            ),
            "interrupted_dir": str(args.interrupted_dir),
            "contract_report": str(args.dry_report),
            "tail_root": str(args.tail_root),
            "preserved_fragment_count": len(interrupted),
            "isolated_recovery_count": len(tail_reports),
            "original_report_present": (args.interrupted_dir / "report.json").is_file(),
            "fragments_recomputed": False,
        },
        "results": validated,
        "summary": {
            "videos": len(validated),
            "rows": sum(int(item["rows"]) for item in validated),
            "nodes": sum(int(item["nodes"]) for item in validated),
            "edges": sum(int(item["edges"]) for item in validated),
            "full_fold": len(validated) == int(scope["held_out_video_count"]),
            "known_recovery_seconds": sum(float(item["seconds"] or 0.0) for item in validated),
            "score_command": score_command,
        },
    }
    run_bh.atomic_create_json(args.out_dir / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

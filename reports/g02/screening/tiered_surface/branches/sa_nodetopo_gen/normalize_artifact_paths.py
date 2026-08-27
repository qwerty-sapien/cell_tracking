#!/usr/bin/env python3
"""Normalize banked JSON paths to repository-relative merge-stable paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/Users/agent/Desktop/cell_tracking").resolve()
WORKTREE = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relativeize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: relativeize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relativeize(item) for item in value]
    if isinstance(value, str):
        for prefix in (str(WORKTREE) + "/", str(ROOT) + "/"):
            if value.startswith(prefix):
                return value[len(prefix) :]
    return value


def rewrite(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = relativeize(value)
    path.write_bytes(canonical_bytes(value))
    return value


def main() -> int:
    manifest_path = GENERATED / "raw_universe" / "manifest.json"
    audit_checkpoint_path = GENERATED / "COVERAGE_IDENTITY_AUDIT.checkpoint.json"
    audit_path = GENERATED / "COVERAGE_IDENTITY_AUDIT.json"
    population_checkpoint_path = GENERATED / "ACTION_POPULATION.checkpoint.json"
    curve_path = GENERATED / "FALSE_NODE_STRESS_CURVE.json"
    result_path = GENERATED / "RESULT.json"

    rewrite(manifest_path)
    rewrite(audit_checkpoint_path)
    rewrite(audit_path)
    rewrite(population_checkpoint_path)
    rewrite(curve_path)
    result = rewrite(result_path)
    result["raw_universe"]["manifest_sha256"] = sha256_file(manifest_path)
    result["coverage_identity_audit"]["sha256"] = sha256_file(audit_path)
    result["action_population"]["sha256"] = sha256_file(curve_path)
    result_path.write_bytes(canonical_bytes(result))

    initial_preflight = HERE / "idle_preflight.json"
    resume_preflight = HERE / "idle_preflight_resume.json"
    rewrite(initial_preflight)
    rewrite(resume_preflight)
    for guard_path, preflight_path in (
        (HERE / "resource_guard.json", initial_preflight),
        (HERE / "resource_guard_resume.json", resume_preflight),
    ):
        guard = rewrite(guard_path)
        guard["preflight_sha256"] = sha256_file(preflight_path)
        guard_path.write_bytes(canonical_bytes(guard))
    print(
        json.dumps(
            {
                "status": "PASS",
                "result_sha256": sha256_file(result_path),
                "manifest_sha256": sha256_file(manifest_path),
                "audit_sha256": sha256_file(audit_path),
                "curve_sha256": sha256_file(curve_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

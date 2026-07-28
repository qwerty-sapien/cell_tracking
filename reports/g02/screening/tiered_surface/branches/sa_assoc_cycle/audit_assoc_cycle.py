#!/usr/bin/env python3
"""Authenticate whether the existing cache can support genuine cycle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "reports/g02/trackastra_h1_feature_cache/report.json":
        "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/PREREGISTRATION_v3_per_class_authenticated_weight.json":
        "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json":
        "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v3/b0_sealed.json":
        "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    "reports/g02/screening/tiered_surface/branches/association_expansion.md":
        "37dd4b9a31b5b8cf420dc6744c97d5165bc29faa83cd3d7990ebdafffbc6f5be",
}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit(root: Path) -> dict[str, Any]:
    inputs = []
    for relative, expected in EXPECTED.items():
        path = root / relative
        if not path.is_file():
            raise AuditError(f"missing frozen input: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise AuditError(
                f"frozen input hash mismatch: {relative}: {observed}"
            )
        inputs.append({"path": relative, "sha256": observed, "pass": True})

    report = load_json(root / next(iter(EXPECTED)))
    files = report.get("files", [])
    held = [row for row in files if row.get("identity", "").startswith("train:44b6_")]
    if len(held) != 36:
        raise AuditError(f"expected 36 fit-only 44b6 videos, found {len(held)}")

    checked = []
    total_bytes = 0
    for row in sorted(held, key=lambda item: item["identity"]):
        relative = row["feature_path"]
        path = root / relative
        observed = sha256(path)
        if observed != row["feature_sha256"]:
            raise AuditError(f"feature hash mismatch: {relative}")
        size = path.stat().st_size
        if size != row["bytes"]:
            raise AuditError(f"feature byte-count mismatch: {relative}")
        total_bytes += size
        checked.append({
            "identity": row["identity"],
            "path": relative,
            "sha256": observed,
            "bytes": size,
        })

    contract = report.get("feature_contract", {})
    forward_count = int(contract.get("trackastra_feature_count", -1))
    output_count = int(contract.get("output_feature_count", -1))
    checkpoint_text = str(contract.get("checkpoint_features", "")).lower()
    reverse_fields = {
        key: value for key, value in contract.items()
        if "reverse" in key.lower() or "endpoint_return" in key.lower()
    }
    genuine_reverse = (
        forward_count >= 2
        and bool(reverse_fields)
        and "reverse" in checkpoint_text
    )

    return {
        "schema": "g02.sa-assoc-cycle.capability-audit.v1",
        "status": (
            "READY_FOR_SEPARATION"
            if genuine_reverse
            else "PARK_NOT_READY_MISSING_AUTHENTIC_REVERSE"
        ),
        "frozen_inputs": inputs,
        "existing_cache": {
            "fit_only_44b6_video_count": len(checked),
            "feature_files_authenticated": len(checked),
            "feature_bytes_authenticated": total_bytes,
            "aggregate_report_video_count": report["aggregate"]["video_count"],
            "aggregate_report_proposal_count": report["aggregate"]["proposal_count"],
            "output_feature_count": output_count,
            "trackastra_feature_count": forward_count,
            "declared_reverse_or_endpoint_return_fields": reverse_fields,
            "files": checked,
        },
        "capability_checks": {
            "forward_cache_authenticated": True,
            "reverse_inference_artifact_present": genuine_reverse,
            "endpoint_return_contract_present": bool(reverse_fields),
            "synthetic_reverse_forbidden": True,
            "separation_population_and_B0_authenticated": True,
        },
        "separation": {
            "executed": False,
            "reason": (
                "The existing cache declares exactly one Trackastra margin and "
                "no reverse-inference or endpoint-return contract. A reverse "
                "score cannot be synthesized from the forward margin."
            ),
            "B0_comparison": None,
            "graph_run": False,
        },
        "tier_proposal": None,
        "resume_condition": (
            "Freeze an immutable authentic reverse-inference cache and exact "
            "endpoint-return/no-link contract, then run the unchanged separation gate."
        ),
        "forbidden_actions": {
            "gpu": False,
            "kaggle": False,
            "budget_write": False,
            "cellect": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "submission": False,
            "runpod": False,
            "dead": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise AuditError(f"refusing to overwrite {args.output}")
    result = audit(args.root.resolve())
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

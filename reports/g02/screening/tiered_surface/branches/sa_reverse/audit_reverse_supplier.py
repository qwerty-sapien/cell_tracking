#!/usr/bin/env python3
"""Authenticate local inputs for a genuine reversed-time Trackastra supplier.

This capability audit never loads a model and never performs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT_INPUTS = {
    "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json":
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "src/celltrack_alpha/action_compiler.py":
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json":
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    "reports/g02/trackastra_h1_feature_cache/report.json":
        "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf",
    "reports/g02/trackastra_pretrained_asset_audit_v1.json":
        "2f809076fa570388d114f2f2bfe0a9be7dd5186cda41d67b3992127f10fa0132",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/PREREGISTRATION_v3_per_class_authenticated_weight.json":
        "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json":
        "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v3/b0_sealed.json":
        "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
}

EXTERNAL_INPUTS = {
    "cache_dataset_manifest.json":
        "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28",
}

SOURCE_FILES = {
    "LICENSE": "cc9ea92a72a63101ff425cc525faeace28ec9fc9c658a0fc01daf236e06326c9",
    "trackastra/model/model.py":
        "53e47b931338ed3450f4f23d6391ad1985f7083fd2754eb4edd6d3b5cd8b964c",
    "trackastra/model/model_parts.py":
        "b218eee5e5acb794b56ce1b36b991a390097ea6fd57960f9992240b01f891595",
    "trackastra/model/rope.py":
        "761a850aee82eaac25e8ab5d772748bcee130a5cd10b3d5a31aa339709ce5290",
    "trackastra/utils/__init__.py":
        "9cc30299112973185878e719aee11459913ac2da7d6b1765bba0a7f2e030189f",
    "trackastra/utils/utils.py":
        "6de65b3996dd1fa006e90de826a3d16e7b1796fa8ddbbff8032b178342a07bff",
}

CHECKPOINT_MEMBER_SHA256 = (
    "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
)
CHECKPOINT_ARCHIVE_SHA256 = (
    "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
)


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def receipt(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing frozen input: {path}")
    observed = sha256(path)
    if observed != expected:
        raise AuditError(f"hash mismatch: {path}: {observed}")
    return {
        "path": str(path),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "pass": True,
    }


def endpoint_return(
    reverse_matrix: list[list[float]],
    index_by_node: dict[int, int],
    source: int,
    target: int,
) -> float:
    """Return a separately inferred reverse score to canonical (source,target)."""
    if source not in index_by_node or target not in index_by_node:
        raise AuditError("endpoint-return identity is missing")
    return float(reverse_matrix[index_by_node[target]][index_by_node[source]])


def audit(
    root: Path,
    fit_cache: Path,
    source_root: Path,
    checkpoint_candidates: list[Path],
) -> dict[str, Any]:
    frozen = [
        receipt(root / relative, expected)
        for relative, expected in ROOT_INPUTS.items()
    ]
    manifest_receipt = receipt(
        fit_cache / "cache_dataset_manifest.json",
        EXTERNAL_INPUTS["cache_dataset_manifest.json"],
    )
    source = [
        receipt(source_root / relative, expected)
        for relative, expected in SOURCE_FILES.items()
    ]

    forward_report = load_json(
        root / "reports/g02/trackastra_h1_feature_cache/report.json"
    )
    forward_rows = sorted(
        (
            row for row in forward_report["files"]
            if row["identity"].startswith("train:44b6_")
        ),
        key=lambda row: row["identity"],
    )
    if len(forward_rows) != 36:
        raise AuditError(f"expected 36 forward 44b6 rows, found {len(forward_rows)}")
    forward_files = []
    for row in forward_rows:
        relative = row["feature_path"]
        path = root / relative
        observed = receipt(path, row["feature_sha256"])
        if observed["bytes"] != row["bytes"]:
            raise AuditError(f"forward byte count mismatch: {relative}")
        forward_files.append({
            **observed,
            "identity": row["identity"],
            "inference_calls": int(row["inference_calls"]),
            "proposal_count": int(row["proposal_count"]),
            "row_count": int(row["row_count"]),
        })

    cache_manifest = load_json(fit_cache / "cache_dataset_manifest.json")
    cache_rows = [
        row for row in cache_manifest["files"]
        if row["identity"].startswith("train:44b6_")
    ]
    identities = {row["identity"] for row in cache_rows}
    if identities != {row["identity"] for row in forward_rows}:
        raise AuditError("forward and sparse-cache 44b6 identities differ")
    suffix_counts = {
        suffix: sum(row["path"].endswith(suffix) for row in cache_rows)
        for suffix in (".bhc", ".json")
    }
    if suffix_counts != {".bhc": 36, ".json": 36}:
        raise AuditError(f"sparse-cache suffix coverage drifted: {suffix_counts}")
    sparse_files = []
    for row in sorted(cache_rows, key=lambda item: item["path"]):
        relative = row["path"]
        observed = receipt(fit_cache / relative, row["sha256"])
        if observed["bytes"] != row["bytes"]:
            raise AuditError(f"sparse-cache byte count mismatch: {relative}")
        sparse_files.append({**observed, "identity": row["identity"]})

    checkpoint_observations = []
    exact_checkpoint = None
    for candidate in checkpoint_candidates:
        if candidate.is_file():
            observed = sha256(candidate)
            checkpoint_observations.append({
                "path": str(candidate),
                "present": True,
                "sha256": observed,
                "bytes": candidate.stat().st_size,
            })
            if observed in {CHECKPOINT_MEMBER_SHA256, CHECKPOINT_ARCHIVE_SHA256}:
                exact_checkpoint = candidate
        else:
            checkpoint_observations.append({
                "path": str(candidate),
                "present": False,
            })

    reverse_cache_candidates = sorted(
        root.glob("reports/g02/**/*reverse*trackastra*")
    )
    reverse_files = [str(path) for path in reverse_cache_candidates if path.is_file()]
    reverse_supplier_present = bool(reverse_files)
    checkpoint_present = exact_checkpoint is not None
    authentic_supplier_present = checkpoint_present and reverse_supplier_present

    forward_calls = sum(row["inference_calls"] for row in forward_files)
    return {
        "schema": "g02.sa-reverse.capability-audit.v1",
        "status": (
            "READY_TO_VALIDATE_EXISTING_REVERSE_SUPPLIER"
            if authentic_supplier_present
            else "STOP_COMPUTE_REQUEST_REQUIRED"
        ),
        "frozen_input_receipts": frozen,
        "source_authentication": {
            "source_commit": "fb265888c8fce2ac16a362ebf182ccd5ea48e936",
            "license": "BSD-3-Clause",
            "files": source,
            "all_pins_match": True,
        },
        "forward_cache": {
            "report_sha256": ROOT_INPUTS[
                "reports/g02/trackastra_h1_feature_cache/report.json"
            ],
            "video_count": len(forward_files),
            "feature_bytes": sum(row["bytes"] for row in forward_files),
            "inference_calls": forward_calls,
            "proposal_count": sum(row["proposal_count"] for row in forward_files),
            "row_count": sum(row["row_count"] for row in forward_files),
            "files": forward_files,
            "authentic": True,
            "direction": "forward_only",
        },
        "sparse_detection_cache": {
            "manifest": manifest_receipt,
            "video_count": len(identities),
            "file_count": len(sparse_files),
            "bhc_count": suffix_counts[".bhc"],
            "json_count": suffix_counts[".json"],
            "bytes": sum(row["bytes"] for row in sparse_files),
            "files": sparse_files,
            "authentic": True,
        },
        "checkpoint": {
            "required_archive_sha256": CHECKPOINT_ARCHIVE_SHA256,
            "required_member_sha256": CHECKPOINT_MEMBER_SHA256,
            "parameter_count": 27456880,
            "tensor_count": 280,
            "tensor_bytes": 109827520,
            "candidates": checkpoint_observations,
            "exact_local_checkpoint_present": checkpoint_present,
        },
        "reverse_supplier": {
            "existing_candidate_files": reverse_files,
            "immutable_authentic_reverse_cache_present": reverse_supplier_present,
            "endpoint_return_contract_frozen_in_lane": True,
            "genuine_reverse_inference_required": not reverse_supplier_present,
            "synthetic_reverse_used": False,
        },
        "decision": {
            "disposition": (
                "READY_TO_VALIDATE"
                if authentic_supplier_present
                else "PARK_NOT_READY_BOUNDED_COMPUTE_REQUEST"
            ),
            "reason": (
                "The authentic forward and sparse detection caches plus pinned source "
                "are local, but neither the exact checkpoint nor an immutable reverse "
                "cache is local. A separately executed reversed-time model pass is "
                "required; forward negation/transposition/duplication is forbidden."
            ),
            "separation_executed": False,
            "graph_executed": False,
            "tier_move_proposed": False,
        },
        "forbidden_actions": {
            "reverse_inference": False,
            "synthetic_reverse": False,
            "separation": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "gpu": False,
            "kaggle": False,
            "budget_debit": False,
            "runpod": False,
            "submission": False,
            "dead": False,
            "shared_state_write": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-candidate", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise AuditError(f"refusing to overwrite: {args.output}")
    result = audit(
        args.root.resolve(),
        args.fit_cache.resolve(),
        args.source_root.resolve(),
        [Path(value).resolve() for value in args.checkpoint_candidate],
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

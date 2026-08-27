#!/usr/bin/env python3
"""Audit R-family supplier provenance and freeze the banked per-class population."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
POP_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr"
    / "phase2_separation_gate/run_v1"
)

PINNED = {
    "n1_interface": (
        ROOT
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "grouped_folds": (
        ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        ROOT / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "locked_plan": (
        ROOT / "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md",
        "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
    ),
    "alignment": (
        ROOT / "reports/g02/screening/tiered_surface/ALIGNMENT.md",
        "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
    ),
    "scope_manifest": (
        ROOT / "reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md",
        "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
    ),
    "per_class_amendment": (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_repr"
        / "phase2_separation_gate/PREREGISTRATION_v2_per_class.json",
        "c3c86ec91307fc19a393f060912b50fe95fe42a0d68102508d154c3fe498458c",
    ),
    "population_manifest": (
        POP_ROOT / "population_manifest.json",
        "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    ),
    "f0_feature_contract": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01"
        / "f0_triview_dense_ta/N0_feature_contract.json",
        "5a41c125e66f2e6611ef262f8e3a03970921831a4fa1144ed36274ab567be547",
    ),
    "f0_prior_weight_audit": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01"
        / "f0_triview_dense_ta/N0_weights_license_manifest.json",
        "b15c491b1166e9f6c5a1bef3d776bd84016f29eb5273603dc1dd2d0bf06803e6",
    ),
}

CELLECT = {
    "contract": (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_repr"
        / "FROZEN_SUPPLIER_CONTRACT_v1.json",
        "fffb09c2865fd70010f5d93d5db9eb4cca48bac851c9abe2f77bbc9b27e83e70",
    ),
    "weight": (
        ROOT / "external/CELLECT/model/U-ext+-x3rd-149.0-4.6540.pth",
        "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5",
    ),
    "license": (
        ROOT / "external/CELLECT/LICENSE",
        "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643",
    ),
    "source": (
        ROOT / "external/CELLECT/unetext3Dn_con7.py",
        "3b44f7c64c4b948ab7cc14bb32d5a86f61d3f4cff0ee3425a5ec32cc6c17d3ba",
    ),
    "cleanroom_model": (
        ROOT / "_LATER/cellect/src/celltrack_alpha/cellect_model.py",
        "82df5c1cc209405e2f3a047098e9a0e647b1cd963cf641c3abf6764bbf75b395",
    ),
    "adapter": (
        ROOT / "_LATER/cellect/src/celltrack_alpha/cellect_adapter.py",
        "bc40dca77702416798cf29692a3477d4ec84d4cc33fd17989172f71d1b8f5153",
    ),
}

ENDPOINT_KEYS = (
    "source_id",
    "target_id",
    "source_t",
    "target_t",
    "source_zyx",
    "target_zyx",
)


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, value: Any) -> None:
    path = HERE / name
    if path.exists():
        raise AuditError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def authenticate_pins() -> dict[str, Any]:
    rows = {}
    for name, (path, expected) in PINNED.items():
        observed = sha256(path)
        if observed != expected:
            raise AuditError(f"{name}: expected {expected}, observed {observed}")
        rows[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "pass": True,
        }
    return rows


def validate_shard(
    population_path: Path, request_path: Path, manifest_row: dict[str, Any]
) -> dict[str, Any]:
    if sha256(population_path) != manifest_row["population_sha256"]:
        raise AuditError(f"population hash mismatch: {population_path}")
    if sha256(request_path) != manifest_row["request_sha256"]:
        raise AuditError(f"request hash mismatch: {request_path}")
    with np.load(population_path, allow_pickle=False) as pop, np.load(
        request_path, allow_pickle=False
    ) as request:
        count = int(manifest_row["count"])
        for key in ENDPOINT_KEYS:
            if key not in pop or key not in request:
                raise AuditError(f"{population_path}: missing endpoint {key}")
            if not np.array_equal(pop[key], request[key]):
                raise AuditError(f"{population_path}: request mismatch {key}")
        required = {"b0", "oracle_delta_S", "oracle_label", "fold", "fp_budget"}
        if not required.issubset(pop.files):
            raise AuditError(f"{population_path}: missing {sorted(required - set(pop.files))}")
        if len(pop["oracle_label"]) != count or len(pop["source_id"]) != count:
            raise AuditError(f"{population_path}: count mismatch")
        delta = pop["oracle_delta_S"]
        label = pop["oracle_label"]
        if not np.isfinite(delta).all() or not np.isfinite(pop["b0"]).all():
            raise AuditError(f"{population_path}: nonfinite value")
        if not np.array_equal(label, (delta > 0.0).astype(np.uint8)):
            raise AuditError(f"{population_path}: label/delta mismatch")
        if set(np.unique(pop["fold"]).tolist()) - {0, 1, 2, 3, 4}:
            raise AuditError(f"{population_path}: invalid fold")
        if np.any(pop["target_t"] != pop["source_t"] + 1):
            raise AuditError(f"{population_path}: non-adjacent endpoint")
        if int(label.sum()) != int(manifest_row["beneficial"]):
            raise AuditError(f"{population_path}: beneficial count mismatch")
        if int((label == 0).sum()) != int(manifest_row["harmful"]):
            raise AuditError(f"{population_path}: harmful count mismatch")
        if int(pop["fp_budget"][0]) != int(manifest_row["fp_budget"]):
            raise AuditError(f"{population_path}: FP budget mismatch")
        fold_values = sorted(int(value) for value in np.unique(pop["fold"]))
    return {
        "count": count,
        "beneficial": int(manifest_row["beneficial"]),
        "harmful": int(manifest_row["harmful"]),
        "fp_budget": int(manifest_row["fp_budget"]),
        "fold_values": fold_values,
        "population_bytes": population_path.stat().st_size,
        "request_bytes": request_path.stat().st_size,
    }


def freeze_population() -> dict[str, Any]:
    source = json.loads((POP_ROOT / "population_manifest.json").read_text())
    inventory = []
    seen = set()
    totals: dict[str, dict[str, int]] = {}
    aggregate = hashlib.sha256()
    for action_class in ("delete", "occupied_swap"):
        rows = source["classes"][action_class]["videos"]
        class_totals = {"actions": 0, "beneficial": 0, "harmful": 0, "fp_budget": 0}
        for row in rows:
            identity = str(row["identity"])
            key = (action_class, identity)
            if key in seen:
                raise AuditError(f"duplicate class/identity: {key}")
            seen.add(key)
            population_path = ROOT / row["population_path"]
            request_path = ROOT / row["request_path"]
            checked = validate_shard(population_path, request_path, row)
            for kind, path, expected in (
                ("population", population_path, row["population_sha256"]),
                ("feature_request", request_path, row["request_sha256"]),
            ):
                relative = str(path.relative_to(ROOT))
                size = path.stat().st_size
                aggregate.update(f"{relative}\\0{expected}\\0{size}\\n".encode())
                inventory.append(
                    {
                        "action_class": action_class,
                        "identity": identity,
                        "kind": kind,
                        "absolute_path": str(path),
                        "relative_path": relative,
                        "sha256": expected,
                        "bytes": size,
                    }
                )
            class_totals["actions"] += checked["count"]
            class_totals["beneficial"] += checked["beneficial"]
            class_totals["harmful"] += checked["harmful"]
            class_totals["fp_budget"] += checked["fp_budget"]
        expected_totals = source["totals"][action_class]
        if class_totals != expected_totals:
            raise AuditError(
                f"{action_class}: totals {class_totals} != manifest {expected_totals}"
            )
        totals[action_class] = class_totals
    return {
        "schema": "g02.sa-repr-multi.population-freeze.v1",
        "status": "FROZEN_COMPLETE",
        "source_manifest": {
            "path": str(POP_ROOT / "population_manifest.json"),
            "sha256": PINNED["population_manifest"][1],
        },
        "read_only": True,
        "family": "44b6",
        "action_classes": ["delete", "occupied_swap"],
        "totals": totals,
        "inventory_file_count": len(inventory),
        "inventory_total_bytes": sum(row["bytes"] for row in inventory),
        "aggregate_rule": "SHA256 over sorted manifest traversal records: relative_path NUL file_sha256 NUL bytes LF",
        "aggregate_sha256": aggregate.hexdigest(),
        "integrity": {
            "manifest_hash_exact": True,
            "all_files_present_and_hash_exact": True,
            "feature_request_endpoints_array_exact": True,
            "labels_equal_delta_S_gt_zero": True,
            "finite_b0_and_delta": True,
            "folds_subset_0_through_4": True,
            "adjacent_frame_endpoints": True,
            "class_totals_exact": True,
            "duplicate_identity_class_pairs": 0,
        },
        "inventory": inventory,
    }


def discover_dense_weights() -> dict[str, Any]:
    roots = [
        Path("/Users/agent/.cache/torch"),
        Path("/Users/agent/.cache/huggingface"),
        ROOT,
    ]
    matched = []
    file_count = 0
    skipped_names = {".git", "__pycache__", "node_modules"}
    for scan_root in roots:
        if not scan_root.exists():
            continue
        for directory, dirs, files in os.walk(scan_root):
            dirs[:] = sorted(name for name in dirs if name not in skipped_names)
            for name in sorted(files):
                file_count += 1
                lower = name.lower()
                if (
                    "dinov2" in lower
                    or "dinov3" in lower
                    or ("vits14" in lower and lower.endswith((".pth", ".pt", ".ckpt")))
                    or (lower.startswith("sam") and lower.endswith((".pth", ".pt", ".ckpt")))
                ):
                    path = Path(directory) / name
                    matched.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
    return {
        "scan_roots": [str(path) for path in roots],
        "files_seen": file_count,
        "semantic_dense_weight_matches": matched,
        "exact_dinov2_vits14_pretrain_matches": [
            row for row in matched if Path(row["path"]).name == "dinov2_vits14_pretrain.pth"
        ],
    }


def authenticate_cellect() -> dict[str, Any]:
    artifacts = {}
    for name, (path, expected) in CELLECT.items():
        observed = sha256(path)
        if observed != expected:
            raise AuditError(f"CELLECT {name}: expected {expected}, observed {observed}")
        artifacts[name] = {
            "path": str(path),
            "sha256": observed,
            "pass": True,
        }
    head = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/CELLECT"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/CELLECT"), "status", "--porcelain=v1"],
        text=True,
    ).strip()
    if head != "3586070926f7f1fd5d8df37456861d22bdc63236" or dirty:
        raise AuditError(f"CELLECT repository identity mismatch head={head} dirty={dirty!r}")
    return {
        "authenticated": True,
        "supplier_id": "F2-CELLECT-PUBLISHED-64D-V1",
        "upstream_commit": head,
        "upstream_tree_clean": True,
        "artifacts": artifacts,
        "offline_support_previously_proven": True,
        "queue_disposition": "EXCLUDED_ALREADY_DISPATCHED_R7_SERIAL_LANE",
    }


def provenance_report(pins: dict[str, Any], dense: dict[str, Any]) -> dict[str, Any]:
    cellect = authenticate_cellect()
    r4_ready = bool(dense["exact_dinov2_vits14_pretrain_matches"])
    if r4_ready:
        # A filename match still lacks pinned source/license/offline-loader proof.
        r4_ready = False
    source_register = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/representation_and_governance.md"
    )
    source_register_sha = sha256(source_register)
    h3_absence = ROOT / "reports/g02/screening/h1_execution.json"
    h3_absence_sha = sha256(h3_absence)
    suppliers = [
        {
            "node": "R2",
            "candidate": "H3-AUTHENTIC-MORPHOLOGY",
            "authentication": "NOT_READY",
            "reason": "No authenticated complete mask/wrfeat identity join is banked; raw Zarr intensity is not a mask.",
            "evidence": {"path": str(h3_absence), "sha256": h3_absence_sha},
            "tier_move": None,
        },
        {
            "node": "R3",
            "candidate": "F1-REAL-WRFEAT",
            "authentication": "NOT_READY_DEPENDENCY_R2",
            "reason": "Authentic R2 morphology is absent.",
            "evidence": {"path": str(h3_absence), "sha256": h3_absence_sha},
            "tier_move": None,
        },
        {
            "node": "R4",
            "candidate": "F0-DINOV2-VITS14-PROVENANCE",
            "authentication": "NOT_READY",
            "reason": "No exact local DINOv2 ViT-S/14 weight was found; source revision, weight license, and offline loader therefore cannot all authenticate.",
            "discovery": dense,
            "evidence": {
                "path": str(PINNED["f0_prior_weight_audit"][0]),
                "sha256": PINNED["f0_prior_weight_audit"][1],
            },
            "tier_move": None,
        },
        {
            "node": "R5",
            "candidate": "F0-XY-DENSE-TA",
            "authentication": "NOT_READY_DEPENDENCY_R4",
            "reason": "The frozen crop/view contract exists, but R4 has no authenticated encoder.",
            "evidence": {
                "path": str(PINNED["f0_feature_contract"][0]),
                "sha256": PINNED["f0_feature_contract"][1],
            },
            "tier_move": None,
        },
        {
            "node": "R6",
            "candidate": "F0-TRIVIEW-DENSE-TA",
            "authentication": "NOT_READY_DEPENDENCY_R4_R5",
            "reason": "The frozen tri-view/permutation contract exists, but R4 and the XY beta cache are absent.",
            "evidence": {
                "path": str(PINNED["f0_feature_contract"][0]),
                "sha256": PINNED["f0_feature_contract"][1],
            },
            "tier_move": None,
        },
        {
            "node": "R7",
            "candidate": "F2-CELLECT-PUBLISHED-64D-V1",
            "authentication": "AUTHENTICATED",
            "reason": "Exact weight/source/license/adapter hashes and clean pinned upstream commit match the banked supplier contract.",
            "evidence": cellect,
            "tier_move": None,
        },
        {
            "node": "R8",
            "candidate": "C3-BIDIR-CYCLE",
            "authentication": "NOT_READY",
            "reason": "No immutable reverse-inference supplier or endpoint-return contract is banked.",
            "evidence": {"path": str(source_register), "sha256": source_register_sha},
            "tier_move": None,
        },
        {
            "node": "R9",
            "candidate": "ENV-GROUPDRO-TRACKLET",
            "authentication": "NOT_READY",
            "reason": "No new tracklet/event evidence supplier is banked; reusing only prior pair rows is ineligible.",
            "evidence": {"path": str(source_register), "sha256": source_register_sha},
            "tier_move": None,
        },
        {
            "node": "R10",
            "candidate": "PEFT-44B6-ADAPTER",
            "authentication": "NOT_READY",
            "reason": "No separately frozen <1% adapter contract paired with new authenticated evidence is banked.",
            "evidence": {"path": str(source_register), "sha256": source_register_sha},
            "tier_move": None,
        },
    ]
    return {
        "schema": "g02.sa-repr-multi.provenance-audit.v1",
        "status": "COMPLETE",
        "frozen_input_authentication": pins,
        "supplier_count": len(suppliers),
        "authenticated_count": sum(
            row["authentication"] == "AUTHENTICATED" for row in suppliers
        ),
        "authenticated_non_r7_count": 0,
        "suppliers": suppliers,
        "governance": {
            "external_downloads": 0,
            "external_services": 0,
            "gpu_jobs": 0,
            "gpu_authorizations_requested_or_held": 0,
            "graph_runs": 0,
            "family_6bba_runs": 0,
            "cv57_runs": 0,
            "submissions": 0,
            "runpod_jobs": 0,
            "pseudo_masks_created": 0,
            "tier_moves": 0,
            "memo_c2_or_branch_b_values_used": 0,
        },
        "interpretation": "Only R7 authenticates, and its serial sample is already dispatched outside this lane. Every other R-family item remains NOT_READY without a scientific negative or tier move.",
    }


def queue_report(provenance: dict[str, Any], population: dict[str, Any]) -> dict[str, Any]:
    ready = []
    excluded = [
        {
            "supplier_id": "F2-CELLECT-PUBLISHED-64D-V1",
            "node": "R7",
            "reason": "Authenticated but already dispatched on the independent serial GPU lane; duplicate request/authorization is forbidden.",
        }
    ]
    return {
        "schema": "g02.sa-repr-multi.ready-sample-queue.v1",
        "status": "NO_NEW_READY_SUPPLIER",
        "population_aggregate_sha256": population["aggregate_sha256"],
        "queue_scope": "Exact frozen 44b6 delete and occupied-swap endpoint sample only",
        "ready_samples": ready,
        "ready_sample_count": len(ready),
        "authenticated_but_excluded": excluded,
        "projection_rule": "Queue only with measured same-implementation projection <5 h; no literature-only or unverified estimate.",
        "authorization_requested": False,
        "authorization_held": False,
        "gpu_launched": False,
        "verdict_barrier": "HOLD_PARK_NOT_READY",
        "next_named_inputs_that_can_reopen": [
            "Exact locally supplied DINOv2 ViT-S/14 weights plus pinned source and license bundle",
            "Authentic complete per-detection mask/wrfeat identity join",
            "Immutable reverse-inference feature supplier for R8",
            "New tracklet/event evidence supplier for R9",
            "Separately frozen <1% adapter contract paired with authenticated new evidence for R10"
        ],
    }


def main() -> int:
    started = time.perf_counter()
    prereg = HERE / "PREREGISTRATION_v1.json"
    prereg_sha = sha256(prereg)
    pins = authenticate_pins()
    population = freeze_population()
    dense = discover_dense_weights()
    provenance = provenance_report(pins, dense)
    queue = queue_report(provenance, population)
    population["preregistration_sha256"] = prereg_sha
    provenance["preregistration_sha256"] = prereg_sha
    queue["preregistration_sha256"] = prereg_sha
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "g02.sa-repr-multi.audit-summary.v1",
        "status": "PARK_NOT_READY_AT_VERDICT_BARRIER",
        "preregistration": {"path": str(prereg), "sha256": prereg_sha},
        "population": {
            "status": population["status"],
            "aggregate_sha256": population["aggregate_sha256"],
            "inventory_file_count": population["inventory_file_count"],
            "totals": population["totals"],
        },
        "provenance": {
            "authenticated_count": provenance["authenticated_count"],
            "authenticated_non_r7_count": provenance["authenticated_non_r7_count"],
        },
        "queue": {
            "ready_sample_count": queue["ready_sample_count"],
            "authorization_requested": False,
            "authorization_held": False,
        },
        "execution": {
            "elapsed_seconds": elapsed,
            "cpu_only": True,
            "gpu": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "submission": False,
            "runpod": False,
        },
        "authoritative_comparison_rules_changed": False,
        "tier_moves": [],
        "uncited_tier_moves": 0,
        "memo_quarantine_pass": True,
        "barrier": "HOLD_FOR_ROOT_RECONCILE; DO NOT WAIT FOR R7",
    }
    write_json("POPULATION_FREEZE.json", population)
    write_json("SUPPLIER_PROVENANCE_AUDIT.json", provenance)
    write_json("READY_SAMPLE_QUEUE.json", queue)
    write_json("AUDIT_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"FAIL CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)

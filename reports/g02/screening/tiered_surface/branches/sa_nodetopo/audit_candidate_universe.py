#!/usr/bin/env python3
"""Fail-closed provenance audit for the SA-NODETOPO candidate prerequisite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PINS = {
    "n1_interface": (
        "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "grouped_folds": (
        "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "fit_provenance": (
        "reports/g02/fit_provenance_v1.json",
        "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f",
    ),
    "locked_plan": (
        "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md",
        "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
    ),
    "dataset_registry": (
        "reports/g02/dataset_registry_v1.json",
        "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158",
    ),
    "fit105_feature_report": (
        "reports/g02/screening/h1_existing_cache_feature_report.json",
        "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf",
    ),
}

LEGACY_REPORTS = (
    "_ARCHIVE/general_model/reports/general_model_edge_training_table_fold0_smoke.json",
    "_ARCHIVE/general_model/reports/general_model_edge_training_table_fold0_clean_smoke.json",
    "_ARCHIVE/general_model/reports/general_model_edge_training_table_fold0_current_weak_routes_4ds.json",
    "_ARCHIVE/general_model/reports/general_model_edge_training_table_fold0_hybrid_rescue_8ds.json",
)

FORBIDDEN_LABEL_COLUMNS = {
    "label",
    "label_status",
    "matched_gt_id",
    "matched_source_gt_id",
    "matched_target_gt_id",
    "nearest_gt_um",
}

IDENTITY_COLUMNS = (
    "dataset",
    "route",
    "clip_start",
    "clip_end",
    "t",
    "node_id",
    "z",
    "y",
    "x",
    "sources",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def authenticate_pins(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, (relative, expected) in PINS.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        result[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": actual == expected,
        }
    return result


def candidate_report_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    unique_nodes: dict[tuple[str, ...], tuple[str, str, str]] = {}
    identity_conflicts: list[dict[str, Any]] = []
    covered_datasets: set[str] = set()
    label_columns: set[str] = set()
    total_rows = 0

    for relative in LEGACY_REPORTS:
        report_path = root / relative
        payload = read_json(report_path)
        node_path = root / str(payload["node_csv"])
        edge_path = root / str(payload["edge_csv"])
        node_actual = sha256_file(node_path)
        edge_actual = sha256_file(edge_path)
        with node_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = sorted(set(IDENTITY_COLUMNS) - set(fields))
            label_columns.update(FORBIDDEN_LABEL_COLUMNS & set(fields))
            rows = 0
            for row in reader:
                rows += 1
                total_rows += 1
                identity = tuple(str(row.get(key, "")) for key in IDENTITY_COLUMNS)
                coordinate = (str(row.get("z", "")), str(row.get("y", "")), str(row.get("x", "")))
                source = (relative, str(payload["node_csv"]), str(rows))
                previous = unique_nodes.get(identity)
                if previous is None:
                    unique_nodes[identity] = coordinate
                elif previous != coordinate:
                    identity_conflicts.append(
                        {"identity": list(identity), "first_coordinate": list(previous), "second_coordinate": list(coordinate), "source": list(source)}
                    )
                covered_datasets.add(str(row.get("dataset", "")))
        reports.append(
            {
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "fold_id": str(payload.get("fold_id")),
                "routes": payload.get("routes"),
                "clip_length": payload.get("clip_length"),
                "clips_per_dataset": payload.get("clips_per_dataset"),
                "source_datasets": payload.get("source_datasets"),
                "node_path": str(node_path),
                "node_expected_sha256": payload.get("node_csv_sha256"),
                "node_actual_sha256": node_actual,
                "node_hash_ok": node_actual == payload.get("node_csv_sha256"),
                "edge_path": str(edge_path),
                "edge_expected_sha256": payload.get("edge_csv_sha256"),
                "edge_actual_sha256": edge_actual,
                "edge_hash_ok": edge_actual == payload.get("edge_csv_sha256"),
                "node_rows_reported": payload.get("node_rows"),
                "node_rows_observed": rows,
                "identity_columns_missing": missing,
                "embedded_label_columns": sorted(FORBIDDEN_LABEL_COLUMNS & set(fields)),
            }
        )

    return reports, {
        "candidate_rows_across_reports": total_rows,
        "unique_candidate_identities": len(unique_nodes),
        "covered_datasets": sorted(covered_datasets),
        "embedded_label_columns": sorted(label_columns),
        "identity_conflict_count": len(identity_conflicts),
        "identity_conflicts": identity_conflicts[:20],
    }


def origin_audit(root: Path) -> dict[str, Any]:
    script = root / "scripts/build_candidate_graph_training_table.py"
    source = script.read_text(encoding="utf-8")
    generate_at = source.find("nodes_by_t, all_nodes = generate_candidates(")
    label_at = source.find("labels = match_nodes_by_time(", generate_at + 1)
    reads_gt_at = source.find('gt = read_geff_arrays(')
    return {
        "script": str(script),
        "script_sha256": sha256_file(script),
        "generator": "celltrack_alpha.proposals.generate_candidates",
        "generation_precedes_label_join": generate_at >= 0 and label_at > generate_at,
        "raw_generation_is_label_independent_in_call_order": generate_at >= 0 and label_at > generate_at,
        "artifact_is_later_gt_annotated": reads_gt_at >= 0 and label_at >= 0,
        "interpretation": (
            "candidate coordinate generation precedes the GT join, but the persisted tables "
            "are training artifacts with embedded GT-derived columns and cover only fixed short clips"
        ),
    }


def extracted_raw_audit(root: Path, fit_identities: set[str]) -> dict[str, Any]:
    raw_root = root / "data/derived/t03_fold0_fit10_images_v1/train"
    zarr_dirs = sorted(path for path in raw_root.glob("*.zarr") if path.is_dir())
    identities = {f"train:{path.name[:-5]}" for path in zarr_dirs}
    by_family = {
        family: len([identity for identity in identities if identity.startswith(f"train:{family}_")])
        for family in ("44b6", "6bba")
    }
    return {
        "root": str(raw_root),
        "zarr_count": len(zarr_dirs),
        "family_counts": by_family,
        "identities": sorted(identities),
        "fit105_overlap_count": len(identities & fit_identities),
        "fit105_overlap_identities": sorted(identities & fit_identities),
        "complete_fit105_raw_coverage": identities >= fit_identities,
        "note": "the directory name says fit10 but contains 12 extracted training Zarrs; identity, not the directory label, is audited",
    }


def n1_contract_evidence(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from celltrack_alpha.action_compiler import (  # type: ignore
        ActionCompileError,
        ActionKind,
        GraphAction,
        compile_action,
    )
    from celltrack_alpha.kaggle_calibration import (  # type: ignore
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
    )

    baseline = SubmissionGraph(
        dataset="synthetic:sa-nodetopo-contract",
        nodes={
            1: SubmissionNode(1, 0, 0, 0, 0),
            2: SubmissionNode(2, 1, 0, 1, 0),
            3: SubmissionNode(3, 2, 0, 2, 0),
        },
        edges=[SubmissionEdge(1, 2), SubmissionEdge(2, 3)],
    )
    valid = GraphAction(
        action_id="contract.restore.4",
        kind=ActionKind.RESTORE_TOPOLOGY,
        add_nodes=(SubmissionNode(4, 1, 0, 0, 1),),
        add_edges=((1, 4), (4, 3)),
        replace_outgoing_from=(1,),
        replace_incoming_to=(3,),
        expected_conflict_edges=((1, 2), (2, 3)),
    )
    compiled = compile_action(baseline, valid)
    isolated_rejected = False
    isolated_error = None
    try:
        compile_action(
            baseline,
            GraphAction(
                action_id="contract.isolated.4",
                kind=ActionKind.RESTORE_TOPOLOGY,
                add_nodes=(SubmissionNode(4, 1, 0, 0, 1),),
            ),
        )
    except ActionCompileError as exc:
        isolated_rejected = True
        isolated_error = str(exc)
    return {
        "scope": "synthetic interface contract only; not scientific candidate evidence",
        "valid_complete_action": {
            "compiled": True,
            "action_sha256": compiled.action_sha256,
            "before_graph_sha256": compiled.before_graph_sha256,
            "after_graph_sha256": compiled.after_graph_sha256,
            "removed_edges": [list(pair) for pair in compiled.removed_edges],
            "observation_distinct": compiled.before_graph_sha256 != compiled.after_graph_sha256,
        },
        "isolated_coordinate_only_control": {
            "rejected": isolated_rejected,
            "error": isolated_error,
        },
        "real_candidate_actions_compiled": 0,
        "reason": "no complete provenance-pinned candidate-to-baseline incident-topology join exists",
    }


def audit(root: Path) -> dict[str, Any]:
    prereg = Path(__file__).with_name("PREREGISTRATION.json")
    prereg_hash = sha256_file(prereg)
    pins = authenticate_pins(root)
    if not all(row["ok"] for row in pins.values()):
        bad = [name for name, row in pins.items() if not row["ok"]]
        raise RuntimeError(f"frozen input mismatch: {bad}")

    feature_report = read_json(root / PINS["fit105_feature_report"][0])
    fit_identities = {str(row["identity"]) for row in feature_report.get("files", [])}
    fit44 = {identity for identity in fit_identities if identity.startswith("train:44b6_")}
    fit6 = {identity for identity in fit_identities if identity.startswith("train:6bba_")}

    reports, candidates = candidate_report_audit(root)
    covered44 = {
        f"train:{dataset}"
        for dataset in candidates["covered_datasets"]
        if str(dataset).startswith("44b6_")
    }
    covered6 = {
        f"train:{dataset}"
        for dataset in candidates["covered_datasets"]
        if str(dataset).startswith("6bba_")
    }
    partial_fit44 = covered44 & fit44
    partial_fit6 = covered6 & fit6

    p0 = read_json(root / PINS["p0"][0])
    p0_node_restore_backgrounds = sorted(
        {
            sample["actions"]["node_restore"]["counts"].get("background_definition")
            for sample in p0.get("samples", [])
        }
    )
    p0_background_counts = sorted(
        {
            sample["actions"]["node_restore"]["counts"].get("raw_background_actions")
            for sample in p0.get("samples", [])
        },
        key=lambda value: (value is not None, str(value)),
    )

    origin = origin_audit(root)
    raw = extracted_raw_audit(root, fit_identities)
    n1 = n1_contract_evidence(root)
    complete44 = len(partial_fit44) == len(fit44) and all(
        row.get("clip_length") == 100 for row in reports
    )
    exact_background = False
    join_complete = False
    authentic_origin = origin["raw_generation_is_label_independent_in_call_order"]
    verdict = "PARK_NO_COMPLETE_AUTHENTIC_CANDIDATE_UNIVERSE"
    blockers = [
        f"legacy candidate artifacts cover only {len(partial_fit44)}/{len(fit44)} fit105 44b6 datasets and one 3-frame clip per dataset",
        "persisted candidate tables contain GT-derived label/join columns; labels are excluded but the artifacts are not a frozen deployment-population bank",
        "no exact unlabeled background count exists for a complete restoration universe",
        "no immutable candidate-to-baseline node/incident-conflict join exists, so zero real N1 restore_topology actions can be formed",
        f"only {raw['zarr_count']} raw training Zarrs are extracted locally and they do not cover fit105",
        "P0 explicitly reports node-restore background as undefined/null and is not a proposal source",
    ]
    return {
        "schema": "celltrack-alpha.sa-nodetopo-provenance-audit.v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "preregistration": {
            "path": str(prereg),
            "sha256": prereg_hash,
            "git_commit": "1abc8c7",
            "outcome_opened_after_commit": True,
        },
        "pins": pins,
        "scope": {
            "fit_only": True,
            "fit105_count": len(fit_identities),
            "fit105_family_counts": {"44b6": len(fit44), "6bba": len(fit6)},
            "audit_used": False,
            "test_split_used": False,
            "cv57_used": False,
            "held_graph_scored": False,
            "second_family_opened": False,
            "labels_used_for_candidate_selection": False,
            "external_action": False,
            "gpu_action": False,
        },
        "proposal_origin": origin,
        "legacy_candidate_artifacts": reports,
        "candidate_identity_audit": {
            **candidates,
            "fit105_partial_dataset_coverage": {
                "44b6": {"covered": len(partial_fit44), "required": len(fit44), "identities": sorted(partial_fit44)},
                "6bba": {"covered": len(partial_fit6), "required": len(fit6), "identities": sorted(partial_fit6)},
            },
            "complete_44b6_coverage": complete44,
            "exact_candidate_to_baseline_join": join_complete,
        },
        "raw_zarr_evidence": raw,
        "background_contract": {
            "exact_unlabeled_background_available": exact_background,
            "p0_background_definitions": p0_node_restore_backgrounds,
            "p0_background_counts": p0_background_counts,
            "stress_curve_executed": False,
            "reason": "a stress curve cannot honestly be instantiated until the complete immutable background universe exists",
        },
        "prerequisite_checks": {
            "authentic_label_independent_origin_procedure": authentic_origin,
            "complete_fit105_44b6_candidate_bank": complete44,
            "exact_unlabeled_background": exact_background,
            "unambiguous_candidate_to_baseline_join": join_complete,
            "real_actions_100_percent_n1_valid": False,
        },
        "n1_contract_evidence": n1,
        "blockers": blockers,
        "decision": {
            "tier_move_proposed": False,
            "current_tier": "PARK_A",
            "banked_p0_headroom_retained": True,
            "restoration_fitting_open": False,
            "family_44b6_graph_run": False,
            "family_6bba_graph_run": False,
            "cv57_run": False,
            "gpu_request_emitted": False,
            "resume_condition": (
                "root freezes a full fit105, label-independent candidate bank with exact deployment background "
                "and an immutable candidate-to-baseline incident-topology join; then rerun this audit"
            ),
        },
        "quarantine": {
            "memo_c2_values_used": False,
            "memo_branch_b_66_223_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

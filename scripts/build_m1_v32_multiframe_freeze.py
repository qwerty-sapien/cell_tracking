"""Build the local-only immutable M1 N3 feasibility package."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from m1_v32_multiframe_lineage import (
    CANDIDATE_ID,
    EXPECTED_FIT_MANIFEST_SHA256,
    FIT_MANIFEST,
    FIXED_SEED,
    FrozenConfig,
    authenticate_contract,
    benchmark_synthetic_forward,
    build_frozen_model,
    environment_receipt,
    export_five_frame_window,
    model_identity,
    run_control_contract,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dag_prompts/2026-08-23-model-clade-renewal-v1/m1_v32_multiframe"
RUNS = OUT / "runs"
MODULE = ROOT / "scripts/m1_v32_multiframe_lineage.py"
BUILDER = ROOT / "scripts/build_m1_v32_multiframe_freeze.py"
TEST = ROOT / "tests/test_m1_v32_multiframe_lineage.py"


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def synthetic_export(identity: str) -> dict[str, Any]:
    nodes = 3
    node_ids = [np.arange(slot * nodes, (slot + 1) * nodes, dtype=np.int64) for slot in range(5)]
    coords = []
    features = []
    for slot in range(5):
        coords.append(
            np.asarray([[node, 10 * node + slot, 20 * node + 2 * slot] for node in range(nodes)], dtype=np.float32)
        )
        value = np.zeros((nodes, 32), dtype=np.float32)
        for node in range(nodes):
            value[node, node] = 1.0
            value[node, 16] = slot / 4.0
        features.append(value)
    logits = []
    for _ in range(4):
        value = np.full((nodes, nodes), -6.0, dtype=np.float32)
        np.fill_diagonal(value, 6.0)
        logits.append(value)
    target = OUT / "fixtures/selected_train_synthetic_export"
    if target.exists():
        for path in sorted(target.iterdir()):
            if path.is_file() and not path.is_symlink():
                path.unlink()
    result = export_five_frame_window(
        project_root=ROOT,
        identity=identity,
        frames=(10, 11, 12, 13, 14),
        node_ids=node_ids,
        coords_um=coords,
        node_features=features,
        raw_pair_logits=logits,
        output_dir=target,
    )
    return {
        "synthetic_only": True,
        "source_identity_used_as_schema_fixture_only": identity,
        "manifest_sha256": result["manifest_sha256"],
        "files": result["files"],
    }


def child_plan(*, code_sha: str, environment_sha: str, inventory_sha: str) -> dict[str, Any]:
    return {
        "schema": "auto-experiment.plan.v1",
        "study_id": "M1-V32-NATIVE-MULTIFRAME-LINEAGE-FIT44B6-v1",
        "mode": "exploratory",
        "purpose": (
            "Falsify one immutable five-frame V32-native lineage realization on grouped fit-only "
            "movies without hyperparameter, threshold, architecture, checkpoint, or seed selection."
        ),
        "mechanism_id": CANDIDATE_ID,
        "authority": {
            "references": [
                "AGENTS.md",
                "LMT.md",
                "reports/CURRENT.md",
                "dag_prompts/2026-08-23-model-clade-renewal-v1/DAG.md",
                "dag_prompts/2026-08-23-model-clade-renewal-v1/N4_native_multiframe_falsifier.md",
            ],
            "external_actions": False,
            "protected_outcomes_open": False,
        },
        "evidence": {
            "primary_sources": [
                "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf"
            ],
            "local_receipts": [
                {
                    "path": "dag_prompts/2026-08-23-model-clade-renewal-v1/runs/V32_NATIVE_MULTIFRAME_SEAM.json",
                    "sha256": "4031b8ad7cb8aadf564a98f862807a26e476dd883b0109e8a8ed31341be89ada",
                },
                {
                    "path": "dag_prompts/2026-08-23-model-clade-renewal-v1/m1_v32_multiframe/CANDIDATE_INVENTORY.json",
                    "sha256": inventory_sha,
                },
            ],
            "extrapolations": [
                "Synthetic wiring controls establish code integrity only, not target-domain efficacy.",
                "Movie identity is the highest authenticated group; embryo/batch provenance is unavailable.",
                "No local-to-T4 whole-pipeline projection is claimed until an authenticated production census and matched calibration exist.",
            ],
        },
        "experimental_units": {
            "primary_unit": "movie_identity",
            "group_key": "movie_identity",
            "hierarchy": ["acquisition_family", "movie_identity", "track", "cell", "frame", "node"],
            "independence_justification": (
                "All windows, frames, nodes, and tracks from a movie remain in one fold; no population-level "
                "biological inference is allowed because embryo/batch provenance is unavailable."
            ),
            "minimum_independent_units": 21,
        },
        "data": {
            "allowlist_manifest": {
                "path": str(FIT_MANIFEST),
                "sha256": EXPECTED_FIT_MANIFEST_SHA256,
            },
            "forbidden_namespaces": [
                "competition_test",
                "test",
                "sealed confirmation family",
                "public leaderboard",
                "private leaderboard",
                "submission scorer",
            ],
            "protected_holdout": "sealed-36-movie-confirmation",
            "holdout_open_once": True,
        },
        "split": {
            "scheme": "nested_group_kfold",
            "split_unit": "movie_identity",
            "group_isolation": True,
            "nested_selection": True,
            "inner_folds": 3,
            "outer_folds": 3,
            "fixed_seeds": [FIXED_SEED],
            "repeats": 1,
            "selection_scope": "fit-only-44b6-fixed-realization-no-selection",
            "time_ordered": False,
        },
        "outcome_firewall": {
            "selection_uses_holdout": False,
            "adaptive_reuse": False,
            "read_audit": True,
            "forbidden_paths": [
                "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
                "competition_test",
                "sealed confirmation outputs",
                "submission.csv",
            ],
            "decoy_test": True,
            "enforcement_tests": [
                "exact fit-manifest SHA and 21-identity allowlist",
                "deny every non-allowlisted identity",
                "deny competition/test/confirmation/leaderboard/submission/scorer tokens before open or hash",
                "inventory every opened source/manifest path",
            ],
        },
        "hypotheses": [
            {
                "id": "H1-M1-FIT44B6",
                "H0": "The immutable M1 complete-graph adjusted edge Jaccard delta is below +0.005 or a raw-edge/node-recall safety delta is negative.",
                "Ha": "The immutable M1 complete-graph adjusted edge Jaccard delta is at least +0.005 with raw-edge and node-recall deltas nonnegative.",
                "estimand": "paired complete-graph M1 minus byte-exact V32 delta aggregated by movie identity",
                "minimum_effect": 0.005,
                "falsifier": "any adjusted, raw-edge, node-recall, control, integrity, or runtime gate fails",
                "terminal_on_fail": True,
            }
        ],
        "experiments": [
            {
                "id": "E1-M1-FIT44B6-GRAPH-FALSIFIER",
                "hypotheses": ["H1-M1-FIT44B6"],
                "kind": "score",
                "confirmatory": False,
                "candidate_count": 1,
                "candidate_ids": [CANDIDATE_ID],
                "inputs": ["exact selected-train 21-movie fit allowlist only"],
                "method": (
                    "one immutable 5-frame/128-hidden/4-head/4-block/seed-104729 lineage head; "
                    "V32 encoder frozen; AdamW lr=1e-4 weight_decay=0.01 batch=16 for exactly 50 "
                    "epochs; fixed final epoch only; no checkpoint selection; inherited V32 loss; "
                    "nested movie-group isolation"
                ),
                "outputs": ["paired complete-graph movie ledger", "control receipts", "read audit"],
                "decision_criterion": {
                    "type": "effect_threshold",
                    "metric": "adjusted edge Jaccard delta",
                    "operator": ">=",
                    "threshold": 0.005,
                },
                "pass_rule": "adjusted delta >=0.005 and raw-edge delta >=0 and node-recall delta >=0, with all controls/integrity/runtime gates PASS",
                "fail_rule": "stop exact realization on first failed ordered gate; no tuning or same-identity retry",
                "dependencies": ["N3 GO", "authenticated exporter", "whole-pipeline runtime envelope strictly below 8 hours"],
                "status": "blocked_dependency",
            }
        ],
        "controls": {
            "negative": [
                "group-preserving temporal circular shift",
                "candidate-parent identity permutation",
                "cross-track identity break",
            ],
            "positive": [
                "frozen synthetic planted-lineage recovery",
                "byte-exact deterministic V32 baseline replay",
            ],
        },
        "selection": {
            "hyperparameter_candidates": 1,
            "seed_candidates": 1,
            "candidate_inventory": {
                "path": "dag_prompts/2026-08-23-model-clade-renewal-v1/m1_v32_multiframe/CANDIDATE_INVENTORY.json",
                "sha256": inventory_sha,
                "count": 1,
            },
            "visual_selection": False,
            "all_choices_inner_only": True,
        },
        "multiplicity": {
            "alpha": 0.05,
            "procedure": "none_descriptive",
            "family": ["H1-M1-FIT44B6"],
            "stop_at_first_failure": True,
            "exchangeability_justification": "No p-value or confirmatory population claim; one exact deterministic threshold gate is descriptive exploratory evidence.",
        },
        "inference": {"claims_p_value": False},
        "uncertainty": {
            "method": "cluster_bootstrap",
            "resampling_unit": "movie_identity",
            "repetitions": 10000,
            "confidence": 0.95,
            "recompute_full_metric": True,
        },
        "stopping": {
            "interim_looks": 1,
            "alpha_spending": None,
            "scientific": ["first ordered graph/safety gate failure"],
            "integrity": ["source/config/data/firewall/control hash mismatch"],
            "resource": ["LMT resource guard or runtime envelope failure"],
        },
        "execution": {
            "code_sha256": code_sha,
            "environment_sha256": environment_sha,
            "fixed_seeds": [FIXED_SEED],
            "max_trials": 1,
            "outcome_adaptive_changes": False,
            "retain_all_runs": True,
        },
        "decision": {
            "promotion_rule": "all ordered exploratory fit-only graph, safety, controls, firewall, and integrity gates pass unchanged",
            "successor_requires_new_identity": True,
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    auth = authenticate_contract(ROOT)
    config = FrozenConfig()
    config.validate()
    env = environment_receipt()
    model = model_identity(build_frozen_model())
    controls = run_control_contract()
    runtime = benchmark_synthetic_forward()
    fixture = synthetic_export(auth["fit_identities"][0])
    code_sha = sha256_file(MODULE)

    config_sha = write_json(OUT / "FROZEN_CONFIG.json", {
        "schema": "celltrack-alpha.m1-v32-five-frame-config.v1",
        **asdict(config),
        "config_sha256": config.sha256,
        "selection_candidates": {"architectures": 1, "configurations": 1, "seeds": 1},
        "threshold_or_decoder_frozen_here": False,
    })
    write_json(OUT / "EXPORT_SCHEMA.json", {
        "schema": "celltrack-alpha.m1-v32-five-frame-export-schema.v1",
        "candidate_id": CANDIDATE_ID,
        "window": {"frames": 5, "consecutive": True, "adjacent_pair_matrices": 4},
        "arrays": {
            "node_ids.npy": {"dtype": "int64", "shape": ["sum_nodes"]},
            "frame_slots.npy": {"dtype": "int8", "shape": ["sum_nodes"], "range": [0, 4]},
            "frames.npy": {"dtype": "int64", "shape": ["sum_nodes"]},
            "coords_um.npy": {"dtype": "float32", "shape": ["sum_nodes", 3]},
            "node_features_f32.npy": {"dtype": "float32", "shape": ["sum_nodes", 32]},
            "node_counts.npy": {"dtype": "int64", "shape": [5]},
            "raw_pair_logits_f32.npy": {"dtype": "float32", "shape": ["sum_adjacent_pair_products"]},
            "raw_pair_shapes.npy": {"dtype": "int64", "shape": [4, 2]},
        },
        "raw_pair_values": "exact pre-softmax/pre-sigmoid V32 pair logits in C row-major order",
        "firewall": "in-memory arrays only; exact selected-train allowlist authenticated before write; no path discovery",
        "fixture": fixture,
    })
    write_json(RUNS / "SOURCE_AUTH_RECEIPT.json", {
        "schema": "celltrack-alpha.m1-v32-source-auth.v1",
        "candidate_id": CANDIDATE_ID,
        "status": "PASS",
        "fit_manifest_sha256": auth["fit_manifest_sha256"],
        "fit_identity_count": len(auth["fit_identities"]),
        "fit_identity_set_sha256": __import__("hashlib").sha256("\n".join(auth["fit_identities"]).encode()).hexdigest(),
        "source_sha256": auth["source_sha256"],
        "opened_paths": list(auth["opened_paths"]),
        "protected_outcomes_opened": False,
    })
    write_json(RUNS / "CONTROL_RECEIPT.json", controls)
    write_json(RUNS / "LOCAL_RUNTIME_RECEIPT.json", runtime)
    write_json(RUNS / "ENVIRONMENT_RECEIPT.json", env)
    write_json(RUNS / "MODEL_IDENTITY.json", {
        "schema": "celltrack-alpha.m1-v32-model-identity.v1",
        "candidate_id": CANDIDATE_ID,
        **model,
        "initialization": "torch.manual_seed(104729), exact source state-dict digest algorithm",
        "scientific_model_trained": False,
    })
    inventory = {
        "schema": "celltrack-alpha.m1-v32-candidate-inventory.v1",
        "candidate_count": 1,
        "candidates": [{
            "candidate_id": CANDIDATE_ID,
            "config_sha256": config.sha256,
            "config_file_sha256": config_sha,
            "code_sha256": code_sha,
            "model_state_sha256": model["state_sha256"],
            "environment_sha256": env["sha256"],
            "seed": FIXED_SEED,
        }],
        "selection_forbidden": True,
    }
    inventory_sha = write_json(OUT / "CANDIDATE_INVENTORY.json", inventory)
    plan = child_plan(code_sha=code_sha, environment_sha=env["sha256"], inventory_sha=inventory_sha)
    plan_sha = write_json(OUT / "M1_EXPLORATORY_CHILD_PLAN.json", plan)

    package_records = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in {"PACKAGE_INVENTORY.json", "M1_N3_HANDOFF.json", "M1_EXPLORATORY_CHILD_PLAN.validation.json"}:
            continue
        package_records.append({
            "path": str(path.relative_to(OUT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    package = {
        "schema": "celltrack-alpha.m1-v32-local-package-inventory.v1",
        "candidate_id": CANDIDATE_ID,
        "records": package_records,
        "source_code": {
            str(MODULE.relative_to(ROOT)): code_sha,
            str(BUILDER.relative_to(ROOT)): sha256_file(BUILDER),
            str(TEST.relative_to(ROOT)): sha256_file(TEST),
        },
    }
    package_sha = write_json(OUT / "PACKAGE_INVENTORY.json", package)
    validation_path = OUT / "M1_EXPLORATORY_CHILD_PLAN.validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
    validation_pass = bool(
        validation
        and not validation.get("errors")
        and validation.get("valid") is True
        and validation.get("plan", {}).get("sha256") == plan_sha
    )
    predicates = {
        "exact_sources_and_fit_allowlist_authenticated": True,
        "selected_train_exporter_and_schema_frozen": True,
        "single_architecture_configuration_seed": True,
        "deterministic_synthetic_forward": controls["checks"]["deterministic_replay"],
        "planted_lineage_positive_control": controls["checks"]["planted_lineage_positive_recovery_exact"],
        "three_negative_controls_distinct": all(
            controls["checks"][key]
            for key in (
                "temporal_shift_is_output_distinct",
                "parent_permutation_is_output_distinct",
                "identity_break_is_output_distinct",
            )
        ),
        "exploratory_child_plan_validated": validation_pass,
        "whole_pipeline_remote_upper_strictly_below_eight_hours": False,
    }
    handoff = {
        "schema": "celltrack-alpha.m1-n3-handoff.v1",
        "node": "N3",
        "candidate_id": CANDIDATE_ID,
        "status": "PARK_RUNTIME_IDENTITY",
        "predicates": predicates,
        "predicate_pass_count": sum(predicates.values()),
        "predicate_count": len(predicates),
        "reason": runtime["reason"],
        "advance_allowed": False,
        "plan_sha256": plan_sha,
        "plan_validation_sha256": sha256_file(validation_path) if validation_path.is_file() else None,
        "code_sha256": code_sha,
        "environment_sha256": env["sha256"],
        "config_sha256": config.sha256,
        "model_state_sha256": model["state_sha256"],
        "package_inventory_sha256": package_sha,
        "scientific_outcomes_opened": False,
        "protected_outcomes_opened": False,
        "external_compute_used": False,
        "actions_not_taken": [
            "no selected-train scientific fit or graph scoring",
            "no protected confirmation/test/leaderboard/submission/scorer access",
            "no Kaggle push or T4 launch",
            "no hyperparameter/threshold/architecture/checkpoint/seed comparison",
        ],
        "unblock_requirement": (
            "Authenticate a selected-train per-movie frame/node/pair census and a matched local-to-T4 "
            "whole-pipeline calibration, then run the one frozen multi-scale ladder without changing this candidate."
        ),
    }
    write_json(RUNS / "M1_N3_HANDOFF.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

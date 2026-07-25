import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_temporal_rank_oof_graph_gate",
    ROOT / "scripts/evaluate_temporal_rank_oof_graph_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_hash_pinned_and_fail_closed():
    contract = MODULE.load_contract(
        ROOT / "reports/g02/temporal_rank_oof_graph_gate_contract_v1.json",
        MODULE.DEFAULT_CONTRACT_SHA256,
    )
    assert contract["scope"]["promotion_authorized"] is False
    assert contract["scope"]["validation12_access"] is False
    assert contract["scope"]["cv57_access"] is False
    with pytest.raises(MODULE.TemporalOOFGraphGateError, match="SHA-256"):
        MODULE.load_contract(
            ROOT / "reports/g02/temporal_rank_oof_graph_gate_contract_v1.json",
            "0" * 64,
        )


def test_held_fold_partitions_score_each_identity_once(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "_fold",
        lambda identity, fold_count: int(identity.rsplit("-", 1)[1]) % fold_count,
    )
    identities = [f"train:6bba-video-{index}" for index in range(12)]
    partitions = MODULE.held_fold_partitions(identities, 5)
    held = [identity for _, values in partitions for identity in values]
    assert sorted(held) == sorted(identities)
    for fit, values in partitions:
        assert not set(fit).intersection(values)


def test_validate_fit_scope_rejects_any_sealed_overlap():
    train = [f"train:6bba_{index:08x}" for index in range(69)]
    train += [f"train:44b6_{index:08x}" for index in range(36)]
    validation = [f"train:6bba_v{index:07d}" for index in range(12)]
    scope = {
        "fit_only": True,
        "audit_used": False,
        "held_out_used": False,
        "test_used": False,
        "train_video_ids": train,
        "validation_video_ids": validation,
    }
    assert len(MODULE.validate_fit_scope(scope, dict(scope), set())) == 69
    with pytest.raises(
        MODULE.TemporalOOFGraphGateError, match="overlaps sealed"
    ):
        MODULE.validate_fit_scope(scope, dict(scope), {train[0]})


def test_graph_gate_requires_material_adjusted_gain():
    control = {
        "adj_edge_jaccard": 0.88,
        "edge_jaccard": 0.89,
        "node_recall": 0.99,
    }
    gate = {
        "minimum_adjusted_edge_jaccard_delta": 0.005,
        "require_nonnegative_raw_edge_jaccard_delta": True,
        "minimum_node_recall_delta": 0.0,
    }
    low = {
        "adj_edge_jaccard": 0.8849,
        "edge_jaccard": 0.90,
        "node_recall": 0.99,
    }
    high = {
        "adj_edge_jaccard": 0.8851,
        "edge_jaccard": 0.90,
        "node_recall": 0.99,
    }
    assert (
        MODULE.graph_gate_decision(control, low, gate)[
            "counterfactual_gate_status"
        ]
        == "PARK"
    )
    result = MODULE.graph_gate_decision(control, high, gate)
    assert result["counterfactual_gate_status"] == "PASS-GRAPH"
    assert result["promotion_authorized"] is False


def test_candidate_manifest_passes_current_governance():
    gate_spec = importlib.util.spec_from_file_location(
        "model_candidate_gate",
        ROOT / "src/celltrack_alpha/model_candidate_gate.py",
    )
    gate_module = importlib.util.module_from_spec(gate_spec)
    assert gate_spec.loader is not None
    gate_spec.loader.exec_module(gate_module)
    candidate = json.loads(
        (
            ROOT
            / "reports/g02/model_candidate_temporal_rank_oof_graph_diagnostic_v1.json"
        ).read_text()
    )
    policy = json.loads(
        (ROOT / "reports/g02/compute_governance_v1.json").read_text()
    )
    result = gate_module.validate_candidate(candidate, policy)
    assert result["training_scope"] == "fit_only"
    assert result["holdout_access"] is False

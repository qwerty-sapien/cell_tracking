from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.m1_v32_multiframe_lineage import (
    CANDIDATE_ID,
    EXPECTED_FIT_MANIFEST_SHA256,
    FIXED_SEED,
    FrozenConfig,
    M1ContractError,
    assert_input_path_allowed,
    assert_selected_identity,
    authenticate_contract,
    build_frozen_model,
    export_five_frame_window,
    model_identity,
    run_control_contract,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture_arrays(nodes: int = 3):
    ids = [np.arange(slot * nodes, (slot + 1) * nodes, dtype=np.int64) for slot in range(5)]
    coords = [
        np.stack(
            [
                np.arange(nodes, dtype=np.float32),
                np.arange(nodes, dtype=np.float32) + slot,
                np.arange(nodes, dtype=np.float32) * 2 + slot,
            ],
            axis=1,
        )
        for slot in range(5)
    ]
    features = []
    for slot in range(5):
        value = np.zeros((nodes, 32), dtype=np.float32)
        value[:, 0] = np.arange(nodes, dtype=np.float32)
        value[:, 1] = slot
        features.append(value)
    logits = []
    for slot in range(4):
        value = np.full((nodes, nodes), -4.0, dtype=np.float32)
        np.fill_diagonal(value, 4.0 + slot)
        logits.append(value)
    return ids, coords, features, logits


def test_exact_selected_train_and_source_contract_authenticates() -> None:
    contract = authenticate_contract(ROOT)
    assert contract["fit_manifest_sha256"] == EXPECTED_FIT_MANIFEST_SHA256
    assert len(contract["fit_identities"]) == 21
    assert len(set(contract["fit_identities"])) == 21
    assert set(contract["source_sha256"]) == {"train", "predict", "cache", "transformer"}
    assert all(identity.startswith("train:44b6_") for identity in contract["fit_identities"])
    assert not any("6bba" in path.lower() for path in contract["opened_paths"])
    assert not any("competition" in path.lower() for path in contract["opened_paths"])


@pytest.mark.parametrize(
    "identity",
    [
        "train:6bba_deadbeef",
        "competition_test:44b6_24264f12",
        "train:44b6_not_selected",
        "submission:44b6_24264f12",
    ],
)
def test_forbidden_or_unselected_identity_fails_closed(identity: str) -> None:
    contract = authenticate_contract(ROOT)
    with pytest.raises(M1ContractError):
        assert_selected_identity(identity, contract)


def test_prospective_path_firewall_never_opens_forbidden_namespace() -> None:
    contract = authenticate_contract(ROOT)
    identity = contract["fit_identities"][0]
    token = identity.split(":", 1)[1]
    assert assert_input_path_allowed(Path("/input/train") / f"{token}.zarr", contract) == identity
    for path in (
        Path("/input/competition_test") / f"{token}.zarr",
        Path("/input/train/6bba_deadbeef.zarr"),
        Path("/input/submission") / f"{token}.zarr",
        Path("/input/train/44b6_not_selected.zarr"),
    ):
        with pytest.raises(M1ContractError):
            assert_input_path_allowed(path, contract)


def test_export_is_deterministic_schema_bound_and_pre_activation(tmp_path: Path) -> None:
    contract = authenticate_contract(ROOT)
    identity = contract["fit_identities"][0]
    ids, coords, features, logits = _fixture_arrays()
    first = export_five_frame_window(
        project_root=ROOT,
        identity=identity,
        frames=[10, 11, 12, 13, 14],
        node_ids=ids,
        coords_um=coords,
        node_features=features,
        raw_pair_logits=logits,
        output_dir=tmp_path / "first",
    )
    second = export_five_frame_window(
        project_root=ROOT,
        identity=identity,
        frames=[10, 11, 12, 13, 14],
        node_ids=ids,
        coords_um=coords,
        node_features=features,
        raw_pair_logits=logits,
        output_dir=tmp_path / "second",
    )
    assert first["schema"] == "celltrack-alpha.m1-v32-five-frame-export.v1"
    assert first["node_feature_dim"] == 32
    assert first["frame_count"] == 5
    assert first["raw_values_are_pre_activation"] is True
    assert first["protected_outcomes_opened"] is False
    assert first["files"] == second["files"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    stored = np.load(tmp_path / "first" / "node_features_f32.npy", allow_pickle=False)
    assert stored.dtype == np.float32
    assert stored.shape == (15, 32)
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["opened_paths"] == list(contract["opened_paths"])


def test_export_rejects_shape_nonfinite_frame_and_output_drift(tmp_path: Path) -> None:
    contract = authenticate_contract(ROOT)
    identity = contract["fit_identities"][0]
    ids, coords, features, logits = _fixture_arrays()
    bad_features = list(features)
    bad_features[2] = np.zeros((3, 31), dtype=np.float32)
    with pytest.raises(M1ContractError):
        export_five_frame_window(
            project_root=ROOT,
            identity=identity,
            frames=[10, 11, 12, 13, 14],
            node_ids=ids,
            coords_um=coords,
            node_features=bad_features,
            raw_pair_logits=logits,
            output_dir=tmp_path / "shape",
        )
    bad_logits = list(logits)
    bad_logits[0] = bad_logits[0].copy()
    bad_logits[0][0, 0] = np.nan
    with pytest.raises(M1ContractError):
        export_five_frame_window(
            project_root=ROOT,
            identity=identity,
            frames=[10, 11, 12, 13, 14],
            node_ids=ids,
            coords_um=coords,
            node_features=features,
            raw_pair_logits=bad_logits,
            output_dir=tmp_path / "nonfinite",
        )
    with pytest.raises(M1ContractError):
        export_five_frame_window(
            project_root=ROOT,
            identity=identity,
            frames=[10, 11, 13, 14, 15],
            node_ids=ids,
            coords_um=coords,
            node_features=features,
            raw_pair_logits=logits,
            output_dir=tmp_path / "frames",
        )


def test_one_fixed_config_seed_and_model_identity() -> None:
    config = FrozenConfig()
    config.validate()
    assert config.candidate_id == CANDIDATE_ID
    assert config.seed == FIXED_SEED
    assert (config.frame_count, config.node_feature_dim) == (5, 32)
    assert (config.hidden_dim, config.attention_heads, config.transformer_blocks) == (128, 4, 4)
    assert config.training_scope == "lineage_head_only_v32_encoder_frozen"
    assert (config.optimizer, config.learning_rate, config.weight_decay) == ("AdamW", 1e-4, 0.01)
    assert (config.epochs, config.batch_size) == (50, 16)
    assert config.checkpoint_policy == "fixed_final_epoch_50_no_selection"
    with pytest.raises(M1ContractError):
        FrozenConfig(seed=FIXED_SEED + 1).validate()
    first = model_identity(build_frozen_model())
    second = model_identity(build_frozen_model())
    assert first == second
    assert first["parameter_count"] > 0


def test_planted_lineage_and_negative_control_contracts_pass() -> None:
    result = run_control_contract()
    assert result["status"] == "PASS"
    assert result["scientific_outcome_opened"] is False
    assert result["synthetic_only"] is True
    assert result["positive_recovery_fraction"] == 1.0
    assert result["parent_permutation_recovery_fraction"] < 1.0
    assert all(result["checks"].values())
    assert len(set(result["output_sha256"].values())) == 4


def test_frozen_package_plan_and_park_handoff_are_self_consistent() -> None:
    base = ROOT / "dag_prompts/2026-08-23-model-clade-renewal-v1/m1_v32_multiframe"
    inventory = json.loads((base / "PACKAGE_INVENTORY.json").read_text())
    for record in inventory["records"]:
        path = base / record["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    for relative, expected in inventory["source_code"].items():
        assert sha256_file(ROOT / relative) == expected

    validation_path = base / "M1_EXPLORATORY_CHILD_PLAN.validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["warnings"] == []
    plan_path = base / "M1_EXPLORATORY_CHILD_PLAN.json"
    assert validation["plan"]["sha256"] == sha256_file(plan_path)

    handoff = json.loads((base / "runs/M1_N3_HANDOFF.json").read_text())
    assert handoff["status"] == "PARK_RUNTIME_IDENTITY"
    assert handoff["advance_allowed"] is False
    assert handoff["scientific_outcomes_opened"] is False
    assert handoff["protected_outcomes_opened"] is False
    assert handoff["external_compute_used"] is False
    assert handoff["predicates"]["exploratory_child_plan_validated"] is True
    assert handoff["predicates"]["whole_pipeline_remote_upper_strictly_below_eight_hours"] is False

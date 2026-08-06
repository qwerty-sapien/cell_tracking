from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from scripts import g03_r4_bank_extension as r4


def test_divisions_off_guard_uses_submission_edge_api_and_rejects_forks_merges() -> None:
    linear = SimpleNamespace(
        edges=[
            SimpleNamespace(source_id=1, target_id=2),
            SimpleNamespace(source_id=2, target_id=3),
        ]
    )
    r4._assert_divisions_off(linear)
    fork = SimpleNamespace(
        edges=[
            SimpleNamespace(source_id=1, target_id=2),
            SimpleNamespace(source_id=1, target_id=3),
        ]
    )
    with pytest.raises(r4.R4BankError, match="contains a division"):
        r4._assert_divisions_off(fork)
    merge = SimpleNamespace(
        edges=[
            SimpleNamespace(source_id=1, target_id=3),
            SimpleNamespace(source_id=2, target_id=3),
        ]
    )
    with pytest.raises(r4.R4BankError, match="incoming merge"):
        r4._assert_divisions_off(merge)


def _write_json(path: Path, value: dict) -> str:
    path.write_bytes(r4.canonical_json(value))
    return r4.sha256(path)


def _population() -> r4.Population:
    development = tuple(f"train:6bba_{index:08x}" for index in range(69))
    held_fit = tuple(f"train:44b6_{index + 1000:08x}" for index in range(36))
    hidden = (
        "test:44b6_00001000",
        "test:44b6_00001001",
        "test:6bba_00001002",
        "test:6bba_00001003",
    )
    fold_sizes = (15, 14, 14, 13, 13)
    folds: dict[str, list[dict[str, str]]] = {}
    cursor = 0
    for fold, size in enumerate(fold_sizes):
        folds[str(fold)] = [
            {
                "identity": identity,
                "group_id": f"group-{cursor + offset:03d}",
                "registry_stratum": "6bba|medium|moderate",
            }
            for offset, identity in enumerate(development[cursor : cursor + size])
        ]
        cursor += size

    def role_rows(values: tuple[str, ...]) -> list[dict[str, str]]:
        return [
            {
                "identity": identity,
                "group_id": f"role-{index:03d}-{identity}",
                "registry_stratum": "target-blind",
            }
            for index, identity in enumerate(values)
        ]

    population_sha = "a" * 64
    manifest_sha = "b" * 64
    amendment_sha = "c" * 64
    manifest = {
        "schema": r4.POPULATION_SCHEMA,
        "status": "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population_sha256": population_sha,
        "roles": {
            "development_nested_oof": {
                "video_count": 69,
                "videos": role_rows(development),
            },
            "held_fit_one_shot_confirmation": {
                "video_count": 36,
                "videos": role_rows(held_fit),
            },
            "sealed_hidden_test_probe": {
                "video_count": 4,
                "videos": role_rows(hidden),
            },
        },
        "outer_folds": folds,
        "constraints": {
            "division_labels_used_for_split": False,
            "all_association_banks_use_one_R1_scientific_configuration": True,
            "development_bank_count_required": 69,
            "held_fit_bank_count_required": 36,
            "hidden_test_bank_count_required": 4,
            "contaminated_audit20_bank_count_required": 0,
        },
    }
    amendment = {
        "schema": r4.POPULATION_AMENDMENT_SCHEMA,
        "status": "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population_manifest": {
            "file_sha256": manifest_sha,
            "population_sha256": population_sha,
        },
        "association_bank_scope": {
            "development_bank_count": 69,
            "held_fit_bank_count": 36,
            "hidden_test_bank_count": 4,
            "contaminated_audit20_bank_count": 0,
            "development_and_held_fit_bank_hashes_sealed_before_development_scoring": True,
            "hidden_test_inference_only_after_local_confirmation_gate": True,
        },
    }
    return r4.validate_population_values(
        manifest,
        amendment,
        manifest_sha256=manifest_sha,
        amendment_sha256=amendment_sha,
        expected_population_sha256=population_sha,
    )


STUB_R1 = r'''from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

zarr = SimpleNamespace(open_array=lambda *args, **kwargs: None)

class ZipStore:
    def __init__(self, path, mode="r"):
        self.path = path
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False

class Edge:
    def __init__(self, source, target):
        self.source_id = source
        self.target_id = target

class Graph:
    def __init__(self, dataset):
        self.dataset = dataset
        self.nodes = {1: object(), 2: object()}
        self.edges = [Edge(1, 2)]

def configure_runtime(contract):
    return None

def load_trackastra(contract):
    return object()

def load_adapter(contract):
    return object()

def run_video(identity, model, adapter, store, contract):
    nodes = np.asarray([[1, 0, 1, 2, 3], [2, 1, 2, 3, 4]], dtype=np.float64)
    edges = np.asarray([[1, 2, 0.9, 1.0]], dtype=np.float64)
    return nodes, edges, {
        "identity": identity,
        "node_count": 2,
        "candidate_edge_count": 1,
        "inference_calls": 1,
        "peak_tokens": 2,
        "elapsed_seconds": 0.001,
        "wrfeat": {"all_rows_nonzero": True, "image_conditioned_channel_count": 12},
    }

def graph_from_arrays(identity, nodes, edges, contract):
    return Graph(identity.split(":", 1)[1])

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_association_bank(path, identity, nodes, edges):
    Path(path).write_bytes(("target-free-bank:" + identity).encode("ascii"))
    return _sha(path)

def write_graph(path, graph, contract):
    Path(path).write_bytes(("divisions-off-parent:" + graph.dataset).encode("ascii"))
    return _sha(path)
'''


def _binding_bundle(tmp_path: Path) -> tuple[r4.R1Binding, dict[str, Path]]:
    implementation = tmp_path / "r1.py"
    implementation.write_text(STUB_R1, encoding="ascii")
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    archive = tmp_path / "ctc.zip"
    archive.write_bytes(b"archive")
    raw = tmp_path / "raw.zip"
    raw.write_bytes(b"synthetic-raw-not-opened")
    source_root = tmp_path / "trackastra-source"
    source_root.mkdir()
    (source_root / "model.py").write_bytes(b"frozen source")
    source_sha = r4.tree_sha256(source_root)
    implementation_sha = r4.sha256(implementation)
    adapter_sha = r4.sha256(adapter)
    adapter_state_sha = "1" * 64
    model_sha = r4.sha256(model)
    archive_sha = r4.sha256(archive)
    held_identities = [f"train:44b6_{index + 1000:08x}" for index in range(36)]
    contract = {
        "schema": r4.R1_CONTRACT_SCHEMA,
        "status": "SEALED_BEFORE_HELD_FIT_SCORE_READ",
        "implementation_sha256": implementation_sha,
        "scope": {
            "family": "44b6",
            "identity_count": 36,
            "identities": held_identities,
            "training": False,
            "gpu": False,
            "kaggle": False,
            "cv57": False,
            "submission": False,
        },
        "model_config": {"feat_dim": 12, "window": 4},
        "adapter": {
            "path": str(adapter),
            "file_sha256": adapter_sha,
            "state_tensor_sha256": adapter_state_sha,
            "architecture": "synthetic exact adapter",
            "training_steps": 0,
            "seed": 20260801,
        },
        "compiler": {
            "divisions": False,
            "association_threshold": 0.35,
        },
        "runtime": {"device": "cpu", "hard_wall_seconds": 600},
        "inputs": {
            "trackastra_checkpoint": {
                "model_sha256": model_sha,
                "archive_sha256": archive_sha,
            },
            "trackastra_source_tree_sha256": source_sha,
            "raw_archive": {
                "path": str(raw),
                "sha256": r4.sha256(raw),
                "bytes": raw.stat().st_size,
            },
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_sha = _write_json(contract_path, contract)
    amendment = {
        "schema": r4.R1_AMENDMENT_SCHEMA,
        "status": "SEALED_BEFORE_HELD_FIT_SCORE_READ",
        "amends_contract_sha256": contract_sha,
        "scientific_configuration_changed": False,
        "integrity": {
            "ground_truth_access": False,
            "held_fit_score_read": False,
        },
    }
    amendment_path = tmp_path / "amendment.json"
    amendment_sha = _write_json(amendment_path, amendment)
    preflight = {
        "schema": r4.R1_PREFLIGHT_SCHEMA,
        "status": "PASS_SCORE_BLIND_PREFLIGHT",
        "contract": {"sha256": contract_sha},
        "implementation_sha256": implementation_sha,
        "adapter": {
            "file_sha256": adapter_sha,
            "state_tensor_sha256": adapter_state_sha,
        },
        "trackastra_checkpoint": {
            "model_sha256": model_sha,
            "archive_sha256": archive_sha,
        },
        "trackastra_source_tree_sha256": source_sha,
        "gt_access": False,
        "held_fit_score_read": False,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = _write_json(preflight_path, preflight)
    held_bank_sha = "2" * 64
    held_parent_sha = "3" * 64
    inference_seal = {
        "schema": r4.R1_INFERENCE_SEAL_SCHEMA,
        "status": "SEALED_ALL_36_BEFORE_SCORE_READ",
        "contract": {"sha256": contract_sha},
        "population": {
            "family": "44b6",
            "identity_count": 36,
            "identities": held_identities,
        },
        "aggregate": {
            "association_bank_sha256": held_bank_sha,
            "fragment_sha256": held_parent_sha,
            "predicted_forks": 0,
        },
        "gt_access": False,
        "held_fit_score_read": False,
    }
    inference_seal_path = tmp_path / "inference_seal.json"
    inference_seal_sha = _write_json(inference_seal_path, inference_seal)
    promotion = {
        "schema": r4.PROMOTION_SCHEMA,
        "status": "R1_FINALIZED_AND_HASHES_PROMOTED_BY_ROOT",
        "r1_finalized": True,
        "artifacts": {
            "contract_sha256": contract_sha,
            "decision_amendment_sha256": amendment_sha,
            "implementation_sha256": implementation_sha,
            "adapter_file_sha256": adapter_sha,
            "adapter_state_tensor_sha256": adapter_state_sha,
            "checkpoint_model_sha256": model_sha,
            "checkpoint_archive_sha256": archive_sha,
            "trackastra_source_tree_sha256": source_sha,
            "preflight_sha256": preflight_sha,
            "inference_seal_sha256": inference_seal_sha,
            "held_fit_association_bank_sha256": held_bank_sha,
            "held_fit_parent_graph_sha256": held_parent_sha,
            "validated_result_sha256": "4" * 64,
            "validation_receipt_sha256": "5" * 64,
        },
        "gt_access_by_r4": False,
        "score_read_by_r4": False,
        "score_values_in_receipt": False,
    }
    promotion_path = tmp_path / "promotion.json"
    promotion_sha = _write_json(promotion_path, promotion)
    paths = {
        "promotion_path": promotion_path,
        "contract_path": contract_path,
        "amendment_path": amendment_path,
        "implementation_path": implementation,
        "adapter_path": adapter,
        "checkpoint_model_path": model,
        "checkpoint_archive_path": archive,
        "source_root": source_root,
        "preflight_path": preflight_path,
        "inference_seal_path": inference_seal_path,
    }
    binding = r4.validate_r1_binding(
        promotion_sha256=promotion_sha,
        **paths,
    )
    return binding, paths


def test_population_contract_is_exact_and_target_blind() -> None:
    population = _population()
    assert len(population.development) == 69
    assert len(population.held_fit) == 36
    assert len(population.hidden) == 4
    assert sorted(population.development_fold_by_identity.values()).count(0) == 15
    assert set(population.development_fold_by_identity) == set(population.development)


def test_promotion_binds_score_free_r1_chain_and_rejects_drift(tmp_path: Path) -> None:
    binding, paths = _binding_bundle(tmp_path)
    assert binding.held_fit_association_bank_sha256 == "2" * 64
    assert binding.held_fit_parent_graph_sha256 == "3" * 64
    paths["implementation_path"].write_text(STUB_R1 + "\n# drift\n", encoding="ascii")
    with pytest.raises(r4.R4BankError, match="R1 implementation: SHA-256 mismatch"):
        r4.validate_r1_binding(
            promotion_sha256=r4.sha256(paths["promotion_path"]),
            **paths,
        )


def test_dev69_atomic_seal_resume_and_tamper_detection(tmp_path: Path) -> None:
    binding, paths = _binding_bundle(tmp_path)
    population = _population()
    output = tmp_path / "dev69"
    seal = r4.run_role(
        role="development_nested_oof",
        output=output,
        binding=binding,
        population=population,
        implementation_path=paths["implementation_path"],
        adapter_path=paths["adapter_path"],
        checkpoint_model_path=paths["checkpoint_model_path"],
        checkpoint_archive_path=paths["checkpoint_archive_path"],
        source_root=paths["source_root"],
        authorization_receipt_sha256="f" * 64,
        authorization_receipt_kind="r4_dev69_preflight",
    )
    seal_path = output / "DEV69_INFERENCE_SEAL.json"
    original_sha = r4.sha256(seal_path)
    original_mtime = seal_path.stat().st_mtime_ns
    assert seal["identity_count"] == 69
    assert seal["gt_access"] is False
    assert seal["score_read"] is False
    assert seal["association_banks_target_free"] is True
    assert seal["parent_graph_mode"] == "DIVISIONS_OFF"
    assert seal["local_105_bank_scope"]["total_video_count"] == 105
    assert seal["local_105_bank_scope"][
        "all_hashes_sealed_before_R4_development_scoring"
    ] is True
    assert len(list((output / "videos").iterdir())) == 69

    time.sleep(0.001)
    resumed = r4.run_role(
        role="development_nested_oof",
        output=output,
        binding=binding,
        population=population,
        implementation_path=paths["implementation_path"],
        adapter_path=paths["adapter_path"],
        checkpoint_model_path=paths["checkpoint_model_path"],
        checkpoint_archive_path=paths["checkpoint_archive_path"],
        source_root=paths["source_root"],
        authorization_receipt_sha256="f" * 64,
        authorization_receipt_kind="r4_dev69_preflight",
    )
    assert resumed == seal
    assert r4.sha256(seal_path) == original_sha
    assert seal_path.stat().st_mtime_ns == original_mtime

    with pytest.raises(r4.R4BankError, match="authorization receipt hash drifted"):
        r4.run_role(
            role="development_nested_oof",
            output=output,
            binding=binding,
            population=population,
            implementation_path=paths["implementation_path"],
            adapter_path=paths["adapter_path"],
            checkpoint_model_path=paths["checkpoint_model_path"],
            checkpoint_archive_path=paths["checkpoint_archive_path"],
            source_root=paths["source_root"],
            authorization_receipt_sha256="0" * 64,
            authorization_receipt_kind="r4_dev69_preflight",
        )

    first = output / "videos" / r4.slug(population.development[0])
    (first / "association_bank.npz").write_bytes(b"tampered")
    with pytest.raises(r4.R4BankError, match="bank hash drifted"):
        r4.run_role(
            role="development_nested_oof",
            output=output,
            binding=binding,
            population=population,
            implementation_path=paths["implementation_path"],
            adapter_path=paths["adapter_path"],
            checkpoint_model_path=paths["checkpoint_model_path"],
            checkpoint_archive_path=paths["checkpoint_archive_path"],
            source_root=paths["source_root"],
            authorization_receipt_sha256="f" * 64,
            authorization_receipt_kind="r4_dev69_preflight",
        )


def test_partial_video_never_gets_overwritten(tmp_path: Path) -> None:
    binding, paths = _binding_bundle(tmp_path)
    population = _population()
    output = tmp_path / "partial"
    partial = output / "videos" / r4.slug(population.development[0])
    partial.mkdir(parents=True)
    bank = partial / "association_bank.npz"
    bank.write_bytes(b"preserve-partial")
    before = r4.sha256(bank)
    with pytest.raises(r4.R4BankError, match="partial video state"):
        r4.run_role(
            role="development_nested_oof",
            output=output,
            binding=binding,
            population=population,
            implementation_path=paths["implementation_path"],
            adapter_path=paths["adapter_path"],
            checkpoint_model_path=paths["checkpoint_model_path"],
            checkpoint_archive_path=paths["checkpoint_archive_path"],
            source_root=paths["source_root"],
            authorization_receipt_sha256="f" * 64,
            authorization_receipt_kind="r4_dev69_preflight",
        )
    assert r4.sha256(bank) == before


def test_hidden4_requires_score_free_local_confirmation_before_any_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _paths = _binding_bundle(tmp_path)
    population = _population()
    dev_seal_sha = "6" * 64
    receipt = {
        "schema": r4.LOCAL_CONFIRMATION_SCHEMA,
        "status": "PASS_LOCAL_CONFIRMATION_GATE_HIDDEN4_AUTHORIZED",
        "hidden4_inference_authorized": True,
        "population_sha256": population.population_sha256,
        "r1_binding_sha256": binding.binding_sha256,
        "development_seal_sha256": dev_seal_sha,
        "selection_complete_before_confirmation": True,
        "parent_and_overlay_graph_hashes_sealed": True,
        "held_fit_target_access_count": 1,
        "score_values_in_receipt": False,
        "test_raw_members_opened": False,
        "hidden4_started": False,
        "deployable_config_sha256": "7" * 64,
        "held_fit_parent_graph_sha256": "8" * 64,
        "held_fit_overlay_graph_sha256": "9" * 64,
    }
    r4.validate_local_confirmation_value(
        receipt,
        receipt_sha256="d" * 64,
        binding=binding,
        population=population,
        development_seal_sha256=dev_seal_sha,
    )
    bad = dict(receipt)
    bad["hidden4_inference_authorized"] = False
    with pytest.raises(r4.R4BankError, match="authorization is absent"):
        r4.validate_local_confirmation_value(
            bad,
            receipt_sha256="d" * 64,
            binding=binding,
            population=population,
            development_seal_sha256=dev_seal_sha,
        )

    missing = tmp_path / "NO_LOCAL_CONFIRMATION.json"
    namespace = argparse.Namespace(
        command="infer-hidden4",
        local_confirmation=missing,
        local_confirmation_sha256="e" * 64,
    )

    class FakeParser:
        def parse_args(self) -> argparse.Namespace:
            return namespace

    monkeypatch.setattr(r4, "parser", lambda: FakeParser())
    monkeypatch.setattr(
        r4,
        "_binding_from_args",
        lambda _args: pytest.fail("R1 binding touched before local-confirmation gate"),
    )
    with pytest.raises(r4.R4BankError, match="missing regular file"):
        r4.main()

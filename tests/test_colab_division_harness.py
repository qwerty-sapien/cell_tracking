from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.colab_division_harness import (
    ARTIFACT_SCHEMA,
    AtomicCheckpointStore,
    HarnessError,
    SwapGuard,
    adjudicate_resume_runtime,
    append_colab_session,
    append_colab_session_start,
    capture_training_state,
    colab_hours_by_gpu_model,
    publish_artifact,
    read_colab_ledger,
    reconcile_interrupted_colab_sessions,
    restore_training_state,
    validate_objective_contract,
    verify_artifact,
    verify_serialized_training_state,
    write_idempotent_json,
)


SHA = "a" * 64


def _log_bytes() -> bytes:
    return (
        json.dumps(
            {
                "schema": "celltrack-alpha.full-stage-log.v1",
                "run_id": "test",
                "event": "config_echo",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def test_immutable_artifact_is_idempotent_and_tamper_evident(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_id": "toy-artifact",
    }
    first = publish_artifact(
        root,
        manifest=manifest,
        payloads={"nested/payload.bin": b"exact bytes"},
    )
    second = publish_artifact(
        root,
        manifest=manifest,
        payloads={"nested/payload.bin": b"exact bytes"},
    )
    assert first["tree_sha256"] == second["tree_sha256"]
    (root / "nested/payload.bin").write_bytes(b"tampered")
    with pytest.raises(HarnessError, match="mismatch"):
        verify_artifact(root)


def test_idempotent_json_is_create_once_without_hard_links(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert write_idempotent_json(path, {"value": 1})[0] is True
    assert write_idempotent_json(path, {"value": 1})[0] is False
    with pytest.raises(HarnessError, match="conflicts"):
        write_idempotent_json(path, {"value": 2})


def test_complete_state_restore_recovers_rng_and_semantic_digest() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = model(x).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    provenance = {
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
    }
    _, digest, payload = capture_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        sampler_state={"epoch": 4, "cursor": 19, "permutation_sha256": SHA},
        global_step=31,
        epoch=4,
        provenance=provenance,
    )
    self_test = verify_serialized_training_state(
        payload,
        expected_state_digest=digest,
        expected_provenance=provenance,
    )
    assert self_test["status"] == "PASS_SEMANTIC_DIGEST_BEFORE_COMPUTE"
    with pytest.raises(HarnessError, match="semantic digest mismatch"):
        verify_serialized_training_state(
            payload,
            expected_state_digest="f" * 64,
            expected_provenance=provenance,
        )
    expected_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    with torch.no_grad():
        model.weight.add_(100)
    restored = restore_training_state(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_provenance=provenance,
    )
    observed_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert observed_random == expected_random
    assert restored == {
        "sampler": {"epoch": 4, "cursor": 19, "permutation_sha256": SHA},
        "global_step": 31,
        "epoch": 4,
        "state_digest": digest,
    }


def test_checkpoint_pointer_is_monotonic(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path / "state")
    provenance = {
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
    }
    store.publish_bytes(
        global_step=10,
        publication_index=0,
        checkpoint=b"checkpoint-10",
        log_jsonl=_log_bytes(),
        state_digest="4" * 64,
        provenance=provenance,
        session={"id": "one"},
    )
    assert store.verify_latest()["pointer"]["global_step"] == 10
    with pytest.raises(HarnessError, match="regress"):
        store.publish_bytes(
            global_step=9,
            publication_index=1,
            checkpoint=b"checkpoint-9",
            log_jsonl=_log_bytes(),
            state_digest="5" * 64,
            provenance=provenance,
            session={"id": "one"},
        )


def test_swap_guard_records_zero_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = iter([1000, 1000, 1000])
    monkeypatch.setattr(
        "scripts.colab_division_harness.linux_swapout_bytes", lambda: next(samples)
    )
    guard = SwapGuard()
    guard.sample()
    guard.sample()
    assert guard.receipt()["growth_bytes"] == 0
    assert guard.receipt()["samples"] == 2


def test_cross_gpu_resume_is_logged_without_a_bit_identical_claim() -> None:
    producer = {
        "devices": [{"name": "NVIDIA T4", "capability": [7, 5]}],
        "torch": "one",
    }
    assigned = {
        "devices": [{"name": "NVIDIA L4", "capability": [8, 9]}],
        "torch": "one",
    }
    with pytest.raises(HarnessError, match="not allowed"):
        adjudicate_resume_runtime(producer, assigned, allow_gpu_transition=False)
    receipt = adjudicate_resume_runtime(producer, assigned, allow_gpu_transition=True)
    assert receipt["status"] == "LOGGED_RUNTIME_TRANSITION_NO_BIT_IDENTICAL_CLAIM"
    assert receipt["exact_runtime_match"] is False
    assert receipt["same_gpu_model_and_capability"] is False


def test_colab_ledger_never_commingles_kaggle_hours(tmp_path: Path) -> None:
    ledger = tmp_path / "COLAB_HOURS_LEDGER.jsonl"
    append_colab_session_start(
        ledger,
        session_id="session-one",
        started_utc="2026-08-05T01:00:00+00:00",
        assigned_gpu_model="Tesla T4",
        hardware_fingerprint_sha256="1" * 64,
        receipt_sha256="2" * 64,
    )
    record = append_colab_session(
        ledger,
        session_id="session-one",
        started_utc="2026-08-05T01:00:00+00:00",
        ended_utc="2026-08-05T02:30:00+00:00",
        assigned_gpu_model="Tesla T4",
        hardware_fingerprint_sha256="1" * 64,
        status="COMPLETE",
        receipt_sha256=SHA,
    )
    assert record["session_wall_hours"] == 1.5
    assert record["gpu_model_bucket"] == "NVIDIA T4"
    assert record["kaggle_gpu_hours_included"] == 0.0
    append_colab_session_start(
        ledger,
        session_id="session-two",
        started_utc="2026-08-05T03:00:00+00:00",
        assigned_gpu_model="NVIDIA L4",
        hardware_fingerprint_sha256="3" * 64,
        receipt_sha256="4" * 64,
    )
    append_colab_session(
        ledger,
        session_id="session-two",
        started_utc="2026-08-05T03:00:00+00:00",
        ended_utc="2026-08-05T05:00:00+00:00",
        assigned_gpu_model="NVIDIA L4",
        hardware_fingerprint_sha256="3" * 64,
        status="COMPLETE",
        receipt_sha256="5" * 64,
    )
    assert colab_hours_by_gpu_model(ledger) == {"NVIDIA L4": 2.0, "NVIDIA T4": 1.5}
    rows = read_colab_ledger(ledger)
    assert all("cumulative_colab_hours" not in row for row in rows)
    rows[-1]["total_colab_hours"] = 3.5
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="ascii")
    with pytest.raises(HarnessError, match="naively sums"):
        read_colab_ledger(ledger)


def test_interrupted_session_is_reconciled_as_device_specific_upper_bound(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "COLAB_HOURS_LEDGER.jsonl"
    append_colab_session_start(
        ledger,
        session_id="interrupted-t4",
        started_utc="2026-08-05T01:00:00+00:00",
        assigned_gpu_model="NVIDIA T4",
        hardware_fingerprint_sha256="1" * 64,
        receipt_sha256="2" * 64,
    )
    receipts = reconcile_interrupted_colab_sessions(
        ledger,
        before_utc="2026-08-05T03:00:00+00:00",
        exclude_session_id="new-session",
        latest_producer_session={
            "session_id": "interrupted-t4",
            "checkpoint_started_utc": "2026-08-05T02:00:00+00:00",
        },
    )
    assert len(receipts) == 1
    assert receipts[0]["duration_basis"] == "UPPER_BOUND_LAST_CHECKPOINT_PLUS_CADENCE"
    assert receipts[0]["duration_uncertainty_seconds"] == 600.0
    assert receipts[0]["session_wall_hours"] == pytest.approx(7.0 / 6.0)
    assert colab_hours_by_gpu_model(ledger) == {"NVIDIA T4": pytest.approx(7.0 / 6.0)}


def test_pull_requires_latest_and_recomputes_semantic_digest(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path / "remote")
    with pytest.raises(HarnessError, match="cannot read"):
        store.pull_latest_verified(tmp_path / "local")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    provenance = {
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
    }
    _, digest, payload = capture_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        sampler_state={"epoch": 0, "cursor": 0, "permutation_sha256": SHA},
        global_step=0,
        epoch=0,
        provenance=provenance,
    )
    store.publish_bytes(
        global_step=0,
        publication_index=0,
        checkpoint=payload,
        log_jsonl=_log_bytes(),
        state_digest=digest,
        provenance=provenance,
        session={"id": "bootstrap"},
    )
    pulled = store.pull_latest_verified(tmp_path / "local")
    assert pulled["semantic"]["state_digest"] == digest
    assert pulled["pointer"]["manifest_sha256"] == store.verify_latest()["pointer"]["manifest_sha256"]


def test_training_objective_is_fail_closed_until_taxonomy() -> None:
    pending = {"status": "PENDING_APNCC_V32_TAXONOMY"}
    with pytest.raises(HarnessError, match="pending"):
        validate_objective_contract(pending)
    frozen = {
        "status": "FROZEN_AFTER_APNCC_V32_TAXONOMY",
        "governance": {
            "owner_scope_classification": "NEW_LONGER_CONTEXT_ARCHITECTURE_OUTSIDE_WEIGHT_REFINEMENT_RULE",
            "owner_scope_receipt_sha256": "ac4270f896168b47945bf661d8aed0fb833510d3ec2b37159e7f19e1828381a6",
        },
        "deployment_gate_preregistration": {
            "sha256": "a1c4b2f97aa5d28f88f5d2d4e35a2d9f35db6b09934920ecb9e982521c74b634"
        },
        "taxonomy": {"sha256": SHA},
        "training": {
            "target": "parent_event_detection",
            "data_source": "REAL_BIOHUB_COMPETITION_TRAIN_ONLY",
            "architecture_class": "NEW_LONGER_TEMPORAL_CONTEXT",
            "temporal_context_frames": 5,
            "division_loss_weight": 4.0,
            "daughter_assignment": "EXISTING_CONDITIONAL_PAIR_HEAD_FROZEN",
            "initialization": "PARTIAL_SHAPE_COMPATIBLE_SPATIAL_ONLY",
            "v32_weights_trainable": False,
            "resume_hardware_policy": "ALLOW_LOGGED_GPU_TRANSITION_NO_BIT_IDENTICAL_CLAIM",
            "checkpoint_interval_seconds": 600,
            "session_target_hours": 5.0,
            "entrypoint_sha256": "e" * 64,
        },
        "deployment_gate": {
            "k": 22,
            "minimum_tp": 5,
            "maximum_fp": 17,
            "minimum_precision": 5 / 22,
        },
        "metrics": {
            "primary": ["precision_at_22", "average_precision", "ap_lift"],
            "minimum_ap_lift": 8462.0,
            "auroc_is_a_gate": False,
        },
        "evaluation": {
            "population_manifest_sha256": "c" * 64,
            "scoring_code_sha256": "d" * 64,
            "scores_sealed_before_labels": True,
            "post_score_threshold_or_top_k_sweep": False,
        },
    }
    assert validate_objective_contract(frozen) == frozen
    frozen["governance"]["owner_scope_classification"] = "WEIGHT_REFINEMENT"
    with pytest.raises(HarnessError, match="new-architecture"):
        validate_objective_contract(frozen)

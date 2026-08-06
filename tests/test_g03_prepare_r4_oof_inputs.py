from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "g03_prepare_r4_oof_inputs.py"
SPEC = importlib.util.spec_from_file_location("g03_prepare_r4_oof_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)

import g03_division_decode_manifest as config_builder


CONFIGURATIONS = [asdict(row) for row in config_builder.build_configurations()]


def write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.decode.canonical_json_bytes(value))
    return MODULE.sha256_file(path)


def write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return MODULE.sha256_file(path)


def role_rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "identity": f"{prefix}{index:08x}",
            "group_id": f"group-{prefix.replace(':', '-')}-{index:03d}",
            "registry_stratum": "synthetic-target-blind-stratum",
        }
        for index in range(count)
    ]


def population_manifest() -> dict[str, Any]:
    development = role_rows("train:6bba_", 69)
    held = role_rows("train:44b6_", 36)
    hidden = role_rows("test:hidden_", 4)
    audit = role_rows("train:audit_", 20)
    folds = {str(index): [] for index in range(5)}
    for index, row in enumerate(development):
        folds[str(index % 5)].append(dict(row))
    roles = {
        MODULE.orchestrator.DEVELOPMENT_ROLE: {
            "description": "synthetic development",
            "video_count": 69,
            "target_access": "score_presealed_only",
            "videos": development,
        },
        MODULE.orchestrator.HELD_ROLE: {
            "description": "synthetic held fit",
            "video_count": 36,
            "target_access": "one_shot_only",
            "videos": held,
        },
        MODULE.orchestrator.HIDDEN_ROLE: {
            "description": "synthetic hidden",
            "video_count": 4,
            "target_access": "locked",
            "videos": hidden,
        },
        MODULE.orchestrator.AUDIT_ROLE: {
            "description": "synthetic excluded audit",
            "video_count": 20,
            "target_access": "forbidden",
            "videos": audit,
        },
    }
    constraints = {
        "division_labels_used_for_split": False,
        "all_association_banks_use_one_R1_scientific_configuration": True,
        "development_bank_count_required": 69,
        "held_fit_bank_count_required": 36,
        "hidden_test_bank_count_required": 4,
        "contaminated_audit20_bank_count_required": 0,
    }
    value = {
        "schema": MODULE.POPULATION_SCHEMA,
        "status": "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "seed": "synthetic-real-schema-v1",
        "sources": {},
        "roles": roles,
        "outer_folds": folds,
        "constraints": constraints,
    }
    value["population_sha256"] = MODULE.decode.canonical_sha256(
        {
            "seed": value["seed"],
            "roles": roles,
            "outer_folds": folds,
            "constraints": constraints,
        }
    )
    return value


def artifact_aggregate(rows: list[dict[str, Any]], field: str) -> str:
    return MODULE._aggregate_artifacts(rows, field)


@pytest.fixture
def sealed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    configuration = {
        "schema": MODULE.decode.CONFIG_SCHEMA,
        "status": "LOCKED_BEFORE_OUTCOMES",
        "axes": {},
        "invariants": {},
        "active_configuration_count": 18_144,
        "configuration_count_including_null": 18_145,
        "configurations": CONFIGURATIONS,
        "configuration_sha256": MODULE.decode.canonical_sha256(CONFIGURATIONS),
    }
    configuration_path = tmp_path / "contracts" / "configuration.json"
    configuration_sha = write_json(configuration_path, configuration)
    protocol = {
        "schema": MODULE.PROTOCOL_SCHEMA,
        "status": "SEALED_BEFORE_ANY_G03_DIVISION_OUTCOME",
        "configuration_manifest": {
            "path": str(configuration_path),
            "file_sha256": configuration_sha,
            "configuration_sha256": configuration["configuration_sha256"],
            "active_configuration_count": 18_144,
            "configuration_count_including_null": 18_145,
        },
    }
    protocol_path = tmp_path / "contracts" / "protocol.json"
    protocol_sha = write_json(protocol_path, protocol)
    population = population_manifest()
    population_path = tmp_path / "contracts" / "population.json"
    population_sha = write_json(population_path, population)
    amendment = {
        "schema": MODULE.POPULATION_AMENDMENT_SCHEMA,
        "status": "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "amends": {"protocol_sha256": protocol_sha},
        "population_manifest": {
            "file_sha256": population_sha,
            "population_sha256": population["population_sha256"],
        },
        "association_bank_scope": {
            "required_total_video_count": 109,
            "development_bank_count": 69,
            "held_fit_bank_count": 36,
            "hidden_test_bank_count": 4,
            "contaminated_audit20_bank_count": 0,
            "development_and_held_fit_bank_hashes_sealed_before_development_scoring": True,
        },
    }
    amendment_path = tmp_path / "contracts" / "amendment.json"
    amendment_sha = write_json(amendment_path, amendment)

    held_ids = MODULE._role_identities(population, MODULE.orchestrator.HELD_ROLE)
    r1_root = tmp_path / "r1"
    contract_sha = hashlib.sha256(b"synthetic-r1-contract").hexdigest()
    held_rows = []
    for ordinal, identity in enumerate(held_ids, start=1):
        slug = identity.replace(":", "__", 1)
        bank_sha = write_bytes(
            r1_root / "association_banks" / f"{slug}.npz",
            f"target-free-held-bank:{identity}\n".encode("ascii"),
        )
        parent_sha = write_bytes(
            r1_root / "fragments" / f"{slug}.csv",
            f"target-free-held-parent:{identity}\n".encode("ascii"),
        )
        held_rows.append(
            {
                "schema": "celltrack-alpha.g03-r1-inference-video.v1",
                "identity": identity,
                "ordinal": ordinal,
                "contract_sha256": contract_sha,
                "association_bank_sha256": bank_sha,
                "association_bank_complete": True,
                "fragment_sha256": parent_sha,
                "predicted_forks": 0,
                "gt_access": False,
                "held_fit_score_read": False,
            }
        )
    # The promoted R1 seal preserves inference order, while the R4 population is
    # canonical identity order.  Exact membership, not incidental row order, is
    # the scientific invariant.
    sealed_held_rows = held_rows[::2] + held_rows[1::2]
    r1_seal = {
        "schema": MODULE.R1_INFERENCE_SEAL_SCHEMA,
        "status": "SEALED_ALL_36_BEFORE_SCORE_READ",
        "contract": {"path": "synthetic-contract.json", "sha256": contract_sha},
        "preflight": {"path": "synthetic-preflight.json", "sha256": "1" * 64},
        "population": {
            "family": "44b6",
            "identity_count": 36,
            "identities": [row["identity"] for row in sealed_held_rows],
        },
        "adapter": {"training_steps": 0},
        "compiler": {"divisions": False},
        "aggregate": {
            "association_bank_sha256": artifact_aggregate(
                sealed_held_rows, "association_bank_sha256"
            ),
            "fragment_sha256": artifact_aggregate(
                sealed_held_rows, "fragment_sha256"
            ),
            "predicted_forks": 0,
        },
        "videos": sealed_held_rows,
        "gt_access": False,
        "held_fit_score_read": False,
        "gpu_hours": 0.0,
    }
    r1_seal_path = r1_root / "INFERENCE_SEAL.json"
    r1_seal_sha = write_json(r1_seal_path, r1_seal)
    monkeypatch.setattr(MODULE, "PROMOTED_R1_INFERENCE_SHA256", r1_seal_sha)

    binding = {
        "contract_sha256": contract_sha,
        "decision_amendment_sha256": "2" * 64,
        "implementation_sha256": "3" * 64,
        "adapter_file_sha256": "4" * 64,
        "adapter_state_tensor_sha256": "5" * 64,
        "checkpoint_model_sha256": "6" * 64,
        "checkpoint_archive_sha256": "7" * 64,
        "trackastra_source_tree_sha256": "8" * 64,
        "preflight_sha256": "9" * 64,
        "inference_seal_sha256": r1_seal_sha,
        "held_fit_association_bank_sha256": r1_seal["aggregate"][
            "association_bank_sha256"
        ],
        "held_fit_parent_graph_sha256": r1_seal["aggregate"]["fragment_sha256"],
        "validated_result_sha256": "a" * 64,
        "validation_receipt_sha256": "b" * 64,
        "promotion_sha256": "c" * 64,
        "scientific_configuration_sha256": "d" * 64,
    }
    binding["binding_sha256"] = MODULE._binding_sha256(binding)
    development_ids = MODULE._role_identities(
        population, MODULE.orchestrator.DEVELOPMENT_ROLE
    )
    development_root = tmp_path / "development"
    development_rows = []
    fold_by_identity = {
        row["identity"]: int(fold)
        for fold, rows in population["outer_folds"].items()
        for row in rows
    }
    for identity in development_ids:
        slug = identity.replace(":", "__", 1)
        bank_sha = write_bytes(
            development_root / "videos" / slug / "association_bank.npz",
            f"target-free-development-bank:{identity}\n".encode("ascii"),
        )
        parent_sha = write_bytes(
            development_root / "videos" / slug / "DIVISIONS_OFF_PARENT.csv",
            f"target-free-development-parent:{identity}\n".encode("ascii"),
        )
        development_rows.append(
            {
                "schema": "celltrack-alpha.g03-r4-bank-extension-video.v1",
                "identity": identity,
                "role": MODULE.orchestrator.DEVELOPMENT_ROLE,
                "outer_fold": fold_by_identity[identity],
                "population_sha256": population["population_sha256"],
                "population_manifest_sha256": population_sha,
                "r1_binding_sha256": binding["binding_sha256"],
                "r1_contract_sha256": binding["contract_sha256"],
                "r1_scientific_configuration_sha256": binding[
                    "scientific_configuration_sha256"
                ],
                "association_bank_sha256": bank_sha,
                "parent_graph_sha256": parent_sha,
                "association_bank_target_free": True,
                "parent_graph_mode": "DIVISIONS_OFF",
                "gt_access": False,
                "score_read": False,
            }
        )
    dev_bank_aggregate = artifact_aggregate(
        development_rows, "association_bank_sha256"
    )
    dev_parent_aggregate = artifact_aggregate(development_rows, "parent_graph_sha256")
    development_seal = {
        "schema": MODULE.DEV_SEAL_SCHEMA,
        "status": "SEALED_EXACT_DEV69_BEFORE_R4_GT_OR_SCORER_ACCESS",
        "role": MODULE.orchestrator.DEVELOPMENT_ROLE,
        "population_sha256": population["population_sha256"],
        "population_manifest_sha256": population_sha,
        "population_amendment_sha256": amendment_sha,
        "identity_count": 69,
        "identities": list(development_ids),
        "outer_fold_sizes": {
            fold: sum(value == int(fold) for value in fold_by_identity.values())
            for fold in MODULE.orchestrator.FOLD_IDS
        },
        "r1_binding": binding,
        "local_confirmation_gate_sha256": None,
        "aggregate": {
            "association_bank_sha256": dev_bank_aggregate,
            "parent_graph_sha256": dev_parent_aggregate,
            "predicted_forks": 0,
        },
        "local_105_bank_scope": {
            "held_fit_video_count": 36,
            "held_fit_association_bank_sha256": binding[
                "held_fit_association_bank_sha256"
            ],
            "held_fit_parent_graph_sha256": binding["held_fit_parent_graph_sha256"],
            "development_video_count": 69,
            "development_association_bank_sha256": dev_bank_aggregate,
            "development_parent_graph_sha256": dev_parent_aggregate,
            "total_video_count": 105,
            "all_hashes_sealed_before_R4_development_scoring": True,
        },
        "videos": development_rows,
        "association_banks_target_free": True,
        "parent_graph_mode": "DIVISIONS_OFF",
        "sealed_before_R4_gt_or_scorer_access": True,
        "gt_access": False,
        "score_read": False,
        "gpu_hours": 0.0,
    }
    development_seal_path = development_root / "DEV69_SEAL.json"
    development_seal_sha = write_json(development_seal_path, development_seal)

    apncc_report_path = tmp_path / "apncc" / "report.json"
    apncc_report_sha = write_json(
        apncc_report_path,
        {"schema": "historical-report-opaque-to-builder", "outcomes": "not parsed"},
    )
    apncc_pairs = []
    for identity in held_ids:
        slug = identity.replace(":", "__", 1)
        path = apncc_report_path.parent / "control_fragments" / f"{slug}.csv"
        digest = write_bytes(path, f"apncc-control:{identity}\n".encode("ascii"))
        apncc_pairs.append((identity, digest))
    monkeypatch.setattr(MODULE, "APNCC_REPORT_SHA256", apncc_report_sha)
    monkeypatch.setattr(
        MODULE, "APNCC_AGGREGATE_SHA256", MODULE._fragment_aggregate(apncc_pairs)
    )

    return {
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha,
        "configuration_path": configuration_path,
        "configuration_sha256": configuration_sha,
        "population_path": population_path,
        "population_sha256": population_sha,
        "amendment_path": amendment_path,
        "amendment_sha256": amendment_sha,
        "development_seal_path": development_seal_path,
        "development_seal_sha256": development_seal_sha,
        "r1_inference_seal_path": r1_seal_path,
        "r1_inference_seal_sha256": r1_seal_sha,
        "apncc_report_path": apncc_report_path,
        "population": population,
        "development_seal": development_seal,
    }


def pretarget_call(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    return MODULE.prepare_pretarget_input(
        **{key: value for key, value in inputs.items() if key.endswith("_path") or key.endswith("_sha256")},
        output=output,
    )


def make_plan(inputs: dict[str, Any], pretarget_path: Path, output: Path) -> tuple[str, dict[str, Any]]:
    pretarget = json.loads(pretarget_path.read_text(encoding="ascii"))
    coverage, coverage_sha, r1 = MODULE.orchestrator.verify_exact_fit105_coverage(
        pretarget["fit105_coverage"],
        population=inputs["population"],
        base=pretarget_path.parent,
    )
    runner_path = ROOT / MODULE.orchestrator.RUNNER_SOURCE_RELATIVE
    contracts = {
        "protocol": pretarget["protocol"],
        "configuration_manifest": pretarget["configuration_manifest"],
        "population_manifest": pretarget["population_manifest"],
        "population_amendment": pretarget["population_amendment"],
    }
    payload = {
        "input_manifest": {
            "path": str(pretarget_path),
            "sha256": MODULE.sha256_file(pretarget_path),
        },
        "contracts": contracts,
        "protocol_identity_sha256": inputs["protocol_sha256"],
        "population_identity_sha256": inputs["population"]["population_sha256"],
        "population_amendment_sha256": inputs["amendment_sha256"],
        "configuration_count": 18_145,
        "configuration_names_sha256": "e" * 64,
        "execution_source_bindings": {
            "runner": {
                "path": str(runner_path.resolve()),
                "sha256": MODULE.sha256_file(runner_path),
            }
        },
        "selection_metric": "synthetic",
        "inner_tie_order": ["synthetic"],
        "deployable_rule": "synthetic",
        "deployable_metric_repooling": False,
        "fit105_coverage": {
            "video_count": 105,
            "development_count": 69,
            "held_fit_count": 36,
            "coverage_sha256": coverage_sha,
            "r1_scientific_configuration": r1,
            "records": list(coverage),
        },
        "outer_folds": list(
            MODULE.orchestrator.build_fixed_fold_plan(inputs["population"])
        ),
        "forbidden_roles": [MODULE.orchestrator.AUDIT_ROLE],
        "target_or_outcome_read": False,
        "score_executed": False,
        "hidden_test_locked_until_passed_local_gate": True,
    }
    artifact = MODULE.orchestrator._artifact(
        MODULE.orchestrator.PRETARGET_PLAN_SCHEMA,
        "SEALED_PRE_TARGET_NO_SCORE_OR_GT_READ",
        payload,
    )
    return write_json(output, artifact), artifact


def test_pretarget_builder_emits_exact_real_schema_local105(
    sealed_inputs: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "PRETARGET_INPUT.json"
    report = pretarget_call(sealed_inputs, output)
    manifest = json.loads(output.read_text(encoding="ascii"))
    sealed_r1 = json.loads(
        sealed_inputs["r1_inference_seal_path"].read_text(encoding="ascii")
    )
    assert tuple(sealed_r1["population"]["identities"]) != MODULE._role_identities(
        sealed_inputs["population"], MODULE.orchestrator.HELD_ROLE
    )
    assert report["video_count"] == 105
    assert manifest["schema"] == MODULE.orchestrator.PRETARGET_INPUT_SCHEMA
    records = manifest["fit105_coverage"]["records"]
    assert len(records) == 105
    assert sum(row["role"] == MODULE.orchestrator.DEVELOPMENT_ROLE for row in records) == 69
    assert sum(row["role"] == MODULE.orchestrator.HELD_ROLE for row in records) == 36
    assert all(
        row["parent_graph"] == row["reference_graph"]
        for row in records
        if row["role"] == MODULE.orchestrator.DEVELOPMENT_ROLE
    )
    assert all(
        "control_fragments" in row["reference_graph"]["path"]
        for row in records
        if row["role"] == MODULE.orchestrator.HELD_ROLE
    )
    assert not any(
        row["role"] in {MODULE.orchestrator.AUDIT_ROLE, MODULE.orchestrator.HIDDEN_ROLE}
        for row in records
    )


def test_pretarget_builder_fails_closed_on_tamper_population_and_overwrite(
    sealed_inputs: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "PRETARGET_INPUT.json"
    pretarget_call(sealed_inputs, output)
    with pytest.raises(MODULE.PrepareInputsError, match="overwrite"):
        pretarget_call(sealed_inputs, output)

    first = sealed_inputs["development_seal"]["videos"][0]["identity"]
    bank = (
        sealed_inputs["development_seal_path"].parent
        / "videos"
        / first.replace(":", "__", 1)
        / "association_bank.npz"
    )
    bank.write_bytes(bank.read_bytes() + b"tamper")
    with pytest.raises(MODULE.PrepareInputsError, match="SHA-256 mismatch"):
        pretarget_call(sealed_inputs, tmp_path / "TAMPERED.json")
    assert not (tmp_path / "TAMPERED.json").exists()

    population = json.loads(sealed_inputs["population_path"].read_text())
    population["roles"][MODULE.orchestrator.DEVELOPMENT_ROLE]["videos"].pop()
    population_sha = write_json(tmp_path / "contracts" / "population-drift.json", population)
    drifted = dict(sealed_inputs)
    drifted["population_path"] = tmp_path / "contracts" / "population-drift.json"
    drifted["population_sha256"] = population_sha
    with pytest.raises(MODULE.PrepareInputsError, match="population"):
        pretarget_call(drifted, tmp_path / "POPULATION_DRIFT.json")
    assert not (tmp_path / "POPULATION_DRIFT.json").exists()


def test_runner_builder_binds_plan_final_sources_and_opaque_archive(
    sealed_inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE.runner,
        "SCOREALL_DEPENDENCIES",
        ("src/celltrack_alpha/official_eval.py",),
    )
    pretarget = tmp_path / "PRETARGET_INPUT.json"
    pretarget_call(sealed_inputs, pretarget)
    plan_path = tmp_path / "PRETARGET_PLAN.json"
    plan_sha, _plan = make_plan(sealed_inputs, pretarget, plan_path)
    archive = tmp_path / "competition.zip"
    archive_sha = write_bytes(archive, b"opaque competition archive bytes; no member read")
    output = tmp_path / "RUNNER_INPUT.json"
    report = MODULE.prepare_runner_input(
        repo_root=ROOT,
        pretarget_plan_path=plan_path,
        pretarget_plan_sha256=plan_sha,
        development_seal_path=sealed_inputs["development_seal_path"],
        development_seal_sha256=sealed_inputs["development_seal_sha256"],
        competition_archive_path=archive,
        competition_archive_sha256=archive_sha,
        output=output,
    )
    manifest = json.loads(output.read_text(encoding="ascii"))
    assert manifest["schema"] == MODULE.runner.RUNNER_INPUT_SCHEMA
    assert manifest["local105_inventory"] == {
        "pretarget_plan_sha256": plan_sha,
        "coverage_sha256": report["coverage_sha256"],
        "video_count": 105,
    }
    assert report["runner_sha256"] == MODULE.orchestrator.PROMOTED_RUNNER_SHA256
    assert set(manifest["scorer_dependencies"]) == set(
        MODULE.runner.SCOREALL_DEPENDENCIES
    )
    assert set(manifest["one_pass_dependencies"]) == set(
        MODULE.runner.ONE_PASS_DEPENDENCIES
    )
    for relative, reference in manifest["one_pass_dependencies"].items():
        assert reference == {
            "path": str((ROOT / relative).resolve()),
            "sha256": MODULE.sha256_file(ROOT / relative),
        }
    assert manifest["competition_archive"]["sha256"] == archive_sha


def test_runner_builder_rejects_runner_and_archive_substitution(
    sealed_inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE.runner,
        "SCOREALL_DEPENDENCIES",
        ("src/celltrack_alpha/official_eval.py",),
    )
    pretarget = tmp_path / "PRETARGET_INPUT.json"
    pretarget_call(sealed_inputs, pretarget)
    plan_path = tmp_path / "PRETARGET_PLAN.json"
    _plan_sha, plan = make_plan(sealed_inputs, pretarget, plan_path)
    counterfeit = write_bytes(tmp_path / "counterfeit-runner.py", b"counterfeit\n")
    plan["payload"]["execution_source_bindings"]["runner"] = {
        "path": str(tmp_path / "counterfeit-runner.py"),
        "sha256": counterfeit,
    }
    plan["payload_sha256"] = MODULE.decode.canonical_sha256(plan["payload"])
    substituted_plan = tmp_path / "SUBSTITUTED_PLAN.json"
    substituted_sha = write_json(substituted_plan, plan)
    archive = tmp_path / "competition.zip"
    archive_sha = write_bytes(archive, b"opaque archive")
    with pytest.raises(MODULE.PrepareInputsError, match="runner"):
        MODULE.prepare_runner_input(
            repo_root=ROOT,
            pretarget_plan_path=substituted_plan,
            pretarget_plan_sha256=substituted_sha,
            development_seal_path=sealed_inputs["development_seal_path"],
            development_seal_sha256=sealed_inputs["development_seal_sha256"],
            competition_archive_path=archive,
            competition_archive_sha256=archive_sha,
            output=tmp_path / "BAD_RUNNER_INPUT.json",
        )
    assert not (tmp_path / "BAD_RUNNER_INPUT.json").exists()

    valid_plan = tmp_path / "VALID_PLAN.json"
    valid_plan_sha, _ = make_plan(sealed_inputs, pretarget, valid_plan)
    with pytest.raises(MODULE.PrepareInputsError, match="archive.*mismatch"):
        MODULE.prepare_runner_input(
            repo_root=ROOT,
            pretarget_plan_path=valid_plan,
            pretarget_plan_sha256=valid_plan_sha,
            development_seal_path=sealed_inputs["development_seal_path"],
            development_seal_sha256=sealed_inputs["development_seal_sha256"],
            competition_archive_path=archive,
            competition_archive_sha256="f" * 64,
            output=tmp_path / "BAD_ARCHIVE_INPUT.json",
        )
    assert not (tmp_path / "BAD_ARCHIVE_INPUT.json").exists()


def test_cli_has_no_hidden_or_audit_input_surface() -> None:
    help_text = MODULE.parser().format_help()
    assert "hidden" not in help_text.lower()
    assert "audit" not in help_text.lower()

#!/usr/bin/env python3
"""Build the two target-blind input manifests for the sealed G-03 R4 OOF run.

This program hashes target-free banks, parent graphs, sealed metadata, source
files, and the competition archive as opaque bytes.  It never imports the
scorer, opens an archive member, reads ground truth, or interprets a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

try:
    import g03_division_decode as decode
    import g03_division_oof_orchestrator as orchestrator
    import g03_division_oof_runner as runner
except ModuleNotFoundError:  # pragma: no cover - repository-root imports
    from scripts import g03_division_decode as decode
    from scripts import g03_division_oof_orchestrator as orchestrator
    from scripts import g03_division_oof_runner as runner


PRETARGET_STATUS = "SEALED_BEFORE_DEVELOPMENT_SCORING"
RUNNER_STATUS = "SEALED_BEFORE_R4_DEVELOPMENT_SCORING"
PROTOCOL_SCHEMA = "celltrack-alpha.g03-r4-division-protocol-contract.v1"
POPULATION_SCHEMA = "celltrack-alpha.g03-r4-division-population.v1"
POPULATION_AMENDMENT_SCHEMA = (
    "celltrack-alpha.g03-r4-division-population-amendment.v1"
)
DEV_SEAL_SCHEMA = "celltrack-alpha.g03-r4-dev69-bank-seal.v1"
R1_INFERENCE_SEAL_SCHEMA = "celltrack-alpha.g03-r1-inference-seal.v1"
APNCC_REPORT_SHA256 = orchestrator.APNCC_REFERENCE["report_sha256"]
APNCC_AGGREGATE_SHA256 = orchestrator.APNCC_REFERENCE[
    "aggregate_graph_sha256"
]
PROMOTED_R1_INFERENCE_SHA256 = (
    "c9cf2f7c8a63fecb4d5c822b5f93538f8df609c6577c68245d9f25d4d4c67ea1"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
R1_BINDING_FIELDS = {
    "contract_sha256",
    "decision_amendment_sha256",
    "implementation_sha256",
    "adapter_file_sha256",
    "adapter_state_tensor_sha256",
    "checkpoint_model_sha256",
    "checkpoint_archive_sha256",
    "trackastra_source_tree_sha256",
    "preflight_sha256",
    "inference_seal_sha256",
    "held_fit_association_bank_sha256",
    "held_fit_parent_graph_sha256",
    "validated_result_sha256",
    "validation_receipt_sha256",
    "promotion_sha256",
    "scientific_configuration_sha256",
    "binding_sha256",
}


class PrepareInputsError(RuntimeError):
    """A target-blind input binding failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrepareInputsError(message)


def _sha(value: Any, label: str) -> str:
    result = str(value)
    require(bool(SHA256_RE.fullmatch(result)), f"{label} is not a lowercase SHA-256")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    wanted = _sha(expected, label)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    observed = sha256_file(path)
    require(observed == wanted, f"{label} SHA-256 mismatch: {observed}")
    return observed


def _load_verified_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    _require_hash(path, expected, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepareInputsError(f"cannot read {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{label} keys drifted; missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )


def _absolute_ref(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(digest, str(path))}


def _resolve_ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    require(isinstance(value, Mapping), f"{label} reference is missing")
    _strict(value, {"path", "sha256"}, f"{label} reference")
    raw = Path(str(value["path"]))
    path = raw if raw.is_absolute() else base / raw
    return path.resolve(), _sha(value["sha256"], label)


def _verify_ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    path, digest = _resolve_ref(value, base, label)
    _require_hash(path, digest, label)
    return path, digest


def _write_no_overwrite(path: Path, value: Mapping[str, Any]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    payload = decode.canonical_json_bytes(value)
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PrepareInputsError(f"refusing to overwrite: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _role_identities(population: Mapping[str, Any], role: str) -> tuple[str, ...]:
    roles = population.get("roles")
    record = roles.get(role) if isinstance(roles, Mapping) else None
    rows = record.get("videos") if isinstance(record, Mapping) else None
    require(isinstance(rows, list), f"population role is missing: {role}")
    identities = tuple(
        str(row.get("identity")) for row in rows if isinstance(row, Mapping)
    )
    require(
        len(identities) == len(rows) and len(set(identities)) == len(rows),
        f"population role is malformed: {role}",
    )
    return identities


def _validate_contracts(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    population: Mapping[str, Any],
    population_sha256: str,
    amendment: Mapping[str, Any],
    amendment_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    require(
        protocol.get("schema") == PROTOCOL_SCHEMA
        and protocol.get("status") == "SEALED_BEFORE_ANY_G03_DIVISION_OUTCOME",
        "R4 protocol schema or pre-outcome status drifted",
    )
    config_binding = protocol.get("configuration_manifest")
    require(isinstance(config_binding, Mapping), "protocol configuration binding is missing")
    require(
        config_binding.get("file_sha256") == configuration_sha256,
        "protocol does not bind the configuration file",
    )
    try:
        orchestrator._configuration_names(configuration, protocol)
    except orchestrator.DivisionOOFError as exc:
        raise PrepareInputsError(str(exc)) from exc

    require(
        population.get("schema") == POPULATION_SCHEMA
        and population.get("status")
        == "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population schema or pre-outcome status drifted",
    )
    expected_population_sha = decode.canonical_sha256(
        {
            "seed": population.get("seed"),
            "roles": population.get("roles"),
            "outer_folds": population.get("outer_folds"),
            "constraints": population.get("constraints"),
        }
    )
    require(
        population.get("population_sha256") == expected_population_sha,
        "population membership SHA-256 drifted",
    )
    counts = {
        orchestrator.DEVELOPMENT_ROLE: orchestrator.DEVELOPMENT_COUNT,
        orchestrator.HELD_ROLE: orchestrator.HELD_COUNT,
        orchestrator.HIDDEN_ROLE: orchestrator.HIDDEN_COUNT,
        orchestrator.AUDIT_ROLE: 20,
    }
    identities_by_role = {
        role: _role_identities(population, role) for role in counts
    }
    for role, expected_count in counts.items():
        require(
            len(identities_by_role[role]) == expected_count,
            f"population count drifted: {role}",
        )
    require(
        all(value.startswith("train:6bba_") for value in identities_by_role[orchestrator.DEVELOPMENT_ROLE]),
        "development population is not exact 6bba",
    )
    require(
        all(value.startswith("train:44b6_") for value in identities_by_role[orchestrator.HELD_ROLE]),
        "held-fit population is not exact 44b6",
    )
    role_sets = {role: set(values) for role, values in identities_by_role.items()}
    role_names = tuple(role_sets)
    require(
        all(
            role_sets[left].isdisjoint(role_sets[right])
            for index, left in enumerate(role_names)
            for right in role_names[index + 1 :]
        ),
        "population roles overlap",
    )
    try:
        orchestrator.build_fixed_fold_plan(population)
    except orchestrator.DivisionOOFError as exc:
        raise PrepareInputsError(str(exc)) from exc
    constraints = population.get("constraints")
    require(isinstance(constraints, Mapping), "population constraints are missing")
    require(
        constraints.get("division_labels_used_for_split") is False
        and constraints.get("all_association_banks_use_one_R1_scientific_configuration")
        is True
        and constraints.get("development_bank_count_required")
        == orchestrator.DEVELOPMENT_COUNT
        and constraints.get("held_fit_bank_count_required") == orchestrator.HELD_COUNT
        and constraints.get("contaminated_audit20_bank_count_required") == 0,
        "population target-blind bank constraints drifted",
    )

    require(
        amendment.get("schema") == POPULATION_AMENDMENT_SCHEMA
        and amendment.get("status")
        == "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population amendment schema or pre-outcome status drifted",
    )
    population_binding = amendment.get("population_manifest")
    require(isinstance(population_binding, Mapping), "amendment population binding is missing")
    require(
        population_binding.get("file_sha256") == population_sha256
        and population_binding.get("population_sha256") == expected_population_sha,
        "amendment does not bind the exact population",
    )
    amended = amendment.get("amends")
    require(
        isinstance(amended, Mapping)
        and amended.get("protocol_sha256") == protocol_sha256,
        "population amendment does not bind the exact protocol",
    )
    scope = amendment.get("association_bank_scope")
    require(isinstance(scope, Mapping), "amended bank scope is missing")
    require(
        scope.get("required_total_video_count") == orchestrator.FIT105_COUNT + 4
        and scope.get("development_bank_count") == orchestrator.DEVELOPMENT_COUNT
        and scope.get("held_fit_bank_count") == orchestrator.HELD_COUNT
        and scope.get("contaminated_audit20_bank_count") == 0
        and scope.get("development_and_held_fit_bank_hashes_sealed_before_development_scoring")
        is True,
        "amended local105 bank scope drifted",
    )
    _sha(amendment_sha256, "population amendment")
    return (
        identities_by_role[orchestrator.DEVELOPMENT_ROLE],
        identities_by_role[orchestrator.HELD_ROLE],
    )


def _aggregate_artifacts(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["identity"])):
        digest.update(str(row["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha(row[field], f"{field} aggregate member").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fragment_aggregate(rows: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for identity, file_hash in sorted(rows):
        payload = json.dumps(
            {"identity": identity, "sha256": _sha(file_hash, identity)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _binding_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: value[key] for key in sorted(R1_BINDING_FIELDS - {"binding_sha256"})}
    return hashlib.sha256(decode.canonical_json_bytes(payload)).hexdigest()


def _validate_seals_and_build_coverage(
    *,
    population: Mapping[str, Any],
    population_file_sha256: str,
    amendment_sha256: str,
    development_identities: tuple[str, ...],
    held_identities: tuple[str, ...],
    development_seal_path: Path,
    development_seal_sha256: str,
    r1_inference_seal_path: Path,
    r1_inference_seal_sha256: str,
    apncc_report_path: Path,
) -> dict[str, Any]:
    require(
        r1_inference_seal_sha256 == PROMOTED_R1_INFERENCE_SHA256,
        "R1 inference seal is not the promoted held36 seal",
    )
    development_seal = _load_verified_json(
        development_seal_path, development_seal_sha256, "development seal"
    )
    r1_seal = _load_verified_json(
        r1_inference_seal_path, r1_inference_seal_sha256, "R1 inference seal"
    )
    _require_hash(apncc_report_path, APNCC_REPORT_SHA256, "APNCC 44b6 report")

    require(
        development_seal.get("schema") == DEV_SEAL_SCHEMA
        and development_seal.get("status")
        == "SEALED_EXACT_DEV69_BEFORE_R4_GT_OR_SCORER_ACCESS"
        and development_seal.get("role") == orchestrator.DEVELOPMENT_ROLE
        and development_seal.get("identity_count") == orchestrator.DEVELOPMENT_COUNT
        and tuple(development_seal.get("identities", ())) == development_identities
        and development_seal.get("population_sha256")
        == population.get("population_sha256")
        and development_seal.get("population_manifest_sha256")
        == population_file_sha256
        and development_seal.get("population_amendment_sha256") == amendment_sha256
        and development_seal.get("association_banks_target_free") is True
        and development_seal.get("parent_graph_mode") == "DIVISIONS_OFF"
        and development_seal.get("sealed_before_R4_gt_or_scorer_access") is True
        and development_seal.get("gt_access") is False
        and development_seal.get("score_read") is False,
        "development seal target-blind dev69 contract drifted",
    )
    fold_by_identity = {
        str(row["identity"]): int(fold)
        for fold, rows in population["outer_folds"].items()
        for row in rows
    }
    expected_fold_sizes = {
        fold: sum(value == int(fold) for value in fold_by_identity.values())
        for fold in orchestrator.FOLD_IDS
    }
    require(
        development_seal.get("outer_fold_sizes") == expected_fold_sizes
        and development_seal.get("local_confirmation_gate_sha256") is None
        and development_seal.get("aggregate", {}).get("predicted_forks") == 0,
        "development seal fold or pre-score metadata drifted",
    )
    binding = development_seal.get("r1_binding")
    require(isinstance(binding, Mapping), "development seal R1 binding is missing")
    _strict(binding, R1_BINDING_FIELDS, "development seal R1 binding")
    for field in R1_BINDING_FIELDS:
        _sha(binding[field], f"R1 binding {field}")
    require(
        binding["binding_sha256"] == _binding_sha256(binding),
        "development seal R1 binding SHA-256 drifted",
    )
    require(
        binding["inference_seal_sha256"] == r1_inference_seal_sha256,
        "development seal does not bind the R1 inference seal",
    )

    require(
        r1_seal.get("schema") == R1_INFERENCE_SEAL_SCHEMA
        and r1_seal.get("status") == "SEALED_ALL_36_BEFORE_SCORE_READ"
        and r1_seal.get("population", {}).get("family") == "44b6"
        and r1_seal.get("population", {}).get("identity_count")
        == orchestrator.HELD_COUNT
        and len(r1_seal.get("population", {}).get("identities", ()))
        == orchestrator.HELD_COUNT
        and set(r1_seal.get("population", {}).get("identities", ()))
        == set(held_identities)
        and r1_seal.get("contract", {}).get("sha256")
        == binding["contract_sha256"]
        and r1_seal.get("compiler", {}).get("divisions") is False
        and r1_seal.get("aggregate", {}).get("predicted_forks") == 0
        and r1_seal.get("gt_access") is False
        and r1_seal.get("held_fit_score_read") is False,
        "R1 inference seal target-blind held36 contract drifted",
    )
    development_rows = development_seal.get("videos")
    held_rows = r1_seal.get("videos")
    require(
        isinstance(development_rows, list)
        and len(development_rows) == orchestrator.DEVELOPMENT_COUNT
        and tuple(str(row.get("identity")) for row in development_rows if isinstance(row, Mapping))
        == development_identities,
        "development seal video population drifted",
    )
    require(
        isinstance(held_rows, list)
        and len(held_rows) == orchestrator.HELD_COUNT
        and len(
            [row for row in held_rows if isinstance(row, Mapping)]
        )
        == orchestrator.HELD_COUNT
        and {
            str(row.get("identity")) for row in held_rows if isinstance(row, Mapping)
        }
        == set(held_identities),
        "R1 inference-seal video population drifted",
    )

    scope = development_seal.get("local_105_bank_scope")
    require(isinstance(scope, Mapping), "development seal local105 scope is missing")
    require(
        scope.get("development_video_count") == orchestrator.DEVELOPMENT_COUNT
        and scope.get("held_fit_video_count") == orchestrator.HELD_COUNT
        and scope.get("total_video_count") == orchestrator.FIT105_COUNT
        and scope.get("all_hashes_sealed_before_R4_development_scoring") is True,
        "development seal local105 scope drifted",
    )

    records: list[dict[str, Any]] = []
    dev_root = development_seal_path.parent / "videos"
    for row in development_rows:
        require(isinstance(row, Mapping), "development seal video row is malformed")
        identity = str(row["identity"])
        require(
            row.get("association_bank_target_free") is True
            and row.get("parent_graph_mode") == "DIVISIONS_OFF"
            and row.get("role") == orchestrator.DEVELOPMENT_ROLE
            and row.get("outer_fold") == fold_by_identity[identity]
            and row.get("population_sha256") == population.get("population_sha256")
            and row.get("population_manifest_sha256") == population_file_sha256
            and row.get("r1_binding_sha256") == binding["binding_sha256"]
            and row.get("r1_contract_sha256") == binding["contract_sha256"]
            and row.get("r1_scientific_configuration_sha256")
            == binding["scientific_configuration_sha256"]
            and row.get("gt_access") is False
            and row.get("score_read") is False,
            f"development video is not target-blind: {identity}",
        )
        video_root = dev_root / identity.replace(":", "__", 1)
        bank = video_root / "association_bank.npz"
        parent = video_root / "DIVISIONS_OFF_PARENT.csv"
        records.append(
            {
                "identity": identity,
                "role": orchestrator.DEVELOPMENT_ROLE,
                "r1_configuration_name": "R1_ZERO_SHOT__"
                + str(binding["scientific_configuration_sha256"]),
                "association_bank": _absolute_ref(bank, row["association_bank_sha256"]),
                "parent_graph": _absolute_ref(parent, row["parent_graph_sha256"]),
                "reference_graph": _absolute_ref(parent, row["parent_graph_sha256"]),
            }
        )

    held_bank_root = r1_inference_seal_path.parent / "association_banks"
    held_parent_root = r1_inference_seal_path.parent / "fragments"
    apncc_root = apncc_report_path.parent / "control_fragments"
    expected_apncc_names = {
        f"{identity.replace(':', '__', 1)}.csv" for identity in held_identities
    }
    require(
        apncc_root.is_dir()
        and not apncc_root.is_symlink()
        and {path.name for path in apncc_root.iterdir()} == expected_apncc_names,
        "APNCC control_fragments are not exact held36",
    )
    apncc_pairs: list[tuple[str, str]] = []
    for row in held_rows:
        require(isinstance(row, Mapping), "R1 inference-seal video row is malformed")
        identity = str(row["identity"])
        require(
            row.get("association_bank_complete") is True
            and row.get("contract_sha256") == binding["contract_sha256"]
            and row.get("predicted_forks") == 0
            and row.get("gt_access") is False
            and row.get("held_fit_score_read") is False,
            f"R1 held video is not target-blind: {identity}",
        )
        slug = identity.replace(":", "__", 1)
        bank = held_bank_root / f"{slug}.npz"
        parent = held_parent_root / f"{slug}.csv"
        reference = apncc_root / f"{slug}.csv"
        require(reference.is_file() and not reference.is_symlink(), f"APNCC fragment missing: {identity}")
        reference_sha = sha256_file(reference)
        apncc_pairs.append((identity, reference_sha))
        records.append(
            {
                "identity": identity,
                "role": orchestrator.HELD_ROLE,
                "r1_configuration_name": "R1_ZERO_SHOT__"
                + str(binding["scientific_configuration_sha256"]),
                "association_bank": _absolute_ref(bank, row["association_bank_sha256"]),
                "parent_graph": _absolute_ref(parent, row["fragment_sha256"]),
                "reference_graph": _absolute_ref(reference, reference_sha),
            }
        )
    require(
        _fragment_aggregate(apncc_pairs) == APNCC_AGGREGATE_SHA256,
        "APNCC held36 aggregate graph SHA-256 drifted",
    )
    require(
        _aggregate_artifacts(development_rows, "association_bank_sha256")
        == development_seal.get("aggregate", {}).get("association_bank_sha256")
        == scope.get("development_association_bank_sha256"),
        "development association-bank aggregate drifted",
    )
    require(
        _aggregate_artifacts(development_rows, "parent_graph_sha256")
        == development_seal.get("aggregate", {}).get("parent_graph_sha256")
        == scope.get("development_parent_graph_sha256"),
        "development parent aggregate drifted",
    )
    require(
        _aggregate_artifacts(held_rows, "association_bank_sha256")
        == r1_seal.get("aggregate", {}).get("association_bank_sha256")
        == binding["held_fit_association_bank_sha256"]
        == scope.get("held_fit_association_bank_sha256"),
        "R1 held association-bank aggregate drifted",
    )
    require(
        _aggregate_artifacts(held_rows, "fragment_sha256")
        == r1_seal.get("aggregate", {}).get("fragment_sha256")
        == binding["held_fit_parent_graph_sha256"]
        == scope.get("held_fit_parent_graph_sha256"),
        "R1 held parent aggregate drifted",
    )

    # Population equality is established above.  Only now are bank/graph bytes read.
    for record in records:
        for kind in ("association_bank", "parent_graph", "reference_graph"):
            path, digest = _resolve_ref(record[kind], Path.cwd(), f"{record['identity']} {kind}")
            _require_hash(path, digest, f"{record['identity']} {kind}")
    records.sort(key=lambda value: str(value["identity"]))
    return {
        "sealed_before_development_scoring": True,
        "r1_scientific_configuration": {
            "name": "R1_ZERO_SHOT__" + str(binding["scientific_configuration_sha256"]),
            "inference_seal": _absolute_ref(
                r1_inference_seal_path, r1_inference_seal_sha256
            ),
        },
        "records": records,
    }


def prepare_pretarget_input(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    configuration_path: Path,
    configuration_sha256: str,
    population_path: Path,
    population_sha256: str,
    amendment_path: Path,
    amendment_sha256: str,
    development_seal_path: Path,
    development_seal_sha256: str,
    r1_inference_seal_path: Path,
    r1_inference_seal_sha256: str,
    apncc_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = _load_verified_json(protocol_path, protocol_sha256, "R4 protocol")
    configuration = _load_verified_json(
        configuration_path, configuration_sha256, "configuration manifest"
    )
    population = _load_verified_json(
        population_path, population_sha256, "population manifest"
    )
    amendment = _load_verified_json(
        amendment_path, amendment_sha256, "population amendment"
    )
    development, held = _validate_contracts(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        configuration=configuration,
        configuration_sha256=configuration_sha256,
        population=population,
        population_sha256=population_sha256,
        amendment=amendment,
        amendment_sha256=amendment_sha256,
    )
    coverage = _validate_seals_and_build_coverage(
        population=population,
        population_file_sha256=population_sha256,
        amendment_sha256=amendment_sha256,
        development_identities=development,
        held_identities=held,
        development_seal_path=development_seal_path,
        development_seal_sha256=development_seal_sha256,
        r1_inference_seal_path=r1_inference_seal_path,
        r1_inference_seal_sha256=r1_inference_seal_sha256,
        apncc_report_path=apncc_report_path,
    )
    manifest = {
        "schema": orchestrator.PRETARGET_INPUT_SCHEMA,
        "status": PRETARGET_STATUS,
        "protocol": _absolute_ref(protocol_path, protocol_sha256),
        "configuration_manifest": _absolute_ref(
            configuration_path, configuration_sha256
        ),
        "population_manifest": _absolute_ref(population_path, population_sha256),
        "population_amendment": _absolute_ref(amendment_path, amendment_sha256),
        "fit105_coverage": coverage,
    }
    try:
        normalized, coverage_sha, _r1 = orchestrator.verify_exact_fit105_coverage(
            coverage, population=population, base=output.parent
        )
    except orchestrator.DivisionOOFError as exc:
        raise PrepareInputsError(str(exc)) from exc
    require(len(normalized) == orchestrator.FIT105_COUNT, "normalized local105 drifted")
    file_sha = _write_no_overwrite(output, manifest)
    return {
        "status": PRETARGET_STATUS,
        "path": str(output.resolve()),
        "sha256": file_sha,
        "coverage_sha256": coverage_sha,
        "video_count": len(normalized),
    }


def _validate_plan_for_runner(
    plan: Mapping[str, Any], plan_sha256: str, repo_root: Path
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    try:
        payload = orchestrator._validate_artifact(
            plan,
            orchestrator.PRETARGET_PLAN_SCHEMA,
            "SEALED_PRE_TARGET_NO_SCORE_OR_GT_READ",
            "pretarget plan",
        )
    except orchestrator.DivisionOOFError as exc:
        raise PrepareInputsError(str(exc)) from exc
    source_bindings = payload.get("execution_source_bindings")
    require(
        isinstance(source_bindings, Mapping) and set(source_bindings) == {"runner"},
        "pretarget plan lacks the exact runner source binding",
    )
    runner_path, runner_sha = _verify_ref(
        source_bindings["runner"], Path.cwd(), "pretarget runner source"
    )
    expected_runner = (repo_root / orchestrator.RUNNER_SOURCE_RELATIVE).resolve()
    require(runner_path == expected_runner, "pretarget runner path drifted")
    require(
        runner_sha == orchestrator.PROMOTED_RUNNER_SHA256,
        "pretarget plan does not bind the promoted runner",
    )
    coverage = payload.get("fit105_coverage")
    require(isinstance(coverage, Mapping), "pretarget plan lacks local105 coverage")
    rows = coverage.get("records")
    require(
        isinstance(rows, list)
        and len(rows) == orchestrator.FIT105_COUNT
        and coverage.get("video_count") == orchestrator.FIT105_COUNT
        and coverage.get("development_count") == orchestrator.DEVELOPMENT_COUNT
        and coverage.get("held_fit_count") == orchestrator.HELD_COUNT,
        "pretarget plan local105 counts drifted",
    )
    roles = [str(row.get("role")) for row in rows if isinstance(row, Mapping)]
    require(
        len(roles) == orchestrator.FIT105_COUNT
        and roles.count(orchestrator.DEVELOPMENT_ROLE)
        == orchestrator.DEVELOPMENT_COUNT
        and roles.count(orchestrator.HELD_ROLE) == orchestrator.HELD_COUNT
        and orchestrator.AUDIT_ROLE not in roles
        and orchestrator.HIDDEN_ROLE not in roles,
        "pretarget plan local105 roles drifted",
    )
    require(
        coverage.get("coverage_sha256") == decode.canonical_sha256(rows),
        "pretarget plan local105 coverage SHA-256 drifted",
    )
    require(payload.get("forbidden_roles") == [orchestrator.AUDIT_ROLE], "audit role is not forbidden")
    for row in rows:
        require(isinstance(row, Mapping), "pretarget local105 row is malformed")
        for kind in ("association_bank", "parent_graph", "reference_graph"):
            _verify_ref(row.get(kind), Path.cwd(), f"plan {row.get('identity')} {kind}")
        if row.get("role") == orchestrator.DEVELOPMENT_ROLE:
            require(
                row.get("parent_graph") == row.get("reference_graph"),
                f"development reference is not its same-video parent: {row.get('identity')}",
            )
    contracts = payload.get("contracts")
    require(isinstance(contracts, Mapping), "pretarget contracts are missing")
    for key in ("protocol", "configuration_manifest", "population_manifest", "population_amendment"):
        _verify_ref(contracts.get(key), Path.cwd(), f"pretarget contract {key}")
    _sha(plan_sha256, "pretarget plan")
    return str(coverage["coverage_sha256"]), tuple(rows)


def prepare_runner_input(
    *,
    repo_root: Path,
    pretarget_plan_path: Path,
    pretarget_plan_sha256: str,
    development_seal_path: Path,
    development_seal_sha256: str,
    competition_archive_path: Path,
    competition_archive_sha256: str,
    output: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    require(
        repo_root == Path(__file__).resolve().parents[1],
        "repo root is not the generator's executing repository",
    )
    plan = _load_verified_json(
        pretarget_plan_path, pretarget_plan_sha256, "pretarget plan"
    )
    coverage_sha, rows = _validate_plan_for_runner(
        plan, pretarget_plan_sha256, repo_root
    )
    development_seal = _load_verified_json(
        development_seal_path, development_seal_sha256, "development seal"
    )
    require(
        development_seal.get("schema") == DEV_SEAL_SCHEMA
        and development_seal.get("status")
        == "SEALED_EXACT_DEV69_BEFORE_R4_GT_OR_SCORER_ACCESS"
        and development_seal.get("role") == orchestrator.DEVELOPMENT_ROLE
        and development_seal.get("identity_count") == orchestrator.DEVELOPMENT_COUNT
        and development_seal.get("association_banks_target_free") is True
        and development_seal.get("parent_graph_mode") == "DIVISIONS_OFF"
        and development_seal.get("sealed_before_R4_gt_or_scorer_access") is True
        and development_seal.get("gt_access") is False
        and development_seal.get("score_read") is False,
        "development seal target-blind contract drifted",
    )
    plan_payload = plan["payload"]
    plan_contracts = plan_payload["contracts"]
    plan_r1 = plan_payload["fit105_coverage"]["r1_scientific_configuration"]
    require(
        development_seal.get("population_sha256")
        == plan_payload.get("population_identity_sha256")
        and development_seal.get("population_manifest_sha256")
        == plan_contracts["population_manifest"]["sha256"]
        and development_seal.get("population_amendment_sha256")
        == plan_contracts["population_amendment"]["sha256"]
        and development_seal.get("r1_binding", {}).get("inference_seal_sha256")
        == plan_r1.get("inference_seal_sha256"),
        "development seal contract binding differs from the pretarget plan",
    )
    development_rows = {
        str(row.get("identity")): row
        for row in development_seal.get("videos", ())
        if isinstance(row, Mapping)
    }
    plan_development = {
        str(row.get("identity")): row
        for row in rows
        if row.get("role") == orchestrator.DEVELOPMENT_ROLE
    }
    require(
        len(development_rows) == orchestrator.DEVELOPMENT_COUNT
        and set(development_rows) == set(plan_development),
        "development seal population differs from the pretarget plan",
    )
    for identity, sealed in development_rows.items():
        planned = plan_development[identity]
        require(
            planned.get("association_bank", {}).get("sha256")
            == sealed.get("association_bank_sha256")
            and planned.get("parent_graph", {}).get("sha256")
            == sealed.get("parent_graph_sha256"),
            f"development seal hash differs from plan: {identity}",
        )

    source_paths = {
        "decoder_source": repo_root / "scripts/g03_division_decode.py",
        "orchestrator_source": repo_root / "scripts/g03_division_oof_orchestrator.py",
        "scoreall_source": repo_root / "scripts/scoreall.py",
    }
    source_refs = {}
    for key, path in source_paths.items():
        require(path.is_file() and not path.is_symlink(), f"{key} is not a regular source")
        source_refs[key] = _absolute_ref(path, sha256_file(path))
    runner_path = repo_root / orchestrator.RUNNER_SOURCE_RELATIVE
    runner_sha = _require_hash(
        runner_path, orchestrator.PROMOTED_RUNNER_SHA256, "promoted R4 runner"
    )
    plan_runner_path, plan_runner_sha = _resolve_ref(
        plan["payload"]["execution_source_bindings"]["runner"],
        Path.cwd(),
        "sealed runner",
    )
    require(
        plan_runner_path == runner_path.resolve() and plan_runner_sha == runner_sha,
        "runner bytes differ from the pretarget seal",
    )
    dependencies = {}
    for relative in runner.SCOREALL_DEPENDENCIES:
        path = (repo_root / relative).resolve()
        require(path.is_relative_to(repo_root), f"scorer dependency escaped repository: {relative}")
        require(
            path.is_file() and not path.is_symlink(),
            f"scorer dependency is not a regular source: {relative}",
        )
        dependencies[relative] = _absolute_ref(path, sha256_file(path))
    one_pass_dependencies = {}
    for relative in runner.ONE_PASS_DEPENDENCIES:
        path = (repo_root / relative).resolve()
        require(
            path.is_relative_to(repo_root),
            f"one-pass dependency escaped repository: {relative}",
        )
        require(
            path.is_file() and not path.is_symlink(),
            f"one-pass dependency is not a regular source: {relative}",
        )
        one_pass_dependencies[relative] = _absolute_ref(path, sha256_file(path))
    archive_sha = _require_hash(
        competition_archive_path,
        competition_archive_sha256,
        "competition archive opaque bytes",
    )
    manifest = {
        "schema": runner.RUNNER_INPUT_SCHEMA,
        "status": RUNNER_STATUS,
        "pretarget_plan": _absolute_ref(pretarget_plan_path, pretarget_plan_sha256),
        "local105_inventory": {
            "pretarget_plan_sha256": pretarget_plan_sha256,
            "coverage_sha256": coverage_sha,
            "video_count": orchestrator.FIT105_COUNT,
        },
        "development_seal": _absolute_ref(
            development_seal_path, development_seal_sha256
        ),
        **source_refs,
        "one_pass_dependencies": one_pass_dependencies,
        "scorer_dependencies": dependencies,
        "competition_archive": _absolute_ref(competition_archive_path, archive_sha),
    }
    file_sha = _write_no_overwrite(output, manifest)
    return {
        "status": RUNNER_STATUS,
        "path": str(output.resolve()),
        "sha256": file_sha,
        "coverage_sha256": coverage_sha,
        "runner_sha256": runner_sha,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    pretarget = commands.add_parser("pretarget", help="write the exact local105 pretarget input")
    for name in ("protocol", "configuration", "population", "amendment"):
        pretarget.add_argument(f"--{name}", required=True, type=Path)
        pretarget.add_argument(f"--{name}-sha256", required=True)
    pretarget.add_argument("--development-seal", required=True, type=Path)
    pretarget.add_argument("--development-seal-sha256", required=True)
    pretarget.add_argument("--r1-inference-seal", required=True, type=Path)
    pretarget.add_argument("--r1-inference-seal-sha256", required=True)
    pretarget.add_argument("--apncc-report", required=True, type=Path)
    pretarget.add_argument("--output", required=True, type=Path)

    runner_input = commands.add_parser("runner", help="write the development runner input")
    runner_input.add_argument("--repo-root", required=True, type=Path)
    runner_input.add_argument("--pretarget-plan", required=True, type=Path)
    runner_input.add_argument("--pretarget-plan-sha256", required=True)
    runner_input.add_argument("--development-seal", required=True, type=Path)
    runner_input.add_argument("--development-seal-sha256", required=True)
    runner_input.add_argument("--competition-archive", required=True, type=Path)
    runner_input.add_argument("--competition-archive-sha256", required=True)
    runner_input.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "pretarget":
        report = prepare_pretarget_input(
            protocol_path=args.protocol,
            protocol_sha256=args.protocol_sha256,
            configuration_path=args.configuration,
            configuration_sha256=args.configuration_sha256,
            population_path=args.population,
            population_sha256=args.population_sha256,
            amendment_path=args.amendment,
            amendment_sha256=args.amendment_sha256,
            development_seal_path=args.development_seal,
            development_seal_sha256=args.development_seal_sha256,
            r1_inference_seal_path=args.r1_inference_seal,
            r1_inference_seal_sha256=args.r1_inference_seal_sha256,
            apncc_report_path=args.apncc_report,
            output=args.output,
        )
    else:
        report = prepare_runner_input(
            repo_root=args.repo_root,
            pretarget_plan_path=args.pretarget_plan,
            pretarget_plan_sha256=args.pretarget_plan_sha256,
            development_seal_path=args.development_seal,
            development_seal_sha256=args.development_seal_sha256,
            competition_archive_path=args.competition_archive,
            competition_archive_sha256=args.competition_archive_sha256,
            output=args.output,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

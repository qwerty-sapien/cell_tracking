#!/usr/bin/env python3
"""Fail-closed, target-blind orchestration for G-03 R4 division selection.

This module separates every target-bearing transition.  ``prepare`` verifies the
sealed decoder, population amendment, and exact fit105 bank/parent/reference
inventory, then writes a no-overwrite pre-target plan.  Later helpers accept only
hash-pinned metric receipts, freeze each inner-OOF winner before its outer-fold
receipt, choose the deployable configuration by modal winner count, compute the
five-fold development PASS gate, and prepare one-shot held-fit and gated hidden-test
markers.  Held preparation recomputes that gate; caller assertions are never enough.

There is deliberately no scorer, ground-truth loader, Kaggle client, or submission
interface in this file.  Creating a marker records authorization prerequisites; it
does not open a target or perform an external action.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:  # Direct script execution and repository-root imports use different paths.
    import g03_division_decode as decode
except ModuleNotFoundError:  # pragma: no cover - exercised by repository pytest.
    from scripts import g03_division_decode as decode


PRETARGET_INPUT_SCHEMA = "celltrack-alpha.g03-r4-oof-pretarget-input.v1"
PRETARGET_PLAN_SCHEMA = "celltrack-alpha.g03-r4-oof-pretarget-plan.v1"
INNER_SCORE_SCHEMA = "celltrack-alpha.g03-r4-inner-oof-score-receipt.v1"
INNER_WINNER_SCHEMA = "celltrack-alpha.g03-r4-inner-oof-winner-seal.v1"
OUTER_SCORE_SCHEMA = "celltrack-alpha.g03-r4-outer-holdout-score-receipt.v1"
OUTER_RESULT_SCHEMA = "celltrack-alpha.g03-r4-outer-holdout-result.v1"
DEPLOYABLE_SCHEMA = "celltrack-alpha.g03-r4-deployable-seal.v1"
DEPLOYABLE_STATUS = "PASS_DEVELOPMENT_GATE_SEALED_BEFORE_HELD_FIT_TARGET_ACCESS"
HELD_INVENTORY_SCHEMA = "celltrack-alpha.g03-r4-held-fit-inventory.v1"
R1_BINDING_SCHEMA = "celltrack-alpha.g03-r1-validated-result-binding.v1"
HELD_MARKER_SCHEMA = "celltrack-alpha.g03-r4-held-fit-access-marker.v1"
LOCAL_GATE_SCHEMA = "celltrack-alpha.g03-r4-local-confirmation-gate.v1"
HIDDEN_INVENTORY_SCHEMA = "celltrack-alpha.g03-r4-hidden-probe-inventory.v1"
HIDDEN_MARKER_SCHEMA = "celltrack-alpha.g03-r4-hidden-probe-marker.v1"

FIT105_ROLES = (
    "development_nested_oof",
    "held_fit_one_shot_confirmation",
)
DEVELOPMENT_ROLE = FIT105_ROLES[0]
HELD_ROLE = FIT105_ROLES[1]
HIDDEN_ROLE = "sealed_hidden_test_probe"
AUDIT_ROLE = "excluded_contaminated_audit20"
FOLD_IDS = ("0", "1", "2", "3", "4")
FIT105_COUNT = 105
DEVELOPMENT_COUNT = 69
HELD_COUNT = 36
HIDDEN_COUNT = 4
CONFIGURATION_COUNT = 18_145
CAMPAIGN_BEST_SCORE = "0.881522069960"
SAME_POPULATION_REFERENCE_LABEL = "DIVISIONS_OFF_EXACT_SAME_DEV69_NESTED_OOF_PARENT"
SCORER_DEPENDENCY_PATHS = (
    "src/celltrack_alpha/official_eval.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py",
)
ONE_PASS_DEPENDENCY_PATHS = (
    "src/celltrack_alpha/action_compiler.py",
    "src/celltrack_alpha/kaggle_calibration.py",
)
RUNNER_SOURCE_RELATIVE = "scripts/g03_division_oof_runner.py"
PROMOTED_RUNNER_SHA256 = (
    "cc37fc63729d4f8d5418951dabc4ecd2ff1ccc2b7f684011793263af73cde304"
)

APNCC_REFERENCE = {
    "id": "APNCC_44b6_CONTROL",
    "score": "0.866534526036",
    "aggregate_graph_sha256": (
        "48b99c5db7024c0998b779d2dfcf30bade9c12d74ec679325cc1974ceb9f7ccb"
    ),
    "report_sha256": (
        "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d"
    ),
}
R1_REFERENCE_ID = "R1_DIVISIONS_OFF"
HELD_SENTINELS = (
    "HELD_FIT_ACCESS_MARKER.json",
    "HELD_FIT_SCORE_RECEIPT.json",
    "HELD_FIT_TERMINAL_RECEIPT.json",
    "R4_OVERLAY_OUTPUT_RECEIPT.json",
)
HIDDEN_SENTINELS = (
    "HIDDEN_PROBE_MARKER.json",
    "HIDDEN_PROBE_SUBMISSION_RECEIPT.json",
    "HIDDEN_PROBE_TERMINAL_RECEIPT.json",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


class DivisionOOFError(RuntimeError):
    """An R4 orchestration contract failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise DivisionOOFError(
            f"{label} keys drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha(value: Any, label: str) -> str:
    result = str(value)
    if not SHA256_RE.fullmatch(result):
        raise DivisionOOFError(f"{label} is not a lowercase SHA-256")
    return result


def _resolve(base: Path, raw: Any) -> Path:
    candidate = Path(str(raw))
    return candidate if candidate.is_absolute() else base / candidate


def _ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be a path/SHA reference")
    _strict(value, {"path", "sha256"}, label)
    return _resolve(base, value["path"]), _sha(value["sha256"], label)


def _verified_ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    path, expected = _ref(value, base, label)
    try:
        decode._verify_file(path, expected, label)
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc
    return path, expected


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DivisionOOFError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DivisionOOFError(f"{label} must be a JSON object")
    return value


def _load_verified_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    try:
        return decode._load_verified_json(path, _sha(expected, label), label)
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc


def _absolute_ref(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(sha256, "reference hash")}


def _artifact(schema: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        "schema": schema,
        "status": status,
        "payload": body,
        "payload_sha256": decode.canonical_sha256(body),
    }


def _validate_artifact(value: Any, schema: str, status: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be an object")
    _strict(value, {"schema", "status", "payload", "payload_sha256"}, label)
    payload = value.get("payload")
    if (
        value.get("schema") != schema
        or value.get("status") != status
        or not isinstance(payload, Mapping)
        or value.get("payload_sha256") != decode.canonical_sha256(payload)
    ):
        raise DivisionOOFError(f"{label} identity or status drifted")
    return payload


def load_artifact_reference(
    reference: Any,
    *,
    base: Path,
    schema: str,
    status: str,
    label: str,
) -> tuple[dict[str, Any], Mapping[str, Any], Path, str]:
    path, expected = _verified_ref(reference, base, label)
    artifact = _load_json(path, label)
    payload = _validate_artifact(artifact, schema, status, label)
    return artifact, payload, path, expected


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically create canonical JSON, refusing every pre-existing target."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise DivisionOOFError(f"refusing to overwrite sealed artifact: {path}")
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
            raise DivisionOOFError(f"refusing to overwrite sealed artifact: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return decode.sha256_file(path)


def _role_identities(population: Mapping[str, Any], role: str) -> tuple[str, ...]:
    roles = population.get("roles")
    record = roles.get(role) if isinstance(roles, Mapping) else None
    videos = record.get("videos") if isinstance(record, Mapping) else None
    if not isinstance(videos, list):
        raise DivisionOOFError(f"population role is missing: {role}")
    identities = tuple(
        str(row.get("identity")) for row in videos if isinstance(row, Mapping)
    )
    if len(identities) != len(videos) or len(set(identities)) != len(identities):
        raise DivisionOOFError(f"population role identities are malformed: {role}")
    return identities


def _fold_identities(population: Mapping[str, Any], fold: str) -> tuple[str, ...]:
    folds = population.get("outer_folds")
    rows = folds.get(fold) if isinstance(folds, Mapping) else None
    if not isinstance(rows, list):
        raise DivisionOOFError(f"population fold is missing: {fold}")
    values = tuple(str(row.get("identity")) for row in rows if isinstance(row, Mapping))
    if len(values) != len(rows):
        raise DivisionOOFError(f"population fold is malformed: {fold}")
    return values


def _configuration_names(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    rows = manifest.get("configurations")
    config_record = protocol.get("configuration_manifest")
    if (
        manifest.get("schema") != decode.CONFIG_SCHEMA
        or manifest.get("status") != "LOCKED_BEFORE_OUTCOMES"
        or not isinstance(rows, list)
        or len(rows) != CONFIGURATION_COUNT
        or manifest.get("configuration_count_including_null") != CONFIGURATION_COUNT
        or manifest.get("active_configuration_count") != CONFIGURATION_COUNT - 1
        or not isinstance(config_record, Mapping)
    ):
        raise DivisionOOFError("configuration manifest shape or status drifted")
    observed_hash = decode.canonical_sha256(rows)
    if (
        manifest.get("configuration_sha256") != observed_hash
        or config_record.get("configuration_sha256") != observed_hash
    ):
        raise DivisionOOFError("configuration identity drifted")
    names = tuple(
        str(row.get("name")) for row in rows if isinstance(row, Mapping)
    )
    if (
        len(names) != CONFIGURATION_COUNT
        or len(set(names)) != CONFIGURATION_COUNT
        or any(not name for name in names)
        or names.count("DIVISIONS_OFF") != 1
    ):
        raise DivisionOOFError("configuration names are incomplete or non-unique")
    return names


def verify_exact_fit105_coverage(
    coverage: Any,
    *,
    population: Mapping[str, Any],
    base: Path,
) -> tuple[tuple[dict[str, Any], ...], str, dict[str, str]]:
    """Validate metadata coverage first; only then touch any bank/graph bytes."""

    if not isinstance(coverage, Mapping):
        raise DivisionOOFError("fit105 coverage must be an object")
    _strict(
        coverage,
        {"sealed_before_development_scoring", "r1_scientific_configuration", "records"},
        "fit105 coverage",
    )
    r1 = coverage.get("r1_scientific_configuration")
    records = coverage.get("records")
    if (
        coverage.get("sealed_before_development_scoring") is not True
        or not isinstance(r1, Mapping)
        or not isinstance(records, list)
    ):
        raise DivisionOOFError("fit105 coverage is not sealed before development scoring")
    _strict(r1, {"name", "inference_seal"}, "R1 scientific configuration")
    r1_name = str(r1.get("name"))
    if not r1_name:
        raise DivisionOOFError("R1 scientific configuration name is empty")
    # Parsing references is metadata-only.  No referenced file is opened before the
    # exact population and role equality checks below have succeeded.
    parsed: list[tuple[dict[str, Any], dict[str, tuple[Path, str]]]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise DivisionOOFError(f"fit105 record {index} is malformed")
        _strict(
            raw,
            {
                "identity",
                "role",
                "r1_configuration_name",
                "association_bank",
                "parent_graph",
                "reference_graph",
            },
            f"fit105 record {index}",
        )
        refs = {
            name: _ref(raw[name], base, f"fit105 {name} {index}")
            for name in ("association_bank", "parent_graph", "reference_graph")
        }
        parsed.append((dict(raw), refs))
    expected_by_role = {
        DEVELOPMENT_ROLE: set(_role_identities(population, DEVELOPMENT_ROLE)),
        HELD_ROLE: set(_role_identities(population, HELD_ROLE)),
    }
    observed_by_role = {role: set() for role in FIT105_ROLES}
    for row, _refs in parsed:
        role = str(row["role"])
        identity = str(row["identity"])
        if role not in observed_by_role:
            raise DivisionOOFError(f"fit105 contains forbidden role: {role}")
        if identity in observed_by_role[role]:
            raise DivisionOOFError(f"fit105 duplicate identity: {identity}")
        if row["r1_configuration_name"] != r1_name:
            raise DivisionOOFError("fit105 banks do not share one R1 configuration")
        observed_by_role[role].add(identity)
    if (
        len(parsed) != FIT105_COUNT
        or len(observed_by_role[DEVELOPMENT_ROLE]) != DEVELOPMENT_COUNT
        or len(observed_by_role[HELD_ROLE]) != HELD_COUNT
        or observed_by_role != expected_by_role
    ):
        raise DivisionOOFError("exact 105-bank dev69/held36 coverage is not satisfied")
    audit = set(_role_identities(population, AUDIT_ROLE))
    if audit & set().union(*observed_by_role.values()):
        raise DivisionOOFError("audit20 is forbidden from R4 coverage")
    for name in ("association_bank", "parent_graph", "reference_graph"):
        paths = [str(refs[name][0].resolve()) for _row, refs in parsed]
        if len(paths) != len(set(paths)):
            raise DivisionOOFError(f"fit105 {name} paths are not one-per-identity")

    inference_path, inference_hash = _verified_ref(
        r1["inference_seal"], base, "R1 inference seal"
    )
    normalized: list[dict[str, Any]] = []
    for row, refs in parsed:
        normalized_row: dict[str, Any] = {
            "identity": str(row["identity"]),
            "role": str(row["role"]),
            "r1_configuration_name": r1_name,
        }
        for name, (path, expected) in refs.items():
            try:
                decode._verify_file(path, expected, f"fit105 {name} {row['identity']}")
            except decode.DivisionDecodeError as exc:
                raise DivisionOOFError(str(exc)) from exc
            normalized_row[name] = _absolute_ref(path, expected)
        normalized.append(normalized_row)
    normalized.sort(key=lambda row: str(row["identity"]))
    identity = decode.canonical_sha256(normalized)
    return (
        tuple(normalized),
        identity,
        {
            "name": r1_name,
            "inference_seal_path": str(inference_path.resolve()),
            "inference_seal_sha256": inference_hash,
        },
    )


def build_fixed_fold_plan(population: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    development = set(_role_identities(population, DEVELOPMENT_ROLE))
    folds = {fold: _fold_identities(population, fold) for fold in FOLD_IDS}
    flattened = [identity for fold in FOLD_IDS for identity in folds[fold]]
    if len(flattened) != DEVELOPMENT_COUNT or set(flattened) != development:
        raise DivisionOOFError("five fixed folds do not exactly cover dev69")
    result = []
    for outer in FOLD_IDS:
        inner = tuple(fold for fold in FOLD_IDS if fold != outer)
        inner_identities = tuple(
            identity for fold in inner for identity in folds[fold]
        )
        if set(inner_identities) & set(folds[outer]):
            raise DivisionOOFError("outer holdout leaked into inner tuning folds")
        result.append(
            {
                "outer_fold": outer,
                "outer_identities": list(folds[outer]),
                "inner_folds": list(inner),
                "inner_identities": list(inner_identities),
                "inner_fold_identities": {
                    fold: list(folds[fold]) for fold in inner
                },
                "tuning_scope": "EXACTLY_FOUR_FIXED_GROUPED_6BBA_FOLDS",
                "outer_evaluation_scope": "WINNER_ONLY_AFTER_INNER_WINNER_SEAL",
            }
        )
    return tuple(result)


def _verify_effective_contracts(
    manifest: Mapping[str, Any], manifest_path: Path, repo_root: Path
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[str, ...],
    dict[str, dict[str, str]],
]:
    _strict(
        manifest,
        {
            "schema",
            "status",
            "protocol",
            "configuration_manifest",
            "population_manifest",
            "population_amendment",
            "fit105_coverage",
        },
        "pre-target input",
    )
    if (
        manifest.get("schema") != PRETARGET_INPUT_SCHEMA
        or manifest.get("status") != "SEALED_BEFORE_DEVELOPMENT_SCORING"
    ):
        raise DivisionOOFError("pre-target input is not sealed before scoring")
    base = manifest_path.parent
    refs: dict[str, dict[str, str]] = {}
    loaded: dict[str, Mapping[str, Any]] = {}
    for key, label in (
        ("protocol", "R4 protocol"),
        ("configuration_manifest", "configuration manifest"),
        ("population_manifest", "R4 population manifest"),
        ("population_amendment", "R4 population amendment"),
    ):
        path, expected = _verified_ref(manifest[key], base, label)
        refs[key] = _absolute_ref(path, expected)
        loaded[key] = _load_json(path, label)
    protocol = loaded["protocol"]
    configuration = loaded["configuration_manifest"]
    population = loaded["population_manifest"]
    amendment = loaded["population_amendment"]
    try:
        decode._verify_protocol(
            protocol,
            protocol_path=Path(refs["protocol"]["path"]),
            protocol_sha256=refs["protocol"]["sha256"],
            configuration_path=Path(refs["configuration_manifest"]["path"]),
            configuration_sha256=refs["configuration_manifest"]["sha256"],
            repo_root=repo_root,
        )
        roles = decode._verify_population_contract(
            population=population,
            population_path=Path(refs["population_manifest"]["path"]),
            population_file_sha256=refs["population_manifest"]["sha256"],
            amendment=amendment,
            amendment_path=Path(refs["population_amendment"]["path"]),
            amendment_sha256=refs["population_amendment"]["sha256"],
            protocol=protocol,
            protocol_sha256=refs["protocol"]["sha256"],
            repo_root=repo_root,
        )
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc
    if roles[AUDIT_ROLE] & (roles[DEVELOPMENT_ROLE] | roles[HELD_ROLE]):
        raise DivisionOOFError("audit20 overlaps an R4 score-eligible role")
    names = _configuration_names(configuration, protocol)
    return protocol, configuration, population, amendment, names, refs


def prepare_pre_target_plan(
    *,
    input_manifest: Path,
    input_manifest_sha256: str,
    repo_root: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = _load_verified_json(
        input_manifest, input_manifest_sha256, "pre-target input"
    )
    protocol, _configuration, population, amendment, names, refs = (
        _verify_effective_contracts(manifest, input_manifest, repo_root.resolve())
    )
    coverage, coverage_sha, r1 = verify_exact_fit105_coverage(
        manifest["fit105_coverage"], population=population, base=input_manifest.parent
    )
    folds = build_fixed_fold_plan(population)
    runner_path = (repo_root.resolve() / RUNNER_SOURCE_RELATIVE).resolve()
    executing_runner_path = Path(__file__).resolve().with_name(
        "g03_division_oof_runner.py"
    )
    if runner_path != executing_runner_path or not runner_path.is_file():
        raise DivisionOOFError(
            "R4 runner path is not the executing orchestrator's repository runner"
        )
    runner_sha = decode.sha256_file(runner_path)
    if runner_sha != PROMOTED_RUNNER_SHA256:
        raise DivisionOOFError(f"promoted R4 runner SHA-256 drifted: {runner_sha}")
    runner_source = _absolute_ref(runner_path, runner_sha)
    payload = {
        "input_manifest": _absolute_ref(input_manifest, input_manifest_sha256),
        "contracts": refs,
        "protocol_identity_sha256": refs["protocol"]["sha256"],
        "population_identity_sha256": str(population["population_sha256"]),
        "population_amendment_sha256": refs["population_amendment"]["sha256"],
        "configuration_count": len(names),
        "configuration_names_sha256": decode.canonical_sha256(list(names)),
        "execution_source_bindings": {"runner": runner_source},
        "selection_metric": protocol["protocol"]["selection_metric"],
        "inner_tie_order": [
            "higher pooled absolute combined graph score",
            "higher pooled division Jaccard",
            "lower pooled division FP",
            "higher pooled adjusted edge Jaccard",
            "higher pooled raw edge Jaccard",
            "higher pooled node recall",
            "fewer pooled predicted forks",
            "lexicographically smallest config name",
        ],
        "deployable_rule": (
            "highest outer-winner count, then lexicographically smallest config name"
        ),
        "deployable_metric_repooling": False,
        "fit105_coverage": {
            "video_count": len(coverage),
            "development_count": DEVELOPMENT_COUNT,
            "held_fit_count": HELD_COUNT,
            "coverage_sha256": coverage_sha,
            "r1_scientific_configuration": r1,
            "records": list(coverage),
        },
        "outer_folds": list(folds),
        "forbidden_roles": [AUDIT_ROLE],
        "target_or_outcome_read": False,
        "score_executed": False,
        "hidden_test_locked_until_passed_local_gate": (
            amendment["association_bank_scope"][
                "hidden_test_inference_only_after_local_confirmation_gate"
            ]
            is True
        ),
    }
    artifact = _artifact(
        PRETARGET_PLAN_SCHEMA, "SEALED_PRE_TARGET_NO_SCORE_OR_GT_READ", payload
    )
    file_sha = atomic_create_json(output, artifact)
    return {
        "status": artifact["status"],
        "path": str(output.resolve()),
        "sha256": file_sha,
        "payload_sha256": artifact["payload_sha256"],
        "fit105_coverage_sha256": coverage_sha,
    }


def _fold_plan(plan: Mapping[str, Any], outer: str) -> Mapping[str, Any]:
    rows = plan.get("outer_folds")
    matches = [
        row
        for row in rows if isinstance(row, Mapping) and row.get("outer_fold") == outer
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise DivisionOOFError(f"pre-target plan lacks unique outer fold {outer}")
    return matches[0]


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DivisionOOFError(f"{label} must be a nonnegative integer")
    return value


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DivisionOOFError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DivisionOOFError(f"{label} must be finite in [0, 1]")
    return result


def _finite_combined(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DivisionOOFError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.1:
        raise DivisionOOFError(f"{label} must be finite in [0, 1.1]")
    return result


def validate_metric_bundle(value: Any, label: str) -> dict[str, Any]:
    """Validate edge and division confusion terms as distinct named objects."""

    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be an object")
    _strict(
        value,
        {"combined_graph_score", "node_recall", "predicted_forks", "edge", "division"},
        label,
    )
    edge = value.get("edge")
    division = value.get("division")
    if not isinstance(edge, Mapping) or not isinstance(division, Mapping):
        raise DivisionOOFError(f"{label} must keep edge and division terms separate")
    _strict(edge, {"tp", "fp", "fn", "jaccard", "adjusted_jaccard"}, f"{label} edge")
    _strict(division, {"tp", "fp", "fn", "jaccard"}, f"{label} division")
    combined = _finite_combined(value["combined_graph_score"], f"{label} combined score")
    node = _finite_unit(value["node_recall"], f"{label} node recall")
    forks = _nonnegative_int(value["predicted_forks"], f"{label} predicted forks")
    edge_counts = {
        key: _nonnegative_int(edge[key], f"{label} edge {key}")
        for key in ("tp", "fp", "fn")
    }
    division_counts = {
        key: _nonnegative_int(division[key], f"{label} division {key}")
        for key in ("tp", "fp", "fn")
    }
    edge_jaccard = _finite_unit(edge["jaccard"], f"{label} edge Jaccard")
    adjusted = _finite_unit(
        edge["adjusted_jaccard"], f"{label} adjusted edge Jaccard"
    )
    division_jaccard = _finite_unit(
        division["jaccard"], f"{label} division Jaccard"
    )
    for counts, observed, term in (
        (edge_counts, edge_jaccard, "edge"),
        (division_counts, division_jaccard, "division"),
    ):
        denominator = counts["tp"] + counts["fp"] + counts["fn"]
        expected = 1.0 if denominator == 0 else counts["tp"] / denominator
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise DivisionOOFError(f"{label} {term} Jaccard disagrees with TP/FP/FN")
    division_denominator = sum(division_counts.values())
    expected_combined = adjusted + (
        0.1 * division_jaccard if division_denominator > 0 else 0.0
    )
    if not math.isclose(combined, expected_combined, rel_tol=0.0, abs_tol=1e-12):
        raise DivisionOOFError(
            f"{label} combined score disagrees with official adjusted-plus-division formula"
        )
    return {
        "combined_graph_score": combined,
        "node_recall": node,
        "predicted_forks": forks,
        "edge": {**edge_counts, "jaccard": edge_jaccard, "adjusted_jaccard": adjusted},
        "division": {**division_counts, "jaccard": division_jaccard},
    }


def _graph_hashes(
    value: Any, identities: Sequence[str], label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(identities):
        raise DivisionOOFError(f"{label} must bind exactly the evaluated identities")
    return {
        identity: _sha(value[identity], f"{label} {identity}")
        for identity in identities
    }


def _validate_parent_fold(
    value: Any, identities: Sequence[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be an object")
    _strict(
        value,
        {
            "reference_label",
            "evaluated_identities",
            "graph_sha256_by_identity",
            "gt_division_count",
            "metrics",
        },
        label,
    )
    expected = list(identities)
    if (
        value.get("reference_label") != SAME_POPULATION_REFERENCE_LABEL
        or list(value.get("evaluated_identities", ())) != expected
    ):
        raise DivisionOOFError(f"{label} same-population parent binding drifted")
    metrics = validate_metric_bundle(value.get("metrics"), f"{label} metrics")
    gt_count = _nonnegative_int(value.get("gt_division_count"), f"{label} GT divisions")
    if gt_count != metrics["division"]["tp"] + metrics["division"]["fn"]:
        raise DivisionOOFError(f"{label} GT division count disagrees with parent TP/FN")
    return {
        "reference_label": SAME_POPULATION_REFERENCE_LABEL,
        "evaluated_identities": expected,
        "graph_sha256_by_identity": _graph_hashes(
            value.get("graph_sha256_by_identity"), expected, f"{label} graph hashes"
        ),
        "gt_division_count": gt_count,
        "metrics": metrics,
    }


def _validate_candidate_fold(
    value: Any,
    identities: Sequence[str],
    *,
    gt_division_count: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be an object")
    _strict(
        value,
        {
            "evaluated_identities",
            "graph_sha256_by_identity",
            "duplicate_survivors",
            "metrics",
        },
        label,
    )
    expected = list(identities)
    if list(value.get("evaluated_identities", ())) != expected:
        raise DivisionOOFError(f"{label} is not the exact same-video evaluation")
    metrics = validate_metric_bundle(value.get("metrics"), f"{label} metrics")
    if metrics["division"]["tp"] + metrics["division"]["fn"] != gt_division_count:
        raise DivisionOOFError(f"{label} GT division count disagrees with candidate TP/FN")
    return {
        "evaluated_identities": expected,
        "graph_sha256_by_identity": _graph_hashes(
            value.get("graph_sha256_by_identity"), expected, f"{label} graph hashes"
        ),
        "duplicate_survivors": _nonnegative_int(
            value.get("duplicate_survivors"), f"{label} duplicate survivors"
        ),
        "metrics": metrics,
    }


def _metric_deltas(candidate: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, float]:
    return {
        "combined_graph_score": float(candidate["combined_graph_score"])
        - float(parent["combined_graph_score"]),
        "adjusted_edge_jaccard": float(candidate["edge"]["adjusted_jaccard"])
        - float(parent["edge"]["adjusted_jaccard"]),
        "raw_edge_jaccard": float(candidate["edge"]["jaccard"])
        - float(parent["edge"]["jaccard"]),
        "node_recall": float(candidate["node_recall"])
        - float(parent["node_recall"]),
    }


def _metric_bundles_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("predicted_forks",):
        if int(left[key]) != int(right[key]):
            return False
    for section, keys in (("edge", ("tp", "fp", "fn")), ("division", ("tp", "fp", "fn"))):
        if any(int(left[section][key]) != int(right[section][key]) for key in keys):
            return False
    scalar_pairs = (
        (left["combined_graph_score"], right["combined_graph_score"]),
        (left["node_recall"], right["node_recall"]),
        (left["edge"]["jaccard"], right["edge"]["jaccard"]),
        (left["edge"]["adjusted_jaccard"], right["edge"]["adjusted_jaccard"]),
        (left["division"]["jaccard"], right["division"]["jaccard"]),
    )
    return all(
        math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in scalar_pairs
    )


def _validate_execution_sources(
    value: Any,
    *,
    scorer_source: Any,
    expected_runner_source: Any,
    base: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DivisionOOFError(f"{label} must be an object")
    _strict(
        value,
        {
            "runner",
            "decoder",
            "orchestrator",
            "scoreall",
            "scoreall_dependencies",
            "one_pass_dependencies",
        },
        label,
    )
    normalized: dict[str, Any] = {}
    for name in ("runner", "decoder", "orchestrator", "scoreall"):
        path, digest = _verified_ref(value[name], base, f"{label} {name}")
        normalized[name] = _absolute_ref(path, digest)
    expected_runner_path, expected_runner_sha = _verified_ref(
        expected_runner_source, Path.cwd(), f"{label} sealed runner"
    )
    repository_runner = Path(__file__).resolve().with_name("g03_division_oof_runner.py")
    if expected_runner_path != repository_runner:
        raise DivisionOOFError(
            f"{label} sealed runner path is not the repository runner"
        )
    if expected_runner_sha != PROMOTED_RUNNER_SHA256:
        raise DivisionOOFError(f"{label} sealed runner is not the promoted runner")
    if (
        Path(normalized["runner"]["path"]) != expected_runner_path
        or normalized["runner"]["sha256"] != expected_runner_sha
    ):
        raise DivisionOOFError(f"{label} runner differs from sealed pre-target binding")
    if (
        Path(normalized["decoder"]["path"]) != Path(decode.__file__).resolve()
        or Path(normalized["orchestrator"]["path"]) != Path(__file__).resolve()
    ):
        raise DivisionOOFError(
            f"{label} decoder/orchestrator paths are not the executing frozen sources"
        )
    scorer_path, scorer_sha = _ref(scorer_source, base, f"{label} scorer source")
    if (
        Path(normalized["scoreall"]["path"]) != scorer_path.resolve()
        or normalized["scoreall"]["sha256"] != scorer_sha
    ):
        raise DivisionOOFError(f"{label} scoreall differs from scorer source")
    dependencies = value.get("scoreall_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(SCORER_DEPENDENCY_PATHS):
        raise DivisionOOFError(f"{label} scorer dependency set drifted")
    normalized["scoreall_dependencies"] = {}
    for relative in SCORER_DEPENDENCY_PATHS:
        path, digest = _verified_ref(
            dependencies[relative], base, f"{label} dependency {relative}"
        )
        normalized["scoreall_dependencies"][relative] = _absolute_ref(path, digest)
    one_pass_dependencies = value.get("one_pass_dependencies")
    if (
        not isinstance(one_pass_dependencies, Mapping)
        or set(one_pass_dependencies) != set(ONE_PASS_DEPENDENCY_PATHS)
    ):
        raise DivisionOOFError(f"{label} one-pass dependency set drifted")
    normalized["one_pass_dependencies"] = {}
    repository_root = Path(__file__).resolve().parents[1]
    for relative in ONE_PASS_DEPENDENCY_PATHS:
        path, digest = _verified_ref(
            one_pass_dependencies[relative],
            base,
            f"{label} one-pass dependency {relative}",
        )
        if path != (repository_root / relative).resolve():
            raise DivisionOOFError(
                f"{label} one-pass dependency path drifted: {relative}"
            )
        normalized["one_pass_dependencies"][relative] = _absolute_ref(path, digest)
    return normalized


def _inner_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["pooled_inner_oof"]
    return (
        -metrics["combined_graph_score"],
        -metrics["division"]["jaccard"],
        metrics["division"]["fp"],
        -metrics["edge"]["adjusted_jaccard"],
        -metrics["edge"]["jaccard"],
        -metrics["node_recall"],
        metrics["predicted_forks"],
        row["configuration_name"],
    )


def select_inner_winner(
    *,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    configuration_names: Sequence[str],
    receipt_base: Path = Path.cwd(),
) -> tuple[str, Mapping[str, Any]]:
    if receipt.get("schema") != INNER_SCORE_SCHEMA or receipt.get("status") != "SEALED_INNER_OOF_SCORES":
        raise DivisionOOFError("inner score receipt schema or status drifted")
    _strict(
        receipt,
        {
            "schema",
            "status",
            "pretarget_plan_sha256",
            "outer_fold",
            "inner_folds",
            "evaluated_identities",
            "configuration_count",
            "scorer_source",
            "execution_sources",
            "parent_folds",
            "rows",
        },
        "inner score receipt",
    )
    outer = str(receipt["outer_fold"])
    fold = _fold_plan(plan, outer)
    if (
        list(receipt["inner_folds"]) != list(fold["inner_folds"])
        or set(receipt["evaluated_identities"]) != set(fold["inner_identities"])
        or len(receipt["evaluated_identities"]) != len(fold["inner_identities"])
        or set(receipt["evaluated_identities"]) & set(fold["outer_identities"])
    ):
        raise DivisionOOFError("inner receipt is not restricted to the four tuning folds")
    _verified_ref(receipt["scorer_source"], receipt_base, "inner scorer source")
    _validate_execution_sources(
        receipt.get("execution_sources"),
        scorer_source=receipt["scorer_source"],
        expected_runner_source=(plan.get("execution_source_bindings") or {}).get(
            "runner"
        ),
        base=receipt_base,
        label="inner execution sources",
    )
    parent_folds_raw = receipt.get("parent_folds")
    if not isinstance(parent_folds_raw, Mapping) or set(parent_folds_raw) != set(fold["inner_folds"]):
        raise DivisionOOFError("inner receipt lacks exact same-video parent fold bindings")
    parent_folds = {
        fold_id: _validate_parent_fold(
            parent_folds_raw[fold_id],
            fold["inner_fold_identities"][fold_id],
            f"inner parent fold {fold_id}",
        )
        for fold_id in fold["inner_folds"]
    }
    expected_names = tuple(configuration_names)
    rows = receipt.get("rows")
    if (
        not isinstance(rows, list)
        or receipt.get("configuration_count") != len(expected_names)
        or len(rows) != len(expected_names)
    ):
        raise DivisionOOFError("inner receipt configuration coverage drifted")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise DivisionOOFError(f"inner score row {index} is malformed")
        _strict(raw, {"configuration_name", "fold_evaluations", "pooled_inner_oof"}, f"inner row {index}")
        fold_evaluations = raw.get("fold_evaluations")
        if not isinstance(fold_evaluations, Mapping) or set(fold_evaluations) != set(fold["inner_folds"]):
            raise DivisionOOFError("inner score row does not contain exactly four folds")
        normalized_folds = {
            key: _validate_candidate_fold(
                fold_evaluations[key],
                fold["inner_fold_identities"][key],
                gt_division_count=parent_folds[key]["gt_division_count"],
                label=f"inner row {index} fold {key}",
            )
            for key in fold["inner_folds"]
        }
        pooled = validate_metric_bundle(
            raw["pooled_inner_oof"], f"inner row {index} pooled"
        )
        recomputed_pooled = _pool_fold_metrics(
            [
                {
                    "evaluated_identities": normalized_folds[key]["evaluated_identities"],
                    "candidate_metrics": normalized_folds[key]["metrics"],
                }
                for key in fold["inner_folds"]
            ],
            "candidate",
        )
        if not _metric_bundles_match(pooled, recomputed_pooled):
            raise DivisionOOFError("inner pooled metric does not match its four fold metrics")
        if any(
            normalized_folds[key]["duplicate_survivors"] != 0
            for key in fold["inner_folds"]
        ):
            raise DivisionOOFError("inner receipt contains a duplicate fork survivor")
        normalized.append(
            {
                "configuration_name": str(raw["configuration_name"]),
                "fold_evaluations": normalized_folds,
                "pooled_inner_oof": pooled,
            }
        )
    names = [row["configuration_name"] for row in normalized]
    if len(set(names)) != len(names) or set(names) != set(expected_names):
        raise DivisionOOFError("inner receipt does not score every sealed config exactly once")
    winner = min(normalized, key=_inner_rank)
    return winner["configuration_name"], winner


def make_inner_winner_artifact(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    configuration_names: Sequence[str],
    receipt_base: Path = Path.cwd(),
) -> dict[str, Any]:
    expected_plan = _sha(plan_sha256, "pre-target plan")
    if receipt.get("pretarget_plan_sha256") != expected_plan:
        raise DivisionOOFError("inner score receipt is bound to a different pre-target plan")
    winner, row = select_inner_winner(
        plan=plan,
        receipt=receipt,
        configuration_names=configuration_names,
        receipt_base=receipt_base,
    )
    fold = _fold_plan(plan, str(receipt["outer_fold"]))
    return _artifact(
        INNER_WINNER_SCHEMA,
        "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
        {
            "pretarget_plan_sha256": expected_plan,
            "inner_score_receipt_sha256": _sha(receipt_sha256, "inner score receipt"),
            "outer_fold": str(receipt["outer_fold"]),
            "outer_identities": list(fold["outer_identities"]),
            "configuration_name": winner,
            "pooled_inner_oof": row["pooled_inner_oof"],
            "selection_order": list(plan["inner_tie_order"]),
            "outer_evaluation_authorized_for_winner_only": True,
            "outer_metric_read_before_winner_seal": False,
        },
    )


def make_outer_result_artifact(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    winner_artifact: Mapping[str, Any],
    winner_sha256: str,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    receipt_base: Path = Path.cwd(),
) -> dict[str, Any]:
    winner = _validate_artifact(
        winner_artifact,
        INNER_WINNER_SCHEMA,
        "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
        "inner winner seal",
    )
    expected_plan = _sha(plan_sha256, "pre-target plan")
    _strict(
        winner,
        {
            "pretarget_plan_sha256",
            "inner_score_receipt_sha256",
            "outer_fold",
            "outer_identities",
            "configuration_name",
            "pooled_inner_oof",
            "selection_order",
            "outer_evaluation_authorized_for_winner_only",
            "outer_metric_read_before_winner_seal",
        },
        "inner winner payload",
    )
    if winner.get("pretarget_plan_sha256") != expected_plan:
        raise DivisionOOFError("inner winner seal is bound to a different pre-target plan")
    _sha(winner.get("inner_score_receipt_sha256"), "inner score receipt")
    validate_metric_bundle(winner.get("pooled_inner_oof"), "sealed inner winner")
    if list(winner.get("selection_order", ())) != list(plan.get("inner_tie_order", ())):
        raise DivisionOOFError("inner winner selection order drifted")
    _strict(
        receipt,
        {
            "schema",
            "status",
            "pretarget_plan_sha256",
            "inner_winner_seal_sha256",
            "outer_fold",
            "configuration_name",
            "evaluated_identities",
            "scorer_source",
            "execution_sources",
            "candidate",
            "parent",
            "gt_division_count",
        },
        "outer holdout receipt",
    )
    outer = str(winner["outer_fold"])
    fold = _fold_plan(plan, outer)
    if (
        receipt.get("schema") != OUTER_SCORE_SCHEMA
        or receipt.get("status") != "SEALED_SINGLE_WINNER_OUTER_HOLDOUT_SCORE"
        or receipt.get("pretarget_plan_sha256") != expected_plan
        or receipt.get("inner_winner_seal_sha256") != _sha(winner_sha256, "winner seal")
        or str(receipt.get("outer_fold")) != outer
        or receipt.get("configuration_name") != winner["configuration_name"]
        or list(receipt.get("evaluated_identities", ())) != list(fold["outer_identities"])
        or list(winner.get("outer_identities", ())) != list(fold["outer_identities"])
        or winner.get("outer_evaluation_authorized_for_winner_only") is not True
        or winner.get("outer_metric_read_before_winner_seal") is not False
    ):
        raise DivisionOOFError("outer receipt is not the sealed winner on its untouched holdout")
    _verified_ref(receipt["scorer_source"], receipt_base, "outer scorer source")
    _validate_execution_sources(
        receipt.get("execution_sources"),
        scorer_source=receipt["scorer_source"],
        expected_runner_source=(plan.get("execution_source_bindings") or {}).get(
            "runner"
        ),
        base=receipt_base,
        label="outer execution sources",
    )
    parent = _validate_parent_fold(
        {
            **dict(receipt["parent"]),
            "gt_division_count": receipt["gt_division_count"],
        }
        if isinstance(receipt.get("parent"), Mapping)
        else receipt.get("parent"),
        fold["outer_identities"],
        "outer holdout parent",
    )
    candidate = _validate_candidate_fold(
        receipt.get("candidate"),
        fold["outer_identities"],
        gt_division_count=parent["gt_division_count"],
        label="outer holdout candidate",
    )
    if receipt.get("gt_division_count") != parent["gt_division_count"]:
        raise DivisionOOFError("outer holdout GT division count drifted")
    deltas = _metric_deltas(candidate["metrics"], parent["metrics"])
    return _artifact(
        OUTER_RESULT_SCHEMA,
        "SEALED_OUTER_HOLDOUT_RESULT",
        {
            "pretarget_plan_sha256": expected_plan,
            "inner_winner_seal_sha256": _sha(winner_sha256, "winner seal"),
            "outer_score_receipt_sha256": _sha(
                receipt_sha256, "outer score receipt"
            ),
            "outer_fold": outer,
            "configuration_name": str(winner["configuration_name"]),
            "evaluated_identities": list(fold["outer_identities"]),
            "candidate": candidate,
            "parent": parent,
            "gt_division_count": parent["gt_division_count"],
            "deltas": deltas,
        },
    )


def _pool_fold_metrics(folds: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    total_videos = sum(len(row["evaluated_identities"]) for row in folds)
    if total_videos <= 0:
        raise DivisionOOFError("development gate has no evaluated videos")
    bundles = [row[f"{side}_metrics"] for row in folds]
    weights = [len(row["evaluated_identities"]) for row in folds]
    edge_counts = {
        key: sum(int(metrics["edge"][key]) for metrics in bundles)
        for key in ("tp", "fp", "fn")
    }
    division_counts = {
        key: sum(int(metrics["division"][key]) for metrics in bundles)
        for key in ("tp", "fp", "fn")
    }

    def video_weighted(path: Sequence[str]) -> float:
        values = []
        for metrics, weight in zip(bundles, weights, strict=True):
            value: Any = metrics
            for key in path:
                value = value[key]
            values.append(float(value) * weight)
        return math.fsum(values) / total_videos

    edge_denominator = sum(edge_counts.values())
    division_denominator = sum(division_counts.values())
    edge_weights = [
        int(metrics["edge"]["tp"])
        + int(metrics["edge"]["fp"])
        + int(metrics["edge"]["fn"])
        for metrics in bundles
    ]
    edge_weight_total = sum(edge_weights)
    if edge_weight_total <= 0:
        raise DivisionOOFError("development gate has no edge denominator")
    adjusted = math.fsum(
        weight * float(metrics["edge"]["adjusted_jaccard"])
        for weight, metrics in zip(edge_weights, bundles, strict=True)
    ) / edge_weight_total
    division_jaccard = (
        1.0
        if division_denominator == 0
        else division_counts["tp"] / division_denominator
    )
    return {
        "combined_graph_score": adjusted
        + (0.1 * division_jaccard if division_denominator > 0 else 0.0),
        "node_recall": video_weighted(("node_recall",)),
        "predicted_forks": sum(int(metrics["predicted_forks"]) for metrics in bundles),
        "edge": {
            **edge_counts,
            "jaccard": 1.0
            if edge_denominator == 0
            else edge_counts["tp"] / edge_denominator,
            "adjusted_jaccard": adjusted,
        },
        "division": {
            **division_counts,
            "jaccard": division_jaccard,
        },
    }


def _development_gate_from_folds(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != len(FOLD_IDS) or [str(row.get("outer_fold")) for row in folds] != list(FOLD_IDS):
        raise DivisionOOFError("development gate requires ordered results for all five folds")
    candidate_pooled = _pool_fold_metrics(folds, "candidate")
    parent_pooled = _pool_fold_metrics(folds, "parent")
    pooled_deltas = _metric_deltas(candidate_pooled, parent_pooled)
    duplicate_survivors = sum(int(row["duplicate_survivors"]) for row in folds)
    checks = {
        "all_folds_gt_division_positive": all(
            int(row["gt_division_count"]) > 0 for row in folds
        ),
        "all_fold_combined_attribution_nonnegative": all(
            float(row["deltas"]["combined_graph_score"]) >= 0.0 for row in folds
        ),
        "pooled_division_tp_positive": candidate_pooled["division"]["tp"] > 0,
        "pooled_division_jaccard_at_least_0_30": (
            candidate_pooled["division"]["jaccard"] >= 0.30
        ),
        "duplicate_survivors_zero": duplicate_survivors == 0,
        "pooled_adjusted_edge_nonregression": (
            pooled_deltas["adjusted_edge_jaccard"] >= 0.0
        ),
        "pooled_raw_edge_nonregression": pooled_deltas["raw_edge_jaccard"] >= 0.0,
        "pooled_node_nonregression": pooled_deltas["node_recall"] >= 0.0,
        "campaign_reference_explicit": True,
        "same_population_reference_exact": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DivisionOOFError(f"development gate failed: {', '.join(failed)}")
    return {
        "status": "PASS",
        "campaign_reference": {
            "label": "APNCC_DIRECT_CAMPAIGN_BEST_CROSS_POPULATION_POLICY_REFERENCE",
            "score": CAMPAIGN_BEST_SCORE,
            "comparison_population": "CROSS_POPULATION_NON_CALIBRATED",
            "binding_for_development_gate": False,
        },
        "same_population_reference_label": SAME_POPULATION_REFERENCE_LABEL,
        "folds": [dict(row) for row in folds],
        "pooled_candidate_metrics": candidate_pooled,
        "pooled_parent_metrics": parent_pooled,
        "pooled_deltas": pooled_deltas,
        "duplicate_survivors": duplicate_survivors,
        "checks": checks,
    }


def validate_development_gate(value: Any) -> dict[str, Any]:
    """Recompute every PASS predicate; caller-supplied check booleans have no authority."""

    if not isinstance(value, Mapping):
        raise DivisionOOFError("development gate must be an object")
    _strict(
        value,
        {
            "status",
            "campaign_reference",
            "same_population_reference_label",
            "folds",
            "pooled_candidate_metrics",
            "pooled_parent_metrics",
            "pooled_deltas",
            "duplicate_survivors",
            "checks",
        },
        "development gate",
    )
    campaign = value.get("campaign_reference")
    if not isinstance(campaign, Mapping):
        raise DivisionOOFError("development gate campaign reference must be an object")
    _strict(
        campaign,
        {"label", "score", "comparison_population", "binding_for_development_gate"},
        "development gate campaign reference",
    )
    if dict(campaign) != {
        "label": "APNCC_DIRECT_CAMPAIGN_BEST_CROSS_POPULATION_POLICY_REFERENCE",
        "score": CAMPAIGN_BEST_SCORE,
        "comparison_population": "CROSS_POPULATION_NON_CALIBRATED",
        "binding_for_development_gate": False,
    }:
        raise DivisionOOFError("development gate campaign reference drifted")
    if value.get("same_population_reference_label") != SAME_POPULATION_REFERENCE_LABEL:
        raise DivisionOOFError("development gate same-population reference label drifted")
    raw_folds = value.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != len(FOLD_IDS):
        raise DivisionOOFError("development gate requires exactly five fold comparisons")
    normalized_folds: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_folds):
        if not isinstance(raw, Mapping):
            raise DivisionOOFError(f"development gate fold {index} is malformed")
        _strict(
            raw,
            {
                "outer_fold",
                "outer_result_sha256",
                "configuration_name",
                "evaluated_identities",
                "gt_division_count",
                "candidate_graph_sha256_by_identity",
                "parent_graph_sha256_by_identity",
                "candidate_metrics",
                "parent_metrics",
                "duplicate_survivors",
                "deltas",
            },
            f"development gate fold {index}",
        )
        fold_id = str(raw.get("outer_fold"))
        if fold_id != FOLD_IDS[index]:
            raise DivisionOOFError("development gate folds are not in frozen order")
        configuration_name = str(raw.get("configuration_name"))
        if not configuration_name or configuration_name == "None":
            raise DivisionOOFError(
                f"development gate fold {fold_id} configuration is missing"
            )
        identities = list(raw.get("evaluated_identities", ()))
        if not identities or any(not isinstance(identity, str) for identity in identities):
            raise DivisionOOFError(f"development gate fold {fold_id} identities are malformed")
        gt_count = _nonnegative_int(raw.get("gt_division_count"), f"development fold {fold_id} GT divisions")
        parent_metrics = validate_metric_bundle(raw.get("parent_metrics"), f"development fold {fold_id} parent")
        candidate_metrics = validate_metric_bundle(raw.get("candidate_metrics"), f"development fold {fold_id} candidate")
        if (
            parent_metrics["division"]["tp"] + parent_metrics["division"]["fn"] != gt_count
            or candidate_metrics["division"]["tp"] + candidate_metrics["division"]["fn"] != gt_count
        ):
            raise DivisionOOFError(f"development fold {fold_id} GT division count drifted")
        deltas = _metric_deltas(candidate_metrics, parent_metrics)
        supplied_deltas = raw.get("deltas")
        if not isinstance(supplied_deltas, Mapping) or set(supplied_deltas) != set(deltas):
            raise DivisionOOFError(f"development fold {fold_id} deltas drifted")
        if any(
            not isinstance(supplied_deltas[key], (int, float))
            or isinstance(supplied_deltas[key], bool)
            or not math.isclose(float(supplied_deltas[key]), expected, rel_tol=0.0, abs_tol=1e-15)
            for key, expected in deltas.items()
        ):
            raise DivisionOOFError(f"development fold {fold_id} deltas drifted")
        normalized_folds.append(
            {
                "outer_fold": fold_id,
                "outer_result_sha256": _sha(
                    raw.get("outer_result_sha256"),
                    f"development fold {fold_id} outer result",
                ),
                "configuration_name": configuration_name,
                "evaluated_identities": identities,
                "gt_division_count": gt_count,
                "candidate_graph_sha256_by_identity": _graph_hashes(
                    raw.get("candidate_graph_sha256_by_identity"), identities, f"development fold {fold_id} candidate graphs"
                ),
                "parent_graph_sha256_by_identity": _graph_hashes(
                    raw.get("parent_graph_sha256_by_identity"), identities, f"development fold {fold_id} parent graphs"
                ),
                "candidate_metrics": candidate_metrics,
                "parent_metrics": parent_metrics,
                "duplicate_survivors": _nonnegative_int(
                    raw.get("duplicate_survivors"), f"development fold {fold_id} duplicate survivors"
                ),
                "deltas": deltas,
            }
        )
    recomputed = _development_gate_from_folds(normalized_folds)
    if decode.canonical_sha256(dict(value)) != decode.canonical_sha256(recomputed):
        raise DivisionOOFError("development gate payload is not the machine-computed PASS result")
    return recomputed


def _validate_deployable_development_gate(
    deployable: Mapping[str, Any],
) -> dict[str, Any]:
    gate = validate_development_gate(deployable.get("development_gate"))
    outer_results = deployable.get("outer_results")
    if not isinstance(outer_results, list) or len(outer_results) != len(FOLD_IDS):
        raise DivisionOOFError("deployable gate binding requires exactly five outer results")
    expected_rows = []
    for fold, gate_fold in zip(FOLD_IDS, gate["folds"], strict=True):
        expected_rows.append(
            {
                "outer_fold": fold,
                "configuration_name": gate_fold["configuration_name"],
                "result_sha256": gate_fold["outer_result_sha256"],
            }
        )
    if outer_results != expected_rows:
        raise DivisionOOFError("deployable gate evidence is not bound to its five outer results")
    winners = [row["configuration_name"] for row in expected_rows]
    counts = dict(sorted(Counter(winners).items()))
    try:
        selected = decode.select_modal_outer_winner(winners)
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc
    if (
        deployable.get("outer_winner_counts") != counts
        or deployable.get("configuration_name") != selected
    ):
        raise DivisionOOFError("deployable modal winner does not match bound outer results")
    return gate


def make_deployable_artifact(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    outer_result_artifacts: Sequence[Mapping[str, Any]],
    outer_result_sha256s: Sequence[str],
    configuration_names: Sequence[str],
) -> dict[str, Any]:
    if len(outer_result_artifacts) != 5 or len(outer_result_sha256s) != 5:
        raise DivisionOOFError("deployable selection requires exactly five outer results")
    expected_plan = _sha(plan_sha256, "pre-target plan")
    by_fold: dict[str, tuple[str, str, dict[str, Any]]] = {}
    winner_seals: set[str] = set()
    allowed_configs = set(configuration_names)
    if not allowed_configs or len(allowed_configs) != len(tuple(configuration_names)):
        raise DivisionOOFError("deployable configuration universe is empty or duplicated")
    for index, (artifact, file_sha) in enumerate(
        zip(outer_result_artifacts, outer_result_sha256s, strict=True)
    ):
        payload = _validate_artifact(
            artifact,
            OUTER_RESULT_SCHEMA,
            "SEALED_OUTER_HOLDOUT_RESULT",
            f"outer result {index}",
        )
        _strict(
            payload,
            {
                "pretarget_plan_sha256",
                "inner_winner_seal_sha256",
                "outer_score_receipt_sha256",
                "outer_fold",
                "configuration_name",
                "evaluated_identities",
                "candidate",
                "parent",
                "gt_division_count",
                "deltas",
            },
            f"outer result {index} payload",
        )
        fold = str(payload.get("outer_fold"))
        winner_seal = _sha(
            payload.get("inner_winner_seal_sha256"), f"outer result {index} winner seal"
        )
        _sha(
            payload.get("outer_score_receipt_sha256"),
            f"outer result {index} score receipt",
        )
        fold_plan = _fold_plan(plan, fold)
        if (
            payload.get("pretarget_plan_sha256") != expected_plan
            or fold in by_fold
            or str(payload.get("configuration_name")) not in allowed_configs
            or list(payload.get("evaluated_identities", ()))
            != list(fold_plan["outer_identities"])
            or winner_seal in winner_seals
        ):
            raise DivisionOOFError("outer result plan binding or fold uniqueness drifted")
        identities = list(fold_plan["outer_identities"])
        parent = _validate_parent_fold(
            payload.get("parent"), identities, f"outer result {index} parent"
        )
        candidate = _validate_candidate_fold(
            payload.get("candidate"),
            identities,
            gt_division_count=parent["gt_division_count"],
            label=f"outer result {index} candidate",
        )
        if payload.get("gt_division_count") != parent["gt_division_count"]:
            raise DivisionOOFError("outer result GT division count drifted")
        deltas = _metric_deltas(candidate["metrics"], parent["metrics"])
        supplied_deltas = payload.get("deltas")
        if (
            not isinstance(supplied_deltas, Mapping)
            or set(supplied_deltas) != set(deltas)
            or any(
                not isinstance(supplied_deltas[key], (int, float))
                or isinstance(supplied_deltas[key], bool)
                or not math.isclose(
                    float(supplied_deltas[key]), expected, rel_tol=0.0, abs_tol=1e-15
                )
                for key, expected in deltas.items()
            )
        ):
            raise DivisionOOFError("outer result explicit deltas drifted")
        winner_seals.add(winner_seal)
        by_fold[fold] = (
            str(payload.get("configuration_name")),
            _sha(file_sha, f"outer result {fold}"),
            {
                "outer_fold": fold,
                "outer_result_sha256": _sha(file_sha, f"outer result {fold}"),
                "configuration_name": str(payload.get("configuration_name")),
                "evaluated_identities": identities,
                "gt_division_count": parent["gt_division_count"],
                "candidate_graph_sha256_by_identity": candidate["graph_sha256_by_identity"],
                "parent_graph_sha256_by_identity": parent["graph_sha256_by_identity"],
                "candidate_metrics": candidate["metrics"],
                "parent_metrics": parent["metrics"],
                "duplicate_survivors": candidate["duplicate_survivors"],
                "deltas": deltas,
            },
        )
    if set(by_fold) != set(FOLD_IDS):
        raise DivisionOOFError("deployable selection lacks all five fixed outer folds")
    winners = [by_fold[fold][0] for fold in FOLD_IDS]
    try:
        selected = decode.select_modal_outer_winner(winners)
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc
    counts = Counter(winners)
    development_gate = _development_gate_from_folds(
        [by_fold[fold][2] for fold in FOLD_IDS]
    )
    artifact = _artifact(
        DEPLOYABLE_SCHEMA,
        DEPLOYABLE_STATUS,
        {
            "pretarget_plan_sha256": expected_plan,
            "outer_results": [
                {
                    "outer_fold": fold,
                    "configuration_name": by_fold[fold][0],
                    "result_sha256": by_fold[fold][1],
                }
                for fold in FOLD_IDS
            ],
            "outer_winner_counts": dict(sorted(counts.items())),
            "configuration_name": selected,
            "selection_rule": plan["deployable_rule"],
            "outer_or_held_metrics_used_for_modal_tie": False,
            "metric_repooling_for_deployable_selection": False,
            "development_gate": development_gate,
        },
    )
    _validate_deployable_development_gate(artifact["payload"])
    return artifact


def _decimal_string(value: Any, label: str) -> str:
    result = str(value)
    if not DECIMAL_RE.fullmatch(result):
        raise DivisionOOFError(f"{label} must be a nonnegative decimal string")
    try:
        parsed = Decimal(result)
    except InvalidOperation as exc:  # pragma: no cover - regex is already restrictive.
        raise DivisionOOFError(f"{label} is malformed") from exc
    if not parsed.is_finite():
        raise DivisionOOFError(f"{label} must be finite")
    return result


def exact_closed_reference_set(
    amendment: Mapping[str, Any], r1_binding: Mapping[str, Any], *, base: Path
) -> list[dict[str, str]]:
    references = amendment.get("held_fit_reference_set")
    if not isinstance(references, list) or len(references) != 2:
        raise DivisionOOFError("effective amendment lacks exactly two held-fit references")
    apncc = [row for row in references if isinstance(row, Mapping) and row.get("id") == APNCC_REFERENCE["id"]]
    r1_template = [row for row in references if isinstance(row, Mapping) and row.get("id") == R1_REFERENCE_ID]
    if len(apncc) != 1 or dict(apncc[0]) != APNCC_REFERENCE or len(r1_template) != 1:
        raise DivisionOOFError("APNCC held-fit reference set drifted")
    template = r1_template[0]
    if (
        template.get("score") != "BIND_FROM_VALIDATED_R1_RESULT_BEFORE_R4_HELD_FIT_ACCESS"
        or template.get("aggregate_graph_sha256")
        != "BIND_FROM_R1_INFERENCE_SEAL_BEFORE_R4_HELD_FIT_ACCESS"
    ):
        raise DivisionOOFError("R1 closed-reference placeholder drifted")
    _strict(
        r1_binding,
        {
            "schema",
            "status",
            "scientific_configuration_name",
            "score",
            "aggregate_graph_sha256",
            "validated_result_report",
            "inference_seal",
        },
        "validated R1 binding",
    )
    if (
        r1_binding.get("schema") != R1_BINDING_SCHEMA
        or r1_binding.get("status") != "PASS_VALIDATED_COMPLETE"
    ):
        raise DivisionOOFError("R1 result is not validated and complete")
    _verified_ref(r1_binding["validated_result_report"], base, "validated R1 report")
    _verified_ref(r1_binding["inference_seal"], base, "validated R1 inference seal")
    return [
        dict(APNCC_REFERENCE),
        {
            "id": R1_REFERENCE_ID,
            "score": _decimal_string(r1_binding["score"], "R1 score"),
            "aggregate_graph_sha256": _sha(
                r1_binding["aggregate_graph_sha256"], "R1 aggregate graph"
            ),
        },
    ]


def make_held_marker_artifact(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    deployable_artifact: Mapping[str, Any],
    deployable_sha256: str,
    amendment: Mapping[str, Any],
    inventory: Mapping[str, Any],
    r1_binding: Mapping[str, Any],
    base: Path,
    r1_base: Path | None = None,
) -> dict[str, Any]:
    deployable = _validate_artifact(
        deployable_artifact,
        DEPLOYABLE_SCHEMA,
        DEPLOYABLE_STATUS,
        "deployable seal",
    )
    _strict(
        deployable,
        {
            "pretarget_plan_sha256",
            "outer_results",
            "outer_winner_counts",
            "configuration_name",
            "selection_rule",
            "outer_or_held_metrics_used_for_modal_tie",
            "metric_repooling_for_deployable_selection",
            "development_gate",
        },
        "deployable payload",
    )
    if (
        deployable.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or deployable.get("selection_rule") != plan.get("deployable_rule")
        or deployable.get("outer_or_held_metrics_used_for_modal_tie") is not False
        or deployable.get("metric_repooling_for_deployable_selection") is not False
    ):
        raise DivisionOOFError("deployable selection contract drifted")
    _validate_deployable_development_gate(deployable)
    _strict(
        inventory,
        {
            "schema",
            "status",
            "pretarget_plan_sha256",
            "deployable_seal_sha256",
            "configuration_name",
            "overlay_prediction_seal",
            "records",
        },
        "held-fit inventory",
    )
    if (
        inventory.get("schema") != HELD_INVENTORY_SCHEMA
        or inventory.get("status") != "SEALED_BEFORE_HELD_FIT_TARGET_ACCESS"
        or inventory.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or inventory.get("deployable_seal_sha256") != _sha(deployable_sha256, "deployable seal")
        or inventory.get("configuration_name") != deployable["configuration_name"]
    ):
        raise DivisionOOFError("held-fit inventory binding drifted")
    overlay_seal_path, overlay_seal_sha = _verified_ref(
        inventory["overlay_prediction_seal"], base, "held-fit overlay prediction seal"
    )
    if overlay_seal_path.name != "SEAL.json":
        raise DivisionOOFError("held-fit overlay prediction reference is not SEAL.json")
    try:
        overlay_summary = decode.verify_sealed_output(
            overlay_seal_path.parent,
            expected_seal_sha256=overlay_seal_sha,
        )
    except decode.DivisionDecodeError as exc:
        raise DivisionOOFError(str(exc)) from exc
    if (
        overlay_summary.get("status") != "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING"
        or overlay_summary.get("configuration_name") != deployable["configuration_name"]
        or overlay_summary.get("video_count") != HELD_COUNT
    ):
        raise DivisionOOFError("held-fit overlay prediction seal scope drifted")
    overlay_seal = _load_json(overlay_seal_path, "held-fit overlay prediction seal")
    overlay_graph_rows = overlay_seal.get("graphs")
    if not isinstance(overlay_graph_rows, list):
        raise DivisionOOFError("held-fit overlay prediction seal lacks graph inventory")
    sealed_overlay_graphs = {
        str(row.get("identity")): row
        for row in overlay_graph_rows
        if isinstance(row, Mapping)
    }
    coverage = plan.get("fit105_coverage")
    coverage_rows = coverage.get("records") if isinstance(coverage, Mapping) else None
    held_parent = {
        str(row["identity"]): row["parent_graph"]
        for row in coverage_rows
        if isinstance(row, Mapping) and row.get("role") == HELD_ROLE
    } if isinstance(coverage_rows, list) else {}
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != HELD_COUNT:
        raise DivisionOOFError("held-fit inventory must contain exactly 36 records")
    normalized = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise DivisionOOFError(f"held-fit row {index} is malformed")
        _strict(row, {"identity", "parent_graph", "overlay_graph"}, f"held-fit row {index}")
        identity = str(row["identity"])
        if identity in seen or identity not in held_parent:
            raise DivisionOOFError("held-fit inventory is not exact held36")
        seen.add(identity)
        parent_path, parent_sha = _ref(row["parent_graph"], base, f"held parent {identity}")
        sealed_parent_path, sealed_parent_sha = _ref(
            held_parent[identity], Path.cwd(), f"sealed held parent {identity}"
        )
        if (
            parent_path.resolve() != sealed_parent_path.resolve()
            or parent_sha != sealed_parent_sha
        ):
            raise DivisionOOFError("held-fit parent graph differs from fit105 seal")
        _verified_ref(row["parent_graph"], base, f"held parent {identity}")
        overlay_path, overlay_sha = _verified_ref(
            row["overlay_graph"], base, f"held overlay {identity}"
        )
        sealed_overlay = sealed_overlay_graphs.get(identity)
        if (
            not isinstance(sealed_overlay, Mapping)
            or (overlay_seal_path.parent / str(sealed_overlay.get("path"))).resolve()
            != overlay_path.resolve()
            or sealed_overlay.get("sha256") != overlay_sha
        ):
            raise DivisionOOFError("held overlay graph differs from decoder prediction seal")
        normalized.append(
            {
                "identity": identity,
                "parent_graph": _absolute_ref(parent_path, parent_sha),
                "overlay_graph": _absolute_ref(overlay_path, overlay_sha),
            }
        )
    if seen != set(held_parent):
        raise DivisionOOFError("held-fit inventory is not exact held36")
    if set(sealed_overlay_graphs) != seen:
        raise DivisionOOFError("decoder prediction seal is not exact held36")
    audit_identities = set(plan.get("forbidden_roles", ()))
    if AUDIT_ROLE not in audit_identities:
        raise DivisionOOFError("pre-target plan does not preserve audit20 refusal")
    r1_name = coverage["r1_scientific_configuration"]["name"]
    if r1_binding.get("scientific_configuration_name") != r1_name:
        raise DivisionOOFError("held reference is not the sealed R1 scientific configuration")
    references = exact_closed_reference_set(
        amendment, r1_binding, base=base if r1_base is None else r1_base
    )
    normalized.sort(key=lambda row: row["identity"])
    return _artifact(
        HELD_MARKER_SCHEMA,
        "MARKER_WRITTEN_BEFORE_HELD_FIT_TARGET_OPEN",
        {
            "pretarget_plan_sha256": _sha(plan_sha256, "pre-target plan"),
            "deployable_seal_sha256": _sha(deployable_sha256, "deployable seal"),
            "configuration_name": deployable["configuration_name"],
            "population_role": HELD_ROLE,
            "family": "44b6",
            "video_count": HELD_COUNT,
            "overlay_prediction_seal": _absolute_ref(overlay_seal_path, overlay_seal_sha),
            "graph_inventory_sha256": decode.canonical_sha256(normalized),
            "graphs": normalized,
            "closed_reference_set": references,
            "closed_reference_ids": [APNCC_REFERENCE["id"], R1_REFERENCE_ID],
            "parent_and_overlay_scored_in_same_single_access_required": True,
            "allowed_target_accesses": 1,
            "target_opened": False,
            "scientifically_valid_retry_allowed": False,
        },
    )


def write_held_fit_marker(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, str]:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in HELD_SENTINELS if (run_dir / name).exists() or (run_dir / name).is_symlink()]
    if existing:
        raise DivisionOOFError(f"held-fit one-shot already consumed or tainted: {existing}")
    marker = run_dir / HELD_SENTINELS[0]
    sha = atomic_create_json(marker, artifact)
    return {"path": str(marker), "sha256": sha}


def make_hidden_marker_artifact(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    deployable_artifact: Mapping[str, Any],
    deployable_sha256: str,
    held_marker_sha256: str,
    local_gate: Mapping[str, Any],
    local_gate_sha256: str,
    inventory: Mapping[str, Any],
    population: Mapping[str, Any],
    base: Path,
) -> dict[str, Any]:
    deployable = _validate_artifact(
        deployable_artifact,
        DEPLOYABLE_SCHEMA,
        DEPLOYABLE_STATUS,
        "deployable seal",
    )
    _strict(
        deployable,
        {
            "pretarget_plan_sha256",
            "outer_results",
            "outer_winner_counts",
            "configuration_name",
            "selection_rule",
            "outer_or_held_metrics_used_for_modal_tie",
            "metric_repooling_for_deployable_selection",
            "development_gate",
        },
        "deployable payload",
    )
    if (
        deployable.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or deployable.get("outer_or_held_metrics_used_for_modal_tie") is not False
        or deployable.get("metric_repooling_for_deployable_selection") is not False
    ):
        raise DivisionOOFError("hidden marker deployable binding drifted")
    _validate_deployable_development_gate(deployable)
    _strict(
        local_gate,
        {
            "schema",
            "status",
            "pretarget_plan_sha256",
            "deployable_seal_sha256",
            "held_fit_access_marker_sha256",
            "checks",
        },
        "local confirmation gate",
    )
    checks = local_gate.get("checks")
    required_checks = {
        "same_population_score_pass",
        "division_jaccard_pass",
        "division_tp_positive",
        "duplicate_survivors_zero",
        "edge_non_regression",
        "node_non_regression",
        "all_outer_folds_nonnegative_delta",
    }
    if (
        local_gate.get("schema") != LOCAL_GATE_SCHEMA
        or local_gate.get("status") != "PASS_LOCAL_CONFIRMATION_GATE"
        or local_gate.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or local_gate.get("deployable_seal_sha256") != _sha(deployable_sha256, "deployable seal")
        or local_gate.get("held_fit_access_marker_sha256") != _sha(held_marker_sha256, "held marker")
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(checks[key] is not True for key in required_checks)
    ):
        raise DivisionOOFError("hidden4 remains locked: local confirmation gate did not pass")
    _strict(
        inventory,
        {
            "schema",
            "status",
            "pretarget_plan_sha256",
            "deployable_seal_sha256",
            "configuration_name",
            "submission_zip",
            "records",
        },
        "hidden4 inventory",
    )
    if (
        inventory.get("schema") != HIDDEN_INVENTORY_SCHEMA
        or inventory.get("status") != "SEALED_BEFORE_HIDDEN_PROBE"
        or inventory.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or inventory.get("deployable_seal_sha256") != _sha(deployable_sha256, "deployable seal")
        or inventory.get("configuration_name") != deployable["configuration_name"]
    ):
        raise DivisionOOFError("hidden4 inventory binding drifted")
    hidden = set(_role_identities(population, HIDDEN_ROLE))
    audit = set(_role_identities(population, AUDIT_ROLE))
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != HIDDEN_COUNT:
        raise DivisionOOFError("hidden probe inventory must contain exactly hidden4")
    normalized = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise DivisionOOFError(f"hidden row {index} is malformed")
        _strict(row, {"identity", "association_bank", "decoded_graph"}, f"hidden row {index}")
        identity = str(row["identity"])
        if identity in seen or identity not in hidden or identity in audit:
            raise DivisionOOFError("hidden inventory contains non-hidden or audit identity")
        seen.add(identity)
        bank_path, bank_sha = _verified_ref(row["association_bank"], base, f"hidden bank {identity}")
        graph_path, graph_sha = _verified_ref(row["decoded_graph"], base, f"hidden graph {identity}")
        normalized.append(
            {
                "identity": identity,
                "association_bank": _absolute_ref(bank_path, bank_sha),
                "decoded_graph": _absolute_ref(graph_path, graph_sha),
            }
        )
    if seen != hidden:
        raise DivisionOOFError("hidden inventory is not exact hidden4")
    submission_path, submission_sha = _verified_ref(
        inventory["submission_zip"], base, "hidden submission ZIP"
    )
    normalized.sort(key=lambda row: row["identity"])
    return _artifact(
        HIDDEN_MARKER_SCHEMA,
        "MARKER_WRITTEN_AFTER_LOCAL_PASS_BEFORE_SUBMISSION",
        {
            "pretarget_plan_sha256": _sha(plan_sha256, "pre-target plan"),
            "deployable_seal_sha256": _sha(deployable_sha256, "deployable seal"),
            "held_fit_access_marker_sha256": _sha(held_marker_sha256, "held marker"),
            "local_gate_sha256": _sha(local_gate_sha256, "local gate"),
            "configuration_name": deployable["configuration_name"],
            "population_role": HIDDEN_ROLE,
            "video_count": HIDDEN_COUNT,
            "records": normalized,
            "submission_zip": _absolute_ref(submission_path, submission_sha),
            "submission_made": False,
            "target_or_hidden_outcome_read": False,
        },
    )


def _load_plan(path: Path, sha256: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    artifact = _load_verified_json(path, sha256, "pre-target plan")
    payload = _validate_artifact(
        artifact,
        PRETARGET_PLAN_SCHEMA,
        "SEALED_PRE_TARGET_NO_SCORE_OR_GT_READ",
        "pre-target plan",
    )
    return artifact, payload


def _names_from_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    contracts = plan.get("contracts")
    if not isinstance(contracts, Mapping):
        raise DivisionOOFError("pre-target plan contract references are missing")
    config_path, config_sha = _verified_ref(
        contracts.get("configuration_manifest"), Path.cwd(), "configuration manifest"
    )
    protocol_path, _protocol_sha = _verified_ref(
        contracts.get("protocol"), Path.cwd(), "R4 protocol"
    )
    names = _configuration_names(
        _load_json(config_path, "configuration manifest"),
        _load_json(protocol_path, "R4 protocol"),
    )
    if (
        plan.get("configuration_count") != len(names)
        or plan.get("configuration_names_sha256")
        != decode.canonical_sha256(list(names))
        or config_sha != contracts["configuration_manifest"]["sha256"]
    ):
        raise DivisionOOFError("pre-target plan configuration binding drifted")
    return names


def seal_inner_winner_from_files(
    *,
    plan_path: Path,
    plan_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    output: Path,
) -> dict[str, str]:
    _plan_artifact, plan = _load_plan(plan_path, plan_sha256)
    receipt = _load_verified_json(receipt_path, receipt_sha256, "inner score receipt")
    artifact = make_inner_winner_artifact(
        plan=plan,
        plan_sha256=plan_sha256,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        configuration_names=_names_from_plan(plan),
        receipt_base=receipt_path.parent,
    )
    output_sha = atomic_create_json(output, artifact)
    return {"path": str(output.resolve()), "sha256": output_sha}


def seal_outer_result_from_files(
    *,
    plan_path: Path,
    plan_sha256: str,
    winner_path: Path,
    winner_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    output: Path,
) -> dict[str, str]:
    _plan_artifact, plan = _load_plan(plan_path, plan_sha256)
    winner = _load_verified_json(winner_path, winner_sha256, "inner winner seal")
    receipt = _load_verified_json(receipt_path, receipt_sha256, "outer score receipt")
    artifact = make_outer_result_artifact(
        plan=plan,
        plan_sha256=plan_sha256,
        winner_artifact=winner,
        winner_sha256=winner_sha256,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        receipt_base=receipt_path.parent,
    )
    output_sha = atomic_create_json(output, artifact)
    return {"path": str(output.resolve()), "sha256": output_sha}


def seal_deployable_from_files(
    *,
    plan_path: Path,
    plan_sha256: str,
    outer_results: Sequence[tuple[Path, str]],
    output: Path,
) -> dict[str, str]:
    _plan_artifact, plan = _load_plan(plan_path, plan_sha256)
    artifacts = [
        _load_verified_json(path, expected, f"outer result {index}")
        for index, (path, expected) in enumerate(outer_results)
    ]
    artifact = make_deployable_artifact(
        plan=plan,
        plan_sha256=plan_sha256,
        outer_result_artifacts=artifacts,
        outer_result_sha256s=[expected for _path, expected in outer_results],
        configuration_names=_names_from_plan(plan),
    )
    output_sha = atomic_create_json(output, artifact)
    return {"path": str(output.resolve()), "sha256": output_sha}


def prepare_held_marker_from_files(
    *,
    plan_path: Path,
    plan_sha256: str,
    deployable_path: Path,
    deployable_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
    r1_binding_path: Path,
    r1_binding_sha256: str,
    run_dir: Path,
) -> dict[str, str]:
    _plan_artifact, plan = _load_plan(plan_path, plan_sha256)
    deployable = _load_verified_json(deployable_path, deployable_sha256, "deployable seal")
    inventory = _load_verified_json(inventory_path, inventory_sha256, "held-fit inventory")
    r1_binding = _load_verified_json(r1_binding_path, r1_binding_sha256, "validated R1 binding")
    contracts = plan.get("contracts")
    if not isinstance(contracts, Mapping):
        raise DivisionOOFError("pre-target plan contract references are missing")
    amendment_path, _amendment_sha = _verified_ref(
        contracts.get("population_amendment"), Path.cwd(), "R4 population amendment"
    )
    amendment = _load_json(amendment_path, "R4 population amendment")
    artifact = make_held_marker_artifact(
        plan=plan,
        plan_sha256=plan_sha256,
        deployable_artifact=deployable,
        deployable_sha256=deployable_sha256,
        amendment=amendment,
        inventory=inventory,
        r1_binding=r1_binding,
        base=inventory_path.parent,
        r1_base=r1_binding_path.parent,
    )
    return write_held_fit_marker(run_dir, artifact)


def write_hidden_probe_marker(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, str]:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        name
        for name in HIDDEN_SENTINELS
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]
    if existing:
        raise DivisionOOFError(f"hidden probe already consumed or tainted: {existing}")
    marker = run_dir / HIDDEN_SENTINELS[0]
    sha = atomic_create_json(marker, artifact)
    return {"path": str(marker), "sha256": sha}


def prepare_hidden_marker_from_files(
    *,
    plan_path: Path,
    plan_sha256: str,
    deployable_path: Path,
    deployable_sha256: str,
    held_marker_path: Path,
    held_marker_sha256: str,
    local_gate_path: Path,
    local_gate_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
    run_dir: Path,
) -> dict[str, str]:
    _plan_artifact, plan = _load_plan(plan_path, plan_sha256)
    deployable = _load_verified_json(deployable_path, deployable_sha256, "deployable seal")
    held_marker = _load_verified_json(held_marker_path, held_marker_sha256, "held-fit marker")
    held_payload = _validate_artifact(
        held_marker,
        HELD_MARKER_SCHEMA,
        "MARKER_WRITTEN_BEFORE_HELD_FIT_TARGET_OPEN",
        "held-fit marker",
    )
    if (
        held_payload.get("pretarget_plan_sha256") != _sha(plan_sha256, "pre-target plan")
        or held_payload.get("deployable_seal_sha256")
        != _sha(deployable_sha256, "deployable seal")
    ):
        raise DivisionOOFError("held marker does not bind this plan and deployable")
    local_gate = _load_verified_json(local_gate_path, local_gate_sha256, "local confirmation gate")
    inventory = _load_verified_json(inventory_path, inventory_sha256, "hidden4 inventory")
    contracts = plan.get("contracts")
    if not isinstance(contracts, Mapping):
        raise DivisionOOFError("pre-target plan contract references are missing")
    population_path, _population_sha = _verified_ref(
        contracts.get("population_manifest"), Path.cwd(), "R4 population manifest"
    )
    population = _load_json(population_path, "R4 population manifest")
    artifact = make_hidden_marker_artifact(
        plan=plan,
        plan_sha256=plan_sha256,
        deployable_artifact=deployable,
        deployable_sha256=deployable_sha256,
        held_marker_sha256=held_marker_sha256,
        local_gate=local_gate,
        local_gate_sha256=local_gate_sha256,
        inventory=inventory,
        population=population,
        base=inventory_path.parent,
    )
    return write_hidden_probe_marker(run_dir, artifact)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="write the sealed pre-target plan")
    prepare.add_argument("--repo-root", required=True, type=Path)
    prepare.add_argument("--input-manifest", required=True, type=Path)
    prepare.add_argument("--input-manifest-sha256", required=True)
    prepare.add_argument("--output", required=True, type=Path)

    inner = subcommands.add_parser(
        "seal-inner-winner", help="freeze one four-fold inner-OOF winner"
    )
    inner.add_argument("--plan", required=True, type=Path)
    inner.add_argument("--plan-sha256", required=True)
    inner.add_argument("--score-receipt", required=True, type=Path)
    inner.add_argument("--score-receipt-sha256", required=True)
    inner.add_argument("--output", required=True, type=Path)

    outer = subcommands.add_parser(
        "seal-outer-result", help="record the presealed winner on one outer holdout"
    )
    outer.add_argument("--plan", required=True, type=Path)
    outer.add_argument("--plan-sha256", required=True)
    outer.add_argument("--winner", required=True, type=Path)
    outer.add_argument("--winner-sha256", required=True)
    outer.add_argument("--score-receipt", required=True, type=Path)
    outer.add_argument("--score-receipt-sha256", required=True)
    outer.add_argument("--output", required=True, type=Path)

    deploy = subcommands.add_parser(
        "seal-deployable", help="choose modal outer winner with lexicographic tie only"
    )
    deploy.add_argument("--plan", required=True, type=Path)
    deploy.add_argument("--plan-sha256", required=True)
    deploy.add_argument(
        "--outer-result",
        action="append",
        nargs=2,
        metavar=("PATH", "SHA256"),
        required=True,
    )
    deploy.add_argument("--output", required=True, type=Path)

    held = subcommands.add_parser(
        "prepare-held-marker",
        help="write the held-fit36 marker only after recomputing development PASS",
    )
    held.add_argument("--plan", required=True, type=Path)
    held.add_argument("--plan-sha256", required=True)
    held.add_argument("--deployable", required=True, type=Path)
    held.add_argument("--deployable-sha256", required=True)
    held.add_argument("--inventory", required=True, type=Path)
    held.add_argument("--inventory-sha256", required=True)
    held.add_argument("--r1-binding", required=True, type=Path)
    held.add_argument("--r1-binding-sha256", required=True)
    held.add_argument("--run-dir", required=True, type=Path)

    hidden = subcommands.add_parser(
        "prepare-hidden-marker", help="write a hidden4 marker only after local PASS"
    )
    hidden.add_argument("--plan", required=True, type=Path)
    hidden.add_argument("--plan-sha256", required=True)
    hidden.add_argument("--deployable", required=True, type=Path)
    hidden.add_argument("--deployable-sha256", required=True)
    hidden.add_argument("--held-marker", required=True, type=Path)
    hidden.add_argument("--held-marker-sha256", required=True)
    hidden.add_argument("--local-gate", required=True, type=Path)
    hidden.add_argument("--local-gate-sha256", required=True)
    hidden.add_argument("--inventory", required=True, type=Path)
    hidden.add_argument("--inventory-sha256", required=True)
    hidden.add_argument("--run-dir", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_pre_target_plan(
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            repo_root=args.repo_root,
            output=args.output,
        )
    elif args.command == "seal-inner-winner":
        report = seal_inner_winner_from_files(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            receipt_path=args.score_receipt,
            receipt_sha256=args.score_receipt_sha256,
            output=args.output,
        )
    elif args.command == "seal-outer-result":
        report = seal_outer_result_from_files(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            winner_path=args.winner,
            winner_sha256=args.winner_sha256,
            receipt_path=args.score_receipt,
            receipt_sha256=args.score_receipt_sha256,
            output=args.output,
        )
    elif args.command == "seal-deployable":
        if len(args.outer_result) != 5:
            raise DivisionOOFError("seal-deployable requires five --outer-result pairs")
        report = seal_deployable_from_files(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            outer_results=[(Path(path), sha) for path, sha in args.outer_result],
            output=args.output,
        )
    elif args.command == "prepare-held-marker":
        report = prepare_held_marker_from_files(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            deployable_path=args.deployable,
            deployable_sha256=args.deployable_sha256,
            inventory_path=args.inventory,
            inventory_sha256=args.inventory_sha256,
            r1_binding_path=args.r1_binding,
            r1_binding_sha256=args.r1_binding_sha256,
            run_dir=args.run_dir,
        )
    elif args.command == "prepare-hidden-marker":
        report = prepare_hidden_marker_from_files(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            deployable_path=args.deployable,
            deployable_sha256=args.deployable_sha256,
            held_marker_path=args.held_marker,
            held_marker_sha256=args.held_marker_sha256,
            local_gate_path=args.local_gate,
            local_gate_sha256=args.local_gate_sha256,
            inventory_path=args.inventory,
            inventory_sha256=args.inventory_sha256,
            run_dir=args.run_dir,
        )
    else:  # pragma: no cover - argparse owns this invariant.
        raise DivisionOOFError(f"unsupported command: {args.command}")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

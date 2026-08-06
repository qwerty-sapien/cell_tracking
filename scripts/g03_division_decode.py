#!/usr/bin/env python3
"""Deterministic, target-blind G-03 R4 division decoding on CPU.

The only execution input is a sealed manifest of association-bank and base-graph
hashes.  This module has no target, scorer, audit, or CV57 interface.  It verifies
the sealed R4 protocol and all frozen implementation dependencies before importing
the N1 compiler, generates predictions from observations only, and writes a final
seal only after every prediction record and decoded graph exists.

Persistence has two deliberately observable meanings. ``daughter_persistence``
requires that many one-frame steps on each selected base-graph branch;
``grandchild_persistence`` requires that many deterministic, highest-probability
steps in the complete association bank.  Evidence nodes are owned by the event but
are never emitted as invented edges.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


INPUT_SCHEMA = "celltrack-alpha.g03-r4-division-decode-input.v1"
BANK_SCHEMA = "celltrack-alpha.g03-r1-association-bank.v1"
CONFIG_SCHEMA = "celltrack-alpha.g03-division-decode-config-manifest.v1"
PROTOCOL_SCHEMA = "celltrack-alpha.g03-r4-division-protocol-contract.v1"
POPULATION_SCHEMA = "celltrack-alpha.g03-r4-division-population.v1"
POPULATION_AMENDMENT_SCHEMA = (
    "celltrack-alpha.g03-r4-division-population-amendment.v1"
)
PREDICTION_SCHEMA = "celltrack-alpha.g03-r4-division-predictions.v1"
SEAL_SCHEMA = "celltrack-alpha.g03-r4-division-prediction-seal.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_ACTION_COMPILER_SHA256 = (
    "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"
)
FROZEN_GRAPH_MODEL_SHA256 = (
    "d8f93181c38a319c79c8075fd730281d9aa8521ddefb6fdae8f64b5b27c373e4"
)


class DivisionDecodeError(RuntimeError):
    """The target-blind decode contract failed closed."""


@dataclass(frozen=True)
class DivisionDecodeConfig:
    name: str
    divisions: bool
    fork_threshold: float | None
    max_primary_secondary_gap: float | None
    late_window: int
    daughter_persistence: int
    grandchild_persistence: int
    duplicate_temporal_window: int
    per_video_cap: int | None


@dataclass(frozen=True)
class BankNode:
    node_id: int
    frame: int
    coords_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class CandidateEdge:
    source: int
    target: int
    probability: float
    distance_um: float


@dataclass(frozen=True)
class AssociationBank:
    identity: str
    nodes: Mapping[int, BankNode]
    edges: tuple[CandidateEdge, ...]


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    equivalence_key: str
    identity: str
    mode: str
    late_depth: int
    anchor_parent: int
    emitted_parent: int
    primary_child: int
    secondary_child: int
    children: tuple[int, int]
    parent_lineage: int
    daughter_lineages: tuple[int, int]
    primary_probability: float
    secondary_probability: float
    primary_secondary_gap: float
    continuation_nodes: tuple[int, ...]
    daughter_persistence_nodes: tuple[int, ...]
    grandchild_persistence_nodes: tuple[int, ...]
    occupied_nodes: tuple[int, ...]


@dataclass(frozen=True)
class FrozenIncidentIndex:
    """Exact source/target incidence over one immutable baseline graph."""

    edges: frozenset[tuple[int, int]]
    by_source: Mapping[int, tuple[tuple[int, int], ...]]
    by_target: Mapping[int, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class DecodeRun:
    predictions: Mapping[str, Any]
    graphs: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if not SHA256_RE.fullmatch(result):
        raise DivisionDecodeError(f"{label} is not a lowercase SHA-256")
    return result


def _verify_file(path: Path, expected: str, label: str) -> str:
    wanted = _require_sha256(expected, f"{label} hash")
    if not path.is_file():
        raise DivisionDecodeError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != wanted:
        raise DivisionDecodeError(
            f"{label} SHA-256 mismatch: expected {wanted}, observed {observed}"
        )
    return observed


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DivisionDecodeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DivisionDecodeError(f"{label} must be a JSON object")
    return value


def _load_verified_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    _verify_file(path, expected, label)
    return _load_json(path, label)


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise DivisionDecodeError(
            f"{label} keys drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _resolve(base: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else base / path


def _reference(value: Any, base: Path, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise DivisionDecodeError(f"{label} reference must be an object")
    _strict_keys(value, {"path", "sha256"}, f"{label} reference")
    return _resolve(base, value["path"]), _require_sha256(
        value["sha256"], f"{label} reference"
    )


def _repo_path(repo_root: Path, raw: Any) -> Path:
    candidate = Path(str(raw))
    return candidate if candidate.is_absolute() else repo_root / candidate


def _verify_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    protocol_sha256: str,
    configuration_path: Path,
    configuration_sha256: str,
    repo_root: Path,
) -> None:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "SEALED_BEFORE_ANY_G03_DIVISION_OUTCOME"
    ):
        raise DivisionDecodeError("R4 protocol is not the sealed pre-outcome contract")
    rules = protocol.get("protocol")
    if not isinstance(rules, Mapping) or any(
        (
            rules.get("adaptive_search_forbidden") is not True,
            rules.get("configuration_splicing_forbidden") is not True,
            rules.get("duplicate_survivors_required") != 0,
            rules.get("gt_for_candidate_generation_forbidden") is not True,
        )
    ):
        raise DivisionDecodeError("R4 target-blind protocol invariants drifted")

    config_record = protocol.get("configuration_manifest")
    if not isinstance(config_record, Mapping):
        raise DivisionDecodeError("R4 protocol has no configuration-manifest record")
    if config_record.get("file_sha256") != configuration_sha256:
        raise DivisionDecodeError("configuration manifest is not protocol-pinned")

    design_path = _repo_path(repo_root, rules.get("design_path"))
    _verify_file(design_path, str(rules.get("design_sha256")), "sealed R4 design")
    generator_path = _repo_path(repo_root, config_record.get("generator_path"))
    _verify_file(
        generator_path,
        str(config_record.get("generator_sha256")),
        "sealed configuration generator",
    )
    dependencies = protocol.get("frozen_dependencies")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise DivisionDecodeError("R4 protocol has no frozen dependencies")
    if "src/celltrack_alpha/action_compiler.py" not in dependencies:
        raise DivisionDecodeError("R4 protocol does not pin the N1 compiler")
    for relative, expected in sorted(dependencies.items()):
        _verify_file(
            _repo_path(repo_root, relative),
            str(expected),
            f"frozen dependency {relative}",
        )

    # Keep arguments live in the contract audit and reject symlink/path confusion by
    # requiring that the bytes verified above still have the caller-supplied hashes.
    _verify_file(protocol_path, protocol_sha256, "R4 protocol")
    _verify_file(configuration_path, configuration_sha256, "configuration manifest")


def _load_configuration(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any], name: str
) -> DivisionDecodeConfig:
    configurations = manifest.get("configurations")
    if (
        manifest.get("schema") != CONFIG_SCHEMA
        or manifest.get("status") != "LOCKED_BEFORE_OUTCOMES"
        or not isinstance(configurations, list)
        or len(configurations) != 18_145
        or manifest.get("configuration_count_including_null") != 18_145
        or manifest.get("active_configuration_count") != 18_144
    ):
        raise DivisionDecodeError("configuration manifest shape or status drifted")
    observed_config_hash = canonical_sha256(configurations)
    expected_config_hash = str(
        protocol["configuration_manifest"].get("configuration_sha256")
    )
    if (
        manifest.get("configuration_sha256") != observed_config_hash
        or observed_config_hash != expected_config_hash
    ):
        raise DivisionDecodeError("configuration identity drifted")
    matches = [row for row in configurations if isinstance(row, Mapping) and row.get("name") == name]
    if len(matches) != 1:
        raise DivisionDecodeError(f"configuration name is not uniquely sealed: {name}")
    row = matches[0]
    _strict_keys(
        row,
        {
            "name",
            "divisions",
            "fork_threshold",
            "max_primary_secondary_gap",
            "late_window",
            "daughter_persistence",
            "grandchild_persistence",
            "duplicate_temporal_window",
            "per_video_cap",
        },
        "configuration row",
    )
    config = DivisionDecodeConfig(
        name=str(row["name"]),
        divisions=bool(row["divisions"]),
        fork_threshold=(
            None if row["fork_threshold"] is None else float(row["fork_threshold"])
        ),
        max_primary_secondary_gap=(
            None
            if row["max_primary_secondary_gap"] is None
            else float(row["max_primary_secondary_gap"])
        ),
        late_window=int(row["late_window"]),
        daughter_persistence=int(row["daughter_persistence"]),
        grandchild_persistence=int(row["grandchild_persistence"]),
        duplicate_temporal_window=int(row["duplicate_temporal_window"]),
        per_video_cap=(
            None if row["per_video_cap"] is None else int(row["per_video_cap"])
        ),
    )
    if config.divisions:
        if (
            config.fork_threshold is None
            or config.max_primary_secondary_gap is None
            or not 0.0 <= config.fork_threshold <= 1.0
            or config.max_primary_secondary_gap < 0.0
            or min(
                config.late_window,
                config.daughter_persistence,
                config.grandchild_persistence,
                config.duplicate_temporal_window,
            )
            < 0
            or (config.per_video_cap is not None and config.per_video_cap <= 0)
        ):
            raise DivisionDecodeError("active configuration values are malformed")
    elif config.name != "DIVISIONS_OFF" or config.per_video_cap != 0:
        raise DivisionDecodeError("null configuration drifted")
    return config


def select_modal_outer_winner(winners: Sequence[str]) -> str:
    """Apply the amended deployable tie rule without metric re-pooling."""

    if len(winners) != 5 or any(not value for value in winners):
        raise DivisionDecodeError("deployable selection requires five outer winners")
    counts: dict[str, int] = defaultdict(int)
    for value in winners:
        counts[str(value)] += 1
    highest = max(counts.values())
    return min(name for name, count in counts.items() if count == highest)


def _verify_population_contract(
    *,
    population: Mapping[str, Any],
    population_path: Path,
    population_file_sha256: str,
    amendment: Mapping[str, Any],
    amendment_path: Path,
    amendment_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    repo_root: Path,
) -> Mapping[str, set[str]]:
    _strict_keys(
        population,
        {
            "schema",
            "status",
            "seed",
            "sources",
            "roles",
            "outer_folds",
            "constraints",
            "population_sha256",
        },
        "R4 population manifest",
    )
    if (
        population.get("schema") != POPULATION_SCHEMA
        or population.get("status")
        != "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME"
    ):
        raise DivisionDecodeError("R4 population manifest is not pre-outcome locked")
    population_identity = canonical_sha256(
        {
            "seed": population.get("seed"),
            "roles": population.get("roles"),
            "outer_folds": population.get("outer_folds"),
            "constraints": population.get("constraints"),
        }
    )
    if population.get("population_sha256") != population_identity:
        raise DivisionDecodeError("R4 population identity drifted")

    roles = population.get("roles")
    if not isinstance(roles, Mapping):
        raise DivisionDecodeError("R4 population roles are malformed")
    expected_roles = {
        "development_nested_oof": (69, "train:6bba_"),
        "held_fit_one_shot_confirmation": (36, "train:44b6_"),
        "sealed_hidden_test_probe": (4, "test:"),
        "excluded_contaminated_audit20": (20, "train:"),
    }
    identities_by_role: dict[str, set[str]] = {}
    for role, (count, prefix) in expected_roles.items():
        record = roles.get(role)
        if not isinstance(record, Mapping) or record.get("video_count") != count:
            raise DivisionDecodeError(f"R4 population role count drifted: {role}")
        videos = record.get("videos")
        if not isinstance(videos, list) or len(videos) != count:
            raise DivisionDecodeError(f"R4 population role inventory drifted: {role}")
        identities = {str(row.get("identity")) for row in videos if isinstance(row, Mapping)}
        if len(identities) != count or any(not value.startswith(prefix) for value in identities):
            raise DivisionDecodeError(f"R4 population role identities drifted: {role}")
        identities_by_role[role] = identities
    role_names = tuple(expected_roles)
    for left in range(len(role_names)):
        for right in range(left + 1, len(role_names)):
            if identities_by_role[role_names[left]] & identities_by_role[role_names[right]]:
                raise DivisionDecodeError("R4 population roles overlap")
    folds = population.get("outer_folds")
    if (
        not isinstance(folds, Mapping)
        or set(folds) != {"0", "1", "2", "3", "4"}
        or any(not isinstance(rows, list) for rows in folds.values())
    ):
        raise DivisionDecodeError("R4 population outer-fold inventory drifted")
    fold_rows = [row for rows in folds.values() for row in rows]
    fold_identities = [str(row.get("identity")) for row in fold_rows if isinstance(row, Mapping)]
    fold_groups = [str(row.get("group_id")) for row in fold_rows if isinstance(row, Mapping)]
    if (
        len(fold_rows) != 69
        or set(fold_identities) != identities_by_role["development_nested_oof"]
        or len(fold_identities) != len(set(fold_identities))
        or len(fold_groups) != len(set(fold_groups))
    ):
        raise DivisionDecodeError("R4 grouped outer folds leak or lack development coverage")
    constraints = population.get("constraints")
    if not isinstance(constraints, Mapping) or any(
        (
            constraints.get("division_labels_used_for_split") is not False,
            constraints.get("held_fit_can_select_configuration") is not False,
            constraints.get("contaminated_audit20_bank_count_required") != 0,
            constraints.get("development_bank_count_required") != 69,
            constraints.get("held_fit_bank_count_required") != 36,
            constraints.get("hidden_test_bank_count_required") != 4,
        )
    ):
        raise DivisionDecodeError("R4 population constraints drifted")

    if (
        amendment.get("schema") != POPULATION_AMENDMENT_SCHEMA
        or amendment.get("status")
        != "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME"
    ):
        raise DivisionDecodeError("R4 population amendment is not pre-outcome sealed")
    amended = amendment.get("amends")
    population_record = amendment.get("population_manifest")
    scientific = amendment.get("scientific_configuration")
    binding = amendment.get("binding_roles")
    bank_scope = amendment.get("association_bank_scope")
    if not all(
        isinstance(value, Mapping)
        for value in (amended, population_record, scientific, binding, bank_scope)
    ):
        raise DivisionDecodeError("R4 population amendment sections are malformed")
    if (
        amended.get("protocol_sha256") != protocol_sha256
        or amended.get("design_sha256") != protocol["protocol"].get("design_sha256")
        or population_record.get("file_sha256") != population_file_sha256
        or population_record.get("population_sha256") != population_identity
        or scientific.get("modal_outer_winner_rule")
        != "highest outer-winner count, then lexicographically smallest config name"
        or scientific.get("modal_tie_uses_metric_repooling") is not False
        or binding.get("development_nested_oof", {}).get("video_count") != 69
        or binding.get("held_fit_one_shot_confirmation", {}).get("video_count") != 36
        or binding.get("sealed_hidden_test_probe", {}).get("video_count") != 4
        or binding.get("excluded_contaminated_audit20", {}).get(
            "target_access_for_R4_allowed"
        )
        is not False
        or bank_scope.get("development_bank_count") != 69
        or bank_scope.get("held_fit_bank_count") != 36
        or bank_scope.get("hidden_test_bank_count") != 4
        or bank_scope.get("contaminated_audit20_bank_count") != 0
        or bank_scope.get("development_and_held_fit_bank_hashes_sealed_before_development_scoring")
        is not True
        or bank_scope.get("hidden_test_inference_only_after_local_confirmation_gate")
        is not True
    ):
        raise DivisionDecodeError("R4 population amendment binding drifted")
    generator_path = _repo_path(repo_root, population_record.get("generator_path"))
    _verify_file(
        generator_path,
        str(population_record.get("generator_sha256")),
        "sealed population generator",
    )
    _verify_file(population_path, population_file_sha256, "R4 population manifest")
    _verify_file(amendment_path, amendment_sha256, "R4 population amendment")
    return identities_by_role


def _verify_stage_gate(
    *,
    stage: str,
    raw_gate: Any,
    manifest_base: Path,
    config: DivisionDecodeConfig,
    protocol_sha256: str,
    population_sha256: str,
    configuration_file_sha256: str,
    configuration_sha256: str,
) -> str | None:
    if stage == "development_staging":
        if raw_gate is not None:
            raise DivisionDecodeError("development stage must not carry a later-stage gate")
        return None
    gate_path, gate_hash = _reference(raw_gate, manifest_base, f"{stage} gate")
    gate = _load_verified_json(gate_path, gate_hash, f"{stage} gate")
    if stage == "development_nested_oof":
        if (
            gate.get("schema") != "celltrack-alpha.g03-r4-bank-coverage-seal.v1"
            or gate.get("status")
            != "SEALED_105_DEVELOPMENT_AND_HELD_FIT_BANKS_BEFORE_OOF"
            or gate.get("development_bank_count") != 69
            or gate.get("held_fit_bank_count") != 36
            or gate.get("bank_count") != 105
            or gate.get("all_bank_hashes_verified") is not True
            or gate.get("r1_scientific_configuration_count") != 1
            or not SHA256_RE.fullmatch(str(gate.get("bank_inventory_sha256", "")))
            or not SHA256_RE.fullmatch(
                str(gate.get("r1_inference_seal_sha256", ""))
            )
        ):
            raise DivisionDecodeError("development 105-bank coverage gate is closed")
    elif stage == "held_fit_one_shot_confirmation":
        winners = gate.get("outer_winners")
        if (
            gate.get("schema")
            != "celltrack-alpha.g03-r4-deployable-config-seal.v1"
            or gate.get("status") != "SEALED_DEPLOYABLE_CONFIG_BEFORE_HELD_FIT"
            or not isinstance(winners, list)
            or gate.get("selected_config") != select_modal_outer_winner(winners)
            or gate.get("selected_config") != config.name
        ):
            raise DivisionDecodeError("held-fit deployable configuration gate drifted")
    elif stage == "sealed_hidden_test_probe":
        if (
            gate.get("schema") != "celltrack-alpha.g03-r4-local-confirmation-gate.v1"
            or gate.get("status") != "PASS_R4_LOCAL_CONFIRMATION_GATE"
            or gate.get("selected_config") != config.name
        ):
            raise DivisionDecodeError("hidden-test local confirmation gate is closed")
    else:
        raise DivisionDecodeError(f"unsupported R4 execution stage: {stage}")
    if (
        gate.get("protocol_contract_sha256") != protocol_sha256
        or gate.get("population_sha256") != population_sha256
        or gate.get("configuration_manifest_file_sha256")
        != configuration_file_sha256
        or gate.get("configuration_sha256") != configuration_sha256
    ):
        raise DivisionDecodeError(f"{stage} gate references drifted")
    return gate_hash


def load_association_bank(
    path: Path, *, expected_sha256: str, expected_identity: str
) -> AssociationBank:
    """Load one R1 bank only after exact byte verification."""

    _verify_file(path, expected_sha256, f"{expected_identity} association bank")
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_members = {
                "metadata",
                "node_id",
                "frame",
                "coords_zyx",
                "edge_source",
                "edge_target",
                "probability",
                "distance_um",
            }
            if set(archive.files) != expected_members:
                raise DivisionDecodeError("association-bank member set drifted")
            metadata_raw = np.asarray(archive["metadata"])
            if metadata_raw.dtype != np.uint8 or metadata_raw.ndim != 1:
                raise DivisionDecodeError("association-bank metadata encoding drifted")
            metadata = json.loads(metadata_raw.tobytes().decode("ascii"))
            if not isinstance(metadata, dict):
                raise DivisionDecodeError("association-bank metadata must be an object")
            _strict_keys(
                metadata,
                {
                    "schema",
                    "identity",
                    "probability_dtype",
                    "distance_unit",
                    "complete_candidate_bank",
                },
                "association-bank metadata",
            )
            if (
                metadata.get("schema") != BANK_SCHEMA
                or metadata.get("identity") != expected_identity
                or metadata.get("probability_dtype") != "float32"
                or metadata.get("distance_unit") != "micrometre"
                or metadata.get("complete_candidate_bank") is not True
            ):
                raise DivisionDecodeError("association-bank metadata contract drifted")
            node_id = np.asarray(archive["node_id"])
            frame = np.asarray(archive["frame"])
            coords = np.asarray(archive["coords_zyx"])
            edge_source = np.asarray(archive["edge_source"])
            edge_target = np.asarray(archive["edge_target"])
            probability = np.asarray(archive["probability"])
            distance_um = np.asarray(archive["distance_um"])
    except DivisionDecodeError:
        raise
    except (OSError, ValueError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise DivisionDecodeError(f"cannot parse association bank: {path}") from exc

    if (
        node_id.dtype != np.int64
        or frame.dtype != np.int32
        or coords.dtype != np.float32
        or edge_source.dtype != np.int64
        or edge_target.dtype != np.int64
        or probability.dtype != np.float32
        or distance_um.dtype != np.float32
    ):
        raise DivisionDecodeError("association-bank array dtypes drifted")
    if (
        node_id.ndim != 1
        or frame.shape != node_id.shape
        or coords.shape != (len(node_id), 3)
        or edge_source.ndim != 1
        or edge_target.shape != edge_source.shape
        or probability.shape != edge_source.shape
        or distance_um.shape != edge_source.shape
        or len(node_id) == 0
    ):
        raise DivisionDecodeError("association-bank array shapes drifted")
    if (
        not np.isfinite(coords).all()
        or not np.isfinite(probability).all()
        or not np.isfinite(distance_um).all()
        or bool(np.any(probability < 0.0))
        or bool(np.any(probability > 1.0))
        or bool(np.any(distance_um < 0.0))
    ):
        raise DivisionDecodeError("association-bank numeric domain is invalid")
    ids = [int(value) for value in node_id]
    if len(ids) != len(set(ids)):
        raise DivisionDecodeError("association-bank node IDs repeat")
    nodes = {
        node: BankNode(
            node_id=node,
            frame=int(frame[index]),
            coords_zyx=tuple(float(value) for value in coords[index]),
        )
        for index, node in enumerate(ids)
    }
    edges: list[CandidateEdge] = []
    pairs: set[tuple[int, int]] = set()
    for source_raw, target_raw, score_raw, distance_raw in zip(
        edge_source,
        edge_target,
        probability,
        distance_um,
        strict=True,
    ):
        source, target = int(source_raw), int(target_raw)
        pair = (source, target)
        if pair in pairs:
            raise DivisionDecodeError("association-bank candidate edge repeats")
        pairs.add(pair)
        if source not in nodes or target not in nodes:
            raise DivisionDecodeError("association-bank candidate endpoint is missing")
        if nodes[target].frame != nodes[source].frame + 1:
            raise DivisionDecodeError("association-bank edge is not exactly one frame forward")
        edges.append(
            CandidateEdge(
                source=source,
                target=target,
                probability=float(score_raw),
                distance_um=float(distance_raw),
            )
        )
    edges.sort(key=lambda row: (row.source, row.target))
    return AssociationBank(expected_identity, nodes, tuple(edges))


def _compiler_api(repo_root: Path) -> Mapping[str, Any]:
    source = str((repo_root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from celltrack_alpha.action_compiler import (
            ActionKind,
            GraphAction,
            compile_action,
            graph_sha256,
        )
        from celltrack_alpha.kaggle_calibration import (
            read_submission,
            validate_graphs,
            write_submission,
        )
    except ImportError as exc:
        raise DivisionDecodeError("cannot import the frozen N1 graph interface") from exc
    return {
        "ActionKind": ActionKind,
        "GraphAction": GraphAction,
        "compile_action": compile_action,
        "graph_sha256": graph_sha256,
        "read_submission": read_submission,
        "validate_graphs": validate_graphs,
        "write_submission": write_submission,
    }


def _load_base_graph(
    path: Path,
    *,
    expected_sha256: str,
    identity: str,
    bank: AssociationBank,
    api: Mapping[str, Any],
) -> Any:
    _verify_file(path, expected_sha256, f"{identity} base graph")
    try:
        graphs = api["read_submission"](path)
    except (OSError, ValueError) as exc:
        raise DivisionDecodeError(f"cannot parse {identity} base graph") from exc
    if len(graphs) != 1:
        raise DivisionDecodeError("base fragment must contain exactly one graph")
    graph = next(iter(graphs.values()))
    split, separator, dataset = identity.partition(":")
    if separator != ":" or split not in {"train", "test"} or graph.dataset != dataset:
        raise DivisionDecodeError("base graph identity disagrees with the input manifest")
    for node_id, node in graph.nodes.items():
        bank_node = bank.nodes.get(int(node_id))
        if bank_node is None:
            raise DivisionDecodeError("base graph contains a node absent from its bank")
        observed = (
            int(node.t),
            int(node.z),
            int(node.y),
            int(node.x),
        )
        expected = (bank_node.frame,) + tuple(
            int(round(value)) for value in bank_node.coords_zyx
        )
        if observed != expected:
            raise DivisionDecodeError("base graph node geometry disagrees with its bank")
    return graph


def _adjacency(graph: Any) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]], set[tuple[int, int]]]:
    outgoing_raw: dict[int, list[int]] = defaultdict(list)
    incoming_raw: dict[int, list[int]] = defaultdict(list)
    edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
        pair = (int(edge.source_id), int(edge.target_id))
        edges.add(pair)
        outgoing_raw[pair[0]].append(pair[1])
        incoming_raw[pair[1]].append(pair[0])
    outgoing = {key: tuple(sorted(value)) for key, value in outgoing_raw.items()}
    incoming = {key: tuple(sorted(value)) for key, value in incoming_raw.items()}
    return outgoing, incoming, edges


def _lineage_root(node: int, incoming: Mapping[int, tuple[int, ...]]) -> int:
    seen = {node}
    current = node
    while len(incoming.get(current, ())) == 1:
        current = incoming[current][0]
        if current in seen:
            raise DivisionDecodeError("base graph contains a lineage cycle")
        seen.add(current)
    return current


def _selected_path(
    start: int,
    depth: int,
    *,
    outgoing: Mapping[int, tuple[int, ...]],
    incoming: Mapping[int, tuple[int, ...]],
    nodes: Mapping[int, Any],
) -> tuple[int, ...] | None:
    current = start
    result: list[int] = []
    for _ in range(depth):
        successors = outgoing.get(current, ())
        if len(successors) != 1:
            return None
        target = successors[0]
        if len(incoming.get(target, ())) != 1:
            return None
        if int(nodes[target].t) != int(nodes[current].t) + 1:
            return None
        result.append(target)
        current = target
    return tuple(result)


def _bank_path(
    start: int,
    depth: int,
    *,
    outgoing: Mapping[int, tuple[CandidateEdge, ...]],
) -> tuple[int, ...] | None:
    current = start
    result: list[int] = []
    for _ in range(depth):
        candidates = outgoing.get(current, ())
        if not candidates:
            return None
        winner = min(candidates, key=lambda row: (-row.probability, row.target))
        result.append(winner.target)
        current = winner.target
    return tuple(result)


def _late_anchors(
    source: int,
    maximum: int,
    *,
    outgoing: Mapping[int, tuple[int, ...]],
    incoming: Mapping[int, tuple[int, ...]],
    nodes: Mapping[int, Any],
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    current = source
    reverse_path = [source]
    result: list[tuple[int, int, tuple[int, ...]]] = []
    for depth in range(1, maximum + 1):
        predecessors = incoming.get(current, ())
        if len(predecessors) != 1:
            break
        predecessor = predecessors[0]
        if outgoing.get(predecessor, ()) != (current,):
            break
        if int(nodes[current].t) != int(nodes[predecessor].t) + 1:
            break
        reverse_path.append(predecessor)
        continuation = tuple(reversed(reverse_path[:-1]))
        result.append((depth, predecessor, continuation))
        current = predecessor
    return tuple(result)


def _event_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(value)


def _candidate_rank(event: CandidateEvent, config_name: str) -> tuple[Any, ...]:
    mode_rank = 0 if event.mode == "DIRECT" else 1
    evidence = tuple(
        sorted(
            set(
                event.continuation_nodes
                + event.daughter_persistence_nodes
                + event.grandchild_persistence_nodes
            )
        )
    )
    return (
        -event.secondary_probability,
        event.primary_secondary_gap,
        mode_rank,
        event.anchor_parent,
        event.children,
        evidence,
        config_name,
        event.late_depth,
        event.emitted_parent,
        event.event_id,
    )


def generate_candidates(
    bank: AssociationBank, graph: Any, config: DivisionDecodeConfig
) -> tuple[tuple[CandidateEvent, ...], Mapping[str, int]]:
    """Generate the complete config-filtered, observation-only event population."""

    if not config.divisions:
        return (), {
            "bank_sources_with_two_forward_candidates": 0,
            "candidate_pairs_considered": 0,
            "candidate_pairs_threshold_pass": 0,
            "persistence_pass": 0,
            "raw_event_count": 0,
        }
    outgoing, incoming, base_edges = _adjacency(graph)
    del base_edges
    bank_outgoing_raw: dict[int, list[CandidateEdge]] = defaultdict(list)
    for edge in bank.edges:
        bank_outgoing_raw[edge.source].append(edge)
    bank_outgoing = {
        source: tuple(sorted(rows, key=lambda row: (-row.probability, row.target)))
        for source, rows in bank_outgoing_raw.items()
    }
    base_nodes = set(int(value) for value in graph.nodes)
    events: list[CandidateEvent] = []
    source_count = pair_count = threshold_count = persistence_count = 0
    for source in sorted(bank_outgoing):
        if source not in base_nodes:
            continue
        eligible = tuple(
            row
            for row in bank_outgoing[source]
            if row.target in base_nodes
            and int(graph.nodes[row.target].t) == int(graph.nodes[source].t) + 1
        )
        if len(eligible) < 2:
            continue
        source_count += 1
        for left, right in itertools.combinations(eligible, 2):
            pair_count += 1
            primary, secondary = sorted(
                (left, right), key=lambda row: (-row.probability, row.target)
            )
            gap = primary.probability - secondary.probability
            if (
                secondary.probability + 1.0e-12 < float(config.fork_threshold)
                or gap > float(config.max_primary_secondary_gap) + 1.0e-12
            ):
                continue
            threshold_count += 1
            children = tuple(sorted((primary.target, secondary.target)))
            daughter_paths = tuple(
                _selected_path(
                    child,
                    config.daughter_persistence,
                    outgoing=outgoing,
                    incoming=incoming,
                    nodes=graph.nodes,
                )
                for child in children
            )
            grandchild_paths = tuple(
                _bank_path(
                    child,
                    config.grandchild_persistence,
                    outgoing=bank_outgoing,
                )
                for child in children
            )
            if any(path is None for path in daughter_paths + grandchild_paths):
                continue
            daughter_nodes = tuple(
                node for path in daughter_paths for node in (path or ())
            )
            grandchild_nodes = tuple(
                node for path in grandchild_paths for node in (path or ())
            )
            left_branch = set(children[:1]) | set(daughter_paths[0] or ()) | set(
                grandchild_paths[0] or ()
            )
            right_branch = set(children[1:]) | set(daughter_paths[1] or ()) | set(
                grandchild_paths[1] or ()
            )
            if left_branch & right_branch:
                continue
            daughter_lineages = tuple(
                sorted(_lineage_root(child, incoming) for child in children)
            )
            if daughter_lineages[0] == daughter_lineages[1]:
                continue
            persistence_count += 1
            parent_lineage = _lineage_root(source, incoming)
            equivalence_payload = {
                "identity": bank.identity,
                "parent_lineage": parent_lineage,
                "daughter_lineages": list(daughter_lineages),
            }
            equivalence_key = _event_hash(equivalence_payload)

            representations = [("DIRECT", 0, source, ())]
            representations.extend(
                (
                    f"LATE({late_depth})",
                    late_depth,
                    anchor,
                    continuation,
                )
                for late_depth, anchor, continuation in _late_anchors(
                    source,
                    config.late_window,
                    outgoing=outgoing,
                    incoming=incoming,
                    nodes=graph.nodes,
                )
            )
            for mode, late_depth, anchor, continuation in representations:
                occupied = tuple(
                    sorted(
                        {
                            source,
                            anchor,
                            *children,
                            *continuation,
                            *daughter_nodes,
                            *grandchild_nodes,
                        }
                    )
                )
                event_payload = {
                    **equivalence_payload,
                    "mode": mode,
                    "late_depth": late_depth,
                    "anchor_parent": anchor,
                    "emitted_parent": source,
                    "primary_child": primary.target,
                    "secondary_child": secondary.target,
                    "children": list(children),
                    "daughter_persistence_nodes": list(daughter_nodes),
                    "grandchild_persistence_nodes": list(grandchild_nodes),
                }
                events.append(
                    CandidateEvent(
                        event_id=_event_hash(event_payload),
                        equivalence_key=equivalence_key,
                        identity=bank.identity,
                        mode=mode,
                        late_depth=late_depth,
                        anchor_parent=anchor,
                        emitted_parent=source,
                        primary_child=primary.target,
                        secondary_child=secondary.target,
                        children=children,
                        parent_lineage=parent_lineage,
                        daughter_lineages=daughter_lineages,
                        primary_probability=primary.probability,
                        secondary_probability=secondary.probability,
                        primary_secondary_gap=gap,
                        continuation_nodes=tuple(continuation),
                        daughter_persistence_nodes=daughter_nodes,
                        grandchild_persistence_nodes=grandchild_nodes,
                        occupied_nodes=occupied,
                    )
                )
    ordered = tuple(sorted(events, key=lambda row: _candidate_rank(row, config.name)))
    if len({row.event_id for row in ordered}) != len(ordered):
        raise DivisionDecodeError("candidate event identity collision")
    return ordered, {
        "bank_sources_with_two_forward_candidates": source_count,
        "candidate_pairs_considered": pair_count,
        "candidate_pairs_threshold_pass": threshold_count,
        "persistence_pass": persistence_count,
        "raw_event_count": len(ordered),
    }


def suppress_candidates(
    events: Sequence[CandidateEvent],
    config: DivisionDecodeConfig,
    *,
    incident_index: FrozenIncidentIndex,
) -> tuple[tuple[CandidateEvent, ...], tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    """Apply exact, temporal, node, frozen-edge ownership, then cap."""

    rejections: list[Mapping[str, Any]] = []
    exact_winner: dict[str, CandidateEvent] = {}
    exact_survivors: list[CandidateEvent] = []
    for event in events:
        winner = exact_winner.get(event.equivalence_key)
        if winner is not None:
            rejections.append(
                {
                    "event_id": event.event_id,
                    "reason": "exact_equivalence",
                    "winner_event_id": winner.event_id,
                }
            )
            continue
        exact_winner[event.equivalence_key] = event
        exact_survivors.append(event)

    temporal_survivors: list[CandidateEvent] = []
    temporal_index: dict[
        tuple[int, tuple[int, int], int], tuple[int, CandidateEvent]
    ] = {}
    for event in exact_survivors:
        possible = [
            temporal_index[
                (event.parent_lineage, event.daughter_lineages, late_depth)
            ]
            for late_depth in range(
                event.late_depth - config.duplicate_temporal_window,
                event.late_depth + config.duplicate_temporal_window + 1,
            )
            if (
                event.parent_lineage,
                event.daughter_lineages,
                late_depth,
            )
            in temporal_index
        ]
        winner = (
            min(possible, key=lambda row: row[0])[1]
            if possible
            else None
        )
        if winner is not None:
            rejections.append(
                {
                    "event_id": event.event_id,
                    "reason": "duplicate_temporal_window",
                    "winner_event_id": winner.event_id,
                }
            )
            continue
        position = len(temporal_survivors)
        temporal_survivors.append(event)
        temporal_index.setdefault(
            (event.parent_lineage, event.daughter_lineages, event.late_depth),
            (position, event),
        )

    node_owners: dict[int, CandidateEvent] = {}
    conflict_edge_owners: dict[tuple[int, int], CandidateEvent] = {}
    accepted: list[CandidateEvent] = []
    cap = config.per_video_cap
    for event in temporal_survivors:
        collisions = sorted(
            {
                node_owners[node].event_id
                for node in event.occupied_nodes
                if node in node_owners
            }
        )
        if collisions:
            rejections.append(
                {
                    "event_id": event.event_id,
                    "reason": "node_ownership",
                    "winner_event_ids": collisions,
                }
            )
            continue
        conflicts = _incident_conflicts(
            incident_index, event.emitted_parent, event.children
        )
        conflict_collisions = sorted(
            {
                conflict_edge_owners[pair].event_id
                for pair in conflicts
                if pair in conflict_edge_owners
            }
        )
        if conflict_collisions:
            rejections.append(
                {
                    "event_id": event.event_id,
                    "reason": "conflict_edge_ownership",
                    "winner_event_ids": conflict_collisions,
                    "owned_conflict_edges": [
                        list(pair)
                        for pair in conflicts
                        if pair in conflict_edge_owners
                    ],
                }
            )
            continue
        if cap is not None and len(accepted) >= cap:
            rejections.append(
                {
                    "event_id": event.event_id,
                    "reason": "per_video_cap",
                    "cap": cap,
                }
            )
            continue
        accepted.append(event)
        for node in event.occupied_nodes:
            node_owners[node] = event
        for pair in conflicts:
            conflict_edge_owners[pair] = event

    equivalence_keys = [event.equivalence_key for event in accepted]
    duplicate_survivors = len(equivalence_keys) - len(set(equivalence_keys))
    if duplicate_survivors:
        raise DivisionDecodeError("FAIL_CLOSED_DUPLICATE_FORK")
    if cap is not None and len(accepted) > cap:
        raise DivisionDecodeError("per-video cap was not enforced")
    if len(events) != len(accepted) + len(rejections):
        raise DivisionDecodeError("suppression accounting lost a candidate event")
    counts: dict[str, int] = defaultdict(int)
    for row in rejections:
        counts[str(row["reason"])] += 1
    return tuple(accepted), tuple(rejections), {
        "exact_equivalence_survivors": len(exact_survivors),
        "temporal_survivors": len(temporal_survivors),
        "accepted_events": len(accepted),
        "duplicate_survivors": duplicate_survivors,
        "rejected_exact_equivalence": counts["exact_equivalence"],
        "rejected_duplicate_temporal_window": counts[
            "duplicate_temporal_window"
        ],
        "rejected_node_ownership": counts["node_ownership"],
        "rejected_conflict_edge_ownership": counts["conflict_edge_ownership"],
        "rejected_per_video_cap": counts["per_video_cap"],
    }


def _build_incident_index(
    edges: Iterable[tuple[int, int]],
) -> FrozenIncidentIndex:
    normalized = tuple((int(pair[0]), int(pair[1])) for pair in edges)
    if len(normalized) != len(set(normalized)):
        raise DivisionDecodeError("baseline incident index contains duplicate edges")
    by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in normalized:
        by_source[pair[0]].append(pair)
        by_target[pair[1]].append(pair)
    return FrozenIncidentIndex(
        edges=frozenset(normalized),
        by_source={key: tuple(sorted(value)) for key, value in by_source.items()},
        by_target={key: tuple(sorted(value)) for key, value in by_target.items()},
    )


def _incident_conflicts(
    index: FrozenIncidentIndex, parent: int, children: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    conflicts = set(index.by_source.get(int(parent), ()))
    for child in children:
        conflicts.update(index.by_target.get(int(child), ()))
    return tuple(sorted(conflicts))


def _event_record(event: CandidateEvent) -> dict[str, Any]:
    return asdict(event)


def _materialize_division_actions_one_pass(
    *,
    baseline: Any,
    actions: Sequence[Any],
    api: Mapping[str, Any],
    baseline_graph_sha256: str | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Materialize validated division primitives with one full graph construction."""

    validate_graphs = api.get("validate_graphs")
    graph_sha256 = api.get("graph_sha256")
    compile_action = api.get("compile_action")
    graph_action = api.get("GraphAction")
    action_kind = api.get("ActionKind")
    if not (
        callable(validate_graphs)
        and callable(graph_sha256)
        and callable(compile_action)
        and graph_action is not None
        and action_kind is not None
    ):
        raise DivisionDecodeError("one-pass materializer lacks the frozen compiler API")
    compiler_globals = getattr(compile_action, "__globals__", {})
    prepare_action = compiler_globals.get("_prepare_action")
    action_sha256 = compiler_globals.get("action_sha256")
    frozen_graph_sha256 = compiler_globals.get("graph_sha256")
    frozen_validate_graphs = compiler_globals.get("validate_graphs")
    frozen_graph_action = compiler_globals.get("GraphAction")
    frozen_action_kind = compiler_globals.get("ActionKind")
    submission_edge = compiler_globals.get("SubmissionEdge")
    if not (
        callable(prepare_action)
        and callable(action_sha256)
        and callable(submission_edge)
    ):
        raise DivisionDecodeError(
            "one-pass materializer cannot reach the frozen preparation oracle"
        )
    if not (
        graph_sha256 is frozen_graph_sha256
        and validate_graphs is frozen_validate_graphs
        and graph_action is frozen_graph_action
        and action_kind is frozen_action_kind
    ):
        raise DivisionDecodeError("one-pass compiler API identity coherence drifted")
    compiler_module = str(getattr(compile_action, "__module__", ""))
    if not (
        compiler_module
        and getattr(prepare_action, "__module__", None) == compiler_module
        and getattr(action_sha256, "__module__", None) == compiler_module
        and getattr(prepare_action, "__globals__", None) is compiler_globals
        and getattr(action_sha256, "__globals__", None) is compiler_globals
    ):
        raise DivisionDecodeError("one-pass frozen compiler private API provenance drifted")
    compiler_source = Path(str(compile_action.__code__.co_filename)).resolve()
    prepare_source = Path(str(prepare_action.__code__.co_filename)).resolve()
    action_sha_source = Path(str(action_sha256.__code__.co_filename)).resolve()
    if not (
        compiler_source.is_file()
        and not compiler_source.is_symlink()
        and prepare_source == compiler_source
        and action_sha_source == compiler_source
    ):
        raise DivisionDecodeError("one-pass frozen compiler source provenance drifted")
    graph_model_module = str(getattr(submission_edge, "__module__", ""))
    graph_model = sys.modules.get(graph_model_module)
    graph_model_file = getattr(graph_model, "__file__", None)
    if not graph_model_module or graph_model_file is None:
        raise DivisionDecodeError("one-pass graph-model source provenance is unavailable")
    graph_model_source = Path(str(graph_model_file)).resolve()
    expected_compiler_source = (
        Path(__file__).resolve().parents[1]
        / "src/celltrack_alpha/action_compiler.py"
    ).resolve()
    expected_graph_model_source = (
        Path(__file__).resolve().parents[1]
        / "src/celltrack_alpha/kaggle_calibration.py"
    ).resolve()
    if not (
        compiler_source == expected_compiler_source
        and graph_model_source == expected_graph_model_source
        and graph_model_source.is_file()
        and not graph_model_source.is_symlink()
    ):
        raise DivisionDecodeError("one-pass frozen source paths drifted")
    compiler_source_sha256 = sha256_file(compiler_source)
    graph_model_source_sha256 = sha256_file(graph_model_source)
    if not (
        compiler_source_sha256 == FROZEN_ACTION_COMPILER_SHA256
        and graph_model_source_sha256 == FROZEN_GRAPH_MODEL_SHA256
    ):
        raise DivisionDecodeError("one-pass frozen source hashes drifted")
    if not (
        getattr(graph_model, "SubmissionEdge", None) is submission_edge
        and getattr(graph_model, "validate_graphs", None) is validate_graphs
    ):
        raise DivisionDecodeError("one-pass graph-model API identity coherence drifted")

    validate_graphs({baseline.dataset: baseline})
    baseline_nodes = dict(baseline.nodes)
    baseline_node_ids = set(baseline_nodes)
    baseline_edges = {
        (int(edge.source_id), int(edge.target_id)) for edge in baseline.edges
    }
    incident_index = _build_incident_index(baseline_edges)

    action_ids: set[str] = set()
    added_edge_owner: dict[tuple[int, int], str] = {}
    removed_edge_owner: dict[tuple[int, int], str] = {}
    source_owner: dict[int, str] = {}
    target_owner: dict[int, str] = {}
    boundary_owner: dict[int, str] = {}
    all_add_edges: set[tuple[int, int]] = set()
    all_remove_edges: set[tuple[int, int]] = set()
    action_hashes: list[str] = []

    for action in sorted(actions, key=lambda value: str(value.action_id)):
        action_id = str(action.action_id)
        if action_id in action_ids:
            raise DivisionDecodeError("one-pass action IDs repeat")
        action_ids.add(action_id)
        if action.kind is not action_kind.DIVISION_FORK:
            raise DivisionDecodeError("one-pass materializer accepts division_fork only")
        action_hashes.append(_require_sha256(action_sha256(action), "one-pass action"))
        replace_sources = {int(value) for value in action.replace_outgoing_from}
        replace_targets = {int(value) for value in action.replace_incoming_to}
        expected_conflicts = {
            (int(pair[0]), int(pair[1])) for pair in action.expected_conflict_edges
        }
        actual_conflicts: set[tuple[int, int]] = set()
        for source in replace_sources:
            actual_conflicts.update(incident_index.by_source.get(source, ()))
        for target in replace_targets:
            actual_conflicts.update(incident_index.by_target.get(target, ()))
        if expected_conflicts != actual_conflicts:
            raise DivisionDecodeError(
                "one-pass incident-conflict proof differs from the frozen baseline"
            )
        referenced_nodes = {
            int(value) for pair in action.add_edges for value in pair
        } | replace_sources | replace_targets | {
            int(value) for pair in expected_conflicts for value in pair
        }
        if not referenced_nodes <= baseline_node_ids:
            raise DivisionDecodeError("one-pass action references a missing baseline node")
        sparse = type(baseline)(
            dataset=baseline.dataset,
            nodes={node_id: baseline_nodes[node_id] for node_id in referenced_nodes},
            edges=[submission_edge(*pair) for pair in sorted(expected_conflicts)],
        )
        added_ids, add_edges, delete_edges, prepared_conflicts = prepare_action(
            sparse, action
        )
        if added_ids or delete_edges:
            raise DivisionDecodeError("one-pass division action primitive drifted")
        removed = delete_edges | prepared_conflicts
        for pair in add_edges:
            if pair in added_edge_owner:
                raise DivisionDecodeError("one-pass added edge ownership overlaps")
            if pair in removed_edge_owner and removed_edge_owner[pair] != action_id:
                raise DivisionDecodeError(
                    "one-pass edge is added and removed by different actions"
                )
            added_edge_owner[pair] = action_id
        for pair in removed:
            if pair in removed_edge_owner:
                raise DivisionDecodeError("one-pass removed edge ownership overlaps")
            if pair in added_edge_owner and added_edge_owner[pair] != action_id:
                raise DivisionDecodeError(
                    "one-pass edge is removed and added by different actions"
                )
            removed_edge_owner[pair] = action_id
        for source in replace_sources:
            if source in source_owner:
                raise DivisionDecodeError("one-pass outgoing ownership overlaps")
            source_owner[source] = action_id
        for target in replace_targets:
            if target in target_owner:
                raise DivisionDecodeError("one-pass incoming ownership overlaps")
            target_owner[target] = action_id
        for boundary in replace_sources | replace_targets:
            if boundary in boundary_owner:
                raise DivisionDecodeError(
                    "one-pass cross-category boundary ownership overlaps"
                )
            boundary_owner[boundary] = action_id
        all_add_edges.update(add_edges)
        all_remove_edges.update(removed)

    final_edges = (baseline_edges - all_remove_edges) | all_add_edges
    graph = type(baseline)(
        dataset=baseline.dataset,
        nodes=baseline_nodes,
        edges=[submission_edge(*pair) for pair in sorted(final_edges)],
    )
    validate_graphs({graph.dataset: graph})
    graph_sha = _require_sha256(graph_sha256(graph), "one-pass semantic graph")
    before_graph_sha256 = (
        _require_sha256(baseline_graph_sha256, "one-pass baseline graph")
        if baseline_graph_sha256 is not None
        else _require_sha256(graph_sha256(baseline), "one-pass baseline graph")
    )
    return graph, {
        "schema": "celltrack-alpha.g03-r4-n1-one-pass-install.v1",
        "compiler_path": "frozen_prepare_action_one_pass",
        "action_count": len(actions),
        "before_graph_sha256": before_graph_sha256,
        "after_graph_sha256": graph_sha,
        "baseline_edge_count": len(baseline_edges),
        "added_edge_count": len(all_add_edges),
        "removed_edge_count": len(all_remove_edges),
        "final_edge_count": len(final_edges),
        "removed_edges": [list(pair) for pair in sorted(all_remove_edges)],
        "action_set_sha256": canonical_sha256(action_hashes),
        "frozen_action_sha256_validation": True,
        "compiler_source_sha256": compiler_source_sha256,
        "graph_model_source_sha256": graph_model_source_sha256,
        "incident_index_built_once": True,
        "order_independence_verified_by_disjoint_primitive_ownership": True,
        "full_graph_constructions": 1,
        "full_graph_hashes": 1,
    }


def decode_video(
    bank: AssociationBank,
    graph: Any,
    config: DivisionDecodeConfig,
    api: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    base_hash = api["graph_sha256"](graph)
    if not config.divisions:
        return graph.copy(), {
            "identity": bank.identity,
            "base_graph_sha256": base_hash,
            "decoded_graph_sha256": base_hash,
            "generation": {
                "bank_sources_with_two_forward_candidates": 0,
                "candidate_pairs_considered": 0,
                "candidate_pairs_threshold_pass": 0,
                "persistence_pass": 0,
                "raw_event_count": 0,
            },
            "suppression": {
                "exact_equivalence_survivors": 0,
                "temporal_survivors": 0,
                "accepted_events": 0,
                "duplicate_survivors": 0,
                "rejected_exact_equivalence": 0,
                "rejected_duplicate_temporal_window": 0,
                "rejected_node_ownership": 0,
                "rejected_conflict_edge_ownership": 0,
                "rejected_per_video_cap": 0,
            },
            "accepted": [],
            "rejected": [],
            "n1_batch": None,
        }

    _outgoing, _incoming, edges = _adjacency(graph)
    incident_index = _build_incident_index(edges)
    events, generation = generate_candidates(bank, graph, config)
    accepted, rejections, suppression = suppress_candidates(
        events, config, incident_index=incident_index
    )
    actions = []
    accepted_records: list[dict[str, Any]] = []
    for event in accepted:
        conflicts = _incident_conflicts(
            incident_index, event.emitted_parent, event.children
        )
        action = api["GraphAction"](
            action_id=f"r4.{event.event_id[:40]}",
            kind=api["ActionKind"].DIVISION_FORK,
            add_edges=tuple((event.emitted_parent, child) for child in event.children),
            replace_outgoing_from=(event.emitted_parent,),
            replace_incoming_to=event.children,
            expected_conflict_edges=conflicts,
        )
        record = _event_record(event)
        record["action"] = {
            "action_id": action.action_id,
            "kind": action.kind.value,
            "add_edges": [list(pair) for pair in action.add_edges],
            "replace_outgoing_from": list(action.replace_outgoing_from),
            "replace_incoming_to": list(action.replace_incoming_to),
            "expected_conflict_edges": [list(pair) for pair in conflicts],
        }
        actions.append(action)
        accepted_records.append(record)

    if actions:
        decoded, one_pass = _materialize_division_actions_one_pass(
            baseline=graph,
            actions=tuple(actions),
            api=api,
            baseline_graph_sha256=base_hash,
        )
        decoded_hash = str(one_pass["after_graph_sha256"])
        batch_identity = {
            **one_pass,
            "before_graph_sha256": base_hash,
        }
        batch_record: Mapping[str, Any] | None = {
            "install_sha256": canonical_sha256(batch_identity),
            **batch_identity,
        }
    else:
        decoded = graph.copy()
        decoded_hash = base_hash
        batch_record = None
    if batch_record is not None and batch_record["after_graph_sha256"] != decoded_hash:
        raise DivisionDecodeError("N1 one-pass graph identity drifted")
    return decoded, {
        "identity": bank.identity,
        "base_graph_sha256": base_hash,
        "decoded_graph_sha256": decoded_hash,
        "generation": generation,
        "suppression": suppression,
        "accepted": accepted_records,
        "rejected": list(rejections),
        "n1_batch": batch_record,
    }


def _load_input_manifest(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    observed = _verify_file(path, expected_sha256, "R4 decode input manifest")
    value = _load_json(path, "R4 decode input manifest")
    _strict_keys(
        value,
        {
            "schema",
            "status",
            "stage",
            "protocol_contract",
            "configuration_manifest",
            "population_manifest",
            "population_amendment",
            "stage_gate",
            "videos",
        },
        "R4 decode input manifest",
    )
    if (
        value.get("schema") != INPUT_SCHEMA
        or value.get("status") != "SEALED_TARGET_BLIND_INPUTS"
    ):
        raise DivisionDecodeError("input manifest is not sealed target-blind input")
    videos = value.get("videos")
    if not isinstance(videos, list) or not videos:
        raise DivisionDecodeError("input manifest contains no videos")
    identities: list[str] = []
    for index, row in enumerate(videos):
        if not isinstance(row, Mapping):
            raise DivisionDecodeError(f"video row {index} is not an object")
        _strict_keys(
            row,
            {"identity", "association_bank", "base_graph"},
            f"video row {index}",
        )
        identity = str(row["identity"])
        if not re.fullmatch(r"(?:train|test):[A-Za-z0-9_.-]+", identity):
            raise DivisionDecodeError(f"video row {index} identity is invalid")
        _reference(row["association_bank"], path.parent, f"video {identity} bank")
        _reference(row["base_graph"], path.parent, f"video {identity} graph")
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise DivisionDecodeError("input manifest video identities repeat")
    return value, observed


def run_from_manifest(
    *,
    input_manifest: Path,
    input_manifest_sha256: str,
    config_name: str,
    repo_root: Path,
) -> DecodeRun:
    input_value, input_hash = _load_input_manifest(
        input_manifest, input_manifest_sha256
    )
    protocol_path, protocol_hash = _reference(
        input_value["protocol_contract"], input_manifest.parent, "R4 protocol"
    )
    config_path, config_file_hash = _reference(
        input_value["configuration_manifest"],
        input_manifest.parent,
        "configuration manifest",
    )
    population_path, population_file_hash = _reference(
        input_value["population_manifest"],
        input_manifest.parent,
        "R4 population manifest",
    )
    amendment_path, amendment_hash = _reference(
        input_value["population_amendment"],
        input_manifest.parent,
        "R4 population amendment",
    )
    protocol = _load_verified_json(protocol_path, protocol_hash, "R4 protocol")
    configuration_manifest = _load_verified_json(
        config_path, config_file_hash, "configuration manifest"
    )
    population = _load_verified_json(
        population_path, population_file_hash, "R4 population manifest"
    )
    amendment = _load_verified_json(
        amendment_path, amendment_hash, "R4 population amendment"
    )
    _verify_protocol(
        protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        configuration_path=config_path,
        configuration_sha256=config_file_hash,
        repo_root=repo_root,
    )
    config = _load_configuration(configuration_manifest, protocol, config_name)
    identities_by_role = _verify_population_contract(
        population=population,
        population_path=population_path,
        population_file_sha256=population_file_hash,
        amendment=amendment,
        amendment_path=amendment_path,
        amendment_sha256=amendment_hash,
        protocol=protocol,
        protocol_sha256=protocol_hash,
        repo_root=repo_root,
    )
    stage = str(input_value["stage"])
    observed_identities = {str(row["identity"]) for row in input_value["videos"]}
    contaminated = identities_by_role["excluded_contaminated_audit20"]
    if observed_identities & contaminated:
        raise DivisionDecodeError("contaminated audit20 identity is forbidden for R4")
    if stage == "development_staging":
        expected = identities_by_role["development_nested_oof"]
        if not observed_identities or not observed_identities < expected:
            raise DivisionDecodeError("development staging must be a strict nonempty subset")
        score_eligible = False
    elif stage == "development_nested_oof":
        expected = identities_by_role["development_nested_oof"]
        if observed_identities != expected:
            raise DivisionDecodeError("development OOF requires all 69 sealed identities")
        score_eligible = True
    elif stage == "held_fit_one_shot_confirmation":
        expected = identities_by_role["held_fit_one_shot_confirmation"]
        if observed_identities != expected:
            raise DivisionDecodeError("held-fit confirmation requires all 36 identities")
        score_eligible = True
    elif stage == "sealed_hidden_test_probe":
        expected = identities_by_role["sealed_hidden_test_probe"]
        if observed_identities != expected:
            raise DivisionDecodeError("hidden-test probe requires all four identities")
        score_eligible = False
    else:
        raise DivisionDecodeError(f"unsupported R4 execution stage: {stage}")
    stage_gate_hash = _verify_stage_gate(
        stage=stage,
        raw_gate=input_value["stage_gate"],
        manifest_base=input_manifest.parent,
        config=config,
        protocol_sha256=protocol_hash,
        population_sha256=str(population["population_sha256"]),
        configuration_file_sha256=config_file_hash,
        configuration_sha256=str(configuration_manifest["configuration_sha256"]),
    )
    api = _compiler_api(repo_root)

    graphs: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for row in sorted(input_value["videos"], key=lambda value: str(value["identity"])):
        identity = str(row["identity"])
        bank_path, bank_hash = _reference(
            row["association_bank"], input_manifest.parent, f"{identity} bank"
        )
        graph_path, graph_file_hash = _reference(
            row["base_graph"], input_manifest.parent, f"{identity} graph"
        )
        bank = load_association_bank(
            bank_path,
            expected_sha256=bank_hash,
            expected_identity=identity,
        )
        base_graph = _load_base_graph(
            graph_path,
            expected_sha256=graph_file_hash,
            identity=identity,
            bank=bank,
            api=api,
        )
        decoded, record = decode_video(bank, base_graph, config, api)
        record = dict(record)
        record["association_bank_sha256"] = bank_hash
        record["base_graph_file_sha256"] = graph_file_hash
        record["bank_node_count"] = len(bank.nodes)
        record["bank_candidate_edge_count"] = len(bank.edges)
        graphs[identity] = decoded
        records.append(record)

    duplicate_survivors = sum(
        int(row["suppression"]["duplicate_survivors"]) for row in records
    )
    if duplicate_survivors != 0:
        raise DivisionDecodeError("FAIL_CLOSED_DUPLICATE_FORK")
    predictions = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_PREDICTIONS_BEFORE_ANY_SCORE_READ",
        "input_manifest_sha256": input_hash,
        "protocol_contract_sha256": protocol_hash,
        "population_manifest": {
            "file_sha256": population_file_hash,
            "population_sha256": population["population_sha256"],
        },
        "population_amendment_sha256": amendment_hash,
        "stage": stage,
        "stage_gate_sha256": stage_gate_hash,
        "configuration_manifest": {
            "file_sha256": config_file_hash,
            "configuration_sha256": configuration_manifest[
                "configuration_sha256"
            ],
        },
        "configuration": asdict(config),
        "semantics": {
            "daughter_persistence": "selected_base_graph_one_frame_steps",
            "grandchild_persistence": (
                "complete_bank_highest_probability_one_frame_steps"
            ),
            "direct_late_equivalence": (
                "predicted_parent_lineage_plus_unordered_daughter_lineages"
            ),
            "equivalence_precedes_node_ownership": True,
            "evidence_edges_emitted": False,
        },
        "scope": {
            "target_access": False,
            "gt_access": False,
            "scorer_access": False,
            "audit_confirmation_access": False,
            "cv57_access": False,
            "gpu_hours": 0.0,
            "score_eligible_after_seal": score_eligible,
            "contaminated_audit20_access": False,
        },
        "videos": records,
        "aggregate": {
            "video_count": len(records),
            "bank_node_count": sum(int(row["bank_node_count"]) for row in records),
            "bank_candidate_edge_count": sum(
                int(row["bank_candidate_edge_count"]) for row in records
            ),
            "raw_event_count": sum(
                int(row["generation"]["raw_event_count"]) for row in records
            ),
            "accepted_event_count": sum(
                int(row["suppression"]["accepted_events"]) for row in records
            ),
            "duplicate_survivors": duplicate_survivors,
        },
    }
    predictions["prediction_identity_sha256"] = canonical_sha256(predictions)
    return DecodeRun(predictions=predictions, graphs=graphs)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _slug(identity: str) -> str:
    return identity.replace(":", "__", 1)


def write_sealed_output(
    output: Path, run: DecodeRun, *, repo_root: Path
) -> Mapping[str, Any]:
    """Atomically publish predictions and graphs, writing the seal last."""

    if output.exists():
        raise DivisionDecodeError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    api = _compiler_api(repo_root)
    try:
        graphs_dir = staging / "graphs"
        graphs_dir.mkdir()
        prediction_path = staging / "PREDICTIONS.json"
        _write_bytes(prediction_path, canonical_json_bytes(run.predictions))
        graph_rows = []
        records = {
            str(row["identity"]): row for row in run.predictions.get("videos", [])
        }
        for identity in sorted(run.graphs):
            graph = run.graphs[identity]
            relative = Path("graphs") / f"{_slug(identity)}.csv"
            path = staging / relative
            api["write_submission"]({graph.dataset: graph}, path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            expected_graph_hash = records[identity]["decoded_graph_sha256"]
            observed_graph_hash = api["graph_sha256"](graph)
            if observed_graph_hash != expected_graph_hash:
                raise DivisionDecodeError("decoded graph changed before sealing")
            graph_rows.append(
                {
                    "identity": identity,
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "graph_sha256": observed_graph_hash,
                }
            )
        seal = {
            "schema": SEAL_SCHEMA,
            "status": (
                "SEALED_COMPLETE_STAGING_NOT_SCORE_ELIGIBLE"
                if run.predictions["stage"] == "development_staging"
                else "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING"
            ),
            "input_manifest_sha256": run.predictions["input_manifest_sha256"],
            "protocol_contract_sha256": run.predictions[
                "protocol_contract_sha256"
            ],
            "population_sha256": run.predictions["population_manifest"][
                "population_sha256"
            ],
            "population_amendment_sha256": run.predictions[
                "population_amendment_sha256"
            ],
            "stage": run.predictions["stage"],
            "configuration_name": run.predictions["configuration"]["name"],
            "prediction_identity_sha256": run.predictions[
                "prediction_identity_sha256"
            ],
            "predictions": {
                "path": "PREDICTIONS.json",
                "sha256": sha256_file(prediction_path),
                "bytes": prediction_path.stat().st_size,
            },
            "graphs": graph_rows,
            "scope": {
                "target_access": False,
                "gt_access": False,
                "scorer_access": False,
                "audit_confirmation_access": False,
                "cv57_access": False,
                "all_outputs_complete": True,
            },
        }
        _write_bytes(staging / "SEAL.json", canonical_json_bytes(seal))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists():
            raise DivisionDecodeError(f"refusing to overwrite output: {output}")
        os.rename(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verified = verify_sealed_output(output)
    return {
        **verified,
        "output": str(output),
        "seal_sha256": sha256_file(output / "SEAL.json"),
    }


def verify_sealed_output(
    output: Path, *, expected_seal_sha256: str | None = None
) -> Mapping[str, Any]:
    seal_path = output / "SEAL.json"
    if expected_seal_sha256 is not None:
        _verify_file(seal_path, expected_seal_sha256, "R4 output seal")
    seal = _load_json(seal_path, "R4 output seal")
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status")
        not in {
            "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING",
            "SEALED_COMPLETE_STAGING_NOT_SCORE_ELIGIBLE",
        }
        or seal.get("scope", {}).get("all_outputs_complete") is not True
        or seal.get("scope", {}).get("target_access") is not False
        or seal.get("scope", {}).get("gt_access") is not False
        or seal.get("scope", {}).get("scorer_access") is not False
    ):
        raise DivisionDecodeError("output seal contract drifted")
    prediction_record = seal.get("predictions")
    if not isinstance(prediction_record, Mapping):
        raise DivisionDecodeError("output seal has no prediction record")
    if prediction_record.get("path") != "PREDICTIONS.json":
        raise DivisionDecodeError("sealed prediction path drifted")
    prediction_path = output / "PREDICTIONS.json"
    _verify_file(
        prediction_path,
        str(prediction_record.get("sha256")),
        "sealed predictions",
    )
    if prediction_path.stat().st_size != int(prediction_record.get("bytes", -1)):
        raise DivisionDecodeError("sealed prediction byte count drifted")
    predictions = _load_json(prediction_path, "sealed predictions")
    stored_prediction_identity = predictions.get("prediction_identity_sha256")
    identity_payload = dict(predictions)
    identity_payload.pop("prediction_identity_sha256", None)
    if (
        predictions.get("status") != "SEALED_PREDICTIONS_BEFORE_ANY_SCORE_READ"
        or stored_prediction_identity != canonical_sha256(identity_payload)
        or stored_prediction_identity != seal.get("prediction_identity_sha256")
        or predictions.get("stage") != seal.get("stage")
        or predictions.get("population_manifest", {}).get("population_sha256")
        != seal.get("population_sha256")
        or (
            seal.get("status") == "SEALED_COMPLETE_STAGING_NOT_SCORE_ELIGIBLE"
            and predictions.get("scope", {}).get("score_eligible_after_seal")
            is not False
        )
    ):
        raise DivisionDecodeError("sealed prediction identity drifted")
    graph_rows = seal.get("graphs")
    if not isinstance(graph_rows, list) or len(graph_rows) != int(
        predictions.get("aggregate", {}).get("video_count", -1)
    ):
        raise DivisionDecodeError("sealed graph inventory drifted")
    prediction_videos = {
        str(row.get("identity")): row
        for row in predictions.get("videos", [])
        if isinstance(row, Mapping)
    }
    seen_identities: set[str] = set()
    for row in graph_rows:
        if not isinstance(row, Mapping):
            raise DivisionDecodeError("sealed graph record is malformed")
        identity = str(row.get("identity"))
        relative = Path(str(row.get("path")))
        if (
            identity in seen_identities
            or relative.is_absolute()
            or relative.parts != ("graphs", f"{_slug(identity)}.csv")
            or identity not in prediction_videos
            or row.get("graph_sha256")
            != prediction_videos[identity].get("decoded_graph_sha256")
        ):
            raise DivisionDecodeError("sealed graph binding drifted")
        seen_identities.add(identity)
        graph_path = output / relative
        _verify_file(graph_path, str(row.get("sha256")), "sealed decoded graph")
        if graph_path.stat().st_size != int(row.get("bytes", -1)):
            raise DivisionDecodeError("sealed graph byte count drifted")
    return {
        "status": seal["status"],
        "configuration_name": seal["configuration_name"],
        "prediction_identity_sha256": seal["prediction_identity_sha256"],
        "video_count": len(graph_rows),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", required=True, type=Path)
    result.add_argument("--input-manifest", required=True, type=Path)
    result.add_argument("--input-manifest-sha256", required=True)
    result.add_argument("--config-name", required=True)
    destination = result.add_mutually_exclusive_group(required=True)
    destination.add_argument("--dry-run", action="store_true")
    destination.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run = run_from_manifest(
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        config_name=args.config_name,
        repo_root=args.repo_root,
    )
    if args.dry_run:
        report = {
            "status": "PASS_TARGET_BLIND_DRY_RUN_NO_WRITES",
            "stage": run.predictions["stage"],
            "score_eligible_after_seal": run.predictions["scope"][
                "score_eligible_after_seal"
            ],
            "configuration_name": run.predictions["configuration"]["name"],
            "prediction_identity_sha256": run.predictions[
                "prediction_identity_sha256"
            ],
            "video_count": run.predictions["aggregate"]["video_count"],
            "raw_event_count": run.predictions["aggregate"]["raw_event_count"],
            "accepted_event_count": run.predictions["aggregate"][
                "accepted_event_count"
            ],
            "duplicate_survivors": run.predictions["aggregate"][
                "duplicate_survivors"
            ],
            "writes": False,
        }
    else:
        report = write_sealed_output(args.output, run, repo_root=args.repo_root)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

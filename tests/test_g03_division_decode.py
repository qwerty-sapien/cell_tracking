from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pytest

from scripts.g03_division_decode import (
    BANK_SCHEMA,
    INPUT_SCHEMA,
    POPULATION_AMENDMENT_SCHEMA,
    POPULATION_SCHEMA,
    PROTOCOL_SCHEMA,
    AssociationBank,
    CandidateEvent,
    DivisionDecodeConfig,
    DivisionDecodeError,
    _build_incident_index,
    _compiler_api,
    _incident_conflicts,
    _load_base_graph,
    decode_video,
    load_association_bank,
    run_from_manifest,
    select_modal_outer_winner,
    sha256_file,
    suppress_candidates,
    verify_sealed_output,
    write_sealed_output,
)
from scripts.g03_division_decode_manifest import build_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
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


def write_json(path: Path, value: Any) -> str:
    path.write_bytes(canonical_bytes(value))
    return sha256_file(path)


def synthetic_rows(*, include_collisions: bool = False, include_second: bool = False):
    rows = [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (9, 1),
        (10, 2),
        (11, 3),
        (12, 4),
    ]
    graph_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (9, 10), (10, 11), (11, 12)]
    bank_edges = [
        (1, 2, 0.99),
        (2, 3, 0.90),
        (2, 10, 0.80),
        (3, 4, 0.95),
        (4, 5, 0.94),
        (9, 10, 0.90),
        (10, 11, 0.93),
        (11, 12, 0.92),
    ]
    if include_collisions:
        rows.extend([(19, 1), (13, 2), (14, 3), (15, 4)])
        graph_edges.extend([(19, 13), (13, 14), (14, 15)])
        bank_edges.extend(
            [(19, 13, 0.90), (2, 13, 0.75), (13, 14, 0.91), (14, 15, 0.90)]
        )
    if include_second:
        rows.extend(
            [
                (20, 0),
                (21, 1),
                (22, 2),
                (23, 3),
                (24, 4),
                (29, 1),
                (30, 2),
                (31, 3),
                (32, 4),
            ]
        )
        graph_edges.extend(
            [
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (29, 30),
                (30, 31),
                (31, 32),
            ]
        )
        bank_edges.extend(
            [
                (20, 21, 0.98),
                (21, 22, 0.85),
                (21, 30, 0.75),
                (22, 23, 0.92),
                (23, 24, 0.91),
                (29, 30, 0.89),
                (30, 31, 0.90),
                (31, 32, 0.89),
            ]
        )
    return rows, graph_edges, bank_edges


def write_bank(
    path: Path,
    *,
    identity: str,
    rows: list[tuple[int, int]],
    bank_edges: list[tuple[int, int, float]],
    metadata_override: Mapping[str, Any] | None = None,
) -> str:
    metadata = {
        "schema": BANK_SCHEMA,
        "identity": identity,
        "probability_dtype": "float32",
        "distance_unit": "micrometre",
        "complete_candidate_bank": True,
    }
    if metadata_override:
        metadata.update(metadata_override)
    node_ids = np.asarray([node for node, _frame in rows], dtype=np.int64)
    frames = np.asarray([frame for _node, frame in rows], dtype=np.int32)
    coords = np.asarray(
        [[float(node), float(node % 7), float(node % 5)] for node in node_ids],
        dtype=np.float32,
    )
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.frombuffer(canonical_bytes(metadata), dtype=np.uint8),
            node_id=node_ids,
            frame=frames,
            coords_zyx=coords,
            edge_source=np.asarray([row[0] for row in bank_edges], dtype=np.int64),
            edge_target=np.asarray([row[1] for row in bank_edges], dtype=np.int64),
            probability=np.asarray([row[2] for row in bank_edges], dtype=np.float32),
            distance_um=np.asarray([1.0 for _row in bank_edges], dtype=np.float32),
        )
    return sha256_file(path)


def make_graph(
    path: Path,
    *,
    dataset: str,
    rows: list[tuple[int, int]],
    graph_edges: list[tuple[int, int]],
) -> tuple[Any, str]:
    api = _compiler_api(REPO_ROOT)
    from celltrack_alpha.kaggle_calibration import (
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
    )

    graph = SubmissionGraph(
        dataset,
        {
            node: SubmissionNode(
                node_id=node,
                t=frame,
                z=node,
                y=node % 7,
                x=node % 5,
            )
            for node, frame in rows
        },
        [SubmissionEdge(source, target) for source, target in graph_edges],
    )
    api["write_submission"]({dataset: graph}, path)
    return graph, sha256_file(path)


def active_config(**changes: Any) -> DivisionDecodeConfig:
    values = {
        "name": "synthetic_active",
        "divisions": True,
        "fork_threshold": 0.35,
        "max_primary_secondary_gap": 1.0,
        "late_window": 1,
        "daughter_persistence": 2,
        "grandchild_persistence": 2,
        "duplicate_temporal_window": 2,
        "per_video_cap": None,
    }
    values.update(changes)
    return DivisionDecodeConfig(**values)


def loaded_fixture(
    tmp_path: Path, *, include_collisions: bool = False, include_second: bool = False
) -> tuple[AssociationBank, Any, Mapping[str, Any]]:
    identity = "train:44b6_fixture"
    rows, graph_edges, bank_edges = synthetic_rows(
        include_collisions=include_collisions,
        include_second=include_second,
    )
    bank_path = tmp_path / "bank.npz"
    bank_hash = write_bank(
        bank_path,
        identity=identity,
        rows=rows,
        bank_edges=bank_edges,
    )
    graph, _graph_hash = make_graph(
        tmp_path / "graph.csv",
        dataset="44b6_fixture",
        rows=rows,
        graph_edges=graph_edges,
    )
    bank = load_association_bank(
        bank_path,
        expected_sha256=bank_hash,
        expected_identity=identity,
    )
    return bank, graph, _compiler_api(REPO_ROOT)


def test_bank_requires_exact_hash_schema_and_numeric_contract(tmp_path: Path) -> None:
    rows, _graph_edges, bank_edges = synthetic_rows()
    bank_path = tmp_path / "bank.npz"
    bank_hash = write_bank(
        bank_path,
        identity="train:44b6_fixture",
        rows=rows,
        bank_edges=bank_edges,
    )
    with pytest.raises(DivisionDecodeError, match="SHA-256 mismatch"):
        load_association_bank(
            bank_path,
            expected_sha256="0" * 64,
            expected_identity="train:44b6_fixture",
        )
    malformed = tmp_path / "malformed.npz"
    malformed_hash = write_bank(
        malformed,
        identity="train:44b6_fixture",
        rows=rows,
        bank_edges=bank_edges,
        metadata_override={"complete_candidate_bank": False},
    )
    with pytest.raises(DivisionDecodeError, match="metadata contract drifted"):
        load_association_bank(
            malformed,
            expected_sha256=malformed_hash,
            expected_identity="train:44b6_fixture",
        )


def test_hidden_test_parent_graph_identity_is_supported_without_new_nodes(
    tmp_path: Path,
) -> None:
    identity = "test:44b6_fixture"
    rows, graph_edges, bank_edges = synthetic_rows()
    bank_path = tmp_path / "hidden_bank.npz"
    bank_hash = write_bank(
        bank_path,
        identity=identity,
        rows=rows,
        bank_edges=bank_edges,
    )
    graph_path = tmp_path / "hidden_graph.csv"
    graph, graph_hash = make_graph(
        graph_path,
        dataset="44b6_fixture",
        rows=rows,
        graph_edges=graph_edges,
    )
    bank = load_association_bank(
        bank_path,
        expected_sha256=bank_hash,
        expected_identity=identity,
    )
    loaded = _load_base_graph(
        graph_path,
        expected_sha256=graph_hash,
        identity=identity,
        bank=bank,
        api=_compiler_api(REPO_ROOT),
    )
    assert set(loaded.nodes) == set(graph.nodes)


def test_direct_wins_late_equivalence_and_compiles_valid_n1_fork(
    tmp_path: Path,
) -> None:
    bank, graph, api = loaded_fixture(tmp_path)
    decoded, record = decode_video(bank, graph, active_config(), api)
    assert record["generation"] == {
        "bank_sources_with_two_forward_candidates": 1,
        "candidate_pairs_considered": 1,
        "candidate_pairs_threshold_pass": 1,
        "persistence_pass": 1,
        "raw_event_count": 2,
    }
    assert record["suppression"]["rejected_exact_equivalence"] == 1
    assert record["suppression"]["duplicate_survivors"] == 0
    assert len(record["accepted"]) == 1
    winner = record["accepted"][0]
    assert winner["mode"] == "DIRECT"
    assert winner["daughter_persistence_nodes"] == (4, 5, 11, 12)
    assert winner["grandchild_persistence_nodes"] == (4, 5, 11, 12)
    assert winner["action"]["add_edges"] == [[2, 3], [2, 10]]
    assert winner["action"]["expected_conflict_edges"] == [[2, 3], [9, 10]]
    final_edges = {(edge.source_id, edge.target_id) for edge in decoded.edges}
    assert set(decoded.nodes) == set(graph.nodes)
    assert (2, 3) in final_edges and (2, 10) in final_edges
    assert (9, 10) not in final_edges
    assert record["n1_batch"]["after_graph_sha256"] == record[
        "decoded_graph_sha256"
    ]
    assert record["n1_batch"]["compiler_path"] == "frozen_prepare_action_one_pass"
    assert record["n1_batch"][
        "order_independence_verified_by_disjoint_primitive_ownership"
    ] is True
    assert record["n1_batch"]["incident_index_built_once"] is True
    assert record["n1_batch"]["full_graph_constructions"] == 1
    assert record["n1_batch"]["full_graph_hashes"] == 1
    api["validate_graphs"]({decoded.dataset: decoded})


def synthetic_event(
    event_id: str,
    parent: int,
    children: tuple[int, int],
    *,
    occupied: tuple[int, ...] | None = None,
) -> CandidateEvent:
    return CandidateEvent(
        event_id=event_id,
        equivalence_key=f"eq-{event_id}",
        identity="train:6bba_fixture",
        mode="DIRECT",
        late_depth=0,
        anchor_parent=parent,
        emitted_parent=parent,
        primary_child=children[0],
        secondary_child=children[1],
        children=children,
        parent_lineage=parent,
        daughter_lineages=children,
        primary_probability=0.9,
        secondary_probability=0.8,
        primary_secondary_gap=0.1,
        continuation_nodes=(),
        daughter_persistence_nodes=(),
        grandchild_persistence_nodes=(),
        occupied_nodes=occupied or (parent, *children),
    )


def test_conflict_ownership_is_integrated_before_cap_and_rejected_events_own_nothing() -> None:
    index = _build_incident_index(((1, 8), (2, 5), (10, 11)))
    first = synthetic_event("first", 1, (3, 4))
    conflict = synthetic_event("conflict", 2, (8, 9))
    replacement = synthetic_event("replacement", 10, (11, 12), occupied=(2, 10, 11, 12))
    config = DivisionDecodeConfig(
        name="synthetic",
        divisions=True,
        fork_threshold=0.5,
        max_primary_secondary_gap=1.0,
        late_window=0,
        daughter_persistence=0,
        grandchild_persistence=0,
        duplicate_temporal_window=0,
        per_video_cap=None,
    )
    accepted, rejected, counts = suppress_candidates(
        (first, conflict, replacement), config, incident_index=index
    )
    assert tuple(event.event_id for event in accepted) == ("first", "replacement")
    assert [row["reason"] for row in rejected] == ["conflict_edge_ownership"]
    assert counts["rejected_conflict_edge_ownership"] == 1
    assert counts["accepted_events"] == 2

    capped = DivisionDecodeConfig(**{**config.__dict__, "per_video_cap": 1})
    accepted, rejected, counts = suppress_candidates(
        (first, conflict, replacement), capped, incident_index=index
    )
    assert tuple(event.event_id for event in accepted) == ("first",)
    assert [row["reason"] for row in rejected] == [
        "conflict_edge_ownership",
        "per_video_cap",
    ]
    assert counts["rejected_conflict_edge_ownership"] == 1
    assert counts["rejected_per_video_cap"] == 1


def test_incident_index_exactly_matches_full_baseline_scan() -> None:
    edges = ((1, 8), (1, 3), (2, 5), (9, 4), (10, 11))
    index = _build_incident_index(edges)
    for parent, children in ((1, (3, 4)), (2, (8, 11)), (99, (98, 97))):
        expected = tuple(
            sorted(
                pair
                for pair in edges
                if pair[0] == parent or pair[1] in set(children)
            )
        )
        assert _incident_conflicts(index, parent, children) == expected


def test_temporal_index_preserves_earliest_ranked_winner() -> None:
    first = synthetic_event("first", 1, (10, 11))
    second = CandidateEvent(
        **{
            **synthetic_event("second", 1, (10, 11)).__dict__,
            "late_depth": 2,
        }
    )
    between = CandidateEvent(
        **{
            **synthetic_event("between", 1, (10, 11)).__dict__,
            "late_depth": 1,
        }
    )
    after = CandidateEvent(
        **{
            **synthetic_event("after", 1, (10, 11)).__dict__,
            "late_depth": 3,
        }
    )
    config = DivisionDecodeConfig(
        name="temporal-index",
        divisions=True,
        fork_threshold=0.5,
        max_primary_secondary_gap=1.0,
        late_window=3,
        daughter_persistence=0,
        grandchild_persistence=0,
        duplicate_temporal_window=1,
        per_video_cap=None,
    )
    _accepted, rejected, counts = suppress_candidates(
        (first, second, between, after),
        config,
        incident_index=_build_incident_index(()),
    )
    temporal = [
        row for row in rejected if row["reason"] == "duplicate_temporal_window"
    ]
    assert temporal == [
        {
            "event_id": "between",
            "reason": "duplicate_temporal_window",
            "winner_event_id": "first",
        },
        {
            "event_id": "after",
            "reason": "duplicate_temporal_window",
            "winner_event_id": "second",
        },
    ]
    assert counts["rejected_duplicate_temporal_window"] == 2


def test_persistence_node_ownership_and_per_video_cap_are_ordered(
    tmp_path: Path,
) -> None:
    bank, graph, api = loaded_fixture(
        tmp_path, include_collisions=True, include_second=True
    )
    config = active_config(per_video_cap=1)
    _decoded, record = decode_video(bank, graph, config, api)
    assert record["suppression"]["accepted_events"] == 1
    assert record["accepted"][0]["emitted_parent"] == 2
    assert record["suppression"]["rejected_node_ownership"] >= 1
    assert record["suppression"]["rejected_per_video_cap"] >= 1
    assert record["suppression"]["duplicate_survivors"] == 0

    too_deep = active_config(daughter_persistence=3, grandchild_persistence=3)
    _decoded, filtered = decode_video(bank, graph, too_deep, api)
    assert filtered["generation"]["persistence_pass"] == 0
    assert filtered["suppression"]["accepted_events"] == 0


def test_divisions_off_is_exact_parent_graph_noop(tmp_path: Path) -> None:
    bank, graph, api = loaded_fixture(tmp_path)
    config = DivisionDecodeConfig(
        name="DIVISIONS_OFF",
        divisions=False,
        fork_threshold=None,
        max_primary_secondary_gap=None,
        late_window=0,
        daughter_persistence=0,
        grandchild_persistence=0,
        duplicate_temporal_window=0,
        per_video_cap=0,
    )
    decoded, record = decode_video(bank, graph, config, api)
    assert api["graph_sha256"](decoded) == api["graph_sha256"](graph)
    assert record["accepted"] == []
    assert record["n1_batch"] is None


def build_manifest_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    identity = "train:6bba_fixture"
    rows, graph_edges, bank_edges = synthetic_rows()
    bank_path = tmp_path / "bank.npz"
    bank_hash = write_bank(
        bank_path,
        identity=identity,
        rows=rows,
        bank_edges=bank_edges,
    )
    graph_path = tmp_path / "graph.csv"
    _graph, graph_hash = make_graph(
        graph_path,
        dataset="6bba_fixture",
        rows=rows,
        graph_edges=graph_edges,
    )

    configuration = build_manifest()
    config_path = tmp_path / "configuration.json"
    config_hash = write_json(config_path, configuration)
    config_name = "t035_g100_l1_dp2_gp2_n2_call"
    assert any(row["name"] == config_name for row in configuration["configurations"])

    design_path = tmp_path / "SEALED_DESIGN.md"
    design_path.write_text("sealed synthetic design\n", encoding="ascii")
    action_compiler = REPO_ROOT / "src/celltrack_alpha/action_compiler.py"
    calibration = REPO_ROOT / "src/celltrack_alpha/kaggle_calibration.py"
    generator = REPO_ROOT / "scripts/g03_division_decode_manifest.py"
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "status": "SEALED_BEFORE_ANY_G03_DIVISION_OUTCOME",
        "protocol": {
            "design_path": str(design_path),
            "design_sha256": sha256_file(design_path),
            "adaptive_search_forbidden": True,
            "configuration_splicing_forbidden": True,
            "duplicate_survivors_required": 0,
            "gt_for_candidate_generation_forbidden": True,
        },
        "configuration_manifest": {
            "path": str(config_path),
            "file_sha256": config_hash,
            "configuration_sha256": configuration["configuration_sha256"],
            "active_configuration_count": 18_144,
            "configuration_count_including_null": 18_145,
            "generator_path": str(generator),
            "generator_sha256": sha256_file(generator),
        },
        "frozen_dependencies": {
            "src/celltrack_alpha/action_compiler.py": sha256_file(action_compiler),
            "src/celltrack_alpha/kaggle_calibration.py": sha256_file(calibration),
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_hash = write_json(protocol_path, protocol)

    development = [identity] + [f"train:6bba_dev{index:02d}" for index in range(68)]
    held_fit = [f"train:44b6_held{index:02d}" for index in range(36)]
    hidden = [
        "test:44b6_hidden00",
        "test:44b6_hidden01",
        "test:6bba_hidden00",
        "test:6bba_hidden01",
    ]
    contaminated = [f"train:audit_exposed{index:02d}" for index in range(20)]

    def rows_for(identities: list[str], role: str) -> list[dict[str, str]]:
        return [
            {
                "identity": value,
                "group_id": f"{role}-group-{index:03d}",
                "registry_stratum": f"{role}|synthetic",
            }
            for index, value in enumerate(identities)
        ]

    development_rows = rows_for(development, "development")
    folds = {
        str(fold): [
            row for ordinal, row in enumerate(development_rows) if ordinal % 5 == fold
        ]
        for fold in range(5)
    }
    constraints = {
        "family_specific_configuration_splicing": False,
        "held_fit_can_select_configuration": False,
        "held_fit_decoder_overlay_evaluations_allowed": 1,
        "confirmation_can_select_configuration": False,
        "division_labels_used_for_split": False,
        "split_stratification_source": "target-blind frozen registry_stratum",
        "all_association_banks_use_one_R1_scientific_configuration": True,
        "development_bank_count_required": 69,
        "held_fit_bank_count_required": 36,
        "hidden_test_bank_count_required": 4,
        "contaminated_audit20_bank_count_required": 0,
    }
    roles = {
        "development_nested_oof": {
            "description": "synthetic development",
            "video_count": 69,
            "target_access": "presealed_predictions_only",
            "videos": development_rows,
        },
        "held_fit_one_shot_confirmation": {
            "description": "synthetic held fit",
            "video_count": 36,
            "target_access": "one_shot",
            "videos": rows_for(held_fit, "held"),
        },
        "sealed_hidden_test_probe": {
            "description": "synthetic hidden",
            "video_count": 4,
            "target_access": "downstream_only",
            "videos": rows_for(hidden, "hidden"),
        },
        "excluded_contaminated_audit20": {
            "description": "synthetic contaminated",
            "video_count": 20,
            "target_access": "forbidden_for_this_R4_node",
            "videos": rows_for(contaminated, "audit"),
        },
    }
    population = {
        "schema": POPULATION_SCHEMA,
        "status": "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "seed": "synthetic-fivefold",
        "sources": {},
        "roles": roles,
        "outer_folds": folds,
        "constraints": constraints,
    }
    population["population_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "seed": population["seed"],
                "roles": roles,
                "outer_folds": folds,
                "constraints": constraints,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    population_path = tmp_path / "population.json"
    population_file_hash = write_json(population_path, population)
    population_generator = tmp_path / "population_generator.py"
    population_generator.write_text("# sealed synthetic population generator\n", encoding="ascii")
    amendment = {
        "schema": POPULATION_AMENDMENT_SCHEMA,
        "status": "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "amends": {
            "design_sha256": sha256_file(design_path),
            "protocol_sha256": protocol_hash,
        },
        "scientific_configuration": {
            "modal_outer_winner_rule": (
                "highest outer-winner count, then lexicographically smallest config name"
            ),
            "modal_tie_uses_metric_repooling": False,
        },
        "population_manifest": {
            "file_sha256": population_file_hash,
            "population_sha256": population["population_sha256"],
            "generator_path": str(population_generator),
            "generator_sha256": sha256_file(population_generator),
        },
        "binding_roles": {
            "development_nested_oof": {"video_count": 69},
            "held_fit_one_shot_confirmation": {"video_count": 36},
            "sealed_hidden_test_probe": {"video_count": 4},
            "excluded_contaminated_audit20": {
                "target_access_for_R4_allowed": False
            },
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
    amendment_path = tmp_path / "population_amendment.json"
    amendment_hash = write_json(amendment_path, amendment)
    input_value = {
        "schema": INPUT_SCHEMA,
        "status": "SEALED_TARGET_BLIND_INPUTS",
        "stage": "development_staging",
        "protocol_contract": {"path": str(protocol_path), "sha256": protocol_hash},
        "configuration_manifest": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "population_manifest": {
            "path": str(population_path),
            "sha256": population_file_hash,
        },
        "population_amendment": {
            "path": str(amendment_path),
            "sha256": amendment_hash,
        },
        "stage_gate": None,
        "videos": [
            {
                "identity": identity,
                "association_bank": {
                    "path": str(bank_path),
                    "sha256": bank_hash,
                },
                "base_graph": {"path": str(graph_path), "sha256": graph_hash},
            }
        ],
    }
    input_path = tmp_path / "INPUT.json"
    input_hash = write_json(input_path, input_value)
    return input_path, input_hash, config_name


def test_dry_run_identity_matches_atomic_pre_score_seal_and_tamper_fails(
    tmp_path: Path,
) -> None:
    input_path, input_hash, config_name = build_manifest_fixture(tmp_path)
    run = run_from_manifest(
        input_manifest=input_path,
        input_manifest_sha256=input_hash,
        config_name=config_name,
        repo_root=REPO_ROOT,
    )
    replay = run_from_manifest(
        input_manifest=input_path,
        input_manifest_sha256=input_hash,
        config_name=config_name,
        repo_root=REPO_ROOT,
    )
    assert replay.predictions == run.predictions
    assert {
        identity: _compiler_api(REPO_ROOT)["graph_sha256"](graph)
        for identity, graph in replay.graphs.items()
    } == {
        identity: _compiler_api(REPO_ROOT)["graph_sha256"](graph)
        for identity, graph in run.graphs.items()
    }
    output = tmp_path / "sealed"
    receipt = write_sealed_output(output, run, repo_root=REPO_ROOT)
    assert receipt["prediction_identity_sha256"] == run.predictions[
        "prediction_identity_sha256"
    ]
    assert receipt["status"] == "SEALED_COMPLETE_STAGING_NOT_SCORE_ELIGIBLE"
    assert verify_sealed_output(
        output, expected_seal_sha256=receipt["seal_sha256"]
    )["video_count"] == 1
    predictions = json.loads((output / "PREDICTIONS.json").read_text(encoding="ascii"))
    assert predictions["scope"]["target_access"] is False
    assert predictions["scope"]["gt_access"] is False
    assert predictions["scope"]["scorer_access"] is False
    assert predictions["scope"]["audit_confirmation_access"] is False
    assert predictions["scope"]["cv57_access"] is False
    assert predictions["scope"]["score_eligible_after_seal"] is False
    assert predictions["scope"]["contaminated_audit20_access"] is False
    with pytest.raises(DivisionDecodeError, match="refusing to overwrite"):
        write_sealed_output(output, run, repo_root=REPO_ROOT)

    graph_path = next((output / "graphs").glob("*.csv"))
    graph_path.write_bytes(graph_path.read_bytes() + b"\n")
    with pytest.raises(DivisionDecodeError, match="SHA-256 mismatch"):
        verify_sealed_output(output)


def test_cli_dry_run_writes_nothing_and_has_no_scoring_surface(tmp_path: Path) -> None:
    input_path, input_hash, config_name = build_manifest_fixture(tmp_path)
    command = [
        sys.executable,
        "scripts/g03_division_decode.py",
        "--repo-root",
        str(REPO_ROOT),
        "--input-manifest",
        str(input_path),
        "--input-manifest-sha256",
        input_hash,
        "--config-name",
        config_name,
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "PASS_TARGET_BLIND_DRY_RUN_NO_WRITES"
    assert receipt["writes"] is False
    assert receipt["stage"] == "development_staging"
    assert receipt["score_eligible_after_seal"] is False
    assert receipt["duplicate_survivors"] == 0
    source = (REPO_ROOT / "scripts/g03_division_decode.py").read_text(encoding="utf-8")
    assert "official_eval" not in source
    assert "division_metrics" not in source
    assert "scoreall" not in source


def test_input_manifest_hash_drift_fails_before_any_bank_load(tmp_path: Path) -> None:
    input_path, _input_hash, config_name = build_manifest_fixture(tmp_path)
    with pytest.raises(DivisionDecodeError, match="SHA-256 mismatch"):
        run_from_manifest(
            input_manifest=input_path,
            input_manifest_sha256="f" * 64,
            config_name=config_name,
            repo_root=REPO_ROOT,
        )


def test_contaminated_audit_identity_is_refused_before_bank_load(tmp_path: Path) -> None:
    input_path, _input_hash, config_name = build_manifest_fixture(tmp_path)
    value = json.loads(input_path.read_text(encoding="ascii"))
    value["videos"][0]["identity"] = "train:audit_exposed00"
    input_hash = write_json(input_path, value)
    with pytest.raises(DivisionDecodeError, match="audit20 identity is forbidden"):
        run_from_manifest(
            input_manifest=input_path,
            input_manifest_sha256=input_hash,
            config_name=config_name,
            repo_root=REPO_ROOT,
        )


def test_modal_outer_tie_is_lexicographic_without_metric_repooling() -> None:
    assert select_modal_outer_winner(["z", "a", "z", "a", "q"]) == "a"
    assert select_modal_outer_winner(["z", "a", "z", "q", "z"]) == "z"

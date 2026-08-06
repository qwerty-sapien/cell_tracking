from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import synth_div_sim2real as model


def _tiny_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Columns are [t,z,y,x,track_id].  Node 2 divides into nodes 4 and 5.
    nodes = np.asarray(
        [
            [0, 0.0, 0.0, 0.0, 10],
            [0, 0.0, 12.0, 0.0, 20],
            [1, 0.0, 1.0, 0.0, 10],
            [1, 0.0, 13.0, 0.0, 20],
            [2, 0.0, 2.0, -1.5, 11],
            [2, 0.0, 2.0, 1.5, 12],
            [2, 0.0, 14.0, 0.0, 20],
            [2, 0.0, 7.0, 5.0, 30],
        ],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 2], [1, 3], [2, 4], [2, 5], [3, 6]], dtype=np.int32)
    divisions = np.asarray([2], dtype=np.int32)
    return nodes, edges, divisions


def _write_fixture(path: Path, *, poisonous_volume: bool = True) -> Path:
    nodes, edges, divisions = _tiny_arrays()
    volume = np.asarray([{"must_not_be_loaded": True}], dtype=object) if poisonous_volume else np.zeros(1)
    np.savez(
        path,
        volumes=volume,
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )
    return path


def _sequence_id_for_split(split: str) -> str:
    index = 0
    while True:
        candidate = f"{split}_fixture_{index}"
        if model.split_name(candidate) == split:
            return candidate
        index += 1


def _write_split_fixtures(tmp_path: Path) -> list[Path]:
    return [
        _write_fixture(tmp_path / f"{_sequence_id_for_split(split)}.npz")
        for split in ("train", "validation", "synthetic_test")
    ]


def _one_unreachable_arrays(
    division_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes: list[list[float]] = []
    edges: list[list[int]] = []
    divisions = np.arange(division_count, dtype=np.int32)
    for parent in range(division_count):
        nodes.append([0, 0.0, 0.0, float(parent * 100), float(parent)])
    nodes.append([0, 0.0, 0.0, 10000.0, 1000.0])  # nondivision event negative
    for parent in range(division_count):
        center = float(parent * 100)
        offsets = (20.0, 21.0) if parent == 0 else (-1.0, 1.0)
        for branch, offset in enumerate(offsets):
            child = len(nodes)
            nodes.append([1, 0.0, 0.0, center + offset, float(2000 + 2 * parent + branch)])
            edges.append([parent, child])
    # Exactly eight closer nodes make parent 0's true daughters unreachable.
    for offset in range(1, 9):
        nodes.append([1, 0.0, 0.0, float(offset), float(3000 + offset)])
    successor = len(nodes)
    nodes.append([1, 0.0, 0.0, 10001.0, 1000.0])
    edges.append([division_count, successor])
    return (
        np.asarray(nodes, dtype=np.float32),
        np.asarray(edges, dtype=np.int32),
        divisions,
    )


def _write_arrays(path: Path, arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Path:
    nodes, edges, divisions = arrays
    np.savez(
        path,
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )
    return path


def _zero_bundle(event_probability: float = 0.2) -> model.ModelBundle:
    pair = model.LogisticModel(
        mean=np.zeros(len(model.PAIR_FEATURE_NAMES)),
        scale=np.ones(len(model.PAIR_FEATURE_NAMES)),
        coef=np.zeros(len(model.PAIR_FEATURE_NAMES) + 1),
    )
    event_coef = np.zeros(len(model.EVENT_FEATURE_NAMES) + 1)
    event_coef[0] = np.log(event_probability / (1.0 - event_probability))
    event = model.LogisticModel(
        mean=np.zeros(len(model.EVENT_FEATURE_NAMES)),
        scale=np.ones(len(model.EVENT_FEATURE_NAMES)),
        coef=event_coef,
    )
    return model.ModelBundle(pair, event, {"fixture": True})


def test_label_loader_never_loads_poisonous_volume(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "seq_poison.npz")
    # Accessing this object member with allow_pickle=False proves it is poisonous.
    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="Object arrays"):
            _ = archive["volumes"]
    sequence = model.load_sequence(path)
    assert sequence.nodes.shape == (8, 5)
    assert sequence.divisions.tolist() == [2]


def test_sha256_split_is_deterministic_and_uses_frozen_residues() -> None:
    seen: dict[int, str] = {}
    index = 0
    while len(seen) < 10:
        sequence_id = f"fixture_{index}"
        independent = int(hashlib.sha256(sequence_id.encode()).hexdigest(), 16) % 10
        seen.setdefault(independent, sequence_id)
        assert model.split_residue(sequence_id) == independent
        index += 1
    assert all(model.split_name(seen[value]) == "train" for value in range(8))
    assert model.split_name(seen[8]) == "validation"
    assert model.split_name(seen[9]) == "synthetic_test"


def test_true_daughter_pair_and_frozen_prior_features_are_enumerated(tmp_path: Path) -> None:
    sequence = model.load_sequence(_write_fixture(tmp_path / "seq_pair.npz"))
    candidates = model.enumerate_parent_candidates(sequence, 2)
    assert candidates.true_pair == (4, 5)
    assert (4, 5) in candidates.pairs
    assert len(candidates.next_indices) <= 8
    feature = candidates.features[candidates.pairs.index((4, 5))]
    values = dict(zip(model.PAIR_FEATURE_NAMES, feature, strict=True))
    mean_step = 0.5 * (values["near_daughter_step_um"] + values["far_daughter_step_um"])
    assert values["mean_step_abs_residual_from_1p86_um"] == pytest.approx(
        abs(mean_step - 1.86)
    )
    assert values["sister_separation_abs_residual_from_7p24_um"] == pytest.approx(
        abs(values["sister_separation_um"] - 7.24)
    )
    assert "daughter_midpoint_residual_from_persistence_0p30_um" in values
    assert "lag1_directional_persistence_residual_from_0p30" in values


def test_physical_coordinate_view_is_cached_read_only_and_bit_exact(tmp_path: Path) -> None:
    sequence = model.load_sequence(_write_fixture(tmp_path / "seq_coord_cache.npz"))
    expected = sequence.coords_zyx * sequence.voxel_um_pooled
    first = sequence.physical_coords_zyx_um
    second = sequence.physical_coords_zyx_um
    assert first is second
    assert first.dtype == np.float64
    assert np.array_equal(first, expected)
    assert first.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        first[0, 0] = 1.0


def test_preregistered_prevalence_weights_lower_event_probability() -> None:
    features = np.empty((100, 0), dtype=np.float64)
    labels = np.asarray([1.0] * 10 + [0.0] * 90)
    unweighted = model.fit_logistic(features, labels, l2=1.0)
    weights = np.where(
        labels == 1.0,
        model.POSITIVE_IMPORTANCE_WEIGHT,
        model.NEGATIVE_IMPORTANCE_WEIGHT,
    )
    weighted = model.fit_logistic(features, labels, sample_weight=weights, l2=1.0)
    expected = (10 * model.POSITIVE_IMPORTANCE_WEIGHT) / (
        10 * model.POSITIVE_IMPORTANCE_WEIGHT + 90 * model.NEGATIVE_IMPORTANCE_WEIGHT
    )
    assert weighted.probabilities(features[:1])[0] == pytest.approx(expected, abs=1e-12)
    assert weighted.probabilities(features[:1])[0] < unweighted.probabilities(features[:1])[0]


def test_deterministic_training_model_round_trip_and_hash_validation(tmp_path: Path) -> None:
    paths = _write_split_fixtures(tmp_path)
    first = model.train_model(
        paths, maximum_iterations=40, runtime_cap_seconds=30
    )
    second = model.train_model(
        paths, maximum_iterations=40, runtime_cap_seconds=30
    )
    assert np.array_equal(first.pair.coef, second.pair.coef)
    assert np.array_equal(first.event.coef, second.event.coef)
    manifest_path, arrays_path = model.save_model(first, tmp_path / "division_model.json")
    loaded = model.load_model(manifest_path)
    assert np.array_equal(loaded.pair.mean, first.pair.mean)
    assert np.array_equal(loaded.pair.scale, first.pair.scale)
    assert np.array_equal(loaded.pair.coef, first.pair.coef)
    assert np.array_equal(loaded.event.coef, first.event.coef)
    assert loaded.arrays_sha256 == model.sha256_file(arrays_path)

    document = json.loads(manifest_path.read_text())
    document["calibration"]["additional_logit_shift"] = -2.790791860165793
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(model.ContractError, match="double-shift"):
        model.load_model(manifest_path)


def test_reachability_audit_and_pair_skip_preserve_all_event_labels(tmp_path: Path) -> None:
    arrays = _one_unreachable_arrays()
    paths = [
        _write_arrays(tmp_path / f"{_sequence_id_for_split(split)}.npz", arrays)
        for split in ("train", "validation", "synthetic_test")
    ]
    report = model.audit_reachability(paths, runtime_cap_seconds=30)
    assert report["amendment2_sha256"] == model.AMENDMENT_2_SHA256
    assert report["overall"]["total_division_parents"] == 60
    assert report["overall"]["reachable_division_parents"] == 57
    assert report["overall"]["unreachable_division_parents"] == 3
    for split in ("train", "validation", "synthetic_test"):
        assert report["by_split"][split]["coverage_rate"] == pytest.approx(0.95)
    model.require_reachability_gate(report)

    report_path = tmp_path / "coverage.json"
    model._write_json(report_path, report)
    bundle = model.train_model(
        paths,
        coverage_report_path=report_path,
        coverage_report_sha256=model.sha256_file(report_path),
        maximum_iterations=20,
        runtime_cap_seconds=30,
    )
    pair_coverage = bundle.metadata["population"]["pair_division_reachability"]
    assert pair_coverage == {
        "total_division_parents": 20,
        "reachable_division_parents": 19,
        "unreachable_division_parents": 1,
        "coverage_rate": 0.95,
    }
    # The unreachable positive was skipped only by the pair head, not the event head.
    assert bundle.metadata["population"]["event_positives"] == 20
    assert bundle.metadata["amendment2_sha256"] == model.AMENDMENT_2_SHA256
    assert bundle.metadata["reachability_report_sha256"] == model.sha256_file(report_path)
    with pytest.raises(model.ContractError, match="SHA-256 mismatch"):
        model.train_model(
            paths,
            coverage_report_path=report_path,
            coverage_report_sha256="0" * 64,
            maximum_iterations=1,
            runtime_cap_seconds=30,
        )


def test_reachability_gate_rejects_a_split_below_95_percent(tmp_path: Path) -> None:
    arrays = _one_unreachable_arrays(division_count=19)
    paths = [
        _write_arrays(tmp_path / f"{_sequence_id_for_split(split)}.npz", arrays)
        for split in ("train", "validation", "synthetic_test")
    ]
    report = model.audit_reachability(paths, runtime_cap_seconds=30)
    assert report["by_split"]["train"]["coverage_rate"] == pytest.approx(18 / 19)
    with pytest.raises(model.ContractError, match="reachability gate failed"):
        model.require_reachability_gate(report)
    with pytest.raises(model.ContractError, match="reachability gate failed"):
        model.train_model(paths, maximum_iterations=1, runtime_cap_seconds=30)


def test_production_training_fails_before_fit_without_all_three_split_coverages(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path / f"{_sequence_id_for_split('train')}.npz")
    with pytest.raises(model.ContractError, match="no labelled divisions in validation"):
        model.train_model([path], maximum_iterations=5, runtime_cap_seconds=30)


def test_graph_array_proposals_have_ids_features_and_calibrated_probabilities() -> None:
    nodes, edges, divisions = _tiny_arrays()
    rows = model.score_graph_arrays(
        sequence_id="array_fixture",
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        model=_zero_bundle(0.2),
    )
    assert rows
    by_parent: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_parent.setdefault(int(row["parent_index"]), []).append(row)
        assert 0.0 <= float(row["pair_probability"]) <= 1.0
        assert 0.0 <= float(row["event_probability"]) <= 1.0
        assert float(row["score"]) == pytest.approx(
            float(row["event_probability"]) * float(row["pair_probability"])
        )
        assert set(row["pair_features"]) == set(model.PAIR_FEATURE_NAMES)
        assert set(row["event_features"]) == set(model.EVENT_FEATURE_NAMES)
        assert isinstance(row["parent_track_id"], int)
        assert isinstance(row["daughter_a_track_id"], int)
        assert isinstance(row["daughter_b_track_id"], int)
    for parent_rows in by_parent.values():
        assert sum(float(row["pair_probability"]) for row in parent_rows) == pytest.approx(1.0)
        assert sum(float(row["score"]) for row in parent_rows) == pytest.approx(0.2)


def test_streaming_graph_array_api_matches_compatibility_list() -> None:
    nodes, edges, divisions = _tiny_arrays()
    bundle = _zero_bundle(0.2)
    iterator = model.iter_score_graph_arrays(
        sequence_id="stream_fixture",
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        model=bundle,
    )
    assert iter(iterator) is iterator
    streamed = list(iterator)
    materialized = model.score_graph_arrays(
        sequence_id="stream_fixture",
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        model=bundle,
    )
    assert streamed == materialized

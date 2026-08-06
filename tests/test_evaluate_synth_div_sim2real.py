from __future__ import annotations

from pathlib import Path
import json
import sys
import time
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.action_compiler import compile_action
from celltrack_alpha.kaggle_calibration import (
    read_submission,
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    validate_graphs,
    write_submission,
)
from scripts import evaluate_synth_div_sim2real as adapter
from scripts import synth_div_sim2real as model_api


def tiny_graph() -> SubmissionGraph:
    nodes = {
        1: SubmissionNode(1, 0, 0, 0, 0),
        2: SubmissionNode(2, 1, 0, 4, 4),
        5: SubmissionNode(5, 1, 0, 20, 20),
        3: SubmissionNode(3, 2, 0, 8, 0),
        4: SubmissionNode(4, 2, 0, 8, 8),
        6: SubmissionNode(6, 2, 0, 20, 16),
        7: SubmissionNode(7, 2, 0, 20, 24),
        10: SubmissionNode(10, 3, 0, 0, 0),
    }
    graph = SubmissionGraph(
        "44b6_tiny",
        nodes,
        [
            SubmissionEdge(1, 2),
            SubmissionEdge(2, 3),
            SubmissionEdge(5, 4),
            SubmissionEdge(5, 6),
            # Valid submission gap; intentionally not lag-1 model context.
            SubmissionEdge(1, 10),
        ],
    )
    validate_graphs({graph.dataset: graph})
    return graph


def candidate_arrays(rows: list[tuple[int, int, int, float]]) -> adapter.CandidateArrays:
    score = np.asarray([row[3] for row in rows], dtype=np.float64)
    return adapter.CandidateArrays(
        parent_id=np.asarray([row[0] for row in rows], dtype=np.int64),
        daughter_a_id=np.asarray([row[1] for row in rows], dtype=np.int64),
        daughter_b_id=np.asarray([row[2] for row in rows], dtype=np.int64),
        pair_rank=np.ones(len(rows), dtype=np.int16),
        event_probability=score.copy(),
        pair_probability=np.ones(len(rows), dtype=np.float64),
        score=score,
    )


def write_v2_fixture_model(path: Path) -> None:
    pair = model_api.LogisticModel(
        np.zeros(len(model_api.PAIR_FEATURE_NAMES)),
        np.ones(len(model_api.PAIR_FEATURE_NAMES)),
        np.zeros(len(model_api.PAIR_FEATURE_NAMES) + 1),
    )
    event = model_api.LogisticModel(
        np.zeros(len(model_api.EVENT_FEATURE_NAMES)),
        np.ones(len(model_api.EVENT_FEATURE_NAMES)),
        np.zeros(len(model_api.EVENT_FEATURE_NAMES) + 1),
    )
    training_metadata = {
        "fixture": True,
        "amendment2_sha256": model_api.AMENDMENT_2_SHA256,
        "reachability_report_sha256": "0" * 64,
    }
    model_api.save_model(model_api.ModelBundle(pair, event, training_metadata), path)


def write_registration_fixture(root: Path, datasets: list[str]) -> Path:
    census = root / "CENSUS.json"
    root.mkdir()
    census.write_text("{}\n", encoding="ascii")
    video_root = root / "videos"
    video_root.mkdir()
    transitions = [
        {
            "frame": frame,
            "pool_breaking": False,
            "shift_magnitude_um": 0.0,
            "shift_registered_zyx": [0, 0, 0],
        }
        for frame in range(99)
    ]
    for dataset in datasets:
        (video_root / f"train__{dataset}.json").write_text(
            json.dumps(
                {
                    "split": "train",
                    "dataset": dataset,
                    "transitions": transitions,
                }
            ),
            encoding="ascii",
        )
    return census


def test_native_to_pooled_scale_and_adjacent_context_only() -> None:
    nodes, edges, divisions, node_ids = adapter.graph_to_model_arrays(tiny_graph())
    row = node_ids.index(2)
    # Native y=x=4 at 0.40625 um/px becomes pooled y=x=1 at 1.625 um/px.
    assert nodes[row, 1:4].tolist() == [0.0, 1.0, 1.0]
    pairs = {(node_ids[int(source)], node_ids[int(target)]) for source, target in edges}
    assert (1, 10) not in pairs
    assert pairs == {(1, 2), (2, 3), (5, 4), (5, 6)}
    assert divisions.tolist() == [node_ids.index(5)]


def test_registered_geometry_changes_model_view_not_submission_graph() -> None:
    graph = tiny_graph()
    original_y = graph.nodes[2].y
    plan = adapter.RegistrationPlan(
        identity="train:44b6_tiny",
        cumulative_shift_raw_zyx=np.asarray(
            [[0, 0, 0], [0, -4, -4], [0, -4, -4], [0, -4, -4]],
            dtype=np.int64,
        ),
        pool_breaking_frames=(0,),
        census_sha256="b" * 64,
    )
    nodes, _edges, _divisions, node_ids = adapter.graph_to_model_arrays(
        graph, registration_plan=plan
    )
    row = node_ids.index(2)
    assert nodes[row, 1:4].tolist() == [0.0, 0.0, 0.0]
    assert graph.nodes[2].y == original_y


def test_stream_model_stub_yields_compiler_shaped_non_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = tiny_graph()
    observed: dict[str, object] = {}

    def fake_iter(**kwargs):
        observed.update(kwargs)
        yield {
            "parent_index": kwargs["nodes"][:, 4].astype(int).tolist().index(2),
            "daughter_a_index": kwargs["nodes"][:, 4].astype(int).tolist().index(3),
            "daughter_b_index": kwargs["nodes"][:, 4].astype(int).tolist().index(4),
            "pair_rank": 1,
            "event_probability": 0.2,
            "pair_probability": 0.5,
            "score": 0.1,
        }

    monkeypatch.setattr(adapter.sds, "iter_score_graph_arrays", fake_iter)
    arrays, counts = adapter._score_graph(graph, object())
    assert len(arrays) == 1
    assert arrays.row(0) == (2, 3, 4, 0.1)
    assert counts == {
        "model_rows": 1,
        "graph_distinct_candidates": 1,
        "graph_identical_noops_excluded": 0,
    }
    assert np.array_equal(observed["voxel_um_pooled"], np.asarray([1.625] * 3))

    action = adapter._action_for_ids(graph, 2, 3, 4)
    assert set(action.expected_conflict_edges) == {(2, 3), (5, 4)}
    compiled = compile_action(graph, action)
    assert {(edge.source_id, edge.target_id) for edge in compiled.graph.edges} >= {
        (2, 3),
        (2, 4),
    }
    assert (5, 4) not in {(edge.source_id, edge.target_id) for edge in compiled.graph.edges}


def test_greedy_owns_conflicts_beyond_node_boundary_and_applies_cap_threshold() -> None:
    graph = tiny_graph()
    arrays = candidate_arrays(
        [
            (2, 3, 4, 0.90),  # Owns (5,4) through daughter incoming replacement.
            (5, 6, 7, 0.80),  # Distinct nodes, but also owns (5,4): suppressed.
            (5, 3, 7, 0.70),  # Direct node overlap with first: suppressed.
        ]
    )
    adapter._validate_candidate_arrays(arrays, graph)
    greedy = adapter.greedy_selected_indices(graph, arrays)
    assert greedy.tolist() == [0]
    assert adapter.policy_indices(arrays, greedy, threshold=0.85, cap=None).tolist() == [0]
    assert adapter.policy_indices(arrays, greedy, threshold=0.95, cap=None).tolist() == []
    assert adapter.policy_indices(arrays, greedy, threshold=0.0, cap=1).tolist() == [0]


def test_exact_gt_labels_use_coordinate_match_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = tiny_graph()
    arrays = candidate_arrays([(2, 3, 4, 0.9), (5, 6, 7, 0.2)])
    monkeypatch.setattr(
        adapter,
        "_official_node_matching",
        lambda *_args, **_kwargs: {2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70},
    )
    gt = {
        "nodes": {"node_id": np.asarray([20, 30, 40, 50, 60, 70])},
        "edges": np.asarray([[20, 30], [20, 40], [50, 60]], dtype=np.int64),
    }
    labels, total, reachable = adapter.exact_gt_fork_labels(
        graph, arrays, gt, (1.625, 0.40625, 0.40625)
    )
    assert labels.tolist() == [True, False]
    assert (total, reachable) == (1, 1)


def test_ranking_metrics_ties_and_frontier_counts() -> None:
    metrics = adapter.binary_ranking_metrics(
        np.asarray([0.5, 0.5, 0.2]), np.asarray([1, 0, 0], dtype=np.uint8)
    )
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["pr_auc_average_precision"] == pytest.approx(0.5)
    assert adapter._counts_row(2, 3, 7) == {
        "tp": 2,
        "fp": 3,
        "fn": 5,
        "precision": 0.4,
        "recall": pytest.approx(2 / 7),
        "division_jaccard": 0.2,
    }
    scores = np.asarray([0.1, 0.5, 0.5, 0.5, 0.9])
    labels = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
    order = np.argsort(scores)
    groups = list(
        adapter._group_stream(scores, labels, order, descending=False, chunk_size=2)
    )
    assert [value for group in groups for value in group[1].tolist()] == [1, 3, 1]
    assert [value for group in groups for value in group[2].tolist()] == [0, 2, 1]


def test_generate_cli_has_no_gt_argument_and_evaluation_authenticates_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = adapter.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--model-json",
                "model.json",
                "--fragments",
                "fragments",
                "--output-dir",
                "scores",
                "--gt-archive",
                "forbidden.zip",
            ]
        )

    calls: list[str] = []

    def fail_auth(_path: Path, **_kwargs):
        calls.append("authenticate_scores")
        raise adapter.EvaluationError("tampered score bank")

    monkeypatch.setattr(adapter, "load_frozen_scores", fail_auth)
    with pytest.raises(adapter.EvaluationError, match="tampered score bank"):
        adapter.evaluate_scores(
            tmp_path / "scores.json", tmp_path / "gt.zip", tmp_path / "report.json"
        )
    assert calls == ["authenticate_scores"]
    assert not (tmp_path / "report.json").exists()


def test_monotonic_runtime_deadline_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((10.0, 10.5, 11.000001))
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(ticks))
    deadline = adapter.RuntimeDeadline.start(1.0)
    deadline.check("within cap")
    with pytest.raises(TimeoutError, match="runtime cap exceeded during timed fixture"):
        deadline.check("timed fixture")
    with pytest.raises(adapter.EvaluationError, match="finite positive"):
        adapter.RuntimeDeadline.start(0.0)


def test_end_to_end_generate_and_integrate_safe_fragment_copy(tmp_path: Path) -> None:
    graph = tiny_graph()
    fragment_dir = tmp_path / "fragments"
    fragment_dir.mkdir()
    fragment = fragment_dir / "train__44b6_tiny.csv"
    write_submission({graph.dataset: graph}, fragment)
    original_hash = adapter.sha256_file(fragment)

    model_path = tmp_path / "model.json"
    write_v2_fixture_model(model_path)
    registration_census = write_registration_fixture(
        tmp_path / "registration", ["44b6_tiny"]
    )

    scores_dir = tmp_path / "scores"
    score_manifest = adapter.generate_scores(
        model_path,
        fragment_dir,
        scores_dir,
        registration_census=registration_census,
    )
    assert score_manifest["contains_labels"] is False
    assert score_manifest["gt_archive_opened"] is False
    assert score_manifest["candidate_count"] > 0

    integrated_dir = tmp_path / "integrated"
    result = adapter.integrate_scores(
        scores_dir / "SCORES_MANIFEST.json",
        integrated_dir,
        policies=[(0.0, 1)],
    )
    policy = result["policies"][0]
    assert policy["selected_action_count"] == 1
    output_fragment = integrated_dir / policy["files"][0]["fragment"]
    assert set(read_submission(output_fragment)) == {graph.dataset}
    assert adapter.sha256_file(fragment) == original_hash
    assert output_fragment != fragment


def test_parallel_two_fragment_generation_is_byte_identical_to_serial(
    tmp_path: Path,
) -> None:
    fragment_dir = tmp_path / "parallel_fragments"
    fragment_dir.mkdir()
    first = tiny_graph()
    second = tiny_graph()
    second.dataset = "6bba_tiny"
    write_submission({first.dataset: first}, fragment_dir / "train__44b6_tiny.csv")
    write_submission({second.dataset: second}, fragment_dir / "train__6bba_tiny.csv")

    model_path = tmp_path / "parallel_model.json"
    write_v2_fixture_model(model_path)
    registration_census = write_registration_fixture(
        tmp_path / "parallel_registration", ["44b6_tiny", "6bba_tiny"]
    )
    serial_dir = tmp_path / "serial_scores"
    parallel_dir = tmp_path / "parallel_scores"
    serial = adapter.generate_scores(
        model_path,
        fragment_dir,
        serial_dir,
        registration_census=registration_census,
        workers=1,
    )
    # Some managed macOS sandboxes deny the informational semaphore-limit
    # sysconf call even though process creation itself is available. Bypass only
    # that platform probe; the real ProcessPoolExecutor and workers still run.
    with mock.patch(
        "concurrent.futures.process._check_system_limits", return_value=None
    ):
        parallel = adapter.generate_scores(
            model_path,
            fragment_dir,
            parallel_dir,
            registration_census=registration_census,
            workers=2,
        )

    assert [row["identity"] for row in serial["records"]] == [
        "train:44b6_tiny",
        "train:6bba_tiny",
    ]
    assert [row["candidate_sha256"] for row in serial["records"]] == [
        row["candidate_sha256"] for row in parallel["records"]
    ]
    assert [row["candidate_sha256"] for row in serial["records"]] == [
        "8530037c672550948616ece580424140954b5b7bf6ff320e9b36f324455a7545",
        "8530037c672550948616ece580424140954b5b7bf6ff320e9b36f324455a7545",
    ]
    assert [row["graph_distinct_candidates"] for row in serial["records"]] == [
        row["graph_distinct_candidates"] for row in parallel["records"]
    ]
    for serial_row, parallel_row in zip(serial["records"], parallel["records"], strict=True):
        serial_arrays = adapter._load_candidate_npz(
            serial_dir / serial_row["candidate_file"]
        )
        parallel_arrays = adapter._load_candidate_npz(
            parallel_dir / parallel_row["candidate_file"]
        )
        for key in adapter.CANDIDATE_KEYS:
            assert np.array_equal(
                getattr(serial_arrays, key), getattr(parallel_arrays, key)
            )

    serial_scientific = dict(serial)
    parallel_scientific = dict(parallel)
    serial_scientific.pop("execution")
    parallel_scientific.pop("execution")
    assert serial_scientific == parallel_scientific
    assert parallel["execution"] == {
        "workers": 2,
        "mode": "bounded_process_pool_per_video",
        "maximum_live_graphs": 2,
        "model_loaded_per_worker": True,
        "deterministic_manifest_order": "authenticated fragment inventory order",
        "execution_amendment_3_sha256": adapter.EXECUTION_AMENDMENT_3_SHA256,
        "execution_amendment_4_sha256": adapter.EXECUTION_AMENDMENT_4_SHA256,
    }


def test_parallel_failure_helper_terminates_running_workers() -> None:
    with mock.patch(
        "concurrent.futures.process._check_system_limits", return_value=None
    ):
        executor = adapter.ProcessPoolExecutor(max_workers=1)
    future = executor.submit(time.sleep, 30.0)
    try:
        deadline = time.monotonic() + 5.0
        while not getattr(executor, "_processes", None):
            if time.monotonic() > deadline:
                pytest.fail("fixture worker did not start")
            time.sleep(0.01)
        processes = tuple(executor._processes.values())
        adapter._terminate_executor_workers(executor, join_timeout_seconds=2.0)
        assert processes
        assert all(not process.is_alive() for process in processes)
    finally:
        future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

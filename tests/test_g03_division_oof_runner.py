from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ORCHESTRATOR = load_module(
    "g03_oof_orchestrator_runner_tests",
    ROOT / "scripts" / "g03_division_oof_orchestrator.py",
)
RUNNER = load_module(
    "g03_oof_runner_tests", ROOT / "scripts" / "g03_division_oof_runner.py"
)
CONFIG_MANIFEST = load_module(
    "g03_config_manifest_runner_tests",
    ROOT / "scripts" / "g03_division_decode_manifest.py",
)
DECODE_FIXTURES = load_module(
    "g03_decode_fixtures_runner_tests",
    ROOT / "tests" / "test_g03_division_decode.py",
)
# The production orchestrator is resealed by the root coordinator after these
# isolated runner bytes are final.  Tests bind its oracle to the exact WIP bytes.
ORCHESTRATOR.PROMOTED_RUNNER_SHA256 = RUNNER.sha256_file(
    Path(RUNNER.__file__).resolve()
)
REAL_API = ORCHESTRATOR.decode._compiler_api(ROOT)


@dataclass
class FakeGraph:
    dataset: str
    variant: str

    def copy(self) -> "FakeGraph":
        return FakeGraph(self.dataset, self.variant)


def graph_sha256(graph: FakeGraph) -> str:
    return hashlib.sha256(f"{graph.dataset}\0{graph.variant}".encode("ascii")).hexdigest()


def write_submission(graphs: Mapping[str, FakeGraph], path: Path) -> None:
    assert len(graphs) == 1
    graph = next(iter(graphs.values()))
    path.write_text(
        json.dumps({"dataset": graph.dataset, "variant": graph.variant}, sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def read_submission(path: Path) -> dict[str, FakeGraph]:
    value = json.loads(path.read_text(encoding="ascii"))
    graph = FakeGraph(str(value["dataset"]), str(value["variant"]))
    return {graph.dataset: graph}


API = {
    "graph_sha256": graph_sha256,
    "write_submission": write_submission,
    "read_submission": read_submission,
}


def disjoint_division_fixture(count: int) -> tuple[Any, tuple[Any, ...]]:
    from celltrack_alpha.kaggle_calibration import (
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
    )

    nodes: dict[int, Any] = {}
    edges = []
    actions = []
    for index in range(count):
        parent, alternate, child_a, child_b = (
            4 * index + 1,
            4 * index + 2,
            4 * index + 3,
            4 * index + 4,
        )
        nodes[parent] = SubmissionNode(parent, 0, index, 0, 0)
        nodes[alternate] = SubmissionNode(alternate, 0, index, 1, 0)
        nodes[child_a] = SubmissionNode(child_a, 1, index, 0, 1)
        nodes[child_b] = SubmissionNode(child_b, 1, index, 1, 1)
        edges.extend(
            (SubmissionEdge(parent, child_a), SubmissionEdge(alternate, child_b))
        )
        actions.append(
            REAL_API["GraphAction"](
                action_id=f"division.{index:05d}",
                kind=REAL_API["ActionKind"].DIVISION_FORK,
                add_edges=((parent, child_a), (parent, child_b)),
                replace_outgoing_from=(parent,),
                replace_incoming_to=(child_a, child_b),
                expected_conflict_edges=((parent, child_a), (alternate, child_b)),
            )
        )
    graph = SubmissionGraph("one_pass_fixture", nodes, edges)
    REAL_API["validate_graphs"]({graph.dataset: graph})
    return graph, tuple(actions)


def sequential_graph(baseline: Any, actions: Any) -> Any:
    current = baseline
    for action in sorted(actions, key=lambda value: value.action_id):
        current = REAL_API["compile_action"](current, action).graph
    return current


def test_one_pass_matches_sequential_for_every_subset_and_order() -> None:
    baseline, actions = disjoint_division_fixture(8)
    checked = 0
    for size in range(len(actions) + 1):
        for selected in itertools.combinations(actions, size):
            trusted = sequential_graph(baseline, selected)
            expected = REAL_API["graph_sha256"](trusted)
            for order in (selected, tuple(reversed(selected))):
                observed, receipt = ORCHESTRATOR.decode._materialize_division_actions_one_pass(
                    baseline=baseline,
                    actions=order,
                    api=REAL_API,
                )
                assert REAL_API["graph_sha256"](observed) == expected
                assert receipt["after_graph_sha256"] == expected
                assert receipt["action_count"] == len(selected)
                assert receipt["full_graph_constructions"] == 1
                assert receipt["full_graph_hashes"] == 1
                checked += 1
    assert checked == 512


def test_one_pass_rejects_stale_cross_category_and_source_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, actions = disjoint_division_fixture(2)
    stale = replace(actions[0], expected_conflict_edges=actions[0].expected_conflict_edges[:1])
    with pytest.raises(
        ORCHESTRATOR.decode.DivisionDecodeError, match="incident-conflict"
    ):
        ORCHESTRATOR.decode._materialize_division_actions_one_pass(
            baseline=baseline, actions=(stale,), api=REAL_API
        )

    overlap = replace(actions[0], action_id="division.overlap")
    with pytest.raises(
        ORCHESTRATOR.decode.DivisionDecodeError, match="ownership overlaps"
    ):
        ORCHESTRATOR.decode._materialize_division_actions_one_pass(
            baseline=baseline, actions=(actions[0], overlap), api=REAL_API
        )

    substituted = dict(REAL_API)
    substituted["graph_sha256"] = lambda _graph: "0" * 64
    with pytest.raises(
        ORCHESTRATOR.decode.DivisionDecodeError, match="API identity coherence"
    ):
        ORCHESTRATOR.decode._materialize_division_actions_one_pass(
            baseline=baseline, actions=(actions[0],), api=substituted
        )

    monkeypatch.setattr(
        ORCHESTRATOR.decode, "FROZEN_ACTION_COMPILER_SHA256", "0" * 64
    )
    with pytest.raises(
        ORCHESTRATOR.decode.DivisionDecodeError, match="source hashes"
    ):
        ORCHESTRATOR.decode._materialize_division_actions_one_pass(
            baseline=baseline, actions=(actions[0],), api=REAL_API
        )


def test_production_decode_spool_invokes_one_pass_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    bank, baseline, api = DECODE_FIXTURES.loaded_fixture(fixture_root)
    identity = bank.identity
    config = DECODE_FIXTURES.active_config(per_video_cap=1)
    video = RUNNER.VideoInput(
        identity=identity,
        fold="0",
        association_bank_path=tmp_path / "bank.npz",
        association_bank_sha256="b" * 64,
        parent_graph_path=tmp_path / "parent.csv",
        parent_graph_sha256="c" * 64,
        bank=bank,
        parent_graph=baseline,
    )
    dependency_hashes = {
        relative: RUNNER.sha256_file(ROOT / relative)
        for relative in RUNNER.ONE_PASS_DEPENDENCIES
    }
    context = RUNNER.CoreContext(
        plan={},
        plan_sha256="1" * 64,
        folds={"0": (identity,)},
        videos={identity: video},
        configs=(RUNNER.FixedConfig(name=config.name, value=config),),
        decoder=ORCHESTRATOR.decode,
        orchestrator=ORCHESTRATOR,
        api=api,
        scorer_source={"path": "unused", "sha256": "2" * 64},
        execution_sources={},
        source_hashes={},
        input_manifest_sha256="3" * 64,
        development_seal_sha256="4" * 64,
        graph_cache=RUNNER.GraphCache(
            tmp_path / "graph-cache",
            api=api,
            decoder_source_sha256=RUNNER.sha256_file(
                Path(ORCHESTRATOR.decode.__file__).resolve()
            ),
            one_pass_dependency_sha256s=dependency_hashes,
        ),
    )
    original = ORCHESTRATOR.decode._materialize_division_actions_one_pass
    calls = []

    def observed_materializer(**kwargs: Any) -> Any:
        calls.append(len(kwargs["actions"]))
        return original(**kwargs)

    monkeypatch.setattr(
        ORCHESTRATOR.decode,
        "_materialize_division_actions_one_pass",
        observed_materializer,
    )
    spool = RUNNER.MetricSpool(tmp_path)
    counts = RUNNER.RuntimeCounts()
    try:
        RUNNER._decode_into_spool(
            context=context,
            configs=context.configs,
            identities=(identity,),
            spool=spool,
            counts=counts,
        )
    finally:
        spool.close()
    assert calls == [1]
    assert counts.one_pass_graph_materializations == 1
    assert counts.full_graph_decodes == 1
    assert counts.scorer_invocations == 0


def test_configuration_name_digest_reproduces_production_pretarget_plan() -> None:
    configuration_path = (
        ROOT
        / "reports/g03/screening/wave_r/R4_DIVISION_DECODE_CONFIG_MANIFEST.json"
    )
    protocol_path = (
        ROOT / "reports/g03/screening/wave_r/R4_DIVISION_PROTOCOL_CONTRACT.json"
    )
    plan_path = ROOT / "reports/g03/screening/wave_r/R4_artifacts/oof/PRETARGET_PLAN.json"
    if not all(path.is_file() for path in (configuration_path, protocol_path, plan_path)):
        pytest.skip("production pretarget artifacts are not present in this isolated worktree")
    configuration = json.loads(configuration_path.read_text(encoding="ascii"))
    protocol = json.loads(protocol_path.read_text(encoding="ascii"))
    plan = json.loads(plan_path.read_text(encoding="ascii"))["payload"]
    names = ORCHESTRATOR._configuration_names(configuration, protocol)
    sealed = RUNNER.pretarget_configuration_names_sha256(
        ORCHESTRATOR.decode, names
    )
    assert sealed == plan["configuration_names_sha256"]
    assert sealed != RUNNER.canonical_sha256(list(names))


@dataclass(frozen=True)
class FakeConfig:
    name: str
    divisions: bool = True
    fork_threshold: float | None = 0.5
    max_primary_secondary_gap: float | None = 0.2
    late_window: int = 0
    daughter_persistence: int = 1
    grandchild_persistence: int = 1
    duplicate_temporal_window: int = 0
    per_video_cap: int | None = 1


@dataclass(frozen=True)
class FakeEvent:
    event_id: str
    emitted_parent: int
    children: tuple[int, int]


class FakeDecoder:
    def __init__(self, events: list[str], *, duplicates: int = 0) -> None:
        self.events = events
        self.duplicates = duplicates

    @staticmethod
    def _adjacency(_graph: FakeGraph) -> tuple[dict[Any, Any], dict[Any, Any], tuple[Any, ...]]:
        return {}, {}, ((10, 90), (91, 20), (92, 23))

    @staticmethod
    def _incident_conflicts(
        index: Any, parent: int, children: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        edges = tuple(index)
        return tuple(
            sorted(pair for pair in edges if pair[0] == parent or pair[1] in set(children))
        )

    @staticmethod
    def _build_incident_index(edges: Any) -> tuple[tuple[int, int], ...]:
        return tuple(edges)

    @staticmethod
    def _candidate_rank(event: FakeEvent, config_name: str) -> tuple[Any, ...]:
        return (event.emitted_parent, event.children, config_name, event.event_id)

    def generate_candidates(
        self, bank: Any, _parent: FakeGraph, config: FakeConfig
    ) -> tuple[tuple[FakeEvent, ...], Mapping[str, int]]:
        self.events.append(f"generate:{bank.identity}:{config.name}")
        if not config.divisions:
            generated: tuple[FakeEvent, ...] = ()
        else:
            first = FakeEvent(f"{config.name}.first", 10, (20, 21))
            second = FakeEvent(f"{config.name}.second", 11, (22, 23))
            generated = (first, second) if config.fork_threshold == 0.6 else (first,)
        return generated, {"raw_event_count": len(generated)}

    def suppress_candidates(
        self, events: Any, config: FakeConfig, *, incident_index: Any
    ) -> tuple[tuple[FakeEvent, ...], tuple[Any, ...], Mapping[str, int]]:
        assert incident_index is not None
        accepted = tuple(events)
        if config.per_video_cap is not None:
            accepted = accepted[: config.per_video_cap]
        return accepted, (), {
            "accepted_events": len(accepted),
            "duplicate_survivors": self.duplicates,
        }

    def decode_video(
        self,
        bank: Any,
        parent: FakeGraph,
        config: FakeConfig,
        _api: Mapping[str, Any],
    ) -> tuple[FakeGraph, dict[str, Any]]:
        self.events.append(f"decode:{bank.identity}:{config.name}")
        generated, _generation = self.generate_candidates(bank, parent, config)
        accepted, _rejected, suppression = self.suppress_candidates(
            generated,
            config,
            incident_index=self._build_incident_index(self._adjacency(parent)[2]),
        )
        variant = (
            "parent"
            if not config.divisions
            else "candidate-shared"
            if len(accepted) == 1
            else "candidate-c"
        )
        graph = FakeGraph(parent.dataset, variant)
        return graph, {
            "decoded_graph_sha256": graph_sha256(graph),
            "suppression": suppression,
            "accepted": [
                {
                    "action": {
                        "action_id": f"r4.{event.event_id}",
                        "kind": "division_fork",
                        "add_edges": [
                            [event.emitted_parent, child] for child in event.children
                        ],
                        "replace_outgoing_from": [event.emitted_parent],
                        "replace_incoming_to": list(event.children),
                        "expected_conflict_edges": [
                            list(pair)
                            for pair in self._incident_conflicts(
                                self._build_incident_index(
                                    self._adjacency(parent)[2]
                                ),
                                event.emitted_parent,
                                event.children,
                            )
                        ],
                    }
                }
                for event in accepted
            ],
        }


def sample_metric(graph: FakeGraph) -> dict[str, int | float]:
    if graph.variant == "parent":
        edge = (7, 1, 2)
        division = (0, 0, 2)
        adjusted = 0.78
        node = 0.89
    elif graph.variant == "candidate-shared":
        edge = (8, 1, 1)
        division = (1, 0, 1)
        adjusted = 0.79
        node = 0.90
    else:
        edge = (7, 2, 1)
        division = (1, 2, 1)
        adjusted = 0.75
        node = 0.88
    return {
        "edge_tp": edge[0],
        "edge_fp": edge[1],
        "edge_fn": edge[2],
        "division_tp": division[0],
        "division_fp": division[1],
        "division_fn": division[2],
        "num_pred_nodes": 10,
        "node_recall": node,
        "total_node_ratio": 1.0,
        "edge_jaccard": edge[0] / sum(edge),
        "adj_edge_jaccard": adjusted,
    }


class BackendState:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.opens = 0
        self.invocations: list[tuple[str, str]] = []


class FakeBackend:
    def __init__(self, state: BackendState) -> None:
        self.state = state

    def __enter__(self) -> "FakeBackend":
        self.state.opens += 1
        self.state.events.append("backend-enter")
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def score(
        self, identity: str, graph: FakeGraph, source: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert source["semantic_graph_sha256"] == graph_sha256(graph)
        self.state.invocations.append((identity, graph.variant))
        return sample_metric(graph)

    def summarise(self, rows: list[dict[str, Any]]) -> Mapping[str, Any]:
        edge = {
            key: sum(int(row[f"edge_{key}"]) for row in rows)
            for key in ("tp", "fp", "fn")
        }
        division = {
            key: sum(int(row[f"division_{key}"]) for row in rows)
            for key in ("tp", "fp", "fn")
        }
        edge_total = sum(edge.values())
        division_total = sum(division.values())
        edge_jaccard = edge["tp"] / edge_total if edge_total else 1.0
        division_jaccard = division["tp"] / division_total if division_total else 1.0
        adjusted = sum(float(row["adj_edge_jaccard"]) for row in rows) / len(rows)
        node = sum(float(row["node_recall"]) for row in rows) / len(rows)
        return {
            "score": adjusted + 0.1 * division_jaccard,
            "node_recall": node,
            "edge_jaccard": edge_jaccard,
            "adj_edge_jaccard": adjusted,
            "division_jaccard": division_jaccard,
        }


def make_plan(folds: dict[str, tuple[str, ...]], configuration_count: int = 3) -> dict[str, Any]:
    rows = []
    for outer in RUNNER.FOLD_IDS:
        inner = [fold for fold in RUNNER.FOLD_IDS if fold != outer]
        rows.append(
            {
                "outer_fold": outer,
                "outer_identities": list(folds[outer]),
                "inner_folds": inner,
                "inner_identities": [identity for fold in inner for identity in folds[fold]],
                "inner_fold_identities": {fold: list(folds[fold]) for fold in inner},
            }
        )
    runner_path = (ROOT / "scripts" / "g03_division_oof_runner.py").resolve()
    return {
        "configuration_count": configuration_count,
        "outer_folds": rows,
        "execution_source_bindings": {
            "runner": {
                "path": str(runner_path),
                "sha256": RUNNER.sha256_file(runner_path),
            }
        },
        "inner_tie_order": ["synthetic-frozen-order"],
        "deployable_rule": "modal then lex",
    }


def make_context(
    tmp_path: Path,
    *,
    events: list[str] | None = None,
    duplicates: int = 0,
    configuration_count: int = 3,
) -> tuple[Any, BackendState, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    event_log = [] if events is None else events
    folds: dict[str, tuple[str, ...]] = {}
    videos = {}
    for fold_index, fold in enumerate(RUNNER.FOLD_IDS):
        identities = tuple(
            f"train:6bba_{fold_index * 2 + offset:08x}" for offset in range(2)
        )
        folds[fold] = identities
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            videos[identity] = RUNNER.VideoInput(
                identity=identity,
                fold=fold,
                association_bank_path=tmp_path / "unused-bank.npz",
                association_bank_sha256=hashlib.sha256(f"bank:{identity}".encode()).hexdigest(),
                parent_graph_path=tmp_path / "unused-parent.csv",
                parent_graph_sha256=hashlib.sha256(f"parent-file:{identity}".encode()).hexdigest(),
                bank=SimpleNamespace(identity=identity),
                parent_graph=FakeGraph(dataset, "parent"),
            )
    scorer_path = tmp_path / "synthetic_scorer.py"
    scorer_path.write_text("# target-free synthetic scorer\n", encoding="ascii")
    scorer_sha = RUNNER.sha256_file(scorer_path)
    scorer_ref = {"path": str(scorer_path), "sha256": scorer_sha}
    runner_path = (ROOT / "scripts" / "g03_division_oof_runner.py").resolve()
    runner_ref = {
        "path": str(runner_path),
        "sha256": RUNNER.sha256_file(runner_path),
    }
    decoder_path = Path(ORCHESTRATOR.decode.__file__).resolve()
    orchestrator_path = (ROOT / "scripts" / "g03_division_oof_orchestrator.py").resolve()
    execution_sources = {
        "runner": runner_ref,
        "decoder": {
            "path": str(decoder_path),
            "sha256": RUNNER.sha256_file(decoder_path),
        },
        "orchestrator": {
            "path": str(orchestrator_path),
            "sha256": RUNNER.sha256_file(orchestrator_path),
        },
        "scoreall": dict(scorer_ref),
        "scoreall_dependencies": {
            relative: dict(scorer_ref)
            for relative in ORCHESTRATOR.SCORER_DEPENDENCY_PATHS
        },
        "one_pass_dependencies": {
            relative: {
                "path": str((ROOT / relative).resolve()),
                "sha256": RUNNER.sha256_file(ROOT / relative),
            }
            for relative in ORCHESTRATOR.ONE_PASS_DEPENDENCY_PATHS
        },
    }
    decoder = FakeDecoder(event_log, duplicates=duplicates)
    cache = RUNNER.GraphCache(
        tmp_path / "graph-cache",
        api=API,
        decoder_source_sha256="d" * 64,
    )
    context = RUNNER.CoreContext(
        plan=make_plan(folds, configuration_count),
        plan_sha256="1" * 64,
        folds=folds,
        videos=videos,
        configs=tuple(
            RUNNER.FixedConfig(name=config.name, value=config)
            for config in (
                FakeConfig("cfg-a", duplicate_temporal_window=0),
                FakeConfig("cfg-b", duplicate_temporal_window=1),
                FakeConfig("cfg-c", fork_threshold=0.6, per_video_cap=2),
            )
        ),
        decoder=decoder,
        orchestrator=ORCHESTRATOR,
        api=API,
        scorer_source=scorer_ref,
        execution_sources=execution_sources,
        source_hashes={
            "runner": runner_ref["sha256"],
            "decoder": "d" * 64,
            "orchestrator": "e" * 64,
            "scoreall": scorer_sha,
            "one_pass_dependencies": {
                relative: RUNNER.sha256_file(ROOT / relative)
                for relative in RUNNER.ONE_PASS_DEPENDENCIES
            },
        },
        input_manifest_sha256="2" * 64,
        development_seal_sha256="3" * 64,
        graph_cache=cache,
    )
    state = BackendState(event_log)
    return context, state, lambda: FakeBackend(state)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def test_full_18145_lattice_matches_naive_decode_on_two_videos(tmp_path: Path) -> None:
    manifest_configs = CONFIG_MANIFEST.build_configurations()
    fixed = tuple(
        RUNNER.FixedConfig(name=config.name, value=config)
        for config in manifest_configs
    )
    keys = [RUNNER._candidate_generation_key(config.value) for config in fixed]
    active_runs = 1 + sum(
        left != right for left, right in zip(keys[1:], keys[2:], strict=False)
    )
    assert len(fixed) == 18_145
    assert keys[0] == (False,)
    assert active_runs == len(set(keys[1:])) == 1_008
    assert all(keys[index] == keys[index + 17] for index in range(1, 18_145, 18))

    context, state, _factory = make_context(
        tmp_path,
        events=[],
        configuration_count=len(fixed),
    )
    context = replace(context, configs=fixed)
    identities = tuple(context.folds["0"])
    naive: dict[tuple[int, str], tuple[str, int, int]] = {}
    for config_index, config in enumerate(fixed):
        for identity in identities:
            video = context.videos[identity]
            _graph, record = context.decoder.decode_video(
                video.bank, video.parent_graph, config.value, context.api
            )
            naive[(config_index, identity)] = (
                record["decoded_graph_sha256"],
                int(record["suppression"]["accepted_events"]),
                int(record["suppression"]["duplicate_survivors"]),
            )

    spool = RUNNER.MetricSpool(tmp_path)
    counts = RUNNER.RuntimeCounts()
    try:
        RUNNER._decode_into_spool(
            context=context,
            configs=fixed,
            identities=identities,
            spool=spool,
            counts=counts,
        )
        observed = {
            (int(config_index), str(identity)): (
                str(graph_sha), int(forks), int(duplicates)
            )
            for config_index, identity, graph_sha, forks, duplicates in spool.connection.execute(
                "SELECT config_index,identity,graph_sha256,predicted_forks,"
                "duplicate_survivors FROM config_graph"
            )
        }
    finally:
        spool.close()
    assert observed == naive
    assert counts.decoded_graph_requests == 2 * 18_145
    assert counts.candidate_generation_requests == 2 * 1_008
    assert counts.candidate_generation_reuses == 2 * (18_144 - 1_008)
    assert counts.full_graph_decodes == 4
    assert counts.action_set_reuses == 2 * (18_145 - 2)
    assert counts.inert_action_set_requests == 2
    assert state.opens == 0
    assert not list(tmp_path.glob(".g03-r4-oof-spool-*"))


def winner_artifact(context: Any, outer_fold: str = "0", config: str = "cfg-a") -> dict[str, Any]:
    return ORCHESTRATOR._artifact(
        ORCHESTRATOR.INNER_WINNER_SCHEMA,
        "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
        {
            "pretarget_plan_sha256": context.plan_sha256,
            "inner_score_receipt_sha256": "4" * 64,
            "outer_fold": outer_fold,
            "outer_identities": list(context.folds[outer_fold]),
            "configuration_name": config,
            "pooled_inner_oof": ORCHESTRATOR.validate_metric_bundle(
                {
                    "combined_graph_score": 0.84,
                    "node_recall": 0.9,
                    "predicted_forks": 8,
                    "edge": {
                        "tp": 8,
                        "fp": 1,
                        "fn": 1,
                        "jaccard": 0.8,
                        "adjusted_jaccard": 0.79,
                    },
                    "division": {"tp": 1, "fp": 0, "fn": 1, "jaccard": 0.5},
                },
                "synthetic winner",
            ),
            "selection_order": list(context.plan["inner_tie_order"]),
            "outer_evaluation_authorized_for_winner_only": True,
            "outer_metric_read_before_winner_seal": False,
        },
    )


def test_inner_scores_all_configs_after_decode_with_parent_and_graph_dedup(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    context, state, factory = make_context(tmp_path, events=events)
    output = tmp_path / "inner-output"
    RUNNER.run_inner_core(
        context=context,
        outer_fold="0",
        backend_factory=factory,
        output=output,
        work_root=tmp_path,
    )
    receipt = read_json(output / "INNER_SCORE_RECEIPT.json")
    diagnostics = read_json(output / "RUNNER_DIAGNOSTICS.json")
    assert set(receipt) == {
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
    }
    assert receipt["inner_folds"] == ["1", "2", "3", "4"]
    assert len(receipt["evaluated_identities"]) == 8
    assert len(receipt["rows"]) == 3
    assert set(receipt["parent_folds"]) == {"1", "2", "3", "4"}
    assert all(
        parent["reference_label"] == ORCHESTRATOR.SAME_POPULATION_REFERENCE_LABEL
        and parent["gt_division_count"] == 4
        for parent in receipt["parent_folds"].values()
    )
    assert all(
        set(row) == {"configuration_name", "fold_evaluations", "pooled_inner_oof"}
        for row in receipt["rows"]
    )
    winner, _ = ORCHESTRATOR.select_inner_winner(
        plan=context.plan,
        receipt=receipt,
        configuration_names=("cfg-a", "cfg-b", "cfg-c"),
    )
    assert winner == "cfg-a"
    assert diagnostics["counts"] == {
        "decoded_graph_requests": 24,
        "candidate_generation_requests": 16,
            "candidate_generation_reuses": 8,
            "full_graph_decodes": 16,
            "one_pass_graph_materializations": 0,
            "action_set_reuses": 8,
        "avoided_full_graph_decodes": 8,
        "inert_action_set_requests": 0,
        "parent_graph_requests": 8,
        "unique_identity_graphs": 24,
        "graph_cache_hits": 0,
        "graph_cache_misses": 24,
        "scorer_invocations": 24,
    }
    assert len(state.invocations) == 24
    assert all(identity not in context.folds["0"] for identity, _variant in state.invocations)
    backend_at = events.index("backend-enter")
    assert sum(event.startswith("decode:") for event in events[:backend_at]) == 16
    assert sum(event.startswith("generate:") for event in events[:backend_at]) == 32
    assert all(
        event.startswith(("decode:", "generate:"))
        for event in events[:backend_at]
    )
    assert not list(tmp_path.glob(".g03-r4-oof-spool-*"))


def test_graph_cache_resumes_but_metric_scoring_does_not(tmp_path: Path) -> None:
    context, first_state, first_factory = make_context(tmp_path)
    RUNNER.run_inner_core(
        context=context,
        outer_fold="0",
        backend_factory=first_factory,
        output=tmp_path / "first",
        work_root=tmp_path,
    )
    second_state = BackendState([])
    RUNNER.run_inner_core(
        context=context,
        outer_fold="0",
        backend_factory=lambda: FakeBackend(second_state),
        output=tmp_path / "second",
        work_root=tmp_path,
    )
    first = read_json(tmp_path / "first" / "RUNNER_DIAGNOSTICS.json")
    second = read_json(tmp_path / "second" / "RUNNER_DIAGNOSTICS.json")
    assert first["counts"]["graph_cache_misses"] == 24
    assert second["counts"]["graph_cache_misses"] == 0
    assert second["counts"]["graph_cache_hits"] == 24
    assert second["counts"]["scorer_invocations"] == 24
    assert len(first_state.invocations) == len(second_state.invocations) == 24
    assert not list(tmp_path.glob(".g03-r4-oof-spool-*"))


def test_inner_refuses_config_drift_and_duplicate_survivor_before_target_open(
    tmp_path: Path,
) -> None:
    context, state, factory = make_context(tmp_path / "count", configuration_count=4)
    with pytest.raises(RUNNER.OOFRunnerError, match="every fixed configuration"):
        RUNNER.run_inner_core(
            context=context,
            outer_fold="0",
            backend_factory=factory,
            output=tmp_path / "count-output",
            work_root=tmp_path,
        )
    assert state.opens == 0
    context, state, factory = make_context(tmp_path / "duplicate", duplicates=1)
    with pytest.raises(RUNNER.OOFRunnerError, match="duplicate fork survivor"):
        RUNNER.run_inner_core(
            context=context,
            outer_fold="0",
            backend_factory=factory,
            output=tmp_path / "duplicate-output",
            work_root=tmp_path,
        )
    assert state.opens == 0
    assert not (tmp_path / "duplicate-output").exists()


def test_action_equivalence_drift_fails_before_scorer_or_graph_publication(
    tmp_path: Path,
) -> None:
    context, state, factory = make_context(tmp_path, events=[])

    class DriftDecoder(FakeDecoder):
        def decode_video(
            self,
            bank: Any,
            parent: FakeGraph,
            config: FakeConfig,
            api: Mapping[str, Any],
        ) -> tuple[FakeGraph, dict[str, Any]]:
            graph, record = super().decode_video(bank, parent, config, api)
            record = copy.deepcopy(record)
            record["accepted"][0]["action"]["add_edges"][0][1] += 1000
            return graph, record

    context = replace(context, decoder=DriftDecoder(state.events))
    output = tmp_path / "drift-output"
    with pytest.raises(
        RUNNER.OOFRunnerError,
        match="full decoder differs from its exact action-set plan",
    ):
        RUNNER.run_inner_core(
            context=context,
            outer_fold="0",
            backend_factory=factory,
            output=output,
            work_root=tmp_path,
        )
    assert state.opens == 0
    assert not output.exists()
    assert not any(context.graph_cache.root.rglob("CACHE_ENTRY.json"))


def test_outer_requires_presealed_winner_then_scores_candidate_and_parent_once(
    tmp_path: Path,
) -> None:
    context, state, factory = make_context(tmp_path)
    bad = winner_artifact(context, config="cfg-a")
    bad["payload"]["outer_fold"] = "1"
    bad["payload_sha256"] = ORCHESTRATOR.decode.canonical_sha256(bad["payload"])
    with pytest.raises(RUNNER.OOFRunnerError, match="winner outer fold drifted"):
        RUNNER.run_outer_core(
            context=context,
            outer_fold="0",
            winner_artifact=bad,
            winner_sha256="5" * 64,
            backend_factory=factory,
            output=tmp_path / "bad-outer",
            work_root=tmp_path,
        )
    assert state.opens == 0
    assert not any(event.startswith("decode:") for event in state.events)

    good = winner_artifact(context)
    output = tmp_path / "outer"
    RUNNER.run_outer_core(
        context=context,
        outer_fold="0",
        winner_artifact=good,
        winner_sha256="5" * 64,
        backend_factory=factory,
        output=output,
        work_root=tmp_path,
    )
    receipt = read_json(output / "OUTER_SCORE_RECEIPT.json")
    diagnostics = read_json(output / "RUNNER_DIAGNOSTICS.json")
    assert receipt["configuration_name"] == "cfg-a"
    assert receipt["evaluated_identities"] == list(context.folds["0"])
    assert receipt["gt_division_count"] == 4
    assert receipt["parent"]["reference_label"] == ORCHESTRATOR.SAME_POPULATION_REFERENCE_LABEL
    assert set(receipt["candidate"]["graph_sha256_by_identity"]) == set(context.folds["0"])
    assert set(receipt["parent"]["graph_sha256_by_identity"]) == set(context.folds["0"])
    assert diagnostics["counts"]["scorer_invocations"] == 4
    assert diagnostics["counts"]["unique_identity_graphs"] == 4
    result = ORCHESTRATOR.make_outer_result_artifact(
        plan=context.plan,
        plan_sha256=context.plan_sha256,
        winner_artifact=good,
        winner_sha256="5" * 64,
        receipt=receipt,
        receipt_sha256="6" * 64,
        receipt_base=tmp_path,
    )
    assert result["payload"]["deltas"]["combined_graph_score"] > 0


def test_runner_has_no_held_audit_hidden_or_gpu_command() -> None:
    parser = RUNNER.parser()
    actions = [action for action in parser._actions if hasattr(action, "choices")]
    commands = next(action.choices for action in actions if action.choices)
    assert set(commands) == {"inner", "outer"}
    with pytest.raises(SystemExit):
        parser.parse_args(["held"])


def test_nonfinite_aggregate_fails_without_mutation_or_publication(tmp_path: Path) -> None:
    row = sample_metric(FakeGraph("6bba_synthetic", "candidate-shared"))
    raw_nan = dict(row)
    raw_nan["edge_tp"] = math.nan
    with pytest.raises(RUNNER.OOFRunnerError, match="NONFINITE_SCORER_METRIC"):
        RUNNER.metric_bundle(
            rows=[raw_nan],
            predicted_forks=1,
            summarise=lambda _rows: {},
            orchestrator=ORCHESTRATOR,
            label="synthetic raw NaN",
        )
    assert math.isnan(raw_nan["edge_tp"])
    summary = FakeBackend(BackendState([])).summarise([dict(row)])
    summary = dict(summary)
    summary["division_jaccard"] = math.nan
    with pytest.raises(RUNNER.OOFRunnerError, match="NONFINITE_AGGREGATE_METRIC"):
        RUNNER.metric_bundle(
            rows=[row],
            predicted_forks=1,
            summarise=lambda _rows: summary,
            orchestrator=ORCHESTRATOR,
            label="synthetic NaN",
        )
    assert math.isnan(summary["division_jaccard"])
    with pytest.raises(RUNNER.OOFRunnerError, match="NONFINITE_JSON_DIAGNOSTIC"):
        RUNNER.atomic_output_bundle(
            tmp_path / "nan-output",
            receipt_name="RECEIPT.json",
            receipt={"metric": math.nan},
            diagnostics={"ok": True},
        )
    assert not (tmp_path / "nan-output").exists()


def test_atomic_outputs_refuse_overwrite_and_cache_tamper(tmp_path: Path) -> None:
    output = tmp_path / "sealed"
    RUNNER.atomic_output_bundle(
        output,
        receipt_name="RECEIPT.json",
        receipt={"metric": 1.0},
        diagnostics={"ok": True},
    )
    original = (output / "RECEIPT.json").read_bytes()
    with pytest.raises(RUNNER.OOFRunnerError, match="overwrite"):
        RUNNER.atomic_output_bundle(
            output,
            receipt_name="RECEIPT.json",
            receipt={"metric": 0.0},
            diagnostics={"ok": False},
        )
    assert (output / "RECEIPT.json").read_bytes() == original

    context, _state, _factory = make_context(tmp_path / "cache")
    identity = context.folds["0"][0]
    video = context.videos[identity]
    graph = FakeGraph(identity.split(":", 1)[1], "candidate-shared")
    graph_hash = graph_sha256(graph)
    _path, reused = context.graph_cache.publish(
        video=video, graph=graph, graph_sha256=graph_hash
    )
    assert reused is False
    receipt_path = (
        context.graph_cache.root
        / context.graph_cache._slug(identity)
        / graph_hash
        / "CACHE_ENTRY.json"
    )
    cache_receipt = read_json(receipt_path)
    cache_receipt["parent_graph_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(cache_receipt) + "\n", encoding="ascii")
    fresh_cache = RUNNER.GraphCache(
        context.graph_cache.root,
        api=API,
        decoder_source_sha256="d" * 64,
    )
    with pytest.raises(RUNNER.OOFRunnerError, match="binding drifted"):
        fresh_cache.validate(video, graph_hash)


def test_verified_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Value:\n    number: int\n",
        encoding="ascii",
    )
    with pytest.raises(RUNNER.OOFRunnerError, match="SHA-256 mismatch"):
        RUNNER.import_verified_module(source, "0" * 64, "synthetic source")
    loaded = RUNNER.import_verified_module(
        source, RUNNER.sha256_file(source), "synthetic source"
    )
    assert loaded.Value(3).number == 3

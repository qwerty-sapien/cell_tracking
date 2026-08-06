#!/usr/bin/env python3
"""Truth-blind real-graph adapter for ``SYNTH-DIV-SIM2REAL-v2-REACHABLE``.

The command-line interface deliberately separates proposal generation from
label evaluation:

``generate``
    Reads only a frozen model and prediction fragments.  It converts native
    anisotropic coordinates to the simulator's pooled-isotropic coordinate
    convention, scores every parent/pair proposal, and seals compact NPZ files.

``evaluate``
    First authenticates the complete sealed score set.  Only then does it open
    the GT archive and label proposals through the official 7-um node matcher.

``integrate``
    Reads only sealed scores and baseline fragments.  It greedily selects
    node-disjoint actions, compiles exact incident-conflict ownership, and
    emits safe fragment copies plus a scorer-ready manifest.

No GT value, scorer result, texture, contrast, or intensity enters proposal
generation, model scoring, or graph selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _location in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_location) not in sys.path:
        sys.path.insert(0, str(_location))

try:  # Importable both as ``scripts.evaluate_...`` and as a direct script.
    from scripts import synth_div_sim2real as sds
except ImportError:  # pragma: no cover - direct-script fallback
    import synth_div_sim2real as sds  # type: ignore[no-redef]

from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    action_sha256,
    compile_action,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    read_submission,
    validate_graphs,
    write_submission,
)
from celltrack_alpha.frame_registration import (  # noqa: E402
    RegistrationPlan,
    coordinate_views,
    load_registration_plan,
    sha256_file as registration_sha256_file,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


NODE = sds.NODE
SCORE_SCHEMA = "celltrack-alpha.synth-div-sim2real-real-scores.v2"
EVALUATION_SCHEMA = "celltrack-alpha.synth-div-sim2real-real-evaluation.v2"
INTEGRATION_SCHEMA = "celltrack-alpha.synth-div-sim2real-integration.v2"
NATIVE_SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
POOLED_SCALE_ZYX_UM = np.asarray((1.625, 1.625, 1.625), dtype=np.float64)
MATCH_RADIUS_UM = 7.0
MAX_NEXT_FRAME_CANDIDATES = 8
DEFAULT_RUNTIME_CAP_SECONDS = 14_400.0
DEFAULT_REGISTRATION_CENSUS = (
    ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
)
EXECUTION_AMENDMENT_3 = (
    ROOT / "reports/g04/lanes/synth_div_sim2real_v1/EXECUTION_AMENDMENT_3.json"
)
EXECUTION_AMENDMENT_3_SHA256 = (
    "3e82d991f87f1e41f73b4a7e1109d2b0dfa05574ebf5120778d676a41bb1a7f8"
)
EXECUTION_AMENDMENT_4 = (
    ROOT / "reports/g04/lanes/synth_div_sim2real_v1/EXECUTION_AMENDMENT_4.json"
)
EXECUTION_AMENDMENT_4_SHA256 = (
    "5f87a28432f86805dc83a832bb08bc0e8d9ba7cf4f2455b5f5ffd59e2319dbf3"
)
MAX_GENERATION_WORKERS = 4
FRAGMENT_RE = re.compile(r"^(train|test)__([A-Za-z0-9][A-Za-z0-9_.-]*)\.csv$")
CANDIDATE_KEYS = (
    "parent_id",
    "daughter_a_id",
    "daughter_b_id",
    "pair_rank",
    "event_probability",
    "pair_probability",
    "score",
)
DEFAULT_THRESHOLDS = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.7,
    0.9,
)
DEFAULT_CAPS: tuple[int | None, ...] = (1, 2, 3, 5, 8, None)


class EvaluationError(RuntimeError):
    """A frozen input or truth-firewall contract was violated."""


@dataclass(frozen=True)
class RuntimeDeadline:
    started: float
    cap_seconds: float

    @classmethod
    def start(cls, cap_seconds: float) -> "RuntimeDeadline":
        if not math.isfinite(cap_seconds) or cap_seconds <= 0.0:
            raise EvaluationError("runtime cap must be a finite positive number of seconds")
        return cls(time.monotonic(), float(cap_seconds))

    def check(self, context: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.cap_seconds:
            raise TimeoutError(
                f"runtime cap exceeded during {context}: "
                f"{elapsed:.3f}s > {self.cap_seconds:.3f}s"
            )

    def remaining(self, context: str) -> float:
        self.check(context)
        return max(0.0, self.cap_seconds - (time.monotonic() - self.started))


@dataclass(frozen=True)
class FragmentRecord:
    split: str
    dataset: str
    identity: str
    path: Path
    graph: SubmissionGraph


@dataclass(frozen=True)
class FragmentInventory:
    split: str
    dataset: str
    identity: str
    path: Path
    fragment_sha256: str
    baseline_graph_sha256: str


@dataclass(frozen=True)
class GenerationTask:
    inventory: FragmentInventory
    candidate_path: Path


_GENERATION_WORKER_MODEL: sds.ModelBundle | None = None
_GENERATION_WORKER_CENSUS: Path | None = None
_GENERATION_WORKER_DEADLINE: RuntimeDeadline | None = None


@dataclass(frozen=True)
class CandidateArrays:
    parent_id: np.ndarray
    daughter_a_id: np.ndarray
    daughter_b_id: np.ndarray
    pair_rank: np.ndarray
    event_probability: np.ndarray
    pair_probability: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return int(self.score.size)

    def row(self, index: int) -> tuple[int, int, int, float]:
        return (
            int(self.parent_id[index]),
            int(self.daughter_a_id[index]),
            int(self.daughter_b_id[index]),
            float(self.score[index]),
        )


@dataclass(frozen=True)
class GraphEdgeIndex:
    edges: frozenset[tuple[int, int]]
    outgoing: Mapping[int, frozenset[tuple[int, int]]]
    incoming: Mapping[int, frozenset[tuple[int, int]]]


def sha256_file(
    path: Path,
    *,
    deadline: RuntimeDeadline | None = None,
    context: str = "file hashing",
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block_index, block in enumerate(iter(lambda: handle.read(1 << 20), b"")):
            if deadline is not None and block_index % 64 == 0:
                deadline.check(context)
            digest.update(block)
    if deadline is not None:
        deadline.check(context)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
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


def _write_new_json(path: Path, value: Any) -> str:
    if path.exists():
        raise EvaluationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _write_new_npz(path: Path, arrays: CandidateArrays) -> str:
    if path.exists():
        raise EvaluationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                parent_id=arrays.parent_id,
                daughter_a_id=arrays.daughter_a_id,
                daughter_b_id=arrays.daughter_b_id,
                pair_rank=arrays.pair_rank,
                event_probability=arrays.event_probability,
                pair_probability=arrays.pair_probability,
                score=arrays.score,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _write_new_submission(path: Path, graph: SubmissionGraph) -> str:
    if path.exists():
        raise EvaluationError(f"refusing to overwrite fragment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_submission({graph.dataset: graph}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _candidate_key(dataset: str, parent: int, daughter_a: int, daughter_b: int) -> str:
    first, second = sorted((int(daughter_a), int(daughter_b)))
    return f"{dataset}:{int(parent)}:{first}:{second}"


def _action_id(candidate_key: str) -> str:
    return "sd2r:" + hashlib.sha256(candidate_key.encode("ascii")).hexdigest()[:24]


def _edge_set(graph: SubmissionGraph) -> set[tuple[int, int]]:
    return {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}


def _edge_index(graph: SubmissionGraph) -> GraphEdgeIndex:
    edges = frozenset(_edge_set(graph))
    outgoing: dict[int, set[tuple[int, int]]] = defaultdict(set)
    incoming: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for pair in edges:
        outgoing[pair[0]].add(pair)
        incoming[pair[1]].add(pair)
    return GraphEdgeIndex(
        edges,
        {key: frozenset(value) for key, value in outgoing.items()},
        {key: frozenset(value) for key, value in incoming.items()},
    )


def _action_for_ids(
    graph: SubmissionGraph,
    parent: int,
    daughter_a: int,
    daughter_b: int,
    *,
    edge_index: GraphEdgeIndex | None = None,
) -> GraphAction:
    daughters = tuple(sorted((int(daughter_a), int(daughter_b))))
    key = _candidate_key(graph.dataset, parent, *daughters)
    indexed = _edge_index(graph) if edge_index is None else edge_index
    conflicts = tuple(sorted(
        set(indexed.outgoing.get(int(parent), ()))
        | set(indexed.incoming.get(daughters[0], ()))
        | set(indexed.incoming.get(daughters[1], ()))
    ))
    return GraphAction(
        action_id=_action_id(key),
        kind=ActionKind.DIVISION_FORK,
        add_edges=((int(parent), daughters[0]), (int(parent), daughters[1])),
        replace_outgoing_from=(int(parent),),
        replace_incoming_to=daughters,
        expected_conflict_edges=conflicts,
    )


def _is_noop(
    graph: SubmissionGraph,
    action: GraphAction,
    *,
    edge_index: GraphEdgeIndex | None = None,
) -> bool:
    del graph, edge_index
    # Conflict edges are, by construction, a subset of the baseline.  The
    # replacement is therefore graph-identical iff it removes and re-adds the
    # exact same edge set.
    return set(action.expected_conflict_edges) == set(action.add_edges)


def discover_fragments(fragment_dir: Path) -> list[FragmentRecord]:
    records: list[FragmentRecord] = []
    seen: set[str] = set()
    for path in sorted(fragment_dir.glob("*.csv"), key=lambda value: value.name):
        match = FRAGMENT_RE.fullmatch(path.name)
        if match is None:
            raise EvaluationError(f"invalid fragment filename: {path.name}")
        split, dataset = match.groups()
        identity = f"{split}:{dataset}"
        if identity in seen:
            raise EvaluationError(f"duplicate fragment identity: {identity}")
        graphs = read_submission(path)
        if set(graphs) != {dataset} or graphs[dataset].dataset != dataset:
            raise EvaluationError(f"fragment identity mismatch: {path}")
        seen.add(identity)
        records.append(FragmentRecord(split, dataset, identity, path.resolve(), graphs[dataset]))
    if not records:
        raise EvaluationError(f"fragment directory contains no CSV fragments: {fragment_dir}")
    return records


def discover_fragment_inventory(
    fragment_dir: Path, *, deadline: RuntimeDeadline
) -> list[FragmentInventory]:
    """Authenticate fragments sequentially without retaining their graphs."""

    inventory: list[FragmentInventory] = []
    seen: set[str] = set()
    for path in sorted(fragment_dir.glob("*.csv"), key=lambda value: value.name):
        deadline.check("fragment inventory authentication")
        match = FRAGMENT_RE.fullmatch(path.name)
        if match is None:
            raise EvaluationError(f"invalid fragment filename: {path.name}")
        split, dataset = match.groups()
        identity = f"{split}:{dataset}"
        if identity in seen:
            raise EvaluationError(f"duplicate fragment identity: {identity}")
        resolved = path.resolve()
        fragment_hash = sha256_file(
            resolved,
            deadline=deadline,
            context=f"fragment inventory hash for {identity}",
        )
        graphs = read_submission(resolved)
        if set(graphs) != {dataset} or graphs[dataset].dataset != dataset:
            raise EvaluationError(f"fragment identity mismatch: {resolved}")
        inventory.append(
            FragmentInventory(
                split,
                dataset,
                identity,
                resolved,
                fragment_hash,
                graph_sha256(graphs[dataset]),
            )
        )
        seen.add(identity)
        del graphs
    if not inventory:
        raise EvaluationError(f"fragment directory contains no CSV fragments: {fragment_dir}")
    return inventory


def graph_to_model_arrays(
    graph: SubmissionGraph,
    *,
    registration_plan: RegistrationPlan | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Convert a native submission graph to pooled-isotropic label arrays.

    Original CSV coordinates remain untouched in every output graph.  Only the
    model view is converted: ``native_coord * native_um / 1.625``.
    Non-adjacent graph edges are excluded from lag-1 context because the
    synthetic model contract admits only exact next-frame lineage edges.
    """

    ordered = sorted(graph.nodes.values(), key=lambda node: (int(node.t), int(node.node_id)))
    node_ids = [int(node.node_id) for node in ordered]
    if len(node_ids) != len(set(node_ids)):
        raise EvaluationError(f"duplicate node IDs in {graph.dataset}")
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    coordinates = np.asarray(
        [[float(node.z), float(node.y), float(node.x)] for node in ordered],
        dtype=np.float64,
    )
    timepoints = np.asarray([int(node.t) for node in ordered], dtype=np.int64)
    if registration_plan is None:
        candidate_coordinates = coordinates
    else:
        try:
            registered = coordinate_views(
                registration_plan,
                timepoints,
                coordinates,
            )
        except ValueError as exc:
            raise EvaluationError(
                f"registration preprocessing failed for {graph.dataset}: {exc}"
            ) from exc
        candidate_coordinates = np.asarray(
            registered.candidate_zyx, dtype=np.float64
        )
    pooled = (
        candidate_coordinates
        * NATIVE_SCALE_ZYX_UM[None, :]
        / POOLED_SCALE_ZYX_UM[None, :]
    )
    nodes = np.column_stack(
        (
            timepoints.astype(np.float64),
            pooled,
            np.asarray(node_ids, dtype=np.float64),
        )
    )
    adjacent: list[tuple[int, int]] = []
    for edge in graph.edges:
        source = graph.nodes[int(edge.source_id)]
        target = graph.nodes[int(edge.target_id)]
        if int(target.t) == int(source.t) + 1:
            adjacent.append((index[int(edge.source_id)], index[int(edge.target_id)]))
    edges = np.asarray(sorted(adjacent), dtype=np.int64).reshape(-1, 2)
    outdegree = Counter(int(source) for source, _target in edges.tolist())
    divisions = np.asarray(sorted(row for row, count in outdegree.items() if count == 2), dtype=np.int64)
    return nodes, edges, divisions, node_ids


def _score_graph(
    graph: SubmissionGraph,
    model: sds.ModelBundle,
    *,
    deadline: RuntimeDeadline | None = None,
    registration_plan: RegistrationPlan | None = None,
) -> tuple[CandidateArrays, dict[str, int]]:
    nodes, edges, divisions, node_ids = graph_to_model_arrays(
        graph,
        registration_plan=registration_plan,
    )
    iterator = sds.iter_score_graph_arrays(
        sequence_id=graph.dataset,
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        model=model,
        voxel_um_pooled=POOLED_SCALE_ZYX_UM,
    )
    columns: dict[str, list[Any]] = {key: [] for key in CANDIDATE_KEYS}
    skipped_noop = 0
    indexed = _edge_index(graph)
    for row_index, row in enumerate(iterator):
        if deadline is not None and row_index % 8192 == 0:
            deadline.check(f"candidate scoring for {graph.dataset}")
        parent = node_ids[int(row["parent_index"])]
        daughter_a = node_ids[int(row["daughter_a_index"])]
        daughter_b = node_ids[int(row["daughter_b_index"])]
        if daughter_a > daughter_b:
            daughter_a, daughter_b = daughter_b, daughter_a
        action = _action_for_ids(
            graph, parent, daughter_a, daughter_b, edge_index=indexed
        )
        if _is_noop(graph, action, edge_index=indexed):
            skipped_noop += 1
            continue
        columns["parent_id"].append(parent)
        columns["daughter_a_id"].append(daughter_a)
        columns["daughter_b_id"].append(daughter_b)
        columns["pair_rank"].append(int(row["pair_rank"]))
        columns["event_probability"].append(float(row["event_probability"]))
        columns["pair_probability"].append(float(row["pair_probability"]))
        columns["score"].append(float(row["score"]))
    arrays = CandidateArrays(
        parent_id=np.asarray(columns["parent_id"], dtype=np.int64),
        daughter_a_id=np.asarray(columns["daughter_a_id"], dtype=np.int64),
        daughter_b_id=np.asarray(columns["daughter_b_id"], dtype=np.int64),
        pair_rank=np.asarray(columns["pair_rank"], dtype=np.int16),
        event_probability=np.asarray(columns["event_probability"], dtype=np.float64),
        pair_probability=np.asarray(columns["pair_probability"], dtype=np.float64),
        score=np.asarray(columns["score"], dtype=np.float64),
    )
    _validate_candidate_arrays(arrays, graph)
    return arrays, {
        "model_rows": len(arrays) + skipped_noop,
        "graph_distinct_candidates": len(arrays),
        "graph_identical_noops_excluded": skipped_noop,
    }


def _validate_candidate_arrays(arrays: CandidateArrays, graph: SubmissionGraph | None = None) -> None:
    lengths = {len(np.asarray(getattr(arrays, key))) for key in CANDIDATE_KEYS}
    if len(lengths) != 1:
        raise EvaluationError("candidate array lengths differ")
    for key in ("parent_id", "daughter_a_id", "daughter_b_id", "pair_rank"):
        if np.asarray(getattr(arrays, key)).dtype.kind not in "iu":
            raise EvaluationError(f"candidate {key} is not integral")
    for key in ("event_probability", "pair_probability", "score"):
        values = np.asarray(getattr(arrays, key), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise EvaluationError(f"candidate {key} contains invalid probabilities")
    if np.any(arrays.daughter_a_id >= arrays.daughter_b_id):
        raise EvaluationError("daughter IDs must be distinct and canonically ordered")
    if len(arrays) > 1:
        order = np.lexsort((arrays.daughter_b_id, arrays.daughter_a_id, arrays.parent_id))
        if np.any(
            (arrays.parent_id[order][1:] == arrays.parent_id[order][:-1])
            & (arrays.daughter_a_id[order][1:] == arrays.daughter_a_id[order][:-1])
            & (arrays.daughter_b_id[order][1:] == arrays.daughter_b_id[order][:-1])
        ):
            raise EvaluationError("duplicate parent/daughter equivalence key")
    if not np.allclose(
        arrays.score,
        arrays.event_probability * arrays.pair_probability,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise EvaluationError("candidate score is not event_probability * pair_probability")
    if graph is not None:
        node_ids = set(graph.nodes)
        indexed = _edge_index(graph)
        for parent, daughter_a, daughter_b in zip(
            arrays.parent_id,
            arrays.daughter_a_id,
            arrays.daughter_b_id,
            strict=True,
        ):
            if {int(parent), int(daughter_a), int(daughter_b)} - node_ids:
                raise EvaluationError("candidate references a missing graph node")
            parent_t = int(graph.nodes[int(parent)].t)
            if any(
                int(graph.nodes[int(daughter)].t) != parent_t + 1
                for daughter in (daughter_a, daughter_b)
            ):
                raise EvaluationError("candidate daughters are not exactly next-frame nodes")
            action = _action_for_ids(
                graph,
                int(parent),
                int(daughter_a),
                int(daughter_b),
                edge_index=indexed,
            )
            if _is_noop(graph, action, edge_index=indexed):
                raise EvaluationError("sealed candidate contains a graph-identical no-op")


def _load_candidate_npz(path: Path, graph: SubmissionGraph | None = None) -> CandidateArrays:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(CANDIDATE_KEYS):
                raise EvaluationError(f"candidate NPZ schema mismatch: {path}")
            arrays = CandidateArrays(**{key: np.asarray(archive[key]) for key in CANDIDATE_KEYS})
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"failed to read candidate NPZ {path}: {exc}") from exc
    _validate_candidate_arrays(arrays, graph)
    return arrays


def _load_inventory_graph(
    inventory: FragmentInventory, deadline: RuntimeDeadline
) -> SubmissionGraph:
    actual_hash = sha256_file(
        inventory.path,
        deadline=deadline,
        context=f"worker fragment hash for {inventory.identity}",
    )
    if actual_hash != inventory.fragment_sha256:
        raise EvaluationError(f"fragment changed after inventory authentication: {inventory.identity}")
    graphs = read_submission(inventory.path)
    if set(graphs) != {inventory.dataset}:
        raise EvaluationError(f"worker fragment identity mismatch: {inventory.identity}")
    graph = graphs[inventory.dataset]
    if graph_sha256(graph) != inventory.baseline_graph_sha256:
        raise EvaluationError(f"worker graph hash mismatch: {inventory.identity}")
    return graph


def _score_generation_task(
    task: GenerationTask,
    *,
    model: sds.ModelBundle,
    registration_census: Path,
    deadline: RuntimeDeadline,
) -> dict[str, Any]:
    inventory = task.inventory
    deadline.check(f"generate worker video {inventory.identity}")
    graph = _load_inventory_graph(inventory, deadline)
    registration_plan = load_registration_plan(registration_census, inventory.identity)
    arrays, counts = _score_graph(
        graph,
        model,
        deadline=deadline,
        registration_plan=registration_plan,
    )
    candidate_hash = _write_new_npz(task.candidate_path, arrays)
    deadline.check(f"generate worker video {inventory.identity} completion")
    return {
        "identity": inventory.identity,
        "split": inventory.split,
        "dataset": inventory.dataset,
        "family": inventory.dataset.split("_", 1)[0],
        "fragment_path": str(inventory.path),
        "fragment_sha256": inventory.fragment_sha256,
        "baseline_graph_sha256": inventory.baseline_graph_sha256,
        "candidate_file_name": task.candidate_path.name,
        "candidate_sha256": candidate_hash,
        "registration_receipt": registration_plan.receipt(),
        **counts,
    }


def _initialize_generation_worker(
    model_json: str,
    registration_census: str,
    deadline_started: float,
    runtime_cap_seconds: float,
) -> None:
    global _GENERATION_WORKER_MODEL
    global _GENERATION_WORKER_CENSUS
    global _GENERATION_WORKER_DEADLINE
    _GENERATION_WORKER_MODEL = sds.load_model(Path(model_json))
    _GENERATION_WORKER_CENSUS = Path(registration_census)
    _GENERATION_WORKER_DEADLINE = RuntimeDeadline(
        float(deadline_started), float(runtime_cap_seconds)
    )
    _GENERATION_WORKER_DEADLINE.check("parallel worker initialization")


def _parallel_generation_worker(task: GenerationTask) -> dict[str, Any]:
    if (
        _GENERATION_WORKER_MODEL is None
        or _GENERATION_WORKER_CENSUS is None
        or _GENERATION_WORKER_DEADLINE is None
    ):
        raise EvaluationError("parallel generation worker was not initialized")
    return _score_generation_task(
        task,
        model=_GENERATION_WORKER_MODEL,
        registration_census=_GENERATION_WORKER_CENSUS,
        deadline=_GENERATION_WORKER_DEADLINE,
    )


def _terminate_executor_workers(
    executor: ProcessPoolExecutor, *, join_timeout_seconds: float = 5.0
) -> None:
    """Bound and terminate only this executor's live child processes."""

    raw_processes = getattr(executor, "_processes", None)
    processes = tuple(raw_processes.values()) if raw_processes else ()
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + float(join_timeout_seconds)
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        if process.is_alive():
            process.join(timeout=max(0.0, deadline - time.monotonic()))


def generate_scores(
    model_json: Path,
    fragment_dir: Path,
    output_dir: Path,
    *,
    registration_census: Path = DEFAULT_REGISTRATION_CENSUS,
    runtime_cap_seconds: float = DEFAULT_RUNTIME_CAP_SECONDS,
    workers: int = 1,
) -> dict[str, Any]:
    deadline = RuntimeDeadline.start(runtime_cap_seconds)
    deadline.check("generate initialization")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_GENERATION_WORKERS:
        raise EvaluationError(f"workers must be an integer in [1,{MAX_GENERATION_WORKERS}]")
    if output_dir.exists():
        raise EvaluationError(f"refusing to reuse score output directory: {output_dir}")
    if workers > 1:
        amendments = (
            (EXECUTION_AMENDMENT_3, EXECUTION_AMENDMENT_3_SHA256, 3),
            (EXECUTION_AMENDMENT_4, EXECUTION_AMENDMENT_4_SHA256, 4),
        )
        for amendment_path, expected_hash, number in amendments:
            amendment_hash = sha256_file(
                amendment_path,
                deadline=deadline,
                context=f"Execution Amendment {number} authentication",
            )
            if amendment_hash != expected_hash:
                raise EvaluationError(
                    f"Execution Amendment {number} SHA-256 mismatch"
                )
    output_dir.mkdir(parents=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir()
    model = sds.load_model(model_json)
    inventory = discover_fragment_inventory(fragment_dir, deadline=deadline)
    tasks = [
        GenerationTask(
            row,
            candidate_dir / f"{row.split}__{row.dataset}.npz",
        )
        for row in inventory
    ]
    raw_records: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            raw_records.append(
                _score_generation_task(
                    task,
                    model=model,
                    registration_census=registration_census,
                    deadline=deadline,
                )
            )
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_generation_worker,
            initargs=(
                str(model_json.resolve()),
                str(registration_census.resolve()),
                deadline.started,
                deadline.cap_seconds,
            ),
        )
        futures = [executor.submit(_parallel_generation_worker, task) for task in tasks]
        try:
            # Results are consumed in authenticated inventory order regardless
            # of child completion order.
            for task, future in zip(tasks, futures, strict=True):
                deadline.check(f"await parallel video {task.inventory.identity}")
                raw_records.append(
                    future.result(
                        timeout=deadline.remaining(
                            f"await parallel video {task.inventory.identity}"
                        )
                    )
                )
                deadline.check(f"parallel video {task.inventory.identity} completion")
        except BaseException:
            for future in futures:
                future.cancel()
            _terminate_executor_workers(executor)
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    manifest_records: list[dict[str, Any]] = []
    total_candidates = 0
    for inventory_row, row in zip(inventory, raw_records, strict=True):
        if row.get("identity") != inventory_row.identity:
            raise EvaluationError("parallel generation result order drifted")
        candidate_path = candidate_dir / str(row.pop("candidate_file_name"))
        row["candidate_file"] = str(candidate_path.relative_to(output_dir))
        total_candidates += int(row["graph_distinct_candidates"])
        manifest_records.append(row)
    manifest = {
        "schema": SCORE_SCHEMA,
        "node": NODE,
        "status": "FROZEN_BEFORE_GT_OPEN",
        "contains_labels": False,
        "contains_voxels": False,
        "gt_archive_opened": False,
        "runtime_cap_seconds": float(runtime_cap_seconds),
        "feature_policy": "geometry/lineage/motion only; no texture, contrast, or intensity",
        "model": {
            "manifest_path": str(model_json.resolve()),
            "manifest_sha256": str(model.json_sha256 or sha256_file(model_json)),
            "arrays_sha256": str(model.arrays_sha256),
        },
        "coordinate_adapter": {
            "native_scale_zyx_um": NATIVE_SCALE_ZYX_UM.tolist(),
            "model_pooled_scale_zyx_um": POOLED_SCALE_ZYX_UM.tolist(),
            "formula": (
                "model_coord = registered_native_coord * native_scale_um / 1.625"
            ),
            "registration_preprocessing": {
                "standing": True,
                "census": str(registration_census.resolve()),
                "census_sha256": registration_sha256_file(registration_census),
                "candidate_topology_and_geometry": "registered",
                "baseline_fragment_and_output": "original",
                "applications_per_video": 1,
            },
        },
        "candidate_generator": {
            "maximum_next_frame_candidates": MAX_NEXT_FRAME_CANDIDATES,
            "daughter_pairs": "all unordered pairs among deterministic nearest-8",
            "score": "P_real(division at parent) * P(pair | division,parent)",
            "graph_identical_noops_excluded_truth_blind": True,
        },
        "fragment_root": str(fragment_dir.resolve()),
        "video_count": len(inventory),
        "candidate_count": total_candidates,
        "records": manifest_records,
        "execution": {
            "workers": workers,
            "mode": "serial" if workers == 1 else "bounded_process_pool_per_video",
            "maximum_live_graphs": workers,
            "model_loaded_per_worker": workers > 1,
            "deterministic_manifest_order": "authenticated fragment inventory order",
            "execution_amendment_3_sha256": (
                EXECUTION_AMENDMENT_3_SHA256 if workers > 1 else None
            ),
            "execution_amendment_4_sha256": (
                EXECUTION_AMENDMENT_4_SHA256 if workers > 1 else None
            ),
        },
    }
    deadline.check("generate manifest publication")
    _write_new_json(output_dir / "SCORES_MANIFEST.json", manifest)
    return manifest


def load_frozen_scores(
    manifest_path: Path, *, deadline: RuntimeDeadline | None = None
) -> tuple[dict[str, Any], list[FragmentRecord]]:
    """Authenticate every score and baseline file before returning.

    Evaluation callers must invoke this function before constructing a
    ``ZipStore``.  This ordering is the code-level real-label firewall.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid score manifest: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCORE_SCHEMA
        or manifest.get("node") != NODE
        or manifest.get("status") != "FROZEN_BEFORE_GT_OPEN"
        or manifest.get("contains_labels") is not False
        or manifest.get("gt_archive_opened") is not False
    ):
        raise EvaluationError("score manifest does not satisfy the frozen no-label contract")
    model_row = manifest.get("model")
    if not isinstance(model_row, dict):
        raise EvaluationError("score manifest model record is absent")
    model_path = Path(str(model_row.get("manifest_path", "")))
    if (
        not model_path.is_file()
        or sha256_file(
            model_path,
            deadline=deadline,
            context="frozen model manifest authentication",
        )
        != model_row.get("manifest_sha256")
    ):
        raise EvaluationError("frozen model manifest hash mismatch")
    model = sds.load_model(model_path)
    if model.arrays_sha256 != model_row.get("arrays_sha256"):
        raise EvaluationError("frozen model arrays hash mismatch")
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise EvaluationError("score manifest execution record is absent")
    manifest_workers = execution.get("workers")
    if (
        isinstance(manifest_workers, bool)
        or not isinstance(manifest_workers, int)
        or not 1 <= manifest_workers <= MAX_GENERATION_WORKERS
    ):
        raise EvaluationError("score manifest worker count is invalid")
    expected_mode = "serial" if manifest_workers == 1 else "bounded_process_pool_per_video"
    if (
        execution.get("mode") != expected_mode
        or execution.get("maximum_live_graphs") != manifest_workers
        or execution.get("model_loaded_per_worker") is not (manifest_workers > 1)
        or execution.get("deterministic_manifest_order")
        != "authenticated fragment inventory order"
    ):
        raise EvaluationError("score manifest execution contract mismatch")
    if manifest_workers > 1:
        if (
            execution.get("execution_amendment_3_sha256")
            != EXECUTION_AMENDMENT_3_SHA256
            or sha256_file(
                EXECUTION_AMENDMENT_3,
                deadline=deadline,
                context="Execution Amendment 3 manifest authentication",
            )
            != EXECUTION_AMENDMENT_3_SHA256
        ):
            raise EvaluationError("parallel score manifest does not pin Execution Amendment 3")
        if (
            execution.get("execution_amendment_4_sha256")
            != EXECUTION_AMENDMENT_4_SHA256
            or sha256_file(
                EXECUTION_AMENDMENT_4,
                deadline=deadline,
                context="Execution Amendment 4 manifest authentication",
            )
            != EXECUTION_AMENDMENT_4_SHA256
        ):
            raise EvaluationError("parallel score manifest does not pin Execution Amendment 4")
    elif (
        execution.get("execution_amendment_3_sha256") is not None
        or execution.get("execution_amendment_4_sha256") is not None
    ):
        raise EvaluationError("serial score manifest unexpectedly pins parallel execution amendment")
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or int(manifest.get("video_count", -1)) != len(raw_records):
        raise EvaluationError("score manifest record count mismatch")
    records: list[FragmentRecord] = []
    seen: set[str] = set()
    candidate_count = 0
    for row in raw_records:
        if deadline is not None:
            deadline.check("frozen score record authentication")
        if not isinstance(row, dict):
            raise EvaluationError("score manifest record is not an object")
        split, dataset = str(row["identity"]).split(":", 1)
        if row.get("split") != split or row.get("dataset") != dataset:
            raise EvaluationError("score manifest identity fields disagree")
        identity = f"{split}:{dataset}"
        if identity in seen:
            raise EvaluationError(f"duplicate score identity: {identity}")
        seen.add(identity)
        fragment = Path(str(row["fragment_path"]))
        if not fragment.is_file() or sha256_file(
            fragment,
            deadline=deadline,
            context=f"fragment authentication for {identity}",
        ) != row.get("fragment_sha256"):
            raise EvaluationError(f"fragment hash mismatch: {identity}")
        graphs = read_submission(fragment)
        if set(graphs) != {dataset}:
            raise EvaluationError(f"fragment graph identity mismatch: {identity}")
        graph = graphs[dataset]
        if graph_sha256(graph) != row.get("baseline_graph_sha256"):
            raise EvaluationError(f"baseline graph hash mismatch: {identity}")
        candidate_path = manifest_path.parent / str(row["candidate_file"])
        if not candidate_path.is_file() or sha256_file(
            candidate_path,
            deadline=deadline,
            context=f"candidate bank authentication for {identity}",
        ) != row.get("candidate_sha256"):
            raise EvaluationError(f"candidate score hash mismatch: {identity}")
        arrays = _load_candidate_npz(candidate_path, graph)
        if len(arrays) != int(row.get("graph_distinct_candidates", -1)):
            raise EvaluationError(f"candidate count mismatch: {identity}")
        candidate_count += len(arrays)
        records.append(FragmentRecord(split, dataset, identity, fragment, graph))
    if candidate_count != int(manifest.get("candidate_count", -1)):
        raise EvaluationError("aggregate candidate count mismatch")
    return manifest, records


def _candidate_path(manifest_path: Path, manifest: Mapping[str, Any], identity: str) -> Path:
    row = next(value for value in manifest["records"] if value["identity"] == identity)
    return manifest_path.parent / str(row["candidate_file"])


def greedy_selected_indices(graph: SubmissionGraph, arrays: CandidateArrays) -> np.ndarray:
    """Return the threshold-independent greedy survivor sequence."""

    if len(arrays) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort(
        (
            arrays.daughter_b_id,
            arrays.daughter_a_id,
            arrays.parent_id,
            -arrays.score,
        )
    )
    occupied: set[int] = set()
    owned_conflicts: set[tuple[int, int]] = set()
    selected: list[int] = []
    indexed = _edge_index(graph)
    for raw_index in order:
        index = int(raw_index)
        parent, daughter_a, daughter_b, _score = arrays.row(index)
        action = _action_for_ids(
            graph, parent, daughter_a, daughter_b, edge_index=indexed
        )
        boundary = {parent, daughter_a, daughter_b}
        conflicts = set(action.expected_conflict_edges)
        if boundary & occupied or conflicts & owned_conflicts:
            continue
        occupied.update(boundary)
        owned_conflicts.update(conflicts)
        selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def policy_indices(
    arrays: CandidateArrays,
    greedy: np.ndarray,
    *,
    threshold: float,
    cap: int | None,
) -> np.ndarray:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be finite and in [0,1]")
    if cap is not None and cap <= 0:
        raise EvaluationError("cap must be positive or 'all'")
    eligible = greedy[arrays.score[greedy] >= float(threshold)]
    return eligible if cap is None else eligible[: int(cap)]


def _compile_disjoint_actions(
    baseline: SubmissionGraph, actions: Sequence[GraphAction]
) -> tuple[SubmissionGraph, str]:
    """Compile validated actions as one exact union transaction.

    ``action_compiler.compile_batch`` currently rejects an edge that one
    division action truthfully owns as a baseline conflict and also re-adds
    (for example, retaining one existing child while adding its sister).
    Each action is therefore validated by the canonical ``compile_action``
    seam, then the already conflict-disjoint set is unioned once.  This keeps
    exact incident ownership without weakening the compiler contract.
    """

    if not actions:
        raise EvaluationError("cannot compile an empty action transaction")
    if len({action.action_id for action in actions}) != len(actions):
        raise EvaluationError("duplicate action ID in transaction")
    removed: set[tuple[int, int]] = set()
    additions: set[tuple[int, int]] = set()
    action_hashes: list[str] = []
    for action in sorted(actions, key=lambda value: value.action_id):
        compiled = compile_action(baseline, action)
        conflicts = set(compiled.removed_edges)
        added = set(action.add_edges)
        if conflicts & (removed | additions) or added & (removed | additions):
            raise EvaluationError("selected actions share an added or owned conflict edge")
        removed.update(conflicts)
        additions.update(added)
        action_hashes.append(action_sha256(action))
    final_edges = (_edge_set(baseline) - removed) | additions
    graph = SubmissionGraph(
        baseline.dataset,
        dict(baseline.nodes),
        [SubmissionEdge(source, target) for source, target in sorted(final_edges)],
    )
    validate_graphs({graph.dataset: graph})
    payload = {
        "before_graph_sha256": graph_sha256(baseline),
        "action_sha256s": action_hashes,
        "after_graph_sha256": graph_sha256(graph),
        "removed_edges": [list(pair) for pair in sorted(removed)],
    }
    batch_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return graph, batch_hash


def _official_node_matching(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[int, int]:
    """Original prediction ID -> original GT ID under the official matcher."""

    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options
    from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph

    ordered_pred = sorted(graph.nodes.values(), key=lambda node: (int(node.t), int(node.node_id)))
    pred_original = [int(node.node_id) for node in ordered_pred]
    pred_rows = [
        {
            "dataset": graph.dataset,
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": int(node.z),
            "y": int(node.y),
            "x": int(node.x),
        }
        for node in ordered_pred
    ]
    pred = prediction_to_graph(pred_rows, [])
    truth = geff_arrays_to_graph(dict(gt))
    gt_original = [
        int(node_id)
        for node_id, t in sorted(
            zip(gt["nodes"]["node_id"], gt["nodes"]["t"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        )
    ]
    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred.match(
                truth,
                matching=DistanceMatching(max_distance=MATCH_RADIUS_UM, scale=scale_zyx),
            )
    finally:
        set_options(show_progress=previous)
    attrs = pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    result: dict[int, int] = {}
    gt_seen: set[int] = set()
    for pred_id, gt_id in attrs.iter_rows():
        if gt_id is None or int(gt_id) == -1:
            continue
        original_pred = pred_original[int(pred_id)]
        original_gt = gt_original[int(gt_id)]
        if original_pred in result or original_gt in gt_seen:
            raise EvaluationError("official node matching was not one-to-one")
        result[original_pred] = original_gt
        gt_seen.add(original_gt)
    return result


def exact_gt_fork_labels(
    graph: SubmissionGraph,
    arrays: CandidateArrays,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, int, int]:
    """Label exact direct-child forks after official coordinate matching."""

    pred_to_gt = _official_node_matching(graph, gt, scale_zyx)
    successors: dict[int, set[int]] = defaultdict(set)
    for source, target in np.asarray(gt["edges"], dtype=np.int64).reshape(-1, 2):
        successors[int(source)].add(int(target))
    gt_dividers = {source for source, children in successors.items() if len(children) == 2}
    labels = np.zeros(len(arrays), dtype=np.bool_)
    reached: set[int] = set()
    for index in range(len(arrays)):
        parent, daughter_a, daughter_b, _score = arrays.row(index)
        gt_parent = pred_to_gt.get(parent)
        gt_a = pred_to_gt.get(daughter_a)
        gt_b = pred_to_gt.get(daughter_b)
        if (
            gt_parent is not None
            and gt_a is not None
            and gt_b is not None
            and gt_a != gt_b
            and gt_parent in gt_dividers
            and {gt_a, gt_b} == successors[gt_parent]
        ):
            labels[index] = True
            reached.add(gt_parent)
    if int(labels.sum()) != len(reached):
        raise EvaluationError("multiple candidate keys map to one exact GT division")
    return labels, len(gt_dividers), len(reached)


def _group_stream(
    scores: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    *,
    descending: bool,
    chunk_size: int = 1_000_000,
    deadline: RuntimeDeadline | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield sorted tie groups as ``(start_position, count, positives)`` arrays.

    Only the integer argsort and one bounded chunk are resident.  A final tie
    group is carried across chunk boundaries so metrics remain exact.
    """

    ordered = order[::-1] if descending else order
    carry_score: float | None = None
    carry_start = 0
    carry_count = 0
    carry_positive = 0
    for base in range(0, len(ordered), chunk_size):
        if deadline is not None:
            deadline.check("exact ranking tie-group scan")
        indices = ordered[base : base + chunk_size]
        chunk_scores = np.asarray(scores[indices], dtype=np.float64)
        chunk_labels = np.asarray(labels[indices], dtype=np.uint8)
        boundaries = np.flatnonzero(chunk_scores[1:] != chunk_scores[:-1]) + 1
        starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
        ends = np.concatenate((boundaries, np.asarray([len(indices)], dtype=np.int64)))
        counts = ends - starts
        positives = np.add.reduceat(chunk_labels.astype(np.int64), starts)
        group_scores = chunk_scores[starts]

        first = 0
        if carry_score is not None:
            if len(group_scores) and float(group_scores[0]) == carry_score:
                carry_count += int(counts[0])
                carry_positive += int(positives[0])
                first = 1
                if len(group_scores) > 1:
                    yield (
                        np.asarray([carry_start], dtype=np.int64),
                        np.asarray([carry_count], dtype=np.int64),
                        np.asarray([carry_positive], dtype=np.int64),
                    )
                    carry_score = None
            else:
                yield (
                    np.asarray([carry_start], dtype=np.int64),
                    np.asarray([carry_count], dtype=np.int64),
                    np.asarray([carry_positive], dtype=np.int64),
                )
                carry_score = None

        remaining = len(group_scores) - first
        if remaining <= 0:
            continue
        last = len(group_scores) - 1
        if first < last:
            selected = slice(first, last)
            yield (
                (base + starts[selected]).astype(np.int64, copy=False),
                counts[selected].astype(np.int64, copy=False),
                positives[selected].astype(np.int64, copy=False),
            )
        carry_score = float(group_scores[last])
        carry_start = int(base + starts[last])
        carry_count = int(counts[last])
        carry_positive = int(positives[last])
    if carry_score is not None:
        yield (
            np.asarray([carry_start], dtype=np.int64),
            np.asarray([carry_count], dtype=np.int64),
            np.asarray([carry_positive], dtype=np.int64),
        )


def binary_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    deadline: RuntimeDeadline | None = None,
) -> dict[str, float | int | None]:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if (
        scores.ndim != 1
        or scores.dtype.kind != "f"
        or labels.shape != scores.shape
        or not np.all(np.isfinite(scores))
    ):
        raise EvaluationError("invalid binary ranking arrays")
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    auroc: float | None = None
    average_precision: float | None = None
    if positives:
        # Tie order is irrelevant because tie groups are aggregated exactly;
        # quicksort avoids mergesort's second full-size workspace.
        order = np.argsort(scores, kind="quicksort")
        if deadline is not None:
            deadline.check("exact ranking argsort")
        if negatives:
            positive_rank_sum = 0.0
            for starts, counts, group_positive in _group_stream(
                scores, labels, order, descending=False, deadline=deadline
            ):
                average_ranks = starts.astype(np.float64) + (counts.astype(np.float64) + 1.0) / 2.0
                positive_rank_sum += float(np.dot(group_positive, average_ranks))
            auroc = (
                positive_rank_sum - positives * (positives + 1) / 2.0
            ) / (positives * negatives)

        seen_count = 0
        seen_positive = 0
        ap = 0.0
        for _starts, counts, group_positive in _group_stream(
            scores, labels, order, descending=True, deadline=deadline
        ):
            cumulative_count = seen_count + np.cumsum(counts)
            cumulative_positive = seen_positive + np.cumsum(group_positive)
            precision = cumulative_positive / cumulative_count
            ap += float(np.sum((group_positive / positives) * precision))
            seen_count = int(cumulative_count[-1])
            seen_positive = int(cumulative_positive[-1])
        average_precision = ap
    return {
        "candidate_count": int(labels.size),
        "positive_count": positives,
        "negative_count": negatives,
        "auroc": auroc,
        "pr_auc_average_precision": average_precision,
    }


def calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    deadline: RuntimeDeadline | None = None,
) -> dict[str, Any]:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    edges = np.asarray((0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    score_sums = np.zeros(len(edges) - 1, dtype=np.float64)
    positive_sums = np.zeros(len(edges) - 1, dtype=np.int64)
    squared_error_sum = 0.0
    log_loss_sum = 0.0
    for start in range(0, len(scores), 1_000_000):
        if deadline is not None:
            deadline.check("calibration chunk scan")
        chunk_scores = np.asarray(scores[start : start + 1_000_000], dtype=np.float64)
        chunk_labels = np.asarray(labels[start : start + 1_000_000], dtype=np.float64)
        bins = np.minimum(np.searchsorted(edges, chunk_scores, side="right") - 1, len(edges) - 2)
        counts += np.bincount(bins, minlength=len(counts)).astype(np.int64)
        score_sums += np.bincount(bins, weights=chunk_scores, minlength=len(counts))
        positive_sums += np.bincount(bins, weights=chunk_labels, minlength=len(counts)).astype(np.int64)
        squared_error_sum += float(np.sum((chunk_scores - chunk_labels) ** 2))
        clipped = np.clip(chunk_scores, 1e-15, 1.0 - 1e-15)
        log_loss_sum += float(
            np.sum(-(chunk_labels * np.log(clipped) + (1.0 - chunk_labels) * np.log(1.0 - clipped)))
        )
    rows = [
        {
            "left_closed": float(left),
            "right": float(right),
            "count": int(counts[index]),
            "mean_score": float(score_sums[index] / counts[index]) if counts[index] else None,
            "observed_positive_rate": (
                float(positive_sums[index] / counts[index]) if counts[index] else None
            ),
        }
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]
    return {
        "brier": squared_error_sum / len(scores) if len(scores) else None,
        "log_loss": log_loss_sum / len(scores) if len(scores) else None,
        "reliability": rows,
    }


def _counts_row(tp: int, fp: int, total_gt: int) -> dict[str, Any]:
    fn = max(0, int(total_gt) - int(tp))
    denominator = int(tp) + int(fp) + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / total_gt) if total_gt else None,
        "division_jaccard": float(tp / denominator) if denominator else None,
    }


def _policy_grid(
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    caps: Sequence[int | None] = DEFAULT_CAPS,
) -> list[tuple[float, int | None]]:
    return [(float(threshold), cap) for threshold in thresholds for cap in caps]


def evaluate_scores(
    manifest_path: Path,
    gt_archive: Path,
    output: Path,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    caps: Sequence[int | None] = DEFAULT_CAPS,
    runtime_cap_seconds: float = DEFAULT_RUNTIME_CAP_SECONDS,
) -> dict[str, Any]:
    deadline = RuntimeDeadline.start(runtime_cap_seconds)
    deadline.check("evaluation initialization")
    if output.exists():
        raise EvaluationError(f"refusing to overwrite evaluation output: {output}")
    # Mandatory ordering: all score/baseline bytes authenticate before GT opens.
    manifest, records = load_frozen_scores(manifest_path, deadline=deadline)
    deadline.check("frozen score authentication")
    if any(record.split != "train" for record in records):
        raise EvaluationError("GT evaluation is permitted only for train fragments")

    import zarr
    from zarr.storage import ZipStore

    manifest_rows = {str(row["identity"]): row for row in manifest["records"]}
    total_candidates = int(manifest["candidate_count"])
    family_candidate_counts: Counter[str] = Counter()
    for record in records:
        family_candidate_counts[record.dataset.split("_", 1)[0]] += int(
            manifest_rows[record.identity]["graph_distinct_candidates"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.TemporaryDirectory(prefix="sd2r-real-eval-", dir=output.parent)
    scratch_root = Path(scratch.name)
    if total_candidates:
        scores = np.memmap(
            scratch_root / "all_scores.f64", mode="w+", dtype=np.float64, shape=(total_candidates,)
        )
        all_labels_store = np.memmap(
            scratch_root / "all_labels.u8", mode="w+", dtype=np.uint8, shape=(total_candidates,)
        )
    else:
        scores = np.empty(0, dtype=np.float64)
        all_labels_store = np.empty(0, dtype=np.uint8)
    family_scores: dict[str, np.ndarray] = {}
    family_labels: dict[str, np.ndarray] = {}
    for family, count in family_candidate_counts.items():
        if count:
            family_scores[family] = np.memmap(
                scratch_root / f"{family}.scores.f64",
                mode="w+",
                dtype=np.float64,
                shape=(count,),
            )
            family_labels[family] = np.memmap(
                scratch_root / f"{family}.labels.u8",
                mode="w+",
                dtype=np.uint8,
                shape=(count,),
            )
        else:
            family_scores[family] = np.empty(0, dtype=np.float64)
            family_labels[family] = np.empty(0, dtype=np.uint8)
    global_offset = 0
    family_offsets: Counter[str] = Counter()
    total_gt = 0
    reachable_gt = 0
    total_selected: dict[tuple[float, int | None], list[int]] = {
        policy: [0, 0] for policy in _policy_grid(thresholds, caps)
    }
    family_selected: dict[str, dict[tuple[float, int | None], list[int]]] = defaultdict(
        lambda: {policy: [0, 0] for policy in _policy_grid(thresholds, caps)}
    )
    family_gt: Counter[str] = Counter()
    video_records: list[dict[str, Any]] = []

    with ZipStore(gt_archive, mode="r") as store:
        for record in records:
            deadline.check(f"evaluate video {record.identity}")
            arrays = _load_candidate_npz(
                _candidate_path(manifest_path, manifest, record.identity), record.graph
            )
            gt = read_geff_arrays(store, "train", record.dataset)
            group = zarr.open_group(store=store, path=f"train/{record.dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            candidate_labels, video_gt, video_reachable = exact_gt_fork_labels(
                record.graph, arrays, gt, scale
            )
            greedy = greedy_selected_indices(record.graph, arrays)
            family = record.dataset.split("_", 1)[0]
            next_global = global_offset + len(arrays)
            scores[global_offset:next_global] = arrays.score
            labels_video = candidate_labels.astype(np.uint8)
            all_labels_store[global_offset:next_global] = labels_video
            global_offset = next_global
            family_offset = family_offsets[family]
            next_family = family_offset + len(arrays)
            family_scores[family][family_offset:next_family] = arrays.score
            family_labels[family][family_offset:next_family] = labels_video
            family_offsets[family] = next_family
            total_gt += video_gt
            reachable_gt += video_reachable
            family_gt[family] += video_gt
            policy_rows: list[dict[str, Any]] = []
            for policy in _policy_grid(thresholds, caps):
                deadline.check(f"evaluate policies for {record.identity}")
                threshold, cap = policy
                selected = policy_indices(arrays, greedy, threshold=threshold, cap=cap)
                tp = int(labels_video[selected].sum())
                fp = int(len(selected) - tp)
                total_selected[policy][0] += tp
                total_selected[policy][1] += fp
                family_selected[family][policy][0] += tp
                family_selected[family][policy][1] += fp
                policy_rows.append(
                    {
                        "threshold": threshold,
                        "per_video_cap": "all" if cap is None else cap,
                        "selected": int(len(selected)),
                        "tp": tp,
                        "fp": fp,
                    }
                )
            video_records.append(
                {
                    "identity": record.identity,
                    "family": family,
                    "candidate_count": len(arrays),
                    "positive_candidate_count": int(labels_video.sum()),
                    "gt_divisions": video_gt,
                    "reachable_gt_divisions": video_reachable,
                    "greedy_survivor_count_at_zero_threshold": int(len(greedy)),
                    "policies": policy_rows,
                }
            )
            deadline.check(f"evaluate video {record.identity} completion")

    if global_offset != total_candidates:
        raise EvaluationError("evaluation did not fill the authenticated candidate population")
    if any(family_offsets[family] != count for family, count in family_candidate_counts.items()):
        raise EvaluationError("evaluation family candidate offsets drifted")
    if isinstance(scores, np.memmap):
        scores.flush()
    if isinstance(all_labels_store, np.memmap):
        all_labels_store.flush()
    for value in (*family_scores.values(), *family_labels.values()):
        if isinstance(value, np.memmap):
            value.flush()
    frontier = []
    for threshold, cap in _policy_grid(thresholds, caps):
        tp, fp = total_selected[(threshold, cap)]
        frontier.append(
            {
                "threshold": threshold,
                "per_video_cap": "all" if cap is None else cap,
                "predicted_forks": tp + fp,
                **_counts_row(tp, fp, total_gt),
                "families": {
                    family: _counts_row(
                        family_selected[family][(threshold, cap)][0],
                        family_selected[family][(threshold, cap)][1],
                        family_gt[family],
                    )
                    for family in sorted(family_gt)
                },
            }
        )
    deadline.check("global real ranking metrics")
    global_ranking = binary_ranking_metrics(
        scores, all_labels_store, deadline=deadline
    )
    deadline.check("global real calibration metrics")
    global_calibration = calibration_metrics(
        scores, all_labels_store, deadline=deadline
    )
    deadline.check("family real ranking metrics")
    family_report = {
        family: {
            "gt_divisions": family_gt[family],
            "ranking": binary_ranking_metrics(
                family_scores[family], family_labels[family], deadline=deadline
            ),
        }
        for family in sorted(family_gt)
    }
    deadline.check("evaluation report publication")
    report = {
        "schema": EVALUATION_SCHEMA,
        "node": NODE,
        "status": "COMPLETE_REAL_LABEL_EVALUATION",
        "runtime_cap_seconds": float(runtime_cap_seconds),
        "score_manifest": str(manifest_path.resolve()),
        "score_manifest_sha256": sha256_file(manifest_path),
        "truth_firewall": {
            "all_score_fragment_and_model_hashes_verified_before_gt_open": True,
            "labels_used_for_generation_or_scores": False,
            "labels_used_for_model_refit": False,
        },
        "label_definition": (
            "candidate parent and both direct daughters map one-to-one, within the official "
            "7-um coordinate matcher, to one exact GT outdegree-2 parent and its two direct children"
        ),
        "gt_archive": {
            "path": str(gt_archive.resolve()),
            "bytes": gt_archive.stat().st_size,
            "sha256": sha256_file(
                gt_archive,
                deadline=deadline,
                context="GT archive authentication hash",
            ),
        },
        "ranking": global_ranking,
        "calibration": global_calibration,
        "real_divisions": {
            "total": total_gt,
            "reachable_by_exact_candidate": reachable_gt,
            "unreachable": total_gt - reachable_gt,
        },
        "families": family_report,
        "frontier": frontier,
        "videos": video_records,
    }
    deadline.check("evaluation report publication")
    _write_new_json(output, report)
    del scores, all_labels_store, family_scores, family_labels
    scratch.cleanup()
    return report


def _policy_name(threshold: float, cap: int | None) -> str:
    threshold_text = format(float(threshold), ".12g").replace(".", "p")
    return f"t{threshold_text}_cap{'all' if cap is None else cap}"


def integrate_scores(
    manifest_path: Path,
    output_dir: Path,
    *,
    policies: Sequence[tuple[float, int | None]],
    runtime_cap_seconds: float = DEFAULT_RUNTIME_CAP_SECONDS,
) -> dict[str, Any]:
    deadline = RuntimeDeadline.start(runtime_cap_seconds)
    deadline.check("integration initialization")
    manifest, records = load_frozen_scores(manifest_path, deadline=deadline)
    deadline.check("integration score authentication")
    if output_dir.exists():
        raise EvaluationError(f"refusing to reuse integration directory: {output_dir}")
    if not policies:
        raise EvaluationError("at least one integration policy is required")
    for threshold, cap in policies:
        deadline.check("integration policy directory setup")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise EvaluationError("integration threshold must be finite and in [0,1]")
        if cap is not None and cap <= 0:
            raise EvaluationError("integration cap must be positive or 'all'")
    names = [_policy_name(threshold, cap) for threshold, cap in policies]
    if len(names) != len(set(names)):
        raise EvaluationError("integration policy directory names collide")
    output_dir.mkdir(parents=True)
    if len(set(policies)) != len(policies):
        raise EvaluationError("duplicate integration policy")
    policy_state: dict[tuple[float, int | None], dict[str, Any]] = {}
    for threshold, cap in policies:
        policy_dir = output_dir / _policy_name(threshold, cap)
        fragment_dir = policy_dir / "fragments"
        fragment_dir.mkdir(parents=True)
        policy_state[(threshold, cap)] = {
            "name": _policy_name(threshold, cap),
            "threshold": float(threshold),
            "per_video_cap": "all" if cap is None else int(cap),
            "fragment_root": str(fragment_dir.relative_to(output_dir)),
            "selected_action_count": 0,
            "files": [],
            "fragment_dir_path": fragment_dir,
        }
    # Candidate banks and the all-threshold greedy sequence are loaded once per
    # video even when the explicit 114-row full grid is requested.
    for record in records:
        deadline.check(f"integrate video {record.identity}")
        arrays = _load_candidate_npz(
            _candidate_path(manifest_path, manifest, record.identity), record.graph
        )
        greedy = greedy_selected_indices(record.graph, arrays)
        indexed = _edge_index(record.graph)
        for threshold, cap in policies:
            deadline.check(f"integrate policies for {record.identity}")
            state = policy_state[(threshold, cap)]
            selected = policy_indices(arrays, greedy, threshold=threshold, cap=cap)
            actions = [
                _action_for_ids(
                    record.graph,
                    *arrays.row(int(index))[:3],
                    edge_index=indexed,
                )
                for index in selected
            ]
            if actions:
                output_graph, batch_hash_value = _compile_disjoint_actions(
                    record.graph, actions
                )
                batch_hash: str | None = batch_hash_value
            else:
                output_graph = record.graph.copy()
                batch_hash = None
            validate_graphs({record.dataset: output_graph})
            destination = Path(state["fragment_dir_path"]) / f"{record.split}__{record.dataset}.csv"
            fragment_hash = _write_new_submission(destination, output_graph)
            reread = read_submission(destination)
            if graph_sha256(reread[record.dataset]) != graph_sha256(output_graph):
                raise EvaluationError(f"fragment semantic round-trip failed: {record.identity}")
            state["selected_action_count"] += len(actions)
            state["files"].append(
                {
                    "identity": record.identity,
                    "fragment": str(destination.relative_to(output_dir)),
                    "fragment_sha256": fragment_hash,
                    "baseline_graph_sha256": graph_sha256(record.graph),
                    "candidate_graph_sha256": graph_sha256(output_graph),
                    "selected_action_count": len(actions),
                    "selected_candidate_keys": [
                        _candidate_key(record.dataset, *arrays.row(int(index))[:3])
                        for index in selected
                    ],
                    "batch_sha256": batch_hash,
                }
            )
        deadline.check(f"integrate video {record.identity} completion")
    policy_reports: list[dict[str, Any]] = []
    for policy in policies:
        state = policy_state[policy]
        state.pop("fragment_dir_path")
        policy_reports.append(state)
    report = {
        "schema": INTEGRATION_SCHEMA,
        "node": NODE,
        "status": "FRONTIER_FRAGMENTS_READY_FOR_EXACT_OFFICIAL_SCORER",
        "runtime_cap_seconds": float(runtime_cap_seconds),
        "contains_labels": False,
        "selection_uses_scores_only": True,
        "score_manifest": str(manifest_path.resolve()),
        "score_manifest_sha256": sha256_file(manifest_path),
        "selection": (
            "descending score; deterministic candidate-key ties; exact node-disjoint and "
            "incident-conflict-disjoint ownership; atomic division_fork compilation"
        ),
        "policies": policy_reports,
    }
    deadline.check("integration manifest publication")
    _write_new_json(output_dir / "INTEGRATION_MANIFEST.json", report)
    return report


def _parse_cap(value: str) -> int | None:
    if value == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cap must be a positive integer or 'all'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("cap must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="seal truth-blind real graph scores")
    generate.add_argument("--model-json", type=Path, required=True)
    generate.add_argument("--fragments", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--registration-census",
        type=Path,
        default=DEFAULT_REGISTRATION_CENSUS,
    )
    generate.add_argument(
        "--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS
    )
    generate.add_argument("--workers", type=int, default=1)

    evaluate = subparsers.add_parser(
        "evaluate", help="authenticate frozen scores, then open real GT and report the full grid"
    )
    evaluate.add_argument("--scores-manifest", type=Path, required=True)
    evaluate.add_argument("--gt-archive", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, action="append")
    evaluate.add_argument("--cap", type=_parse_cap, action="append")
    evaluate.add_argument(
        "--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS
    )

    integrate = subparsers.add_parser(
        "integrate", help="compile selected score-only policies into safe fragment copies"
    )
    integrate.add_argument("--scores-manifest", type=Path, required=True)
    integrate.add_argument("--output-dir", type=Path, required=True)
    integrate.add_argument("--threshold", type=float, action="append")
    integrate.add_argument("--cap", type=_parse_cap, action="append")
    integrate.add_argument(
        "--full-grid",
        action="store_true",
        help="explicitly emit all 114 preregistered policies (large output)",
    )
    integrate.add_argument(
        "--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_scores(
            args.model_json,
            args.fragments,
            args.output_dir,
            registration_census=args.registration_census,
            runtime_cap_seconds=args.runtime_cap_seconds,
            workers=args.workers,
        )
        print(json.dumps({"status": result["status"], "candidate_count": result["candidate_count"]}))
        return 0
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    caps = tuple(args.cap) if args.cap else DEFAULT_CAPS
    if args.command == "evaluate":
        result = evaluate_scores(
            args.scores_manifest,
            args.gt_archive,
            args.output,
            thresholds=thresholds,
            caps=caps,
            runtime_cap_seconds=args.runtime_cap_seconds,
        )
        print(json.dumps({"status": result["status"], **result["ranking"]}))
        return 0
    if args.command == "integrate":
        if not args.full_grid and (args.threshold is None or args.cap is None):
            raise EvaluationError(
                "integration requires explicit --threshold and --cap, or --full-grid"
            )
        policies = _policy_grid(
            DEFAULT_THRESHOLDS if args.full_grid else thresholds,
            DEFAULT_CAPS if args.full_grid else caps,
        )
        result = integrate_scores(
            args.scores_manifest,
            args.output_dir,
            policies=policies,
            runtime_cap_seconds=args.runtime_cap_seconds,
        )
        print(json.dumps({"status": result["status"], "policy_count": len(result["policies"])}))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

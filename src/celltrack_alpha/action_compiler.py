"""Deterministic, feasibility-exact graph action compilation.

The compiler is intentionally independent of labels, model scores, and the official
metric.  A fully specified action owns every incident conflict that it replaces.
Every action must be valid alone against the frozen baseline, and a batch is accepted
only when the union of its primitives is unambiguous and graph-valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from celltrack_alpha.kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    validate_graphs,
)


INTERFACE_SCHEMA_VERSION = 1
_ACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ActionCompileError(ValueError):
    """An action or batch violates the frozen feasibility contract."""


class ActionKind(str, Enum):
    NOOP = "noop"
    ADD_EDGE = "add_edge"
    DELETE_EDGES = "delete_edges"
    SWAP = "swap"
    DIVISION_FORK = "division_fork"
    RESTORE_TOPOLOGY = "restore_topology"


EdgePair = tuple[int, int]


@dataclass(frozen=True)
class GraphAction:
    """One atomic, self-contained graph mutation.

    ``expected_conflict_edges`` must exactly enumerate every baseline edge whose
    source is in ``replace_outgoing_from`` or whose target is in
    ``replace_incoming_to``.  This is the fail-closed incident-conflict proof.
    """

    action_id: str
    kind: ActionKind
    add_nodes: tuple[SubmissionNode, ...] = field(default_factory=tuple)
    add_edges: tuple[EdgePair, ...] = field(default_factory=tuple)
    delete_edges: tuple[EdgePair, ...] = field(default_factory=tuple)
    replace_outgoing_from: tuple[int, ...] = field(default_factory=tuple)
    replace_incoming_to: tuple[int, ...] = field(default_factory=tuple)
    expected_conflict_edges: tuple[EdgePair, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompiledAction:
    action: GraphAction
    graph: SubmissionGraph
    action_sha256: str
    before_graph_sha256: str
    after_graph_sha256: str
    removed_edges: tuple[EdgePair, ...]


@dataclass(frozen=True)
class CompiledBatch:
    actions: tuple[GraphAction, ...]
    graph: SubmissionGraph
    batch_sha256: str
    before_graph_sha256: str
    after_graph_sha256: str
    action_sha256s: tuple[str, ...]
    removed_edges: tuple[EdgePair, ...]


def _edge_set(graph: SubmissionGraph) -> set[EdgePair]:
    return {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}


def _unique(values: Iterable[Any], label: str) -> tuple[Any, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ActionCompileError(f"{label} contains duplicates")
    return result


def _node_payload(node: SubmissionNode) -> dict[str, int]:
    values = {
        "node_id": int(node.node_id),
        "t": int(node.t),
        "z": int(node.z),
        "y": int(node.y),
        "x": int(node.x),
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ActionCompileError("node fields must be finite integers")
    return values


def _canonical_action_payload(action: GraphAction) -> dict[str, Any]:
    return {
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "action_id": action.action_id,
        "kind": action.kind.value,
        "add_nodes": sorted(
            (_node_payload(node) for node in action.add_nodes),
            key=lambda row: row["node_id"],
        ),
        "add_edges": [list(pair) for pair in sorted(action.add_edges)],
        "delete_edges": [list(pair) for pair in sorted(action.delete_edges)],
        "replace_outgoing_from": sorted(action.replace_outgoing_from),
        "replace_incoming_to": sorted(action.replace_incoming_to),
        "expected_conflict_edges": [
            list(pair) for pair in sorted(action.expected_conflict_edges)
        ],
    }


def action_sha256(action: GraphAction) -> str:
    payload = json.dumps(
        _canonical_action_payload(action),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def graph_sha256(graph: SubmissionGraph) -> str:
    validate_graphs({graph.dataset: graph})
    payload = {
        "dataset": graph.dataset,
        "nodes": sorted(
            (_node_payload(node) for node in graph.nodes.values()),
            key=lambda row: row["node_id"],
        ),
        "edges": [list(pair) for pair in sorted(_edge_set(graph))],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_kind(action: GraphAction, baseline: SubmissionGraph) -> None:
    has_mutation = any(
        (
            action.add_nodes,
            action.add_edges,
            action.delete_edges,
            action.replace_outgoing_from,
            action.replace_incoming_to,
            action.expected_conflict_edges,
        )
    )
    if action.kind is ActionKind.NOOP:
        if has_mutation:
            raise ActionCompileError("noop action cannot contain mutations")
        return
    if action.kind is ActionKind.ADD_EDGE:
        if (
            not action.add_edges
            or action.add_nodes
            or action.delete_edges
            or action.replace_outgoing_from
            or action.replace_incoming_to
            or action.expected_conflict_edges
        ):
            raise ActionCompileError("add_edge must contain only added edges")
        edges = _edge_set(baseline)
        outgoing = {source for source, _target in edges}
        incoming = {target for _source, target in edges}
        if any(source in outgoing or target in incoming for source, target in action.add_edges):
            raise ActionCompileError("add_edge requires free source and target endpoints")
        return
    if action.kind is ActionKind.DELETE_EDGES:
        if (
            not action.delete_edges
            or action.add_nodes
            or action.add_edges
            or action.replace_outgoing_from
            or action.replace_incoming_to
            or action.expected_conflict_edges
        ):
            raise ActionCompileError("delete_edges must contain only explicit deletions")
        return
    if action.kind in (ActionKind.SWAP, ActionKind.DIVISION_FORK):
        if action.add_nodes or action.delete_edges or not action.add_edges:
            raise ActionCompileError(f"{action.kind.value} has an invalid primitive set")
        added_sources = {source for source, _target in action.add_edges}
        added_targets = {target for _source, target in action.add_edges}
        if added_sources != set(action.replace_outgoing_from):
            raise ActionCompileError("replacement sources must equal added-edge sources")
        if added_targets != set(action.replace_incoming_to):
            raise ActionCompileError("replacement targets must equal added-edge targets")
        if action.kind is ActionKind.DIVISION_FORK:
            if len(action.add_edges) != 2 or len(added_sources) != 1 or len(added_targets) != 2:
                raise ActionCompileError("division_fork requires one parent and two daughters")
        return
    if action.kind is ActionKind.RESTORE_TOPOLOGY:
        if not action.add_nodes or not action.add_edges or action.delete_edges:
            raise ActionCompileError(
                "restore_topology requires nodes and edges and no explicit deletions"
            )
        restored = {int(node.node_id) for node in action.add_nodes}
        incident = {
            endpoint
            for pair in action.add_edges
            for endpoint in pair
            if endpoint in restored
        }
        if incident != restored:
            raise ActionCompileError("every restored node must have added topology")
        return
    raise ActionCompileError(f"unsupported action kind: {action.kind!r}")


def _prepare_action(
    baseline: SubmissionGraph, action: GraphAction
) -> tuple[set[int], set[EdgePair], set[EdgePair], set[EdgePair]]:
    validate_graphs({baseline.dataset: baseline})
    if not _ACTION_ID.fullmatch(action.action_id):
        raise ActionCompileError(f"invalid action_id: {action.action_id!r}")
    if not isinstance(action.kind, ActionKind):
        raise ActionCompileError("kind must be an ActionKind")

    added_node_ids = tuple(int(node.node_id) for node in action.add_nodes)
    _unique(added_node_ids, "add_nodes")
    add_edges = set(_unique(action.add_edges, "add_edges"))
    delete_edges = set(_unique(action.delete_edges, "delete_edges"))
    replace_sources = set(
        _unique(action.replace_outgoing_from, "replace_outgoing_from")
    )
    replace_targets = set(
        _unique(action.replace_incoming_to, "replace_incoming_to")
    )
    expected_conflicts = set(
        _unique(action.expected_conflict_edges, "expected_conflict_edges")
    )

    baseline_nodes = set(baseline.nodes)
    if baseline_nodes & set(added_node_ids):
        raise ActionCompileError("added node ID already exists in baseline")
    if not replace_sources <= baseline_nodes or not replace_targets <= baseline_nodes:
        raise ActionCompileError("replacement boundary references a missing baseline node")

    baseline_edges = _edge_set(baseline)
    if not delete_edges <= baseline_edges:
        raise ActionCompileError("explicit deletion references a missing baseline edge")
    actual_conflicts = {
        pair
        for pair in baseline_edges
        if pair[0] in replace_sources or pair[1] in replace_targets
    }
    if expected_conflicts != actual_conflicts:
        missing = sorted(actual_conflicts - expected_conflicts)
        extra = sorted(expected_conflicts - actual_conflicts)
        raise ActionCompileError(
            f"incident-conflict proof mismatch; missing={missing}, extra={extra}"
        )

    final_nodes = baseline_nodes | set(added_node_ids)
    if any(source not in final_nodes or target not in final_nodes for source, target in add_edges):
        raise ActionCompileError("added edge has a missing final endpoint")
    if any(source == target for source, target in add_edges):
        raise ActionCompileError("self-edge is forbidden")

    _validate_kind(action, baseline)
    return set(added_node_ids), add_edges, delete_edges, expected_conflicts


def _make_graph(
    baseline: SubmissionGraph,
    add_nodes: Sequence[SubmissionNode],
    add_edges: set[EdgePair],
    remove_edges: set[EdgePair],
) -> SubmissionGraph:
    nodes = dict(baseline.nodes)
    nodes.update((int(node.node_id), node) for node in add_nodes)
    edges = (_edge_set(baseline) - remove_edges) | add_edges
    graph = SubmissionGraph(
        dataset=baseline.dataset,
        nodes=nodes,
        edges=[SubmissionEdge(source, target) for source, target in sorted(edges)],
    )
    validate_graphs({graph.dataset: graph})
    return graph


def compile_action(baseline: SubmissionGraph, action: GraphAction) -> CompiledAction:
    """Compile one action transaction or reject it without a partial graph."""

    _added_ids, add_edges, delete_edges, expected_conflicts = _prepare_action(
        baseline, action
    )
    removed = delete_edges | expected_conflicts
    graph = _make_graph(baseline, action.add_nodes, add_edges, removed)
    before = graph_sha256(baseline)
    after = graph_sha256(graph)
    return CompiledAction(
        action=action,
        graph=graph,
        action_sha256=action_sha256(action),
        before_graph_sha256=before,
        after_graph_sha256=after,
        removed_edges=tuple(sorted(removed)),
    )


def compile_batch(
    baseline: SubmissionGraph, actions: Sequence[GraphAction]
) -> CompiledBatch:
    """Compile an order-independent batch whose actions are each valid alone."""

    if not actions:
        raise ActionCompileError("batch must contain at least one explicit action")
    ids = [action.action_id for action in actions]
    _unique(ids, "batch action IDs")
    ordered = tuple(sorted(actions, key=lambda action: action.action_id))

    added_node_owner: dict[int, str] = {}
    added_edge_owner: dict[EdgePair, str] = {}
    removed_edge_owner: dict[EdgePair, str] = {}
    source_owner: dict[int, str] = {}
    target_owner: dict[int, str] = {}
    all_nodes: list[SubmissionNode] = []
    all_add_edges: set[EdgePair] = set()
    all_remove_edges: set[EdgePair] = set()
    action_hashes: list[str] = []

    for action in ordered:
        compiled = compile_action(baseline, action)
        added_ids, add_edges, delete_edges, expected_conflicts = _prepare_action(
            baseline, action
        )
        removed = delete_edges | expected_conflicts
        for node_id in added_ids:
            if node_id in added_node_owner:
                raise ActionCompileError(
                    f"node {node_id} added by both {added_node_owner[node_id]} and {action.action_id}"
                )
            added_node_owner[node_id] = action.action_id
        for pair in add_edges:
            if pair in added_edge_owner:
                raise ActionCompileError(
                    f"edge {pair} added by both {added_edge_owner[pair]} and {action.action_id}"
                )
            if pair in removed_edge_owner:
                raise ActionCompileError(f"edge {pair} is removed by another action")
            added_edge_owner[pair] = action.action_id
        for pair in removed:
            if pair in removed_edge_owner:
                raise ActionCompileError(
                    f"edge {pair} removed by both {removed_edge_owner[pair]} and {action.action_id}"
                )
            if pair in added_edge_owner:
                raise ActionCompileError(f"edge {pair} is added by another action")
            removed_edge_owner[pair] = action.action_id
        for source in action.replace_outgoing_from:
            if source in source_owner:
                raise ActionCompileError(
                    f"outgoing boundary {source} owned by multiple actions"
                )
            source_owner[source] = action.action_id
        for target in action.replace_incoming_to:
            if target in target_owner:
                raise ActionCompileError(
                    f"incoming boundary {target} owned by multiple actions"
                )
            target_owner[target] = action.action_id

        all_nodes.extend(action.add_nodes)
        all_add_edges.update(add_edges)
        all_remove_edges.update(removed)
        action_hashes.append(compiled.action_sha256)

    graph = _make_graph(
        baseline,
        tuple(all_nodes),
        all_add_edges,
        all_remove_edges,
    )
    before = graph_sha256(baseline)
    after = graph_sha256(graph)
    batch_payload = {
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "before_graph_sha256": before,
        "action_sha256s": action_hashes,
        "after_graph_sha256": after,
    }
    batch_hash = hashlib.sha256(
        json.dumps(
            batch_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return CompiledBatch(
        actions=ordered,
        graph=graph,
        batch_sha256=batch_hash,
        before_graph_sha256=before,
        after_graph_sha256=after,
        action_sha256s=tuple(action_hashes),
        removed_edges=tuple(sorted(all_remove_edges)),
    )


def interface_schema() -> Mapping[str, Any]:
    """Return the frozen, JSON-serializable public interface contract."""

    return {
        "schema": "celltrack-alpha.feasibility-exact-action-compiler.v1",
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "action_kinds": [kind.value for kind in ActionKind],
        "graph_action_fields": [
            "action_id",
            "kind",
            "add_nodes",
            "add_edges",
            "delete_edges",
            "replace_outgoing_from",
            "replace_incoming_to",
            "expected_conflict_edges",
        ],
        "invariants": [
            "baseline graph validates before compilation",
            "every action validates alone against the frozen baseline",
            "expected conflict edges exactly equal the declared incident boundary",
            "all final edge endpoints exist",
            "all final edges point forward in time",
            "final indegree is at most one (no merges)",
            "final outdegree is at most two",
            "batch action IDs and primitive ownership are unique",
            "batch result is independent of input action order",
            "no partial graph is returned on failure",
        ],
        "hashing": {
            "algorithm": "sha256",
            "encoding": "canonical sorted compact ASCII JSON",
            "identities": ["action", "baseline graph", "result graph", "batch"],
        },
    }


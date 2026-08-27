"""Deterministic, graph-only reference contract for D1-PERSISTENT-FORK.

This is deliberately not a trained model.  It freezes the motif grammar and the
stable one-to-one suppression that a later compact frozen-feature head must feed.
It never reads held outcomes or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Edge = tuple[int, int]


@dataclass(frozen=True)
class ForkCandidate:
    parent: int
    child_a: int
    child_b: int
    grandchild_a: int | None
    grandchild_b: int | None
    score: float
    equivalence_key: str

    @property
    def children(self) -> tuple[int, int]:
        return tuple(sorted((self.child_a, self.child_b)))

    @property
    def grandchildren(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                node
                for node in (self.grandchild_a, self.grandchild_b)
                if node is not None
            )
        )

    @property
    def persistent(self) -> bool:
        return self.grandchild_a is not None and self.grandchild_b is not None

    @property
    def occupied_nodes(self) -> frozenset[int]:
        return frozenset((self.parent, *self.children, *self.grandchildren))


@dataclass(frozen=True)
class SuppressionRecord:
    candidate: ForkCandidate
    reason: str
    winner_key: str


def _rank_key(candidate: ForkCandidate) -> tuple[float, int, tuple[int, int], tuple[int, ...], str]:
    """Total ordering: model score first, then stable graph/event identity."""

    return (
        -float(candidate.score),
        int(candidate.parent),
        candidate.children,
        candidate.grandchildren,
        candidate.equivalence_key,
    )


def _validate_candidate(candidate: ForkCandidate) -> None:
    if candidate.child_a == candidate.child_b:
        raise ValueError("fork children must be distinct")
    if candidate.parent in candidate.children:
        raise ValueError("fork parent cannot also be a child")
    if candidate.grandchild_a is not None and candidate.grandchild_a == candidate.child_a:
        raise ValueError("branch A grandchild cannot equal its child")
    if candidate.grandchild_b is not None and candidate.grandchild_b == candidate.child_b:
        raise ValueError("branch B grandchild cannot equal its child")
    if not candidate.equivalence_key:
        raise ValueError("equivalence key must be non-empty")


def suppress_candidates(
    candidates: Iterable[ForkCandidate],
    *,
    require_persistence: bool,
) -> tuple[tuple[ForkCandidate, ...], tuple[SuppressionRecord, ...]]:
    """Select one-to-one scorer-equivalence events before graph emission."""

    accepted: list[ForkCandidate] = []
    suppressed: list[SuppressionRecord] = []
    used_equivalence_keys: set[str] = set()
    occupied: set[int] = set()

    for candidate in sorted(candidates, key=_rank_key):
        _validate_candidate(candidate)
        if require_persistence and not candidate.persistent:
            suppressed.append(SuppressionRecord(candidate, "missing_branch_persistence", ""))
            continue
        if candidate.equivalence_key in used_equivalence_keys:
            winner = next(
                row for row in accepted if row.equivalence_key == candidate.equivalence_key
            )
            suppressed.append(
                SuppressionRecord(candidate, "duplicate_equivalence_class", winner.equivalence_key)
            )
            continue
        if candidate.occupied_nodes.intersection(occupied):
            winner = next(row for row in accepted if row.occupied_nodes.intersection(candidate.occupied_nodes))
            suppressed.append(
                SuppressionRecord(candidate, "one_to_one_node_conflict", winner.equivalence_key)
            )
            continue
        accepted.append(candidate)
        used_equivalence_keys.add(candidate.equivalence_key)
        occupied.update(candidate.occupied_nodes)

    return tuple(accepted), tuple(suppressed)


def emit_forks(base_edges: Iterable[Edge], selected: Iterable[ForkCandidate]) -> tuple[Edge, ...]:
    """Emit only the two parent-child edges; persistence edges are evidence, not invented."""

    output = set((int(source), int(target)) for source, target in base_edges)
    for candidate in selected:
        output.add((candidate.parent, candidate.child_a))
        output.add((candidate.parent, candidate.child_b))
    return tuple(sorted(output))


def validate_graph(nodes_t: dict[int, int], edges: Iterable[Edge]) -> None:
    """Fail closed on the D1 graph invariants."""

    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    seen: set[Edge] = set()
    for source, target in edges:
        if (source, target) in seen:
            raise ValueError("duplicate edge")
        seen.add((source, target))
        if source not in nodes_t or target not in nodes_t:
            raise ValueError("dangling endpoint")
        if nodes_t[target] <= nodes_t[source]:
            raise ValueError("edge must point forward in time")
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
    if max(incoming.values(), default=0) > 1:
        raise ValueError("indegree exceeds one")
    if max(outgoing.values(), default=0) > 2:
        raise ValueError("outdegree exceeds two")


def duplicate_survivors(selected: Iterable[ForkCandidate]) -> int:
    keys = [candidate.equivalence_key for candidate in selected]
    return len(keys) - len(set(keys))

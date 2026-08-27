"""AE-P5: frozen C0/C2-KBEST-STABILITY assignment rule over occupied conflicts.

Implements `AE_P5_C0_ASSIGNMENT_CONTRACT.json` exactly, under the human
implementation authorization of 2026-07-30. Scope is unchanged otherwise:
fit-only 44b6, held labels forbidden, no graph run, no 6bba, no CV57.

C0 (base rule)
    unit        one complete N1-valid occupied-conflict action
    component   transitive closure of actions sharing any owned source or
                destination endpoint
    feasible    a subset of component actions with pairwise-disjoint N1
                ownership; the empty subset is the explicit no-op
    evidence    the authenticated 49th feature, b0[:, 48]: normalized frozen
                Trackastra challenger-minus-incumbent margin for the conflict row
    cost        negative sum of the action's margins; no-op costs exactly 0.0
    MAP         minimum total float64 cost, tie-broken by (total_cost, then the
                lexicographically sorted tuple of canonical N1 action SHA-256)

C2 (fixed stability rule)
    k=8 best feasible assignments ordered by (total_cost asc, canonical
    assignment tuple asc). An edit may be introduced only with >=7/8 membership
    and removed only with <=1/8 membership, subject to the min-marginal
    tolerance. The whole component falls back byte-semantically to APNCC if
    fewer than eight feasible assignments exist, or the conjunction fails.

Non-finite or missing evidence sends the whole component back to APNCC.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from celltrack_alpha.action_compiler import (
        ActionCompileError,
        ActionKind,
        GraphAction,
        action_sha256,
        compile_batch,
    )
    from celltrack_alpha.kaggle_calibration import SubmissionGraph, read_submission
except ModuleNotFoundError as exc:  # pragma: no cover - CLI environment guard
    raise RuntimeError(
        "AE-P5 requires the repository src/ package on PYTHONPATH (PYTHONPATH=src)."
    ) from exc

K_BEST = 8
INTRODUCE_MIN_MEMBERSHIP = 7
REMOVE_MAX_MEMBERSHIP = 1
NO_OP_COST = 0.0
MARGIN_FEATURE_INDEX = 48  # the authenticated 49th feature
B0_FEATURE_WIDTH = 49


class AeP5Error(ValueError):
    """Raised when the frozen contract cannot be honoured."""


@dataclass(frozen=True)
class Action:
    """One occupied-conflict action, keyed by the endpoints it owns."""

    video: str
    row: int
    source_id: int
    target_id: int
    source_t: int
    target_t: int
    margin: float
    incumbent_edges: tuple[tuple[int, int], ...] = ()

    @property
    def cost(self) -> float:
        """Negative sum of the action's challenger-minus-incumbent margins."""
        return -float(self.margin)

    @property
    def owned(self) -> tuple[tuple[object, ...], ...]:
        """All N1 primitives named in the frozen C0 contract.

        The prior implementation used only source/destination slots.  A complete
        occupied swap also owns each incumbent edge it removes, its challenger
        edge, and the explicit no-op endpoint for that challenger.  The latter is
        intentionally represented separately even though rows are deduplicated by
        challenger pair: it keeps the empty/no-op alternative explicit in the
        component identity without inventing evidence or a forward-derived score.
        """
        return (
            ("source_outgoing", self.source_id),
            ("destination_incoming", self.target_id),
            *(("incumbent_edge", source, target) for source, target in self.incumbent_edges),
            ("challenger_edge", self.source_id, self.target_id),
            ("no_op_endpoint", self.source_id, self.target_id),
        )

    def graph_action(self) -> GraphAction:
        """The exact frozen N1 action whose ownership this row represents."""
        return GraphAction(
            action_id=f"swap:{self.source_id}:{self.target_id}",
            kind=ActionKind.SWAP,
            add_edges=((self.source_id, self.target_id),),
            replace_outgoing_from=(self.source_id,),
            replace_incoming_to=(self.target_id,),
            expected_conflict_edges=self.incumbent_edges,
        )

    def canonical_sha256(self) -> str:
        """Frozen N1 canonical action digest, including all incumbent conflicts."""
        return action_sha256(self.graph_action())


def _baseline_edges(graph: SubmissionGraph) -> tuple[set[tuple[int, int]], dict[int, tuple[tuple[int, int], ...]], dict[int, tuple[tuple[int, int], ...]]]:
    edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    outgoing: dict[int, list[tuple[int, int]]] = {}
    incoming: dict[int, list[tuple[int, int]]] = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append((source, target))
        incoming.setdefault(target, []).append((source, target))
    return (
        edges,
        {key: tuple(sorted(value)) for key, value in outgoing.items()},
        {key: tuple(sorted(value)) for key, value in incoming.items()},
    )


def load_actions(shard: Path, baseline: SubmissionGraph) -> list[Action]:
    """Read one frozen population shard and restore its complete N1 boundary.

    The population carries the exact candidate endpoints and frozen margin but not
    the incumbent edges.  Those are deterministically recovered from the pinned
    APNCC control graph: all and only edges with the candidate source or target at
    the replacement boundary.  No oracle labels are read.
    """
    baseline_edges, outgoing, incoming = _baseline_edges(baseline)
    with np.load(shard, allow_pickle=False) as data:
        for field in ("source_id", "target_id", "source_t", "target_t", "b0"):
            if field not in data.files:
                raise AeP5Error(f"{shard.name} is missing frozen field {field}")
        b0 = data["b0"]
        if b0.ndim != 2 or b0.shape[1] != B0_FEATURE_WIDTH:
            raise AeP5Error(
                f"{shard.name} b0 width is {b0.shape}, expected (n, {B0_FEATURE_WIDTH})"
            )
        video = shard.stem.replace("__", ":", 1)
        actions = []
        for index in range(len(b0)):
            source = int(data["source_id"][index])
            target = int(data["target_id"][index])
            source_t = int(data["source_t"][index])
            target_t = int(data["target_t"][index])
            if (
                source not in baseline.nodes
                or target not in baseline.nodes
                or int(baseline.nodes[source].t) != source_t
                or int(baseline.nodes[target].t) != target_t
                or target_t != source_t + 1
                or (source, target) in baseline_edges
                or len(outgoing.get(source, ())) != 1
                or len(incoming.get(target, ())) != 1
            ):
                raise AeP5Error(
                    f"{shard.name} row {index} does not match the frozen occupied-swap boundary"
                )
            incumbent_edges = tuple(
                sorted(set(outgoing[source]) | set(incoming[target]))
            )
            if len(incumbent_edges) not in (1, 2):
                raise AeP5Error(
                    f"{shard.name} row {index} has malformed incumbent boundary"
                )
            actions.append(
                Action(
                    video=video,
                    row=index,
                    source_id=source,
                    target_id=target,
                    source_t=source_t,
                    target_t=target_t,
                    margin=float(b0[index, MARGIN_FEATURE_INDEX]),
                    incumbent_edges=incumbent_edges,
                )
            )
        return actions


def components(actions: Sequence[Action]) -> list[list[Action]]:
    """Transitive closure over shared owned endpoints."""
    parent = list(range(len(actions)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    first_seen: dict[tuple[object, ...], int] = {}
    for index, action in enumerate(actions):
        for slot in action.owned:
            if slot in first_seen:
                union(first_seen[slot], index)
            else:
                first_seen[slot] = index

    grouped: dict[int, list[Action]] = {}
    for index, action in enumerate(actions):
        grouped.setdefault(find(index), []).append(action)
    return [
        sorted(group, key=lambda item: item.canonical_sha256())
        for group in grouped.values()
    ]


def _disjoint(chosen: Iterable[Action]) -> bool:
    slots: set[tuple[object, ...]] = set()
    for action in chosen:
        for slot in action.owned:
            if slot in slots:
                return False
            slots.add(slot)
    return True


def feasible_assignments(component: Sequence[Action]) -> list[tuple[Action, ...]]:
    """Every pairwise-disjoint subset, including the explicit empty no-op."""
    found: list[tuple[Action, ...]] = []
    for size in range(len(component) + 1):
        for subset in combinations(component, size):
            if _disjoint(subset):
                found.append(subset)
    return found


def _assignment_key(subset: Sequence[Action]) -> tuple[float, tuple[str, ...]]:
    total = NO_OP_COST + math.fsum(action.cost for action in subset)
    return total, tuple(sorted(action.canonical_sha256() for action in subset))


def _min_marginal_tolerance(best_cost: float) -> float:
    return 100.0 * float(np.finfo(np.float64).eps) * max(1.0, abs(best_cost))


def _compiled_feasible_assignments(
    baseline: SubmissionGraph, raw_assignments: Sequence[tuple[Action, ...]]
) -> tuple[list[tuple[Action, ...]], int]:
    """Filter pairwise-disjoint candidates through frozen N1 batch compilation."""
    feasible: list[tuple[Action, ...]] = []
    rejected = 0
    for subset in raw_assignments:
        if not subset:
            feasible.append(subset)
            continue
        try:
            compile_batch(baseline, [action.graph_action() for action in subset])
        except ActionCompileError:
            rejected += 1
        else:
            feasible.append(subset)
    return feasible, rejected


def solve_component(
    component: Sequence[Action], baseline: SubmissionGraph | None = None
) -> dict[str, Any]:
    """Apply C0 then the C2 fixed stability rule to one component."""
    if any(not math.isfinite(action.margin) for action in component):
        return {
            "size": len(component),
            "decision": "APNCC_FALLBACK",
            "reason": "missing_or_nonfinite_evidence",
            "selected": (),
            "feasible_assignment_count": None,
        }

    raw_assignments = feasible_assignments(component)
    # When the pairwise ownership upper bound is below k, N1 compilation cannot
    # restore enough assignments.  This is the cheapest sound falsifier for the
    # overwhelming majority of components.
    if len(raw_assignments) < K_BEST:
        ranked = sorted(raw_assignments, key=_assignment_key)
        best = ranked[0]
        best_cost, _ = _assignment_key(best)
        record = {
            "size": len(component),
            "c0_map_cost": best_cost,
            "c0_map_size": len(best),
            "decision": "APNCC_FALLBACK",
            "selected": (),
            "n1_batch_compiles": 0,
            "n1_batch_rejections": 0,
        }
        if baseline is None:
            record.update(
                {
                    "feasible_assignment_count": len(raw_assignments),
                    "reason": "fewer_than_eight_feasible_assignments",
                }
            )
        else:
            record.update(
                {
                    "feasible_assignment_count_upper_bound": len(raw_assignments),
                    "reason": "fewer_than_eight_feasible_assignments_upper_bound",
                }
            )
        return record

    if baseline is None:
        # Unit tests exercise the pure combinatorial rule; production always
        # supplies the authenticated baseline and gets this additional check.
        assignments = list(raw_assignments)
        batch_rejections = 0
    else:
        assignments, batch_rejections = _compiled_feasible_assignments(
            baseline, raw_assignments
        )
    ranked = sorted(assignments, key=_assignment_key)
    best = ranked[0]
    best_cost, _ = _assignment_key(best)

    # C0 MAP is defined regardless; C2 decides whether it may be applied.
    record: dict[str, Any] = {
        "size": len(component),
        "feasible_assignment_count": len(assignments),
        "pairwise_feasible_assignment_upper_bound": len(raw_assignments),
        "n1_batch_compiles": len(raw_assignments) - batch_rejections - 1,
        "n1_batch_rejections": batch_rejections,
        "c0_map_cost": best_cost,
        "c0_map_size": len(best),
    }

    if len(assignments) < K_BEST:
        record.update(
            {
                "decision": "APNCC_FALLBACK",
                "reason": "fewer_than_eight_feasible_assignments",
                "selected": (),
            }
        )
        return record

    top = ranked[:K_BEST]
    tolerance = _min_marginal_tolerance(best_cost)
    membership: dict[str, int] = {}
    for subset in top:
        for action in subset:
            membership[action.canonical_sha256()] = (
                membership.get(action.canonical_sha256(), 0) + 1
            )

    selected = []
    for action in component:
        digest = action.canonical_sha256()
        count = membership.get(digest, 0)
        in_best = action in best
        if in_best and count < INTRODUCE_MIN_MEMBERSHIP:
            record.update(
                {
                    "decision": "APNCC_FALLBACK",
                    "reason": "introduce_membership_below_seven_eighths",
                    "selected": (),
                }
            )
            return record
        if not in_best and count > REMOVE_MAX_MEMBERSHIP:
            record.update(
                {
                    "decision": "APNCC_FALLBACK",
                    "reason": "remove_membership_above_one_eighth",
                    "selected": (),
                }
            )
            return record
        if in_best:
            without = tuple(item for item in best if item is not action)
            marginal = _assignment_key(without)[0] - best_cost
            if marginal <= tolerance:
                record.update(
                    {
                        "decision": "APNCC_FALLBACK",
                        "reason": "min_marginal_within_tolerance",
                        "selected": (),
                    }
                )
                return record
            selected.append(action)

    record.update({"decision": "APPLY_RANK1", "reason": None, "selected": tuple(selected)})
    return record


def run(population_root: Path, control_root: Path) -> dict[str, Any]:
    shards = sorted(population_root.glob("*.npz"))
    if not shards:
        raise AeP5Error(f"no frozen population shards under {population_root}")

    totals = {
        "shards": len(shards),
        "actions": 0,
        "components": 0,
        "components_applied": 0,
        "components_fallback": 0,
        "actions_selected": 0,
        "n1_batch_compiles": 0,
        "n1_batch_rejections": 0,
    }
    reasons: dict[str, int] = {}
    size_histogram: dict[int, int] = {}
    feasible_histogram: dict[int, int] = {}
    selected_rows: list[dict[str, Any]] = []

    for shard in shards:
        control_path = control_root / f"{shard.stem}.csv"
        if not control_path.is_file():
            raise AeP5Error(f"missing pinned control fragment: {control_path}")
        graphs = read_submission(control_path)
        if len(graphs) != 1:
            raise AeP5Error(f"control fragment is not a single graph: {control_path}")
        baseline = next(iter(graphs.values()))
        actions = load_actions(shard, baseline)
        totals["actions"] += len(actions)
        for component in components(actions):
            totals["components"] += 1
            size_histogram[len(component)] = size_histogram.get(len(component), 0) + 1
            record = solve_component(component, baseline)
            totals["n1_batch_compiles"] += int(record.get("n1_batch_compiles", 0))
            totals["n1_batch_rejections"] += int(record.get("n1_batch_rejections", 0))
            count = record.get("feasible_assignment_count")
            if count is not None:
                feasible_histogram[count] = feasible_histogram.get(count, 0) + 1
            if record["decision"] == "APPLY_RANK1":
                totals["components_applied"] += 1
                for action in record["selected"]:
                    totals["actions_selected"] += 1
                    selected_rows.append(
                        {
                            "video": action.video,
                            "source_id": action.source_id,
                            "target_id": action.target_id,
                            "margin": action.margin,
                            "action_sha256": action.canonical_sha256(),
                        }
                    )
            else:
                totals["components_fallback"] += 1
                reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1

    return {
        "schema": "celltrack-alpha.ae-p5-c0-c2-rank1.v1",
        "status": "COMPLETE",
        "scope": {
            "source_partition": "fit-train105",
            "family": "44b6",
            "held_labels_used": False,
            "graph_run": False,
            "family_6bba_run": False,
            "cv57_run": False,
        },
        "rule": {
            "k": K_BEST,
            "introduce_membership_minimum": f"{INTRODUCE_MIN_MEMBERSHIP}/{K_BEST}",
            "remove_membership_maximum": f"{REMOVE_MAX_MEMBERSHIP}/{K_BEST}",
            "no_op_cost": NO_OP_COST,
            "evidence_feature_index": MARGIN_FEATURE_INDEX,
        },
        "ownership": {
            "source": "candidate source outgoing boundary",
            "destination": "candidate target incoming boundary",
            "incumbent_edge": "every exact baseline edge removed by the N1 replacement boundary",
            "challenger_edge": "the exact candidate edge added by the swap",
            "no_op_endpoint": "the explicit no-op alternative keyed by the candidate endpoint pair",
        },
        "totals": totals,
        "component_size_histogram": dict(sorted(size_histogram.items())),
        "feasible_assignment_count_histogram": dict(sorted(feasible_histogram.items())),
        "fallback_reasons": dict(sorted(reasons.items())),
        "rank1_selected_actions": selected_rows,
        "barrier": "RANK1_EMITTED_NO_GRAPH_RUN",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = run(args.population_root, args.control_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"totals": result["totals"], "fallbacks": result["fallback_reasons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

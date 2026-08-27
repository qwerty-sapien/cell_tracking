#!/usr/bin/env python3
"""Infrastructure-only adapter for bounded exact winner graph compilation.

The frozen scorer calls the historical batch compiler, which materializes and
hashes a complete single-action graph for every selected action even though the
batch result consumes only each action hash and retained-anchor set.  This
adapter preserves every action/batch validation and every observable batch
field while computing the final graph only once per video.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import score_winner_once as target


a0 = target.a1r3.a1r.base.a0
_REFERENCE_COMPILE = a0.compile_replace_components


def compile_replace_components_bounded(
    baseline: Any,
    actions: Sequence[Any],
) -> Any:
    """Exact observable-equivalent batch compile without per-action graph hashes."""

    if not actions:
        return _REFERENCE_COMPILE(baseline, actions)

    a0.validate_graphs({baseline.dataset: baseline})
    ordered = tuple(sorted(actions, key=lambda item: item.action_id))
    if len({item.action_id for item in ordered}) != len(ordered):
        raise a0.ScientificMismatch("replace_component action IDs are not unique")

    before_edges = a0.edge_set(baseline)
    baseline_out: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
    baseline_in: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
    for pair in before_edges:
        baseline_out[pair[0]].add(pair)
        baseline_in[pair[1]].add(pair)

    expected_identity = f"train:{baseline.dataset}"
    added_owner: dict[tuple[int, int], str] = {}
    removed_owner: dict[tuple[int, int], str] = {}
    source_owner: dict[int, str] = {}
    target_owner: dict[int, str] = {}
    additions: set[tuple[int, int]] = set()
    removals: set[tuple[int, int]] = set()
    action_hashes: list[str] = []
    retained_anchors: list[tuple[int, int]] = []

    for action in ordered:
        if action.identity != expected_identity:
            raise a0.ScientificMismatch(
                f"action identity {action.identity} does not match {expected_identity}"
            )
        collections = {
            "missing_edges": action.missing_edges,
            "add_edges": action.add_edges,
            "remove_edges": action.remove_edges,
            "replace_sources": action.replace_sources,
            "replace_targets": action.replace_targets,
        }
        for label, values in collections.items():
            if tuple(sorted(set(values))) != values:
                raise a0.ScientificMismatch(
                    f"{action.action_id} {label} is not sorted unique"
                )

        local_additions = set(action.add_edges)
        local_removals = set(action.remove_edges)
        missing = set(action.missing_edges)
        if not local_removals <= before_edges:
            raise a0.ScientificMismatch(
                f"{action.action_id} removes a non-baseline edge"
            )
        if not missing <= local_additions or missing & before_edges:
            raise a0.ScientificMismatch(
                f"{action.action_id} has an invalid missing-edge set"
            )
        actual_closure: set[tuple[int, int]] = set()
        for source in action.replace_sources:
            actual_closure.update(baseline_out[source])
        for target_id in action.replace_targets:
            actual_closure.update(baseline_in[target_id])
        if local_removals != actual_closure:
            raise a0.ScientificMismatch(
                f"{action.action_id} does not own its complete incident closure"
            )
        if {pair[0] for pair in local_additions} != set(action.replace_sources):
            raise a0.ScientificMismatch(
                f"{action.action_id} replacement sources drifted"
            )
        if {pair[1] for pair in local_additions} != set(action.replace_targets):
            raise a0.ScientificMismatch(
                f"{action.action_id} replacement targets drifted"
            )
        if any(
            source not in baseline.nodes
            or target_id not in baseline.nodes
            or source == target_id
            for source, target_id in local_additions
        ):
            raise a0.ScientificMismatch(
                f"{action.action_id} has invalid added endpoints"
            )

        for pair in action.add_edges:
            previous = added_owner.get(pair)
            if previous is not None:
                raise a0.ScientificMismatch(
                    f"edge {pair} added by both {previous} and {action.action_id}"
                )
            remover = removed_owner.get(pair)
            if remover is not None and remover != action.action_id:
                raise a0.ScientificMismatch(
                    f"edge {pair} removed by {remover} and added by {action.action_id}"
                )
            added_owner[pair] = action.action_id
        for pair in action.remove_edges:
            previous = removed_owner.get(pair)
            if previous is not None:
                raise a0.ScientificMismatch(
                    f"edge {pair} removed by both {previous} and {action.action_id}"
                )
            adder = added_owner.get(pair)
            if adder is not None and adder != action.action_id:
                raise a0.ScientificMismatch(
                    f"edge {pair} added by {adder} and removed by {action.action_id}"
                )
            removed_owner[pair] = action.action_id
        for node_id in action.replace_sources:
            previous = source_owner.get(node_id)
            if previous is not None:
                raise a0.ScientificMismatch(
                    f"source boundary {node_id} owned by {previous} and {action.action_id}"
                )
            source_owner[node_id] = action.action_id
        for node_id in action.replace_targets:
            previous = target_owner.get(node_id)
            if previous is not None:
                raise a0.ScientificMismatch(
                    f"target boundary {node_id} owned by {previous} and {action.action_id}"
                )
            target_owner[node_id] = action.action_id

        additions.update(action.add_edges)
        removals.update(action.remove_edges)
        action_hashes.append(a0.action_sha256(action))
        retained_anchors.extend(action.retained_anchors)

    candidate = a0.graph_with_edges(
        baseline, (before_edges - removals) | additions
    )
    before_hash = a0.graph_sha256(baseline)
    after_hash = a0.graph_sha256(candidate)
    batch_hash = a0.canonical_sha256(
        {
            "schema": "celltrack-alpha.replace-component-batch.v1",
            "before_graph_sha256": before_hash,
            "action_sha256s": action_hashes,
            "after_graph_sha256": after_hash,
        }
    )
    return a0.CompiledComponentBatch(
        actions=ordered,
        graph=candidate,
        before_graph_sha256=before_hash,
        after_graph_sha256=after_hash,
        edge_set_sha256=a0.edge_set_sha256(a0.edge_set(candidate)),
        batch_sha256=batch_hash,
        action_sha256s=tuple(action_hashes),
        retained_anchors=tuple(sorted(retained_anchors)),
    )


def main() -> int:
    if hasattr(target.CompositeGuard, "active_monitor"):
        raise RuntimeError("compatibility alias is unexpectedly already present")
    if a0.compile_replace_components is not _REFERENCE_COMPILE:
        raise RuntimeError("historical compiler is unexpectedly already patched")
    target.CompositeGuard.active_monitor = property(lambda self: self.monitor)
    a0.compile_replace_components = compile_replace_components_bounded
    return target.main()


if __name__ == "__main__":
    raise SystemExit(main())

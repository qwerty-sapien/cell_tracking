#!/usr/bin/env python3
"""Fail-closed D1 division emission for an already-built submission graph.

The input is a complete, divisions-off submission CSV plus a candidate receipt
from a division arm.  This module deliberately does not discover candidates,
fit a model, or invoke the official scorer.  It only validates the frozen D1
contract and applies its already-selected fork motifs atomically with frozen N1
conflict ownership.

It is copied unchanged into the Lane 1 submission package by
``build_division_submission_kernel.py``.  The sibling ``frozen_d1_decoder.py``
is a byte-for-byte copy of the D1 decoder that was validated against the
official scorer fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from frozen_d1_decoder import ForkCandidate, SuppressionRecord, duplicate_survivors, suppress_candidates

from celltrack_alpha.action_compiler import ActionKind, GraphAction, compile_batch, graph_sha256
from celltrack_alpha.kaggle_calibration import SubmissionGraph, read_submission, write_submission


SCHEMA = "celltrack-alpha.lane1-division-submission.v1"
CANDIDATE_SCHEMA = "celltrack-alpha.d1-division-candidates.v1"
FROZEN_D1_DECODER_SHA256 = "e780a398a6586e0c6acd7fb89d54626e732b254992252db91e20f5385a4bbcf6"
FROZEN_D1_EVIDENCE_SHA256 = "f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9"
FROZEN_N1_COMPILER_SHA256 = "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class DivisionEmissionError(ValueError):
    """A candidate receipt cannot safely be emitted into the submission graph."""


@dataclass(frozen=True)
class ParsedCandidate:
    """A receipt candidate after graph-local D1 validation."""

    dataset: str
    candidate_id: str
    fork: ForkCandidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DivisionEmissionError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DivisionEmissionError(f"non-finite JSON literal {token!r} in {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, DivisionEmissionError) as exc:
        if isinstance(exc, DivisionEmissionError):
            raise
        raise DivisionEmissionError(f"cannot parse JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DivisionEmissionError(f"JSON root must be an object: {path}")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DivisionEmissionError(f"{label} must be an integer")
    return int(value)


def _require_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DivisionEmissionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DivisionEmissionError(f"{label} must be finite")
    return result


def _require_key(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_KEY.fullmatch(value):
        raise DivisionEmissionError(
            f"{label} must match {_SAFE_KEY.pattern!r}; got {value!r}"
        )
    return value


def _aggregate_graph_sha256(graphs: Mapping[str, SubmissionGraph]) -> str:
    payload = [
        {"dataset": dataset, "graph_sha256": graph_sha256(graphs[dataset])}
        for dataset in sorted(graphs)
    ]
    return _canonical_sha256(payload)


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def validate_frozen_contract() -> dict[str, Any]:
    """Authenticate the copied D1/N1 sources and reuse the frozen D1 receipt.

    The evidence JSON is deliberately inspected, not re-executed: rerunning the
    official scorer fixtures is out of scope for a packaging rehearsal.
    """

    root = _package_root()
    decoder = root / "frozen_d1_decoder.py"
    evidence = root / "frozen_d1_evidence.json"
    compiler = root / "celltrack_alpha" / "action_compiler.py"
    actual = {
        "d1_decoder_sha256": _sha256_file(decoder),
        "d1_evidence_sha256": _sha256_file(evidence),
        "n1_compiler_sha256": _sha256_file(compiler),
    }
    expected = {
        "d1_decoder_sha256": FROZEN_D1_DECODER_SHA256,
        "d1_evidence_sha256": FROZEN_D1_EVIDENCE_SHA256,
        "n1_compiler_sha256": FROZEN_N1_COMPILER_SHA256,
    }
    if actual != expected:
        raise DivisionEmissionError(
            f"frozen contract hash mismatch: expected={expected}, observed={actual}"
        )
    receipt = _strict_json(evidence)
    fixtures = receipt.get("official_fixtures")
    counterfactuals = receipt.get("counterfactuals")
    if (
        receipt.get("official_fixture_count") != 14
        or not isinstance(fixtures, list)
        or len(fixtures) != 14
        or any(
            not isinstance(item, Mapping)
            or item.get("expected_tp_fp_fn") != item.get("observed_tp_fp_fn")
            for item in fixtures
        )
        or receipt.get("counterfactual_count") != 51
        or not isinstance(counterfactuals, list)
        or len(counterfactuals) != 51
        or receipt.get("duplicate_survivors") != 0
        or receipt.get("repeated_output_byte_identical") is not True
        or receipt.get("held_graph_scoring") is not False
        or receipt.get("cv57_access") is not False
        or receipt.get("gpu_action") is not False
    ):
        raise DivisionEmissionError("frozen D1 evidence receipt is incomplete or drifted")
    return {
        **actual,
        "official_fixture_count": 14,
        "counterfactual_count": 51,
        "duplicate_survivors": 0,
        "reused_without_rerun": True,
    }


def _validate_gate(receipt: Mapping[str, Any]) -> None:
    gate = receipt.get("gate")
    if not isinstance(gate, Mapping):
        raise DivisionEmissionError("non-empty candidate receipt requires a 44b6 gate object")
    if gate.get("family") != "44b6":
        raise DivisionEmissionError("candidate receipt must be gated on the 44b6 family")
    if _require_score(gate.get("combined_score_delta"), "gate.combined_score_delta") <= 0.0:
        raise DivisionEmissionError("44b6 combined-score delta must be strictly positive")
    if _require_score(gate.get("division_jaccard"), "gate.division_jaccard") < 0.30:
        raise DivisionEmissionError("44b6 division Jaccard must be at least 0.30")
    for field in (
        "adjusted_edge_jaccard_delta",
        "raw_edge_jaccard_delta",
        "node_recall_delta",
    ):
        if _require_score(gate.get(field), f"gate.{field}") < 0.0:
            raise DivisionEmissionError(f"44b6 non-regression gate failed: {field}")
    if _require_int(gate.get("duplicate_survivors"), "gate.duplicate_survivors") != 0:
        raise DivisionEmissionError("44b6 gate must have zero duplicate survivors")
    artifact_sha = gate.get("source_artifact_sha256")
    if not isinstance(artifact_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
        raise DivisionEmissionError("44b6 gate must carry its source artifact SHA-256")


def _candidate_from_row(
    row: Mapping[str, Any], graphs: Mapping[str, SubmissionGraph]
) -> ParsedCandidate:
    required = {
        "dataset",
        "candidate_id",
        "parent_id",
        "child_a_id",
        "child_b_id",
        "grandchild_a_id",
        "grandchild_b_id",
        "score",
        "equivalence_key",
    }
    if set(row) != required:
        raise DivisionEmissionError(
            f"candidate fields must be exactly {sorted(required)}; got {sorted(row)}"
        )
    dataset = row["dataset"]
    if not isinstance(dataset, str) or not dataset or dataset not in graphs:
        raise DivisionEmissionError(f"candidate references an unknown dataset: {dataset!r}")
    candidate_id = _require_key(row["candidate_id"], "candidate_id")
    parent = _require_int(row["parent_id"], "parent_id")
    child_a = _require_int(row["child_a_id"], "child_a_id")
    child_b = _require_int(row["child_b_id"], "child_b_id")
    grandchild_a = _require_int(row["grandchild_a_id"], "grandchild_a_id")
    grandchild_b = _require_int(row["grandchild_b_id"], "grandchild_b_id")
    node_ids = (parent, child_a, child_b, grandchild_a, grandchild_b)
    if len(set(node_ids)) != len(node_ids):
        raise DivisionEmissionError("D1 fork motif nodes must be pairwise distinct")
    graph = graphs[dataset]
    missing = [node_id for node_id in node_ids if node_id not in graph.nodes]
    if missing:
        raise DivisionEmissionError(f"D1 fork motif has missing graph nodes: {missing}")
    parent_node = graph.nodes[parent]
    child_a_node = graph.nodes[child_a]
    child_b_node = graph.nodes[child_b]
    grandchild_a_node = graph.nodes[grandchild_a]
    grandchild_b_node = graph.nodes[grandchild_b]
    if child_a_node.t != parent_node.t + 1 or child_b_node.t != parent_node.t + 1:
        raise DivisionEmissionError("D1 daughters must be direct next-frame children")
    if (
        grandchild_a_node.t != child_a_node.t + 1
        or grandchild_b_node.t != child_b_node.t + 1
    ):
        raise DivisionEmissionError("D1 persistence evidence must be direct next-frame grandchildren")
    base_edges = {(edge.source_id, edge.target_id) for edge in graph.edges}
    required_persistence = {(child_a, grandchild_a), (child_b, grandchild_b)}
    if not required_persistence <= base_edges:
        raise DivisionEmissionError(
            "D1 persistence evidence must already be present in the base graph; it is never invented"
        )
    if {(parent, child_a), (parent, child_b)}.intersection(base_edges):
        raise DivisionEmissionError(
            "D1 daughters must be unlinked births in the base graph; N1 replaces all conflicting ownership rather than re-adding a retained continuation"
        )
    if sum(edge.source_id == parent for edge in graph.edges) > 1:
        raise DivisionEmissionError("base graph parent already has a division; D1 only opens new forks")
    return ParsedCandidate(
        dataset=dataset,
        candidate_id=candidate_id,
        fork=ForkCandidate(
            parent=parent,
            child_a=child_a,
            child_b=child_b,
            grandchild_a=grandchild_a,
            grandchild_b=grandchild_b,
            score=_require_score(row["score"], "score"),
            equivalence_key=_require_key(row["equivalence_key"], "equivalence_key"),
        ),
    )


def _parse_candidates(
    receipt: Mapping[str, Any], graphs: Mapping[str, SubmissionGraph], base_submission_sha256: str, *, allow_empty: bool
) -> tuple[dict[str, list[ParsedCandidate]], dict[str, Any]]:
    if receipt.get("schema") != CANDIDATE_SCHEMA:
        raise DivisionEmissionError("candidate receipt schema is not the frozen D1 input schema")
    if receipt.get("base_submission_sha256") != base_submission_sha256:
        raise DivisionEmissionError("candidate receipt does not bind to this exact base submission CSV")
    rows = receipt.get("candidates")
    if not isinstance(rows, list):
        raise DivisionEmissionError("candidate receipt candidates must be a list")
    if not rows and not allow_empty:
        raise DivisionEmissionError("empty candidate receipt is allowed only for an explicit dry-run")
    if rows:
        _validate_gate(receipt)
    elif set(receipt) != {"schema", "base_submission_sha256", "candidates"}:
        raise DivisionEmissionError("empty dry-run receipt may contain only schema, base hash, and candidates")

    parsed: dict[str, list[ParsedCandidate]] = defaultdict(list)
    candidate_ids: set[tuple[str, str]] = set()
    exact_forks: set[tuple[str, ForkCandidate]] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            raise DivisionEmissionError("each candidate must be an object")
        candidate = _candidate_from_row(item, graphs)
        identity = (candidate.dataset, candidate.candidate_id)
        if identity in candidate_ids:
            raise DivisionEmissionError(f"duplicate candidate ID: {identity}")
        candidate_ids.add(identity)
        fork_identity = (candidate.dataset, candidate.fork)
        if fork_identity in exact_forks:
            raise DivisionEmissionError("candidate receipt contains an exact duplicate fork")
        exact_forks.add(fork_identity)
        parsed[candidate.dataset].append(candidate)
    return dict(parsed), {"candidate_receipt_sha256": _canonical_sha256(receipt)}


def _compile_dataset(
    graph: SubmissionGraph, candidates: Iterable[ParsedCandidate]
) -> tuple[SubmissionGraph, dict[str, Any]]:
    rows = list(candidates)
    by_fork = {candidate.fork: candidate for candidate in rows}
    accepted, suppressed = suppress_candidates(
        (candidate.fork for candidate in rows), require_persistence=True
    )
    if duplicate_survivors(accepted) != 0:
        raise DivisionEmissionError("frozen D1 suppression left a duplicate survivor")
    base_edges = {(edge.source_id, edge.target_id) for edge in graph.edges}
    actions: list[GraphAction] = []
    for fork in accepted:
        parsed = by_fork[fork]
        replace_sources = (fork.parent,)
        replace_targets = tuple(sorted(fork.children))
        conflicts = tuple(
            sorted(
                edge
                for edge in base_edges
                if edge[0] in replace_sources or edge[1] in replace_targets
            )
        )
        action_identity = _canonical_sha256(
            {
                "dataset": graph.dataset,
                "candidate_id": parsed.candidate_id,
                "fork": {
                    "parent": fork.parent,
                    "children": list(fork.children),
                    "grandchildren": list(fork.grandchildren),
                    "equivalence_key": fork.equivalence_key,
                },
            }
        )
        actions.append(
            GraphAction(
                action_id=f"d1.fork.{action_identity[:32]}",
                kind=ActionKind.DIVISION_FORK,
                add_edges=((fork.parent, fork.child_a), (fork.parent, fork.child_b)),
                replace_outgoing_from=replace_sources,
                replace_incoming_to=replace_targets,
                expected_conflict_edges=conflicts,
            )
        )
    if actions:
        compiled = compile_batch(graph, actions)
        output = compiled.graph
        batch_sha256: str | None = compiled.batch_sha256
        removed_edges = [list(edge) for edge in compiled.removed_edges]
    else:
        output = graph.copy()
        batch_sha256 = None
        removed_edges = []
    reasons = Counter(record.reason for record in suppressed)
    return output, {
        "input_candidates": len(rows),
        "accepted_forks": len(accepted),
        "suppressed_candidates": len(suppressed),
        "suppression_reasons": dict(sorted(reasons.items())),
        "duplicate_survivors": duplicate_survivors(accepted),
        "accepted_equivalence_keys": [fork.equivalence_key for fork in accepted],
        "batch_sha256": batch_sha256,
        "removed_conflict_edges": removed_edges,
    }


def _atomic_write_submission(graphs: Mapping[str, SubmissionGraph], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lane1-division-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        write_submission(dict(graphs), staged)
        staged.replace(output)


def emit_submission(
    *, base_submission: Path, candidate_receipt: Path, output: Path, allow_empty: bool
) -> dict[str, Any]:
    frozen = validate_frozen_contract()
    graphs = read_submission(base_submission)
    base_sha256 = _sha256_file(base_submission)
    base_graph_sha256 = _aggregate_graph_sha256(graphs)
    receipt = _strict_json(candidate_receipt)
    candidates_by_dataset, candidate_meta = _parse_candidates(
        receipt, graphs, base_sha256, allow_empty=allow_empty
    )
    output_graphs: dict[str, SubmissionGraph] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    for dataset in sorted(graphs):
        output_graph, stats = _compile_dataset(graphs[dataset], candidates_by_dataset.get(dataset, ()))
        output_graphs[dataset] = output_graph
        dataset_stats[dataset] = stats
    _atomic_write_submission(output_graphs, output)
    round_tripped = read_submission(output)
    output_graph_sha256 = _aggregate_graph_sha256(round_tripped)
    input_candidate_count = sum(stats["input_candidates"] for stats in dataset_stats.values())
    accepted_forks = sum(stats["accepted_forks"] for stats in dataset_stats.values())
    duplicate_count = sum(stats["duplicate_survivors"] for stats in dataset_stats.values())
    if duplicate_count != 0:
        raise DivisionEmissionError("output retained duplicate D1 event survivors")
    no_op = input_candidate_count == 0
    if no_op and output_graph_sha256 != base_graph_sha256:
        raise DivisionEmissionError("empty divisions-on dry-run changed the scorer input graph")
    return {
        "schema": SCHEMA,
        "base_submission": str(base_submission),
        "base_submission_sha256": base_sha256,
        "base_graph_sha256": base_graph_sha256,
        "output_submission": str(output),
        "output_submission_sha256": _sha256_file(output),
        "output_graph_sha256": output_graph_sha256,
        "output_schema": {
            "validated_by": "celltrack_alpha.kaggle_calibration.read_submission",
            "divisions_enabled": True,
            "indegree_max": 1,
            "outdegree_max": 2,
        },
        "frozen_contract": frozen,
        "candidate": candidate_meta,
        "datasets": dataset_stats,
        "input_candidate_count": input_candidate_count,
        "accepted_forks": accepted_forks,
        "duplicate_survivors": duplicate_count,
        "official_scorer_rerun": False,
        "score_change": 0.0 if no_op else None,
        "score_change_basis": (
            "exact canonical scorer-input graph identity after empty divisions-on round trip"
            if no_op
            else "not scored in packaging lane; a non-empty candidate changes scorer input"
        ),
        "verdict": "PASS_EMPTY_DIVISIONS_ON" if no_op else "PASS_D1_EMISSION",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report or args.output.with_suffix(".division_receipt.json")
    try:
        report = emit_submission(
            base_submission=args.base_submission.resolve(),
            candidate_receipt=args.candidate_receipt.resolve(),
            output=args.output.resolve(),
            allow_empty=bool(args.allow_empty),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("verdict", "accepted_forks", "duplicate_survivors", "score_change")}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"DIVISION_EMISSION_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

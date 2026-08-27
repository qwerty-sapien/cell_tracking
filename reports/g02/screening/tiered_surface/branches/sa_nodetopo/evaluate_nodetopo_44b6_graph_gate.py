#!/usr/bin/env python3
"""Score the frozen 50-action NODETOPO population on the 44b6 graph gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from zarr.storage import ZipStore


ROOT = Path(
    os.environ.get(
        "CELLTRACK_ROOT", str(Path(__file__).resolve().parents[6])
    )
).resolve()
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionNode,
    read_submission,
    sha256_file,
    write_submission,
)


SCHEMA = "celltrack-alpha.sa-nodetopo-44b6-graph-gate.v1"
GENERATOR = (
    ROOT / "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen"
)
PREREGISTRATION = GENERATOR / "PREREGISTRATION.json"
CURVE = GENERATOR / "generated/FALSE_NODE_STRESS_CURVE.json"
ACTIONS = GENERATOR / "generated/action_population.jsonl.gz"
RAW_MANIFEST = GENERATOR / "generated/raw_universe/manifest.json"
BASELINE_MANIFEST = ROOT / "reports/g02/fitcache_117/manifest.json"
BASELINE_FRAGMENTS = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
FOLDS = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"

PINS = {
    PREREGISTRATION: "c6529b5da9e6bbf76f08067fd68d248a56aa8dcb6104512e91905d8d58ecaa87",
    CURVE: "fff2b78042af1112bfa570aec3b83dc6c82bb448d944804d98ec1cc0f3e292e3",
    ACTIONS: "6be1c77b4869c0a83eb3e540f13b16d71ba097b30bd808647d688bea0a5a5324",
    RAW_MANIFEST: "40a7ffdcbbfc20201ce74f0c6174df5b695f9b73503836ebf270679ee8dda22c",
    BASELINE_MANIFEST: "20ef8056bea8afb60f76cf99357c7b62215b2db6c408d78a1c3ba21c41431a33",
    FOLDS: "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
}
SCORER_SOURCE_PINS = {
    ROOT
    / "src/celltrack_alpha/official_eval.py": (
        "22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6"
    ),
    ROOT
    / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py": (
        "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444"
    ),
    ROOT
    / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py": (
        "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9"
    ),
}
RAW_ARCHIVE_BYTES = 87_393_127_165
ADJUSTED_FLOOR = 0.005
RAW_FLOOR = 0.0
NODE_RECALL_FLOOR = 0.0


class NodeTopoGraphGateError(RuntimeError):
    """The frozen NODETOPO graph-gate contract was violated."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NodeTopoGraphGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise NodeTopoGraphGateError(f"{label} must be an object")
    return value


def _verify_pins() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise NodeTopoGraphGateError(
                f"frozen input identity drifted: {path}; expected {expected}, got {actual}"
            )
        observed[str(path.relative_to(ROOT))] = actual
    raw_root = os.environ.get("CELLTRACK_RAW_ROOT")
    if raw_root:
        if not Path(raw_root).resolve().is_dir():
            raise NodeTopoGraphGateError("extracted competition root missing")
    elif RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise NodeTopoGraphGateError("raw archive byte size drifted")
    for path, expected in SCORER_SOURCE_PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise NodeTopoGraphGateError(
                f"banked official scorer identity drifted: {path}; "
                f"expected {expected}, got {actual}"
            )
        observed[str(path.relative_to(ROOT))] = actual
    return observed


def _node(raw: Mapping[str, Any]) -> SubmissionNode:
    return SubmissionNode(
        node_id=int(raw["node_id"]),
        t=int(raw["t"]),
        z=int(raw["z"]),
        y=int(raw["y"]),
        x=int(raw["x"]),
    )


def _pair(raw: Sequence[Any]) -> tuple[int, int]:
    if len(raw) != 2:
        raise NodeTopoGraphGateError("edge pair must contain exactly two endpoints")
    return int(raw[0]), int(raw[1])


def _action(raw: Mapping[str, Any]) -> GraphAction:
    try:
        kind = ActionKind(str(raw["action_kind"]))
        return GraphAction(
            action_id=str(raw["action_id"]),
            kind=kind,
            add_nodes=tuple(_node(value) for value in raw["add_nodes"]),
            add_edges=tuple(_pair(value) for value in raw["add_edges"]),
            replace_outgoing_from=tuple(
                int(value) for value in raw["replace_outgoing_from"]
            ),
            replace_incoming_to=tuple(
                int(value) for value in raw["replace_incoming_to"]
            ),
            expected_conflict_edges=tuple(
                _pair(value) for value in raw["expected_conflict_edges"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NodeTopoGraphGateError("malformed frozen action row") from exc


def _load_actions() -> list[tuple[str, GraphAction]]:
    rows: list[tuple[str, GraphAction]] = []
    try:
        with gzip.open(ACTIONS, "rt", encoding="ascii", newline="") as handle:
            for line in handle:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise NodeTopoGraphGateError("action row must be an object")
                rows.append((str(raw["identity"]), _action(raw)))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NodeTopoGraphGateError("cannot read frozen action population") from exc
    if (
        len(rows) != 50
        or len({action.action_id for _identity, action in rows}) != 50
        or {action.kind for _identity, action in rows}
        != {ActionKind.RESTORE_TOPOLOGY}
    ):
        raise NodeTopoGraphGateError(
            "population must contain exactly 50 unique restore_topology actions"
        )
    return rows


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: int(sum(float(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }


def _decision(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    adjusted = float(candidate["adj_edge_jaccard"]) - float(
        baseline["adj_edge_jaccard"]
    )
    raw = float(candidate["edge_jaccard"]) - float(
        baseline["edge_jaccard"]
    )
    recall = float(candidate["node_recall"]) - float(
        baseline["node_recall"]
    )
    passed = (
        adjusted >= ADJUSTED_FLOOR
        and raw >= RAW_FLOOR
        and recall >= NODE_RECALL_FLOOR
    )
    return {
        "verdict": "GO_44B6" if passed else "KILL_44B6_STOP",
        "adjusted_edge_jaccard_delta": adjusted,
        "raw_edge_jaccard_delta": raw,
        "node_recall_delta": recall,
        "gates": {
            "adjusted_delta_ge_0_005": adjusted >= ADJUSTED_FLOOR,
            "raw_delta_ge_0": raw >= RAW_FLOOR,
            "node_recall_delta_ge_0": recall >= NODE_RECALL_FLOOR,
        },
        "all_gates_passed": passed,
        "second_family_opened": False,
        "dead_promotion_performed": False,
        "tier_move_performed": False,
    }


def _canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def run(output: Path) -> dict[str, Any]:
    if (
        os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
        or os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
    ):
        raise NodeTopoGraphGateError("the zero-swap local guard is required")
    if output.exists():
        raise NodeTopoGraphGateError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    pins = _verify_pins()

    preregistration = _load_json(PREREGISTRATION, "generator preregistration")
    curve = _load_json(CURVE, "frozen action curve")
    raw_manifest = _load_json(RAW_MANIFEST, "raw-universe manifest")
    baseline_manifest = _load_json(BASELINE_MANIFEST, "baseline manifest")
    if (
        preregistration.get("scope", {}).get("family") != "44b6"
        or preregistration.get("scope", {}).get("identity_count") != 36
        or curve.get("population_count") != 50
        or curve.get("action_population_sha256") != PINS[ACTIONS]
        or curve.get("single_n1_compile_passed") != 50
        or curve.get("single_n1_compile_total") != 50
        or curve.get("official_scorer_run") is not False
        or curve.get("graph_metric_run") is not False
    ):
        raise NodeTopoGraphGateError("frozen generator contract drifted")

    identities = tuple(
        str(row["identity"]) for row in raw_manifest.get("videos", ())
    )
    if (
        len(identities) != 36
        or len(set(identities)) != 36
        or any(not value.startswith("train:44b6_") for value in identities)
    ):
        raise NodeTopoGraphGateError("44b6 identity population drifted")
    scorer_sources = {
        str(path.relative_to(ROOT)): digest
        for path, digest in SCORER_SOURCE_PINS.items()
    }

    action_rows = _load_actions()
    by_identity: dict[str, list[GraphAction]] = defaultdict(list)
    for identity, action in action_rows:
        if identity not in set(identities):
            raise NodeTopoGraphGateError(
                f"action escaped the frozen 44b6 population: {identity}"
            )
        by_identity[identity].append(action)
    if len(by_identity) != 24:
        raise NodeTopoGraphGateError("50-action population must affect 24 videos")

    budget50 = {
        str(identity): value
        for identity, value in curve["false_node_budget_stress_curve"][-1][
            "batch_receipts"
        ].items()
    }
    if (
        curve["false_node_budget_stress_curve"][-1]["budget"] != 50
        or set(budget50) != set(by_identity)
    ):
        raise NodeTopoGraphGateError("budget-50 N1 receipt coverage drifted")

    output.mkdir(parents=True)
    candidate_dir = output / "candidate_fragments"
    candidate_dir.mkdir()
    baseline_graphs = {}
    candidate_graphs = {}
    graph_records = []
    baseline_paths: list[Path] = []
    candidate_paths: list[Path] = []
    for identity in identities:
        dataset = identity.split(":", 1)[1]
        baseline_path = BASELINE_FRAGMENTS / f"{identity.replace(':', '__', 1)}.csv"
        expected_fragment = baseline_manifest["videos"][identity]["files"][
            "fragments"
        ]["sha256"]
        actual_fragment = sha256_file(baseline_path)
        if actual_fragment != expected_fragment:
            raise NodeTopoGraphGateError(
                f"baseline fragment identity drifted: {identity}"
            )
        loaded = read_submission(baseline_path)
        baseline = loaded.get(dataset)
        if baseline is None or len(loaded) != 1:
            raise NodeTopoGraphGateError(
                f"baseline fragment graph identity drifted: {identity}"
            )
        baseline_hash = graph_sha256(baseline)
        actions = tuple(by_identity.get(identity, ()))
        if actions:
            compiled = compile_batch(baseline, actions)
            receipt = budget50[identity]
            if (
                compiled.before_graph_sha256 != receipt["before_graph_sha256"]
                or compiled.after_graph_sha256 != receipt["after_graph_sha256"]
                or compiled.batch_sha256 != receipt["batch_sha256"]
                or len(actions) != int(receipt["action_count"])
            ):
                raise NodeTopoGraphGateError(
                    f"budget-50 N1 replay drifted: {identity}"
                )
            candidate = compiled.graph
            candidate_hash = compiled.after_graph_sha256
            removed_edges = len(compiled.removed_edges)
        else:
            candidate = baseline.copy()
            candidate_hash = graph_sha256(candidate)
            removed_edges = 0
        candidate_path = candidate_dir / baseline_path.name
        write_submission({dataset: candidate}, candidate_path)
        baseline_graphs[identity] = baseline
        candidate_graphs[identity] = candidate
        baseline_paths.append(baseline_path)
        candidate_paths.append(candidate_path)
        graph_records.append(
            {
                "identity": identity,
                "action_count": len(actions),
                "removed_edges": removed_edges,
                "baseline_fragment_sha256": actual_fragment,
                "candidate_fragment_sha256": sha256_file(candidate_path),
                "baseline_graph_sha256": baseline_hash,
                "candidate_graph_sha256": candidate_hash,
                "observation_distinct": baseline_hash != candidate_hash,
            }
        )

    control_manifest = scoreall.validate_fragment_set(
        baseline_paths, identities
    )
    candidate_manifest = scoreall.validate_fragment_set(
        candidate_paths, identities
    )
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        baseline_rows, baseline_provenance, baseline_coverage = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )
    baseline_summary = scoreall.official_summary(baseline_rows)
    candidate_summary = scoreall.official_summary(candidate_rows)
    decision = _decision(baseline_summary, candidate_summary)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_44B6_BARRIER",
        "mechanism": "BLOCKPEAK-RESTORE-V1",
        "scope": {
            "family": "44b6",
            "fit_only": True,
            "identity_count": 36,
            "action_count": 50,
            "affected_video_count": 24,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
            "cv57_used": False,
            "second_family_used": False,
            "external_action": False,
        },
        "frozen_inputs": {
            "sha256": pins,
            "raw_archive_path": str(RAW_ARCHIVE.relative_to(ROOT)),
            "raw_archive_bytes": RAW_ARCHIVE_BYTES,
            "scorer_sources": scorer_sources,
        },
        "n1_replay": {
            "single_actions_banked": "50/50",
            "budget50_batches_replayed": f"{len(by_identity)}/{len(by_identity)}",
            "all_receipts_exact": True,
        },
        "graph": {
            "baseline": {
                "summary": baseline_summary,
                "edge_counts": _edge_counts(baseline_rows),
                "input_provenance": baseline_provenance,
                "coverage": baseline_coverage,
            },
            "candidate": {
                "summary": candidate_summary,
                "edge_counts": _edge_counts(candidate_rows),
                "input_provenance": candidate_provenance,
                "coverage": candidate_coverage,
            },
            "observation_distinct_video_count": sum(
                int(row["observation_distinct"]) for row in graph_records
            ),
            "videos": graph_records,
        },
        "decision": decision,
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_fragment_dir": str(candidate_dir.relative_to(ROOT)),
            "streaming": "one full-video GT/prediction graph at a time",
        },
    }
    _canonical_json(output / "report.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args.output.resolve())
    except (NodeTopoGraphGateError, KeyError, OSError, ValueError) as exc:
        print(f"NODETOPO 44b6 gate failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

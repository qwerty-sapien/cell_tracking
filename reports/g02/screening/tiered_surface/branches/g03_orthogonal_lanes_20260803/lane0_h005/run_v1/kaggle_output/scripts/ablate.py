#!/usr/bin/env python3
"""Run deterministic post-hoc graph ablations with the frozen official scorer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SUBMISSION_COLUMNS,
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    validate_graphs,
)


SUB = ROOT / "kaggle_kernel/bh9o6/bh9out/sub.csv"
ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
PART_SALT = "ablate-v1"

# This exact set is frozen independently of future submission-file contents.
FROZEN_IDS = (
    "44b6_0b24845f",
    "44b6_1d530831",
    "44b6_949adeb1",
    "44b6_a21120c2",
    "44b6_a2bb48bb",
    "44b6_c15fded2",
    "44b6_c771cb04",
    "6bba_09961292",
    "6bba_12665c0e",
    "6bba_23af9eeb",
    "6bba_3abfe10a",
    "6bba_55b7eebe",
    "6bba_705ec2c9",
    "6bba_8b7818bf",
    "6bba_907271db",
    "6bba_a5e926bb",
    "6bba_aeee7805",
    "6bba_d1acb6ff",
    "6bba_d5eae175",
    "6bba_e5e44988",
    "6bba_ebff6e76",
    "6bba_f20478e9",
)


class AblateError(ValueError):
    """An input or result violates the fixed ablation contract."""


@dataclass(frozen=True)
class Cand:
    name: str
    mode: str
    cut_um: float | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.mode == "base":
            policy = "identity_edges"
        elif self.mode == "edge":
            policy = "drop_edges_over_physical_cutoff"
        else:
            policy = "drop_farther_daughter_when_fork_exceeds_cutoff"
        return {
            "name": self.name,
            "mode": self.mode,
            "cutoff_um": self.cut_um,
            "node_policy": "preserve_all",
            "policy": policy,
        }


# Candidate names, order, and parameters are global and fixed before labels load.
CANDS = (
    Cand("base", "base"),
    Cand("edge10", "edge", 10.0),
    Cand("edge8", "edge", 8.0),
    Cand("edge6", "edge", 6.0),
    Cand("edge4", "edge", 4.0),
    Cand("div10", "div", 10.0),
    Cand("div8", "div", 8.0),
)


def train_id(dataset: str) -> str:
    """Map a plain submission ID to the local train identity in memory."""

    if ":" in dataset:
        raise AblateError(f"expected a plain dataset ID, got {dataset!r}")
    return f"train:{dataset}"


def _row_int(row: Mapping[str, str], field: str, line: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise AblateError(f"CSV row {line}: invalid integer field {field!r}") from exc


def load_csv(
    path: Path,
    frozen_ids: Sequence[str] = FROZEN_IDS,
) -> tuple[dict[str, SubmissionGraph], int]:
    """Parse the aggregate CSV structurally and require exact frozen coverage."""

    frozen = tuple(frozen_ids)
    if len(frozen) != len(set(frozen)):
        raise AblateError("frozen dataset IDs contain duplicates")
    expected = set(frozen)
    graphs: dict[str, SubmissionGraph] = {}
    rows = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
                raise AblateError(f"unexpected submission header: {reader.fieldnames}")
            for row in reader:
                line = reader.line_num
                if None in row or any(row.get(name) is None for name in SUBMISSION_COLUMNS):
                    raise AblateError(f"CSV row {line}: malformed column count")
                _row_int(row, "id", line)
                dataset = str(row["dataset"])
                if dataset not in expected:
                    raise AblateError(f"CSV row {line}: dataset is not frozen: {dataset!r}")
                train_id(dataset)
                graph = graphs.setdefault(dataset, SubmissionGraph(dataset, {}, []))
                row_type = row["row_type"]
                if row_type == "node":
                    node = SubmissionNode(
                        node_id=_row_int(row, "node_id", line),
                        t=_row_int(row, "t", line),
                        z=_row_int(row, "z", line),
                        y=_row_int(row, "y", line),
                        x=_row_int(row, "x", line),
                    )
                    _row_int(row, "source_id", line)
                    _row_int(row, "target_id", line)
                    if node.node_id in graph.nodes:
                        raise AblateError(
                            f"CSV row {line}: duplicate node ID {node.node_id} in {dataset}"
                        )
                    graph.nodes[node.node_id] = node
                elif row_type == "edge":
                    for field in ("node_id", "t", "z", "y", "x"):
                        _row_int(row, field, line)
                    graph.edges.append(
                        SubmissionEdge(
                            _row_int(row, "source_id", line),
                            _row_int(row, "target_id", line),
                        )
                    )
                else:
                    raise AblateError(f"CSV row {line}: unexpected row_type {row_type!r}")
                rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AblateError(f"cannot parse submission CSV {path}: {exc}") from exc

    missing = sorted(expected - set(graphs))
    extra = sorted(set(graphs) - expected)
    if missing or extra or len(graphs) != len(frozen):
        raise AblateError(f"frozen dataset coverage mismatch: missing={missing}, extra={extra}")
    try:
        validate_graphs(graphs)
    except ValueError as exc:
        raise AblateError(f"invalid source graph: {exc}") from exc
    return {dataset: graphs[dataset] for dataset in sorted(graphs)}, rows


def hash_part(
    frozen_ids: Sequence[str] = FROZEN_IDS,
    salt: str = PART_SALT,
) -> dict[str, dict[str, str]]:
    """Return a fixed balanced partition ranked only by salted ID hashes."""

    ids = tuple(frozen_ids)
    if len(ids) < 2 or len(ids) % 2:
        raise AblateError("the frozen tune/eval set must have an even size of at least two")
    ranked: list[tuple[str, str]] = []
    for dataset in ids:
        identity = train_id(dataset)
        digest = hashlib.sha256(f"{salt}\0{identity}".encode("ascii")).hexdigest()
        ranked.append((digest, identity))
    ranked.sort()
    tune = {identity for _, identity in ranked[: len(ranked) // 2]}
    result: dict[str, dict[str, str]] = {}
    for digest, identity in sorted(ranked, key=lambda item: item[1]):
        result[identity] = {
            "hash": digest,
            "part": "tune" if identity in tune else "eval",
        }
    return result


def edge_dist(
    graph: SubmissionGraph,
    edge: SubmissionEdge,
    scale_zyx: Sequence[float],
) -> float:
    if len(scale_zyx) != 3 or any(float(value) <= 0.0 for value in scale_zyx):
        raise AblateError(f"invalid physical scale {scale_zyx!r}")
    source = graph.nodes[edge.source_id]
    target = graph.nodes[edge.target_id]
    return math.sqrt(
        sum(
            ((float(a) - float(b)) * float(scale)) ** 2
            for a, b, scale in zip(
                source.coord_zyx,
                target.coord_zyx,
                scale_zyx,
                strict=True,
            )
        )
    )


def _divs(edges: Iterable[SubmissionEdge]) -> int:
    out = Counter(edge.source_id for edge in edges)
    return sum(value == 2 for value in out.values())


def apply_cand(
    graph: SubmissionGraph,
    cand: Cand,
    scale_zyx: Sequence[float],
) -> tuple[SubmissionGraph, dict[str, Any]]:
    """Apply one fixed edge-only transform and prove node preservation."""

    if cand not in CANDS:
        raise AblateError(f"candidate is not predeclared: {cand!r}")
    result = graph.copy()
    if cand.mode == "edge":
        assert cand.cut_um is not None
        result.edges = [
            edge
            for edge in graph.edges
            if edge_dist(graph, edge, scale_zyx) <= cand.cut_um
        ]
    elif cand.mode == "div":
        assert cand.cut_um is not None
        outgoing: dict[int, list[SubmissionEdge]] = defaultdict(list)
        for edge in graph.edges:
            outgoing[edge.source_id].append(edge)
        drop: set[tuple[int, int]] = set()
        for edges in outgoing.values():
            if len(edges) != 2:
                continue
            ranked = sorted(
                (edge_dist(graph, edge, scale_zyx), edge.target_id, edge)
                for edge in edges
            )
            if ranked[-1][0] > cand.cut_um:
                far = ranked[-1][2]
                drop.add((far.source_id, far.target_id))
        result.edges = [
            edge
            for edge in graph.edges
            if (edge.source_id, edge.target_id) not in drop
        ]
    elif cand.mode != "base":
        raise AblateError(f"unsupported candidate mode {cand.mode!r}")

    if result.nodes != graph.nodes:
        raise AblateError(f"candidate {cand.name} changed source nodes")
    try:
        validate_graphs({graph.dataset: result})
    except ValueError as exc:
        raise AblateError(f"candidate {cand.name} made {graph.dataset} invalid: {exc}") from exc
    return result, {
        "after_divisions": _divs(result.edges),
        "after_edges": len(result.edges),
        "after_nodes": len(result.nodes),
        "before_divisions": _divs(graph.edges),
        "before_edges": len(graph.edges),
        "before_nodes": len(graph.nodes),
        "removed_edges": len(graph.edges) - len(result.edges),
        "removed_nodes": len(graph.nodes) - len(result.nodes),
        "scale_zyx": [float(value) for value in scale_zyx],
    }


def _scale(store: Any, dataset: str) -> tuple[float, float, float]:
    group = scoreall.zarr.open_group(
        store=store,
        path=f"train/{dataset}.zarr",
        mode="r",
    )
    return tuple(float(value) for value in scoreall.parse_scale(dict(group.attrs)))


def _part_score(
    store: Any,
    graphs: Mapping[str, SubmissionGraph],
    parts: Mapping[str, Mapping[str, str]],
    cand: Cand,
    part: str,
    sub_hash: str,
    scales: Mapping[str, Sequence[float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    for dataset in sorted(graphs):
        identity = train_id(dataset)
        if parts[identity]["part"] != part:
            continue
        output, stat = apply_cand(graphs[dataset], cand, scales[dataset])
        row = scoreall.score_graph(
            store,
            identity,
            output,
            source={
                "candidate": cand.name,
                "mode": "posthoc_ablation",
                "submission_sha256": sub_hash,
            },
        )
        row.pop("elapsed_seconds", None)
        rows.append(row)
        stats[identity] = stat
    return rows, stats


def pick_best(records: Mapping[str, Mapping[str, Any]]) -> str:
    """Select strictly from tune summaries, with declaration order as tie-break."""

    best = ""
    best_score = -math.inf
    for cand in CANDS:
        record = records.get(cand.name)
        if record is None:
            raise AblateError(f"missing candidate record {cand.name!r}")
        value = record.get("summary", {}).get("tune", {}).get("score")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AblateError(f"candidate {cand.name} has no finite tune score")
        if float(value) > best_score:
            best = cand.name
            best_score = float(value)
    return best


def _totals(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    fields = (
        "before_nodes",
        "after_nodes",
        "removed_nodes",
        "before_edges",
        "after_edges",
        "removed_edges",
        "before_divisions",
        "after_divisions",
    )
    return {field: sum(int(row[field]) for row in stats.values()) for field in fields}


def score_all(
    store: Any,
    graphs: Mapping[str, SubmissionGraph],
    parts: Mapping[str, Mapping[str, str]],
    sub_hash: str,
) -> tuple[list[dict[str, Any]], str]:
    """Score tune first, freeze the winner, then touch eval labels."""

    scales = {dataset: _scale(store, dataset) for dataset in sorted(graphs)}
    records: dict[str, dict[str, Any]] = {}
    for cand in CANDS:
        rows, stats = _part_score(
            store, graphs, parts, cand, "tune", sub_hash, scales
        )
        records[cand.name] = {
            "candidate": cand.as_dict(),
            "rows": list(rows),
            "stats": dict(stats),
            "summary": {"tune": scoreall.official_summary(rows)},
        }

    selected = pick_best(records)

    # Evaluation happens only after ``selected`` is frozen above. Eval scores are
    # reported for diagnosis but never enter ``pick_best``.
    for cand in CANDS:
        rows, stats = _part_score(
            store, graphs, parts, cand, "eval", sub_hash, scales
        )
        record = records[cand.name]
        record["rows"].extend(rows)
        record["stats"].update(stats)
        record["summary"]["eval"] = scoreall.official_summary(rows)
        record["summary"]["all22"] = scoreall.official_summary(record["rows"])

    output: list[dict[str, Any]] = []
    for cand in CANDS:
        record = records[cand.name]
        samples = []
        for row in sorted(record["rows"], key=lambda value: value["identity"]):
            samples.append({**row, "partition": parts[row["identity"]]["part"]})
        output.append(
            {
                "candidate": record["candidate"],
                "diagnostics": {
                    "datasets": record["stats"],
                    "totals": _totals(record["stats"]),
                },
                "samples": samples,
                "summary": record["summary"],
            }
        )
    return output, selected


def clean_json(value: Any) -> Any:
    """Return strict-JSON-safe data without changing deterministic ordering."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(
        clean_json(value),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _load_split(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblateError(f"cannot read scorer split manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AblateError("scorer split manifest must be an object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.sub.is_file():
        raise AblateError(f"submission CSV does not exist: {args.sub}")
    if not args.zip.is_file():
        raise AblateError(f"competition archive does not exist: {args.zip}")
    if not args.split.is_file():
        raise AblateError(f"scorer split manifest does not exist: {args.split}")

    hash_before, sub_bytes = scoreall.sha256_with_size(args.sub)
    graphs, row_count = load_csv(args.sub)
    hash_after, bytes_after = scoreall.sha256_with_size(args.sub)
    if (hash_before, sub_bytes) != (hash_after, bytes_after):
        raise AblateError("submission CSV changed while it was being parsed")

    split = _load_split(args.split)
    try:
        scorer_hashes = scoreall.verify_scorer_sources(split)
        archive_ids = set(scoreall.archive_train_video_ids(args.zip))
    except scoreall.ScoreAllError as exc:
        raise AblateError(str(exc)) from exc
    expected = {train_id(dataset) for dataset in FROZEN_IDS}
    missing = sorted(expected - archive_ids)
    if missing:
        raise AblateError(f"official train store is missing frozen IDs: {missing}")

    parts = hash_part()
    store = scoreall.ZipStore(args.zip, mode="r")
    try:
        candidates, selected = score_all(store, graphs, parts, hash_before)
    finally:
        store.close()

    selected_record = next(
        record for record in candidates if record["candidate"]["name"] == selected
    )
    split_hash, split_bytes = scoreall.sha256_with_size(args.split)
    result = {
        "schema_version": 1,
        "kind": "posthoc_graph_ablation",
        "status": "completed",
        "contract": {
            "candidate_scope": "global",
            "eval_used_for_selection": False,
            "label_tuning_by_dataset": False,
            "node_policy": "preserve_all",
            "run_order": ["score_all_tune", "select_on_tune", "score_all_eval"],
            "selection_metric": "official_summary.score",
        },
        "input": {
            "submission": {
                "bytes": sub_bytes,
                "dataset_count": len(graphs),
                "dataset_ids_are_plain": True,
                "mode": "read_only",
                "row_count": row_count,
                "sha256": hash_before,
            },
            "store": {
                "frozen_ids_present_in_train": True,
                "train_video_count": len(archive_ids),
            },
            "scorer_manifest": {
                "bytes": split_bytes,
                "sha256": split_hash,
                "source_hashes": scorer_hashes,
            },
        },
        "mapping": [
            {"submission_id": dataset, "score_identity": train_id(dataset)}
            for dataset in FROZEN_IDS
        ],
        "partition": {
            "algorithm": "sha256_rank_then_equal_halves",
            "assignments": parts,
            "eval_count": sum(row["part"] == "eval" for row in parts.values()),
            "salt": PART_SALT,
            "tune_count": sum(row["part"] == "tune" for row in parts.values()),
        },
        "candidates": candidates,
        "selected": {
            "name": selected,
            "summary": selected_record["summary"],
        },
    }
    return clean_json(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", type=Path, default=SUB, help="frozen aggregate CSV")
    parser.add_argument("--zip", type=Path, default=ARCHIVE, help="local competition ZIP")
    parser.add_argument("--split", type=Path, default=SPLIT, help="frozen scorer manifest")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        text = json_text(run(args))
        if args.out is None:
            sys.stdout.write(text)
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
    except (AblateError, OSError) as exc:
        print(f"ablate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize and exactly score the preregistered Trackastra fit105 sweep."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    read_submission,
    validate_graphs,
    write_submission,
)


SCHEMA = "celltrack-alpha.trackastra-native-base-fit105-sweep.v1"
CONTROL_SCORE = 0.8815220699597315


class NativeSweepError(RuntimeError):
    """A bank, graph, exact-score, or selection invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def safe_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_number(item) for item in value]
    return value


def configurations(prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = prereg["a2_grid"]
    rows = []
    for radius in grid["radius_um"]:
        for cap in grid["neighbor_cap"]:
            for association in grid["association_threshold"]:
                for division in grid["division_threshold"]:
                    off = division == "OFF"
                    name = (
                        f"r{int(radius):02d}_c{int(cap):02d}_a{int(round(100 * association)):03d}_"
                        + ("off" if off else f"d{int(round(100 * float(division))):03d}")
                    )
                    rows.append(
                        {
                            "name": name,
                            "radius_um": float(radius),
                            "neighbor_cap": int(cap),
                            "association_threshold": float(association),
                            "division_threshold": None if off else float(division),
                            "divisions": not off,
                            "solver": "trackastra-global-descending-weight-greedy",
                        }
                    )
    if len(rows) != 108 or len({row["name"] for row in rows}) != 108:
        raise NativeSweepError("preregistered configuration grid did not produce 108 variants")
    return rows


def load_bank(bank_dir: Path, identity: str) -> tuple[np.ndarray, np.ndarray, str]:
    slug = identity.replace(":", "__", 1)
    path = bank_dir / "videos" / f"{slug}.npz"
    receipt = bank_dir / "videos" / f"{slug}.json"
    if not path.is_file() or not receipt.is_file():
        raise NativeSweepError(f"bank files missing for {identity}")
    attestation = json.loads(receipt.read_text(encoding="ascii"))
    if (
        attestation.get("schema") != "celltrack-alpha.trackastra-native-video-bank.v1"
        or attestation.get("identity") != identity
        or attestation.get("sha256") != sha256(path)
    ):
        raise NativeSweepError(f"bank attestation mismatch for {identity}")
    with np.load(path, allow_pickle=False) as value:
        nodes = np.asarray(value["nodes"], dtype=np.float64)
        edges = np.asarray(value["edges"], dtype=np.float64)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or edges.ndim != 2
        or edges.shape[1] != 4
        or not np.isfinite(nodes).all()
        or not np.isfinite(edges).all()
    ):
        raise NativeSweepError(f"bank array schema mismatch for {identity}")
    if np.max(np.abs(nodes - np.rint(nodes)), initial=0.0) > 1e-5:
        raise NativeSweepError(f"native node coordinates are not submission-integral for {identity}")
    return nodes, edges, attestation["sha256"]


def filtered_edges(edges: np.ndarray, config: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    radius = float(config["radius_um"])
    cap = int(config["neighbor_cap"])
    by_source: dict[int, list[tuple[int, float, float]]] = {}
    for source_f, target_f, weight_f, distance_f in edges:
        source, target = int(source_f), int(target_f)
        weight, distance = float(weight_f), float(distance_f)
        if distance <= radius + 1e-9:
            by_source.setdefault(source, []).append((target, weight, distance))
    eligible = []
    for source, candidates in by_source.items():
        for target, weight, _distance in sorted(
            candidates, key=lambda row: (row[2], row[0])
        )[:cap]:
            eligible.append((source, target, weight))
    return sorted(eligible, key=lambda row: (-row[2], row[0], row[1]))


def solve(edges: np.ndarray, config: Mapping[str, Any]) -> tuple[list[tuple[int, int]], dict[str, int]]:
    association = float(config["association_threshold"])
    division = config["division_threshold"]
    incoming: set[int] = set()
    outgoing: Counter[int] = Counter()
    selected: list[tuple[int, int]] = []
    considered = 0
    for source, target, weight in filtered_edges(edges, config):
        if weight < association:
            break
        considered += 1
        if target in incoming:
            continue
        degree = outgoing[source]
        if degree >= (2 if config["divisions"] else 1):
            continue
        if degree == 1 and division is not None and weight < float(division):
            continue
        incoming.add(target)
        outgoing[source] += 1
        selected.append((source, target))
    return selected, {
        "threshold_considered": considered,
        "selected_edges": len(selected),
        "forks": sum(value == 2 for value in outgoing.values()),
    }


def graph_from_solution(
    dataset: str,
    nodes: np.ndarray,
    selected: Iterable[tuple[int, int]],
) -> SubmissionGraph:
    edges = tuple((int(source), int(target)) for source, target in selected)
    used = {node for pair in edges for node in pair}
    node_map = {
        int(node_id): SubmissionNode(
            int(node_id), int(t), int(round(z)), int(round(y)), int(round(x))
        )
        for node_id, t, z, y, x in nodes
        if int(node_id) in used
    }
    graph = SubmissionGraph(
        dataset=dataset,
        nodes=node_map,
        edges=[SubmissionEdge(source, target) for source, target in edges],
    )
    validate_graphs({dataset: graph})
    return graph


def materialize(
    *,
    bank_dir: Path,
    output: Path,
    identities: tuple[str, ...],
    configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    summaries = {
        row["name"]: {
            "configuration": row,
            "video_count": 0,
            "node_count": 0,
            "edge_count": 0,
            "fork_count": 0,
            "videos": [],
        }
        for row in configs
    }
    bank_hashes = {}
    fragment_roots = {row["name"]: output / "variants" / row["name"] for row in configs}
    for root in fragment_roots.values():
        root.mkdir(parents=True)
    for ordinal, identity in enumerate(identities, start=1):
        dataset = identity.split(":", 1)[1]
        nodes, bank_edges, bank_hash = load_bank(bank_dir, identity)
        bank_hashes[identity] = bank_hash
        linked_graphs: dict[tuple[tuple[int, int], ...], Path] = {}
        for config in configs:
            selected, counts = solve(bank_edges, config)
            signature = tuple(selected)
            destination = fragment_roots[config["name"]] / f"{identity.replace(':', '__', 1)}.csv"
            if signature in linked_graphs:
                os.link(linked_graphs[signature], destination)
                graph = graph_from_solution(dataset, nodes, signature)
            else:
                graph = graph_from_solution(dataset, nodes, signature)
                write_submission({dataset: graph}, destination)
                linked_graphs[signature] = destination
            row = summaries[config["name"]]
            row["video_count"] += 1
            row["node_count"] += len(graph.nodes)
            row["edge_count"] += len(graph.edges)
            row["fork_count"] += counts["forks"]
            row["videos"].append(
                {
                    "identity": identity,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    **counts,
                    "fragment": str(destination),
                    "fragment_sha256": sha256(destination),
                }
            )
        print(f"native materialize {ordinal:03d}/105 {identity}", flush=True)
    for name, row in summaries.items():
        if row["video_count"] != 105:
            raise NativeSweepError(f"variant {name} does not cover 105 videos")
        digest = hashlib.sha256()
        for video in row["videos"]:
            digest.update(str(video["identity"]).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(video["fragment_sha256"]).encode("ascii"))
            digest.update(b"\n")
        row["aggregate_graph_sha256"] = digest.hexdigest()
        atomic_json(fragment_roots[name] / "manifest.json", row)
    return summaries, bank_hashes


def edge_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }


def score_one(task: tuple[str, str, tuple[str, ...], str]) -> dict[str, Any]:
    name, fragment_root, identities, archive = task
    from zarr.storage import ZipStore

    paths = scoreall.resolve_fragment_paths(fragment_root)
    manifest = scoreall.validate_fragment_set(paths, identities)
    with ZipStore(Path(archive), mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    return {
        "name": name,
        "summary": safe_number(summary),
        "edge_counts": edge_counts(rows),
        "coverage": coverage,
        "input_provenance": provenance,
    }


def exact_score_all(
    *,
    output: Path,
    configs: list[dict[str, Any]],
    identities: tuple[str, ...],
    archive: Path,
    workers: int,
    structural: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for row in configs:
        groups.setdefault(str(structural[row["name"]]["aggregate_graph_sha256"]), []).append(row["name"])
    representatives = {digest: sorted(names)[0] for digest, names in groups.items()}
    tasks = [
        (name, str(output / "variants" / name), identities, str(archive))
        for name in representatives.values()
    ]
    results: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {pool.submit(score_one, task): task[0] for task in tasks}
        for completed, future in enumerate(as_completed(pending), start=1):
            result = future.result()
            results[result["name"]] = result
            atomic_json(output / "scores" / f"{result['name']}.json", result)
            print(
                f"native exact score {completed:03d}/{len(tasks):03d} {result['name']} "
                f"S={result['summary']['score']:.12f}",
                flush=True,
            )
    by_digest = {
        digest: results[representative]
        for digest, representative in representatives.items()
    }
    for digest, names in groups.items():
        representative = representatives[digest]
        source = by_digest[digest]
        for name in names:
            if name == representative:
                source["score_derivation"] = "DIRECT_FROZEN_OFFICIAL_SCORER"
                source["aggregate_graph_sha256"] = digest
                atomic_json(output / "scores" / f"{name}.json", source)
                continue
            alias = {
                "name": name,
                "summary": source["summary"],
                "edge_counts": source["edge_counts"],
                "coverage": source["coverage"],
                "input_provenance": {
                    "mode": "byte-identical-graph-alias",
                    "representative": representative,
                    "aggregate_graph_sha256": digest,
                },
                "score_derivation": "BYTE_IDENTICAL_TO_DIRECT_EXACT_SCORER_REPRESENTATIVE",
                "aggregate_graph_sha256": digest,
            }
            results[name] = alias
            atomic_json(output / "scores" / f"{name}.json", alias)
    if len(results) != 108:
        raise NativeSweepError("not all 108 configurations received an exact score")
    return results


def select_result(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        name, row = item
        summary = row["summary"]
        return (
            -float(summary["score"]),
            -float(summary["division_jaccard"]),
            -float(summary["adj_edge_jaccard"]),
            -float(summary["edge_jaccard"]),
            -float(summary["node_recall"]),
            name,
        )

    name, row = sorted(results.items(), key=key)[0]
    return {"name": name, **row}


def hybrid_graph(apncc: SubmissionGraph, native: SubmissionGraph) -> tuple[SubmissionGraph, dict[str, int]]:
    outgoing: dict[int, list[int]] = {}
    for edge in native.edges:
        outgoing.setdefault(edge.source_id, []).append(edge.target_id)
    forks = {
        parent: tuple(sorted(children))
        for parent, children in outgoing.items()
        if len(children) == 2
    }
    admissible = {
        parent: children
        for parent, children in forks.items()
        if parent in apncc.nodes and all(child in apncc.nodes for child in children)
    }
    parents = set(admissible)
    children = {child for pair in admissible.values() for child in pair}
    base = {
        (edge.source_id, edge.target_id)
        for edge in apncc.edges
        if edge.source_id not in parents and edge.target_id not in children
    }
    overlay = {(parent, child) for parent, pair in admissible.items() for child in pair}
    combined = sorted(base | overlay)
    result = SubmissionGraph(
        dataset=apncc.dataset,
        nodes=dict(apncc.nodes),
        edges=[SubmissionEdge(source, target) for source, target in combined],
    )
    validate_graphs({result.dataset: result})
    return result, {
        "native_forks": len(forks),
        "inserted_forks": len(admissible),
        "removed_apncc_edges": len(apncc.edges) - len(base),
        "inserted_native_edges": len(overlay),
        "final_edges": len(result.edges),
    }


def materialize_hybrid(
    *,
    output: Path,
    selected_name: str,
    apncc_dir: Path,
    identities: tuple[str, ...],
    archive: Path,
) -> dict[str, Any]:
    destination = output / "hybrid" / "fragments"
    destination.mkdir(parents=True)
    totals: Counter[str] = Counter()
    videos = []
    for identity in identities:
        filename = f"{identity.replace(':', '__', 1)}.csv"
        apncc_graphs = read_submission(apncc_dir / filename)
        native_graphs = read_submission(output / "variants" / selected_name / filename)
        if len(apncc_graphs) != 1 or len(native_graphs) != 1:
            raise NativeSweepError(f"hybrid input graph ambiguity for {identity}")
        apncc = next(iter(apncc_graphs.values()))
        native = next(iter(native_graphs.values()))
        graph, counts = hybrid_graph(apncc, native)
        write_submission({graph.dataset: graph}, destination / filename)
        totals.update(counts)
        videos.append({"identity": identity, **counts})
    scored = score_one(("APNCC_NATIVE_FORK_HYBRID", str(destination), identities, str(archive)))
    return {
        "name": "APNCC_NATIVE_FORK_HYBRID",
        "source_native_configuration": selected_name,
        "summary": scored["summary"],
        "edge_counts": scored["edge_counts"],
        "coverage": scored["coverage"],
        "input_provenance": scored["input_provenance"],
        "totals": dict(totals),
        "videos": videos,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise NativeSweepError(f"refusing to overwrite output: {args.output}")
    prereg = json.loads(args.prereg.read_text(encoding="ascii"))
    manifest = json.loads(args.fit_manifest.read_text(encoding="ascii"))
    bank_report = json.loads((args.bank / "report.json").read_text(encoding="ascii"))
    apncc_report = json.loads(args.apncc_report.read_text(encoding="ascii"))
    identities = tuple(str(value) for value in manifest.get("identities", []))
    if (
        prereg.get("schema") != "celltrack-alpha.native-base-wave-preregistration.v1"
        or prereg.get("outcome_blind_at_recording") is not True
        or len(identities) != 105
        or bank_report.get("status") != "COMPLETE"
        or bank_report.get("aggregate", {}).get("video_count") != 105
        or apncc_report.get("status") != "COMPLETED_EXACT_FIT105_CONTROL"
        or tuple(apncc_report.get("selection", {}).get("video_ids", [])) != identities
    ):
        raise NativeSweepError("preregistration, bank, or APNCC control is incomplete")
    split = json.loads(args.split.read_text(encoding="ascii"))
    scorer_hashes = scoreall.verify_scorer_sources(split)
    configs = configurations(prereg)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    structural, bank_hashes = materialize(
        bank_dir=args.bank,
        output=args.output,
        identities=identities,
        configs=configs,
    )
    scores = exact_score_all(
        output=args.output,
        configs=configs,
        identities=identities,
        archive=args.zip,
        workers=args.workers,
        structural=structural,
    )
    apncc_summary = apncc_report["control"]["summary"]
    fit105_apncc_score = float(apncc_summary["score"])
    for name, row in scores.items():
        row["configuration"] = structural[name]["configuration"]
        row["structural"] = {
            key: structural[name][key]
            for key in ("video_count", "node_count", "edge_count", "fork_count")
        }
        row["delta_vs_fit105_apncc_control"] = (
            float(row["summary"]["score"]) - fit105_apncc_score
        )
        row["delta_vs_cv57_control"] = float(row["summary"]["score"]) - CONTROL_SCORE
    selected = select_result(scores)
    hybrid = materialize_hybrid(
        output=args.output,
        selected_name=selected["name"],
        apncc_dir=args.apncc_report.parent / "fragments",
        identities=identities,
        archive=args.zip,
    )
    hybrid["delta_vs_fit105_apncc_control"] = (
        float(hybrid["summary"]["score"]) - fit105_apncc_score
    )
    hybrid["delta_vs_cv57_control"] = float(hybrid["summary"]["score"]) - CONTROL_SCORE
    ranking = sorted(
        scores.values(),
        key=lambda row: (
            -float(row["summary"]["score"]),
            -float(row["summary"]["division_jaccard"]),
            -float(row["summary"]["adj_edge_jaccard"]),
            -float(row["summary"]["edge_jaccard"]),
            -float(row["summary"]["node_recall"]),
            row["name"],
        ),
    )
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_EXACT_108_VARIANT_SWEEP",
        "population": {
            "video_count": 105,
            "family_counts": manifest["scope"]["family_counts"],
            "validation12_access": False,
            "cv57_access": False,
        },
        "apncc_fit105_control": {
            "summary": apncc_summary,
            "edge_counts": apncc_report["control"]["edge_counts"],
        },
        "cv57_reference_score": CONTROL_SCORE,
        "a1_defaults": {
            "divisions_off": scores["r12_c10_a050_off"],
            "divisions_on": scores["r12_c10_a050_d050"],
        },
        "selected": selected,
        "hybrid": hybrid,
        "ranking": ranking,
        "selection_rule": prereg["selection_rule"],
        "provenance": {
            "preregistration": str(args.prereg),
            "preregistration_sha256": sha256(args.prereg),
            "fit_manifest": str(args.fit_manifest),
            "fit_manifest_sha256": sha256(args.fit_manifest),
            "bank_report": str(args.bank / "report.json"),
            "bank_report_sha256": sha256(args.bank / "report.json"),
            "bank_video_hashes": bank_hashes,
            "apncc_report": str(args.apncc_report),
            "apncc_report_sha256": sha256(args.apncc_report),
            "frozen_scorer_source_hashes": scorer_hashes,
        },
        "execution": {
            "device": "cpu",
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "submission": False,
        },
    }
    atomic_json(args.output / "FIT105_RESULTS.json", safe_number(result))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    branch = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
    value.add_argument("--prereg", type=Path, default=branch / "PREREGISTRATION.json")
    value.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT / "kaggle_kernel/trackastra-h1-feature-cache-prep/cache_dataset_manifest.json",
    )
    value.add_argument(
        "--bank",
        type=Path,
        default=branch / "run_v1/kaggle_output/trackastra_native_fit105_bank_v1",
    )
    value.add_argument(
        "--apncc-report",
        type=Path,
        default=branch / "apncc_fit105_control/report.json",
    )
    value.add_argument("--zip", type=Path, default=scoreall.ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    value.add_argument("--split", type=Path, default=scoreall.ROOT / "reports/g02/grouped_3fold_audit20_v1.json")
    value.add_argument("--workers", type=int, default=6)
    value.add_argument("--output", type=Path, default=branch / "fit105_sweep")
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("--workers must be between 1 and 12")
    report = run(args)
    print(json.dumps({"selected": report["selected"]["name"], "summary": report["selected"]["summary"]}, sort_keys=True))

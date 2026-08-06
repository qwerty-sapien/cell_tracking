#!/usr/bin/env python3
"""Seal the exact fit117 G1-selected graph cohort for the velocity audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "reports/g04/lanes/velocity_aware_candidate_window_20260804",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import g04_score_pathology_repairs as repair  # noqa: E402
import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
from celltrack_alpha.frame_registration import load_registration_plan  # noqa: E402


DEFAULT_PREPARE = ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json"
DEFAULT_BASE = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_EXAMPLES = ROOT / "reports/g02/extended_radius_shell_prepare_v1/examples"
DEFAULT_CENSUS = ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
DEFAULT_OUTPUT = ROOT / "reports/g04/lanes/velocity_aware_candidate_window_20260804/fit117_g1_selected_v1"


class CohortError(RuntimeError):
    """A fit117 selection, graph, or provenance invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise CohortError(f"refusing to overwrite cohort artifact: {path}")
    temporary.write_bytes(canonical_bytes(dict(value)))
    os.replace(temporary, path)


def inventory(root: Path, suffix: str) -> set[str]:
    return {path.stem.replace("__", ":", 1) for path in root.glob(f"train__*{suffix}")}


def aggregate(entries: list[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes(entries)).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CohortError(f"refusing to overwrite cohort: {args.output}")
    started = time.perf_counter()
    prepare = repair.load_json(args.prepare)
    videos = prepare.get("videos")
    if not isinstance(videos, list):
        raise CohortError("fit117 prepare video list is absent")
    selected = tuple(str(row["identity"]) for row in videos)
    scope = prepare.get("scope", {})
    scope_selected = tuple(map(str, scope.get("train_video_ids", []))) + tuple(
        map(str, scope.get("validation_video_ids", []))
    )
    if (
        prepare.get("status") != "prepared"
        or len(selected) != 117
        or len(set(selected)) != 117
        or selected != scope_selected
        or int(prepare.get("totals", {}).get("positive_pairs", -1)) != 456
    ):
        raise CohortError("canonical fit117 prepare selection or 456-positive shell drifted")
    expected = set(selected)
    base_inventory = inventory(args.base_fragments, ".csv")
    cache_inventory = inventory(args.cache, ".bhc")
    metadata_inventory = {
        path.name[:-5].replace("__", ":", 1)
        for path in args.cache.glob("train__*.json")
        if not path.name.endswith(".intent.json")
    }
    example_inventory = inventory(args.examples, ".pt")
    if any(value != expected for value in (base_inventory, cache_inventory, metadata_inventory, example_inventory)):
        raise CohortError(
            "fit117 base/cache/example inventory mismatch: "
            f"base={len(base_inventory)} cache={len(cache_inventory)} "
            f"metadata={len(metadata_inventory)} examples={len(example_inventory)}"
        )
    census_master = repair.load_json(args.census)
    census_paths = {
        f"{row['split']}:{row['dataset']}": args.census.parent / str(row["path"])
        for row in census_master["videos"]
    }
    census = repair.census_by_identity(args.census)
    if not expected.issubset(census):
        raise CohortError("G1 census does not cover exact fit117 selection")

    args.output.mkdir(parents=True)
    fragment_root = args.output / "fragments"
    fragment_root.mkdir()
    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sha256(args.edge_artifact),
        args.edge_auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise CohortError("frozen fit117 edge scorer did not restore")
    if edge_record.get("sha256") != "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034":
        raise CohortError("frozen edge scorer identity drifted")

    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(selected, start=1):
        slug = identity.replace(":", "__", 1)
        base_path = args.base_fragments / f"{slug}.csv"
        cache_path = args.cache / f"{slug}.bhc"
        metadata_path = args.cache / f"{slug}.json"
        example_path = args.examples / f"{slug}.pt"
        output_path = fragment_root / f"{slug}.csv"
        base_rows = repair.read_fragment(base_path)
        video = census[identity]
        flagged = {
            int(row["frame"])
            for row in video["transitions"]
            if bool(row["pool_breaking"])
        }
        detail: dict[str, int] = {
            "removed_raw_edges": 0,
            "added_registered_edges": 0,
            "removed_conflicting_synthetic_edges": 0,
            "skipped_pruned_pairs": 0,
        }
        if flagged:
            cache = repair.load_cache(args.cache, identity)
            plan = load_registration_plan(args.census, identity)
            registered = repair.translated_frames(cache, video["transitions"], plan=plan)
            probabilities, no_link, scored = repair.score_pairs(cache, registered, scorer)
            replacements = repair.registered_local_edges(registered, probabilities, no_link, flagged)
            raw_ids = {int(node_id) for frame in cache.detections for node_id in frame.node_ids}
            output_rows, detail = repair.apply_registered(base_rows, replacements, raw_ids)
            scored_count = sum(int(scored[frame].stats.candidate_count) for frame in flagged)
        else:
            output_rows = [dict(row) for row in base_rows]
            scored_count = 0
        repair.write_fragment(output_path, output_rows)
        dataset, frames, edges = __import__(
            "run_velocity_window_audit"
        ).read_fragment(output_path)
        if identity != f"train:{dataset}" or set(frames) != set(range(100)):
            raise CohortError(f"sealed output graph identity drifted: {identity}")
        for key, value in detail.items():
            counts[key] += int(value)
        counts["pool_breaking_transitions"] += len(flagged)
        counts["videos_with_pool_breaking_transitions"] += int(bool(flagged))
        counts["nodes"] += sum(len(frame[0]) for frame in frames.values())
        counts["edges"] += len(edges)
        record = {
            "ordinal": ordinal,
            "identity": identity,
            "partition": videos[ordinal - 1]["partition"],
            "pool_breaking_transitions": len(flagged),
            "scored_candidate_count": scored_count,
            "graph": {"nodes": sum(len(frame[0]) for frame in frames.values()), "edges": len(edges)},
            "g1_edit": detail,
            "inputs": {
                "base_fragment": {"path": str(base_path), "bytes": base_path.stat().st_size, "sha256": sha256(base_path)},
                "cache": {"path": str(cache_path), "bytes": cache_path.stat().st_size, "sha256": sha256(cache_path)},
                "cache_metadata": {"path": str(metadata_path), "bytes": metadata_path.stat().st_size, "sha256": sha256(metadata_path)},
                "shell_example": {
                    "path": str(example_path),
                    "bytes": example_path.stat().st_size,
                    "sha256": sha256(example_path),
                    "prepare_sha256": str(videos[ordinal - 1]["sha256"]),
                },
                "registration_census": {
                    "path": str(census_paths[identity]),
                    "sha256": sha256(census_paths[identity]),
                },
            },
            "output": {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": sha256(output_path)},
        }
        if record["inputs"]["shell_example"]["sha256"] != record["inputs"]["shell_example"]["prepare_sha256"]:
            raise CohortError(f"shell example identity drifted: {identity}")
        records.append(record)
        print(f"fit117 G1 {ordinal:03d}/117 {identity} jumps={len(flagged)}", flush=True)

    output_inventory = inventory(fragment_root, ".csv")
    if output_inventory != expected or len(list(fragment_root.glob("*.csv"))) != 117:
        raise CohortError("sealed G1 output inventory is not exactly fit117")
    manifest = {
        "schema": "celltrack-alpha.velocity-aware-fit117-g1-cohort.v1",
        "status": "PASS_SEALED_EXACT_FIT117_G1_SELECTED_GRAPH",
        "selection": {
            "count": 117,
            "identities": list(selected),
            "train_count": len(scope["train_video_ids"]),
            "validation_count": len(scope["validation_video_ids"]),
            "family_counts": dict(Counter(value.split(":", 1)[1].split("_", 1)[0] for value in selected)),
        },
        "contract": {
            "base": "actual fit117 selected graph",
            "g1": "whole-frame transform before fixed-r10/cap8 candidate construction; unchanged scorer and legacy graph matcher",
            "cv57_substitution": False,
            "labels_read": False,
            "shell_positive_count": 456,
            "expected_post_g1_residual_shell": 366,
            "conservation_guard_relaxed": False,
        },
        "totals": dict(counts),
        "aggregate": {
            "algorithm": "sha256(canonical per-video records)",
            "sha256": aggregate(records),
        },
        "videos": records,
        "provenance": {
            "prepare": {"path": str(args.prepare), "sha256": sha256(args.prepare)},
            "census": {"path": str(args.census), "sha256": sha256(args.census)},
            "edge_scorer": edge_record,
            "edge_auxiliary": {"path": str(args.edge_auxiliary), "sha256": sha256(args.edge_auxiliary)},
            "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "scorer_called_only_for_pool_breaking_videos": True,
            "gt_archive_opened": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    write_new(args.output / "SEALED_COHORT.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--base-fragments", type=Path, default=DEFAULT_BASE)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    value.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": report["status"], "totals": report["totals"]}, sort_keys=True))

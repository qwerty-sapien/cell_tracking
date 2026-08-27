#!/usr/bin/env python3
"""Compare two preregistered APNCC direct-swap policies on fit-validation12."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import appearance_swap, bh_edge, bh_graph, bh_model  # noqa: E402
from evaluate_selected_graph_conflict_gate import _edge_counts, _realization  # noqa: E402
from evaluate_selected_graph_direct_swap import _solve_direct  # noqa: E402


SCHEMA = "celltrack-alpha.appearance-direct-policy-fit-screen.v1"
DEFAULT_FIT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
POLICY_SHA256 = (
    "8e546fb86f245eaa3a40a45c4c830d3e639b50001ff54814cccba027f8115b65"
)


class PolicyFitError(RuntimeError):
    """The fit-only policy comparison contract was violated."""


def _policy(value: Any, name: str) -> tuple[float, int, float]:
    if not isinstance(value, Mapping):
        raise PolicyFitError(f"{name} policy is missing")
    cap = value.get("proposal_cap")
    if not isinstance(cap, Mapping):
        raise PolicyFitError(f"{name} proposal cap is missing")
    try:
        threshold = float(value["appearance_threshold"])
        minimum = int(cap["minimum_per_video"])
        fraction = float(cap["selected_edge_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyFitError(f"{name} policy is malformed") from exc
    if minimum <= 0 or not 0.0 < fraction <= 0.02:
        raise PolicyFitError(f"{name} proposal cap is outside bounds")
    return threshold, minimum, fraction


def load_policy(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sweep._sha256(path) != expected_sha256:
        raise PolicyFitError("policy manifest SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="ascii"))
    if value.get("schema") != "celltrack-alpha.appearance-direct-policy-fit.v1":
        raise PolicyFitError("unexpected policy schema")
    control = _policy(value.get("control"), "control")
    candidate = _policy(value.get("candidate"), "candidate")
    if candidate[0] >= control[0] or candidate[1:] != control[1:]:
        raise PolicyFitError("candidate must lower only the appearance threshold")
    promotion = value.get("promotion")
    if not isinstance(promotion, Mapping):
        raise PolicyFitError("promotion contract is missing")
    return value


def _cap(policy: tuple[float, int, float], selected_edges: int) -> int:
    return max(policy[1], int(math.ceil(policy[2] * selected_edges)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise PolicyFitError(f"refusing to overwrite output: {args.output}")
    policy = load_policy(args.policy, args.policy_sha256)
    control_policy = _policy(policy["control"], "control")
    candidate_policy = _policy(policy["candidate"], "candidate")

    source = json.loads(args.selection_report.read_text(encoding="ascii"))
    scope = source.get("scope")
    if (
        source.get("schema") != "celltrack-alpha.selected-graph-appearance-audit.v1"
        or not isinstance(scope, Mapping)
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or len(scope.get("validation_video_ids", [])) != 12
    ):
        raise PolicyFitError("fit-validation selection report is inadmissible")
    selected_ids = tuple(str(value) for value in scope["validation_video_ids"])

    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = set(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.zip),
        )
    )
    overlap = sorted(set(selected_ids).intersection(cv57))
    if overlap:
        raise PolicyFitError(f"fit-validation selection overlaps CV57: {overlap}")

    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected_ids).issubset(cache_paths) or not set(
        selected_ids
    ).issubset(meta_paths):
        raise PolicyFitError("cache coverage is incomplete")

    contract, runtime_gate = appearance_swap.load_contract(
        args.runtime_contract,
        expected_sha256=args.runtime_contract_sha256,
        root=ROOT,
    )
    if (
        float(contract["appearance"]["threshold"]) != control_policy[0]
        or int(contract["proposal_cap"]["minimum_per_video"])
        != control_policy[1]
        or float(contract["proposal_cap"]["selected_edge_fraction"])
        != control_policy[2]
    ):
        raise PolicyFitError("control policy drifted from the banked contract")

    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise PolicyFitError("frozen edge scorer did not restore")

    args.output.mkdir(parents=True)
    control_dir = args.output / "control_fragments"
    candidate_dir = args.output / "candidate_fragments"
    control_dir.mkdir()
    candidate_dir.mkdir()
    videos: list[dict[str, Any]] = []
    started = time.perf_counter()

    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected_ids, start=1):
            dataset = identity.split(":", 1)[1]
            meta = sweep._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            probabilities, no_link, association = run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=bh_edge
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            adjacent_edges, adjacent_stats = bh_graph.build_adjacent_edges(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                topology_mode="legacy",
            )
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            refiner = sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = bh_graph.close_one_frame_gaps(
                graph_nodes,
                adjacent_edges,
                image_refine_callback=refiner,
            )
            _nodes, baseline_edges, _post = bh_graph.postprocess_graph(
                gap_nodes, gap_edges
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            baseline_selected = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids
                and int(edge.target_id) in raw_ids
            }
            candidate_cap = _cap(candidate_policy, len(baseline_selected))
            all_proposals, proposal_counts = appearance_swap.propose_swaps(
                cache,
                scorer,
                baseline_selected,
                runtime_gate,
                image,
                geometry_threshold=float(contract["geometry_gate"]["threshold"]),
                appearance_threshold=candidate_policy[0],
                maximum=max(1, len(baseline_selected)),
            )
            candidate_proposals = all_proposals[:candidate_cap]
            control_cap = _cap(control_policy, len(baseline_selected))
            control_proposals = [
                row
                for row in all_proposals
                if float(row["confidence"]) >= control_policy[0]
            ][:control_cap]
            control = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=tuple(int(value) for value in image.shape),
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=control_proposals,
                probabilities=probabilities,
                fragment_dir=control_dir,
            )
            candidate = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=tuple(int(value) for value in image.shape),
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=candidate_proposals,
                probabilities=probabilities,
                fragment_dir=candidate_dir,
            )
            control_realization = _realization(
                Path(control["fragment"]), control_proposals
            )
            candidate_realization = _realization(
                Path(candidate["fragment"]), candidate_proposals
            )
            if (
                control_realization["retained_conflicts"] != 0
                or candidate_realization["retained_conflicts"] != 0
            ):
                raise PolicyFitError(
                    f"direct swap realization failed for {identity}"
                )
            videos.append(
                {
                    "identity": identity,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_generation": proposal_counts,
                    "control_proposals": len(control_proposals),
                    "candidate_proposals": len(candidate_proposals),
                    "control_realization": control_realization,
                    "candidate_realization": candidate_realization,
                }
            )
            print(
                f"policy fit {index:02d}/{len(selected_ids)} {identity} "
                f"control={len(control_proposals)} "
                f"candidate={len(candidate_proposals)}",
                flush=True,
            )

        control_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(control_dir)), selected_ids
        )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(candidate_dir)), selected_ids
        )
        control_rows, control_provenance, control_coverage = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )

    control_summary = scoreall.official_summary(control_rows)
    candidate_summary = scoreall.official_summary(candidate_rows)
    adjusted_delta = float(candidate_summary["adj_edge_jaccard"]) - float(
        control_summary["adj_edge_jaccard"]
    )
    raw_delta = float(candidate_summary["edge_jaccard"]) - float(
        control_summary["edge_jaccard"]
    )
    recall_delta = float(candidate_summary["node_recall"]) - float(
        control_summary["node_recall"]
    )
    promotion = policy["promotion"]
    passed = (
        adjusted_delta
        >= float(promotion["minimum_adjusted_edge_jaccard_delta"])
        and (
            not bool(promotion["require_nonnegative_raw_edge_jaccard_delta"])
            or raw_delta >= 0.0
        )
        and recall_delta >= float(promotion["minimum_node_recall_delta"])
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "role": "untouched-fit-validation12",
            "video_ids": list(selected_ids),
            "cv57_overlap": overlap,
        },
        "policies": {
            "control": policy["control"],
            "candidate": policy["candidate"],
        },
        "control": {
            "summary": control_summary,
            "edge_counts": _edge_counts(control_rows),
            "coverage": control_coverage,
            "input_provenance": control_provenance,
        },
        "candidate": {
            "summary": candidate_summary,
            "edge_counts": _edge_counts(candidate_rows),
            "coverage": candidate_coverage,
            "input_provenance": candidate_provenance,
        },
        "videos": videos,
        "decision": {
            "adjusted_edge_jaccard_delta": adjusted_delta,
            "raw_edge_jaccard_delta": raw_delta,
            "node_recall_delta": recall_delta,
            "promotion_contract": promotion,
            "status": "CONFIRM-CV57" if passed else "PARK",
        },
        "provenance": {
            "policy_manifest": str(args.policy),
            "policy_manifest_sha256": args.policy_sha256,
            "selection_report": str(args.selection_report),
            "selection_report_sha256": sweep._sha256(args.selection_report),
            "runtime_contract": str(args.runtime_contract),
            "runtime_contract_sha256": contract["contract_sha256"],
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/appearance_direct_policy_fit_v1.json",
    )
    value.add_argument("--policy-sha256", default=POLICY_SHA256)
    value.add_argument(
        "--selection-report",
        type=Path,
        default=ROOT / "reports/g02/selected_graph_appearance_audit_v1.json",
    )
    value.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "reports/g02/appearance_direct_swap_contract_v1.json",
    )
    value.add_argument(
        "--runtime-contract-sha256",
        default=(
            "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7"
        ),
    )
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=DEFAULT_FIT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument(
        "--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY
    )
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/appearance_direct_policy_fit_v1",
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report["decision"], sort_keys=True))

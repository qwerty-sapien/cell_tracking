#!/usr/bin/env python3
"""Audit label-free full-pool Trackastra endpoint and token coverage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import appearance_swap, bh_edge, bh_model  # noqa: E402
from evaluate_trackastra_zero_wrfeat_signal import (  # noqa: E402
    select_window_nodes,
    sha256,
)
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-full-pool-feasibility.v1"
CONTRACT_SCHEMA = (
    "celltrack-alpha.trackastra-full-pool-feasibility-contract.v1"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/trackastra_full_pool_feasibility_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "0bb92b8ddc899b387d33ddc3a0327f99e4c1c3de271543c2d364c3d696f4b761"
)


class TrackastraFullPoolError(RuntimeError):
    """The sealed full-pool feasibility audit failed closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackastraFullPoolError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise TrackastraFullPoolError(f"{label} must be a JSON object")
    return value


def _require(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise TrackastraFullPoolError(f"{label} identity mismatch: {path}")


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path, expected_sha256, "contract")
    value = _load_json(path, "contract")
    pool = value.get("proposal_pool")
    window = value.get("trackastra_window")
    scope = value.get("scope")
    gate = value.get("gate")
    runtime = value.get("runtime_projection")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (pool, window, scope, gate, runtime)
        )
    ):
        raise TrackastraFullPoolError("contract is malformed")
    if (
        pool.get("appearance_threshold_applied") is not False
        or pool.get("proposal_cap_applied") is not False
        or pool.get("sampled_background") is not False
        or pool.get("label_access") is not False
        or int(window.get("frames", 0)) != 4
        or int(window.get("max_tokens", 0)) != 1024
        or scope.get("fit_train105_only") is not True
        or scope.get("training") is not False
        or any(
            scope.get(key) is not False
            for key in (
                "validation12_access",
                "cv57_access",
                "promotion_authorized",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
        or int(gate.get("expected_video_count", 0)) != 69
    ):
        raise TrackastraFullPoolError("contract is not fail-closed")
    return value


def transition_coverage(
    *,
    groups: Sequence[Mapping[str, Any]],
    by_node: Mapping[int, tuple[int, np.ndarray]],
    frame_nodes: Mapping[int, Sequence[int]],
    window_frames: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Return exact transition endpoint coverage without using labels."""

    by_source_t: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        challenger = tuple(int(value) for value in group["challenger"])
        conflicts = tuple(
            tuple(int(value) for value in pair)
            for pair in group["conflicts"]
        )
        if not conflicts or any(
            node not in by_node
            for pair in (challenger, *conflicts)
            for node in pair
        ):
            raise TrackastraFullPoolError(
                "proposal identity is absent or conflict-free"
            )
        source_t = int(by_node[challenger[0]][0])
        if int(by_node[challenger[1]][0]) != source_t + 1:
            raise TrackastraFullPoolError(
                "challenger is not an adjacent forward link"
            )
        for source, target in conflicts:
            if (
                int(by_node[source][0]) != source_t
                or int(by_node[target][0]) != source_t + 1
            ):
                raise TrackastraFullPoolError(
                    "incumbent escaped the challenger transition"
                )
        by_source_t[source_t].append(group)

    result = []
    for source_t in sorted(by_source_t):
        transition_groups = by_source_t[source_t]
        required = {
            node
            for group in transition_groups
            for pair in (
                tuple(int(value) for value in group["challenger"]),
                *(
                    tuple(int(value) for value in conflict)
                    for conflict in group["conflicts"]
                ),
            )
            for node in pair
        }
        all_frames = sorted(frame_nodes)
        min_frame, max_frame = all_frames[0], all_frames[-1]
        start = min(
            max(source_t - 1, min_frame),
            max(min_frame, max_frame - 3),
        )
        full_window_nodes = {
            int(node)
            for frame in range(start, start + window_frames)
            for node in frame_nodes.get(frame, ())
        }
        required_inside_window = required <= full_window_nodes
        token_cap_pass = len(required) <= max_tokens
        selected: tuple[int, ...] = ()
        cropped = len(full_window_nodes) > max_tokens
        missing = len(required)
        if required_inside_window and token_cap_pass:
            selected, cropped = select_window_nodes(
                by_node=by_node,
                frame_nodes=frame_nodes,
                source_frame=source_t,
                required=required,
                window_frames=window_frames,
                max_tokens=max_tokens,
            )
            missing = len(required - set(selected))
        result.append(
            {
                "source_t": source_t,
                "proposal_count": len(transition_groups),
                "required_node_count": len(required),
                "full_window_node_count": len(full_window_nodes),
                "selected_token_count": len(selected),
                "cropped": cropped,
                "required_inside_window": required_inside_window,
                "token_cap_pass": token_cap_pass,
                "missing_endpoint_count": missing,
            }
        )
    return result


def _filter_geometry(
    groups: Sequence[Mapping[str, Any]],
    gate: torch.nn.Module,
    threshold: float,
) -> list[dict[str, Any]]:
    result = []
    with torch.inference_mode():
        for group in groups:
            relative = np.asarray(
                group["relative_features"], dtype=np.float32
            )
            probabilities = torch.sigmoid(
                gate(torch.from_numpy(relative))
            ).numpy()
            if float(np.min(probabilities)) >= threshold:
                result.append(
                    {
                        "challenger": [
                            int(value) for value in group["challenger"]
                        ],
                        "conflicts": [
                            [int(value) for value in pair]
                            for pair in group["conflicts"]
                        ],
                    }
                )
    return result


def _video(
    *,
    identity: str,
    expected_selected_edges: int,
    cache_root: Path,
    scorer: Any,
    geometry_gate: torch.nn.Module,
    geometry_threshold: float,
    window_frames: int,
    max_tokens: int,
) -> dict[str, Any]:
    slug = identity.replace(":", "__", 1)
    meta_path = cache_root / f"{slug}.json"
    cache_path = cache_root / f"{slug}.bhc"
    meta = _load_json(meta_path, f"{identity} cache metadata")
    cache = bh_model.load_sparse_cache(
        cache_path, expected_identity=sweep._cache_identity(meta)
    )
    selected, _probabilities, _no_link, graph = _baseline(cache, scorer)
    if int(graph["raw_selected_edges"]) != expected_selected_edges:
        raise TrackastraFullPoolError(
            f"{identity} selected graph identity drifted"
        )
    groups, counts = appearance_swap.proposal_feature_groups(
        cache, scorer, selected
    )
    filtered = _filter_geometry(
        groups, geometry_gate, geometry_threshold
    )
    by_node: dict[int, tuple[int, np.ndarray]] = {}
    frame_nodes: dict[int, tuple[int, ...]] = {}
    for frame in cache.detections:
        ids = tuple(int(value) for value in frame.node_ids)
        frame_nodes[int(frame.frame_index)] = ids
        for node, coordinate in zip(
            ids, np.asarray(frame.coords_zyx), strict=True
        ):
            if node in by_node:
                raise TrackastraFullPoolError(
                    f"{identity} repeats a node identity"
                )
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )
    transitions = transition_coverage(
        groups=filtered,
        by_node=by_node,
        frame_nodes=frame_nodes,
        window_frames=window_frames,
        max_tokens=max_tokens,
    )
    report = {
        "identity": identity,
        "raw_selected_edges": len(selected),
        "candidate_pairs": int(counts["candidate_pairs"]),
        "raw_proposals": int(counts["raw_proposals"]),
        "geometry_proposals": len(filtered),
        "nonempty_transitions": len(transitions),
        "cropped_transitions": sum(
            int(row["cropped"]) for row in transitions
        ),
        "maximum_required_nodes": max(
            (int(row["required_node_count"]) for row in transitions),
            default=0,
        ),
        "maximum_full_window_nodes": max(
            (int(row["full_window_node_count"]) for row in transitions),
            default=0,
        ),
        "missing_endpoint_count": sum(
            int(row["missing_endpoint_count"]) for row in transitions
        ),
        "token_cap_failures": sum(
            not bool(row["token_cap_pass"]) for row in transitions
        ),
        "window_membership_failures": sum(
            not bool(row["required_inside_window"]) for row in transitions
        ),
        "transitions": transitions,
    }
    print(
        f"{identity}: {len(filtered)} proposals / {len(transitions)} "
        f"transitions / max required {report['maximum_required_nodes']}",
        flush=True,
    )
    del groups, filtered, transitions, cache
    gc.collect()
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_contract(args.contract, args.contract_sha256)
    inputs = contract["inputs"]
    appearance_path = ROOT / str(inputs["appearance_contract"]["path"])
    prepare_path = ROOT / str(inputs["fit_prepare"]["path"])
    signal_path = ROOT / str(inputs["trackastra_signal_report"]["path"])
    _require(
        appearance_path,
        str(inputs["appearance_contract"]["sha256"]),
        "appearance contract",
    )
    _require(
        prepare_path,
        str(inputs["fit_prepare"]["sha256"]),
        "fit prepare",
    )
    _require(
        signal_path,
        str(inputs["trackastra_signal_report"]["sha256"]),
        "signal report",
    )
    prepare = _load_json(prepare_path, "fit prepare")
    scope = prepare.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise TrackastraFullPoolError("fit prepare scope drifted")
    eligible = tuple(
        str(identity)
        for identity in scope["train_video_ids"]
        if str(identity).startswith("train:6bba_")
    )
    if len(eligible) != int(contract["gate"]["expected_video_count"]):
        raise TrackastraFullPoolError("eligible video count drifted")
    rows = {str(row["identity"]): row for row in prepare["videos"]}
    if not set(eligible) <= set(rows):
        raise TrackastraFullPoolError("fit prepare video coverage is incomplete")

    appearance_contract, geometry_gate = appearance_swap.load_contract(
        appearance_path,
        expected_sha256=str(inputs["appearance_contract"]["sha256"]),
        root=ROOT,
    )
    threshold = float(appearance_contract["geometry_gate"]["threshold"])
    if threshold != float(contract["proposal_pool"]["geometry_threshold"]):
        raise TrackastraFullPoolError("geometry threshold drifted")
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
        raise TrackastraFullPoolError("frozen edge scorer did not restore")

    window_frames = int(contract["trackastra_window"]["frames"])
    max_tokens = int(contract["trackastra_window"]["max_tokens"])
    videos = []
    for identity in eligible:
        videos.append(
            _video(
                identity=identity,
                expected_selected_edges=int(
                    rows[identity]["graph"]["raw_selected_edges"]
                ),
                cache_root=args.cache,
                scorer=scorer,
                geometry_gate=geometry_gate,
                geometry_threshold=threshold,
                window_frames=window_frames,
                max_tokens=max_tokens,
            )
        )
    total_calls = sum(int(row["nonempty_transitions"]) for row in videos)
    runtime = contract["runtime_projection"]
    projected_seconds = (
        total_calls
        * float(runtime["seconds_per_1024_token_call"])
        * float(runtime["safety_factor"])
    )
    projected_hours = projected_seconds / 3600.0
    every_nonempty = all(
        int(row["geometry_proposals"]) > 0
        and int(row["nonempty_transitions"]) > 0
        for row in videos
    )
    token_cap_pass = all(
        int(row["token_cap_failures"]) == 0 for row in videos
    )
    endpoint_pass = all(
        int(row["missing_endpoint_count"]) == 0
        and int(row["window_membership_failures"]) == 0
        for row in videos
    )
    runtime_pass = projected_hours <= float(
        runtime["maximum_projected_hours"]
    )
    passed = (
        len(videos) == int(contract["gate"]["expected_video_count"])
        and every_nonempty
        and token_cap_pass
        and endpoint_pass
        and runtime_pass
    )
    return {
        "schema": SCHEMA,
        "verdict": (
            "PASS_GROUPED_OOF_DESIGN_AUTHORIZED"
            if passed
            else "FAIL_REDESIGN_OR_PARK"
        ),
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
        "implementation_sha256": sha256(Path(__file__)),
        "scope": contract["scope"],
        "aggregate": {
            "video_count": len(videos),
            "raw_proposals": sum(
                int(row["raw_proposals"]) for row in videos
            ),
            "geometry_proposals": sum(
                int(row["geometry_proposals"]) for row in videos
            ),
            "inference_calls": total_calls,
            "cropped_transitions": sum(
                int(row["cropped_transitions"]) for row in videos
            ),
            "maximum_required_nodes": max(
                int(row["maximum_required_nodes"]) for row in videos
            ),
            "maximum_full_window_nodes": max(
                int(row["maximum_full_window_nodes"]) for row in videos
            ),
            "missing_endpoint_count": sum(
                int(row["missing_endpoint_count"]) for row in videos
            ),
            "projected_seconds": projected_seconds,
            "projected_hours": projected_hours,
        },
        "gate": {
            "every_video_nonempty": every_nonempty,
            "token_cap_pass": token_cap_pass,
            "endpoint_coverage_pass": endpoint_pass,
            "runtime_projection_pass": runtime_pass,
            "passed": passed,
        },
        "execution": {
            "device": "cpu",
            "training": False,
            "label_access": False,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "videos": videos,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.contract = args.contract.resolve()
    args.cache = args.cache.resolve()
    args.edge_artifact = args.edge_artifact.resolve()
    args.edge_auxiliary = args.edge_auxiliary.resolve()
    args.support_root = args.support_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise TrackastraFullPoolError(
            f"refusing to overwrite output: {args.output}"
        )
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report["aggregate"], sort_keys=True))
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

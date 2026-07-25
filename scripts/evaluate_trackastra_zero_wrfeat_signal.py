#!/usr/bin/env python3
"""Evaluate frozen Trackastra association signal on a sealed fit-only probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any
import zipfile

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha import bh_model  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-zero-wrfeat-signal.v1"
CONTRACT_SCHEMA = (
    "celltrack-alpha.trackastra-zero-wrfeat-signal-contract.v1"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/trackastra_zero_wrfeat_signal_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "7404891ab6396e3c021653e8df1313e25590b9891d4c496588284206be85c76d"
)
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"


class TrackastraSignalError(RuntimeError):
    """The sealed frozen-checkpoint signal probe failed closed."""


def _cache_identity(meta: Mapping[str, Any]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        return bh_model.CacheIdentity(
            str(model["config_sha256"]),
            str(model["source_sha256"]),
            str(model["weight_sha256"]),
            int(model["pair_batch_size"]),
            int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackastraSignalError(
            "cache metadata lacks a valid model identity"
        ) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackastraSignalError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TrackastraSignalError(f"{label} must be a JSON object")
    return value


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise TrackastraSignalError("contract SHA-256 mismatch")
    value = _load_json(path, "contract")
    scope = value.get("scope")
    probe = value.get("probe")
    gate = value.get("gate")
    data = value.get("data")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (scope, probe, gate, data)
        )
    ):
        raise TrackastraSignalError("contract is malformed")
    if (
        scope.get("fit_train105_only") is not True
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
        or probe.get("checkpoint_frozen") is not True
        or probe.get("training") is not False
        or int(probe.get("window_frames", 0)) != 4
        or int(probe.get("max_tokens", 0)) != 1024
    ):
        raise TrackastraSignalError("contract scope is not fail-closed")
    videos = data.get("videos")
    if not isinstance(videos, list) or len(videos) != 5:
        raise TrackastraSignalError("contract must name exactly five videos")
    return value


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise TrackastraSignalError(f"{label} identity mismatch: {path}")


def _install_pinned_source(source_root: Path, source_files: Mapping[str, Any]):
    for relative, expected in source_files.items():
        _require_file(
            source_root / str(relative), str(expected), f"source {relative}"
        )
    if any(
        name == "trackastra" or name.startswith("trackastra.")
        for name in sys.modules
    ):
        raise TrackastraSignalError(
            "refusing to mix pinned Trackastra with an existing import"
        )
    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
    package = types.ModuleType("trackastra")
    package.__path__ = [str(source_root / "trackastra")]
    package.__package__ = "trackastra"
    sys.modules["trackastra"] = package
    model_package = types.ModuleType("trackastra.model")
    model_package.__path__ = [str(source_root / "trackastra/model")]
    model_package.__package__ = "trackastra.model"
    sys.modules["trackastra.model"] = model_package
    try:
        module = importlib.import_module("trackastra.model.model")
    except Exception as exc:
        raise TrackastraSignalError(
            "cannot import the pinned Trackastra source"
        ) from exc
    return module.TrackingTransformer


def _load_model(
    source_root: Path,
    checkpoint_zip: Path,
    contract: Mapping[str, Any],
) -> torch.nn.Module:
    asset = contract["asset_audit"]
    _require_file(
        checkpoint_zip,
        str(asset["checkpoint_archive_sha256"]),
        "checkpoint archive",
    )
    TrackingTransformer = _install_pinned_source(
        source_root, contract["source_files"]
    )
    try:
        with zipfile.ZipFile(checkpoint_zip) as archive:
            config = yaml.safe_load(archive.read("ctc/config.yaml"))
            payload = archive.read("ctc/model.pt")
    except (OSError, KeyError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        raise TrackastraSignalError(
            "cannot load checkpoint archive"
        ) from exc
    if (
        not isinstance(config, dict)
        or hashlib.sha256(payload).hexdigest()
        != asset["checkpoint_member_sha256"]
    ):
        raise TrackastraSignalError("checkpoint member identity mismatch")
    model = TrackingTransformer(**config)
    state = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def grouped_rows(
    challenger_pairs: np.ndarray,
    incumbent_pairs: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Group conflict rows by challenger without observing label values."""

    challenger_pairs = np.asarray(challenger_pairs, dtype=np.int64)
    incumbent_pairs = np.asarray(incumbent_pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        challenger_pairs.shape != incumbent_pairs.shape
        or challenger_pairs.shape != (len(labels), 2)
        or np.any(~np.isin(labels, (0, 1)))
    ):
        raise TrackastraSignalError("prepared conflict rows are malformed")
    indices: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(challenger_pairs):
        indices.setdefault((int(pair[0]), int(pair[1])), []).append(index)
    result = []
    for challenger, rows in indices.items():
        observed = {int(labels[index]) for index in rows}
        if len(observed) != 1:
            raise TrackastraSignalError(
                "one challenger has contradictory labels"
            )
        result.append(
            {
                "challenger": challenger,
                "incumbents": tuple(
                    (
                        int(incumbent_pairs[index, 0]),
                        int(incumbent_pairs[index, 1]),
                    )
                    for index in rows
                ),
                "rows": tuple(rows),
                "label": observed.pop(),
            }
        )
    return result


def select_window_nodes(
    *,
    by_node: Mapping[int, tuple[int, np.ndarray]],
    frame_nodes: Mapping[int, Sequence[int]],
    source_frame: int,
    required: set[int],
    window_frames: int,
    max_tokens: int,
) -> tuple[tuple[int, ...], bool]:
    """Select a deterministic full or proposal-centered Trackastra window."""

    all_frames = sorted(frame_nodes)
    if not all_frames:
        raise TrackastraSignalError("cache has no detection frames")
    min_frame, max_frame = all_frames[0], all_frames[-1]
    start = min(max(source_frame - 1, min_frame), max(min_frame, max_frame - 3))
    allowed_frames = tuple(
        frame for frame in range(start, start + window_frames)
        if frame in frame_nodes
    )
    candidates = {
        int(node)
        for frame in allowed_frames
        for node in frame_nodes[frame]
    }
    if not required <= candidates:
        raise TrackastraSignalError(
            "proposal endpoint is outside its Trackastra window"
        )
    if len(candidates) <= max_tokens:
        selected = candidates
        cropped = False
    else:
        if len(required) > max_tokens:
            raise TrackastraSignalError(
                "proposal endpoints exceed the token cap"
            )
        centroid = np.mean(
            [np.asarray(by_node[node][1], dtype=np.float64) for node in required],
            axis=0,
        )
        optional = sorted(
            candidates - required,
            key=lambda node: (
                float(
                    np.sum(
                        (
                            np.asarray(by_node[node][1], dtype=np.float64)
                            - centroid
                        )
                        ** 2
                    )
                ),
                int(by_node[node][0]),
                int(node),
            ),
        )
        selected = set(required)
        selected.update(optional[: max_tokens - len(required)])
        cropped = True
    ordered = tuple(
        sorted(selected, key=lambda node: (int(by_node[node][0]), int(node)))
    )
    return ordered, cropped


def _association(
    model: torch.nn.Module,
    nodes: Sequence[int],
    by_node: Mapping[int, tuple[int, np.ndarray]],
) -> tuple[np.ndarray, dict[int, int]]:
    coordinates = np.stack(
        [
            np.concatenate(
                (
                    np.asarray((by_node[node][0],), dtype=np.float32),
                    np.asarray(by_node[node][1], dtype=np.float32),
                )
            )
            for node in nodes
        ]
    )
    timepoints = torch.from_numpy(coordinates[:, 0].astype(np.int64))
    coords = torch.from_numpy(coordinates).unsqueeze(0)
    features = torch.zeros((1, len(nodes), 12), dtype=torch.float32)
    with torch.inference_mode():
        logits = model(coords, features=features)
        association = model.normalize_output(
            logits, timepoints.unsqueeze(0), coords
        )[0]
    value = association.detach().cpu().numpy().astype(np.float64)
    if value.shape != (len(nodes), len(nodes)) or not np.isfinite(value).all():
        raise TrackastraSignalError(
            "Trackastra returned a malformed association matrix"
        )
    return value, {int(node): index for index, node in enumerate(nodes)}


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (
            float(ranks[labels == 1].sum())
            - positive * (positive + 1) / 2
        )
        / (positive * negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels == 1))
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]
    total_tp = total_fp = 0
    previous_recall = 0.0
    result = 0.0
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and scores[end] == scores[cursor]:
            end += 1
        block = labels[cursor:end]
        total_tp += int(np.sum(block == 1))
        total_fp += int(np.sum(block == 0))
        recall = total_tp / positive
        precision = total_tp / (total_tp + total_fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return float(result)


def _metric_bundle(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise TrackastraSignalError("metric arrays are malformed")
    return {
        "count": len(labels),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "roc_auc": (
            roc_auc(labels, scores)
            if len(set(int(value) for value in labels)) == 2
            else None
        ),
        "average_precision": (
            average_precision(labels, scores)
            if int(np.sum(labels == 1)) > 0
            else None
        ),
    }


def _video(
    *,
    video_contract: Mapping[str, Any],
    conflict_row: Mapping[str, Any],
    temporal_row: Mapping[str, Any],
    cache_root: Path,
    model: torch.nn.Module,
    window_frames: int,
    max_tokens: int,
) -> tuple[dict[str, Any], list[int], list[float], list[float]]:
    identity = str(video_contract["identity"])
    conflict_path = ROOT / str(conflict_row["path"])
    temporal_path = ROOT / str(temporal_row["path"])
    _require_file(
        conflict_path,
        str(video_contract["conflict_sha256"]),
        f"{identity} conflict example",
    )
    _require_file(
        temporal_path,
        str(video_contract["temporal_sha256"]),
        f"{identity} temporal example",
    )
    conflict = torch.load(
        conflict_path, map_location="cpu", weights_only=False
    )
    temporal = torch.load(
        temporal_path, map_location="cpu", weights_only=False
    )
    challenger_pairs = np.asarray(
        conflict["challenger_pairs"], dtype=np.int64
    )
    incumbent_pairs = np.asarray(conflict["selected_pairs"], dtype=np.int64)
    labels = np.asarray(conflict["labels"], dtype=np.int8)
    temporal_pairs = np.asarray(
        temporal["labeled_challenger_pairs"], dtype=np.int64
    )
    temporal_labels = np.asarray(temporal["labeled_labels"], dtype=np.int8)
    temporal_features = np.asarray(
        temporal["labeled_features"], dtype=np.float32
    )
    if (
        not np.array_equal(challenger_pairs, temporal_pairs)
        or not np.array_equal(labels, temporal_labels)
        or temporal_features.shape != (len(labels), 48)
    ):
        raise TrackastraSignalError(
            f"{identity} parent prepared artifacts disagree"
        )
    groups = grouped_rows(challenger_pairs, incumbent_pairs, labels)

    slug = identity.replace(":", "__", 1)
    meta = _load_json(cache_root / f"{slug}.json", f"{identity} cache metadata")
    cache = bh_model.load_sparse_cache(
        cache_root / f"{slug}.bhc",
        expected_identity=_cache_identity(meta),
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
                raise TrackastraSignalError(
                    f"{identity} repeats a detection ID"
                )
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )

    by_transition: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        source, target = group["challenger"]
        if source not in by_node or target not in by_node:
            raise TrackastraSignalError(
                f"{identity} proposal endpoint is absent from cache"
            )
        source_frame = int(by_node[source][0])
        if int(by_node[target][0]) != source_frame + 1:
            raise TrackastraSignalError(
                f"{identity} challenger is not an adjacent forward link"
            )
        by_transition[source_frame].append(group)

    output_labels: list[int] = []
    trackastra_scores: list[float] = []
    pair_ncc_scores: list[float] = []
    inference_calls = cropped_calls = peak_tokens = 0
    for source_frame in sorted(by_transition):
        transition_groups = by_transition[source_frame]
        transition_required = {
            node
            for group in transition_groups
            for pair in (group["challenger"], *group["incumbents"])
            for node in pair
        }
        common_nodes, common_cropped = select_window_nodes(
            by_node=by_node,
            frame_nodes=frame_nodes,
            source_frame=source_frame,
            required=transition_required,
            window_frames=window_frames,
            max_tokens=max_tokens,
        )
        common = None
        if not common_cropped:
            common = _association(model, common_nodes, by_node)
            inference_calls += 1
            peak_tokens = max(peak_tokens, len(common_nodes))
        for group in transition_groups:
            if common is None:
                required = {
                    node
                    for pair in (group["challenger"], *group["incumbents"])
                    for node in pair
                }
                nodes, cropped = select_window_nodes(
                    by_node=by_node,
                    frame_nodes=frame_nodes,
                    source_frame=source_frame,
                    required=required,
                    window_frames=window_frames,
                    max_tokens=max_tokens,
                )
                association, node_index = _association(model, nodes, by_node)
                inference_calls += 1
                cropped_calls += int(cropped)
                peak_tokens = max(peak_tokens, len(nodes))
            else:
                association, node_index = common
            challenger = group["challenger"]
            challenger_score = association[
                node_index[challenger[0]], node_index[challenger[1]]
            ]
            margins = [
                challenger_score
                - association[
                    node_index[incumbent[0]], node_index[incumbent[1]]
                ]
                for incumbent in group["incumbents"]
            ]
            trackastra_scores.append(float(min(margins)))
            pair_ncc_scores.append(
                float(
                    min(
                        temporal_features[index, 44]
                        for index in group["rows"]
                    )
                )
            )
            output_labels.append(int(group["label"]))

    labels_array = np.asarray(output_labels, dtype=np.int8)
    trackastra_array = np.asarray(trackastra_scores, dtype=np.float64)
    ncc_array = np.asarray(pair_ncc_scores, dtype=np.float64)
    report = {
        "identity": identity,
        "raw_selected_edges": int(video_contract["raw_selected_edges"]),
        "labeled_rows": int(video_contract["labeled_rows"]),
        "unique_proposals": len(output_labels),
        "trackastra": _metric_bundle(labels_array, trackastra_array),
        "pair_ncc": _metric_bundle(labels_array, ncc_array),
        "inference_calls": inference_calls,
        "cropped_calls": cropped_calls,
        "peak_tokens": peak_tokens,
    }
    print(
        f"{identity}: {len(output_labels)} proposals, "
        f"{inference_calls} calls, {cropped_calls} cropped",
        flush=True,
    )
    return report, output_labels, trackastra_scores, pair_ncc_scores


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_contract(args.contract, args.contract_sha256)
    audit_record = contract["asset_audit"]
    audit_path = ROOT / str(audit_record["path"])
    _require_file(
        audit_path, str(audit_record["sha256"]), "asset audit"
    )
    audit = _load_json(audit_path, "asset audit")
    if audit.get("verdict") != audit_record["required_verdict"]:
        raise TrackastraSignalError("asset audit verdict drifted")
    data = contract["data"]
    conflict_prepare_path = ROOT / str(data["conflict_prepare"]["path"])
    temporal_prepare_path = ROOT / str(data["temporal_prepare"]["path"])
    _require_file(
        conflict_prepare_path,
        str(data["conflict_prepare"]["sha256"]),
        "conflict prepare",
    )
    _require_file(
        temporal_prepare_path,
        str(data["temporal_prepare"]["sha256"]),
        "temporal prepare",
    )
    conflict_prepare = _load_json(conflict_prepare_path, "conflict prepare")
    temporal_prepare = _load_json(temporal_prepare_path, "temporal prepare")
    conflict_rows = {
        str(row["identity"]): row for row in conflict_prepare["videos"]
    }
    temporal_rows = {
        str(row["identity"]): row for row in temporal_prepare["videos"]
    }
    model = _load_model(args.source_root, args.checkpoint_zip, contract)

    video_reports = []
    labels: list[int] = []
    trackastra_scores: list[float] = []
    pair_ncc_scores: list[float] = []
    for video_contract in data["videos"]:
        identity = str(video_contract["identity"])
        if (
            identity not in conflict_rows
            or identity not in temporal_rows
            or conflict_rows[identity].get("partition") != "train"
            or temporal_rows[identity].get("partition") != "train"
        ):
            raise TrackastraSignalError(
                f"sealed video is absent from fit-train: {identity}"
            )
        result = _video(
            video_contract=video_contract,
            conflict_row=conflict_rows[identity],
            temporal_row=temporal_rows[identity],
            cache_root=args.cache,
            model=model,
            window_frames=int(contract["probe"]["window_frames"]),
            max_tokens=int(contract["probe"]["max_tokens"]),
        )
        video_report, video_labels, video_trackastra, video_ncc = result
        video_reports.append(video_report)
        labels.extend(video_labels)
        trackastra_scores.extend(video_trackastra)
        pair_ncc_scores.extend(video_ncc)

    labels_array = np.asarray(labels, dtype=np.int8)
    trackastra_array = np.asarray(trackastra_scores, dtype=np.float64)
    ncc_array = np.asarray(pair_ncc_scores, dtype=np.float64)
    trackastra_metrics = _metric_bundle(labels_array, trackastra_array)
    ncc_metrics = _metric_bundle(labels_array, ncc_array)
    cutoff = float(
        np.quantile(np.abs(ncc_array), 0.5, method="higher")
    )
    ambiguous = np.abs(ncc_array) <= cutoff
    ambiguous_trackastra = _metric_bundle(
        labels_array[ambiguous], trackastra_array[ambiguous]
    )
    correlation = spearmanr(trackastra_array, ncc_array)
    gate = contract["gate"]
    alternative = gate["alternative_ambiguous_band"]
    support_pass = (
        len(labels_array) >= int(gate["minimum_unique_proposals"])
        and int(np.sum(labels_array == 1))
        >= int(gate["minimum_positive_proposals"])
        and int(np.sum(labels_array == 0))
        >= int(gate["minimum_negative_proposals"])
    )
    full_auc_pass = (
        trackastra_metrics["roc_auc"] is not None
        and ncc_metrics["roc_auc"] is not None
        and float(trackastra_metrics["roc_auc"])
        >= float(gate["minimum_trackastra_roc_auc"])
    )
    gain_pass = (
        full_auc_pass
        and float(trackastra_metrics["roc_auc"])
        - float(ncc_metrics["roc_auc"])
        >= float(gate["minimum_auc_gain_over_pair_ncc"])
    )
    ambiguous_pass = (
        ambiguous_trackastra["count"]
        >= int(alternative["minimum_proposals"])
        and ambiguous_trackastra["positive"]
        >= int(alternative["minimum_positive_proposals"])
        and ambiguous_trackastra["negative"]
        >= int(alternative["minimum_negative_proposals"])
        and ambiguous_trackastra["roc_auc"] is not None
        and float(ambiguous_trackastra["roc_auc"])
        >= float(alternative["minimum_trackastra_roc_auc"])
    )
    passed = support_pass and full_auc_pass and (gain_pass or ambiguous_pass)
    return {
        "schema": SCHEMA,
        "verdict": (
            "PASS_FROZEN_HEAD_DESIGN_AUTHORIZED"
            if passed
            else "PARK_ZERO_WRFEAT_ROUTE"
        ),
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
        "implementation_sha256": sha256(Path(__file__)),
        "scope": contract["scope"],
        "probe": contract["probe"],
        "aggregate": {
            "trackastra": trackastra_metrics,
            "pair_ncc": ncc_metrics,
            "trackastra_auc_delta": (
                float(trackastra_metrics["roc_auc"])
                - float(ncc_metrics["roc_auc"])
            ),
            "score_spearman": {
                "rho": (
                    float(correlation.statistic)
                    if math.isfinite(float(correlation.statistic))
                    else None
                ),
                "pvalue": (
                    float(correlation.pvalue)
                    if math.isfinite(float(correlation.pvalue))
                    else None
                ),
            },
            "ambiguous_band": {
                "absolute_pair_ncc_cutoff": cutoff,
                "trackastra": ambiguous_trackastra,
            },
        },
        "gate": {
            "support_pass": support_pass,
            "full_trackastra_auc_pass": full_auc_pass,
            "auc_gain_pass": gain_pass,
            "ambiguous_band_pass": ambiguous_pass,
            "passed": passed,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "inference_calls": sum(
                int(row["inference_calls"]) for row in video_reports
            ),
            "cropped_calls": sum(
                int(row["cropped_calls"]) for row in video_reports
            ),
            "peak_tokens": max(
                int(row["peak_tokens"]) for row in video_reports
            ),
        },
        "videos": video_reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-zip", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.contract = args.contract.resolve()
    args.source_root = args.source_root.resolve()
    args.checkpoint_zip = args.checkpoint_zip.resolve()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise TrackastraSignalError(
            f"refusing to overwrite output: {args.output}"
        )
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

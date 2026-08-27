#!/usr/bin/env python3
"""Run the one allowed fold-0 frozen-alpha versus motion-hybrid comparison.

The held-out image is opened only after split/checkpoint and frozen-alpha
artifact provenance pass.  Adapter center logits are discarded.  The official
GEFF scorer is invoked only after the alpha and hybrid fragments are final, so
held-out labels cannot tune the parameter-free reranker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OFFICIAL = ROOT / "external/kaggle-cell-tracking-competition/src"
for value in (str(SRC), str(OFFICIAL), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from celltrack_alpha.bh_graph import V45_GRAPH_CONFIG  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.ta_hybrid import (  # noqa: E402
    HybridRerankError,
    rerank_alpha_edges,
    sample_child_motion,
)
from celltrack_alpha.ta_inference import (  # noqa: E402
    GATE_SCHEMA,
    GateError,
    HybridAttestation,
    attest_hybrid_checkpoint,
    iter_sliding_frame_outputs,
    sha256_file,
    write_fragment,
)


DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_PROVENANCE = ROOT / "reports/g02/fit_provenance_v1.json"
DEFAULT_REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
DEFAULT_DATASET = "44b6_24264f12"
DEFAULT_ALPHA_FRAGMENT = ROOT / "reports/g02/f4fs1/frags/train__44b6_24264f12.csv"
DEFAULT_ALPHA_RUNTIME = ROOT / "reports/g02/f4fs1/report.json"
DEFAULT_ALPHA_SCORE = ROOT / "reports/g02/f4fs1/scores.json"
DOWN_SAMPLE = (1, 4, 4)
LABELS = ("frozen_alpha", "alpha_motion_hybrid")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise GateError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON artifact is not an object: {path}")
    return value


def _quantile(attrs: Mapping[str, Any], wanted: str) -> float:
    try:
        value = float(attrs["image_statistics"]["quantiles"][wanted])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"image metadata lacks quantile {wanted}") from exc
    if not math.isfinite(value):
        raise GateError(f"image quantile {wanted} is nonfinite")
    return value


def _alpha_rows(path: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    graphs = read_submission(path)
    if set(graphs) != {dataset}:
        raise GateError("frozen alpha fragment does not contain exactly the selected dataset")
    graph = graphs[dataset]
    nodes = [
        {
            "dataset": dataset,
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": int(edge.source_id), "target_id": int(edge.target_id)}
        for edge in graph.edges
    ]
    if not nodes or not edges:
        raise GateError("frozen alpha fragment is empty")
    return nodes, edges


def _attest_alpha_artifact(
    fragment: Path,
    runtime_path: Path,
    score_path: Path,
    dataset: str,
    node_count: int,
    edge_count: int,
) -> dict[str, Any]:
    fragment_hash = sha256_file(fragment)
    score = _strict_json(score_path)
    files = score.get("input_provenance", {}).get("files", ())
    matches = [
        row
        for row in files
        if isinstance(row, Mapping)
        and row.get("identity") == f"train:{dataset}"
        and row.get("sha256") == fragment_hash
    ]
    if len(matches) != 1 or score.get("selected_video_ids") != [f"train:{dataset}"]:
        raise GateError("frozen alpha score evidence does not bind the supplied fragment")
    runtime = _strict_json(runtime_path)
    summary = runtime.get("summary")
    scope = runtime.get("scope")
    if (
        runtime.get("status") != "completed"
        or not isinstance(summary, Mapping)
        or not isinstance(scope, Mapping)
        or scope.get("selected_video_ids") != [f"train:{dataset}"]
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
        or summary.get("videos") != 1
        or summary.get("nodes") != node_count
        or summary.get("edges") != edge_count
        or not math.isfinite(float(summary.get("seconds", float("nan"))))
        or float(summary["seconds"]) <= 0.0
    ):
        raise GateError("frozen alpha runtime evidence does not match the supplied fragment")
    return {
        "fragment": {
            "path": str(fragment.resolve()),
            "sha256": fragment_hash,
            "bytes": fragment.stat().st_size,
            "nodes": node_count,
            "edges": edge_count,
        },
        "score_evidence": {"path": str(score_path.resolve()), "sha256": sha256_file(score_path)},
        "runtime_evidence": {"path": str(runtime_path.resolve()), "sha256": sha256_file(runtime_path)},
        "inference_seconds": float(summary["seconds"]),
    }


def _load_model(attestation: HybridAttestation, device_name: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from tracking_cellmot.models.temporal_unet import TemporalUNet3D

    from celltrack_alpha.t_adapter import load_trained_model

    backbone = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
    model, restored = load_trained_model(
        backbone,
        attestation.candidate.checkpoint_path,
        expected_sha256=attestation.candidate.checkpoint_sha256,
        expected_size=attestation.candidate.checkpoint_bytes,
        expected_fold=attestation.fold,
    )
    if (
        tuple(restored.source_groups) != attestation.candidate.source_groups
        or tuple(restored.source_video_ids) != attestation.candidate.source_video_ids
    ):
        raise GateError("TemporalAdapterModel restore changed the attested fit-source scope")
    device = torch.device(device_name)
    return model.to(device).eval(), {
        "checkpoint_sha256": restored.checkpoint_sha256,
        "fold": restored.fold,
        "scenario": restored.scenario,
        "completed_steps": restored.completed_steps,
        "source_groups": list(restored.source_groups),
        "source_video_ids": list(restored.source_video_ids),
        "model_tensor_count": restored.model_tensor_count,
        "model_numel": restored.model_numel,
        "device": str(device),
    }


def _pair_outputs(
    model: Any,
    array: Any,
    *,
    original_shape: Sequence[int],
    q_low: float,
    q_high: float,
    device_name: str,
) -> Iterator[np.ndarray]:
    import torch

    if q_high <= q_low:
        raise GateError("held-out image quantile range is invalid")
    device = torch.device(device_name)
    total_frames = int(original_shape[0])
    model_shape = (
        int(original_shape[1]),
        math.ceil(int(original_shape[2]) / DOWN_SAMPLE[1]),
        math.ceil(int(original_shape[3]) / DOWN_SAMPLE[2]),
    )
    with torch.inference_mode():
        for start in range(total_frames - 1):
            raw = array[
                start : start + 2,
                :: DOWN_SAMPLE[0],
                :: DOWN_SAMPLE[1],
                :: DOWN_SAMPLE[2],
            ].astype(np.float32)
            if tuple(raw.shape) != (2, *model_shape) or not np.isfinite(raw).all():
                raise GateError("held-out two-frame image slab is missing or nonfinite")
            normalized = np.clip((raw - q_low) / (q_high - q_low), 0.0, 1.0)
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(2).to(device=device)
            output = model(tensor, scenario="universal")[0]
            if tuple(output.shape) != (2, 6, *model_shape) or not bool(torch.isfinite(output).all().item()):
                raise GateError("TemporalAdapterModel inference output schema drift")
            yield output.detach().cpu().numpy()
            if (start + 1) % 10 == 0 or start + 1 == total_frames - 1:
                print(
                    json.dumps({"completed_pairs": start + 1, "total_pairs": total_frames - 1}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )


def _preflight_selection(attestation: HybridAttestation, archive: Path) -> tuple[Any, dict[str, str]]:
    import scoreall

    split = _strict_json(Path(attestation.split_path))
    scorer_hashes = scoreall.verify_scorer_sources(split)
    archive_ids = scoreall.archive_train_video_ids(archive)
    selection = scoreall.select_video_ids(
        split,
        f"fold{attestation.fold}",
        archive_train_video_ids=archive_ids,
        max_datasets=1,
    )
    if selection.selected_video_ids != (attestation.held_out_video_id,):
        raise GateError("comparison is locked to the deterministic first fold-0 held-out video")
    return selection, scorer_hashes


def _score(
    attestation: HybridAttestation,
    archive: Path,
    selection: Any,
    scorer_hashes: Mapping[str, str],
    fragments: Mapping[str, Path],
) -> dict[str, Any]:
    import scoreall
    from zarr.storage import ZipStore

    manifests = {
        label: scoreall.validate_fragment_set([fragments[label]], selection.selected_video_ids)
        for label in LABELS
    }
    store = ZipStore(archive, mode="r")
    try:
        rows: dict[str, list[dict[str, Any]]] = {}
        input_provenance: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for label in LABELS:
            model_rows, provenance, model_coverage = scoreall.score_fragment_input(store, manifests[label])
            rows[label] = model_rows
            input_provenance[label] = provenance
            coverage[label] = model_coverage
    finally:
        store.close()
    summaries = {label: scoreall.official_summary(rows[label]) for label in LABELS}
    delta: dict[str, float | None] = {}
    for key in ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall"):
        hybrid = float(summaries["alpha_motion_hybrid"].get(key, float("nan")))
        alpha = float(summaries["frozen_alpha"].get(key, float("nan")))
        delta[key] = hybrid - alpha if math.isfinite(hybrid) and math.isfinite(alpha) else None
    archive_hash, archive_bytes = scoreall.sha256_with_size(archive)
    improved = delta["score"] is not None and float(delta["score"]) > 0.0
    return {
        "schema": GATE_SCHEMA,
        "kind": "single-fold0-heldout-frozen-alpha-vs-motion-hybrid",
        "status": "completed_improved" if improved else "stopped_no_improvement",
        "qualification": "one-heldout-video-only-not-full-fold-not-audit-not-kaggle-no-promotion",
        "selection": selection.as_dict(),
        "gate_provenance": attestation.json(),
        "archive": {"path": str(archive.resolve()), "sha256": archive_hash, "bytes": archive_bytes},
        "frozen_scorer_source_hashes": dict(scorer_hashes),
        "summaries": summaries,
        "hybrid_minus_alpha": delta,
        "samples": rows,
        "input_provenance": input_provenance,
        "coverage": coverage,
        "stop_rule": {
            "condition": "hybrid official score must be strictly greater than frozen alpha",
            "passed": improved,
            "further_heldout_runs_allowed": False,
        },
        "evidence_boundary": {
            "held_out_video_count": 1,
            "full_video": True,
            "full_fold": False,
            "audit_used": False,
            "test_used": False,
            "kaggle": False,
            "network": False,
            "external_gpu": False,
            "wandb": False,
            "promotion_claim_allowed": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import zarr
    from zarr.storage import ZipStore

    if args.fold != 0:
        raise GateError("this bounded comparison is locked to fold 0")
    if not args.zip.is_file():
        raise GateError("local competition archive is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise GateError("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attestation = attest_hybrid_checkpoint(
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_metrics=args.candidate_metrics,
        pretrain_checkpoint=args.pretrain_checkpoint,
        pretrain_metrics=args.pretrain_metrics,
        split_manifest=args.split_manifest,
        provenance_manifest=args.provenance_manifest,
        registry_manifest=args.registry_manifest,
        fold=args.fold,
        held_out_dataset=args.dataset,
    )
    selection, scorer_hashes = _preflight_selection(attestation, args.zip)
    alpha_nodes, alpha_edges = _alpha_rows(args.alpha_fragment, attestation.held_out_dataset)
    alpha_evidence = _attest_alpha_artifact(
        args.alpha_fragment,
        args.alpha_runtime_report,
        args.alpha_score_evidence,
        attestation.held_out_dataset,
        len(alpha_nodes),
        len(alpha_edges),
    )
    _write_json_new(
        args.output_dir / "preflight.json",
        {"gate": attestation.json(), "frozen_alpha": alpha_evidence},
    )

    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    from celltrack_baseline.rule_based import parse_scale

    store = ZipStore(args.zip, mode="r")
    model = None
    try:
        group = zarr.open_group(store=store, path=f"train/{attestation.held_out_dataset}.zarr", mode="r")
        array = zarr.open_array(store=store, path=f"train/{attestation.held_out_dataset}.zarr/0", mode="r")
        observed_shape = tuple(int(value) for value in array.shape)
        if observed_shape != attestation.held_out_image_shape:
            raise GateError("held-out archive image shape disagrees with the attested registry")
        attrs = dict(group.attrs)
        scale = tuple(float(value) for value in parse_scale(attrs))
        if not np.allclose(scale, V45_GRAPH_CONFIG.scale_zyx, rtol=0.0, atol=1.0e-12):
            raise GateError("held-out physical scale differs from the frozen alpha graph scale")
        q_low, q_high = _quantile(attrs, "0.001"), _quantile(attrs, "0.999")

        load_started = time.perf_counter()
        model, restore = _load_model(attestation, args.device)
        model_load_seconds = time.perf_counter() - load_started
        inference_started = time.perf_counter()
        motions = sample_child_motion(
            alpha_nodes,
            iter_sliding_frame_outputs(
                _pair_outputs(
                    model,
                    array,
                    original_shape=observed_shape,
                    q_low=q_low,
                    q_high=q_high,
                    device_name=args.device,
                )
            ),
            downsample_zyx=DOWN_SAMPLE,
        )
        adapter_inference_seconds = time.perf_counter() - inference_started
    finally:
        store.close()
        del model
        gc.collect()

    rerank_started = time.perf_counter()
    hybrid = rerank_alpha_edges(
        alpha_nodes,
        alpha_edges,
        motions,
        scale_zyx=V45_GRAPH_CONFIG.scale_zyx,
        dataset=attestation.held_out_dataset,
        attestation=attestation,
    )
    rerank_seconds = time.perf_counter() - rerank_started
    hybrid_nodes = [
        {
            "dataset": attestation.held_out_dataset,
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in hybrid.nodes
    ]
    hybrid_fragment = args.output_dir / "alpha_motion_hybrid" / "frags" / f"train__{attestation.held_out_dataset}.csv"
    hybrid_fragment.parent.mkdir(parents=True, exist_ok=True)
    hybrid_hash = write_fragment(hybrid_fragment, attestation.held_out_dataset, hybrid_nodes, hybrid.edges)
    timing = {
        "frozen_alpha_inference_seconds": alpha_evidence["inference_seconds"],
        "adapter_model_load_seconds": model_load_seconds,
        "adapter_motion_inference_seconds": adapter_inference_seconds,
        "sparse_rerank_seconds": rerank_seconds,
        "hybrid_incremental_seconds": model_load_seconds + adapter_inference_seconds + rerank_seconds,
        "hybrid_total_sequential_seconds": (
            alpha_evidence["inference_seconds"] + model_load_seconds + adapter_inference_seconds + rerank_seconds
        ),
    }
    inference_report = {
        "schema": GATE_SCHEMA,
        "kind": "frozen-alpha-child-motion-only-inference",
        "status": "completed",
        "gate_provenance": attestation.json(),
        "frozen_alpha": alpha_evidence,
        "restore": restore,
        "hybrid": {
            "fragment": {
                "path": str(hybrid_fragment.resolve()),
                "sha256": hybrid_hash,
                "bytes": hybrid_fragment.stat().st_size,
            },
            "motion_sample_count": len(motions),
            "stats": dict(hybrid.stats),
            "provenance": dict(hybrid.provenance),
        },
        "timing": timing,
        "data_access_before_scoring": {
            "image_split": "train",
            "image_video_count": 1,
            "geff_read": False,
            "held_out_labels_read": False,
            "audit_read": False,
            "test_read": False,
        },
        "source_hashes": {
            "scripts/score_alpha_motion_hybrid.py": sha256_file(Path(__file__).resolve()),
            "src/celltrack_alpha/ta_hybrid.py": sha256_file(SRC / "celltrack_alpha/ta_hybrid.py"),
            "src/celltrack_alpha/ta_inference.py": sha256_file(SRC / "celltrack_alpha/ta_inference.py"),
            "src/celltrack_alpha/t_adapter.py": sha256_file(SRC / "celltrack_alpha/t_adapter.py"),
        },
    }
    inference_sha = _write_json_new(args.output_dir / "inference.json", inference_report)

    score_started = time.perf_counter()
    score = _score(
        attestation,
        args.zip,
        selection,
        scorer_hashes,
        {"frozen_alpha": args.alpha_fragment, "alpha_motion_hybrid": hybrid_fragment},
    )
    score["timing"] = {**timing, "official_scoring_seconds": time.perf_counter() - score_started}
    score["inference"] = {
        "path": str((args.output_dir / "inference.json").resolve()),
        "sha256": inference_sha,
    }
    _write_json_new(args.output_dir / "score.json", score)
    return score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain-metrics", type=Path, required=True)
    parser.add_argument("--alpha-fragment", type=Path, default=DEFAULT_ALPHA_FRAGMENT)
    parser.add_argument("--alpha-runtime-report", type=Path, default=DEFAULT_ALPHA_RUNTIME)
    parser.add_argument("--alpha-score-evidence", type=Path, default=DEFAULT_ALPHA_SCORE)
    parser.add_argument("--fold", type=int, choices=(0,), default=0)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--provenance-manifest", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--registry-manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    if not 1 <= args.cpu_threads <= 32:
        raise GateError("--cpu-threads must be in [1,32]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
    except (GateError, HybridRerankError) as exc:
        print(f"score_alpha_motion_hybrid: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str((args.output_dir / "score.json").resolve()),
                "status": result["status"],
                "qualification": result["qualification"],
                "summaries": result["summaries"],
                "hybrid_minus_alpha": result["hybrid_minus_alpha"],
                "timing": result["timing"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

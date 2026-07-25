#!/usr/bin/env python3
"""Predict and score the converged organiser model on the frozen CV57 once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "kaggle-cell-tracking-competition"
for candidate in (UPSTREAM / "src", UPSTREAM / "scripts", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch  # noqa: E402
import tracksdata as td  # noqa: E402
from geff import GeffMetadata  # noqa: E402

import predict_unet_transformer as official_predict  # noqa: E402
from tracking_cellmot.io import open_dataset, save_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)
import train_organizer_f0 as training_contract  # noqa: E402


SCHEMA = "celltrack-alpha.organizer-unet-transformer-f0-cv57.v1"
COMPARATOR = 0.873150
EXPECTED_SCORER_HASHES = {
    "src/tracking_cellmot/division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
    "src/tracking_cellmot/metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
}


class HoldoutGateError(RuntimeError):
    """Raised before holdout access when training or scorer evidence is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutGateError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HoldoutGateError(f"{label} is not a JSON object: {path}")
    return value


def validate_patched_scorer(parity_path: Path) -> dict[str, Any]:
    parity = load_object(parity_path, "scorer parity ledger")
    fixture = parity.get("fixture_report")
    recorded_hashes = parity.get("source_hashes")
    if (
        parity.get("artifact") != "scorer_parity_v1"
        or not isinstance(fixture, Mapping)
        or fixture.get("overall_pass") is not True
        or fixture.get("edge_only_overall_pass") is not True
        or fixture.get("official_checkout") != training_contract.EXPECTED_OFFICIAL_COMMIT
        or not isinstance(recorded_hashes, Mapping)
    ):
        raise HoldoutGateError("patched scorer parity ledger is not green")
    observed = {
        relative: sha256_file(UPSTREAM / relative)
        for relative in EXPECTED_SCORER_HASHES
    }
    if observed != EXPECTED_SCORER_HASHES:
        raise HoldoutGateError("packaged patched scorer source hashes do not match")
    for relative, expected in EXPECTED_SCORER_HASHES.items():
        root_relative = f"external/kaggle-cell-tracking-competition/{relative}"
        if recorded_hashes.get(root_relative) != expected:
            raise HoldoutGateError(f"parity ledger does not bind {root_relative}")
    return {
        "official_checkout": fixture["official_checkout"],
        "fixture_count": len(fixture.get("cases", ())),
        "source_hashes": observed,
        "parity_sha256": sha256_file(parity_path),
    }


def validate_converged_training(training_dir: Path) -> dict[str, Any]:
    """Validate all training evidence before any holdout image/GT is opened."""
    run_path = training_dir / "run.json"
    provenance_path = training_dir / "provenance.json"
    checkpoint_path = training_dir / "checkpoint_last.pth"
    best_path = training_dir / "edge_predictor_best.pth"
    config_path = training_dir / "config.json"
    for path in (run_path, provenance_path, checkpoint_path, best_path, config_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise HoldoutGateError(f"training artifact is missing or empty: {path}")

    run = load_object(run_path, "training run")
    provenance = load_object(provenance_path, "training provenance")
    scope = training_contract.frozen_scope()
    fit_hash = hashlib.sha256(canonical_bytes(list(scope["fit117"]))).hexdigest()
    if (
        run.get("schema") != training_contract.SCHEMA
        or run.get("status") != "converged"
        or int(run.get("epoch", -1)) < 300
        or run.get("fit_video_count") != 117
        or run.get("holdout_evaluated") is not False
    ):
        raise HoldoutGateError("training run is not a converged fit117-only result")
    boundary = run.get("data_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("fit_only") is not True
        or any(boundary.get(key) is not False for key in ("holdout_used", "audit_used", "test_used"))
    ):
        raise HoldoutGateError("training run crossed the fit-only data boundary")
    if (
        provenance.get("schema") != training_contract.SCHEMA
        or provenance.get("split_sha256") != training_contract.EXPECTED_SPLIT_SHA256
        or provenance.get("fit_selection_sha256") != fit_hash
        or provenance.get("fit_video_count") != 117
        or provenance.get("initialization") != "random_scratch_no_released_or_foreign_checkpoint"
        or provenance.get("official_source_hashes") != training_contract.EXPECTED_SOURCE_HASHES
        or provenance.get("data_boundary") != boundary
    ):
        raise HoldoutGateError("training provenance is not the frozen scratch fit117 experiment")
    if run.get("checkpoint_last_sha256") != sha256_file(checkpoint_path):
        raise HoldoutGateError("last-checkpoint hash does not match the run manifest")
    if run.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise HoldoutGateError("best-checkpoint hash does not match the run manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != training_contract.SCHEMA
        or int(checkpoint.get("epoch", -1)) != int(run["epoch"])
        or checkpoint.get("provenance") != provenance
    ):
        raise HoldoutGateError("last checkpoint does not bind the converged run/provenance")
    return {
        "run": run,
        "provenance": provenance,
        "checkpoint": {
            "best_path": best_path,
            "best_sha256": sha256_file(best_path),
            "last_sha256": sha256_file(checkpoint_path),
            "config_sha256": sha256_file(config_path),
        },
        "holdout57": scope["holdout57"],
    }


def _estimated_total_nodes(geff_path: Path) -> float:
    metadata = GeffMetadata.read(geff_path)
    value = (metadata.extra or {}).get("estimated_number_of_nodes")
    return float(value) if value is not None else float("nan")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # These two gates complete before holdout paths are constructed or opened.
    training = validate_converged_training(args.training_dir)
    scorer = validate_patched_scorer(args.parity_ledger)
    if not torch.cuda.is_available():
        raise HoldoutGateError("CV57 inference requires CUDA")

    holdout = tuple(training["holdout57"])
    if len(holdout) != 57 or len(set(holdout)) != 57:
        raise HoldoutGateError("validated terminal holdout is not exactly 57 unique videos")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir()

    device = torch.device("cuda")
    weights_path = training["checkpoint"]["best_path"]
    model, window_size, downsample = official_predict.load_model(weights_path, device)
    cfg = official_predict.PredictConfig(
        det_threshold=0.99,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=False,
        max_parents_per_node=1,
        max_children_per_node=2,
    )

    rows: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    for index, identity in enumerate(holdout, start=1):
        stem = identity.split(":", 1)[1]
        dataset_path = args.data_dir / stem
        if not dataset_path.with_suffix(".zarr").is_dir() or not dataset_path.with_suffix(".geff").exists():
            raise HoldoutGateError(f"missing holdout pair for {identity}")
        print(f"[cv57] {index}/57 {identity}", flush=True)
        coords, edges = official_predict.predict_video(
            model,
            dataset_path,
            device,
            cfg=cfg,
            window_size=window_size,
            unet_batch_size=args.unet_batch_size,
            downsample=downsample,
        )
        graph = official_predict.build_graph(coords, edges)
        prediction_path = prediction_dir / f"{stem}.geff"
        save_graph(graph, prediction_path)

        dataset = open_dataset(dataset_path, normalize=False, require_tracks=True, load_image=False)
        result = compute_metric(graph, dataset.tracks, scale=dataset.scale, max_distance=7.0)
        recall = node_recall(graph, dataset.tracks) if graph.num_nodes() and graph.num_edges() else 0.0
        row = per_sample_metrics(
            result,
            _estimated_total_nodes(dataset_path.with_suffix(".geff")),
            recall,
        )
        rows.append(row)
        per_video.append({"video_id": identity, **row})

    summary = summarise(rows)
    score = float(summary["score"])
    if summary.get("n") != 57 or not math.isfinite(score):
        raise HoldoutGateError("patched scorer did not produce a finite complete CV57 score")
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "name": "frozen_fold0_holdout57",
            "video_count": 57,
            "video_ids": list(holdout),
            "selection_sha256": hashlib.sha256(canonical_bytes(list(holdout))).hexdigest(),
        },
        "model": {
            "architecture": "TemporalUNet3D+SimpleNodeTransformer",
            "best_checkpoint_sha256": training["checkpoint"]["best_sha256"],
            "last_checkpoint_sha256": training["checkpoint"]["last_sha256"],
            "training_epoch": training["run"]["epoch"],
            "best_epoch": training["run"]["best_epoch"],
            "fit_video_count": 117,
        },
        "inference": {
            "det_threshold": 0.99,
            "det_tta": True,
            "pool_kernel_um": 3.0,
            "edge_activation": "softmax",
            "edge_threshold": 0.5,
            "max_parents_per_node": 1,
            "max_children_per_node": 2,
            "use_ilp": False,
        },
        "patched_scorer": scorer,
        "summary": summary,
        "comparison": {
            "baseline": COMPARATOR,
            "holdout_score": score,
            "delta": score - COMPARATOR,
            "beats_baseline": score > COMPARATOR,
        },
        "production": {
            "merged_into_v45": False,
            "action": "report_only",
        },
        "per_video": per_video,
    }
    report_path = args.output_dir / "scores.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_bytes(report) + b"\n")
    os.replace(temporary, report_path)
    print(json.dumps(report["comparison"], sort_keys=True), flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--parity-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

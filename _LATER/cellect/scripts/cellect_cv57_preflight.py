#!/usr/bin/env python3
"""Validate and smoke-test the pinned GPL CELLECT CV57 experiment.

This script does not train, alter graph geometry, emit divisions, or contact
Kaggle.  It binds the exploratory direction to the exact fit117/CV57 data
boundary before any GPU run is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.cellect_adapter import (  # noqa: E402
    CellectAdapterConfig,
    CellectContractError,
    CellectEmbeddingAdapter,
    UPSTREAM_COMMIT,
    batch_hard_temporal_triplet,
    load_published_backbone,
    sample_embeddings_zyx,
    sha256_file,
    validate_upstream_tree,
)


SCHEMA = "celltrack-alpha.cellect-cv57-preflight.v2"
EXPECTED_BASELINE = {
    "adjusted_edge_jaccard": 0.8711885823490648,
    "edge_tp": 34_174,
    "edge_fp": 2_736,
    "edge_fn": 2_094,
}
EXPECTED_FIT_VIDEOS = 117
EXPECTED_HOLDOUT_VIDEOS = 57


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/CELLECT")
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117/manifest.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "reports/g02/graph_cost_sweeps_holdout57_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/cellect_cv57_preflight_v1.json",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CellectContractError(f"JSON root is not an object: {path}")
    return value


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def selected_holdout(split: dict[str, Any]) -> list[str]:
    fold = split.get("folds", {}).get("0")
    if not isinstance(fold, dict):
        raise CellectContractError("fold 0 is missing")
    held = [str(value) for value in fold.get("held_out_video_ids", ())]
    excluded = "train:6bba_f4ae811c"
    if len(held) != 58 or excluded not in held:
        raise CellectContractError("frozen fold-0 58-video selection changed")
    result = [value for value in held if value != excluded]
    if len(result) != EXPECTED_HOLDOUT_VIDEOS:
        raise CellectContractError("exact holdout is not 57 videos")
    return result


def validate_baseline(payload: dict[str, Any]) -> dict[str, float | int]:
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise CellectContractError("baseline ledger has no baseline record")
    observed = {
        "adjusted_edge_jaccard": float(baseline.get("adj_edge_jaccard", float("nan"))),
        "edge_tp": int(baseline.get("edge_tp", -1)),
        "edge_fp": int(baseline.get("edge_fp", -1)),
        "edge_fn": int(baseline.get("edge_fn", -1)),
    }
    if observed != EXPECTED_BASELINE:
        raise CellectContractError(f"exact CV57 baseline changed: {observed}")
    return observed


def smoke(upstream: Path, device_name: str) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise CellectContractError("CUDA smoke requested but unavailable")
    device = torch.device(device_name)
    model = CellectEmbeddingAdapter(load_published_backbone(upstream, device=device)).to(device)
    model.eval()
    input_tensor = torch.ones((1, 2, 32, 32, 8), dtype=torch.float32, device=device)
    started = time.perf_counter()
    with torch.no_grad():
        feature = model(input_tensor)
        points = torch.tensor([[1, 4, 4], [2, 8, 8], [3, 12, 12]], device=device)
        sampled = sample_embeddings_zyx(feature, points)
        positive = torch.tensor(
            [[True, False, False], [False, True, False], [False, False, True]],
            device=device,
        )
        loss, counts = batch_hard_temporal_triplet(sampled, sampled, positive)
    if tuple(feature.shape) != (1, 64, 32, 32, 8):
        raise CellectContractError(f"unexpected CELLECT smoke shape: {tuple(feature.shape)}")
    return {
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_shape": list(feature.shape),
        "finite": bool(torch.isfinite(feature).all().item()),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "synthetic_triplet_loss": float(loss.item()),
        "synthetic_triplet_counts": counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.upstream, args.fit_manifest, args.split, args.baseline):
        if not path.exists():
            raise CellectContractError(f"missing CELLECT preflight input: {path}")
    if args.output.exists():
        raise CellectContractError(f"refusing to overwrite preflight output: {args.output}")

    upstream = validate_upstream_tree(args.upstream)
    try:
        head = subprocess.run(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CellectContractError(f"cannot resolve CELLECT submodule revision: {exc}") from exc
    if head != UPSTREAM_COMMIT:
        raise CellectContractError(f"CELLECT submodule changed: {head} != {UPSTREAM_COMMIT}")

    fit = load_json(args.fit_manifest)
    fit_ids = [str(value) for value in fit.get("video_ids", ())]
    boundary = fit.get("data_boundary")
    if len(fit_ids) != EXPECTED_FIT_VIDEOS or len(set(fit_ids)) != EXPECTED_FIT_VIDEOS:
        raise CellectContractError("fit cache is not the exact 117-video corpus")
    expected_boundary = {
        "audit_used": False,
        "competition_submission_performed": False,
        "held_out_used": False,
        "test_used": False,
        "train_fit_only": True,
    }
    if boundary != expected_boundary:
        raise CellectContractError(f"fit117 data boundary changed: {boundary}")

    split = load_json(args.split)
    holdout_ids = selected_holdout(split)
    overlap = sorted(set(fit_ids) & set(holdout_ids))
    if overlap:
        raise CellectContractError(f"fit117 overlaps exact CV57: {overlap}")
    baseline = validate_baseline(load_json(args.baseline))

    config = CellectAdapterConfig()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "validated",
        "supersedes": {
            "path": "reports/g02/cellect_cv57_preflight_v1.json",
            "status": "superseded_missing_model_source_hash",
        },
        "experiment": {
            "role": "orthogonal learned edge cost over released baseline detections",
            "production_baseline_untouched": True,
            "candidate_gate": "top-k physical neighbours",
            "training_loss": "hard-positive/hard-negative squared-distance triplet",
            "division_output_enabled": False,
            "kaggle_action_performed": False,
        },
        "adapter_config": config.as_dict(),
        "upstream": {**upstream, "git_head": head},
        "fit117": {
            "manifest": str(args.fit_manifest),
            "manifest_sha256": sha256_file(args.fit_manifest),
            "video_count": len(fit_ids),
            "video_ids_sha256": canonical_sha(fit_ids),
            "data_boundary": boundary,
        },
        "cv57": {
            "split": str(args.split),
            "split_sha256": sha256_file(args.split),
            "baseline": str(args.baseline),
            "baseline_sha256": sha256_file(args.baseline),
            "video_count": len(holdout_ids),
            "video_ids_sha256": canonical_sha(holdout_ids),
            **baseline,
        },
        "smoke": smoke(args.upstream, args.device) if args.smoke else {"executed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

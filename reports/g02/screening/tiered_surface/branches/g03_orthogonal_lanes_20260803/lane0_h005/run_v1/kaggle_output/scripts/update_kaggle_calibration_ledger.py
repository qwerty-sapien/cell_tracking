#!/usr/bin/env python3
"""Combine frozen probe artifacts, local CV scores, and optional Kaggle results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.site_calibration import (  # noqa: E402
    residuals,
    select_calibration_model,
)


CV_SCORE_FILES = {
    "baseline": "cv_score_baseline.json",
    "edge_thin_10": "cv_score_1_edge_thin_10.json",
    "long_edge_prune_5um": "cv_score_2_long_edge_prune_5um.json",
    "short_component_prune_1edge": "cv_score_3_short_component_prune_1edge.json",
    "division_rescue_conservative": "cv_score_4_division_rescue_conservative.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=Path("reports/kaggle_g01/calibration_probes/frozen_probe_manifest.json"),
    )
    parser.add_argument(
        "--cv-score-dir",
        type=Path,
        default=Path("reports/kaggle_g01/calibration_probes"),
    )
    parser.add_argument(
        "--kaggle-results",
        type=Path,
        default=None,
        help="Optional JSON mapping probe names to submission_id/status/score.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/kaggle_g01/calibration_probes/calibration_ledger.json"),
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_local_scores(directory: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for name, filename in CV_SCORE_FILES.items():
        path = directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores[name] = {
            "score": finite(payload.get("score", payload.get("summary", {}).get("score"))),
            "report": str(path),
            "submission_sha256": payload.get("submission_sha256"),
            "summary": payload.get("summary"),
        }
    return scores


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.probe_manifest.read_text(encoding="utf-8"))
    local = load_local_scores(args.cv_score_dir)
    kaggle = (
        json.loads(args.kaggle_results.read_text(encoding="utf-8"))
        if args.kaggle_results is not None
        else {}
    )

    entries: list[dict[str, Any]] = []
    baseline_manifest = manifest["baseline"]
    entries.append(
        {
            "name": "baseline",
            "role": "calibration",
            "order": 0,
            "artifact": baseline_manifest,
            "local": local["baseline"],
            "kaggle": kaggle.get("baseline"),
        }
    )
    for variant in manifest["variants"]:
        name = str(variant["name"])
        entries.append(
            {
                "name": name,
                "role": variant["role"],
                "order": int(variant["order_after_baseline"]),
                "artifact": variant,
                "local": local[name],
                "kaggle": kaggle.get(name),
            }
        )

    baseline_local = float(local["baseline"]["score"])
    baseline_kaggle = finite((kaggle.get("baseline") or {}).get("score"))
    for entry in entries:
        local_score = finite(entry["local"].get("score"))
        kaggle_score = finite((entry.get("kaggle") or {}).get("score"))
        entry["local_delta_from_baseline"] = (
            local_score - baseline_local if local_score is not None else None
        )
        entry["raw_absolute_residual"] = (
            abs(local_score - kaggle_score)
            if local_score is not None and kaggle_score is not None
            else None
        )
        entry["kaggle_delta_from_baseline"] = (
            kaggle_score - baseline_kaggle
            if kaggle_score is not None and baseline_kaggle is not None
            else None
        )

    calibration_entries = [
        entry
        for entry in entries
        if entry["role"] == "calibration"
        and finite(entry["local"].get("score")) is not None
        and finite((entry.get("kaggle") or {}).get("score")) is not None
    ]
    model_payload: dict[str, Any] | None = None
    parity_status = "awaiting_kaggle_scores"
    if len(calibration_entries) == 3:
        local_fit = [float(entry["local"]["score"]) for entry in calibration_entries]
        kaggle_fit = [float(entry["kaggle"]["score"]) for entry in calibration_entries]
        model = select_calibration_model(local_fit, kaggle_fit, tolerance=0.02)
        fit_residuals = residuals(model, local_fit, kaggle_fit)
        model_payload = {
            "name": model.name,
            "intercept": model.intercept,
            "slope": model.slope,
            "calibration_names": [entry["name"] for entry in calibration_entries],
            "calibration_residuals": fit_residuals,
        }
        for entry in entries:
            local_score = finite(entry["local"].get("score"))
            kaggle_score = finite((entry.get("kaggle") or {}).get("score"))
            entry["calibrated_local_score"] = (
                model.predict(local_score) if local_score is not None else None
            )
            entry["calibrated_absolute_residual"] = (
                abs(model.predict(local_score) - kaggle_score)
                if local_score is not None and kaggle_score is not None
                else None
            )
        holdouts = [entry for entry in entries if entry["role"] == "holdout"]
        if all(entry.get("calibrated_absolute_residual") is not None for entry in holdouts):
            parity_status = (
                "pass"
                if all(float(entry["calibrated_absolute_residual"]) <= 0.02 for entry in holdouts)
                else "fail"
            )
        else:
            parity_status = "awaiting_holdout_scores"

    payload = {
        "schema_version": 1,
        "kind": "kaggle_local_score_calibration_ledger",
        "probe_manifest": str(args.probe_manifest),
        "kaggle_results": str(args.kaggle_results) if args.kaggle_results else None,
        "absolute_tolerance": 0.02,
        "selection_policy": "identity, then bias, then affine; selected on calibration entries only",
        "parity_status": parity_status,
        "model": model_payload,
        "entries": entries,
        "known_local_limitation": (
            "The locked 12x3-frame CV contains no scored divisions; the division-rescue "
            "holdout currently predicts zero local delta. A Kaggle residual above 0.02 on that "
            "holdout requires division-enriched CV, not holdout redefinition."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parity_status": parity_status, "entries": entries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

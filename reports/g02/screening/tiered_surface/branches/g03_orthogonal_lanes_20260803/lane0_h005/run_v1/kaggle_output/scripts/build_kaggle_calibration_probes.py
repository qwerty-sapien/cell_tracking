#!/usr/bin/env python3
"""Freeze deterministic Kaggle score-calibration submission variants."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    VARIANTS,
    apply_variant,
    graph_totals,
    read_submission,
    sha256_file,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/kaggle_g01/kernel_output/submission.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/kaggle_g01/calibration_probes"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or args.output_dir / "frozen_probe_manifest.json"
    baseline = read_submission(args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, object]] = []
    for order, (name, role, transform) in enumerate(VARIANTS, start=1):
        graphs, stats = apply_variant(baseline, transform)
        output = args.output_dir / f"submission_{order}_{name}.csv"
        write_submission(graphs, output)
        variants.append(
            {
                "order_after_baseline": order,
                "name": name,
                "role": role,
                "path": str(output),
                "sha256": sha256_file(output),
                "totals": graph_totals(graphs.values()),
                "per_dataset": stats,
                "local_cv_score_pre_kaggle": None,
                "kaggle_submission_id": None,
                "kaggle_score": None,
                "absolute_residual": None,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "frozen_kaggle_scorer_calibration_probes",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_policy": (
            "Variant definitions, order, calibration/holdout roles, artifacts, and hashes are "
            "frozen before any G-01 Kaggle score is observed."
        ),
        "parity_threshold_absolute": 0.02,
        "baseline": {
            "path": str(args.baseline),
            "sha256": sha256_file(args.baseline),
            "totals": graph_totals(baseline.values()),
            "local_cv_score_pre_kaggle": 0.8713516075529171,
            "kaggle_submission_id": None,
            "kaggle_score": None,
            "absolute_residual": None,
        },
        "variants": variants,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "variants": variants}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the fit105-only exact label join without fitting or graph scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _pairs(value: Any) -> set[tuple[int, int]]:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise RuntimeError("pair array is malformed")
    return {tuple(map(int, row)) for row in array}


def run(output: Path) -> dict[str, Any]:
    manifest = _load_json(
        ROOT
        / "kaggle_kernel/trackastra-h1-feature-cache-prep/"
        "cache_dataset_manifest.json"
    )
    feature_report = _load_json(
        ROOT / "reports/g02/screening/h1_existing_cache_feature_report.json"
    )
    prepare_report = _load_json(
        ROOT
        / "reports/g02/selected_graph_temporal_background_prepare_v1/"
        "prepare.json"
    )
    identities = tuple(map(str, manifest["identities"]))
    features = {
        str(row["identity"]): ROOT / str(row["feature_path"])
        for row in feature_report["files"]
    }
    prepared = {
        str(row["identity"]): ROOT / str(row["path"])
        for row in prepare_report["videos"]
        if str(row.get("identity", "")) in identities
    }
    if set(features) != set(identities) or set(prepared) != set(identities):
        raise RuntimeError("fit105 identity coverage mismatch")
    rows = []
    for identity in identities:
        feature = torch.load(
            features[identity], map_location="cpu", weights_only=False
        )
        labels = torch.load(
            prepared[identity], map_location="cpu", weights_only=False
        )
        cached = _pairs(feature["challengers"])
        known = _pairs(labels["labeled_challenger_pairs"])
        exact = cached.intersection(known)
        reversed_known = {(target, source) for source, target in known}
        reversed_overlap = cached.intersection(reversed_known)
        rows.append(
            {
                "identity": identity,
                "family": identity.split(":", 1)[1].split("_", 1)[0],
                "cached_proposals": len(cached),
                "known_labeled_pairs": len(known),
                "exact_overlap": len(exact),
                "reversed_overlap": len(reversed_overlap),
            }
        )
    families = {}
    for family in ("44b6", "6bba"):
        selected = [row for row in rows if row["family"] == family]
        families[family] = {
            "videos": len(selected),
            "videos_with_exact_overlap": sum(
                row["exact_overlap"] > 0 for row in selected
            ),
            "exact_overlap": sum(row["exact_overlap"] for row in selected),
            "reversed_overlap": sum(
                row["reversed_overlap"] for row in selected
            ),
            "known_labeled_pairs": sum(
                row["known_labeled_pairs"] for row in selected
            ),
        }
    result = {
        "schema": "celltrack-alpha.h1-zero-wrfeat-label-join-audit.v1",
        "scope": {
            "fit_train105_only": True,
            "validation_access": False,
            "cross_validation_access": False,
            "holdout_access": False,
        },
        "families": families,
        "videos": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps(report["families"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

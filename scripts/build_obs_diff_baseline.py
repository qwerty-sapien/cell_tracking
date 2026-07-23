#!/usr/bin/env python3
"""Build the frozen Phase-0 observation-difference audit from existing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "scripts/compare_submissions.py"
SPEC = importlib.util.spec_from_file_location("compare_submissions", COMPARE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {COMPARE_PATH}")
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


LOCAL_ROOT = (
    ROOT
    / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1"
)
SCREENING = ROOT / "reports/g02/screening"


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _pair_report(
    artifact_a: Any,
    artifact_b: Any,
    *,
    scope: str,
    config_a: str,
    config_b: str,
    local_score_a: float | None,
    local_score_b: float | None,
) -> dict[str, Any]:
    datasets = sorted(set(artifact_a.graphs).union(artifact_b.graphs))
    rows = []
    for dataset in datasets:
        rows.append(
            compare.compare_graphs(
                artifact_a.graphs.get(dataset, compare._empty_graph(dataset)),
                artifact_b.graphs.get(dataset, compare._empty_graph(dataset)),
                max_distance_um=compare.DEFAULT_MAX_DISTANCE_UM,
            )
        )
    aggregate = compare.aggregate_rows(rows)
    families = sorted({row["family"] for row in rows})
    return {
        "scope": scope,
        "config_a": config_a,
        "config_b": config_b,
        "local_score_a": local_score_a,
        "local_score_b": local_score_b,
        "local_delta_a_minus_b": (
            local_score_a - local_score_b
            if local_score_a is not None and local_score_b is not None
            else None
        ),
        "artifacts": {
            "a": artifact_a.identity_dict(),
            "b": artifact_b.identity_dict(),
        },
        "aggregate": aggregate,
        "by_family": {
            family: compare.aggregate_rows(
                [row for row in rows if row["family"] == family]
            )
            for family in families
        },
        "by_dataset": rows,
        "observation_class": compare.classify_distance(aggregate),
    }


def build() -> dict[str, Any]:
    baseline_path = (
        LOCAL_ROOT / "motion_error_weight_0p25/fragments"
    )
    local_specs = [
        (
            "historical-c0",
            LOCAL_ROOT / "motion_error_weight_0/fragments",
            0.8483945155545333,
        ),
        (
            "historical-c2",
            LOCAL_ROOT / "motion_error_weight_2/fragments",
            0.8721499474601118,
        ),
        (
            "banked-c0p25-null",
            baseline_path,
            0.8731495935289003,
        ),
        (
            "learned-bonus-4",
            SCREENING
            / "p0_lb_replay_cv57_v1/"
            "learned_bonus_4__raw_distance_weight_0p05/fragments",
            0.8720910462365732,
        ),
        (
            "learned-bonus-8",
            SCREENING
            / "p0_lb_replay_cv57_v1/"
            "learned_bonus_8__raw_distance_weight_0p05/fragments",
            0.8670102112352311,
        ),
        (
            "learned-bonus-16",
            SCREENING
            / "p0_lb_replay_cv57_v1/"
            "learned_bonus_16__raw_distance_weight_0p05/fragments",
            0.8479018413756242,
        ),
    ]
    baseline = compare.load_prediction(baseline_path, label="banked-c0p25")
    loaded_local = {
        label: compare.load_prediction(path, label=label)
        for label, path, _score in local_specs
    }
    local_pairs = [
        _pair_report(
            loaded_local[label],
            baseline,
            scope="local-cv57",
            config_a=label,
            config_b="banked-c0p25",
            local_score_a=score,
            local_score_b=0.8731495935289003,
        )
        for label, path, score in local_specs
    ]
    local_scores = {label: score for label, _path, score in local_specs}
    matched_pair_specs = [
        ("learned-bonus-4", "historical-c2"),
        ("learned-bonus-16", "historical-c0"),
    ]
    matched_local_pairs = []
    for label_a, label_b in matched_pair_specs:
        pair = _pair_report(
            loaded_local[label_a],
            loaded_local[label_b],
            scope="local-cv57-matched-pair",
            config_a=label_a,
            config_b=label_b,
            local_score_a=local_scores[label_a],
            local_score_b=local_scores[label_b],
        )
        pair["admissibility"] = {
            "resolution_floor": 0.004,
            "local_match": abs(pair["local_delta_a_minus_b"]) < 0.004,
            "observation_distinct": pair["observation_class"]["label"] == "DISTINCT",
            "admissible_for_mechanism_interaction": (
                abs(pair["local_delta_a_minus_b"]) < 0.004
                and pair["observation_class"]["label"] == "DISTINCT"
            ),
            "family_effect_claim": False,
            "reason": (
                "The learned bonus changes association scoring within the graph pipeline; "
                "this can test a mechanism interaction but cannot establish a general "
                "model-family effect."
            ),
        }
        matched_local_pairs.append(pair)

    public_baseline_path = ROOT / "kaggle_kernel/bhsubmitout_v3/submission.csv"
    public_baseline = compare.load_prediction(
        public_baseline_path,
        label="public-c0p25",
    )
    public_specs = [
        (
            "public-c0",
            ROOT / "kaggle_kernel/bhp0-c0-out-v1/submission.csv",
            0.860,
        ),
        (
            "public-c2",
            ROOT / "kaggle_kernel/bhp0-c2-out-v2/submission.csv",
            0.878,
        ),
        (
            "public-c0p25-null",
            public_baseline_path,
            0.877,
        ),
    ]
    public_pairs = [
        {
            **_pair_report(
                compare.load_prediction(path, label=label),
                public_baseline,
                scope="public-hidden",
                config_a=label,
                config_b="public-c0p25",
                local_score_a=None,
                local_score_b=None,
            ),
            "public_displayed_a": public_score,
            "public_displayed_b": 0.877,
            "public_interval_a": [
                public_score - 0.0005,
                public_score + 0.0005,
            ],
            "public_interval_b": [0.8765, 0.8775],
        }
        for label, path, public_score in public_specs
    ]

    reconciliation_path = SCREENING / "obs_diff_c0p25_vs_c2_v1.json"
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    return {
        "schema": "celltrack-alpha.observation-distance-baseline-audit.v1",
        "status": "completed",
        "method": {
            "tool": _file_identity(COMPARE_PATH),
            "scale_zyx_um": list(compare.SCALE_ZYX),
            "max_distance_um": compare.DEFAULT_MAX_DISTANCE_UM,
            "null_envelope": dict(compare.ZERO_NULL_ENVELOPE),
            "distance_is_vector": True,
            "prediction_edge_overlap_name": "edge_jaccard",
            "classification_rule": (
                "Only exact zero observation distance is IDENTICAL; nonzero components "
                "outside the preregistered null envelope are DISTINCT."
            ),
        },
        "frozen_scorer_reconciliation": {
            "source": _file_identity(reconciliation_path),
            "reconciliation": reconciliation["reconciliation"],
            "a_summary": reconciliation["frozen_score"]["a"]["summary"],
            "a_edge_counts": reconciliation["frozen_score"]["a"]["edge_counts"],
            "b_summary": reconciliation["frozen_score"]["b"]["summary"],
            "b_edge_counts": reconciliation["frozen_score"]["b"]["edge_counts"],
        },
        "local_pairs": local_pairs,
        "matched_local_pairs": matched_local_pairs,
        "public_pairs": public_pairs,
        "classification_counts": {
            label: sum(
                pair["observation_class"]["label"] == label
                for pair in [*local_pairs, *matched_local_pairs, *public_pairs]
            )
            for label in ("IDENTICAL", "LOW-INFORMATION", "DISTINCT")
        },
        "interpretation": {
            "strictly_vacuous": [
                pair["config_a"]
                for pair in [*local_pairs, *matched_local_pairs, *public_pairs]
                if pair["observation_class"]["label"] == "IDENTICAL"
                and not pair["config_a"].endswith("-null")
            ],
            "submission_authority": "none; this is a CPU/local audit",
            "next_action": (
                "Use the nonzero vector and local leverage jointly when designing the "
                "dispersion ladder; matched local score alone is insufficient."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCREENING / "obs_diff_baseline_v1.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compare._json_safe(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification_counts": report["classification_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

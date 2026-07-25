#!/usr/bin/env python3
"""Fit-only CG1 abstention-mechanism falsification over frozen proposal scores.

This diagnostic does not train a ranker.  It reuses the parked, hash-pinned
temporal proposal scorer and tests calibration/selection laws on five held-video
folds from fit-train105.  The fit-validation12 partition remains unopened for
selection.  Unlabeled deployment-background samples are never relabeled.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_achievable_headroom import _load_parked_scores  # noqa: E402
from fit_selected_graph_temporal_background_gate import _fold  # noqa: E402


SCHEMA = "celltrack-alpha.cg1-abstention-screen.v1"
FAMILIES = ("44b6", "6bba")
FORBIDDEN_PATH_TOKENS = ("cv57", "holdout")
RETENTION_FLOOR = 0.20
CONFIDENCE = 0.95

DEFAULT_GATE = ROOT / "reports/g02/selected_graph_temporal_rank_gate_v1/report.json"
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_CG0 = ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/screening/candgen/cg1_abstention_screen_v1.json"


class CG1ScreenError(RuntimeError):
    """The CG1 fit-only mechanism-screen contract failed."""


def assert_safe_fit_only_path(path: Path) -> None:
    lowered = str(path).lower()
    token = next((item for item in FORBIDDEN_PATH_TOKENS if item in lowered), None)
    if token is not None:
        raise CG1ScreenError(
            f"forbidden {token!r} artifact reference rejected before read: {path}"
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    assert_safe_fit_only_path(path)
    if not path.is_file():
        raise CG1ScreenError(f"missing required input: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise CG1ScreenError(f"expected JSON object: {path}")
    return value


def _family(identity: str) -> str:
    family = identity.split(":", 1)[1].split("_", 1)[0]
    if family not in FAMILIES:
        raise CG1ScreenError(f"unexpected family: {identity}")
    return family


def _upper_binomial_rate(successes: int, trials: int) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise CG1ScreenError("invalid binomial support")
    if successes == trials:
        return 1.0
    return float(beta.ppf(CONFIDENCE, successes + 1, trials - successes))


def _candidate_thresholds(videos: Sequence[Mapping[str, Any]]) -> np.ndarray:
    positives = [
        np.asarray(video["candidate_scores"])[np.asarray(video["labels"]) == 1]
        for video in videos
    ]
    values = np.concatenate([item for item in positives if len(item)])
    if not len(values):
        return np.asarray([math.inf], dtype=np.float64)
    return np.concatenate(
        (
            np.asarray([math.inf], dtype=np.float64),
            np.unique(values)[::-1],
        )
    )


def _background_at_threshold(
    videos: Sequence[Mapping[str, Any]], threshold: float
) -> tuple[int, int, float, int]:
    accepted = support = 0
    projected = 0.0
    raw_total = 0
    for video in videos:
        scores = np.asarray(video["background_scores"], dtype=np.float64)
        count = int(np.sum(scores >= threshold))
        raw = int(video["raw_background"])
        accepted += count
        support += len(scores)
        raw_total += raw
        if len(scores):
            projected += raw * count / len(scores)
    return accepted, support, projected, raw_total


def _positive_count(
    videos: Sequence[Mapping[str, Any]], threshold: float
) -> int:
    return sum(
        int(
            np.sum(
                (np.asarray(video["candidate_scores"]) >= threshold)
                & (np.asarray(video["labels"]) == 1)
            )
        )
        for video in videos
    )


def _select_point_threshold(
    videos: Sequence[Mapping[str, Any]], *, acceptance_rate: float
) -> float:
    choices: list[tuple[int, float, float]] = []
    for threshold in _candidate_thresholds(videos):
        _, _, projected, raw_total = _background_at_threshold(videos, threshold)
        if projected <= acceptance_rate * raw_total + 1.0e-9:
            choices.append(
                (_positive_count(videos, float(threshold)), -projected, float(threshold))
            )
    return max(choices)[2] if choices else math.inf


def _select_crc_threshold(
    videos: Sequence[Mapping[str, Any]], *, acceptance_rate: float
) -> float:
    choices: list[tuple[int, float, float]] = []
    for threshold in _candidate_thresholds(videos):
        accepted, support, _, raw_total = _background_at_threshold(videos, threshold)
        # Infinite abstention has deterministic zero risk; finite thresholds must
        # earn the one-sided finite-sample certificate.
        upper = 0.0 if math.isinf(float(threshold)) else _upper_binomial_rate(
            accepted, support
        )
        projected_upper = upper * raw_total
        if projected_upper <= acceptance_rate * raw_total + 1.0e-9:
            choices.append(
                (
                    _positive_count(videos, float(threshold)),
                    -projected_upper,
                    float(threshold),
                )
            )
    return max(choices)[2] if choices else math.inf


def _evaluate(
    videos: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    acceptance_rates: Mapping[str, float],
) -> dict[str, Any]:
    rows = []
    for video in videos:
        family = _family(str(video["identity"]))
        threshold = float(thresholds[family])
        candidate_scores = np.asarray(video["candidate_scores"])
        labels = np.asarray(video["labels"])
        background_scores = np.asarray(video["background_scores"])
        kept = candidate_scores >= threshold
        background_kept = background_scores >= threshold
        rows.append(
            {
                "identity": str(video["identity"]),
                "family": family,
                "correct_kept": int(np.sum(kept & (labels == 1))),
                "correct_total": int(np.sum(labels == 1)),
                "harmful_kept": int(np.sum(kept & (labels == 0))),
                "background_kept": int(np.sum(background_kept)),
                "background_support": len(background_scores),
                "raw_background": int(video["raw_background"]),
                "projected_background_fp": (
                    int(video["raw_background"])
                    * int(np.sum(background_kept))
                    / max(len(background_scores), 1)
                ),
            }
        )

    result: dict[str, Any] = {"videos": rows, "families": {}}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        correct_kept = sum(row["correct_kept"] for row in selected)
        correct_total = sum(row["correct_total"] for row in selected)
        bg_kept = sum(row["background_kept"] for row in selected)
        bg_support = sum(row["background_support"] for row in selected)
        raw_background = sum(row["raw_background"] for row in selected)
        projected = sum(row["projected_background_fp"] for row in selected)
        budget = acceptance_rates[family] * raw_background
        upper_rate = _upper_binomial_rate(bg_kept, bg_support)
        summary = {
            "video_count": len(selected),
            "known_correct_kept": correct_kept,
            "known_correct_total": correct_total,
            "correct_retention": correct_kept / max(correct_total, 1),
            "known_harmful_kept": sum(row["harmful_kept"] for row in selected),
            "background_sample_kept": bg_kept,
            "background_sample_support": bg_support,
            "projected_background_fp": projected,
            "projected_background_fp_95pct_upper": upper_rate * raw_background,
            "held_raw_background": raw_background,
            "held_background_budget": budget,
            "point_budget_pass": projected <= budget + 1.0e-9,
            "finite_sample_budget_pass": upper_rate * raw_background <= budget + 1.0e-9,
            "retention_pass": correct_kept / max(correct_total, 1) >= RETENTION_FLOOR,
        }
        summary["local_gate_pass"] = bool(
            summary["point_budget_pass"] and summary["retention_pass"]
        )
        result["families"][family] = summary
    result["both_families_pass"] = all(
        result["families"][family]["local_gate_pass"] for family in FAMILIES
    )
    return result


def _copy_raw(videos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **video,
            "candidate_scores": np.asarray(video["labeled_confidence"], dtype=np.float64),
            "background_scores": np.asarray(
                video["background_confidence"], dtype=np.float64
            ),
        }
        for video in videos
    ]


def _video_tail_rank(videos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        background = np.asarray(video["background_confidence"], dtype=np.float64)

        def tail(values: np.ndarray, *, self_background: bool) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            if not len(background):
                return np.ones(len(values), dtype=np.float64)
            counts = np.asarray(
                [np.sum(background >= value) for value in values], dtype=np.float64
            )
            if self_background:
                counts = np.maximum(counts, 1.0)
            return -counts / len(background)

        result.append(
            {
                **video,
                "candidate_scores": tail(
                    np.asarray(video["labeled_confidence"]), self_background=False
                ),
                "background_scores": tail(background, self_background=True),
            }
        )
    return result


def _fit_pu_transform(
    calibration: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    bins: int = 64,
) -> tuple[list[dict[str, Any]], float]:
    positive = np.concatenate(
        [
            np.asarray(video["labeled_confidence"])[
                np.asarray(video["labels"]) == 1
            ]
            for video in calibration
        ]
    )
    unlabeled = np.concatenate(
        [np.asarray(video["background_confidence"]) for video in calibration]
    )
    if not len(positive) or not len(unlabeled):
        raise CG1ScreenError("PU transform lacks positive or unlabeled support")
    edges = np.unique(
        np.quantile(
            np.concatenate((positive, unlabeled)),
            np.linspace(0.0, 1.0, bins + 1),
        )
    )
    if len(edges) < 3:
        raise CG1ScreenError("PU transform score support is degenerate")
    edges[0], edges[-1] = -math.inf, math.inf
    p_count, _ = np.histogram(positive, bins=edges)
    u_count, _ = np.histogram(unlabeled, bins=edges)
    p_density = (p_count + 0.5) / (len(positive) + 0.5 * len(p_count))
    u_density = (u_count + 0.5) / (len(unlabeled) + 0.5 * len(u_count))
    prior = float(np.clip(np.min(u_density / p_density), 1.0e-6, 1.0))
    posterior = np.clip(prior * p_density / u_density, 0.0, 1.0)

    def transform(values: np.ndarray) -> np.ndarray:
        index = np.searchsorted(edges[1:-1], np.asarray(values), side="right")
        return posterior[index]

    converted = [
        {
            **video,
            "candidate_scores": transform(np.asarray(video["labeled_confidence"])),
            "background_scores": transform(np.asarray(video["background_confidence"])),
        }
        for video in target
    ]
    return converted, prior


def _aggregate_oof(
    fold_results: Sequence[Mapping[str, Any]],
    acceptance_rates: Mapping[str, float],
) -> dict[str, Any]:
    videos = [
        row
        for fold in fold_results
        for row in fold["evaluation"]["videos"]
    ]
    # Reuse _evaluate's family aggregation by converting already-decided rows
    # into score arrays whose single threshold keeps exactly the recorded rows.
    result: dict[str, Any] = {"folds": list(fold_results), "families": {}}
    for family in FAMILIES:
        selected = [row for row in videos if row["family"] == family]
        correct_kept = sum(row["correct_kept"] for row in selected)
        correct_total = sum(row["correct_total"] for row in selected)
        bg_kept = sum(row["background_kept"] for row in selected)
        bg_support = sum(row["background_support"] for row in selected)
        raw_background = sum(row["raw_background"] for row in selected)
        projected = sum(row["projected_background_fp"] for row in selected)
        budget = acceptance_rates[family] * raw_background
        upper = _upper_binomial_rate(bg_kept, bg_support) * raw_background
        family_result = {
            "video_count": len(selected),
            "known_correct_kept": correct_kept,
            "known_correct_total": correct_total,
            "correct_retention": correct_kept / max(correct_total, 1),
            "known_harmful_kept": sum(row["harmful_kept"] for row in selected),
            "background_sample_kept": bg_kept,
            "background_sample_support": bg_support,
            "projected_background_fp": projected,
            "projected_background_fp_95pct_upper": upper,
            "held_raw_background": raw_background,
            "held_background_budget": budget,
            "point_budget_pass": projected <= budget + 1.0e-9,
            "finite_sample_budget_pass": upper <= budget + 1.0e-9,
            "retention_pass": correct_kept / max(correct_total, 1) >= RETENTION_FLOOR,
        }
        family_result["local_gate_pass"] = bool(
            family_result["point_budget_pass"] and family_result["retention_pass"]
        )
        result["families"][family] = family_result
    result["both_families_pass"] = all(
        result["families"][family]["local_gate_pass"] for family in FAMILIES
    )
    return result


def _run_folded(
    raw_videos: Sequence[Mapping[str, Any]],
    acceptance_rates: Mapping[str, float],
    *,
    selector: Callable[..., float],
    family_conditional: bool,
    pu_transform: bool = False,
) -> dict[str, Any]:
    folds = []
    for fold in range(5):
        calibration_raw = [video for video in raw_videos if int(video["fold"]) != fold]
        held_raw = [video for video in raw_videos if int(video["fold"]) == fold]
        thresholds: dict[str, float] = {}
        held_scored: list[dict[str, Any]] = []
        priors: dict[str, float] = {}
        groups = FAMILIES if family_conditional else ("overall",)
        for group in groups:
            calibration = (
                calibration_raw
                if group == "overall"
                else [video for video in calibration_raw if _family(video["identity"]) == group]
            )
            held = (
                held_raw
                if group == "overall"
                else [video for video in held_raw if _family(video["identity"]) == group]
            )
            if pu_transform:
                calibration_scored, prior = _fit_pu_transform(
                    calibration, calibration
                )
                held_group, _ = _fit_pu_transform(calibration, held)
                priors[group] = prior
            else:
                calibration_scored = _copy_raw(calibration)
                held_group = _copy_raw(held)
            rate = (
                acceptance_rates["overall"]
                if group == "overall"
                else acceptance_rates[group]
            )
            threshold = selector(calibration_scored, acceptance_rate=rate)
            if group == "overall":
                thresholds = {family: threshold for family in FAMILIES}
            else:
                thresholds[group] = threshold
            held_scored.extend(held_group)
        evaluation = _evaluate(held_scored, thresholds, acceptance_rates)
        folds.append(
            {
                "fold": fold,
                "held_video_count": len(held_raw),
                "thresholds": thresholds,
                "pu_class_priors": priors,
                "evaluation": evaluation,
            }
        )
    return _aggregate_oof(folds, acceptance_rates)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.gate,
        args.prepare,
        args.reachability,
        args.cg0,
        args.output,
    ):
        assert_safe_fit_only_path(path)
    _, prepare, scored, train_ids, validation_ids, gate_path = _load_parked_scores(
        args.gate, args.prepare
    )
    reachability = _load_json(args.reachability)
    cg0 = _load_json(args.cg0)
    if (
        reachability.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or cg0.get("schema") != "celltrack-alpha.candgen-ceiling-diagnostic.v1"
        or cg0.get("verdict") != "GO"
    ):
        raise CG1ScreenError("CG0/reachability prerequisites do not reconcile")

    reach_by_id = {
        str(row["identity"]): row for row in reachability.get("videos", ())
    }
    train_set = set(train_ids)
    raw_videos = []
    for video in scored:
        identity = str(video["identity"])
        if identity not in train_set:
            continue
        metrics = reach_by_id[identity]["metrics"]
        pool = int(metrics["candidate_pool"]["r10_k08"])
        correct = int(metrics["policy_reachable"]["r10_k08"])
        raw_videos.append(
            {
                **video,
                "raw_background": pool - correct,
                "fold": _fold(identity, 5),
            }
        )

    current = next(
        row
        for row in cg0["policy_ladder"]
        if row["policy"] == "current_r10_k08"
    )
    acceptance_rates = {
        family: float(
            current["families"][family]["background_budget_at_goal"][
                "maximum_background_acceptance_fraction"
            ]
        )
        for family in (*FAMILIES, "overall")
    }
    fit117_budgets = {
        family: int(
            current["families"][family]["background_budget_at_goal"][
                "maximum_background_false_positive_edges"
            ]
        )
        for family in (*FAMILIES, "overall")
    }

    mechanisms = {
        "global_point_threshold_control": _run_folded(
            raw_videos,
            acceptance_rates,
            selector=_select_point_threshold,
            family_conditional=False,
        ),
        "global_conformal_risk_control": _run_folded(
            raw_videos,
            acceptance_rates,
            selector=_select_crc_threshold,
            family_conditional=False,
        ),
        "mondrian_family_conformal_risk_control": _run_folded(
            raw_videos,
            acceptance_rates,
            selector=_select_crc_threshold,
            family_conditional=True,
        ),
        "pu_family_density_ratio": _run_folded(
            raw_videos,
            acceptance_rates,
            selector=_select_point_threshold,
            family_conditional=True,
            pu_transform=True,
        ),
    }

    ranked = _video_tail_rank(raw_videos)
    zero_thresholds = {family: 0.0 for family in FAMILIES}
    video_rank = _evaluate(ranked, zero_thresholds, acceptance_rates)
    mechanisms["per_video_tail_rank"] = {
        "folds": [],
        "families": video_rank["families"],
        "both_families_pass": video_rank["both_families_pass"],
        "transductive_rule": "candidate score must exceed every sampled background score in its video",
    }

    # With the conformal plus-one correction, a sampled tail probability is at
    # least 1/(n+1).  Compare that resolvable rank to each video's exact raw
    # top-k allocation; no label is opened for this bound.
    topk_families: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [video for video in raw_videos if _family(video["identity"]) == family]
        raw_total = sum(int(video["raw_background"]) for video in selected)
        budget_total = acceptance_rates[family] * raw_total
        correct_total = sum(int(np.sum(np.asarray(video["labels"]) == 1)) for video in selected)
        certified = 0
        optimistic = 0
        harmful_optimistic = 0
        for video in selected:
            background = np.asarray(video["background_confidence"])
            labels = np.asarray(video["labels"])
            candidates = np.asarray(video["labeled_confidence"])
            allocated = acceptance_rates[family] * int(video["raw_background"])
            if len(background):
                conservative_ranks = np.asarray(
                    [
                        (1 + np.sum(background >= value))
                        * int(video["raw_background"])
                        / (len(background) + 1)
                        for value in candidates
                    ]
                )
                certified += int(
                    np.sum((conservative_ranks <= allocated) & (labels == 1))
                )
            if len(background):
                optimistic_keep = candidates > np.max(background)
                optimistic += int(np.sum(optimistic_keep & (labels == 1)))
                harmful_optimistic += int(np.sum(optimistic_keep & (labels == 0)))
        topk_families[family] = {
            "video_count": len(selected),
            "known_correct_total": correct_total,
            "certified_known_correct_kept": certified,
            "certified_correct_retention": certified / max(correct_total, 1),
            "optimistic_sample_known_correct_kept": optimistic,
            "optimistic_sample_correct_retention": optimistic / max(correct_total, 1),
            "optimistic_sample_known_harmful_kept": harmful_optimistic,
            "exact_count_background_budget": budget_total,
            "exact_count_budget_pass_by_construction": True,
            "retention_pass": certified / max(correct_total, 1) >= RETENTION_FLOOR,
            "local_gate_pass": (
                certified / max(correct_total, 1) >= RETENTION_FLOOR
            ),
        }
    mechanisms["hard_per_video_topk_budget"] = {
        "folds": [],
        "families": topk_families,
        "both_families_pass": all(
            topk_families[family]["local_gate_pass"] for family in FAMILIES
        ),
    }

    verdict = (
        "GO"
        if any(
            result.get("both_families_pass") is True
            for name, result in mechanisms.items()
            if name != "global_point_threshold_control"
        )
        else "PARK"
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "verdict": verdict,
        "scope": {
            "fit_only": True,
            "train_video_count": len(train_ids),
            "validation_video_count": len(validation_ids),
            "validation_used_for_selection": False,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "preregistration": {
            "folds": 5,
            "held_family_order": ["44b6", "6bba"],
            "correct_edge_retention_floor": RETENTION_FLOOR,
            "fit117_background_budgets": fit117_budgets,
            "acceptance_rates": acceptance_rates,
            "kill_line": (
                "projected held background FP exceeds the CG0 family rate, "
                "or known-correct retention is below 0.20 in either family"
            ),
        },
        "data": {
            "frozen_gate": str(gate_path.relative_to(ROOT)),
            "frozen_gate_previously_fit_on_train105": True,
            "held_out_component": (
                "mechanism operating-point calibration only; the fixed base "
                "scorer is not an outer-fold OOF identity"
            ),
            "prepared_background_sample_count": int(
                prepare["totals"]["background_proposals"]
            ),
            "held_calibration_train105_video_count": len(raw_videos),
            "held_calibration_background_sample_count": sum(
                len(video["background_confidence"]) for video in raw_videos
            ),
            "fit_validation12_opened_for_selection": False,
        },
        "execution": {
            "device": "cpu",
            "model_training": False,
            "graph_replay": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
        "mechanisms": mechanisms,
        "decision": {
            "plan_survivor_count": 0 if verdict != "GO" else 1,
            "cg2_authorized": verdict == "GO",
            "reason": (
                "No non-control mechanism passed both family budgets and the "
                "preregistered correct-edge retention floor."
                if verdict != "GO"
                else "At least one non-control mechanism passed the CG1 local gate."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    parser.add_argument("--cg0", type=Path, default=DEFAULT_CG0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(child) for child in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(args)
    if args.output.exists():
        raise CG1ScreenError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preregistered common-36 video-subsample fidelity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "g02.subsample-fidelity.data-lane.v2"
SEED_NAMESPACE = "g02-subsample-fidelity-v1"
BASE_SEED = 20260729
DRAWS = 10_000
KS = (3, 5, 8, 12)
SCHEMES = ("simple_random", "residual_top10", "family_residual")
NAMES = ("APNCC", "C1", "H1-ZWRFEAT", "SA-DELETE", "SA-SWAP", "NODETOPO")
AMENDMENT_SHA256 = "3c497948189b9518c624e3c008cbfa021bc0072a7067a9ebd93e08163b9673fc"


@dataclass(frozen=True)
class Arm:
    name: str
    adjusted_numerator: np.ndarray
    edge_weight: np.ndarray
    division_tp: np.ndarray
    division_fp: np.ndarray
    division_fn: np.ndarray
    full_score: float
    full_delta: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pair_seed(k: int, scheme: str) -> int:
    material = f"{SEED_NAMESPACE}|{BASE_SEED}|{k}|{scheme}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def rows_to_arm(name: str, rows: list[dict[str, Any]], full_delta: float) -> Arm:
    edge_weight = np.asarray(
        [
            float(row["edge_tp"]) + float(row["edge_fp"]) + float(row["edge_fn"])
            for row in rows
        ],
        dtype=np.float64,
    )
    adjusted = np.asarray(
        [float(row["adj_edge_jaccard"]) for row in rows], dtype=np.float64
    )
    return Arm(
        name=name,
        adjusted_numerator=edge_weight * adjusted,
        edge_weight=edge_weight,
        division_tp=np.asarray(
            [float(row["division_tp"]) for row in rows], dtype=np.float64
        ),
        division_fp=np.asarray(
            [float(row["division_fp"]) for row in rows], dtype=np.float64
        ),
        division_fn=np.asarray(
            [float(row["division_fn"]) for row in rows], dtype=np.float64
        ),
        full_score=score_arrays(
            edge_weight * adjusted,
            edge_weight,
            np.asarray([float(row["division_tp"]) for row in rows]),
            np.asarray([float(row["division_fp"]) for row in rows]),
            np.asarray([float(row["division_fn"]) for row in rows]),
            np.arange(len(rows), dtype=np.int64),
            np.ones(len(rows), dtype=np.float64),
        ),
        full_delta=float(full_delta),
    )


def score_arrays(
    adjusted_numerator: np.ndarray,
    edge_weight: np.ndarray,
    division_tp: np.ndarray,
    division_fp: np.ndarray,
    division_fn: np.ndarray,
    selected: np.ndarray,
    design_weights: np.ndarray,
) -> float:
    adjusted_denominator = float(np.dot(design_weights, edge_weight[selected]))
    if adjusted_denominator <= 0:
        return float("nan")
    adjusted = (
        float(np.dot(design_weights, adjusted_numerator[selected]))
        / adjusted_denominator
    )
    div_tp = float(np.dot(design_weights, division_tp[selected]))
    div_fp = float(np.dot(design_weights, division_fp[selected]))
    div_fn = float(np.dot(design_weights, division_fn[selected]))
    div_total = div_tp + div_fp + div_fn
    return adjusted if div_total <= 0 else adjusted + 0.1 * div_tp / div_total


def score_arm(arm: Arm, selected: np.ndarray, weights: np.ndarray) -> float:
    return score_arrays(
        arm.adjusted_numerator,
        arm.edge_weight,
        arm.division_tp,
        arm.division_fp,
        arm.division_fn,
        selected,
        weights,
    )


def score_rows(
    rows: list[dict[str, Any]], design_weights: np.ndarray | None = None
) -> float:
    """Small public helper used by focused synthetic tests."""

    arm = rows_to_arm("_test", rows, 0.0)
    weights = (
        np.ones(len(rows), dtype=np.float64)
        if design_weights is None
        else design_weights
    )
    return score_arm(arm, np.arange(len(rows), dtype=np.int64), weights)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = average_ranks(values_a)
    rank_b = average_ranks(values_b)
    centered_a = rank_a - np.mean(rank_a)
    centered_b = rank_b - np.mean(rank_b)
    denominator = math.sqrt(
        float(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b))
    )
    return (
        float(np.dot(centered_a, centered_b) / denominator)
        if denominator
        else float("nan")
    )


def allocate_largest_remainder(
    stratum_sizes: dict[str, int], k: int
) -> dict[str, int] | None:
    sizes = {key: size for key, size in sorted(stratum_sizes.items()) if size > 0}
    if k < len(sizes) or k > sum(sizes.values()):
        return None
    total = sum(sizes.values())
    quotas = {key: k * size / total for key, size in sizes.items()}
    allocation = {
        key: min(size, max(1, math.floor(quotas[key])))
        for key, size in sizes.items()
    }
    while sum(allocation.values()) < k:
        available = [key for key in sizes if allocation[key] < sizes[key]]
        key = min(
            available,
            key=lambda item: (
                -(quotas[item] - math.floor(quotas[item])),
                item,
            ),
        )
        allocation[key] += 1
    while sum(allocation.values()) > k:
        available = [key for key in sizes if allocation[key] > 1]
        if not available:
            return None
        key = min(
            available,
            key=lambda item: (
                quotas[item] - math.floor(quotas[item]),
                -allocation[item],
                item,
            ),
        )
        allocation[key] -= 1
    return allocation


def strata_from_reference(
    identities: tuple[str, ...],
    reference_rows: list[dict[str, Any]],
    scheme: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n = len(identities)
    # Official adjusted-edge summary is additive in edge_weight * adjusted Jaccard.
    # The complementary contribution is therefore the frozen adjusted-edge error mass.
    residual = np.asarray(
        [
            (
                float(row["edge_tp"])
                + float(row["edge_fp"])
                + float(row["edge_fn"])
            )
            * (1.0 - float(row["adj_edge_jaccard"]))
            for row in reference_rows
        ],
        dtype=np.float64,
    )
    residual_by_id = {
        identity: float(value)
        for identity, value in zip(identities, residual, strict=True)
    }
    if scheme == "simple_random":
        return {"ALL": np.arange(n, dtype=np.int64)}, residual_by_id
    top_order = sorted(
        range(n), key=lambda index: (-residual[index], identities[index])
    )
    top10 = set(top_order[:10])
    strata: dict[str, list[int]] = {}
    for index in range(n):
        residual_label = "TOP10" if index in top10 else "REST"
        # The amended universe is single-family 44b6, so family_residual is
        # structurally degenerate with residual_top10.
        key = residual_label
        strata.setdefault(key, []).append(index)
    return (
        {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in sorted(strata.items())
        },
        residual_by_id,
    )


def sample_once(
    rng: np.random.Generator,
    strata: dict[str, np.ndarray],
    allocation: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    weights: list[float] = []
    for key in sorted(strata):
        population = strata[key]
        count = allocation[key]
        draw = np.sort(rng.choice(population, size=count, replace=False))
        selected.extend(int(value) for value in draw)
        weights.extend([len(population) / count] * count)
    order = np.argsort(np.asarray(selected), kind="mergesort")
    return (
        np.asarray(selected, dtype=np.int64)[order],
        np.asarray(weights, dtype=np.float64)[order],
    )


def q(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def run_pair(
    k: int,
    scheme: str,
    identities: tuple[str, ...],
    reference: Arm,
    arms: list[Arm],
    reference_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strata, residual_by_id = strata_from_reference(
        identities, reference_rows, scheme
    )
    allocation = allocate_largest_remainder(
        {key: len(value) for key, value in strata.items()}, k
    )
    if allocation is None:
        raise RuntimeError(f"non-estimable allocation for k={k}/{scheme}")
    seed = pair_seed(k, scheme)
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty((DRAWS, len(arms)), dtype=np.float64)
    draw_matrix_hasher = hashlib.sha256()
    draw_matrix_hash_spec = {
        "schema": "g02.subsample-fidelity.draw-matrix-hash.v1",
        "meaning": (
            "sampled common-universe row indices followed by design weights "
            "for every draw, in draw order"
        ),
        "header_encoding": "canonical JSON (sorted keys, compact separators) plus LF",
        "row_encoding": (
            "selected indices as little-endian uint32 C-order bytes, then "
            "design weights as little-endian float64 C-order bytes"
        ),
        "draws": DRAWS,
        "k": k,
        "scheme": scheme,
        "seed_uint64": seed,
        "allocation": allocation,
        "candidate_order": [arm.name for arm in arms],
        "universe_video_count": len(identities),
    }
    draw_matrix_hasher.update(canonical_bytes(draw_matrix_hash_spec) + b"\n")
    draw_zero: dict[str, Any] | None = None
    for draw_index in range(DRAWS):
        selected, weights = sample_once(rng, strata, allocation)
        draw_matrix_hasher.update(
            np.asarray(selected, dtype="<u4", order="C").tobytes(order="C")
        )
        draw_matrix_hasher.update(
            np.asarray(weights, dtype="<f8", order="C").tobytes(order="C")
        )
        reference_score = score_arm(reference, selected, weights)
        for arm_index, arm in enumerate(arms):
            estimates[draw_index, arm_index] = (
                score_arm(arm, selected, weights) - reference_score
            )
        if draw_index == 0:
            draw_zero = {
                "video_ids": [identities[index] for index in selected],
                "design_weights": {
                    identities[index]: float(weight)
                    for index, weight in zip(selected, weights, strict=True)
                },
            }
    assert draw_zero is not None

    full_deltas = np.asarray([arm.full_delta for arm in arms], dtype=np.float64)
    rhos = np.asarray(
        [spearman(estimates[index], full_deltas) for index in range(DRAWS)],
        dtype=np.float64,
    )
    sign_matches = np.sign(estimates) == np.sign(full_deltas)[None, :]
    pooled_sign = float(np.mean(sign_matches))
    draw_sign = np.mean(sign_matches, axis=1)
    per_candidate = []
    widths = []
    for index, arm in enumerate(arms):
        values = estimates[:, index]
        low = q(values, 0.025)
        high = q(values, 0.975)
        width = high - low
        widths.append(width)
        errors = values - arm.full_delta
        per_candidate.append(
            {
                "k": k,
                "scheme": scheme,
                "candidate": arm.name,
                "draws": DRAWS,
                "full_delta": arm.full_delta,
                "mean_estimate": float(np.mean(values)),
                "median_estimate": float(np.median(values)),
                "bias": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "median_absolute_error": float(np.median(np.abs(errors))),
                "ci95_low": low,
                "ci95_high": high,
                "ci95_width": width,
                "sign_agreement": float(np.mean(sign_matches[:, index])),
            }
        )

    median_rho = float(np.median(rhos))
    max_width = max(widths)
    threshold_checks = {
        "abs_median_spearman_ge_0p8": abs(median_rho) >= 0.8,
        "pooled_sign_agreement_ge_0p9": pooled_sign >= 0.9,
        "max_ci95_width_lt_0p005": max_width < 0.005,
    }
    usable_literal = all(threshold_checks.values())
    reusable = usable_literal and median_rho > 0
    top10 = sorted(
        residual_by_id, key=lambda identity: (-residual_by_id[identity], identity)
    )[:10]
    row = {
        "k": k,
        "scheme": scheme,
        "seed": seed,
        "seed_type": "uint64",
        "draws": DRAWS,
        "candidate_count": len(arms),
        "candidate_order": [arm.name for arm in arms],
        "draw_matrix_sha256": draw_matrix_hasher.hexdigest(),
        "draw_matrix_hash_spec": draw_matrix_hash_spec,
        "estimate_matrix_sha256": hashlib.sha256(
            np.asarray(estimates, dtype="<f8", order="C").tobytes(order="C")
        ).hexdigest(),
        "estimate_matrix_hash_spec": (
            "draw-major, candidate-order little-endian float64 C-order bytes"
        ),
        "median_spearman_rho": median_rho,
        "rho_p2p5": q(rhos, 0.025),
        "rho_p97p5": q(rhos, 0.975),
        "pooled_sign_agreement": pooled_sign,
        "draw_sign_agreement_p2p5": q(draw_sign, 0.025),
        "draw_sign_agreement_p97p5": q(draw_sign, 0.975),
        "max_candidate_ci95_width": max_width,
        "threshold_checks": threshold_checks,
        "usable_literal_rule": usable_literal,
        "direction_reversing": median_rho < 0,
        "reusable_positive_direction": reusable,
        "degenerate_duplicate_of": (
            "residual_top10" if scheme == "family_residual" else None
        ),
        "allocation": allocation,
        "stratum_sizes": {key: len(value) for key, value in strata.items()},
        "reference_residual_top10": top10,
        "draw_zero": draw_zero,
    }
    return row, per_candidate


def select_pair(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in rows if row["reusable_positive_direction"]]
    if not usable:
        return None
    minimum_k = min(row["k"] for row in usable)
    contenders = [row for row in usable if row["k"] == minimum_k]
    if any(row["scheme"] == "residual_top10" for row in contenders):
        contenders = [
            row for row in contenders if row["scheme"] != "family_residual"
        ]
    return min(
        contenders,
        key=lambda row: (
            row["max_candidate_ci95_width"],
            -row["pooled_sign_agreement"],
            -row["median_spearman_rho"],
            row["scheme"],
        ),
    )


def markdown_table(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> str:
    lines = [
        "# Common-36 held-44b6 fidelity tables",
        "",
        f"Verdict: **{verdict['verdict']}**.",
        "",
        "| k | scheme | median rho | sign agreement | max 95% width | usable |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | `{row['scheme']}` | "
            f"{row['median_spearman_rho']:.6f} | "
            f"{row['pooled_sign_agreement']:.6f} | "
            f"{row['max_candidate_ci95_width']:.9f} | "
            f"{'YES' if row['reusable_positive_direction'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "All cells use six candidates, exactly 10,000 finite-population draws, "
            "the common 36-video held-44b6 universe, and a common CG0 reference. "
            "`family_residual` is a single-family degenerate form of "
            "`residual_top10` and cannot win a tie.",
            "",
            "This is screening-only evidence. Every promotion still requires the full "
            "frozen 44b6-first graph gate and downstream campaign gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_contract(
    output_dir: Path,
    selected: dict[str, Any],
    decomposition_sha: str,
    universe: dict[str, Any],
) -> tuple[str, str]:
    contract = {
        "schema": "g02.subsample-contract.v1",
        "status": "FROZEN_SCREENING_ONLY",
        "k": selected["k"],
        "scheme": selected["scheme"],
        "video_ids": selected["draw_zero"]["video_ids"],
        "design_weights": selected["draw_zero"]["design_weights"],
        "selection_rule": (
            "draw zero from lexicographically sorted authenticated common-36 video "
            "IDs using the preregistered pair seed and allocation"
        ),
        "seed": selected["seed"],
        "generator": "numpy.random.Generator(PCG64)",
        "estimator": (
            "design-weighted official adjusted-edge sufficient statistics plus "
            "design-weighted micro division counts; candidate minus CG0"
        ),
        "universe": universe,
        "source_decomposition_sha256": decomposition_sha,
        "preregistration_amendment_sha256": AMENDMENT_SHA256,
        "valid_only_for": "strategy screening on standardized held-44b6 candidate-vs-CG0",
        "invalid_for": [
            "promotion decision",
            "6bba",
            "CV57",
            "public leaderboard",
            "full-family gate replacement",
        ],
        "promotion_requires_full_44b6_gate": True,
    }
    content_sha = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    contract["content_sha256_without_self_field"] = content_sha
    path = output_dir / "SUBSAMPLE_CONTRACT.json"
    write_json(path, contract)
    return content_sha, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    decomposition_path = args.decomposition.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decomposition_sha = sha256_file(decomposition_path)
    decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
    if decomposition.get("status") != "PASS":
        raise RuntimeError("decomposition replay did not pass")
    if decomposition["amendment"]["sha256"] != AMENDMENT_SHA256:
        raise RuntimeError("amendment identity mismatch")
    if any(
        check.get("pass") is not True
        for check in decomposition["authentication"]["aggregate_checks"]
    ):
        raise RuntimeError("aggregate replay check failed")

    identities = tuple(decomposition["universe"]["video_ids"])
    reference_rows = decomposition["reference"]["rows"]
    reference = rows_to_arm("CG0", reference_rows, 0.0)
    arms = []
    for name in NAMES:
        candidate = decomposition["candidates"][name]
        arm = rows_to_arm(name, candidate["rows"], candidate["delta_vs_cg0"])
        if abs(arm.full_score - float(candidate["summary"]["score"])) > 1e-12:
            raise RuntimeError(f"{name}: full score reconstruction mismatch")
        arms.append(arm)

    rows = []
    candidate_diagnostics = []
    for k in KS:
        for scheme in SCHEMES:
            row, diagnostics = run_pair(
                k, scheme, identities, reference, arms, reference_rows
            )
            rows.append(row)
            candidate_diagnostics.extend(diagnostics)

    selected = select_pair(rows)
    if selected is None:
        verdict = {
            "verdict": "SUBSAMPLING_UNSAFE",
            "reason": "No (k, scheme) passed all preregistered thresholds with positive rank direction.",
            "selected_pair": None,
            "contract_emitted": False,
        }
        contract_hashes = None
    else:
        verdict = {
            "verdict": "USABLE_SCREENING_SUBSAMPLE",
            "reason": "Minimum k passing all preregistered thresholds with positive rank direction.",
            "selected_pair": {"k": selected["k"], "scheme": selected["scheme"]},
            "contract_emitted": True,
        }
        content_sha, file_sha = write_contract(
            output_dir,
            selected,
            decomposition_sha,
            decomposition["universe"],
        )
        contract_hashes = {
            "content_sha256_without_self_field": content_sha,
            "file_sha256": file_sha,
        }
        verdict["contract_hashes"] = contract_hashes

    tables = {
        "schema": f"{SCHEMA}.tables",
        "status": "COMPLETE",
        "amendment_sha256": AMENDMENT_SHA256,
        "decomposition_path": str(decomposition_path),
        "decomposition_sha256": decomposition_sha,
        "universe": decomposition["universe"],
        "reference": "CG0",
        "candidate_names": list(NAMES),
        "draws_per_pair": DRAWS,
        "rows": rows,
        "candidate_diagnostics": candidate_diagnostics,
        "verdict": verdict,
    }
    write_json(output_dir / "fidelity_tables.json", tables)
    write_json(output_dir / "verdict.json", verdict)
    (output_dir / "fidelity_tables.md").write_text(
        markdown_table(rows, verdict), encoding="utf-8"
    )

    summary_fields = [
        "k",
        "scheme",
        "candidate_count",
        "draws",
        "median_spearman_rho",
        "rho_p2p5",
        "rho_p97p5",
        "pooled_sign_agreement",
        "max_candidate_ci95_width",
        "usable_literal_rule",
        "reusable_positive_direction",
        "degenerate_duplicate_of",
    ]
    with (output_dir / "fidelity_tables.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in summary_fields})

    diagnostic_fields = [
        "k",
        "scheme",
        "candidate",
        "draws",
        "full_delta",
        "mean_estimate",
        "median_estimate",
        "bias",
        "rmse",
        "median_absolute_error",
        "ci95_low",
        "ci95_high",
        "ci95_width",
        "sign_agreement",
    ]
    with (output_dir / "candidate_diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        for diagnostic in candidate_diagnostics:
            writer.writerow(
                {field: diagnostic[field] for field in diagnostic_fields}
            )

    inventory = {
        "schema": f"{SCHEMA}.source-inventory",
        "status": "AUTHENTICATED",
        "amendment_sha256": AMENDMENT_SHA256,
        "decomposition_sha256": decomposition_sha,
        "universe": decomposition["universe"],
        "reference": "CG0",
        "candidates": [
            {
                "candidate": name,
                "video_count": len(decomposition["candidates"][name]["rows"]),
                "standardized_delta_vs_cg0": decomposition["candidates"][name][
                    "delta_vs_cg0"
                ],
                "aggregate_score": decomposition["candidates"][name]["summary"][
                    "score"
                ],
            }
            for name in NAMES
        ],
        "authentication": decomposition["authentication"],
        "historical_scope_diagnostics": decomposition[
            "historical_scope_diagnostics"
        ],
        "failure_semantics": (
            "Any source hash, universe, fragment, label, scorer, or aggregate mismatch "
            "halts before draws and yields no scientific verdict or contract."
        ),
    }
    write_json(output_dir / "source_inventory.json", inventory)

    state_fragment = {
        "schema": "g02.subsample-fidelity.state-fragment.v1",
        "proposal_only": True,
        "root_merge_required": True,
        "tier_moves": [],
        "verdict": verdict["verdict"],
        "selected_pair": verdict["selected_pair"],
        "contract_hashes": contract_hashes,
        "promotion_rule_unchanged": True,
        "live_state_written": False,
        "ledger_written": False,
        "tier_census_written": False,
    }
    write_json(output_dir / "state_fragment.json", state_fragment)

    selected_text = (
        "none; SUBSAMPLING_UNSAFE"
        if selected is None
        else f"k={selected['k']}, {selected['scheme']}"
    )
    (output_dir / "PROPOSED_LEDGER_ROW.md").write_text(
        "| SUBSAMPLE-FIDELITY-20260729 | retrospective common-36 held-44b6 "
        "screening fidelity | CPU-only banked decomposition replay + 10,000 finite-"
        f"population draws/cell | selected `{selected_text}` | `{verdict['verdict']}` "
        "| diagnostics never gate; contract is screening-only and every promotion "
        "still requires full 44b6 | 0 GPU / 0 Kaggle / 0 RunPod / $0 | "
        "`tiered_surface/subsample_fidelity/lanes/data/TERMINAL_REPORT.md` |\n",
        encoding="utf-8",
    )

    execution = {
        "schema": f"{SCHEMA}.execution",
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cells": len(KS) * len(SCHEMES),
        "draws_per_cell": DRAWS,
        "candidate_estimates": len(KS) * len(SCHEMES) * DRAWS * len(arms),
        "cpu_only": True,
        "model_inference": False,
        "graph_construction": False,
        "new_family_or_split": False,
        "gpu": False,
        "kaggle": False,
        "runpod": False,
        "budget_debit": False,
        "tier_change": False,
        "dead_action": False,
    }
    write_json(output_dir / "execution_receipt.json", execution)

    best_line = (
        "No cell passed all thresholds."
        if selected is None
        else (
            f"Minimum reusable pair: k={selected['k']} `{selected['scheme']}`; "
            f"median rho={selected['median_spearman_rho']:.6f}, sign="
            f"{selected['pooled_sign_agreement']:.6f}, max width="
            f"{selected['max_candidate_ci95_width']:.9f}."
        )
    )
    (output_dir / "TERMINAL_REPORT.md").write_text(
        "# Video-subsample fidelity — data lane terminal report\n\n"
        f"**{verdict['verdict']}**. {best_line}\n\n"
        "The scorer-only replay authenticated six banked candidate graphs against "
        "one common CG0 reference on the exact 36-video held-44b6 universe and "
        "reproduced every cited aggregate to absolute tolerance `1e-12` before any "
        "draw. APNCC's CV57 and SA-DELETE's five extra 44b6 rows were retained only "
        "as provenance diagnostics, never mixed into ranks.\n\n"
        "Exactly 10,000 draws were run for each of 12 `(k, scheme)` cells. "
        "`family_residual` is reported as the preregistered one-family degeneracy "
        "and cannot beat `residual_top10` on a tie.\n\n"
        "No model inference, graph construction, new family/split, GPU, Kaggle, "
        "RunPod, budget debit, submission, tier move, DEAD action, or root-owned "
        "state write occurred. This lane proposes state and ledger fragments only. "
        "Every promotion still requires the full frozen 44b6-first graph gate.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-check amended fidelity outputs from the emitted sufficient statistics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DECOMPOSITION = HERE / "amended_decomposition_rows.json"
REPRODUCTION = HERE / "amended_reproduction.json"
MANIFEST = HERE / "amended_source_manifest.json"
GUARD = HERE / "amended_guard_retry.json"
OUTPUT = HERE / "amended_validation_checks.json"
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def score(rows: list[dict[str, Any]], chosen: np.ndarray, weights: np.ndarray) -> float:
    selected = [rows[int(index)] for index in chosen]
    unions = np.asarray([
        float(row["edge_tp"] + row["edge_fp"] + row["edge_fn"])
        for row in selected
    ])
    adjusted = np.asarray([float(row["adj_edge_jaccard"]) for row in selected])
    edge_score = float(np.dot(weights, unions * adjusted) / np.dot(weights, unions))
    division_tp = np.asarray([float(row["division_tp"]) for row in selected])
    division_denominator = np.asarray([
        float(row["division_tp"] + row["division_fp"] + row["division_fn"])
        for row in selected
    ])
    denominator = float(np.dot(weights, division_denominator))
    if denominator == 0.0:
        return edge_score
    return edge_score + 0.1 * float(np.dot(weights, division_tp)) / denominator


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        result[order[begin:end]] = (begin + 1 + end) / 2.0
        begin = end
    return result


def rho(values: np.ndarray, target: np.ndarray) -> float:
    observed = ranks(values)
    if np.std(observed) == 0.0 or np.std(target) == 0.0:
        return float("nan")
    centered_observed = observed - np.mean(observed)
    centered_target = target - np.mean(target)
    return float(
        np.dot(centered_observed, centered_target)
        / math.sqrt(
            float(np.dot(centered_observed, centered_observed))
            * float(np.dot(centered_target, centered_target))
        )
    )


def allocate(strata: dict[str, list[int]], k: int) -> dict[str, int]:
    total = sum(map(len, strata.values()))
    quotas = {key: k * len(value) / total for key, value in strata.items()}
    result = {key: max(1, math.floor(value)) for key, value in quotas.items()}
    while sum(result.values()) < k:
        eligible = [key for key in strata if result[key] < len(strata[key])]
        key = sorted(
            eligible,
            key=lambda item: (-(quotas[item] % 1.0), item),
        )[0]
        result[key] += 1
    while sum(result.values()) > k:
        eligible = [key for key in strata if result[key] > 1]
        key = sorted(
            eligible,
            key=lambda item: (quotas[item] % 1.0, item),
        )[0]
        result[key] -= 1
    return result


def draw(
    generator: np.random.Generator,
    ids: list[str],
    top10: set[str],
    k: int,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if scheme == "simple_random":
        chosen = np.sort(generator.choice(len(ids), k, replace=False))
        return chosen, np.repeat(len(ids) / k, k), {"ALL": k}
    strata: dict[str, list[int]] = {}
    for index, identity in enumerate(ids):
        residual = "TOP10" if identity in top10 else "REST"
        family = identity.split(":", 1)[1].split("_", 1)[0]
        label = residual if scheme == "residual_top10" else f"{family}|{residual}"
        strata.setdefault(label, []).append(index)
    allocation = allocate(strata, k)
    pairs: list[tuple[int, float]] = []
    for label in sorted(strata):
        chosen = np.sort(
            generator.choice(strata[label], allocation[label], replace=False)
        )
        weight = len(strata[label]) / allocation[label]
        pairs.extend((int(index), weight) for index in chosen)
    pairs.sort()
    return (
        np.asarray([index for index, _ in pairs]),
        np.asarray([weight for _, weight in pairs]),
        allocation,
    )


def close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        raise RuntimeError(f"{label}: {observed!r} != {expected!r}")


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite amended validation checks")
    decomposition = load(DECOMPOSITION)
    reproduction = load(REPRODUCTION)
    manifest = load(MANIFEST)
    guard = load(GUARD)
    if guard["status"] != "PASS" or float(guard["swap_growth_mib"]) != 0.0:
        raise RuntimeError("guard receipt is not PASS with zero swap growth")
    if not manifest["all_fragment_hashes_matched"]:
        raise RuntimeError("fragment hash audit failed")
    if not manifest["all_replayed_aggregate_scores_matched_to_1e_12"]:
        raise RuntimeError("aggregate replay audit failed")

    ids = list(decomposition["common_video_ids"])
    candidates = list(reproduction["candidate_order"])
    reference = list(decomposition["reference"]["rows"])
    rows_by_candidate = {
        candidate: list(decomposition["candidates"][candidate]["rows"])
        for candidate in candidates
    }
    all_indices = np.arange(len(ids))
    all_weights = np.ones(len(ids))
    reference_full = score(reference, all_indices, all_weights)
    close(reference_full, float(reproduction["cg0_full_score"]), "CG0 full score")
    full_deltas = np.empty(len(candidates))
    for position, candidate in enumerate(candidates):
        candidate_full = score(rows_by_candidate[candidate], all_indices, all_weights)
        close(
            candidate_full,
            float(reproduction["candidate_full_scores"][candidate]),
            f"{candidate} full score",
        )
        full_deltas[position] = candidate_full - reference_full
        close(
            full_deltas[position],
            float(reproduction["standardized_full_deltas"][candidate]),
            f"{candidate} full delta",
        )

    residual_order = sorted(
        range(len(ids)),
        key=lambda index: (
            -float(reference[index]["edge_fp"] + reference[index]["edge_fn"]),
            ids[index],
        ),
    )
    top10 = {ids[index] for index in residual_order[:10]}
    if top10 != set(reproduction["top10_residual_video_ids"]):
        raise RuntimeError("top-10 residual stratum mismatch")

    target_ranks = ranks(full_deltas)
    checked_cells = []
    for cell in reproduction["cells"]:
        k = int(cell["k"])
        scheme = str(cell["scheme"])
        generator = np.random.Generator(np.random.PCG64(int(cell["seed_uint64"])))
        deltas = np.empty((int(cell["draws"]), len(candidates)))
        correlations = np.empty(int(cell["draws"]))
        agreements = np.empty((int(cell["draws"]), len(candidates)), dtype=bool)
        first = None
        for draw_index in range(int(cell["draws"])):
            chosen, weights, allocation = draw(generator, ids, top10, k, scheme)
            if draw_index == 0:
                first = (chosen, weights, allocation)
            reference_score = score(reference, chosen, weights)
            for candidate_index, candidate in enumerate(candidates):
                deltas[draw_index, candidate_index] = (
                    score(rows_by_candidate[candidate], chosen, weights)
                    - reference_score
                )
            correlations[draw_index] = rho(deltas[draw_index], target_ranks)
            agreements[draw_index] = (
                np.sign(deltas[draw_index]) == np.sign(full_deltas)
            )
        assert first is not None
        finite = correlations[np.isfinite(correlations)]
        median_rho = float(np.median(finite))
        sign_agreement = float(np.mean(agreements))
        widths = []
        for candidate_index, candidate in enumerate(candidates):
            low, high = np.quantile(deltas[:, candidate_index], [0.025, 0.975])
            widths.append(float(high - low))
            diagnostic = cell["candidate_diagnostics"][candidate]
            close(float(low), float(diagnostic["q025"]), f"{k}/{scheme}/{candidate} q025")
            close(float(high), float(diagnostic["q975"]), f"{k}/{scheme}/{candidate} q975")
        maximum_width = max(widths)
        close(median_rho, float(cell["median_spearman_rho"]), f"{k}/{scheme} rho")
        close(
            sign_agreement,
            float(cell["pooled_sign_agreement"]),
            f"{k}/{scheme} sign",
        )
        close(
            maximum_width,
            float(cell["maximum_candidate_95pct_width"]),
            f"{k}/{scheme} width",
        )
        expected_usable = bool(
            abs(median_rho) >= 0.8
            and sign_agreement >= 0.9
            and maximum_width < 0.005
        )
        if expected_usable != bool(cell["literal_usable"]):
            raise RuntimeError(f"{k}/{scheme} acceptance mismatch")
        first_expected = reproduction["draw_zero"][f"{k}|{scheme}"]
        if [ids[int(index)] for index in first[0]] != first_expected["video_ids"]:
            raise RuntimeError(f"{k}/{scheme} draw-zero list mismatch")
        if first[2] != first_expected["allocation"]:
            raise RuntimeError(f"{k}/{scheme} draw-zero allocation mismatch")
        expected_weights = [
            float(item["weight"]) for item in first_expected["design_weights"]
        ]
        if not np.allclose(first[1], expected_weights, rtol=0.0, atol=TOLERANCE):
            raise RuntimeError(f"{k}/{scheme} draw-zero weight mismatch")
        checked_cells.append({
            "k": k,
            "scheme": scheme,
            "seed_uint64": int(cell["seed_uint64"]),
            "candidate_order": candidates,
            "draw_matrix": {
                "shape": list(deltas.shape),
                "dtype": "little-endian-float64",
                "order": "C",
                "sha256": hashlib.sha256(
                    np.asarray(deltas, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            },
            "draw_zero": first_expected,
            "median_spearman_rho": median_rho,
            "pooled_sign_agreement": sign_agreement,
            "maximum_candidate_95pct_width": maximum_width,
            "usable": expected_usable,
        })

    usable = [cell for cell in checked_cells if cell["usable"]]
    if usable or reproduction["selected"] is not None:
        raise RuntimeError("expected no usable cell and no selected contract")
    replay_errors = {
        candidate: float(value["summary_score_abs_error"])
        for candidate, value in decomposition["replay_provenance"].items()
    }
    if any(error > TOLERANCE for error in replay_errors.values()):
        raise RuntimeError("aggregate replay error exceeds tolerance")
    result = {
        "schema": "g02.subsample-fidelity.amended-validation-checks.v1",
        "status": "PASS",
        "verdict": "SUBSAMPLING_UNSAFE",
        "candidate_order": candidates,
        "checked_cells": checked_cells,
        "usable_cell_count": 0,
        "selected_contract": None,
        "aggregate_replay_absolute_errors": replay_errors,
        "artifact_sha256": {
            "amended_decomposition_rows.json": sha256(DECOMPOSITION),
            "amended_reproduction.json": sha256(REPRODUCTION),
            "amended_source_manifest.json": sha256(MANIFEST),
            "amended_guard_retry.json": sha256(GUARD),
        },
        "checks": {
            "all_full_scores_and_deltas_recomputed": True,
            "top10_residual_stratum_recomputed": True,
            "all_120000_draws_recomputed": True,
            "all_cell_metrics_recomputed": True,
            "all_draw_zero_lists_recomputed": True,
            "acceptance_rule_reapplied": True,
            "zero_swap_guard_passed": True,
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({
        "status": result["status"],
        "verdict": result["verdict"],
        "checked_cells": len(checked_cells),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

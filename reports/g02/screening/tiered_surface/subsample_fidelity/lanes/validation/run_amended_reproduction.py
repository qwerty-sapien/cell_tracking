#!/usr/bin/env python3
"""Independent common-36 decomposition replay and fidelity reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
import scoreall  # type: ignore  # noqa: E402


AMENDMENT = (
    ROOT / "reports/g02/screening/tiered_surface/subsample_fidelity/"
    "PREREGISTRATION_AMENDMENT_V1.md"
)
AMENDMENT_SHA256 = "3c497948189b9518c624e3c008cbfa021bc0072a7067a9ebd93e08163b9673fc"
ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
ARCHIVE_BYTES = 87_393_127_165
ARCHIVE_CENTRAL_SHA256 = (
    "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77"
)
ARCHIVE_ENTRY_COUNT = 24_886
SCORER_HASHES = {
    "scripts/scoreall.py":
        "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c",
    "src/celltrack_alpha/official_eval.py":
        "22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py":
        "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py":
        "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
}
REPORTS = {
    "h1": (
        "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json",
        "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
    ),
    "c1": (
        "reports/g02/screening/top50_parallel_dags/branches/c1_ta_hoct_lite/"
        "outputs/c1_n3_44b6_graph.json",
        "6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10",
    ),
    "delete": (
        "reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json",
        "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a",
    ),
    "swap": (
        "reports/g02/screening/tiered_surface/branches/sa_swap/"
        "result_44b6_postrepair_v1/report.json",
        "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864",
    ),
    "node": (
        "reports/g02/screening/tiered_surface/branches/"
        "sa_nodetopo_gate/RESULT_44B6.json",
        "fb9202586242f506a7cd8437d780bc93ba1fcfe5de957e22de8d6a2e73f94834",
    ),
}
CANDIDATE_ORDER = (
    "APNCC", "C1", "H1-ZWRFEAT", "SA-DELETE", "SA-SWAP", "NODETOPO",
)
KS = (3, 5, 8, 12)
SCHEMES = ("simple_random", "residual_top10", "family_residual")
DRAW_COUNT = 10_000


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_inputs() -> dict[str, Any]:
    if (
        os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
        or os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
        or os.environ.get("CELLTRACK_CV57_ACCESS") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValidationError("binding guarded CPU-only environment is absent")
    receipts: list[dict[str, Any]] = []
    for path, expected in [(AMENDMENT, AMENDMENT_SHA256)] + [
        (ROOT / path, expected) for path, expected in SCORER_HASHES.items()
    ] + [
        (ROOT / relative, expected) for relative, expected in REPORTS.values()
    ]:
        observed = sha256_file(path)
        if observed != expected:
            raise ValidationError(f"hash mismatch: {path}")
        receipts.append({
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passes": True,
        })
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise ValidationError("raw archive size/identity mismatch")
    with zipfile.ZipFile(ARCHIVE) as archive:
        rows = [
            {
                "compressed_size": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "filename": info.filename,
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
    # Match the frozen archive-receipt convention used by the banked
    # NODETOPO generators: canonical JSON followed by one LF byte.
    central_hash = hashlib.sha256(canonical_bytes(rows) + b"\n").hexdigest()
    if central_hash != ARCHIVE_CENTRAL_SHA256 or len(rows) != ARCHIVE_ENTRY_COUNT:
        raise ValidationError("raw archive central-directory identity mismatch")
    return {
        "files": receipts,
        "archive": {
            "path": str(ARCHIVE),
            "bytes": ARCHIVE_BYTES,
            "entry_count": len(rows),
            "central_directory_sha256": central_hash,
            "passes": True,
        },
    }


def resolve_report_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if ROOT.resolve() not in path.parents:
        raise ValidationError(f"fragment outside immutable root: {path}")
    return path


def manifest_from_report_rows(
    rows: list[dict[str, Any]],
    common_ids: list[str],
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> scoreall.FragmentManifest:
    by_id = {str(row["identity"]): row for row in rows}
    if set(by_id) != set(common_ids) or len(by_id) != len(rows):
        raise ValidationError("report fragment universe mismatch")
    ordered = []
    for identity in sorted(common_ids):
        row = by_id[identity]
        path = resolve_report_path(str(row[path_key]))
        expected_hash = str(row[hash_key])
        if sha256_file(path) != expected_hash:
            raise ValidationError(f"fragment hash mismatch: {identity} {path}")
        ordered.append((identity, path))
    return scoreall.FragmentManifest(
        tuple(ordered),
        {
            "expected_fragment_count": len(common_ids),
            "discovered_csv_count": len(rows),
            "train_fragment_count": len(rows),
            "ignored_test_fragments": [],
            "missing_relevant_fragments": [],
            "extra_relevant_fragments": [],
            "duplicate_relevant_fragments": {},
            "invalid_fragment_paths": [],
        },
    )


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return scoreall.official_summary(rows)


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError(
            f"{label} mismatch: expected={expected!r}, observed={observed!r}"
        )


def replay_decompositions(
    reports: dict[str, dict[str, Any]],
    common_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    h1 = reports["h1"]
    c1 = reports["c1"]
    swap = reports["swap"]

    manifests = {
        "APNCC": manifest_from_report_rows(
            h1["graph"]["control"]["input_provenance"]["files"], common_ids
        ),
        "C1": manifest_from_report_rows(
            c1["graph"]["c1"]["input_provenance"]["files"], common_ids
        ),
        "H1-ZWRFEAT": manifest_from_report_rows(
            h1["graph"]["candidate"]["input_provenance"]["files"], common_ids
        ),
        "SA-SWAP": manifest_from_report_rows(
            [
                {
                    "identity": row["identity"],
                    "path": row["alpha"]["fragment"],
                    "sha256": row["alpha"]["fragment_sha256"],
                }
                for row in swap["graph"]["videos"]
            ],
            common_ids,
        ),
    }
    expected_aggregate_hashes = {
        "APNCC": h1["graph"]["control"]["input_provenance"]["aggregate_sha256"],
        "C1": c1["graph"]["c1"]["input_provenance"]["aggregate_sha256"],
        "H1-ZWRFEAT": h1["graph"]["candidate"]["input_provenance"]["aggregate_sha256"],
        "SA-SWAP": swap["graph"]["alpha"]["fragment_aggregate_sha256"],
    }
    expected_scores = {
        "APNCC": h1["graph"]["control"]["summary"]["score"],
        "C1": c1["graph"]["c1"]["summary"]["score"],
        "H1-ZWRFEAT": h1["graph"]["candidate"]["summary"]["score"],
        "SA-SWAP": swap["graph"]["alpha"]["summary"]["score"],
    }
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    with ZipStore(ARCHIVE, mode="r") as store:
        for candidate in ("APNCC", "C1", "H1-ZWRFEAT", "SA-SWAP"):
            started = time.perf_counter()
            rows, receipt, coverage = scoreall.score_fragment_input(
                store, manifests[candidate]
            )
            rows_by_id = {row["identity"]: row for row in rows}
            ordered_rows = [rows_by_id[identity] for identity in common_ids]
            if set(rows_by_id) != set(common_ids):
                raise ValidationError(f"{candidate} replay universe mismatch")
            if receipt["aggregate_sha256"] != expected_aggregate_hashes[candidate]:
                raise ValidationError(f"{candidate} aggregate fragment hash mismatch")
            candidate_summary = summary(ordered_rows)
            assert_close(
                candidate_summary["score"], expected_scores[candidate],
                f"{candidate} aggregate score replay",
            )
            rows_by_candidate[candidate] = ordered_rows
            provenance[candidate] = {
                "summary": candidate_summary,
                "expected_summary_score": expected_scores[candidate],
                "summary_score_abs_error": abs(
                    candidate_summary["score"] - expected_scores[candidate]
                ),
                "fragment_provenance": receipt,
                "coverage": coverage,
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(
                f"replayed {candidate}: {len(rows)} videos "
                f"score={candidate_summary['score']:.15f}",
                flush=True,
            )
    return rows_by_candidate, provenance


def direct_rows(
    reports: dict[str, dict[str, Any]],
    common_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    delete_video = {
        row["identity"]: row
        for row in reports["delete"]["family_results"]["44b6"]["videos"]
    }
    node_video = {row["identity"]: row for row in reports["node"]["per_video"]}
    if set(common_ids) != set(node_video):
        raise ValidationError("NODETOPO common universe mismatch")
    if not set(common_ids).issubset(delete_video):
        raise ValidationError("SA-DELETE is missing common-universe rows")
    cg0_rows = [node_video[identity]["baseline_metrics"] for identity in common_ids]
    delete_cg0_rows = [
        delete_video[identity]["baseline_metrics"] for identity in common_ids
    ]
    if canonical_sha256(cg0_rows) != canonical_sha256(delete_cg0_rows):
        raise ValidationError("CG0 baseline rows disagree between DELETE and NODETOPO")
    direct = {
        "SA-DELETE": [
            delete_video[identity]["candidate_metrics"] for identity in common_ids
        ],
        "NODETOPO": [
            node_video[identity]["candidate_metrics"] for identity in common_ids
        ],
    }
    cg0_summary = summary(cg0_rows)
    assert_close(
        cg0_summary["score"], reports["node"]["baseline"]["score"],
        "CG0 common-36 score",
    )
    node_summary = summary(direct["NODETOPO"])
    assert_close(
        node_summary["score"], reports["node"]["candidate"]["score"],
        "NODETOPO common-36 score",
    )
    return direct, cg0_rows, {
        "cg0_summary": cg0_summary,
        "cg0_rows_sha256": canonical_sha256(cg0_rows),
        "delete_common36_summary": summary(direct["SA-DELETE"]),
        "nodetopo_summary": node_summary,
        "delete_historical_rows_excluded": sorted(set(delete_video) - set(common_ids)),
    }


def score_selected(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    edge_union = np.array([
        float(row["edge_tp"] + row["edge_fp"] + row["edge_fn"]) for row in rows
    ])
    adjusted = np.array([float(row["adj_edge_jaccard"]) for row in rows])
    adj_den = float(np.sum(weights * edge_union[indices]))
    adj_num = float(np.sum(weights * edge_union[indices] * adjusted[indices]))
    if adj_den <= 0:
        raise ValidationError("nonpositive adjusted-edge denominator")
    score = adj_num / adj_den
    div_tp = np.array([float(row["division_tp"]) for row in rows])
    div_fp = np.array([float(row["division_fp"]) for row in rows])
    div_fn = np.array([float(row["division_fn"]) for row in rows])
    tp = float(np.sum(weights * div_tp[indices]))
    denom = float(np.sum(weights * (div_tp + div_fp + div_fn)[indices]))
    return score if denom == 0 else score + 0.1 * tp / denom


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        rank = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = rank
        position = end
    return ranks


def spearman(values: np.ndarray, target_ranks: np.ndarray) -> float:
    ranks = average_ranks(values)
    if np.std(ranks) == 0 or np.std(target_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(ranks, target_ranks)[0, 1])


def allocate(strata: dict[str, list[int]], k: int) -> dict[str, int]:
    if k < len(strata):
        raise ValidationError("k smaller than nonempty stratum count")
    n_total = sum(len(indices) for indices in strata.values())
    quotas = {name: k * len(indices) / n_total for name, indices in strata.items()}
    result = {name: max(1, math.floor(quotas[name])) for name in strata}
    while sum(result.values()) < k:
        candidates = [
            name for name in strata if result[name] < len(strata[name])
        ]
        if not candidates:
            raise ValidationError("stratum allocation cannot fill k")
        choice = min(
            candidates,
            key=lambda name: (-(quotas[name] - math.floor(quotas[name])), name),
        )
        result[choice] += 1
    while sum(result.values()) > k:
        candidates = [name for name in strata if result[name] > 1]
        if not candidates:
            raise ValidationError("stratum allocation cannot reduce to k")
        choice = min(
            candidates,
            key=lambda name: (
                quotas[name] - math.floor(quotas[name]), name,
            ),
        )
        result[choice] -= 1
    return result


def seed_for(k: int, scheme: str) -> int:
    value = f"g02-subsample-fidelity-v1|20260729|{k}|{scheme}"
    return int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:8], "big")


def draw_selection(
    rng: np.random.Generator,
    k: int,
    scheme: str,
    ids: list[str],
    top10: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if scheme == "simple_random":
        indices = np.sort(rng.choice(len(ids), size=k, replace=False))
        return indices, np.full(k, len(ids) / k), {"ALL": k}
    strata: dict[str, list[int]] = {}
    for index, identity in enumerate(ids):
        residual = "TOP10" if identity in top10 else "REST"
        family = identity.split(":", 1)[1].split("_", 1)[0]
        key = residual if scheme == "residual_top10" else f"{family}|{residual}"
        strata.setdefault(key, []).append(index)
    allocation = allocate(strata, k)
    selected: list[int] = []
    weights: list[float] = []
    for name in sorted(strata):
        chosen = np.sort(
            rng.choice(strata[name], size=allocation[name], replace=False)
        )
        selected.extend(int(index) for index in chosen)
        weights.extend([len(strata[name]) / allocation[name]] * allocation[name])
    order = np.argsort(selected)
    return (
        np.array(selected, dtype=int)[order],
        np.array(weights, dtype=float)[order],
        allocation,
    )


def reproduce(
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    cg0_rows: list[dict[str, Any]],
    common_ids: list[str],
) -> dict[str, Any]:
    cg0_full = summary(cg0_rows)["score"]
    full_scores = {
        candidate: summary(rows_by_candidate[candidate])["score"]
        for candidate in CANDIDATE_ORDER
    }
    full_deltas = np.array([
        full_scores[candidate] - cg0_full for candidate in CANDIDATE_ORDER
    ])
    target_ranks = average_ranks(full_deltas)
    residual_order = sorted(
        range(len(common_ids)),
        key=lambda index: (
            -float(cg0_rows[index]["edge_fp"] + cg0_rows[index]["edge_fn"]),
            common_ids[index],
        ),
    )
    top10 = {common_ids[index] for index in residual_order[:10]}
    cells: list[dict[str, Any]] = []
    draw_zero: dict[str, Any] = {}
    for k in KS:
        for scheme in SCHEMES:
            seed = seed_for(k, scheme)
            rng = np.random.Generator(np.random.PCG64(seed))
            deltas = np.empty((DRAW_COUNT, len(CANDIDATE_ORDER)), dtype=float)
            rhos = np.empty(DRAW_COUNT, dtype=float)
            signs = np.empty((DRAW_COUNT, len(CANDIDATE_ORDER)), dtype=bool)
            first_selection = None
            first_weights = None
            first_allocation = None
            for draw in range(DRAW_COUNT):
                indices, weights, allocation = draw_selection(
                    rng, k, scheme, common_ids, top10
                )
                if draw == 0:
                    first_selection = indices
                    first_weights = weights
                    first_allocation = allocation
                reference_score = score_selected(cg0_rows, indices, weights)
                for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
                    candidate_score = score_selected(
                        rows_by_candidate[candidate], indices, weights
                    )
                    deltas[draw, candidate_index] = candidate_score - reference_score
                rhos[draw] = spearman(deltas[draw], target_ranks)
                signs[draw] = np.sign(deltas[draw]) == np.sign(full_deltas)
            assert first_selection is not None
            assert first_weights is not None
            finite_rhos = rhos[np.isfinite(rhos)]
            intervals = {}
            widths = []
            for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
                values = deltas[:, candidate_index]
                low, high = np.quantile(values, [0.025, 0.975])
                width = float(high - low)
                widths.append(width)
                errors = values - full_deltas[candidate_index]
                intervals[candidate] = {
                    "full_delta": float(full_deltas[candidate_index]),
                    "q025": float(low),
                    "q975": float(high),
                    "width": width,
                    "bias": float(np.mean(errors)),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "median_absolute_error": float(np.median(np.abs(errors))),
                }
            median_rho = (
                float(np.median(finite_rhos)) if len(finite_rhos) else None
            )
            pooled_sign = float(np.mean(signs))
            max_width = max(widths)
            usable = bool(
                median_rho is not None
                and abs(median_rho) >= 0.8
                and pooled_sign >= 0.9
                and max_width < 0.005
            )
            cell = {
                "k": k,
                "scheme": scheme,
                "seed_uint64": seed,
                "draws": DRAW_COUNT,
                "undefined_rho_draws": int(np.sum(~np.isfinite(rhos))),
                "median_spearman_rho": median_rho,
                "rho_q025": (
                    float(np.quantile(finite_rhos, 0.025))
                    if len(finite_rhos) else None
                ),
                "rho_q975": (
                    float(np.quantile(finite_rhos, 0.975))
                    if len(finite_rhos) else None
                ),
                "pooled_sign_agreement": pooled_sign,
                "draw_sign_agreement_q025": float(
                    np.quantile(np.mean(signs, axis=1), 0.025)
                ),
                "draw_sign_agreement_q975": float(
                    np.quantile(np.mean(signs, axis=1), 0.975)
                ),
                "maximum_candidate_95pct_width": max_width,
                "candidate_diagnostics": intervals,
                "literal_usable": usable,
                "positive_direction_operationally_usable": bool(
                    usable and median_rho is not None and median_rho > 0
                ),
                "scheme_degenerate_with": (
                    "residual_top10" if scheme == "family_residual" else None
                ),
            }
            cells.append(cell)
            draw_zero[f"{k}|{scheme}"] = {
                "video_ids": [common_ids[index] for index in first_selection],
                "design_weights": [
                    {
                        "identity": common_ids[index],
                        "weight": float(weight),
                    }
                    for index, weight in zip(first_selection, first_weights)
                ],
                "allocation": first_allocation,
            }
            print(
                f"cell k={k} scheme={scheme} rho={median_rho} "
                f"sign={pooled_sign:.6f} width={max_width:.9f} usable={usable}",
                flush=True,
            )
    operational = [
        cell for cell in cells
        if cell["positive_direction_operationally_usable"]
    ]
    selected = None
    if operational:
        operational.sort(key=lambda cell: (
            cell["k"],
            cell["maximum_candidate_95pct_width"],
            -cell["pooled_sign_agreement"],
            -cell["median_spearman_rho"],
            cell["scheme"],
        ))
        selected = operational[0]
    return {
        "candidate_order": list(CANDIDATE_ORDER),
        "cg0_full_score": cg0_full,
        "candidate_full_scores": full_scores,
        "standardized_full_deltas": {
            candidate: float(delta)
            for candidate, delta in zip(CANDIDATE_ORDER, full_deltas)
        },
        "top10_residual_video_ids": sorted(top10),
        "cells": cells,
        "draw_zero": draw_zero,
        "selected": (
            None if selected is None else {
                "k": selected["k"],
                "scheme": selected["scheme"],
                "draw_zero": draw_zero[f"{selected['k']}|{selected['scheme']}"],
            }
        ),
    }


def write_table(path: Path, cells: list[dict[str, Any]]) -> None:
    fields = [
        "k", "scheme", "draws", "undefined_rho_draws",
        "median_spearman_rho", "rho_q025", "rho_q975",
        "pooled_sign_agreement", "maximum_candidate_95pct_width",
        "literal_usable", "positive_direction_operationally_usable",
        "scheme_degenerate_with",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow({field: cell[field] for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "amended_source_manifest.json",
        out_dir / "amended_decomposition_rows.json",
        out_dir / "amended_reproduction.json",
        out_dir / "amended_fidelity_table.csv",
    ]
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite amended outputs")

    started = time.perf_counter()
    input_receipts = verify_inputs()
    reports = {name: load_json(relative) for name, (relative, _) in REPORTS.items()}
    common_ids = sorted(row["identity"] for row in reports["h1"]["graph"]["videos"])
    other_universes = {
        "C1": {row["identity"] for row in reports["c1"]["graph"]["videos"]},
        "SA-SWAP": {row["identity"] for row in reports["swap"]["graph"]["videos"]},
        "NODETOPO": {row["identity"] for row in reports["node"]["per_video"]},
    }
    if len(common_ids) != 36 or any(
        set(common_ids) != universe for universe in other_universes.values()
    ):
        raise ValidationError("amended common-36 universe mismatch")

    replay_rows, replay_provenance = replay_decompositions(reports, common_ids)
    direct, cg0_rows, direct_provenance = direct_rows(reports, common_ids)
    rows_by_candidate = {**replay_rows, **direct}
    if tuple(rows_by_candidate) != CANDIDATE_ORDER:
        rows_by_candidate = {
            candidate: rows_by_candidate[candidate] for candidate in CANDIDATE_ORDER
        }
    reproduction = reproduce(rows_by_candidate, cg0_rows, common_ids)

    decomposition = {
        "schema": "g02.subsample-fidelity.amended-independent-decomposition.v1",
        "status": "PASS",
        "amendment_sha256": AMENDMENT_SHA256,
        "common_video_ids": common_ids,
        "common_universe_sha256": canonical_sha256(common_ids),
        "reference": {
            "candidate_id": "CG0",
            "rows": cg0_rows,
            "summary": direct_provenance["cg0_summary"],
        },
        "candidates": {
            candidate: {
                "rows": rows_by_candidate[candidate],
                "summary": summary(rows_by_candidate[candidate]),
                "source": (
                    "official_scorer_decomposition_replay"
                    if candidate in replay_provenance
                    else "banked_per_video_sufficient_statistics"
                ),
            }
            for candidate in CANDIDATE_ORDER
        },
        "replay_provenance": replay_provenance,
        "direct_provenance": direct_provenance,
    }
    manifest = {
        "schema": "g02.subsample-fidelity.amended-independent-source-manifest.v1",
        "status": "PASS",
        "input_receipts": input_receipts,
        "amendment": {
            "path": str(AMENDMENT),
            "sha256": AMENDMENT_SHA256,
            "governance_caveat": (
                "The validation lane originally judged this rescoring/restandardization "
                "outside the original preregistration. Root subsequently authorized the "
                "pre-outcome amendment as a decomposition-only node. Results are valid "
                "only under that amendment and do not retroactively describe the historical "
                "CV57 APNCC or 41-video SA-DELETE deltas."
            ),
        },
        "common_universe": {
            "video_count": 36,
            "family": "44b6",
            "video_ids": common_ids,
            "sha256": canonical_sha256(common_ids),
        },
        "candidate_count": len(CANDIDATE_ORDER),
        "candidate_ids": list(CANDIDATE_ORDER),
        "all_fragment_hashes_matched": True,
        "all_replayed_aggregate_scores_matched_to_1e_12": True,
        "graph_construction_performed": False,
        "model_inference_performed": False,
        "new_family_or_split_opened": False,
        "cv57_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    reproduction_output = {
        "schema": "g02.subsample-fidelity.amended-independent-reproduction.v1",
        "status": "PASS",
        "draw_count_per_cell": DRAW_COUNT,
        "acceptance_rule": {
            "abs_median_rho_min": 0.8,
            "pooled_sign_agreement_min": 0.9,
            "maximum_95pct_width_strict_max": 0.005,
            "positive_rho_required_for_contract": True,
        },
        **reproduction,
    }
    (out_dir / "amended_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (out_dir / "amended_decomposition_rows.json").write_text(
        json.dumps(decomposition, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (out_dir / "amended_reproduction.json").write_text(
        json.dumps(reproduction_output, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_table(out_dir / "amended_fidelity_table.csv", reproduction["cells"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recover and independently score a permissive G-04 division panel.

This script has two deliberately separate outputs:

1. an exact *edge-neutral analytic* curve for the frozen CV57 baseline, where
   the adjusted-edge term is held fixed and only division TP/FP change; and
2. a *measured decoder* frontier obtained by independently rescoring a bounded
   panel of graphs from the recoverable G-03 R4 content-addressed graph bank.

The latter is a dev69 measurement on Trackastra parent graphs.  It is not a
CV57 score and therefore cannot, by itself, promote a submission.  The old R4
SQLite files contain config-to-graph mappings but zero metric rows; this module
never treats those missing metrics as evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/g04/lanes/division_permissive"
DEFAULT_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_GRAPH_CACHE = (
    ROOT / "reports/g03/screening/wave_r/R4_artifacts/oof/graph_cache_v4"
)
DEFAULT_SPOOL_ROOT = (
    ROOT / "reports/g03/screening/wave_r/R4_artifacts/oof/runs_v4"
)
DEFAULT_CV57_FRAGMENTS = ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments"
DEFAULT_APPEARANCE_CACHE = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2"
    / "arm_c_appearance/cache"
)
FROZEN_CV57_BASELINE = Decimal("0.8815220699597315")
FROZEN_CV57_GT_DIVISIONS = 42
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MEASURED_SCHEMA = "celltrack-alpha.g04-division-permissive-measured-frontier.v1"
ANALYTIC_SCHEMA = "celltrack-alpha.g04-division-edge-neutral-curve.v1"
CV57_CACHE_SCHEMA = "celltrack-alpha.g04-division-permissive-cv57-cache-frontier.v1"
PROMOTED_TEST_SCHEMA = "celltrack-alpha.g04-promoted-division-test-package.v1"
EXPECTED_PUBLIC_PARENT_SHA256 = "6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3"
EXPECTED_PUBLIC_PARENT_ROWS = 263_282
EXPECTED_FROZEN_PREDICT_ARM_SHA256 = "8a08d663177317f324ea44595cfb35aa215fdc3d9af34a820041dca1dc82e379"
EXPECTED_RAW_ARCHIVE_BYTES = 87_393_127_165
EXPECTED_RAW_ARCHIVE_SHA256 = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
DEFAULT_PROMOTED_DIR = DEFAULT_OUTPUT / "promoted_unfiltered_geometry_cap5"
DEFAULT_PUBLIC_PARENT = DEFAULT_PROMOTED_DIR / "public_parent/submission.csv"
FROZEN_PREDICT_ARM = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2"
    / "predict_arm.py"
)

getcontext().prec = 50


class PermissiveSweepError(RuntimeError):
    """The permissive sweep input or result failed a consistency check."""


@dataclass(frozen=True)
class GraphBinding:
    config_index: int
    config_name: str
    identity: str
    fold_id: str
    graph_sha256: str
    predicted_forks: int
    duplicate_survivors: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def edge_neutral_case(
    true_forks: int,
    false_forks: int,
    *,
    gt_divisions: int = FROZEN_CV57_GT_DIVISIONS,
    adjusted_edge_score: Decimal = FROZEN_CV57_BASELINE,
) -> dict[str, Any]:
    """Return the exact division-only score for one TP/FP operating point."""

    if isinstance(true_forks, bool) or isinstance(false_forks, bool):
        raise PermissiveSweepError("fork counts must be integers, not booleans")
    if not 0 <= true_forks <= gt_divisions or false_forks < 0:
        raise PermissiveSweepError("division TP/FP counts are outside their valid domain")
    fn = gt_divisions - true_forks
    predicted = true_forks + false_forks
    denominator = true_forks + false_forks + fn
    jaccard = Decimal(true_forks) / Decimal(denominator) if denominator else Decimal(0)
    addition = Decimal("0.1") * jaccard
    combined = adjusted_edge_score + addition
    return {
        "predicted_forks": predicted,
        "tp": true_forks,
        "fp": false_forks,
        "fn": fn,
        "division_jaccard_exact": decimal_text(jaccard),
        "division_addition_exact": decimal_text(addition),
        "combined_score_exact": decimal_text(combined),
    }


def oracle_best_for_predicted_count(
    predicted_forks: int,
    *,
    gt_divisions: int = FROZEN_CV57_GT_DIVISIONS,
    adjusted_edge_score: Decimal = FROZEN_CV57_BASELINE,
) -> dict[str, Any]:
    """Best possible edge-neutral score at an exact predicted-fork count."""

    if isinstance(predicted_forks, bool) or predicted_forks < 0:
        raise PermissiveSweepError("predicted-fork count must be a nonnegative integer")
    tp = min(predicted_forks, gt_divisions)
    return edge_neutral_case(
        tp,
        predicted_forks - tp,
        gt_divisions=gt_divisions,
        adjusted_edge_score=adjusted_edge_score,
    )


def analytic_report(max_predicted_forks: int) -> tuple[dict[str, Any], bytes]:
    if max_predicted_forks < FROZEN_CV57_GT_DIVISIONS:
        raise PermissiveSweepError("analytic curve must include its 42-fork optimum")
    highlighted = {
        "tp15_fp15": edge_neutral_case(15, 15),
        "tp25_fp0": edge_neutral_case(25, 0),
        "tp25_fp5": edge_neutral_case(25, 5),
        "tp25_fp10": edge_neutral_case(25, 10),
        "tp38_fp0": edge_neutral_case(38, 0),
        "tp38_fp718": edge_neutral_case(38, 718),
        "tp38_fp719": edge_neutral_case(38, 719),
    }
    report = {
        "schema": ANALYTIC_SCHEMA,
        "status": "EXACT_EDGE_NEUTRAL_ANALYTIC_NOT_MEASURED_DECODER_EVIDENCE",
        "frozen_cv57": {
            "adjusted_edge_score_exact": decimal_text(FROZEN_CV57_BASELINE),
            "gt_divisions": FROZEN_CV57_GT_DIVISIONS,
            "formula": (
                "J=TP/(TP+FP+FN)=TP/(42+FP); "
                "S=0.8815220699597315+0.1*J; predicted_forks=TP+FP"
            ),
        },
        "oracle_best_curve": {
            "definition": "TP(k)=min(k,42), FP(k)=max(0,k-42); this is an oracle upper envelope, not a deployable ranking",
            "maximum_predicted_forks_in_csv": max_predicted_forks,
            "peak": oracle_best_for_predicted_count(FROZEN_CV57_GT_DIVISIONS),
        },
        "highlighted_points": highlighted,
        "fixed_adjusted_edge_plus_0p005": {
            "tp38_maximum_fp": 718,
            "tp38_fp718_passes": Decimal(
                highlighted["tp38_fp718"]["combined_score_exact"]
            )
            >= FROZEN_CV57_BASELINE + Decimal("0.005"),
            "tp38_fp719_passes": Decimal(
                highlighted["tp38_fp719"]["combined_score_exact"]
            )
            >= FROZEN_CV57_BASELINE + Decimal("0.005"),
            "reconciliation_note": (
                "The historical 1,072-FP number retains +0.001591651042 of "
                "GT-oracle positive-edge gain while holding only false forks edge-neutral. "
                "With the adjusted-edge term fixed at the APNCC baseline, the exact limit is 718 FP."
            ),
        },
        "interpretation": (
            "Every point in this artifact assumes zero adjusted-edge damage. "
            "Measured decoder graphs must be scored separately because fork edits can change the edge term."
        ),
    }
    lines = [
        "predicted_forks,oracle_tp,oracle_fp,oracle_fn,division_jaccard_exact,division_addition_exact,combined_score_exact"
    ]
    for count in range(max_predicted_forks + 1):
        row = oracle_best_for_predicted_count(count)
        lines.append(
            ",".join(
                str(row[key])
                for key in (
                    "predicted_forks",
                    "tp",
                    "fp",
                    "fn",
                    "division_jaccard_exact",
                    "division_addition_exact",
                    "combined_score_exact",
                )
            )
        )
    return report, ("\n".join(lines) + "\n").encode("ascii")


def default_permissive_panel() -> tuple[str, ...]:
    """A bounded panel spanning caps, late decode, persistence, and thresholds."""

    names: list[str] = []
    # Explicit high-FP operating points: low threshold, broad gap, no persistence.
    for late in (0, 1, 2):
        for cap in ("01", "02", "03", "05", "08", "all"):
            names.append(f"t035_g100_l{late}_dp0_gp0_n0_c{cap}")
    # The unrestricted count curve under stronger persistence requirements.
    for daughter, grandchild in (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    ):
        names.append(
            f"t035_g100_l0_dp{daughter}_gp{grandchild}_n0_call"
        )
    # Gap and threshold extremes prevent the permissive panel from being a
    # disguised single-setting read.
    for gap in ("005", "010", "020"):
        names.append(f"t035_g{gap}_l0_dp0_gp0_n0_call")
    for threshold in ("045", "055", "065", "075", "085", "095"):
        names.append(f"t{threshold}_g100_l0_dp0_gp0_n0_call")
    if len(names) != len(set(names)):
        raise AssertionError("default permissive panel contains a duplicate")
    return tuple(names)


def recover_bindings(
    spool_paths: Sequence[Path], config_names: Sequence[str]
) -> tuple[dict[str, dict[str, GraphBinding]], dict[str, Any]]:
    """Recover selected bindings and audit duplicate spool agreement."""

    wanted = set(config_names) | {"__EXACT_SAME_VIDEO_DIVISIONS_OFF_PARENT__"}
    all_rows: dict[tuple[str, str], GraphBinding] = {}
    conflicts: list[dict[str, Any]] = []
    per_spool: list[dict[str, Any]] = []
    for path in sorted(spool_paths):
        if not path.is_file():
            raise PermissiveSweepError(f"spool does not exist: {path}")
        with sqlite3.connect(path) as connection:
            placeholders = ",".join("?" for _ in wanted)
            query = (
                "SELECT config_index,config_name,identity,fold_id,graph_sha256,"
                "predicted_forks,duplicate_survivors FROM config_graph "
                f"WHERE config_name IN ({placeholders}) ORDER BY config_name,identity"
            )
            rows = connection.execute(query, sorted(wanted)).fetchall()
            metric_count = int(connection.execute("SELECT count(*) FROM metric").fetchone()[0])
        per_spool.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "selected_binding_rows": len(rows),
                "metric_rows": metric_count,
            }
        )
        for raw in rows:
            binding = GraphBinding(
                config_index=int(raw[0]),
                config_name=str(raw[1]),
                identity=str(raw[2]),
                fold_id=str(raw[3]),
                graph_sha256=str(raw[4]),
                predicted_forks=int(raw[5]),
                duplicate_survivors=int(raw[6]),
            )
            if not SHA256_RE.fullmatch(binding.graph_sha256):
                raise PermissiveSweepError("spool contains an invalid graph SHA-256")
            key = (binding.config_name, binding.identity)
            prior = all_rows.get(key)
            if prior is not None and prior != binding:
                conflicts.append(
                    {
                        "config_name": binding.config_name,
                        "identity": binding.identity,
                        "first": prior.__dict__,
                        "other": binding.__dict__,
                    }
                )
            else:
                all_rows[key] = binding
    if conflicts:
        raise PermissiveSweepError(f"duplicate spool bindings disagree: {conflicts[:3]}")

    available_names = {key[0] for key in all_rows}
    missing_names = sorted(wanted - available_names)
    if missing_names:
        raise PermissiveSweepError(f"requested configs are absent from the spools: {missing_names}")
    identities = sorted({key[1] for key in all_rows})
    by_identity: dict[str, dict[str, GraphBinding]] = {identity: {} for identity in identities}
    for (name, identity), binding in all_rows.items():
        by_identity[identity][name] = binding
    incomplete = {
        identity: sorted(wanted - set(records))
        for identity, records in by_identity.items()
        if set(records) != wanted
    }
    if incomplete:
        raise PermissiveSweepError(
            f"selected config panel does not cover every recovered identity: {incomplete}"
        )
    if len(identities) != 69:
        raise PermissiveSweepError(f"expected exact dev69 coverage, observed {len(identities)}")
    audit = {
        "spools": per_spool,
        "all_spool_metric_tables_empty": all(row["metric_rows"] == 0 for row in per_spool),
        "duplicate_binding_conflicts": 0,
        "selected_config_count_excluding_parent": len(config_names),
        "identity_count": len(identities),
        "population": "G03 R4 development dev69 (6bba), not CV57",
    }
    return by_identity, audit


def _validate_cache_entry(graph_path: Path, graph_sha256: str) -> dict[str, Any]:
    receipt_path = graph_path.with_name("CACHE_ENTRY.json")
    if not graph_path.is_file() or not receipt_path.is_file():
        raise PermissiveSweepError(f"graph cache entry is incomplete: {graph_path.parent}")
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if receipt.get("semantic_graph_sha256") != graph_sha256:
        raise PermissiveSweepError("cache receipt semantic hash does not match its directory")
    observed_file_hash = sha256_file(graph_path)
    if (
        receipt.get("graph_file_sha256") != observed_file_hash
        or int(receipt.get("graph_file_bytes", -1)) != graph_path.stat().st_size
    ):
        raise PermissiveSweepError("cached graph file receipt drifted")
    return receipt


def _score_identity_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Score all selected unique graphs for one identity in one process."""

    root = Path(str(payload["root"]))
    sys.path[:0] = [
        str(root / "scripts"),
        str(root / "src"),
        str(root / "external/kaggle-cell-tracking-competition/src"),
    ]
    import zarr
    from zarr.storage import ZipStore

    import scoreall
    from celltrack_alpha.kaggle_calibration import read_submission
    from celltrack_alpha.official_eval import official_metric_row
    from celltrack_baseline.rule_based import parse_scale, read_geff_arrays

    identity = str(payload["identity"])
    dataset = identity.split(":", 1)[1]
    archive = Path(str(payload["archive"]))
    cache_root = Path(str(payload["graph_cache"]))
    raw_bindings = list(payload["bindings"])
    by_sha: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for binding in raw_bindings:
        by_sha[str(binding["graph_sha256"])].append(binding)

    result_by_sha: dict[str, dict[str, Any]] = {}
    with ZipStore(archive, mode="r") as store:
        gt = read_geff_arrays(store, "train", dataset)
        group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
        scale = parse_scale(dict(group.attrs))
        for graph_sha in sorted(by_sha):
            graph_path = (
                cache_root / identity.replace(":", "__", 1) / graph_sha / "graph.csv"
            )
            _validate_cache_entry(graph_path, graph_sha)
            graphs = read_submission(graph_path)
            if set(graphs) != {dataset}:
                raise PermissiveSweepError("cached graph identity drifted")
            graph = graphs[dataset]
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            metrics = dict(official_metric_row(nodes, edges, gt, scale))
            result_by_sha[graph_sha] = {
                key: metrics[key]
                for key in (
                    "edge_tp",
                    "edge_fp",
                    "edge_fn",
                    "division_tp",
                    "division_fp",
                    "division_fn",
                    "num_pred_nodes",
                    "node_recall",
                    "total_node_ratio",
                    "edge_jaccard",
                    "adj_edge_jaccard",
                    "division_jaccard",
                )
            }
    rows = []
    for binding in raw_bindings:
        rows.append({**binding, "metrics": result_by_sha[str(binding["graph_sha256"])]})
    return {
        "identity": identity,
        "unique_graphs_scored": len(by_sha),
        "rows": rows,
    }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise PermissiveSweepError(f"nonfinite aggregate {label}")
    return result


def aggregate_measured(
    scored: Sequence[Mapping[str, Any]], config_names: Sequence[str]
) -> list[dict[str, Any]]:
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "external/kaggle-cell-tracking-competition/src"),
    ]
    from celltrack_alpha.official_eval import official_summary

    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for identity_result in scored:
        for row in identity_result["rows"]:
            by_config[str(row["config_name"])].append(row)
    ordered_names = ["__EXACT_SAME_VIDEO_DIVISIONS_OFF_PARENT__", *config_names]
    results: list[dict[str, Any]] = []
    for name in ordered_names:
        rows = by_config[name]
        if len(rows) != 69:
            raise PermissiveSweepError(f"config {name} has {len(rows)} rows, expected 69")
        metric_rows = [dict(row["metrics"]) for row in rows]
        summary = dict(official_summary(metric_rows))
        counts = {
            key: int(sum(int(row["metrics"][key]) for row in rows))
            for key in (
                "edge_tp",
                "edge_fp",
                "edge_fn",
                "division_tp",
                "division_fp",
                "division_fn",
            )
        }
        denominator = (
            counts["division_tp"] + counts["division_fp"] + counts["division_fn"]
        )
        division_jaccard = counts["division_tp"] / denominator if denominator else 0.0
        adjusted = _finite(summary["adj_edge_jaccard"], "adjusted edge Jaccard")
        combined = adjusted + 0.1 * division_jaccard
        if not math.isclose(combined, _finite(summary["score"], "combined score"), abs_tol=1e-12):
            raise PermissiveSweepError("official combined score formula drifted")
        results.append(
            {
                "config_name": name,
                "config_index": int(rows[0]["config_index"]),
                "population": "dev69_6bba",
                "video_count": 69,
                "predicted_forks": int(sum(int(row["predicted_forks"]) for row in rows)),
                "duplicate_survivors": int(
                    sum(int(row["duplicate_survivors"]) for row in rows)
                ),
                "tp": counts["division_tp"],
                "fp": counts["division_fp"],
                "fn": counts["division_fn"],
                "division_jaccard": division_jaccard,
                "adjusted_edge_jaccard": adjusted,
                "combined_score": combined,
                "raw_edge_jaccard": _finite(summary["edge_jaccard"], "edge Jaccard"),
                "node_recall": _finite(summary["node_recall"], "node recall"),
                "edge_tp": counts["edge_tp"],
                "edge_fp": counts["edge_fp"],
                "edge_fn": counts["edge_fn"],
            }
        )
    parent = results[0]
    for row in results:
        row["delta_combined_vs_dev69_parent"] = (
            row["combined_score"] - parent["combined_score"]
        )
        row["delta_adjusted_edge_vs_dev69_parent"] = (
            row["adjusted_edge_jaccard"] - parent["adjusted_edge_jaccard"]
        )
    return results


def nondominated_frontier(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pareto frontier: TP/J/S maximize while FP minimizes."""

    result: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            weak = (
                int(other["tp"]) >= int(candidate["tp"])
                and int(other["fp"]) <= int(candidate["fp"])
                and float(other["division_jaccard"])
                >= float(candidate["division_jaccard"])
                and float(other["combined_score"]) >= float(candidate["combined_score"])
            )
            strict = (
                int(other["tp"]) > int(candidate["tp"])
                or int(other["fp"]) < int(candidate["fp"])
                or float(other["division_jaccard"])
                > float(candidate["division_jaccard"])
                or float(other["combined_score"]) > float(candidate["combined_score"])
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            result.append(dict(candidate))
    return sorted(
        result,
        key=lambda row: (
            int(row["predicted_forks"]),
            int(row["fp"]),
            -int(row["tp"]),
            -float(row["combined_score"]),
            str(row["config_name"]),
        ),
    )


def measured_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "config_name",
        "predicted_forks",
        "tp",
        "fp",
        "fn",
        "division_jaccard",
        "adjusted_edge_jaccard",
        "combined_score",
        "delta_adjusted_edge_vs_dev69_parent",
        "delta_combined_vs_dev69_parent",
    )
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    return ("\n".join(lines) + "\n").encode("ascii")


def cv57_cache_panel() -> tuple[dict[str, Any], ...]:
    """Fixed label-blind policies over the bounded appearance cache.

    The caps count successfully compatible, node-disjoint actions.  The panel
    deliberately includes operating points far beyond the historical cap-one
    high-precision corner.
    """

    policies: list[dict[str, Any]] = []
    families = (
        ("original_mass", "mass", 0.35, 0.6, 1.6, 0.4),
        ("relaxed_mass", "mass", 0.20, 0.4, 2.5, 0.2),
        ("unfiltered_mass", "mass", 0.0, 0.0, None, 0.0),
        ("unfiltered_geometry", "geometry", 0.0, 0.0, None, 0.0),
    )
    for prefix, ranking, balance_min, ratio_min, ratio_max, roundness_min in families:
        for cap in (1, 2, 3, 5, 8):
            policies.append(
                {
                    "name": f"{prefix}_cap{cap}",
                    "ranking": ranking,
                    "balance_min": balance_min,
                    "ratio_min": ratio_min,
                    "ratio_max": ratio_max,
                    "roundness_min": roundness_min,
                    "cap": cap,
                }
            )
    return tuple(policies)


def appearance_features(candidate: Mapping[str, Any]) -> dict[str, float]:
    summaries = candidate["intensity_summaries"]
    mass_1 = float(summaries["daughter_1"]["mass_above_background"])
    mass_2 = float(summaries["daughter_2"]["mass_above_background"])
    parent_mass = float(summaries["parent"]["mass_above_background"])
    larger = max(mass_1, mass_2)
    balance = min(mass_1, mass_2) / larger if larger > 0.0 else 0.0
    ratio = (mass_1 + mass_2) / parent_mass if parent_mass > 0.0 else float("inf")
    mass_error = abs(math.log(ratio)) if ratio > 0.0 and math.isfinite(ratio) else float("inf")
    result = {
        "daughter_mass_balance": balance,
        "daughter_parent_mass_ratio": ratio,
        "parent_xy_roundness": float(summaries["parent"]["xy_roundness"]),
        "absolute_log_mass_error": mass_error,
        "geometry_cost": float(candidate["features"]["geometry_cost"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise PermissiveSweepError("appearance cache contains a nonfinite derived feature")
    return result


def _policy_ordered_candidates(
    candidates: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = dict(raw)
        features = appearance_features(candidate)
        if not (
            features["daughter_mass_balance"] >= float(policy["balance_min"])
            and features["daughter_parent_mass_ratio"] >= float(policy["ratio_min"])
            and (
                policy["ratio_max"] is None
                or features["daughter_parent_mass_ratio"] <= float(policy["ratio_max"])
            )
            and features["parent_xy_roundness"] >= float(policy["roundness_min"])
        ):
            continue
        candidate["g04_appearance_features"] = features
        eligible.append(candidate)
    if policy["ranking"] == "mass":
        key = lambda row: (
            row["g04_appearance_features"]["absolute_log_mass_error"],
            1.0 - row["g04_appearance_features"]["daughter_mass_balance"],
            row["g04_appearance_features"]["geometry_cost"],
            str(row["candidate_key"]),
        )
    elif policy["ranking"] == "geometry":
        key = lambda row: (
            row["g04_appearance_features"]["geometry_cost"],
            row["g04_appearance_features"]["absolute_log_mass_error"],
            str(row["candidate_key"]),
        )
    else:
        raise PermissiveSweepError(f"unsupported CV57 cache ranking: {policy['ranking']}")
    return sorted(eligible, key=key)


def _build_disjoint_fork_graph(
    baseline: Any,
    candidates: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply exact N1-equivalent primitives once after greedy disjoint selection."""

    from celltrack_alpha.kaggle_calibration import SubmissionEdge, SubmissionGraph

    baseline_nodes = set(int(value) for value in baseline.nodes)
    baseline_edges = {
        (int(edge.source_id), int(edge.target_id)) for edge in baseline.edges
    }
    occupied_nodes: set[int] = set()
    owned_conflicts: set[tuple[int, int]] = set()
    selected: list[dict[str, Any]] = []
    incompatible = 0
    overlap_suppressed = 0
    for candidate in _policy_ordered_candidates(candidates, policy):
        action = candidate["action"]
        add_edges = {tuple(int(value) for value in pair) for pair in action["add_edges"]}
        parents = {int(value) for value in action["replace_outgoing_from"]}
        daughters = {int(value) for value in action["replace_incoming_to"]}
        boundary = parents | daughters
        if len(add_edges) != 2 or len(parents) != 1 or len(daughters) != 2:
            incompatible += 1
            continue
        if not boundary <= baseline_nodes:
            incompatible += 1
            continue
        if any(
            int(baseline.nodes[source].t) >= int(baseline.nodes[target].t)
            for source, target in add_edges
        ):
            # The bounded cache was generated against an older parent graph.
            # A cached node tuple that is not a forward edge on APNCC is not a
            # compatible action and must not poison the whole permissive panel.
            incompatible += 1
            continue
        if boundary & occupied_nodes:
            overlap_suppressed += 1
            continue
        actual_conflicts = {
            pair for pair in baseline_edges if pair[0] in parents or pair[1] in daughters
        }
        if actual_conflicts & owned_conflicts:
            overlap_suppressed += 1
            continue
        selected.append(
            {
                "candidate_key": str(candidate["candidate_key"]),
                "add_edges": add_edges,
                "actual_conflicts": actual_conflicts,
                "boundary": boundary,
            }
        )
        occupied_nodes.update(boundary)
        owned_conflicts.update(actual_conflicts)
        if len(selected) >= int(policy["cap"]):
            break

    add_union = {pair for row in selected for pair in row["add_edges"]}
    remove_union = {pair for row in selected for pair in row["actual_conflicts"]}
    final_edges = (baseline_edges - remove_union) | add_union
    graph = SubmissionGraph(
        dataset=baseline.dataset,
        nodes=dict(baseline.nodes),
        edges=[SubmissionEdge(source, target) for source, target in sorted(final_edges)],
    )
    outdegree: dict[int, int] = defaultdict(int)
    for source, _target in final_edges:
        outdegree[source] += 1
    return graph, {
        "selected_action_count": len(selected),
        "selected_candidate_keys": [row["candidate_key"] for row in selected],
        "incompatible_candidates_seen": incompatible,
        "overlap_suppressed": overlap_suppressed,
        "removed_edges": len(remove_union),
        "added_edges": len(add_union),
        "predicted_forks": sum(value == 2 for value in outdegree.values()),
    }


def _score_cv57_cache_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(payload["root"]))
    sys.path[:0] = [
        str(root / "scripts"),
        str(root / "src"),
        str(root / "external/kaggle-cell-tracking-competition/src"),
    ]
    import zarr
    from zarr.storage import ZipStore
    import scoreall
    from celltrack_alpha.kaggle_calibration import read_submission, validate_graphs
    from celltrack_alpha.official_eval import official_metric_row
    from celltrack_baseline.rule_based import parse_scale, read_geff_arrays

    fragment = Path(str(payload["fragment"]))
    cache_path = Path(str(payload["cache_path"]))
    parsed = read_submission(fragment)
    if len(parsed) != 1:
        raise PermissiveSweepError("CV57 fragment contains multiple graphs")
    baseline = next(iter(parsed.values()))
    identity = f"train:{baseline.dataset}"
    if identity != payload["identity"]:
        raise PermissiveSweepError("CV57 fragment identity drifted")
    cache_payload = json.loads(cache_path.read_text(encoding="ascii"))
    if cache_payload.get("contains_labels") is not False or cache_payload.get("contains_voxels") is not False:
        raise PermissiveSweepError("appearance cache firewall marker drifted")
    candidates = list(cache_payload.get("candidates", ()))

    graphs: dict[str, Any] = {"__APNCC_DIVISIONS_OFF_PARENT__": baseline}
    diagnostics: dict[str, dict[str, Any]] = {
        "__APNCC_DIVISIONS_OFF_PARENT__": {
            "selected_action_count": 0,
            "selected_candidate_keys": [],
            "incompatible_candidates_seen": 0,
            "overlap_suppressed": 0,
            "removed_edges": 0,
            "added_edges": 0,
            "predicted_forks": 0,
        }
    }
    for policy in payload["policies"]:
        graph, diagnostic = _build_disjoint_fork_graph(baseline, candidates, policy)
        graphs[str(policy["name"])] = graph
        diagnostics[str(policy["name"])] = diagnostic

    by_selection: dict[tuple[str, ...], list[str]] = defaultdict(list)
    graph_by_selection: dict[tuple[str, ...], Any] = {}
    for name, graph in graphs.items():
        selection = tuple(sorted(diagnostics[name]["selected_candidate_keys"]))
        by_selection[selection].append(name)
        graph_by_selection[selection] = graph
    metrics_by_selection: dict[tuple[str, ...], dict[str, Any]] = {}
    edge_hash_by_selection: dict[tuple[str, ...], str] = {}
    archive = Path(str(payload["archive"]))
    with ZipStore(archive, mode="r") as store:
        gt = read_geff_arrays(store, "train", baseline.dataset)
        group = zarr.open_group(store=store, path=f"train/{baseline.dataset}.zarr", mode="r")
        scale = parse_scale(dict(group.attrs))
        for selection, graph in graph_by_selection.items():
            # Nodes are unchanged and boundary ownership was checked above.
            # Validate each unique selected-action set once, not once per
            # policy plus again during full canonical graph hashing.
            validate_graphs({graph.dataset: graph})
            edge_digest = hashlib.sha256(graph.dataset.encode("ascii") + b"\0")
            for edge in graph.edges:
                edge_digest.update(f"{int(edge.source_id)},{int(edge.target_id)};".encode("ascii"))
            edge_hash_by_selection[selection] = edge_digest.hexdigest()
            nodes, edges = scoreall._prediction_rows(baseline.dataset, graph)
            metric = dict(official_metric_row(nodes, edges, gt, scale))
            metrics_by_selection[selection] = {
                key: metric[key]
                for key in (
                    "edge_tp",
                    "edge_fp",
                    "edge_fn",
                    "division_tp",
                    "division_fp",
                    "division_fn",
                    "num_pred_nodes",
                    "node_recall",
                    "edge_jaccard",
                    "adj_edge_jaccard",
                )
            }
    rows: list[dict[str, Any]] = []
    for selection, names in by_selection.items():
        for name in names:
            rows.append(
                {
                    "config_name": name,
                    "identity": identity,
                    "edge_set_sha256": edge_hash_by_selection[selection],
                    "metrics": metrics_by_selection[selection],
                    "diagnostic": diagnostics[name],
                }
            )
    return {
        "identity": identity,
        "fragment_sha256": sha256_file(fragment),
        "cache_sha256": sha256_file(cache_path),
        "cached_candidates": len(candidates),
        "unique_graphs_scored": len(graph_by_selection),
        "panel_sha256": hashlib.sha256(canonical_bytes(payload["policies"])).hexdigest(),
        "rows": rows,
    }


def _aggregate_cv57_cache(
    scored: Sequence[Mapping[str, Any]], policies: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "external/kaggle-cell-tracking-competition/src"),
    ]
    from celltrack_alpha.official_eval import official_summary

    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for identity_result in scored:
        for row in identity_result["rows"]:
            by_config[str(row["config_name"])].append(row)
    names = ["__APNCC_DIVISIONS_OFF_PARENT__", *(str(row["name"]) for row in policies)]
    results: list[dict[str, Any]] = []
    for name in names:
        rows = by_config[name]
        if len(rows) != 57:
            raise PermissiveSweepError(f"CV57 config {name} has {len(rows)} rows, expected 57")
        metric_rows = []
        for row in rows:
            metric = dict(row["metrics"])
            # Checkpoints written by the first compact worker revision omitted
            # this count.  The official summariser requires the schema key but
            # does not use its total in any returned score component.
            metric.setdefault("num_pred_nodes", 0)
            metric_rows.append(metric)
        summary = dict(official_summary(metric_rows))
        counts = {
            key: int(sum(int(row["metrics"][key]) for row in rows))
            for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
        }
        denominator = counts["division_tp"] + counts["division_fp"] + counts["division_fn"]
        division_jaccard = counts["division_tp"] / denominator if denominator else 0.0
        adjusted = _finite(summary["adj_edge_jaccard"], "CV57 adjusted edge Jaccard")
        combined = adjusted + 0.1 * division_jaccard
        if not math.isclose(combined, _finite(summary["score"], "CV57 score"), abs_tol=1e-12):
            raise PermissiveSweepError("CV57 official combined score formula drifted")
        results.append(
            {
                "config_name": name,
                "population": "cv57_apncc",
                "video_count": 57,
                "selected_actions": sum(int(row["diagnostic"]["selected_action_count"]) for row in rows),
                "predicted_forks": sum(int(row["diagnostic"]["predicted_forks"]) for row in rows),
                "incompatible_candidates_seen": sum(int(row["diagnostic"]["incompatible_candidates_seen"]) for row in rows),
                "overlap_suppressed": sum(int(row["diagnostic"]["overlap_suppressed"]) for row in rows),
                "tp": counts["division_tp"],
                "fp": counts["division_fp"],
                "fn": counts["division_fn"],
                "division_jaccard": division_jaccard,
                "adjusted_edge_jaccard": adjusted,
                "combined_score": combined,
                "raw_edge_jaccard": _finite(summary["edge_jaccard"], "CV57 edge Jaccard"),
                "node_recall": _finite(summary["node_recall"], "CV57 node recall"),
                "edge_tp": counts["edge_tp"],
                "edge_fp": counts["edge_fp"],
                "edge_fn": counts["edge_fn"],
            }
        )
    baseline = results[0]
    for row in results:
        row["delta_adjusted_edge_vs_apncc"] = row["adjusted_edge_jaccard"] - baseline["adjusted_edge_jaccard"]
        row["delta_combined_vs_apncc"] = row["combined_score"] - baseline["combined_score"]
        row["beats_0p881522069960"] = row["combined_score"] > float(FROZEN_CV57_BASELINE)
    return results


def cv57_measured_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "config_name",
        "selected_actions",
        "predicted_forks",
        "tp",
        "fp",
        "fn",
        "division_jaccard",
        "adjusted_edge_jaccard",
        "combined_score",
        "delta_adjusted_edge_vs_apncc",
        "delta_combined_vs_apncc",
        "beats_0p881522069960",
    )
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    return ("\n".join(lines) + "\n").encode("ascii")


def run_cv57_cache_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = cv57_cache_panel()
    panel_sha256 = hashlib.sha256(canonical_bytes(policies)).hexdigest()
    fragment_paths = sorted(args.cv57_fragments.resolve().glob("train__*.csv"))
    cache_paths = sorted(args.appearance_cache.resolve().glob("*.json"))
    if len(fragment_paths) != 57 or len(cache_paths) != 57:
        raise PermissiveSweepError(
            f"expected exact 57 CV57 fragments/caches, observed {len(fragment_paths)}/{len(cache_paths)}"
        )
    cache_by_dataset = {path.stem: path for path in cache_paths}
    payloads = []
    for fragment in fragment_paths:
        dataset = fragment.stem.removeprefix("train__")
        if dataset not in cache_by_dataset:
            raise PermissiveSweepError(f"appearance cache missing CV57 dataset: {dataset}")
        payloads.append(
            {
                "root": str(ROOT),
                "identity": f"train:{dataset}",
                "fragment": str(fragment),
                "cache_path": str(cache_by_dataset[dataset]),
                "archive": str(args.archive.resolve()),
                "policies": policies,
            }
        )
    checkpoint_dir = output / "cv57_cache_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []
    pending_payloads = []
    for payload in payloads:
        checkpoint = checkpoint_dir / f"{str(payload['identity']).replace(':', '__', 1)}.json"
        if checkpoint.is_file():
            recovered = json.loads(checkpoint.read_text(encoding="ascii"))
            if (
                recovered.get("identity") == payload["identity"]
                and recovered.get("panel_sha256") == panel_sha256
                and recovered.get("fragment_sha256") == sha256_file(Path(str(payload["fragment"])))
                and recovered.get("cache_sha256") == sha256_file(Path(str(payload["cache_path"])))
            ):
                scored.append(recovered)
                continue
        pending_payloads.append(payload)
    if scored:
        print(f"recovered {len(scored)}/57 authenticated CV57 checkpoints", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_score_cv57_cache_identity, payload): payload["identity"]
            for payload in pending_payloads
        }
        completed = len(scored)
        for future in as_completed(futures):
            identity = futures[future]
            try:
                result = future.result()
                if result["panel_sha256"] != panel_sha256:
                    raise PermissiveSweepError("CV57 cache worker panel identity drifted")
                scored.append(result)
                checkpoint = checkpoint_dir / f"{identity.replace(':', '__', 1)}.json"
                atomic_write(checkpoint, canonical_bytes(result))
            except Exception as exc:
                raise PermissiveSweepError(f"CV57 cache scoring failed for {identity}: {exc}") from exc
            completed += 1
            if completed % 5 == 0 or completed == len(payloads):
                print(f"independently scored {completed}/{len(payloads)} CV57 videos", flush=True)
    scored.sort(key=lambda row: str(row["identity"]))
    rows = _aggregate_cv57_cache(scored, policies)
    frontier = nondominated_frontier(rows)
    best = max(rows, key=lambda row: (float(row["combined_score"]), str(row["config_name"])))
    baseline = rows[0]
    if not math.isclose(float(baseline["combined_score"]), 0.8815220699597315, abs_tol=1e-12):
        raise PermissiveSweepError(
            f"APNCC CV57 baseline failed exact reproduction: {baseline['combined_score']}"
        )
    report = {
        "schema": CV57_CACHE_SCHEMA,
        "status": "COMPLETE_DIRECTLY_COMPARABLE_CV57_BOUNDED_CACHE_SWEEP",
        "scope": {
            "population": "the exact 57-video APNCC local validation population",
            "cache": "COMPUTED label-blind raw-intensity patch summaries; never H3 masks",
            "boundedness": "at most eight geometry-prefiltered candidates per video; not the full candidate universe",
            "policy_count_excluding_parent": len(policies),
            "policies": list(policies),
        },
        "execution": {
            "device": "cpu",
            "workers": args.workers,
            "unique_graphs_independently_scored": sum(int(row["unique_graphs_scored"]) for row in scored),
            "cached_candidate_count": sum(int(row["cached_candidates"]) for row in scored),
            "archive": str(args.archive.resolve()),
            "fragment_directory": str(args.cv57_fragments.resolve()),
            "appearance_cache_directory": str(args.appearance_cache.resolve()),
        },
        "baseline_reproduction": baseline,
        "best_measured": best,
        "submission_candidate": bool(best["combined_score"] > float(FROZEN_CV57_BASELINE)),
        "all_rows": rows,
        "nondominated_frontier": frontier,
        "input_receipts": [
            {
                "identity": row["identity"],
                "fragment_sha256": row["fragment_sha256"],
                "cache_sha256": row["cache_sha256"],
                "cached_candidates": row["cached_candidates"],
            }
            for row in scored
        ],
    }
    report_path = output / "MEASURED_CV57_CACHE_FRONTIER.json"
    rows_path = output / "MEASURED_CV57_CACHE_ALL_ROWS.csv"
    frontier_path = output / "MEASURED_CV57_CACHE_NONDOMINATED.csv"
    atomic_write(report_path, canonical_bytes(report))
    atomic_write(rows_path, cv57_measured_csv(rows))
    atomic_write(frontier_path, cv57_measured_csv(frontier))
    return {
        "status": report["status"],
        "report": str(report_path),
        "all_rows_csv": str(rows_path),
        "frontier_csv": str(frontier_path),
        "best": best,
        "submission_candidate": report["submission_candidate"],
    }


def _load_frozen_predict_arm() -> Any:
    if sha256_file(FROZEN_PREDICT_ARM) != EXPECTED_FROZEN_PREDICT_ARM_SHA256:
        raise PermissiveSweepError("frozen appearance-prefilter implementation drifted")
    specification = importlib.util.spec_from_file_location(
        "_g04_frozen_predict_arm", FROZEN_PREDICT_ARM
    )
    if specification is None or specification.loader is None:
        raise PermissiveSweepError("could not load frozen appearance-prefilter implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _graph_mapping_for_prefilter(graph: Any) -> dict[str, Any]:
    family = str(graph.dataset).split("_", 1)[0]
    if family not in {"44b6", "6bba"}:
        raise PermissiveSweepError(f"unexpected public-test family: {graph.dataset}")
    return {
        "dataset": str(graph.dataset),
        "family": family,
        "identity": f"test:{graph.dataset}",
        "nodes": {
            int(node_id): {
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": float(node.z),
                "y": float(node.y),
                "x": float(node.x),
            }
            for node_id, node in graph.nodes.items()
        },
        "edges": {
            (int(edge.source_id), int(edge.target_id)) for edge in graph.edges
        },
    }


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _line in handle) - 1)


def run_promoted_test_package(args: argparse.Namespace) -> dict[str, Any]:
    """Apply the frozen geometry-cap-five policy to the scored public parent."""

    sys.path[:0] = [str(ROOT / "src")]
    import zarr
    from zarr.storage import ZipStore
    from celltrack_alpha.action_compiler import graph_sha256
    from celltrack_alpha.kaggle_calibration import (
        read_submission,
        validate_graphs,
        write_submission,
    )

    output = args.promoted_dir.resolve()
    parent_path = args.public_parent.resolve()
    candidate_path = output / "submission.csv"
    package_path = output / "PROMOTED_TEST_PACKAGE.json"
    predicate_path = output / "ELIGIBLE_SUBMISSION_PREDICATE.json"
    readme_path = output / "README.md"
    for path in (candidate_path, package_path, predicate_path, readme_path):
        if path.exists():
            raise PermissiveSweepError(f"refusing to overwrite promoted artifact: {path}")
    if not parent_path.is_file():
        raise PermissiveSweepError(f"authenticated public parent is absent: {parent_path}")
    parent_sha = sha256_file(parent_path)
    parent_rows = _count_csv_rows(parent_path)
    if parent_sha != EXPECTED_PUBLIC_PARENT_SHA256 or parent_rows != EXPECTED_PUBLIC_PARENT_ROWS:
        raise PermissiveSweepError("public-0.881 APNCC parent identity drifted")
    if args.archive.stat().st_size != EXPECTED_RAW_ARCHIVE_BYTES:
        raise PermissiveSweepError("raw archive byte identity drifted")

    local_report_path = output.parent / "MEASURED_CV57_CACHE_FRONTIER.json"
    local_report = json.loads(local_report_path.read_text(encoding="ascii"))
    local_best = dict(local_report["best_measured"])
    if (
        local_best.get("config_name") != "unfiltered_geometry_cap5"
        or not float(local_best["combined_score"]) > float(FROZEN_CV57_BASELINE)
        or int(local_best["tp"]) != 0
    ):
        raise PermissiveSweepError("promoted local result identity/gate drifted")

    graphs = read_submission(parent_path)
    expected_datasets = {
        "44b6_0113de3b",
        "44b6_0b24845f",
        "6bba_05b6850b",
        "6bba_05db0fb1",
    }
    if set(graphs) != expected_datasets:
        raise PermissiveSweepError(f"public-test inventory drifted: {sorted(graphs)}")
    frozen = _load_frozen_predict_arm()
    policy = next(
        row for row in cv57_cache_panel() if row["name"] == "unfiltered_geometry_cap5"
    )
    cache_dir = output / "test_appearance_cache"
    fragment_dir = output / "fragments"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir.mkdir(parents=True, exist_ok=True)
    candidate_graphs: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    with ZipStore(args.archive.resolve(), mode="r") as store:
        for dataset, parent_graph in sorted(graphs.items()):
            mapping = _graph_mapping_for_prefilter(parent_graph)
            candidates, prefilter = frozen.appearance_prefilter(mapping)
            raw = zarr.open_group(
                store=store, path=f"test/{dataset}.zarr", mode="r"
            )["0"]
            enriched: list[dict[str, Any]] = []
            for candidate in candidates:
                record = dict(candidate)
                record["intensity_summaries"] = {
                    role: frozen.patch_summary(raw, mapping["nodes"][int(node_id)])
                    for role, node_id in candidate["patch_nodes"].items()
                }
                enriched.append(record)
            cache_payload = {
                "schema": "celltrack-alpha.g04-computed-test-appearance-cache.v1",
                "dataset": dataset,
                "identity": f"test:{dataset}",
                "source_parent_sha256": parent_sha,
                "source_prefilter_sha256": EXPECTED_FROZEN_PREDICT_ARM_SHA256,
                "candidate_cap": 8,
                "contains_labels": False,
                "contains_voxels": False,
                "provenance": "COMPUTED raw-intensity patch summaries; not H3 authentic masks",
                "prefilter": prefilter,
                "candidates": enriched,
            }
            cache_path = cache_dir / f"{dataset}.json"
            atomic_write(cache_path, canonical_bytes(cache_payload))
            candidate_graph, diagnostic = _build_disjoint_fork_graph(
                parent_graph, enriched, policy
            )
            validate_graphs({dataset: candidate_graph})
            candidate_graphs[dataset] = candidate_graph
            fragment_path = fragment_dir / f"test__{dataset}.csv"
            write_submission({dataset: candidate_graph}, fragment_path)
            reread_fragment = read_submission(fragment_path)
            if graph_sha256(reread_fragment[dataset]) != graph_sha256(candidate_graph):
                raise PermissiveSweepError("written promoted fragment failed semantic round trip")
            diagnostics.append(
                {
                    "identity": f"test:{dataset}",
                    "parent_graph_sha256": graph_sha256(parent_graph),
                    "candidate_graph_sha256": graph_sha256(candidate_graph),
                    "fragment": str(fragment_path),
                    "fragment_sha256": sha256_file(fragment_path),
                    "fragment_bytes": fragment_path.stat().st_size,
                    "cache": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "prefilter": prefilter,
                    **diagnostic,
                }
            )

    validate_graphs(candidate_graphs)
    write_submission(candidate_graphs, candidate_path)
    reread = read_submission(candidate_path)
    if set(reread) != expected_datasets:
        raise PermissiveSweepError("promoted submission round-trip inventory drifted")
    for dataset in expected_datasets:
        if graph_sha256(reread[dataset]) != graph_sha256(candidate_graphs[dataset]):
            raise PermissiveSweepError(f"promoted submission round-trip drifted: {dataset}")
    candidate_sha = sha256_file(candidate_path)
    candidate_rows = _count_csv_rows(candidate_path)
    if candidate_sha == parent_sha:
        raise PermissiveSweepError("promoted test artifact is observation-identical to parent")
    selected_actions = sum(int(row["selected_action_count"]) for row in diagnostics)
    if selected_actions <= 0:
        raise PermissiveSweepError("promoted policy abstained on all public-test videos")

    package = {
        "schema": PROMOTED_TEST_SCHEMA,
        "status": "COMPLETE_OBSERVATION_DISTINCT_TEST_ARTIFACT",
        "policy": policy,
        "local_promotion": {
            "report": str(local_report_path),
            "report_sha256": sha256_file(local_report_path),
            "baseline_score": float(FROZEN_CV57_BASELINE),
            "candidate_score": float(local_best["combined_score"]),
            "delta": float(local_best["delta_combined_vs_apncc"]),
            "tp": int(local_best["tp"]),
            "fp": int(local_best["fp"]),
            "fn": int(local_best["fn"]),
            "division_jaccard": float(local_best["division_jaccard"]),
            "transfer_caveat": (
                "Local gain is edge-only and very small; division TP remained zero, "
                "so public transfer is an empirical calibration probe, not a division-credit claim."
            ),
        },
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha,
            "bytes": parent_path.stat().st_size,
            "rows": parent_rows,
            "public_score": 0.881,
            "source_archive": str((ROOT / "results-bhap.zip").resolve()),
            "source_archive_sha256": sha256_file(ROOT / "results-bhap.zip"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
            "bytes": candidate_path.stat().st_size,
            "rows": candidate_rows,
            "dataset_count": len(candidate_graphs),
            "datasets": sorted(candidate_graphs),
            "selected_actions": selected_actions,
            "predicted_forks": sum(int(row["predicted_forks"]) for row in diagnostics),
            "schema_validated": True,
            "semantic_round_trip_validated": True,
            "observation_distinct_from_parent": True,
        },
        "implementation": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_appearance_prefilter": str(FROZEN_PREDICT_ARM),
            "frozen_appearance_prefilter_sha256": EXPECTED_FROZEN_PREDICT_ARM_SHA256,
            "raw_archive": str(args.archive.resolve()),
            "raw_archive_bytes": args.archive.stat().st_size,
            "raw_archive_sha256": EXPECTED_RAW_ARCHIVE_SHA256,
            "raw_archive_hash_source": "banked immutable pin; byte size reverified",
            "labels_read": False,
            "computed_segmentations_used": False,
            "computed_raw_intensity_summaries_used": True,
        },
        "per_video": diagnostics,
    }
    atomic_write(package_path, canonical_bytes(package))
    predicate = {
        "schema": "celltrack-alpha.g04-eligible-submission-predicate.v1",
        "status": "ELIGIBLE_FOR_ROOT_SERIAL_SUBMISSION",
        "eligible": True,
        "candidate_csv": str(candidate_path),
        "candidate_csv_sha256": candidate_sha,
        "candidate_csv_bytes": candidate_path.stat().st_size,
        "candidate_csv_rows": candidate_rows,
        "package": str(package_path),
        "package_sha256": sha256_file(package_path),
        "predicates": {
            "local_score_gt_0p8815220699597315": True,
            "exact_four_public_test_datasets": True,
            "submission_schema_and_graph_invariants_valid": True,
            "semantic_write_read_round_trip_valid": True,
            "observation_distinct_from_public_parent": True,
            "selected_action_count_gt_zero": True,
            "label_blind_test_construction": True,
            "owner_any_local_improvement_policy_satisfied": True,
            "root_serial_lock_still_required": True,
        },
        "transfer_caveat": package["local_promotion"]["transfer_caveat"],
        "kaggle_action_performed_here": False,
    }
    atomic_write(predicate_path, canonical_bytes(predicate))
    readme = f"""# G-04 Lane A — permissive division frontier\n\nThe exact APNCC CV57 baseline reproduced at `{FROZEN_CV57_BASELINE}`. The bounded 20-policy sweep found **no division TP**. Its nondominated measured frontier was:\n\n| Policy | Applied forks | TP | FP | FN | Division J | Combined S | Delta |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| parent | 0 | 0 | 0 | 42 | 0 | 0.8815220699597315 | 0 |\n| unfiltered geometry cap 1 | 54 | 0 | 2 | 42 | 0 | 0.8815741350411858 | +0.0000520650814543 |\n| unfiltered geometry cap 3 | 155 | 0 | 8 | 42 | 0 | 0.8816069249327657 | +0.0000848549730342 |\n| unfiltered geometry cap 5 | 239 | 0 | 11 | 42 | 0 | 0.8816354767369976 | +0.0001134067772661 |\n\n`unfiltered_geometry_cap5` is promoted only because the owner explicitly authorized any local improvement above the exact baseline. The local gain is edge-only: it does **not** close the zero-division-credit gap, and it is below the established fine-transfer resolution. The public submission is therefore a useful observation-distinct transfer probe, not evidence that division decoding succeeded.\n\nExact analytic artifacts quantify the potential upside separately. For example, at fixed adjusted-edge score, 15 TP / 15 FP gives `J=15/57` and `S=0.9078378594334157105`.\n\nPromoted test CSV SHA-256: `{candidate_sha}`.\n"""
    atomic_write(readme_path, readme.encode("utf-8"))
    return {
        "status": package["status"],
        "candidate_csv": str(candidate_path),
        "candidate_csv_sha256": candidate_sha,
        "selected_actions": selected_actions,
        "package": str(package_path),
        "predicate": str(predicate_path),
        "readme": str(readme_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    resumable_analytic = {
        "ANALYTIC_EDGE_NEUTRAL.json",
        "ANALYTIC_ORACLE_BEST_BY_PREDICTED_COUNT.csv",
    }
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        if not existing <= resumable_analytic:
            raise PermissiveSweepError(
                f"refusing to overwrite completed/non-analytic output directory: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)

    panel = tuple(args.config_name) if args.config_name else default_permissive_panel()
    analytic, curve_csv = analytic_report(args.max_curve_count)
    analytic_path = output / "ANALYTIC_EDGE_NEUTRAL.json"
    curve_path = output / "ANALYTIC_ORACLE_BEST_BY_PREDICTED_COUNT.csv"
    atomic_write(analytic_path, canonical_bytes(analytic))
    atomic_write(curve_path, curve_csv)
    if args.analytic_only:
        return {
            "status": "ANALYTIC_ONLY_COMPLETE",
            "analytic": str(analytic_path),
            "curve": str(curve_path),
        }

    spool_paths = tuple(sorted(args.spool or DEFAULT_SPOOL_ROOT.glob(".*/*.sqlite3")))
    if len(spool_paths) != 4:
        raise PermissiveSweepError(f"expected four recoverable spools, observed {len(spool_paths)}")
    bindings, spool_audit = recover_bindings(spool_paths, panel)
    payloads = []
    for identity, records in sorted(bindings.items()):
        payloads.append(
            {
                "root": str(ROOT),
                "identity": identity,
                "archive": str(args.archive.resolve()),
                "graph_cache": str(args.graph_cache.resolve()),
                "bindings": [record.__dict__ for _, record in sorted(records.items())],
            }
        )
    scored: list[dict[str, Any]] = []
    # The managed macOS sandbox denies SC_SEM_NSEMS_MAX, which makes Python's
    # ProcessPoolExecutor fail before it starts.  Independent archive handles
    # plus native Polars/tracksdata kernels make a bounded thread pool both safe
    # and materially faster than serial evaluation here.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_score_identity_worker, payload): payload["identity"] for payload in payloads}
        completed = 0
        for future in as_completed(futures):
            identity = futures[future]
            try:
                scored.append(future.result())
            except Exception as exc:
                raise PermissiveSweepError(f"scoring failed for {identity}: {exc}") from exc
            completed += 1
            if completed % 5 == 0 or completed == len(payloads):
                print(f"independently scored {completed}/{len(payloads)} dev69 videos", flush=True)
    scored.sort(key=lambda row: str(row["identity"]))
    rows = aggregate_measured(scored, panel)
    frontier = nondominated_frontier(rows)
    best = max(rows, key=lambda row: (float(row["combined_score"]), str(row["config_name"])))
    measured = {
        "schema": MEASURED_SCHEMA,
        "status": "COMPLETE_BOUNDED_PERMISSIVE_PANEL_DEV69_NOT_CV57",
        "panel": {
            "config_count_excluding_parent": len(panel),
            "configuration_names": list(panel),
            "selection": (
                "outcome-blind bounded panel spanning caps 1/2/3/5/8/all, late decode, "
                "persistence, gap, and threshold; not the full 18,145 lattice"
            ),
        },
        "recovery_audit": spool_audit,
        "execution": {
            "device": "cpu",
            "workers": args.workers,
            "unique_graphs_independently_scored": sum(
                int(row["unique_graphs_scored"]) for row in scored
            ),
            "old_spool_metric_rows_consumed": 0,
            "archive": str(args.archive.resolve()),
            "graph_cache": str(args.graph_cache.resolve()),
        },
        "baseline_comparison": {
            "cv57_apncc": decimal_text(FROZEN_CV57_BASELINE),
            "comparison_valid": False,
            "reason": (
                "Measured panel is dev69 6bba on Trackastra parent graphs; absolute scores "
                "are not commensurate with the frozen APNCC CV57 population."
            ),
            "submission_candidate": False,
        },
        "best_measured_combined_dev69": best,
        "all_rows": rows,
        "nondominated_frontier": frontier,
    }
    measured_path = output / "MEASURED_DEV69_FRONTIER.json"
    rows_path = output / "MEASURED_DEV69_ALL_ROWS.csv"
    frontier_path = output / "MEASURED_DEV69_NONDOMINATED.csv"
    atomic_write(measured_path, canonical_bytes(measured))
    atomic_write(rows_path, measured_csv(rows))
    atomic_write(frontier_path, measured_csv(frontier))
    return {
        "status": measured["status"],
        "analytic": str(analytic_path),
        "curve": str(curve_path),
        "measured": str(measured_path),
        "all_rows_csv": str(rows_path),
        "frontier_csv": str(frontier_path),
        "best": best,
        "frontier_rows": len(frontier),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--cv57-fragments", type=Path, default=DEFAULT_CV57_FRAGMENTS)
    parser.add_argument("--appearance-cache", type=Path, default=DEFAULT_APPEARANCE_CACHE)
    parser.add_argument("--promoted-dir", type=Path, default=DEFAULT_PROMOTED_DIR)
    parser.add_argument("--public-parent", type=Path, default=DEFAULT_PUBLIC_PARENT)
    parser.add_argument("--spool", type=Path, action="append", default=[])
    parser.add_argument("--config-name", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-curve-count", type=int, default=72_000)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--cv57-cache-only", action="store_true")
    parser.add_argument("--build-promoted-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > 16:
        raise PermissiveSweepError("--workers must be in [1,16]")
    if args.cv57_cache_only and args.build_promoted_test:
        raise PermissiveSweepError("choose only one execution mode")
    if args.build_promoted_test:
        result = run_promoted_test_package(args)
    elif args.cv57_cache_only:
        result = run_cv57_cache_sweep(args)
    else:
        result = run(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

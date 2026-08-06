#!/usr/bin/env python3
"""Seal and score the single fixed G-03 h005 checkpoint on held-fit36.

``seal`` is score blind: it validates all 36 immutable CUDA association banks
and compiles them with the exact frozen R1 graph compiler.  Only ``score`` may
open the official held-fit36 scorer, after the fragment seal is immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from zarr.storage import ZipStore

from scripts import g03_r1_zero_shot as r1


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = ROOT / "reports/g03/screening/wave_r/R1_ZERO_SHOT_CONTRACT.json"
BASE_CONTRACT_SHA256 = "aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204"
H000_SCORE = ROOT / "reports/g03/screening/phase2_zero_init_v1/h000/CANONICAL_SCORE.json"
H000_SCORE_SHA256 = "d7189066b4122ff11e8a82fef616e292a9053f947c0a2e7dce9b3923ef7fc1c0"
H000 = 0.7630876414338513
HELD_FIT36_APNCC = 0.8665345260361912
PROJECTION_ABS_TOLERANCE = 1.0e-12
SEAL_SCHEMA = "celltrack-alpha.g03-h005-fixed-fragment-seal.v1"
SCORE_SCHEMA = "celltrack-alpha.g03-h005-fixed-held-fit36-score.v1"


class FixedScoreError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedScoreError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FixedScoreError(f"{label} is not an object")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise FixedScoreError(f"{label} SHA-256 mismatch: {path}")


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def aggregate(records: list[dict[str, Any]], field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda value: value["identity"]):
        digest.update(row["identity"].encode("ascii") + b"\0")
        digest.update(row[field].encode("ascii") + b"\n")
    return digest.hexdigest()


def fixed_gate(score: float, hours: float) -> dict[str, Any]:
    if not math.isfinite(score) or not 4.75 <= hours <= 5.25:
        raise FixedScoreError("fixed checkpoint score/hours are outside contract")
    delta = score - H000
    slope = delta / hours
    projected = H000 + 30.0 * slope
    passed = projected + PROJECTION_ABS_TOLERANCE >= HELD_FIT36_APNCC
    return {
        "population": "held-fit36",
        "h000_score": H000,
        "h005_score": score,
        "actual_cumulative_t4_hours": hours,
        "delta_from_h000": delta,
        "slope_adjusted_graph_score_per_t4_hour": slope,
        "linear_projection_at_30_t4_hours": projected,
        "projection_absolute_tolerance": PROJECTION_ABS_TOLERANCE,
        "reference": {
            "name": "APNCC held-fit36",
            "score": HELD_FIT36_APNCC,
        },
        "passes_h030_funding_gate": passed,
        "decision": (
            "H030_ELIGIBLE_BUT_NOT_AUTOMATICALLY_LAUNCHED"
            if passed
            else "STOP_H030_PRESERVE_APPROXIMATELY_25_T4_HOURS"
        ),
        "cross_population_subtraction": False,
        "cv57_access": False,
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    require_hash(args.bank_seal, args.bank_seal_sha256, "CUDA bank seal")
    bank_seal = load_json(args.bank_seal, "CUDA bank seal")
    if (
        bank_seal.get("schema") != "celltrack-alpha.g03-phase2-h005-cuda-bank-seal.v1"
        or bank_seal.get("status") != "SEALED_ALL_36_SCORE_BLIND"
        or bank_seal.get("gt_access") is not False
        or bank_seal.get("score_access") is not False
        or bank_seal.get("cv57_access") is not False
        or bank_seal.get("population", {}).get("name") != "held-fit36"
    ):
        raise FixedScoreError("CUDA bank seal is not fixed-score eligible")
    checkpoint = bank_seal.get("checkpoint", {})
    hours = float(checkpoint.get("cumulative_t4_hours", -1.0))
    if checkpoint.get("sha256") != args.checkpoint_sha256 or not 4.75 <= hours <= 5.25:
        raise FixedScoreError("fixed h005 checkpoint binding drifted")
    contract = r1.load_contract(BASE_CONTRACT, BASE_CONTRACT_SHA256)
    identities = tuple(contract["scope"]["identities"])
    sealed_identities = tuple(bank_seal["population"]["identities"])
    if len(identities) != 36 or set(identities) != set(sealed_identities):
        raise FixedScoreError("held-fit36 identity drift")
    if args.output.exists():
        raise FixedScoreError("fragment output already exists")
    fragment_root = args.output / "fragments"
    fragment_root.mkdir(parents=True)
    bank_rows = {row["identity"]: row for row in bank_seal["results"]}
    records: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(identities, start=1):
        row = bank_rows.get(identity)
        if row is None:
            raise FixedScoreError(f"missing bank receipt: {identity}")
        bank_path = args.bank_seal.parent / row["bank_path"]
        require_hash(bank_path, row["bank_sha256"], identity)
        with np.load(bank_path, allow_pickle=False) as bank:
            nodes = np.asarray(bank["nodes"], dtype=np.float64)
            edges = np.asarray(bank["edges"], dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 5 or edges.ndim != 2 or edges.shape[1] != 4:
            raise FixedScoreError(f"malformed bank: {identity}")
        graph = r1.graph_from_arrays(identity, nodes, edges, contract)
        fragment = fragment_root / f"{identity.replace(':', '__', 1)}.csv"
        fragment_sha = r1.write_graph(fragment, graph, contract)
        records.append(
            {
                "identity": identity,
                "ordinal": ordinal,
                "bank_path": str(bank_path),
                "bank_sha256": row["bank_sha256"],
                "fragment_path": str(fragment),
                "fragment_sha256": fragment_sha,
                "candidate_nodes": int(nodes.shape[0]),
                "candidate_edges": int(edges.shape[0]),
                "prediction_nodes": len(graph.nodes),
                "prediction_edges": len(graph.edges),
            }
        )
    result = {
        "schema": SEAL_SCHEMA,
        "status": "SEALED_36_FIXED_H005_FRAGMENTS_SCORE_BLIND",
        "checkpoint": {"sha256": args.checkpoint_sha256, "cumulative_t4_hours": hours},
        "population": {"name": "held-fit36", "identity_count": 36, "identities": list(identities)},
        "compiler": contract["compiler"],
        "bank_seal": {"path": str(args.bank_seal), "sha256": args.bank_seal_sha256},
        "bank_aggregate_sha256": aggregate(records, "bank_sha256"),
        "fragment_aggregate_sha256": r1.aggregate_fragment_sha(records),
        "videos": records,
        "held_fit_score_read": False,
        "gt_access": False,
        "cv57_access": False,
        "within_trajectory_selection": False,
    }
    r1.atomic_json(args.output / "FRAGMENT_SEAL.json", result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    require_hash(H000_SCORE, H000_SCORE_SHA256, "canonical h000 score")
    require_hash(args.fragment_seal, args.fragment_seal_sha256, "h005 fragment seal")
    seal_value = load_json(args.fragment_seal, "h005 fragment seal")
    if (
        seal_value.get("schema") != SEAL_SCHEMA
        or seal_value.get("status") != "SEALED_36_FIXED_H005_FRAGMENTS_SCORE_BLIND"
        or seal_value.get("held_fit_score_read") is not False
        or seal_value.get("gt_access") is not False
        or seal_value.get("cv57_access") is not False
        or seal_value.get("population", {}).get("name") != "held-fit36"
    ):
        raise FixedScoreError("fragment seal is not score eligible")
    if args.output.exists():
        raise FixedScoreError("fixed score output already exists")
    contract = r1.load_contract(BASE_CONTRACT, BASE_CONTRACT_SHA256)
    r1.configure_runtime(contract)
    r1.verify_scorer(contract)
    r1.configure_repository_imports(contract)
    import scoreall

    identities = tuple(contract["scope"]["identities"])
    fragment_root = args.fragment_seal.parent / "fragments"
    paths = [fragment_root / f"{identity.replace(':', '__', 1)}.csv" for identity in identities]
    for row, path in zip(seal_value["videos"], paths, strict=True):
        require_hash(path, row["fragment_sha256"], row["identity"])
    manifest = scoreall.validate_fragment_set(paths, identities)
    started = time.perf_counter()
    with ZipStore(Path(contract["inputs"]["raw_archive"]["path"]), mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    observed = float(summary["score"])
    hours = float(seal_value["checkpoint"]["cumulative_t4_hours"])
    edge_counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    result = {
        "schema": SCORE_SCHEMA,
        "status": "COMPLETE_FIXED_H005_HELD_FIT36_MEASUREMENT",
        "checkpoint": seal_value["checkpoint"],
        "population": {"name": "held-fit36", "identity_count": 36},
        "candidate": {
            "score": observed,
            "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
            "edge_jaccard": float(summary["edge_jaccard"]),
            "node_recall": float(summary["node_recall"]),
            **edge_counts,
            "division_tp": int(summary["division_tp"]),
            "division_fp": int(summary["division_fp"]),
            "division_fn": int(summary["division_fn"]),
            "division_jaccard": float(summary["division_jaccard"]),
        },
        "trajectory_gate": fixed_gate(observed, hours),
        "execution": {
            "fixed_checkpoint": True,
            "within_trajectory_selection": False,
            "scoring_elapsed_seconds": time.perf_counter() - started,
            "submission": False,
        },
        "provenance": {
            "fragment_seal": {"path": str(args.fragment_seal), "sha256": args.fragment_seal_sha256},
            "checkpoint_sha256": seal_value["checkpoint"]["sha256"],
            "fragment_aggregate_sha256": seal_value["fragment_aggregate_sha256"],
            "scorer_input_aggregate_sha256": provenance["aggregate_sha256"],
            "h000_score": {"path": str(H000_SCORE), "sha256": H000_SCORE_SHA256},
        },
        "score": summary,
        "coverage": coverage,
        "input_provenance": provenance,
        "per_video": rows,
    }
    r1.atomic_json(args.output, sanitize(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make_seal = sub.add_parser("seal")
    make_seal.add_argument("--bank-seal", type=Path, required=True)
    make_seal.add_argument("--bank-seal-sha256", required=True)
    make_seal.add_argument("--checkpoint-sha256", required=True)
    make_seal.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("score")
    measure.add_argument("--fragment-seal", type=Path, required=True)
    measure.add_argument("--fragment-seal-sha256", required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args) if args.command == "seal" else score(args)
    print(json.dumps(sanitize(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

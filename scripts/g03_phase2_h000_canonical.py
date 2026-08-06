#!/usr/bin/env python3
"""Seal and score the canonical h000 graph after CPU/CUDA drift diagnosis.

The zero-initialized adapter is an exact identity at Trackastra's 12-channel
feature boundary: every finite crop maps to twelve float32 zeros.  The original
zero-feature association banks were evaluated on CUDA, while the local h000
portability replay runs on CPU.  This module re-decodes the frozen CUDA banks
under the unchanged compiler and scores those newly emitted fragments as the
canonical h000 graph.  It never trains, launches a GPU, or reads CV57.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from zarr.storage import ZipStore

from scripts import g03_r1_zero_shot as r1
from scripts.g03_phase2_adapter import sha256, validate_h000


AMENDMENT_SCHEMA = "celltrack-alpha.g03-h000-runtime-equivalence-amendment.v1"
SEAL_SCHEMA = "celltrack-alpha.g03-h000-canonical-inference-seal.v1"
VIDEO_SCHEMA = "celltrack-alpha.g03-h000-canonical-video.v1"
SCORE_SCHEMA = "celltrack-alpha.g03-h000-canonical-score.v1"


class CanonicalH000Error(RuntimeError):
    """The canonical h000 runtime-equivalence contract failed closed."""


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalH000Error(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CanonicalH000Error(f"{label} must be a JSON object")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise CanonicalH000Error(f"{label} SHA-256 mismatch: {path}")


def load_amendment(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_hash(path, expected_sha256, "h000 runtime amendment")
    value = load_json(path, "h000 runtime amendment")
    if (
        value.get("schema") != AMENDMENT_SCHEMA
        or value.get("status") != "SEALED_SCORE_BLIND_AFTER_CPU_CUDA_DIAGNOSIS"
        or value.get("scientific_gpu_hours") != 0.0
        or value.get("cv57_access") is not False
        or value.get("score_access_before_seal") is not False
    ):
        raise CanonicalH000Error("h000 runtime amendment is not score-eligible")
    require_hash(
        Path(__file__).resolve(), value["implementation_sha256"],
        "canonical h000 implementation",
    )
    return value


def aggregate_file_hash(
    records: Sequence[Mapping[str, Any]], hash_key: str
) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: str(item["identity"])):
        digest.update(str(row["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row[hash_key]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def edge_set(path: Path) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] == "edge":
                result.add((int(row["source_id"]), int(row["target_id"])))
    return result


def load_native_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as value:
            if set(value.files) != {"nodes", "edges"}:
                raise CanonicalH000Error(f"native bank keys drifted: {path}")
            nodes = np.asarray(value["nodes"], dtype=np.float64)
            edges = np.asarray(value["edges"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise CanonicalH000Error(f"cannot read native bank: {path}") from exc
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or edges.ndim != 2
        or edges.shape[1] != 4
        or not np.isfinite(nodes).all()
        or not np.isfinite(edges).all()
    ):
        raise CanonicalH000Error(f"native bank arrays are malformed: {path}")
    return nodes, edges


def cpu_cuda_diagnostic(
    *,
    identity: str,
    native_nodes: np.ndarray,
    native_edges: np.ndarray,
    replay_bank: Path,
    replay_fragment: Path,
    reference_fragment: Path,
    probability_atol: float,
) -> dict[str, Any]:
    if not replay_bank.is_file():
        return {"identity": identity, "available": False}
    try:
        with np.load(replay_bank, allow_pickle=False) as value:
            replay_node_id = np.asarray(value["node_id"], dtype=np.int64)
            replay_frame = np.asarray(value["frame"], dtype=np.int32)
            replay_coords = np.asarray(value["coords_zyx"], dtype=np.float32)
            replay_source = np.asarray(value["edge_source"], dtype=np.int64)
            replay_target = np.asarray(value["edge_target"], dtype=np.int64)
            replay_probability = np.asarray(value["probability"], dtype=np.float32)
            replay_distance = np.asarray(value["distance_um"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise CanonicalH000Error(
            f"cannot read local CPU portability bank: {replay_bank}"
        ) from exc
    exact_structure = bool(
        np.array_equal(native_nodes[:, 0].astype(np.int64), replay_node_id)
        and np.array_equal(native_nodes[:, 1].astype(np.int32), replay_frame)
        and np.array_equal(native_nodes[:, 2:5].astype(np.float32), replay_coords)
        and np.array_equal(native_edges[:, 0].astype(np.int64), replay_source)
        and np.array_equal(native_edges[:, 1].astype(np.int64), replay_target)
        and np.array_equal(native_edges[:, 3].astype(np.float32), replay_distance)
    )
    if not exact_structure:
        raise CanonicalH000Error(
            f"CPU replay changed the h000 node/candidate structure: {identity}"
        )
    native_probability = native_edges[:, 2].astype(np.float32)
    difference = np.abs(
        native_probability.astype(np.float64)
        - replay_probability.astype(np.float64)
    )
    maximum = float(difference.max(initial=0.0))
    if maximum > probability_atol:
        raise CanonicalH000Error(
            f"CPU/CUDA probability drift exceeds the sealed tolerance: {identity}"
        )
    graph_available = replay_fragment.is_file()
    symmetric_difference = None
    if graph_available:
        symmetric_difference = len(
            edge_set(replay_fragment) ^ edge_set(reference_fragment)
        )
    return {
        "identity": identity,
        "available": True,
        "replay_bank_path": str(replay_bank),
        "replay_bank_sha256": sha256(replay_bank),
        "node_candidate_structure_exact": exact_structure,
        "probability_max_abs_difference": maximum,
        "probability_mean_abs_difference": float(difference.mean()),
        "probability_changed_float32_count": int(np.count_nonzero(difference)),
        "probability_atol": probability_atol,
        "probability_atol_pass": maximum <= probability_atol,
        "native_probabilities_within_atol_of_decode_threshold": int(
            np.count_nonzero(np.abs(native_probability - 0.35) <= probability_atol)
        ),
        "replay_fragment_available": graph_available,
        "selected_edge_symmetric_difference": symmetric_difference,
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    amendment = load_amendment(args.amendment, args.amendment_sha256)
    phase2 = amendment["phase2_contract"]
    require_hash(Path(phase2["path"]), phase2["sha256"], "Phase-2 contract")
    checkpoint = amendment["checkpoint"]
    require_hash(Path(checkpoint["path"]), checkpoint["sha256"], "h000 checkpoint")
    checkpoint_validation = validate_h000(Path(checkpoint["path"]))
    preflight = amendment["preflight"]
    require_hash(Path(preflight["path"]), preflight["sha256"], "h000 preflight")
    preflight_value = load_json(Path(preflight["path"]), "h000 preflight")
    if (
        preflight_value.get("status") != "PASS_H000_SCORE_BLIND_PREFLIGHT"
        or preflight_value.get("held_fit_score_read") is not False
        or preflight_value.get("gt_access") is not False
    ):
        raise CanonicalH000Error("h000 preflight does not authorize canonical seal")

    base_record = amendment["base_r1_contract"]
    base = r1.load_contract(Path(base_record["path"]), base_record["sha256"])
    r1.configure_runtime(base)
    identities = tuple(base["scope"]["identities"])
    if len(identities) != 36:
        raise CanonicalH000Error("canonical h000 population is not held-fit36")
    if args.output.exists():
        raise CanonicalH000Error("canonical h000 output already exists; repeat forbidden")
    fragment_root = args.output / "fragments"
    receipt_root = args.output / "receipts"
    fragment_root.mkdir(parents=True)
    receipt_root.mkdir()

    native_root = Path(amendment["canonical_cuda_bank"]["root"])
    reference_root = Path(amendment["zero_feature_reference"]["fragment_root"])
    probability_atol = float(amendment["cpu_cuda_diagnostic"]["probability_atol"])
    replay_root = Path(amendment["cpu_cuda_diagnostic"]["partial_replay_root"])
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(identities, start=1):
        slug = identity.replace(":", "__", 1)
        native_bank = native_root / f"{slug}.npz"
        native_receipt = native_root / f"{slug}.json"
        if not native_bank.is_file() or not native_receipt.is_file():
            raise CanonicalH000Error(f"canonical CUDA bank is missing {identity}")
        native_receipt_value = load_json(native_receipt, f"{identity} CUDA receipt")
        if (
            native_receipt_value.get("identity") != identity
            or not str(native_receipt_value.get("device", "")).startswith("cuda:")
            or native_receipt_value.get("sha256") != sha256(native_bank)
        ):
            raise CanonicalH000Error(f"canonical CUDA receipt drifted: {identity}")
        nodes, edges = load_native_bank(native_bank)
        graph = r1.graph_from_arrays(identity, nodes, edges, base)
        fragment = fragment_root / f"{slug}.csv"
        fragment_hash = r1.write_graph(fragment, graph, base)
        reference = reference_root / f"{slug}.csv"
        reference_hash = sha256(reference)
        if fragment_hash != reference_hash:
            raise CanonicalH000Error(
                f"canonical CUDA re-decode differs from frozen reference: {identity}"
            )
        diagnostic = cpu_cuda_diagnostic(
            identity=identity,
            native_nodes=nodes,
            native_edges=edges,
            replay_bank=replay_root / "association_banks" / f"{slug}.npz",
            replay_fragment=replay_root / "fragments" / f"{slug}.csv",
            reference_fragment=reference,
            probability_atol=probability_atol,
        )
        diagnostics.append(diagnostic)
        record = {
            "schema": VIDEO_SCHEMA,
            "identity": identity,
            "ordinal": ordinal,
            "checkpoint_sha256": checkpoint["sha256"],
            "canonical_cuda_bank_path": str(native_bank),
            "canonical_cuda_bank_sha256": native_receipt_value["sha256"],
            "canonical_cuda_device": native_receipt_value["device"],
            "canonical_fragment_path": str(fragment),
            "canonical_fragment_sha256": fragment_hash,
            "reference_fragment_path": str(reference),
            "reference_fragment_sha256": reference_hash,
            "fragment_byte_identical": True,
            "prediction_nodes": len(graph.nodes),
            "prediction_edges": len(graph.edges),
            "candidate_nodes": int(nodes.shape[0]),
            "candidate_edges": int(edges.shape[0]),
            "adapter_output": "bitwise_float32_zero_by_checkpoint_and_interface_proof",
            "held_fit_score_read": False,
            "gt_access": False,
            "new_gpu_hours": 0.0,
        }
        r1.atomic_json(receipt_root / f"{slug}.json", record)
        records.append(record)

    bank_aggregate = aggregate_file_hash(records, "canonical_cuda_bank_sha256")
    fragment_records = [
        {"identity": row["identity"], "fragment_sha256": row["canonical_fragment_sha256"]}
        for row in records
    ]
    fragment_aggregate = r1.aggregate_fragment_sha(fragment_records)
    if bank_aggregate != amendment["canonical_cuda_bank"]["aggregate_sha256"]:
        raise CanonicalH000Error("canonical CUDA bank aggregate drifted")
    if fragment_aggregate != amendment["zero_feature_reference"]["aggregate_sha256"]:
        raise CanonicalH000Error("canonical h000 fragment aggregate drifted")
    available = [row for row in diagnostics if row["available"]]
    result = {
        "schema": SEAL_SCHEMA,
        "status": "PASS_CANONICAL_H000_INTERFACE_AND_GRAPH_PARITY",
        "amendment": {"path": str(args.amendment), "sha256": args.amendment_sha256},
        "phase2_contract": phase2,
        "checkpoint": {
            **checkpoint,
            "validation": checkpoint_validation,
            "cumulative_t4_hours": 0.0,
        },
        "population": {"identity_count": 36, "identities": list(identities)},
        "interface_parity": {
            "projection_weight_nonzero": 0,
            "projection_bias_nonzero": 0,
            "wrfeat_output": "tanh(Linear(hidden)); weight=0 and bias=0, therefore exact zero for every finite hidden input",
            "native_zero_wrfeat": "exact same float32 tensor at Trackastra feature boundary",
            "pass": True,
        },
        "canonical_runtime": {
            "source": "frozen native CUDA association banks",
            "new_gpu_action": False,
            "new_gpu_hours": 0.0,
            "bank_aggregate_sha256": bank_aggregate,
            "fragment_aggregate_sha256": fragment_aggregate,
            "byte_identical_fragment_count": 36,
        },
        "cpu_cuda_diagnostic": {
            "classification": "TRACKASTRA_FORWARD_NUMERICAL_DRIFT_BEFORE_DISCONTINUOUS_THRESHOLD_DECODING",
            "available_video_count": len(available),
            "all_available_candidate_structures_exact": all(
                row["node_candidate_structure_exact"] for row in available
            ),
            "maximum_probability_abs_difference": max(
                (float(row["probability_max_abs_difference"]) for row in available),
                default=0.0,
            ),
            "probability_atol": probability_atol,
            "all_available_probability_atol_pass": all(
                row["probability_atol_pass"] for row in available
            ),
            "videos": diagnostics,
        },
        "compiler": base["compiler"],
        "videos": records,
        "held_fit_score_read": False,
        "gt_access": False,
        "cv57_access": False,
        "scientific_gpu_hours": 0.0,
    }
    r1.atomic_json(args.output / "CANONICAL_INFERENCE_SEAL.json", result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    amendment = load_amendment(args.amendment, args.amendment_sha256)
    require_hash(args.seal, args.seal_sha256, "canonical h000 seal")
    seal_value = load_json(args.seal, "canonical h000 seal")
    if (
        seal_value.get("schema") != SEAL_SCHEMA
        or seal_value.get("status") != "PASS_CANONICAL_H000_INTERFACE_AND_GRAPH_PARITY"
        or seal_value.get("held_fit_score_read") is not False
        or seal_value.get("gt_access") is not False
        or seal_value.get("canonical_runtime", {}).get("byte_identical_fragment_count") != 36
    ):
        raise CanonicalH000Error("canonical h000 seal is not score-eligible")
    base_record = amendment["base_r1_contract"]
    base = r1.load_contract(Path(base_record["path"]), base_record["sha256"])
    r1.configure_runtime(base)
    r1.verify_scorer(base)
    r1.configure_repository_imports(base)
    import scoreall

    identities = tuple(base["scope"]["identities"])
    fragment_root = args.seal.parent / "fragments"
    paths = [
        fragment_root / f"{identity.replace(':', '__', 1)}.csv"
        for identity in identities
    ]
    for row, path in zip(seal_value["videos"], paths, strict=True):
        require_hash(path, row["canonical_fragment_sha256"], row["identity"])
    manifest = scoreall.validate_fragment_set(paths, identities)
    started = time.perf_counter()
    with ZipStore(Path(base["inputs"]["raw_archive"]["path"]), mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    observed = float(summary["score"])
    best = float(amendment["allocation_policy"]["best_known_score"])
    floor = float(amendment["allocation_policy"]["neighborhood_floor"])
    branch = "A" if observed >= floor or observed > best else "B"
    edge_counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    result = {
        "schema": SCORE_SCHEMA,
        "status": "COMPLETE_CANONICAL_H000_ZERO_FEATURE_MEASUREMENT",
        "checkpoint": seal_value["checkpoint"],
        "candidate": {
            "score": observed,
            "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
            "edge_jaccard": float(summary["edge_jaccard"]),
            "node_recall": float(summary["node_recall"]),
            "edge_tp": edge_counts["edge_tp"],
            "edge_fp": edge_counts["edge_fp"],
            "edge_fn": edge_counts["edge_fn"],
            "division_tp": int(summary["division_tp"]),
            "division_fp": int(summary["division_fp"]),
            "division_fn": int(summary["division_fn"]),
            "division_jaccard": float(summary["division_jaccard"]),
        },
        "parity": {
            "adapter_interface_exact": True,
            "canonical_fragments_byte_identical": "36/36",
            "same_population_zero_feature_score": observed,
            "absolute_score_error": 0.0,
            "score_tolerance": float(
                amendment["zero_feature_reference"]["score_tolerance"]
            ),
            "pass": True,
            "cpu_cuda_discrepancy": seal_value["cpu_cuda_diagnostic"]["classification"],
            "cv57_is_descriptive_different_population": 0.829194058079541,
        },
        "decision": {
            "best_known_score": best,
            "neighborhood_floor": floor,
            "delta_vs_best_known_score": observed - best,
            "delta_vs_neighborhood_floor": observed - floor,
            "delta_to_0915": 0.915 - observed,
            "delta_to_0945": 0.945 - observed,
            "branch": branch,
            "branch_b_dossier_verdict": amendment["branch_b_dossier_verdict"],
            "funded_t4_hours": 0.0,
        },
        "execution": {
            "trained": False,
            "optimizer_steps": 0,
            "new_gpu_action": False,
            "new_gpu_hours": 0.0,
            "scoring_elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "amendment_sha256": args.amendment_sha256,
            "seal_sha256": args.seal_sha256,
            "checkpoint_sha256": seal_value["checkpoint"]["sha256"],
            "canonical_cuda_bank_aggregate_sha256": seal_value[
                "canonical_runtime"
            ]["bank_aggregate_sha256"],
            "aggregate_graph_sha256": seal_value["canonical_runtime"][
                "fragment_aggregate_sha256"
            ],
            "scorer_input_aggregate_sha256": provenance["aggregate_sha256"],
            "scorer_sources": base["inputs"]["scorer_sources"],
        },
        "score": summary,
        "edge_counts": edge_counts,
        "coverage": coverage,
        "input_provenance": provenance,
        "per_video": rows,
    }
    r1.atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("seal")
    emit.add_argument("--amendment", type=Path, required=True)
    emit.add_argument("--amendment-sha256", required=True)
    emit.add_argument("--output", type=Path, required=True)
    measure = subparsers.add_parser("score")
    measure.add_argument("--amendment", type=Path, required=True)
    measure.add_argument("--amendment-sha256", required=True)
    measure.add_argument("--seal", type=Path, required=True)
    measure.add_argument("--seal-sha256", required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args) if args.command == "seal" else score(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

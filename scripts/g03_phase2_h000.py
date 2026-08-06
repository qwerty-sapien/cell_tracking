#!/usr/bin/env python3
"""Emit and score the fixed G-03 Phase-2 h000 checkpoint.

This runner deliberately reuses the sealed Wave-R R1 population, Trackastra
checkpoint, association-bank construction, decoder, and scorer.  Only the
adapter carrier changes.  Before ground truth is opened, all 36 emitted graph
fragments must be byte-identical to the established zero-feature native graphs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from zarr.storage import ZipStore

from scripts import g03_r1_zero_shot as r1
from scripts.g03_phase2_adapter import (
    CHECKPOINT_SCHEMA,
    Phase2CropAdapter,
    sha256,
    state_tensor_sha256,
    validate_h000,
)


CONTRACT_SCHEMA = "celltrack-alpha.g03-phase2-zero-init-contract.v1"
PREFLIGHT_SCHEMA = "celltrack-alpha.g03-phase2-h000-preflight.v1"
VIDEO_SCHEMA = "celltrack-alpha.g03-phase2-h000-video.v1"
SEAL_SCHEMA = "celltrack-alpha.g03-phase2-h000-inference-seal.v1"
SCORE_SCHEMA = "celltrack-alpha.g03-phase2-h000-score.v1"


class H000Error(RuntimeError):
    """The fixed h000 parity measurement failed closed."""


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H000Error(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise H000Error(f"{label} must be a JSON object")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise H000Error(f"{label} SHA-256 mismatch: {path}")


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_hash(path, expected_sha256, "Phase-2 contract")
    contract = load_json(path, "Phase-2 contract")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "SEALED_BEFORE_H000_INFERENCE"
        or contract.get("h000", {}).get("training") is not False
        or contract.get("h000", {}).get("gpu") is not False
        or contract.get("h000", {}).get("cv57") is not False
        or contract.get("h000", {}).get("submission") is not False
        or contract.get("adapter", {}).get("projection", {}).get("in_features")
        != 48
        or contract.get("adapter", {}).get("projection", {}).get("out_features")
        != 12
        or contract.get("adapter", {}).get("projection", {}).get("weight_init")
        != "zeros"
        or contract.get("adapter", {}).get("projection", {}).get("bias_init")
        != "zeros"
    ):
        raise H000Error("Phase-2 contract is not the sealed zero-init design")
    require_hash(
        Path(__file__).resolve(), contract["implementation_sha256"], "h000 runner"
    )
    return contract


def load_base_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    record = contract["base_r1_contract"]
    return r1.load_contract(Path(record["path"]), str(record["sha256"]))


def load_checkpoint_adapter(
    checkpoint_path: Path, checkpoint_sha256: str
) -> tuple[Phase2CropAdapter, dict[str, Any]]:
    require_hash(checkpoint_path, checkpoint_sha256, "h000 checkpoint")
    validation = validate_h000(checkpoint_path)
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise H000Error("h000 checkpoint schema drifted")
    adapter = Phase2CropAdapter().eval()
    adapter.load_state_dict(value["adapter"]["state_dict"], strict=True)
    return adapter, validation


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    base = load_base_contract(contract)
    r1.configure_runtime(base)
    adapter, checkpoint_validation = load_checkpoint_adapter(
        args.checkpoint, args.checkpoint_sha256
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint["phase2_contract"]["sha256"] != args.contract_sha256
        or checkpoint["trackastra"]["model_sha256"]
        != base["inputs"]["trackastra_checkpoint"]["model_sha256"]
        or checkpoint["trackastra"]["archive_sha256"]
        != base["inputs"]["trackastra_checkpoint"]["archive_sha256"]
    ):
        raise H000Error("h000 checkpoint does not bind the frozen inference carrier")
    raw = base["inputs"]["raw_archive"]
    raw_path = Path(raw["path"])
    if raw_path.stat().st_size != int(raw["bytes"]):
        raise H000Error("raw archive byte count drifted")
    require_hash(raw_path, raw["sha256"], "raw competition archive")
    checkpoint_record = base["inputs"]["trackastra_checkpoint"]
    require_hash(
        Path(checkpoint_record["model_path"]),
        checkpoint_record["model_sha256"],
        "Trackastra model",
    )
    require_hash(
        Path(checkpoint_record["archive_path"]),
        checkpoint_record["archive_sha256"],
        "Trackastra archive",
    )
    source_root = Path(base["inputs"]["trackastra_source_root"])
    if r1.tree_sha256(source_root) != base["inputs"][
        "trackastra_source_tree_sha256"
    ]:
        raise H000Error("Trackastra source tree drifted")
    member_evidence = r1.verify_member_map(base)
    cache_evidence = r1.verify_cache_population(base)
    scorer_evidence = r1.verify_scorer(base)
    dependency_evidence = r1.verify_implementation_dependencies(base)
    reference_root = Path(contract["zero_feature_reference"]["fragment_root"])
    reference_records = []
    for identity in base["scope"]["identities"]:
        fragment = reference_root / f"{identity.replace(':', '__', 1)}.csv"
        if not fragment.is_file():
            raise H000Error(f"zero-feature reference is missing {identity}")
        reference_records.append(
            {"identity": identity, "fragment_sha256": sha256(fragment)}
        )
    reference_aggregate = r1.aggregate_fragment_sha(reference_records)
    if reference_aggregate != contract["zero_feature_reference"][
        "aggregate_fragment_sha256"
    ]:
        raise H000Error("zero-feature reference graph aggregate drifted")
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS_H000_SCORE_BLIND_PREFLIGHT",
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": args.checkpoint_sha256,
            "validation": checkpoint_validation,
        },
        "base_r1_contract": contract["base_r1_contract"],
        "raw_archive": raw,
        "raw_member_map": member_evidence,
        "fit105_cache": cache_evidence,
        "trackastra_checkpoint": checkpoint_record,
        "trackastra_source_tree_sha256": base["inputs"][
            "trackastra_source_tree_sha256"
        ],
        "adapter": {
            "state_tensor_sha256": state_tensor_sha256(adapter.state_dict()),
            "projection_weight_nonzero": int(
                torch.count_nonzero(adapter.projection.weight)
            ),
            "projection_bias_nonzero": int(
                torch.count_nonzero(adapter.projection.bias)
            ),
        },
        "zero_feature_reference": {
            "fragment_root": str(reference_root),
            "identity_count": len(reference_records),
            "aggregate_fragment_sha256": reference_aggregate,
        },
        "scorer_sources": scorer_evidence,
        "implementation_dependencies": dependency_evidence,
        "held_fit_score_read": False,
        "gt_access": False,
        "gpu_hours": 0.0,
    }
    r1.atomic_json(args.output, result)
    return result


def embed_video_zero(
    *,
    image: Any,
    detections: Sequence[Any],
    adapter: Phase2CropAdapter,
    batch_size: int,
) -> tuple[
    dict[int, tuple[int, np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    frame_rows = {
        int(frame.frame_index): (
            np.asarray(frame.node_ids, dtype=np.int64),
            np.asarray(frame.coords_zyx, dtype=np.float32),
        )
        for frame in detections
    }
    if not frame_rows:
        raise H000Error("video has no detections")
    min_frame, max_frame = min(frame_rows), max(frame_rows)
    frame_cache: dict[int, np.ndarray] = {}
    by_node: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    inference_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    count = 0
    output_nonzero = 0
    hidden_nonzero = 0
    hidden_abs_sum = 0.0

    def get_frame(index: int) -> np.ndarray:
        bounded = min(max(index, min_frame), max_frame)
        if bounded not in frame_cache:
            value = np.asarray(image[bounded])
            if value.ndim != 3 or not np.issubdtype(value.dtype, np.number):
                raise H000Error("raw image frame is not numeric Z/Y/X")
            frame_cache[bounded] = value
        return frame_cache[bounded]

    with torch.inference_mode():
        for frame_index in sorted(frame_rows):
            node_ids, coords = frame_rows[frame_index]
            output = []
            source_frames = [
                get_frame(frame_index - 1),
                get_frame(frame_index),
                get_frame(frame_index + 1),
            ]
            for start in range(0, len(node_ids), batch_size):
                stop = min(len(node_ids), start + batch_size)
                crops = r1.extract_crop_batch(source_frames, coords[start:stop])
                hidden = adapter.hidden(crops)
                features = adapter.output_activation(adapter.projection(hidden))
                hidden_nonzero += int(torch.count_nonzero(hidden))
                hidden_abs_sum += float(hidden.abs().sum())
                output_nonzero += int(torch.count_nonzero(features))
                output.append(features.to(device="cpu", dtype=torch.float32).numpy())
            wrfeat = (
                np.concatenate(output, axis=0)
                if output
                else np.empty((0, 12), dtype=np.float32)
            )
            if wrfeat.shape != (len(node_ids), 12) or not np.isfinite(wrfeat).all():
                raise H000Error("wrfeat population is malformed")
            if np.count_nonzero(wrfeat) != 0:
                raise H000Error("h000 emitted nonzero wrfeat on a real crop")
            inference_frames[frame_index] = (node_ids, coords)
            for node, coord, feature in zip(node_ids, coords, wrfeat, strict=True):
                node_id = int(node)
                if node_id in by_node:
                    raise H000Error("detection node ID repeats")
                by_node[node_id] = (frame_index, coord, feature)
            count += len(wrfeat)
            for stale in [value for value in frame_cache if value < frame_index]:
                del frame_cache[stale]
    if count == 0 or hidden_nonzero == 0 or hidden_abs_sum <= 0.0:
        raise H000Error("adapter hidden activations are not crop-dependent")
    evidence = {
        "node_count": count,
        "nonzero_row_count": 0,
        "output_nonzero_value_count": output_nonzero,
        "all_rows_zero": output_nonzero == 0,
        "hidden_nonzero_value_count": hidden_nonzero,
        "hidden_abs_sum": hidden_abs_sum,
        "hidden_activations_nonzero": hidden_nonzero > 0,
        "channel_mean": [0.0] * 12,
        "channel_std": [0.0] * 12,
        "channel_min": [0.0] * 12,
        "channel_max": [0.0] * 12,
    }
    return by_node, inference_frames, evidence


def completed_video(
    identity: str, output: Path, contract_sha256: str
) -> dict[str, Any] | None:
    slug = identity.replace(":", "__", 1)
    fragment = output / "fragments" / f"{slug}.csv"
    bank = output / "association_banks" / f"{slug}.npz"
    receipt_path = output / "receipts" / f"{slug}.json"
    if not fragment.exists() and not bank.exists() and not receipt_path.exists():
        return None
    if not fragment.is_file() or not bank.is_file() or not receipt_path.is_file():
        raise H000Error(f"partial completed-video state: {identity}")
    receipt = load_json(receipt_path, f"{identity} receipt")
    if (
        receipt.get("schema") != VIDEO_SCHEMA
        or receipt.get("identity") != identity
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("fragment_sha256") != sha256(fragment)
        or receipt.get("association_bank_sha256") != sha256(bank)
        or receipt.get("held_fit_score_read") is not False
        or receipt.get("gt_access") is not False
    ):
        raise H000Error(f"completed-video receipt drifted: {identity}")
    return receipt


def infer(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    base = load_base_contract(contract)
    r1.configure_runtime(base)
    require_hash(args.preflight, args.preflight_sha256, "h000 preflight")
    preflight_receipt = load_json(args.preflight, "h000 preflight")
    if (
        preflight_receipt.get("schema") != PREFLIGHT_SCHEMA
        or preflight_receipt.get("status") != "PASS_H000_SCORE_BLIND_PREFLIGHT"
        or preflight_receipt.get("contract", {}).get("sha256")
        != args.contract_sha256
        or preflight_receipt.get("held_fit_score_read") is not False
    ):
        raise H000Error("preflight does not authorize h000 inference")
    if (args.output / "INFERENCE_SEAL.json").exists():
        raise H000Error("h000 inference is already sealed; repeat forbidden")
    adapter, _ = load_checkpoint_adapter(args.checkpoint, args.checkpoint_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    for child in ("fragments", "association_banks", "receipts"):
        (args.output / child).mkdir(exist_ok=True)
    model = r1.load_trackastra(base)
    identities = tuple(base["scope"]["identities"])
    reference_root = Path(contract["zero_feature_reference"]["fragment_root"])
    started = time.monotonic()
    accumulated_prior_seconds = 0.0
    records = []
    raw_path = Path(base["inputs"]["raw_archive"]["path"])
    original_embed = r1.embed_video
    r1.embed_video = embed_video_zero
    try:
        with ZipStore(raw_path, mode="r") as store:
            for ordinal, identity in enumerate(identities, start=1):
                if (
                    accumulated_prior_seconds + time.monotonic() - started
                    >= float(base["runtime"]["hard_wall_seconds"]) - 120.0
                ):
                    raise TimeoutError("h000 hard wall reached before sealing")
                existing = completed_video(identity, args.output, args.contract_sha256)
                if existing is not None:
                    records.append(existing)
                    accumulated_prior_seconds += float(
                        existing["runtime"]["elapsed_seconds"]
                    )
                    continue
                nodes, edges, runtime = r1.run_video(
                    identity, model, adapter, store, base
                )
                graph = r1.graph_from_arrays(identity, nodes, edges, base)
                slug = identity.replace(":", "__", 1)
                fragment = args.output / "fragments" / f"{slug}.csv"
                bank = args.output / "association_banks" / f"{slug}.npz"
                bank_hash = r1.write_association_bank(bank, identity, nodes, edges)
                fragment_hash = r1.write_graph(fragment, graph, base)
                reference = reference_root / f"{slug}.csv"
                reference_hash = sha256(reference)
                receipt = {
                    "schema": VIDEO_SCHEMA,
                    "identity": identity,
                    "ordinal": ordinal,
                    "contract_sha256": args.contract_sha256,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "fragment_path": str(fragment),
                    "fragment_sha256": fragment_hash,
                    "fragment_bytes": fragment.stat().st_size,
                    "reference_fragment_path": str(reference),
                    "reference_fragment_sha256": reference_hash,
                    "fragment_byte_identical_to_zero_feature_reference": (
                        fragment_hash == reference_hash
                    ),
                    "association_bank_path": str(bank),
                    "association_bank_sha256": bank_hash,
                    "association_bank_bytes": bank.stat().st_size,
                    "association_bank_complete": True,
                    "prediction_nodes": len(graph.nodes),
                    "prediction_edges": len(graph.edges),
                    "predicted_forks": 0,
                    "runtime": runtime,
                    "adapter_training_steps": 0,
                    "gt_access": False,
                    "held_fit_score_read": False,
                    "gpu_hours": 0.0,
                }
                r1.atomic_json(args.output / "receipts" / f"{slug}.json", receipt)
                records.append(receipt)
                print(
                    f"h000 infer {ordinal:02d}/36 {identity} "
                    f"nodes={len(graph.nodes)} edges={len(graph.edges)} "
                    f"parity={fragment_hash == reference_hash} "
                    f"elapsed={runtime['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    finally:
        r1.embed_video = original_embed
    if len(records) != 36 or {row["identity"] for row in records} != set(identities):
        raise H000Error("h000 inference did not cover the sealed population")
    matching = sum(
        row["fragment_byte_identical_to_zero_feature_reference"] for row in records
    )
    status = (
        "PASS_ALL_36_BYTE_IDENTICAL_TO_ZERO_FEATURE_REFERENCE"
        if matching == 36
        else "FAIL_DECODING_PATH_MISMATCH_SCORE_CLOSED"
    )
    seal = {
        "schema": SEAL_SCHEMA,
        "status": status,
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": args.checkpoint_sha256,
            "checkpoint_tag": "h000",
            "cumulative_t4_hours": 0.0,
        },
        "preflight": {"path": str(args.preflight), "sha256": args.preflight_sha256},
        "population": {
            "family": "44b6",
            "identity_count": 36,
            "identities": list(identities),
        },
        "adapter": {
            "state_tensor_sha256": state_tensor_sha256(adapter.state_dict()),
            "training_steps": 0,
            "all_real_crop_outputs_bitwise_zero": all(
                row["runtime"]["wrfeat"]["all_rows_zero"] is True
                for row in records
            ),
            "all_real_crop_hidden_activations_nonzero": all(
                row["runtime"]["wrfeat"]["hidden_activations_nonzero"] is True
                for row in records
            ),
        },
        "compiler": base["compiler"],
        "parity": {
            "reference_fragment_root": str(reference_root),
            "byte_identical_fragment_count": matching,
            "mismatched_identities": [
                row["identity"]
                for row in records
                if not row["fragment_byte_identical_to_zero_feature_reference"]
            ],
            "candidate_aggregate_fragment_sha256": r1.aggregate_fragment_sha(
                records
            ),
            "reference_aggregate_fragment_sha256": contract[
                "zero_feature_reference"
            ]["aggregate_fragment_sha256"],
        },
        "aggregate": {
            "fragment_sha256": r1.aggregate_fragment_sha(records),
            "association_bank_sha256": r1.aggregate_bank_sha(records),
            "association_bank_bytes": sum(
                int(row["association_bank_bytes"]) for row in records
            ),
            "node_count": sum(int(row["prediction_nodes"]) for row in records),
            "edge_count": sum(int(row["prediction_edges"]) for row in records),
            "candidate_edge_count": sum(
                int(row["runtime"]["candidate_edge_count"]) for row in records
            ),
            "inference_calls": sum(
                int(row["runtime"]["inference_calls"]) for row in records
            ),
            "peak_tokens": max(
                int(row["runtime"]["peak_tokens"]) for row in records
            ),
            "predicted_forks": 0,
            "elapsed_seconds": accumulated_prior_seconds
            + time.monotonic()
            - started,
        },
        "videos": records,
        "gt_access": False,
        "held_fit_score_read": False,
        "gpu_hours": 0.0,
    }
    r1.atomic_json(args.output / "INFERENCE_SEAL.json", seal)
    if matching != 36:
        raise H000Error(
            "h000 graphs differ from the zero-feature reference; scorer remains closed"
        )
    return seal


def score(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    base = load_base_contract(contract)
    r1.configure_runtime(base)
    require_hash(args.inference_seal, args.inference_seal_sha256, "h000 seal")
    seal = load_json(args.inference_seal, "h000 seal")
    identities = tuple(base["scope"]["identities"])
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status")
        != "PASS_ALL_36_BYTE_IDENTICAL_TO_ZERO_FEATURE_REFERENCE"
        or tuple(seal.get("population", {}).get("identities", [])) != identities
        or seal.get("parity", {}).get("byte_identical_fragment_count") != 36
        or seal.get("gt_access") is not False
        or seal.get("held_fit_score_read") is not False
    ):
        raise H000Error("h000 inference seal is not score-eligible")
    records = seal["videos"]
    fragment_root = args.inference_seal.parent / "fragments"
    bank_root = args.inference_seal.parent / "association_banks"
    for row in records:
        slug = str(row["identity"]).replace(":", "__", 1)
        require_hash(fragment_root / f"{slug}.csv", row["fragment_sha256"], slug)
        require_hash(
            bank_root / f"{slug}.npz", row["association_bank_sha256"], slug
        )
    if r1.aggregate_fragment_sha(records) != seal["aggregate"]["fragment_sha256"]:
        raise H000Error("h000 graph aggregate drifted")
    if r1.aggregate_bank_sha(records) != seal["aggregate"][
        "association_bank_sha256"
    ]:
        raise H000Error("h000 association-bank aggregate drifted")
    r1.verify_scorer(base)
    r1.configure_repository_imports(base)
    import scoreall

    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(fragment_root)), identities
    )
    started = time.perf_counter()
    with ZipStore(Path(base["inputs"]["raw_archive"]["path"]), mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    edge_counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    best = float(contract["allocation_policy"]["best_known_score"])
    floor = float(contract["allocation_policy"]["neighborhood_floor"])
    observed = float(summary["score"])
    branch = "A" if observed >= floor or observed > best else "B"
    result = {
        "schema": SCORE_SCHEMA,
        "status": "COMPLETE_H000_ZERO_FEATURE_PARITY_MEASUREMENT",
        "checkpoint": seal["checkpoint"],
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
            "branch": branch,
        },
        "parity": {
            "population": "held-fit36-44b6",
            "reference": "established zero-wrfeat native fragments",
            "fragment_identity": "36/36 byte-identical",
            "score_tolerance": float(
                contract["zero_feature_reference"]["score_tolerance"]
            ),
            "same_population_reference_score": observed,
            "absolute_score_error": 0.0,
            "pass": True,
            "cv57_native_score_is_different_population": float(
                contract["zero_feature_reference"]["cv57_descriptive_score"]
            ),
        },
        "decision": {
            "best_known_score": best,
            "neighborhood_floor": floor,
            "delta_vs_best_known_score": observed - best,
            "delta_vs_neighborhood_floor": observed - floor,
            "branch": branch,
            "branch_rule": contract["allocation_policy"]["branch_rule"],
            "branch_b_dossier_required_before_gpu": branch == "B",
            "gpu_allocation_t4_hours": 0,
        },
        "execution": {
            "trained": False,
            "optimizer_steps": 0,
            "gpu_hours": 0.0,
            "inference_elapsed_seconds": float(
                seal["aggregate"]["elapsed_seconds"]
            ),
            "scoring_elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "contract_sha256": args.contract_sha256,
            "checkpoint_sha256": seal["checkpoint"]["sha256"],
            "aggregate_graph_sha256": seal["aggregate"]["fragment_sha256"],
            "association_bank_sha256": seal["aggregate"][
                "association_bank_sha256"
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
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--contract", type=Path, required=True)
    pre.add_argument("--contract-sha256", required=True)
    pre.add_argument("--checkpoint", type=Path, required=True)
    pre.add_argument("--checkpoint-sha256", required=True)
    pre.add_argument("--output", type=Path, required=True)
    emit = subparsers.add_parser("infer")
    emit.add_argument("--contract", type=Path, required=True)
    emit.add_argument("--contract-sha256", required=True)
    emit.add_argument("--checkpoint", type=Path, required=True)
    emit.add_argument("--checkpoint-sha256", required=True)
    emit.add_argument("--preflight", type=Path, required=True)
    emit.add_argument("--preflight-sha256", required=True)
    emit.add_argument("--output", type=Path, required=True)
    measure = subparsers.add_parser("score")
    measure.add_argument("--contract", type=Path, required=True)
    measure.add_argument("--contract-sha256", required=True)
    measure.add_argument("--inference-seal", type=Path, required=True)
    measure.add_argument("--inference-seal-sha256", required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args)
    elif args.command == "infer":
        result = infer(args)
    else:
        result = score(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

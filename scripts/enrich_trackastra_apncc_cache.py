#!/usr/bin/env python3
"""Append frozen Trackastra margins to a complete 48-D APNCC base cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_trackastra_apncc_base_cache import (  # noqa: E402
    CHECKPOINT_SCHEMA as BASE_CHECKPOINT_SCHEMA,
    SCHEMA as BASE_REPORT_SCHEMA,
    VIDEO_SCHEMA as BASE_VIDEO_SCHEMA,
    _cache_identity,
    _eligible,
)
from build_trackastra_full_pool_feature_cache import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    DEFAULT_CONTRACT,
    DEFAULT_CONTRACT_SHA256,
    VIDEO_SCHEMA,
    TrackastraFeatureCacheError,
    _atomic_json,
    _atomic_torch_save,
    _checkpoint,
    _load_json,
    _require,
    assert_v1_parity,
    enrich_base_payload,
    load_contract,
    trackastra_margins_from_flattened,
)
from celltrack_alpha import bh_model  # noqa: E402
from evaluate_trackastra_zero_wrfeat_signal import sha256  # noqa: E402
from p4_edge_attention import DEFAULT_CACHE  # noqa: E402
from trackastra_checkpoint_loader import load_mmap_meta  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-full-pool-feature-cache.v2"
DEFAULT_BASE = ROOT / "reports/g02/trackastra_apncc_base_cache_v2"
DEFAULT_OUTPUT = ROOT / "reports/g02/trackastra_full_pool_feature_cache_v2"


def _validate_base_report(
    path: Path,
    *,
    contract_sha256: str,
    expected_identities: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    value = _load_json(path, "APNCC base report")
    files = value.get("files")
    aggregate = value.get("aggregate")
    model_stack = value.get("model_stack")
    if (
        value.get("schema") != BASE_REPORT_SCHEMA
        or value.get("verdict") != "PASS_APNCC_BASE_CACHE"
        or value.get("contract", {}).get("sha256") != contract_sha256
        or value.get("feature_count") != 48
        or not isinstance(files, list)
        or len(files) != len(expected_identities)
        or not isinstance(aggregate, Mapping)
        or int(aggregate.get("video_count", -1))
        != len(expected_identities)
        or not isinstance(model_stack, Mapping)
        or model_stack.get("v45_loaded") is not True
        or model_stack.get("trackastra_loaded") is not False
    ):
        raise TrackastraFeatureCacheError(
            "APNCC base report is incomplete or drifted"
        )
    by_identity = {
        str(row["identity"]): row
        for row in files
        if isinstance(row, Mapping)
    }
    if set(by_identity) != set(expected_identities):
        raise TrackastraFeatureCacheError(
            "APNCC base report identities are incomplete"
        )
    return value, by_identity


def _load_base_video(
    record: Mapping[str, object],
    *,
    identity: str,
    contract_sha256: str,
) -> dict[str, object]:
    feature_path = ROOT / str(record.get("feature_path"))
    checkpoint_path = feature_path.with_suffix(".json")
    checkpoint = _load_json(
        checkpoint_path, f"{identity} APNCC base checkpoint"
    )
    if (
        checkpoint.get("schema") != BASE_CHECKPOINT_SCHEMA
        or checkpoint.get("identity") != identity
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint.get("feature_sha256") != sha256(feature_path)
        or checkpoint.get("feature_sha256") != record.get("feature_sha256")
    ):
        raise TrackastraFeatureCacheError(
            f"{identity} APNCC base checkpoint drifted"
        )
    try:
        payload = torch.load(
            feature_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise TrackastraFeatureCacheError(
            f"cannot restore APNCC base video for {identity}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != BASE_VIDEO_SCHEMA
        or payload.get("identity") != identity
        or payload.get("contract_sha256") != contract_sha256
        or len(payload.get("feature_names", ())) != 48
        or np.asarray(payload.get("row_features")).ndim != 2
        or np.asarray(payload.get("row_features")).shape[1] != 48
    ):
        raise TrackastraFeatureCacheError(
            f"{identity} APNCC base payload drifted"
        )
    return payload


def build(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    contract = load_contract(args.contract, args.contract_sha256)
    implementation_sha256 = sha256(Path(__file__))
    eligible, _prepare_rows, feasibility_rows = _eligible(
        contract, getattr(args, "identity", None)
    )
    base_report, base_rows = _validate_base_report(
        args.base / "report.json",
        contract_sha256=args.contract_sha256,
        expected_identities=eligible,
    )
    inputs = contract["inputs"]
    for label in ("fit_prepare", "full_pool_feasibility", "asset_audit"):
        record = inputs[label]
        _require(ROOT / str(record["path"]), str(record["sha256"]), label)
    if (
        set(eligible) != set(base_rows)
        or eligible[0]
        != contract["checkpoint_loader"]["parity_reference"]["identity"]
    ):
        raise TrackastraFeatureCacheError(
            "Trackastra enrichment video scope drifted"
        )
    model = load_mmap_meta(
        source_root=args.source_root,
        checkpoint_zip=args.checkpoint_zip,
        checkpoint_member=args.checkpoint_member,
        contract=contract,
    )

    args.output.mkdir(parents=True, exist_ok=args.resume)
    videos_root = args.output / "videos"
    videos_root.mkdir(exist_ok=args.resume)
    files: list[dict[str, object]] = []
    resumed = 0
    for ordinal, identity in enumerate(eligible, start=1):
        slug = identity.replace(":", "__", 1)
        feature_path = videos_root / f"{slug}.pt"
        checkpoint_path = videos_root / f"{slug}.json"
        expected_proposals = int(base_rows[identity]["proposal_count"])
        existing = (
            _checkpoint(
                path=checkpoint_path,
                feature_path=feature_path,
                identity=identity,
                contract_sha256=args.contract_sha256,
                implementation_sha256=implementation_sha256,
                expected_proposals=expected_proposals,
            )
            if args.resume
            else None
        )
        if existing is not None:
            files.append(existing)
            resumed += 1
            print(
                f"[{ordinal}/{len(eligible)}] {identity}: enriched resume",
                flush=True,
            )
            continue
        if feature_path.exists() or checkpoint_path.exists():
            raise TrackastraFeatureCacheError(
                f"refusing to overwrite enriched checkpoint for {identity}"
            )
        base = _load_base_video(
            base_rows[identity],
            identity=identity,
            contract_sha256=args.contract_sha256,
        )
        meta = _load_json(
            args.cache / f"{slug}.json", f"{identity} cache metadata"
        )
        detections = bh_model.load_sparse_cache_detections(
            args.cache / f"{slug}.bhc",
            expected_identity=_cache_identity(meta),
        )
        margins, inference = trackastra_margins_from_flattened(
            model=model,
            detections=detections,
            base=base,
            max_tokens=int(contract["features"]["max_tokens"]),
        )
        flattened = enrich_base_payload(base, margins)
        payload = {
            "schema": VIDEO_SCHEMA,
            "identity": identity,
            "contract_sha256": args.contract_sha256,
            "implementation_sha256": implementation_sha256,
            "feature_names": [
                *[f"apncc_context_{index:02d}" for index in range(48)],
                "trackastra_margin",
            ],
            "selected_raw_edges": int(base["selected_raw_edges"]),
            "pool_counts": base["pool_counts"],
            "inference": inference,
            **flattened,
        }
        if identity == eligible[0]:
            assert_v1_parity(payload, contract)
        _atomic_torch_save(feature_path, payload)
        record = {
            "schema": CHECKPOINT_SCHEMA,
            "identity": identity,
            "contract_sha256": args.contract_sha256,
            "implementation_sha256": implementation_sha256,
            "base_feature_sha256": base_rows[identity]["feature_sha256"],
            "feature_path": str(feature_path.relative_to(ROOT)),
            "feature_sha256": sha256(feature_path),
            "bytes": feature_path.stat().st_size,
            "proposal_count": len(flattened["challengers"]),
            "row_count": len(flattened["conflicts"]),
            "transition_count": len(flattened["transition_target_t"]),
            **inference,
        }
        _atomic_json(checkpoint_path, record)
        files.append(record)
        print(
            f"[{ordinal}/{len(eligible)}] {identity}: "
            f"{record['proposal_count']} proposals / "
            f"{record['transition_count']} calls",
            flush=True,
        )
        del base, detections, margins, flattened, payload
        gc.collect()
    aggregate = {
        "video_count": len(files),
        "proposal_count": sum(int(row["proposal_count"]) for row in files),
        "row_count": sum(int(row["row_count"]) for row in files),
        "inference_calls": sum(
            int(row["inference_calls"]) for row in files
        ),
        "cropped_calls": sum(int(row["cropped_calls"]) for row in files),
        "peak_tokens": max(int(row["peak_tokens"]) for row in files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "resumed_video_count": resumed,
    }
    if (
        aggregate["video_count"] != len(eligible)
        or aggregate["proposal_count"]
        != sum(
            int(feasibility_rows[identity]["geometry_proposals"])
            for identity in eligible
        )
        or aggregate["inference_calls"]
        != sum(
            int(feasibility_rows[identity]["nonempty_transitions"])
            for identity in eligible
        )
    ):
        raise TrackastraFeatureCacheError(
            "enriched cache aggregate coverage drifted"
        )
    return {
        "schema": SCHEMA,
        "verdict": "PASS_LABEL_FREE_FEATURE_CACHE",
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
        "implementation_sha256": implementation_sha256,
        "base_report": {
            "path": str((args.base / "report.json").relative_to(ROOT)),
            "sha256": sha256(args.base / "report.json"),
            "implementation_sha256": base_report[
                "implementation_sha256"
            ],
        },
        "scope": contract["scope"],
        "feature_contract": contract["features"],
        "checkpoint_loader": contract["checkpoint_loader"],
        "model_stack": {
            "trackastra_loaded": True,
            "v45_loaded": False,
            "raw_image_access": False,
        },
        "aggregate": aggregate,
        "execution": {
            "device": "cpu",
            "training": False,
            "label_access": False,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "files": files,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-zip", type=Path, required=True)
    parser.add_argument("--checkpoint-member", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--identity",
        help=(
            "run only the contract parity reference; any other identity "
            "fails closed"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "contract",
        "source_root",
        "checkpoint_zip",
        "checkpoint_member",
        "cache",
        "base",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    report_path = args.output / "report.json"
    if report_path.exists():
        raise TrackastraFeatureCacheError(
            f"refusing to overwrite completed report: {report_path}"
        )
    report = build(args)
    _atomic_json(report_path, report)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

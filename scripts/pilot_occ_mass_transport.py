#!/usr/bin/env python3
"""Run the preregistered label-blind OCC-MASS-TRANSPORT CPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

from occ_mass_transport_core import (
    CROP_RADIUS_ZYX,
    EPSILONS,
    SINKHORN_ITERATIONS,
    endpoint_batch,
    extract_reflect_patch,
    feature_names,
    pair_features,
)


ROOT = Path(__file__).resolve().parents[1]
REQUEST_ROOT = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/feature_requests/occupied_swap"
REQUEST_MANIFEST = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/ACTION_REQUEST_MANIFEST_V2.json"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
EXPECTED_MANIFEST_SHA256 = "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
SUPPORTS = (16, 32, 48)
PILOT_IDENTITIES = (
    "train:44b6_0db75fae",
    "train:44b6_12dfb391",
    "train:44b6_144b256d",
)
ROWS_PER_IDENTITY = 96
TOTAL_POPULATION = 45_961


class PilotError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def load_requests() -> list[tuple[str, Path, dict[str, np.ndarray], str]]:
    if sha256(REQUEST_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise PilotError("request manifest hash drift")
    manifest = json.loads(REQUEST_MANIFEST.read_text(encoding="ascii"))
    by_identity = {
        str(row["identity"]): row
        for row in manifest["requests"]
        if row.get("action_class") == "occupied_swap"
    }
    output = []
    for identity in PILOT_IDENTITIES:
        row = by_identity.get(identity)
        if row is None:
            raise PilotError(f"pilot identity absent: {identity}")
        path = REQUEST_ROOT / (identity.replace(":", "__", 1) + ".npz")
        if sha256(path) != row["sha256"]:
            raise PilotError(f"request hash drift: {identity}")
        with np.load(path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        required = {"source_id", "target_id", "source_t", "target_t", "source_zyx", "target_zyx"}
        if set(request) != required or len(request["source_id"]) != row["action_count"]:
            raise PilotError(f"request schema/count drift: {identity}")
        output.append((identity, path, request, str(row["sha256"])))
    return output


def extract(array: zarr.Array, request: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows = min(ROWS_PER_IDENTITY, len(request["source_id"]))
    source = np.stack(
        [
            extract_reflect_patch(array, int(request["source_t"][i]), request["source_zyx"][i])
            for i in range(rows)
        ]
    )
    target = np.stack(
        [
            extract_reflect_patch(array, int(request["target_t"][i]), request["target_zyx"][i])
            for i in range(rows)
        ]
    )
    return source, target


def compute_once(requests: list[tuple[str, Path, dict[str, np.ndarray], str]]) -> tuple[dict[str, Any], bytes]:
    started = time.perf_counter()
    support_features: dict[int, list[np.ndarray]] = {support: [] for support in SUPPORTS}
    rows = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for identity, _path, request, request_sha in requests:
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            array = group["0"]
            source_patches, target_patches = extract(array, request)
            for support in SUPPORTS:
                source = endpoint_batch(source_patches, support=support, device="cpu")
                target = endpoint_batch(target_patches, support=support, device="cpu")
                support_features[support].append(pair_features(source, target))
            rows.append(
                {
                    "identity": identity,
                    "request_sha256": request_sha,
                    "sampled_actions": len(source_patches),
                    "raw_shape": list(array.shape),
                    "crop_shape": list(source_patches.shape[1:]),
                }
            )
    elapsed = time.perf_counter() - started
    joined = {support: np.concatenate(values, axis=0) for support, values in support_features.items()}
    reference = joined[48][:, ::2]
    deviations = {
        str(support): float(np.median(np.abs(joined[support][:, ::2] - reference)))
        for support in SUPPORTS
    }
    residuals = joined[48][:, 1:6:2].reshape(-1)
    eligible = [
        support
        for support in SUPPORTS
        if deviations[str(support)] <= 0.02
        and float(np.quantile(joined[support][:, 1:6:2], 0.99)) <= 1.0e-3
    ]
    selected = min(eligible) if eligible else None
    sampled = int(sum(row["sampled_actions"] for row in rows))
    projection_h = elapsed / sampled * TOTAL_POPULATION / 3600.0
    payload = np.concatenate([joined[support] for support in SUPPORTS], axis=1).tobytes(order="C")
    result = {
        "schema": "celltrack-alpha.occ-mass-transport-pilot.v1",
        "status": "PASS" if selected is not None and projection_h <= 1.0 else "FAIL",
        "scope": {
            "identities": list(PILOT_IDENTITIES),
            "rows_per_identity": ROWS_PER_IDENTITY,
            "sampled_actions": sampled,
            "labels_opened": False,
            "graph_score_opened": False,
        },
        "parameters": {
            "crop_radius_zyx": list(CROP_RADIUS_ZYX),
            "crop_radius_um": (np.asarray(CROP_RADIUS_ZYX) * np.asarray((1.625, 0.40625, 0.40625))).tolist(),
            "supports": list(SUPPORTS),
            "epsilons": list(EPSILONS),
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "selected_support": selected,
            "selection_rule": "smallest support with median abs OT-cost deviation <=0.02 vs K48 and p99 residual <=1e-3",
        },
        "measurements": {
            "elapsed_seconds": elapsed,
            "conservative_local_cpu_projection_h": projection_h,
            "median_abs_cost_deviation_vs_k48": deviations,
            "k48_marginal_residual_p99": float(np.quantile(residuals, 0.99)),
            "finite": all(bool(np.isfinite(value).all()) for value in joined.values()),
            "crop_coverage": 1.0,
        },
        "feature_names": list(feature_names()),
        "requests": rows,
        "numeric_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return result, canonical_bytes(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise PilotError(f"refusing to overwrite {args.output}")
    requests = load_requests()
    first, _ = compute_once(requests)
    second, _ = compute_once(requests)
    # Timing is intentionally excluded from deterministic scientific-byte parity.
    deterministic_first = dict(first)
    deterministic_second = dict(second)
    deterministic_first["measurements"] = dict(first["measurements"])
    deterministic_second["measurements"] = dict(second["measurements"])
    for value in (deterministic_first, deterministic_second):
        value["measurements"].pop("elapsed_seconds")
        value["measurements"].pop("conservative_local_cpu_projection_h")
    hash_first = hashlib.sha256(canonical_bytes(deterministic_first)).hexdigest()
    hash_second = hashlib.sha256(canonical_bytes(deterministic_second)).hexdigest()
    first["determinism"] = {
        "repeat_scientific_sha256": [hash_first, hash_second],
        "byte_identical": hash_first == hash_second,
    }
    if not first["determinism"]["byte_identical"]:
        first["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(first))
    print(json.dumps(first, sort_keys=True, indent=2))
    return 0 if first["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


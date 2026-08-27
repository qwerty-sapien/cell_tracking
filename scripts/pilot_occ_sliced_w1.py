#!/usr/bin/env python3
"""Run the preregistered N2B sliced-W1 fallback pilot."""

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

from occ_mass_transport_core import CROP_RADIUS_ZYX, extract_reflect_patch, sliced_feature_names, sliced_pair_features


ROOT = Path(__file__).resolve().parents[1]
REQUEST_ROOT = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/feature_requests/occupied_swap"
REQUEST_MANIFEST = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/ACTION_REQUEST_MANIFEST_V2.json"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
EXPECTED_MANIFEST_SHA256 = "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
PILOT_IDENTITIES = ("train:44b6_0db75fae", "train:44b6_12dfb391", "train:44b6_144b256d")
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


def load_requests() -> list[tuple[str, dict[str, np.ndarray], str]]:
    if sha256(REQUEST_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise PilotError("request manifest hash drift")
    manifest = json.loads(REQUEST_MANIFEST.read_text(encoding="ascii"))
    by_identity = {str(row["identity"]): row for row in manifest["requests"] if row.get("action_class") == "occupied_swap"}
    output = []
    for identity in PILOT_IDENTITIES:
        row = by_identity[identity]
        path = REQUEST_ROOT / (identity.replace(":", "__", 1) + ".npz")
        if sha256(path) != row["sha256"]:
            raise PilotError(f"request hash drift: {identity}")
        with np.load(path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        output.append((identity, request, str(row["sha256"])))
    return output


def compute_once(requests: list[tuple[str, dict[str, np.ndarray], str]]) -> tuple[dict[str, Any], bytes]:
    started = time.perf_counter()
    features = []
    rows = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for identity, request, request_sha in requests:
            dataset = identity.split(":", 1)[1]
            array = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")["0"]
            count = min(ROWS_PER_IDENTITY, len(request["source_id"]))
            source = np.stack([extract_reflect_patch(array, int(request["source_t"][i]), request["source_zyx"][i]) for i in range(count)])
            target = np.stack([extract_reflect_patch(array, int(request["target_t"][i]), request["target_zyx"][i]) for i in range(count)])
            features.append(sliced_pair_features(source, target))
            rows.append({"identity": identity, "request_sha256": request_sha, "sampled_actions": count, "raw_shape": list(array.shape), "crop_shape": list(source.shape[1:])})
    joined = np.concatenate(features, axis=0)
    elapsed = time.perf_counter() - started
    sampled = len(joined)
    projection_h = elapsed / sampled * TOTAL_POPULATION / 3600.0
    result = {
        "schema": "celltrack-alpha.occ-sliced-w1-pilot.v1",
        "status": "PASS" if np.isfinite(joined).all() and projection_h <= 1.0 else "FAIL",
        "scope": {"identities": list(PILOT_IDENTITIES), "rows_per_identity": ROWS_PER_IDENTITY, "sampled_actions": sampled, "labels_opened": False, "graph_score_opened": False},
        "parameters": {"crop_radius_zyx": list(CROP_RADIUS_ZYX), "crop_radius_um": (np.asarray(CROP_RADIUS_ZYX) * np.asarray((1.625, 0.40625, 0.40625))).tolist(), "projection_bins": 25, "projection_count": 10},
        "measurements": {"elapsed_seconds": elapsed, "conservative_local_cpu_projection_h": projection_h, "finite": bool(np.isfinite(joined).all()), "crop_coverage": 1.0},
        "feature_names": list(sliced_feature_names()),
        "requests": rows,
        "numeric_payload_sha256": hashlib.sha256(joined.tobytes(order="C")).hexdigest(),
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
    deterministic = []
    for value in (first, second):
        reduced = dict(value)
        reduced["measurements"] = dict(value["measurements"])
        reduced["measurements"].pop("elapsed_seconds")
        reduced["measurements"].pop("conservative_local_cpu_projection_h")
        deterministic.append(hashlib.sha256(canonical_bytes(reduced)).hexdigest())
    first["determinism"] = {"repeat_scientific_sha256": deterministic, "byte_identical": deterministic[0] == deterministic[1]}
    if not first["determinism"]["byte_identical"]:
        first["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(first))
    print(json.dumps(first, sort_keys=True, indent=2))
    return 0 if first["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

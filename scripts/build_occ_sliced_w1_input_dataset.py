#!/usr/bin/env python3
"""Build the dedicated immutable input carrier for OCC-SLICED-W1-v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
)
REQUEST_MANIFEST = SOURCE_ROOT / "ACTION_REQUEST_MANIFEST_V2.json"
POPULATION_MANIFEST = SOURCE_ROOT / "run_v1/population_manifest.json"
REQUEST_ROOT = SOURCE_ROOT / "run_v1/feature_requests/occupied_swap"
WHEEL_ROOT = ROOT / "external/Tracking_Support/wheels"
OUTPUT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/occ_sliced_w1_v1/input_dataset"
)

DATASET_ID = "onebitbrain/g02-occ-sliced-w1-input-v1"
REQUEST_MANIFEST_SHA256 = (
    "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
)
POPULATION_MANIFEST_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
WHEELS = {
    "donfig-0.8.1.post1-py3-none-any.whl": (
        "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
    ),
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"
    ),
    "zarr-3.2.1-py3-none-any.whl": (
        "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def checked_copy(source: Path, destination: Path, expected_sha: str, expected_bytes: int) -> None:
    if sha256(source) != expected_sha or source.stat().st_size != expected_bytes:
        raise RuntimeError(f"source asset drift: {source}")
    shutil.copyfile(source, destination)
    if sha256(destination) != expected_sha or destination.stat().st_size != expected_bytes:
        raise RuntimeError(f"copied asset drift: {destination}")


def main() -> int:
    if sha256(REQUEST_MANIFEST) != REQUEST_MANIFEST_SHA256:
        raise RuntimeError("source request manifest drift")
    if sha256(POPULATION_MANIFEST) != POPULATION_MANIFEST_SHA256:
        raise RuntimeError("source population manifest drift")

    payload = json.loads(REQUEST_MANIFEST.read_text(encoding="ascii"))
    rows = sorted(
        (row for row in payload["requests"] if row["action_class"] == "occupied_swap"),
        key=lambda row: row["identity"],
    )
    if len(rows) != 36 or sum(int(row["action_count"]) for row in rows) != 45_961:
        raise RuntimeError("occupied-swap population drift")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected_names = {
        "dataset-metadata.json",
        "occ_sliced_w1_input_manifest.json",
        "action_request_manifest.json",
        "population_manifest.json",
        *WHEELS,
        *(str(row["path"]) for row in rows),
    }
    unexpected = {path.name for path in OUTPUT.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected pre-existing output files: {sorted(unexpected)}")

    checked_copy(
        REQUEST_MANIFEST,
        OUTPUT / "action_request_manifest.json",
        REQUEST_MANIFEST_SHA256,
        REQUEST_MANIFEST.stat().st_size,
    )
    checked_copy(
        POPULATION_MANIFEST,
        OUTPUT / "population_manifest.json",
        POPULATION_MANIFEST_SHA256,
        POPULATION_MANIFEST.stat().st_size,
    )

    request_records = []
    for row in rows:
        identity = str(row["identity"])
        source = REQUEST_ROOT / f"{identity.replace(':', '__')}.npz"
        destination = OUTPUT / str(row["path"])
        checked_copy(source, destination, str(row["sha256"]), int(row["bytes"]))
        request_records.append(
            {
                "identity": identity,
                "action_count": int(row["action_count"]),
                "bytes": int(row["bytes"]),
                "path": destination.name,
                "sha256": str(row["sha256"]),
            }
        )

    wheel_records = []
    for name, digest in sorted(WHEELS.items()):
        source = WHEEL_ROOT / name
        checked_copy(source, OUTPUT / name, digest, source.stat().st_size)
        wheel_records.append(
            {
                "bytes": source.stat().st_size,
                "path": name,
                "sha256": digest,
            }
        )

    input_manifest = {
        "schema": "celltrack-alpha.occ-sliced-w1-input.v1",
        "status": "IMMUTABLE_UPLOAD_READY",
        "dataset_id": DATASET_ID,
        "dataset_version": 1,
        "source_request_manifest": {
            "bytes": REQUEST_MANIFEST.stat().st_size,
            "path": "action_request_manifest.json",
            "sha256": REQUEST_MANIFEST_SHA256,
        },
        "source_population_manifest": {
            "bytes": POPULATION_MANIFEST.stat().st_size,
            "path": "population_manifest.json",
            "sha256": POPULATION_MANIFEST_SHA256,
        },
        "request_count": 45_961,
        "request_file_count": len(request_records),
        "requests": request_records,
        "wheel_count": len(wheel_records),
        "wheels": wheel_records,
    }
    write_json(OUTPUT / "occ_sliced_w1_input_manifest.json", input_manifest)
    write_json(
        OUTPUT / "dataset-metadata.json",
        {
            "id": DATASET_ID,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
            "title": "G02 OCC Sliced W1 Input v1",
        },
    )

    actual_names = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError("built dataset file-set drift")
    total_bytes = sum(path.stat().st_size for path in OUTPUT.iterdir() if path.is_file())
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "file_count": len(actual_names),
                "input_manifest_sha256": sha256(
                    OUTPUT / "occ_sliced_w1_input_manifest.json"
                ),
                "total_bytes": total_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

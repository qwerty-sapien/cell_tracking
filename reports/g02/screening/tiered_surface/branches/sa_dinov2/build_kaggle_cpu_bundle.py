#!/usr/bin/env python3
"""Build the frozen DINOv2 R5 Kaggle CPU input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
SOURCE = Path("/private/tmp/g02_dinov2_source_7764ea0")
WEIGHT = Path(
    "/private/tmp/g02_dinov2_assets/"
    "sha256-b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9/"
    "dinov2_vits14_pretrain.pth"
)
POP_SHA = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object JSON required: {path}")
    return value


def add(files: dict[str, Path], path: Path, archive_name: str | None = None) -> None:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"required DINOv2 input missing: {path}")
    name = archive_name or path.relative_to(ROOT).as_posix()
    if name in files:
        raise RuntimeError(f"duplicate bundle path: {name}")
    files[name] = path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if sha256(WEIGHT) != WEIGHT_SHA or WEIGHT.stat().st_size != 88_283_115:
        raise RuntimeError("DINOv2 checkpoint drifted")

    source_tar = output / "dinov2_source.tar.bin"
    with tarfile.open(source_tar, "w") as handle:
        for path in sorted(SOURCE.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            handle.add(path, arcname=Path("dinov2_source") / path.relative_to(SOURCE))

    gate = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_repr/"
        "phase2_separation_gate"
    )
    population_path = gate / "run_v1/population_manifest.json"
    if sha256(population_path) != POP_SHA:
        raise RuntimeError("DINOv2 population manifest drifted")
    population = load(population_path)
    files: dict[str, Path] = {}
    fixed = (
        HERE / "materialize_dinov2_xy_separation.py",
        HERE / "dinov2_xy_action_featurizer.py",
        HERE / "R5_XY_BETA_STAGING_CONTRACT.json",
        HERE / "KAGGLE_CPU_TRANSPORT_AMENDMENT.json",
        gate / "PREREGISTRATION_v3_per_class_authenticated_weight.json",
        gate / "run_v3/B0_SEALED.json",
        population_path,
    )
    for path in fixed:
        add(files, path)
    for action_class in ("delete", "occupied_swap"):
        for row in population["classes"][action_class]["videos"]:
            add(files, ROOT / str(row["request_path"]))
            add(files, ROOT / str(row["population_path"]))
    for wheel in (
        "donfig-0.8.1.post1-py3-none-any.whl",
        "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "zarr-3.2.1-py3-none-any.whl",
    ):
        add(files, ROOT / "external/Tracking_Support/wheels" / wheel)
    add(files, WEIGHT, "dinov2_vits14_pretrain.pth")
    add(files, source_tar, source_tar.name)

    inventory = []
    archive = output / "dinov2_cpu_bundle.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for name, path in sorted(files.items()):
            inventory.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            handle.write(path, name)
        inventory_bytes = (
            json.dumps(
                {
                    "schema": "celltrack-alpha.dinov2-cpu-bundle-inventory.v1",
                    "files": inventory,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        handle.writestr("dinov2_cpu_bundle_inventory.json", inventory_bytes)

    required_videos = sorted(
        {
            str(row["identity"])
            for action_class in ("delete", "occupied_swap")
            for row in population["classes"][action_class]["videos"]
        }
    )
    request_count = sum(
        int(row["count"])
        for action_class in ("delete", "occupied_swap")
        for row in population["classes"][action_class]["videos"]
    )
    manifest = {
        "schema": "celltrack-alpha.dinov2-r5-kaggle-cpu-bundle.v1",
        "status": "FROZEN_BEFORE_KAGGLE_CPU_OUTCOMES",
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "member_count": len(inventory) + 1,
        },
        "checkpoint": {
            "path": "dinov2_vits14_pretrain.pth",
            "bytes": WEIGHT.stat().st_size,
            "sha256": WEIGHT_SHA,
        },
        "source": {
            "path": source_tar.name,
            "bytes": source_tar.stat().st_size,
            "sha256": sha256(source_tar),
            "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "tree": "2a27257b79b0633b027a21014bc9360e3c1b3f43",
        },
        "population_manifest_sha256": POP_SHA,
        "request_count": request_count,
        "request_file_count": 77,
        "population_file_count": 77,
        "wheel_count": 3,
        "required_train_video_ids": required_videos,
        "unique_endpoint_count": 1_417_863,
        "competition_source": "biohub-cell-tracking-during-development",
        "enable_gpu": False,
        "ram_gib": 16,
    }
    (output / "dinov2_cpu_bundle_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "G02 CELLECT R7 Published Assets v1",
                "id": "onebitbrain/g02-cellect-r7-published-assets-v1",
                "licenses": [{"name": "GPL-2.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

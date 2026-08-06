#!/usr/bin/env python3
"""Build the self-contained, content-addressed DINOv2 Kaggle GPU package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zlib

import numpy as np


SCHEMA = "celltrack-alpha.dinov2-kaggle-gpu-assets-build.v1"
ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
GATE = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
DINO = ROOT / "reports/g02/screening/tiered_surface/branches/sa_dinov2"
SOURCE = Path("/private/tmp/g02_dinov2_source_7764ea0")
WEIGHT = Path(
    "/private/tmp/g02_dinov2_assets/"
    "sha256-b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9/"
    "dinov2_vits14_pretrain.pth"
)
WHEEL_ROOT = ROOT / "external/Tracking_Support/wheels"
TEMPLATE = HERE / "kaggle_kernel_template.py"

PINS = {
    "amendment": {
        "path": GATE / "PREREGISTRATION_v3_per_class_authenticated_weight.json",
        "sha256": "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    },
    "sealed_b0": {
        "path": GATE / "run_v3/B0_SEALED.json",
        "sha256": "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    },
    "population_manifest": {
        "path": GATE / "run_v1/population_manifest.json",
        "sha256": "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    },
    "supplier_contract": {
        "path": DINO / "R5_XY_BETA_STAGING_CONTRACT.json",
        "sha256": "c8e3df7739a3691b6ad9c93234ecc8db9696992c126de3ff343def38a93d5610",
    },
    "featurizer": {
        "path": DINO / "dinov2_xy_action_featurizer.py",
        "sha256": "f93a202a7e662859f92442a4a6932ad371af513c65627838f2ebfd7bfecb9b49",
    },
    "evaluator": {
        "path": GATE / "evaluate_cellect_separation.py",
        "sha256": "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
    },
}
WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_TREE = "2a27257b79b0633b027a21014bc9360e3c1b3f43"
CLASSES = ("delete", "occupied_swap")
EXPECTED_ACTIONS = {"delete": 1_345_497, "occupied_swap": 45_961}
WHEELS = (
    "donfig-0.8.1.post1-py3-none-any.whl",
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "zarr-3.2.1-py3-none-any.whl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins() -> dict[str, str]:
    observed = {}
    for name, pin in PINS.items():
        value = sha256(pin["path"])
        if value != pin["sha256"]:
            raise RuntimeError(f"frozen pin drift: {name}: {value}")
        observed[name] = value
    if WEIGHT.stat().st_size != 88_283_115 or sha256(WEIGHT) != WEIGHT_SHA256:
        raise RuntimeError("DINOv2 checkpoint drift")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    if revision != SOURCE_REVISION or tree != SOURCE_TREE:
        raise RuntimeError("official DINOv2 source identity drift")
    return observed


def deterministic_source_tar(destination: Path) -> None:
    with tarfile.open(destination, "w") as archive:
        for path in sorted(SOURCE.rglob("*")):
            relative = path.relative_to(SOURCE)
            if ".git" in relative.parts or "__pycache__" in relative.parts:
                continue
            info = archive.gettarinfo(str(path), arcname=str(Path("dinov2_source") / relative))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if info.isfile():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def request_unique_endpoints(path: Path) -> int:
    with np.load(path, allow_pickle=False) as loaded:
        rows = np.unique(
            np.concatenate(
                (
                    np.column_stack((loaded["source_t"], loaded["source_zyx"])),
                    np.column_stack((loaded["target_t"], loaded["target_zyx"])),
                )
            ),
            axis=0,
        )
    return int(len(rows))


def canonical_config_bytes(config: dict) -> bytes:
    return (
        json.dumps(config, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    dataset = output / "dataset"
    kernel = output / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir()
    observed_pins = verify_pins()

    shutil.copy2(WEIGHT, dataset / f"weight-{WEIGHT_SHA256}.pth")
    source_tar = dataset / "dinov2-source.tar.bin"
    deterministic_source_tar(source_tar)
    source_sha = sha256(source_tar)

    population = json.loads(PINS["population_manifest"]["path"].read_text())
    requests = []
    class_unique = {}
    for action_class in CLASSES:
        class_endpoints = 0
        action_count = 0
        for row in sorted(
            population["classes"][action_class]["videos"],
            key=lambda item: item["identity"],
        ):
            request = ROOT / row["request_path"]
            observed = sha256(request)
            if observed != row["request_sha256"]:
                raise RuntimeError(f"request drift: {request}")
            target = dataset / f"request-{observed}.npz"
            if target.exists() and sha256(target) != observed:
                raise RuntimeError("content-address collision")
            if not target.exists():
                shutil.copy2(request, target)
            endpoints = request_unique_endpoints(request)
            class_endpoints += endpoints
            action_count += int(row["count"])
            requests.append(
                {
                    "action_class": action_class,
                    "identity": row["identity"],
                    "count": int(row["count"]),
                    "sha256": observed,
                    "unique_endpoints": endpoints,
                }
            )
        if action_count != EXPECTED_ACTIONS[action_class]:
            raise RuntimeError(f"{action_class} action count drift")
        class_unique[action_class] = class_endpoints

    wheels = []
    for name in WHEELS:
        source = WHEEL_ROOT / name
        wheel_sha = sha256(source)
        target = dataset / f"wheel-{wheel_sha}.whl"
        shutil.copy2(source, target)
        wheels.append({"name": name, "sha256": wheel_sha, "bytes": source.stat().st_size})

    config = {
        "schema": "celltrack-alpha.dinov2-kaggle-gpu-embedded-config.v1",
        "supplier_id": "F0-DINOV2-VITS14-XY-V1",
        "source": {
            "sha256": source_sha,
            "bytes": source_tar.stat().st_size,
            "revision": SOURCE_REVISION,
            "tree": SOURCE_TREE,
        },
        "weight": {"sha256": WEIGHT_SHA256, "bytes": WEIGHT.stat().st_size},
        "wheels": wheels,
        "requests": requests,
        "class_unique_endpoint_counts": class_unique,
        "class_action_counts": EXPECTED_ACTIONS,
        "required_video_ids": sorted({row["identity"] for row in requests}),
        "pins": {
            name: pin["sha256"] for name, pin in PINS.items()
        },
        "partial_reuse": {
            "decision": "DISCARD_AND_RECOMPUTE_ALL",
            "quarantined_rows": 724_634,
            "reused_rows": 0,
            "reason": (
                "MPS partials have no complete class manifest and cannot be proven "
                "numerically identical to CUDA; no partial artifact is packaged."
            ),
        },
        "limits": {
            "hard_seconds": 18_000,
            "soft_seconds": 17_940,
            "projection_limit_seconds": 18_000,
        },
    }
    compressed = zlib.compress(canonical_config_bytes(config), level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    template = TEMPLATE.read_text(encoding="utf-8")
    token = "__EMBEDDED_CONFIG_ZLIB_BASE64__"
    if template.count(token) != 1:
        raise RuntimeError("kernel template config token drift")
    generated = template.replace(token, encoded)
    generated_path = kernel / "kaggle_dinov2_gpu.py"
    generated_path.write_text(generated, encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "onebitbrain/g02-dinov2-r5-gpu-bakeoff-v1",
                "title": "G02 DINOv2 R5 GPU bakeoff v1",
                "code_file": "kaggle_dinov2_gpu.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": ["onebitbrain/g02-dinov2-r5-gpu-assets-v1"],
                "competition_sources": ["biohub-cell-tracking-during-development"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "G02 DINOv2 R5 GPU assets v1",
                "id": "onebitbrain/g02-dinov2-r5-gpu-assets-v1",
                "licenses": [{"name": "Apache 2.0"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    dataset_files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(dataset.iterdir())
        if path.name != "dataset-metadata.json"
    ]
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "frozen_pins": observed_pins,
        "embedded_config_sha256": hashlib.sha256(canonical_config_bytes(config)).hexdigest(),
        "generated_kernel_sha256": sha256(generated_path),
        "dataset_file_count": len(dataset_files),
        "dataset_files": dataset_files,
        "request_file_count": len(requests),
        "action_counts": EXPECTED_ACTIONS,
        "class_unique_endpoint_counts": class_unique,
        "partial_reuse": config["partial_reuse"],
        "external_bundle_manifest_present": False,
        "kernel_self_contained": True,
    }
    (output / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

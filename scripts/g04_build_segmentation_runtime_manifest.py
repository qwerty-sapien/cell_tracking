#!/usr/bin/env python3
"""Build the byte-exact G-04 Cellpose CPython-3.12 runtime manifest.

The binary wheels and model are intentionally carried as ordinary files.  This
builder uses only the standard library, records every byte, and never downloads
or installs anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARRIER = (
    ROOT / "kaggle_kernel/g04-computed-segmentation-runtime-cp312-v1"
)
SCHEMA = "celltrack-alpha.g04-computed-segmentation-runtime-carrier.v1"

PYPI_ACQUIRED = {
    "cellpose-3.1.1.2-py3-none-any.whl",
    "fastremap-1.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "fill_voids-2.0.8-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "imagecodecs-2025.3.30-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "llvmlite-0.47.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "natsort-8.4.0-py3-none-any.whl",
    "opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "roifile-2024.9.15-py3-none-any.whl",
    "tifffile-2025.3.30-py3-none-any.whl",
}

EXPECTED_WHEELS = PYPI_ACQUIRED | {
    "deprecated-1.3.1-py2.py3-none-any.whl",
    "donfig-0.8.1.post1-py3-none-any.whl",
    "google_crc32c-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl",
    "imageio-2.37.3-py3-none-any.whl",
    "lazy_loader-0.5-py3-none-any.whl",
    "networkx-3.6.1-py3-none-any.whl",
    "numba-0.65.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "packaging-26.2-py3-none-any.whl",
    "pillow-12.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    "scikit_image-0.26.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "tqdm-4.68.3-py3-none-any.whl",
    "typing_extensions-4.15.0-py3-none-any.whl",
    "wrapt-2.2.2-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
    "zarr-3.2.1-py3-none-any.whl",
}

EXPECTED_NON_WHEELS = {
    "licenses/CELLPOSE_LICENSE.txt",
    "source/celltrack_alpha/__init__.py",
    "source/celltrack_alpha/bh_model.py",
    "source/g04_computed_segmentation.py",
    "weights/cyto3",
}

BASE_RUNTIME = {
    "evidence": {
        "path": "reports/g02/screening/tiered_surface/branches/"
        "g03_orthogonal_lanes_20260803/lane0_h005/runtime_probe_v1/"
        "H005_RUNTIME_PROBE.json",
        "sha256": "44616b10119bb8546e0d587b10f8feed1a4341a9bdaa9a9cb0d60ad6f9bb3c65",
    },
    "python": "3.12.13",
    "abi": "cp312",
    "numpy": "2.0.2",
    "scipy": "1.16.3",
    "torch_cpu_evidence": "2.10.0+cpu",
    "torch_gpu_required": "2.10.0+cu128",
    "policy": (
        "Preserve Kaggle base NumPy/SciPy/PyTorch; the carrier intentionally "
        "contains no wheel for those distributions."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_metadata(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        metadata_names = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"wheel has no unique METADATA member: {path.name}")
        lines = archive.read(metadata_names[0]).decode("utf-8", "replace").splitlines()
        license_members = sorted(
            name
            for name in members
            if ".dist-info/licenses/" in name
            or name.rsplit("/", 1)[-1].upper().startswith(("LICENSE", "COPYING"))
        )

    def first(prefix: str) -> str | None:
        return next(
            (line[len(prefix) :].strip() for line in lines if line.startswith(prefix)),
            None,
        )

    requirements = [
        line[len("Requires-Dist:") :].strip()
        for line in lines
        if line.startswith("Requires-Dist:") and "extra ==" not in line
    ]
    return {
        "distribution": first("Name:"),
        "version": first("Version:"),
        "requires_python": first("Requires-Python:"),
        "requirements": requirements,
        "license_expression": first("License-Expression:") or first("License:"),
        "embedded_license_members": license_members,
    }


def build(carrier: Path) -> dict[str, object]:
    wheels = sorted((carrier / "wheels").glob("*.whl"), key=lambda path: path.name)
    observed = {path.name for path in wheels}
    if observed != EXPECTED_WHEELS:
        raise RuntimeError(
            f"wheel inventory drift: missing={sorted(EXPECTED_WHEELS-observed)}, "
            f"extra={sorted(observed-EXPECTED_WHEELS)}"
        )
    for relative in EXPECTED_NON_WHEELS:
        if not (carrier / relative).is_file():
            raise RuntimeError(f"missing carrier file: {relative}")

    rows: list[dict[str, object]] = []
    for path in wheels:
        relative = path.relative_to(carrier).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "kind": "wheel",
                "origin": (
                    "pypi_pinned_binary_download_2026-08-03"
                    if path.name in PYPI_ACQUIRED
                    else "reused_authenticated_tracking_support_wheelhouse"
                ),
                **wheel_metadata(path),
            }
        )
    for relative in sorted(EXPECTED_NON_WHEELS):
        path = carrier / relative
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "kind": (
                    "model_weight"
                    if relative == "weights/cyto3"
                    else "software_license"
                    if relative.startswith("licenses/")
                    else "source"
                ),
            }
        )

    total_bytes = sum(int(row["bytes"]) for row in rows)
    return {
        "schema": SCHEMA,
        "status": "READY_AUTHENTICATED_OFFLINE_CP312",
        "dataset_id": "onebitbrain/g04-computed-segmentation-runtime-cp312-v1",
        "created_date": "2026-08-03",
        "network_at_install_or_inference": False,
        "provenance_label": "COMPUTED_SEGMENTATIONS",
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "base_runtime": BASE_RUNTIME,
        "install_contract": {
            "method": "python -m pip install --no-index --no-deps --target",
            "supported_abi": "cp312",
            "wheel_count": len(wheels),
            "third_party_imports_before_step0": False,
            "install_before_scientific_imports": True,
        },
        "cellpose": {
            "package_version": "3.1.1.2",
            "declared_numpy_constraint": ">=1.20.0,<2.1",
            "base_numpy_satisfies_declared_constraint": True,
            "model": "cyto3",
            "model_sha256": "2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c",
            "software_license": {
                "spdx": "BSD-3-Clause",
                "sha256": "fa908286fef07e1e8acd05adcdcd7bbb8c7ca4c02f563a4722dad28b94ce0b49",
                "scope": "Cellpose source/software",
            },
            "weight_terms_disclosure": (
                "No separate license text accompanied the downloaded cyto3 weight. "
                "Cellpose 3.1.1.2 states that all models except three named exceptions "
                "are trained on some CC-BY-NC data; cyto3 is not an exception. The "
                "software BSD license is not represented as a separate weight license."
            ),
        },
        "files": sorted(rows, key=lambda row: str(row["path"])),
        "file_count": len(rows),
        "total_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    carrier = args.carrier.resolve()
    value = build(carrier)
    output = carrier / "RUNTIME_CARRIER_MANIFEST.json"
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="ascii") != rendered:
            raise RuntimeError("runtime carrier manifest is absent or stale")
    else:
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(rendered, encoding="ascii")
        temporary.replace(output)
    print(json.dumps({"path": str(output), "sha256": sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

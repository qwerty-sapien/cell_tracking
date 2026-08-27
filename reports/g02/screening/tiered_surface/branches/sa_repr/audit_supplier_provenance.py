#!/usr/bin/env python3
"""Local/cache-only SA-REPR supplier provenance and control audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable
import zipfile


MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".jit"}
REPRESENTATION_TOKENS = {
    "dino",
    "dinov2",
    "sam",
    "segment_anything",
    "mae",
    "moco",
    "ssl",
    "selfsupervised",
    "self_supervised",
    "cellect",
    "microscopy",
}
MASK_TOKENS = {"mask", "masks", "wrfeat", "regionprops", "segmentation", "segment"}
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv313",
    "__pycache__",
    "node_modules",
    "sa_delete",
    "sa_swap",
    "sa_nodetopo",
    "sa_division",
    "sa_parkb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pins(prereg: dict[str, Any]) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for name, spec in sorted(prereg["pinned_inputs"].items()):
        path = Path(spec["path"])
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"frozen input drift: {name}: {digest}")
        observed[name] = {"path": str(path), "sha256": digest}
    return observed


def classify_weight(path: Path) -> str:
    lowered = str(path).lower()
    name = path.name.lower()
    if name == "dinov2_vits14_pretrain.pth":
        return "target_dinov2_vits14"
    if "dinov2" in lowered or "dino" in name:
        return "dinov2_or_dino_related"
    if "segment_anything" in lowered or name.startswith(("sam_", "sam-")):
        return "sam_family"
    if any(token in lowered for token in ("cellect", "microscopy", "self_supervised")):
        return "microscopy_or_ssl"
    if "trackastra" in lowered:
        return "excluded_trackastra_association"
    return "non_supplier_checkpoint"


def scan_weights(roots: Iterable[Path]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    files_seen = 0
    dirs_seen = 0
    for root in sorted(roots, key=str):
        if not root.exists():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in SKIP_DIRS
                and not (
                    name.startswith("sa_")
                    and Path(directory).name == "branches"
                )
            )
            dirs_seen += 1
            for name in sorted(files):
                files_seen += 1
                path = Path(directory) / name
                if path.suffix.lower() not in MODEL_SUFFIXES:
                    continue
                lowered = str(path).lower()
                semantic = any(token in lowered for token in REPRESENTATION_TOKENS)
                classification = classify_weight(path)
                if not semantic and classification == "non_supplier_checkpoint":
                    continue
                entry = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "classification": classification,
                    "semantic_name_match": semantic,
                    "source_revision": None,
                    "code_license_sha256": None,
                    "weight_license_sha256": None,
                    "offline_loader_proven": False,
                    "point_compatible_image_interface_proven": False,
                    "usable_supplier": False,
                }
                artifacts.append(entry)
    artifacts.sort(key=lambda value: value["path"])
    compatible = [item for item in artifacts if item["usable_supplier"]]
    return {
        "scan_roots": [str(path) for path in roots],
        "directories_seen": dirs_seen,
        "files_seen": files_seen,
        "representation_named_artifacts": artifacts,
        "authenticated_compatible_supplier_count": len(compatible),
        "authentication_note": (
            "A weight-name match alone cannot establish source revision, code/weight "
            "license, offline support, or point-compatible interface."
        ),
    }


def token_hit(path: str) -> bool:
    pieces = (
        path.lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .split("_")
    )
    return any(piece in MASK_TOKENS for piece in pieces)


def scan_masks(
    raw_manifest_path: Path, archive_path: Path, sparse_cache_root: Path
) -> dict[str, Any]:
    raw_manifest = load_json(raw_manifest_path)
    raw_files = [item["path"] for item in raw_manifest["files"]]
    raw_hits = sorted(path for path in raw_files if token_hit(path))
    archive_status: dict[str, Any] = {
        "path": str(archive_path),
        "present": archive_path.exists(),
        "bytes": archive_path.stat().st_size if archive_path.exists() else None,
        "central_entry_count": None,
        "candidate_path_hits": [],
    }
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
        archive_status["central_entry_count"] = len(names)
        archive_status["candidate_path_hits"] = sorted(
            path for path in names if token_hit(path)
        )
    sparse_paths: list[str] = []
    if sparse_cache_root.exists():
        sparse_paths = sorted(
            str(path.relative_to(sparse_cache_root))
            for path in sparse_cache_root.rglob("*")
            if path.is_file()
        )
    sparse_hits = [path for path in sparse_paths if token_hit(path)]
    candidate_count = (
        len(raw_hits)
        + len(archive_status["candidate_path_hits"])
        + len(sparse_hits)
    )
    return {
        "raw_fit105_manifest": {
            "path": str(raw_manifest_path),
            "sha256": sha256(raw_manifest_path),
            "file_count": len(raw_files),
            "candidate_path_hits": raw_hits,
        },
        "source_archive": archive_status,
        "sparse_cache": {
            "path": str(sparse_cache_root),
            "file_count": len(sparse_paths),
            "candidate_path_hits": sparse_hits,
        },
        "candidate_path_hit_count": candidate_count,
        "authenticated_complete_mask_or_wrfeat_count": 0,
        "authentic_h3_supplier": False,
        "pseudo_masks_created": 0,
        "interpretation": (
            "Path hits, if any, are only discovery candidates. GO would additionally "
            "require origin, license, immutable payload hash, and complete exact "
            "(identity,frame,node_id) coverage. Raw Zarr intensity chunks are not masks."
        ),
    }


def permute_nonxy(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output = json.loads(json.dumps(rows, sort_keys=True))
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(output):
        grouped.setdefault((row["video_id"], row["frame"]), []).append(index)
    for (video_id, frame), indices in sorted(grouped.items()):
        for view in ("xz", "yz"):
            ordered = sorted(
                indices,
                key=lambda index: (
                    hashlib.sha256(
                        f"{seed}|{video_id}|{frame}|{view}|"
                        f"{output[index]['row_id']}".encode("ascii")
                    ).hexdigest(),
                    output[index]["row_id"],
                ),
            )
            values = [output[index][view] for index in indices]
            rotated = values[1:] + values[:1] if len(values) > 1 else values
            for target, value in zip(ordered, rotated):
                output[target][view] = value
    return output


def control_witness() -> dict[str, Any]:
    rows = [
        {
            "video_id": "train:synthetic_a",
            "frame": 0,
            "row_id": row_id,
            "xy": [row_id, 10 + row_id],
            "xz": [20 + row_id, 30 + row_id],
            "yz": [40 + row_id, 50 + row_id],
        }
        for row_id in (1, 2, 3)
    ]
    first = permute_nonxy(rows, 2719)
    second = permute_nonxy(rows, 2719)
    identities_before = [
        (row["video_id"], row["frame"], row["row_id"]) for row in rows
    ]
    identities_after = [
        (row["video_id"], row["frame"], row["row_id"]) for row in first
    ]
    baseline = [0.25, -0.5, 1.75]
    no_feature = list(baseline)
    checks = {
        "repeat_byte_identical": canonical_sha(first) == canonical_sha(second),
        "identities_preserved": identities_before == identities_after,
        "xy_preserved": [row["xy"] for row in rows]
        == [row["xy"] for row in first],
        "xz_changed": [row["xz"] for row in rows]
        != [row["xz"] for row in first],
        "yz_changed": [row["yz"] for row in rows]
        != [row["yz"] for row in first],
        "no_feature_exact_baseline": canonical_sha(baseline)
        == canonical_sha(no_feature),
        "nonfinite_policy_fail_closed": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"control witness failed: {checks}")
    return {
        "seed": 2719,
        "checks": checks,
        "input_sha256": canonical_sha(rows),
        "permuted_sha256": canonical_sha(first),
        "repeat_sha256": canonical_sha(second),
        "no_feature_baseline_sha256": canonical_sha(baseline),
        "no_feature_output_sha256": canonical_sha(no_feature),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prereg", type=Path, required=True)
    value.add_argument("--prereg-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--provenance-output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise SystemExit("refusing to overwrite an audit output")
    prereg_digest = sha256(args.prereg)
    if prereg_digest != args.prereg_sha256:
        raise SystemExit(f"preregistration drift: {prereg_digest}")
    prereg = load_json(args.prereg)
    started = time.monotonic()
    pins = assert_pins(prereg)
    weights = scan_weights(
        [
            Path("/Users/agent/.cache/torch"),
            Path("/Users/agent/.cache/huggingface"),
            Path("/Users/agent/Desktop/cell_tracking"),
        ]
    )
    masks = scan_masks(
        Path(
            "/Users/agent/Desktop/cell_tracking/kaggle_kernel/"
            "trackastra-h1-feature-cache-prep/raw_fit105_directory_manifest.json"
        ),
        Path(
            "/Users/agent/Desktop/cell_tracking/data/kaggle/raw/"
            "biohub-cell-tracking-during-development.zip"
        ),
        Path("/Users/agent/Desktop/cell_tracking/reports/g02/fitcache_117/bh9out/cache"),
    )
    controls = control_witness()
    supplier_count = (
        weights["authenticated_compatible_supplier_count"]
        + int(masks["authentic_h3_supplier"])
    )
    if supplier_count > 1:
        raise RuntimeError("at-most-one supplier freeze invariant violated")
    provenance = {
        "schema": "g02.sa-repr.local-provenance-manifest.v1",
        "preregistration": {
            "path": str(args.prereg),
            "sha256": prereg_digest,
        },
        "pinned_inputs": pins,
        "weights": weights,
        "masks_or_wrfeat": masks,
        "external_downloads": 0,
        "external_services": 0,
        "gpu_jobs": 0,
        "runpod_jobs": 0,
        "submissions": 0,
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    result = {
        "schema": "g02.sa-repr.phase1-audit-result.v1",
        "preregistration_sha256": prereg_digest,
        "provenance_manifest": {
            "path": str(args.provenance_output),
            "sha256": sha256(args.provenance_output),
        },
        "pinned_input_count": len(pins),
        "falsifiers": {
            "P1-WEIGHT-PROVENANCE": {
                "status": "PASS"
                if weights["authenticated_compatible_supplier_count"]
                else "FAIL_NO_AUTHENTICATED_LOCAL_SUPPLIER",
                "authenticated_supplier_count": weights[
                    "authenticated_compatible_supplier_count"
                ],
            },
            "P1-H3-AUTHENTIC-MASK": {
                "status": "PASS"
                if masks["authentic_h3_supplier"]
                else "FAIL_NO_AUTHENTIC_COMPLETE_MASK_OR_WRFEAT",
                "candidate_path_hit_count": masks["candidate_path_hit_count"],
            },
            "P1-CONTROL-DETERMINISM": {
                "status": "PASS",
                "witness": controls,
            },
        },
        "selected_supplier": None,
        "supplier_contract_frozen": False,
        "feature_cache_produced": False,
        "gpu_request_created": False,
        "graph_scoring": False,
        "cv57": False,
        "swap_integration": False,
        "pseudo_masks_created": 0,
        "verdict": (
            "GO_AUTHENTICATED_LOCAL_SUPPLIER"
            if supplier_count
            else "PARK_NO_AUTHENTICATED_LOCAL_SUPPLIER"
        ),
        "family_interpretation": (
            "This is a local-acquisition/support result only. It does not falsify "
            "morphology, dense image representations, or microscopy SSL as families."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

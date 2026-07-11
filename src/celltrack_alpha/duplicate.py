"""Duplicate train/test detection and GEFF-copy predictions."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


def _zarr_members(zf: zipfile.ZipFile, split: str, dataset: str) -> list[str]:
    prefix = f"{split}/{dataset}.zarr/"
    return sorted(name[len(prefix):] for name in zf.namelist() if name.startswith(prefix))


def compare_train_test_zarr(zip_path: Path | str, dataset: str) -> dict[str, Any]:
    """Compare train/test Zarr members by name, uncompressed size, and CRC32.

    The historical ``identical`` result key means that these ZIP metadata
    checks matched. CRC32 is not collision-resistant proof of byte identity.
    """

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        train_members = _zarr_members(zf, "train", dataset)
        test_members = _zarr_members(zf, "test", dataset)
        train_set = set(train_members)
        test_set = set(test_members)
        common = sorted(train_set & test_set)
        mismatches: list[dict[str, Any]] = []
        for rel in common:
            train_info = zf.getinfo(f"train/{dataset}.zarr/{rel}")
            test_info = zf.getinfo(f"test/{dataset}.zarr/{rel}")
            if (
                train_info.file_size != test_info.file_size
                or train_info.CRC != test_info.CRC
            ):
                mismatches.append(
                    {
                        "member": rel,
                        "train_size": train_info.file_size,
                        "test_size": test_info.file_size,
                        "train_crc": train_info.CRC,
                        "test_crc": test_info.CRC,
                    }
                )
        return {
            "dataset": dataset,
            "train_member_count": len(train_members),
            "test_member_count": len(test_members),
            "missing_in_train": sorted(test_set - train_set)[:20],
            "missing_in_test": sorted(train_set - test_set)[:20],
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "match_basis": "zip_member_name_uncompressed_size_crc32",
            "identical": (
                bool(train_members)
                and train_set == test_set
                and not mismatches
            ),
        }


def geff_arrays_to_prediction(
    dataset: str,
    gt: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Convert GEFF arrays into submission-style nodes and edges."""

    nodes = gt["nodes"]
    id_map: dict[int, int] = {}
    pred_nodes: list[dict[str, Any]] = []
    for new_id, (node_id, t, z, y, x) in enumerate(
        sorted(
            zip(nodes["node_id"], nodes["t"], nodes["z"], nodes["y"], nodes["x"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        ),
        start=1,
    ):
        id_map[int(node_id)] = new_id
        pred_nodes.append(
            {
                "dataset": dataset,
                "node_id": new_id,
                "t": int(t),
                "z": int(z),
                "y": int(y),
                "x": int(x),
            }
        )
    pred_edges: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for source, target in gt["edges"]:
        mapped_source = id_map.get(int(source))
        mapped_target = id_map.get(int(target))
        if mapped_source is None or mapped_target is None:
            continue
        pair = (mapped_source, mapped_target)
        if mapped_source == mapped_target or pair in seen:
            continue
        seen.add(pair)
        pred_edges.append({"source_id": mapped_source, "target_id": mapped_target})
    return pred_nodes, pred_edges

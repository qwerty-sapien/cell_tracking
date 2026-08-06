from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/authenticate_synth_div_payload.py"
SPEC = importlib.util.spec_from_file_location("authenticate_synth_div_payload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auth
SPEC.loader.exec_module(auth)

EXPECTED = auth.ExpectedCounts(static=2, sequences=2, nodes=8, divisions=1)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_payload(root: Path, *, bad_edge_time: bool = False, progress: bool = True) -> Path:
    payload = root / "biohub_synthetic"
    static_dir = payload / "static"
    sequence_dir = payload / "sequences"
    static_dir.mkdir(parents=True)
    sequence_dir.mkdir()

    static_rows = []
    for index in range(2):
        volume = np.zeros((2, 3, 4), dtype=np.uint16)
        centroids = np.asarray([[0.5, 1.0, 2.0 + index * 0.25]], dtype=np.float32)
        path = static_dir / f"vol_{index:05d}.npz"
        np.savez(
            path,
            volume=volume,
            centroids=centroids,
            voxel_um=np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32),
        )
        static_rows.append(
            {
                "file": f"static/{path.name}",
                "n_cells": len(centroids),
                "shape": list(volume.shape),
                "bytes": path.stat().st_size,
            }
        )

    sequence_values = [
        (
            np.asarray(
                [
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 2, 1, 0],
                    [2, 1, 1, 1, 0],
                    [2, 1, 2, 1, 0],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [[0, 3], [0, 2], [1, 3], [2, 4]]
                if bad_edge_time
                else [[0, 1], [0, 2], [1, 3], [2, 4]],
                dtype=np.int32,
            ),
            np.asarray([0], dtype=np.int32),
        ),
        (
            np.asarray(
                [[0, 1, 1, 1, 7], [1, 1, 1, 2, 7], [2, 1, 1, 3, 7]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1], [1, 2]], dtype=np.int32),
            np.asarray([], dtype=np.int32),
        ),
    ]
    sequence_rows = []
    for index, (nodes, edges, divisions) in enumerate(sequence_values):
        volumes = np.zeros((3, 2, 2, 2), dtype=np.uint16)
        path = sequence_dir / f"seq_{index:04d}.npz"
        np.savez(
            path,
            volumes=volumes,
            nodes=nodes,
            edges=edges,
            divisions=divisions,
            voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
        )
        sequence_rows.append(
            {
                "file": f"sequences/{path.name}",
                "T": 3,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "n_divisions": len(divisions),
                "bytes": path.stat().st_size,
            }
        )

    manifest = {"static": static_rows, "sequences": sequence_rows}
    payload_bytes = sum(row["bytes"] for row in static_rows + sequence_rows)
    metadata = {
        "n_static": 2,
        "n_sequences": 2,
        "total_gb": payload_bytes / (1024**3),
        "seq_len": 3,
        "total_divisions": 1,
        "total_nodes": 8,
        "division_rate": 1 / 8,
        "voxel_native_um": [1.625, 0.40625, 0.40625],
        "pooling": "stride 4x in XY (vol[:, ::4, ::4]) -- identical to the official evaluator",
        "real_count_range": [20.0, 30.0],
        "shell_kind": "ellipsoid",
        "motion_calibration": {
            "median_step_um_per_frame": 1.86,
            "lag1_persistence": 0.30,
            "sister_separation_um": 7.24,
        },
        "note_division_rate": "Deliberately over-sampled fixture.",
        "config": {"SYNTH_TEST": "1"},
    }
    _json(payload / "metadata.json", metadata)
    _json(payload / "manifest.json", manifest)
    if progress:
        (payload / "progress.log").write_text("fixture complete\n", encoding="utf-8")
    return payload


def _authenticate(payload: Path) -> dict:
    return auth.authenticate_payload(
        payload, expected=EXPECTED, max_runtime_seconds=30
    )


def test_tiny_payload_authenticates_and_report_is_deterministic(tmp_path: Path) -> None:
    payload = _write_payload(tmp_path)
    first = _authenticate(payload)
    second = _authenticate(payload)

    assert first == second
    assert first["status"] == "AUTHENTICATED"
    assert first["expected"] == {
        "static": 2,
        "sequences": 2,
        "nodes": 8,
        "divisions": 1,
    }
    assert first["validated"]["sequence_nodes"] == 8
    assert first["validated"]["sequence_edges"] == 6
    assert first["validated"]["sequence_divisions"] == 1
    assert first["validated"]["image_arrays_materialized"] is False
    assert len(first["payload"]["files"]) == 7


def test_volume_members_are_never_materialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _write_payload(tmp_path)
    original = auth.np.lib.format.read_array
    loaded: list[str] = []

    def guarded_read_array(handle, *args, **kwargs):
        member_name = Path(handle.name).name
        loaded.append(member_name)
        if member_name in {"volume.npy", "volumes.npy"}:
            raise AssertionError("an image array was materialized")
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(auth.np.lib.format, "read_array", guarded_read_array)
    report = _authenticate(payload)

    assert report["status"] == "AUTHENTICATED"
    assert "volume.npy" not in loaded
    assert "volumes.npy" not in loaded
    assert set(loaded) == {
        "centroids.npy",
        "voxel_um.npy",
        "nodes.npy",
        "edges.npy",
        "divisions.npy",
        "voxel_um_pooled.npy",
    }


def test_tree_hash_is_root_independent_and_progress_log_is_optional(tmp_path: Path) -> None:
    first_payload = _write_payload(tmp_path / "first", progress=False)
    second_payload = tmp_path / "second" / "biohub_synthetic"
    second_payload.parent.mkdir(parents=True)
    shutil.copytree(first_payload, second_payload)

    first = _authenticate(first_payload)
    second = _authenticate(second_payload)
    assert first["payload"]["tree_sha256"] == second["payload"]["tree_sha256"]
    assert first["payload"]["files"] == second["payload"]["files"]


@pytest.mark.parametrize("drift", ["missing", "extra", "bytes"])
def test_inventory_and_manifest_drift_fail_closed(tmp_path: Path, drift: str) -> None:
    payload = _write_payload(tmp_path)
    if drift == "missing":
        (payload / "static/vol_00001.npz").unlink()
    elif drift == "extra":
        (payload / "unexpected.bin").write_bytes(b"drift")
    else:
        manifest = json.loads((payload / "manifest.json").read_text())
        manifest["static"][0]["bytes"] += 1
        _json(payload / "manifest.json", manifest)

    with pytest.raises(auth.AuthenticationError):
        _authenticate(payload)


def test_sequence_time_and_edge_invariant_drift_is_rejected(tmp_path: Path) -> None:
    payload = _write_payload(tmp_path, bad_edge_time=True)
    with pytest.raises(auth.AuthenticationError, match="advance exactly one frame"):
        _authenticate(payload)


def test_cli_writes_atomic_success_and_rejection_reports(tmp_path: Path) -> None:
    payload = _write_payload(tmp_path / "valid")
    output = tmp_path / "receipt.json"
    common = [
        "--payload",
        str(payload),
        "--output",
        str(output),
        "--expected-static-count",
        "2",
        "--expected-sequence-count",
        "2",
        "--expected-total-nodes",
        "8",
        "--expected-total-divisions",
        "1",
        "--max-runtime-seconds",
        "30",
    ]
    assert auth.main(common) == 0
    assert json.loads(output.read_text())["status"] == "AUTHENTICATED"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    (payload / "extra").write_bytes(b"x")
    assert auth.main(common) == 1
    rejected = json.loads(output.read_text())
    assert rejected["status"] == "REJECTED"
    assert "extra" in rejected["error"]
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_runtime_cap_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _write_payload(tmp_path)
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(auth.time, "monotonic", lambda: next(ticks))
    with pytest.raises(auth.AuthenticationError, match="runtime cap"):
        auth.authenticate_payload(payload, expected=EXPECTED, max_runtime_seconds=1)

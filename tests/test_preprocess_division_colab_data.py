from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from scripts.preprocess_division_colab_data import (
    FRAME_FILE,
    PreprocessConfig,
    PreprocessError,
    preprocess,
    verify_output,
)


pytestmark = pytest.mark.filterwarnings("ignore:Duplicate name.*:UserWarning")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_archive(path: Path) -> np.ndarray:
    raw = np.arange(3 * 2 * 8 * 8, dtype=np.uint16).reshape(3, 2, 8, 8)
    with ZipStore(path, mode="w") as store:
        image = zarr.open_group(store=store, path="train/toy.zarr", mode="w")
        image.create_array("0", data=raw, chunks=(1, 2, 8, 8))
        image.attrs["image_statistics"] = {
            "quantiles": {"0.001": 1.0, "0.999": 380.0}
        }
        image.attrs["multiscales"] = [
            {
                "axes": [
                    {"name": "t"},
                    {"name": "z"},
                    {"name": "y"},
                    {"name": "x"},
                ],
                "datasets": [
                    {"coordinateTransformations": [{"scale": [1.0, 2.0, 3.0, 4.0]}]}
                ],
            }
        ]
        graph = zarr.open_group(store=store, path="train/toy.geff", mode="w")
        arrays = {
            "nodes/ids": np.array([10, 11, 12], dtype=np.int64),
            "nodes/props/t/values": np.array([0, 1, 1], dtype=np.int64),
            "nodes/props/z/values": np.array([1, 1, 0], dtype=np.int64),
            "nodes/props/y/values": np.array([4, 5, 6], dtype=np.int64),
            "nodes/props/x/values": np.array([4, 6, 7], dtype=np.int64),
            "edges/ids": np.array([[10, 11], [10, 12]], dtype=np.int64),
        }
        for name, value in arrays.items():
            graph.create_array(name, data=value)
        # Presence of test data proves the builder's selected-split inventory is scoped.
        test = zarr.open_group(store=store, path="test/never-read.zarr", mode="w")
        test.create_array("0", data=np.zeros((1, 1, 1, 1), dtype=np.uint16))
    return raw


def test_stride_carrier_resume_and_tamper_detection(tmp_path: Path) -> None:
    archive = tmp_path / "competition.zip"
    raw = _build_archive(archive)
    output = tmp_path / "carrier"
    config = PreprocessConfig(
        expected_dataset_count=1,
        expected_frames_per_dataset=3,
        expected_raw_shape_zyx=(2, 8, 8),
        expected_output_shape_zyx=(2, 2, 2),
        expected_total_parent_events=1,
        expected_total_division_edges=2,
    )
    receipt = preprocess(
        archive=archive,
        output=output,
        expected_archive_sha256=_sha256(archive),
        expected_archive_bytes=archive.stat().st_size,
        config=config,
        resume=False,
    )
    assert receipt["status"] == "PASS"
    assert receipt["video_count"] == 1
    assert receipt["counts"]["parent_events"] == 1
    assert receipt["counts"]["division_edges"] == 2
    frames = np.load(output / "videos/toy" / FRAME_FILE, allow_pickle=False)
    np.testing.assert_array_equal(frames, raw[:, :, ::4, ::4])
    assert preprocess(
        archive=archive,
        output=output,
        expected_archive_sha256=_sha256(archive),
        expected_archive_bytes=archive.stat().st_size,
        config=config,
        resume=True,
    )["manifest_sha256"] == receipt["manifest_sha256"]
    frame_path = output / "videos/toy" / FRAME_FILE
    with frame_path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(PreprocessError, match="hash drifted"):
        verify_output(output)

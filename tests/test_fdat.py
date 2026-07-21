"""Contract tests for the direct compact-student GEFF/Zarr reader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

import numpy as np
import torch
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.fast_data import FastDataError, FastSource


class FastDataTests(unittest.TestCase):
    @staticmethod
    def _array(group: zarr.Group, name: str, value: np.ndarray) -> None:
        group.create_array(name, data=value, chunks=value.shape)

    def _fixture(self, root: Path, *, time_dtype: np.dtype = np.int64, graph_scale: float = 0.40625) -> tuple[Path, SimpleNamespace]:
        data_root = root / "train"
        data_root.mkdir()
        image = zarr.open_group(str(data_root / "clip.zarr"), mode="w", zarr_format=3)
        pixels = np.arange(2 * 8 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8, 8)
        self._array(image, "0", pixels)
        image.attrs.update(
            {
                "multiscales": [
                    {
                        "axes": [{"name": axis} for axis in ("t", "z", "y", "x")],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                                ]
                            }
                        ],
                    }
                ],
                "image_statistics": {"quantiles": {"0.001": 0.0, "0.999": float(pixels.max())}},
            }
        )
        geff = zarr.open_group(str(data_root / "clip.geff"), mode="w", zarr_format=3)
        nodes = geff.create_group("nodes")
        self._array(nodes, "ids", np.asarray([11, 12], dtype=np.int64))
        props = nodes.create_group("props")
        values = {
            "t": np.asarray([0, 1], dtype=time_dtype),
            "z": np.asarray([1.0, 2.0], dtype=np.float32),
            "y": np.asarray([4.0, 4.0], dtype=np.float32),
            "x": np.asarray([4.0, 4.0], dtype=np.float32),
        }
        for axis, value in values.items():
            prop = props.create_group(axis)
            self._array(prop, "values", value)
        edges = geff.create_group("edges")
        self._array(edges, "ids", np.asarray([[11, 12]], dtype=np.int64))
        geff.attrs["geff"] = {
            "directed": True,
            "geff_version": "1.1",
            "axes": [
                {"name": "t", "scale": 1.0},
                {"name": "z", "scale": 1.625},
                {"name": "y", "scale": graph_scale},
                {"name": "x", "scale": graph_scale},
            ],
        }
        return data_root, SimpleNamespace(group_id="fit-g", video_id="train:clip", stem="clip")

    def test_reads_only_the_two_frame_downsampled_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root, record = self._fixture(Path(temporary))
            source = FastSource(data_root, (1, 4, 4))
            batches = list(source.batches([record], completed_windows=0, deadline=time.monotonic() + 3.0))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(tuple(batch["frames"].shape), (1, 2, 1, 8, 2, 2))
        self.assertEqual(batch["frames"].dtype, torch.float32)
        self.assertEqual(batch["edges"], [{"source_id": 11, "target_id": 12}])
        self.assertEqual(batch["nodes"][0], {"id": 11, "batch": 0, "time": 0, "z": 1.0, "y": 1.0, "x": 1.0})
        self.assertEqual(batch["nodes"][1], {"id": 12, "batch": 0, "time": 1, "z": 2.0, "y": 1.0, "x": 1.0})
        self.assertEqual(batch["voxel_size_um"], (1.625, 1.625, 1.625))
        self.assertEqual(source.window_stats["usable"], 1)

    def test_rejects_noninteger_time_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root, record = self._fixture(Path(temporary), time_dtype=np.float32)
            source = FastSource(data_root, (1, 4, 4))
            with self.assertRaisesRegex(FastDataError, "node table"):
                list(source.batches([record], completed_windows=0, deadline=time.monotonic() + 3.0))

    def test_rejects_physical_scale_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root, record = self._fixture(Path(temporary), graph_scale=0.5)
            source = FastSource(data_root, (1, 4, 4))
            with self.assertRaisesRegex(FastDataError, "metadata decode"):
                list(source.batches([record], completed_windows=0, deadline=time.monotonic() + 3.0))


if __name__ == "__main__":
    unittest.main()

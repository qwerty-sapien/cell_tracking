from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernel/gpuinfer.py"
PACKER = ROOT / "scripts/packfast.py"


def _load():
    spec = importlib.util.spec_from_file_location("gpuinfer", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpuinfer = _load()


def _make_image(root: Path, chunks: tuple[int, int, int, int] = (1, 2, 4, 4), uppercase: bool = False):
    import numpy as np
    import zarr

    image = root / "input/test/tiny.zarr"
    group = zarr.open_group(str(image), mode="w")
    array = np.arange(3 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 4, 4)
    group.create_array("0", data=array, chunks=chunks)
    names = ("T", "Z", "Y", "X") if uppercase else ("t", "z", "y", "x")
    group.attrs.update({
        "image_statistics": {"quantiles": {"0.001": 0, "0.999": int(array.max())}},
        "multiscales": [{
            "axes": [{"name": name} for name in names],
            "datasets": [{"path": "0", "coordinateTransformations": [{
                "type": "scale", "scale": [1, 1.625, 0.40625, 0.40625],
            }]}],
        }],
    })
    return image, array


class FastInferenceTests(unittest.TestCase):
    def test_forbidden_benchmark_inputs(self) -> None:
        cases = {
            "/x/train/a.geff/nodes/ids": "GEFF",
            "/x/annotations.json": "annotation",
            "/x/official_scorer.py": "scorer",
            "/x/dataset_splits.json": "split",
        }
        for path, reason in cases.items():
            self.assertEqual(gpuinfer.forbidden_read(path), reason)
        self.assertIsNone(gpuinfer.forbidden_read("/x/train/a.zarr/0/c/1/0/0/0"))

    def test_gpu_shards_cover_without_duplicates(self) -> None:
        jobs = [
            {"identity": f"train:d{i:03d}", "shape": [100, 64, 256, 256], "dataset": f"d{i:03d}",
             "split": "train", "chunks": 100}
            for i in range(203)
        ]
        shards = gpuinfer.shard_jobs(jobs, 2)
        identities = [job["identity"] for shard in shards for job in shard]
        self.assertEqual(len(identities), 203)
        self.assertEqual(len(set(identities)), 203)
        self.assertLessEqual(abs(len(shards[0]) - len(shards[1])), 1)

    def test_sparse_link_is_bounded_and_one_to_one(self) -> None:
        previous = [
            {"node_id": index + 1, "z": 0, "y": index * 2, "x": 0}
            for index in range(8)
        ]
        current = [
            {"node_id": index + 101, "z": 0, "y": index * 2, "x": 0, "backward_um": [0, 0, 0]}
            for index in range(8)
        ]
        edges, candidates = gpuinfer.sparse_link(previous, current, (1, 1, 1), 3.0, 2, 0.2, 16)
        self.assertLessEqual(candidates, 16)
        self.assertEqual(len({source for source, _ in edges}), len(edges))
        self.assertEqual(len({target for _, target in edges}), len(edges))
        self.assertIn("cKDTree", SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("cdist", SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("linear_sum_assignment", SOURCE.read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "needs scipy")
    def test_crowded_512_link_queries_only_bounded_neighbors(self) -> None:
        import scipy.spatial

        previous = [{"node_id": index + 1, "z": 0, "y": 256, "x": 256} for index in range(512)]
        current = [
            {"node_id": index + 1001, "z": 0, "y": 256, "x": 256, "backward_um": [0, 0, 0]}
            for index in range(512)
        ]
        real_tree = scipy.spatial.cKDTree
        observed = {"queries": 0, "evaluations": 0}

        class CountingTree:
            def __init__(self, points):
                self.tree = real_tree(points)

            def query(self, points, *, k, distance_upper_bound, workers):
                observed["queries"] += 1
                observed["evaluations"] += len(points) * k
                return self.tree.query(points, k=k, distance_upper_bound=distance_upper_bound, workers=workers)

        with mock.patch.object(scipy.spatial, "cKDTree", CountingTree):
            edges, candidates = gpuinfer.sparse_link(previous, current, (1, 1, 1), 8.0, 4, 0.2, 2048)
        self.assertEqual(observed, {"queries": 1, "evaluations": 2048})
        self.assertEqual(candidates, 2048)
        self.assertLessEqual(len(edges), 512)

    def test_uppercase_ngff_axes_retain_scale(self) -> None:
        attrs = {
            "multiscales": [{
                "axes": [
                    {"name": "T", "type": "time"},
                    {"name": "Z", "type": "space"},
                    {"name": "Y", "type": "space"},
                    {"name": "X", "type": "space"},
                ],
                "datasets": [{
                    "coordinateTransformations": [{
                        "type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625],
                    }],
                }],
            }],
        }
        self.assertEqual(gpuinfer._parse_scale(attrs), (1.625, 0.40625, 0.40625))

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"), "needs torch and numpy")
    def test_below_threshold_output_emits_one_deterministic_fallback(self) -> None:
        import numpy as np
        import torch

        output = torch.zeros((4, 2, 2, 3), dtype=torch.float32)
        output[0].fill_(-10.0)
        output[0, 0, 1, 1] = -9.0
        output[0, 1, 0, 0] = -9.0
        output[1:4, 0, 1, 1] = torch.tensor([1.25, -2.5, 3.75])
        first, next_id = gpuinfer._extract_nodes(torch, np, output, 7, 41, (3, 3, 3), 0.99, 8)
        second, second_next_id = gpuinfer._extract_nodes(torch, np, output, 7, 41, (3, 3, 3), 0.99, 8)
        self.assertEqual(first, second)
        self.assertEqual((next_id, second_next_id), (42, 42))
        self.assertEqual(len(first), 1)
        self.assertEqual({key: first[0][key] for key in ("node_id", "t", "z", "y", "x", "fallback")},
                         {"node_id": 41, "t": 7, "z": 0, "y": 4, "x": 4, "fallback": True})
        self.assertEqual(first[0]["backward_um"], [1.25, -2.5, 3.75])
        self.assertAlmostEqual(first[0]["score"], float(torch.sigmoid(torch.tensor(-9.0))), places=10)
        self.assertEqual(gpuinfer._count_frame_fallback(first), 1)

    def test_deadline_crossing_removes_success_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stats = Path(temporary) / "stats.json"
            with mock.patch.object(gpuinfer.time, "time", side_effect=[0.0, 2.0]):
                with self.assertRaisesRegex(gpuinfer.FastError, "deadline"):
                    gpuinfer._atomic_json(stats, {"ok": True}, 1.0)
            self.assertFalse(stats.exists())

    def test_worker_wait_timeout_terminates_worker(self) -> None:
        process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
        try:
            with self.assertRaisesRegex(gpuinfer.FastError, "deadline"):
                gpuinfer._wait_workers([process], time.time() + 0.1)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_compact_package_under_source_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fastsrc.zip"
            completed = subprocess.run(
                [sys.executable, str(PACKER), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertLess(report["source_bytes"], 1_000_000)
            self.assertLess(report["compressed_bytes"], 1_000_000)
            self.assertEqual(report["members"], ["__main__.py", "fastreg.json"])

    def test_empty_frame_postcondition_still_rejects_misuse(self) -> None:
        with self.assertRaisesRegex(gpuinfer.FastError, "emitted no nodes"):
            gpuinfer._require_frame_nodes([], "test:tiny", 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("zarr"), "needs local Zarr runtime")
    def test_alternate_spatial_chunk_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, _array = _make_image(root, chunks=(1, 1, 4, 4))
            job = {"identity": "test:tiny", "split": "test", "dataset": "tiny",
                   "path": str(image), "shape": [3, 2, 4, 4], "chunks": 6}
            with self.assertRaisesRegex(gpuinfer.FastError, "exactly one frame by one spatial volume"):
                gpuinfer._open_image(job, None)

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("zarr"), "needs local Zarr runtime")
    def test_tiny_image_only_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _image, _array = _make_image(root, uppercase=True)
            registry = root / "fastreg.json"
            registry.write_text(json.dumps({
                "artifact": "fast_image_registry", "schema_version": 1,
                "counts": {"videos": 1, "train": 0, "test": 1, "image_chunks": 3},
                "records": [{"split": "test", "dataset": "tiny", "shape": [3, 2, 4, 4], "chunks": 3}],
            }), encoding="ascii")
            output = root / "out"
            completed = subprocess.run(
                [sys.executable, str(SOURCE), "--mode", "benchmark", "--root", str(root / "input"),
                 "--registry", str(registry), "--out", str(output), "--dry-run", "--mock",
                 "--gpus", "cpu", "--max-datasets", "1", "--max-frames", "2",
                 "--max-points", "8", "--neighbor-cap", "2", "--max-candidates", "16"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["coverage"]["completed"], 1)
            self.assertEqual(stats["coverage"]["logical_chunks_read"], 2)
            self.assertTrue(stats["coverage"]["frame_chunk_geometry"])
            self.assertEqual(stats["coverage"]["fallback_frames"], 1)
            self.assertEqual(stats["coverage"]["datasets_with_fallback"], 1)
            self.assertEqual(stats["workers"][0]["datasets"][0]["fallback_frames"], 1)
            self.assertFalse(stats["memory_contract"]["dense_nxn"])
            self.assertEqual(stats["image_only"]["denied_read_attempts"], 0)
            self.assertTrue((output / "benchmark.csv").is_file())
            with (output / "benchmark.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["dataset"] for row in rows}, {"test:tiny"})
            self.assertEqual({int(row["t"]) for row in rows if row["row_type"] == "node"}, {0, 1})

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("zarr"), "needs local Zarr runtime")
    def test_expired_run_writes_no_success_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_image(root)
            registry = root / "fastreg.json"
            registry.write_text(json.dumps({
                "artifact": "fast_image_registry", "schema_version": 1,
                "counts": {"videos": 1, "train": 0, "test": 1, "image_chunks": 3},
                "records": [{"split": "test", "dataset": "tiny", "shape": [3, 2, 4, 4], "chunks": 3}],
            }), encoding="ascii")
            output = root / "out"
            completed = subprocess.run(
                [sys.executable, str(SOURCE), "--mode", "benchmark", "--root", str(root / "input"),
                 "--registry", str(registry), "--out", str(output), "--dry-run", "--mock",
                 "--gpus", "cpu", "--max-frames", "2", "--max-points", "8",
                 "--neighbor-cap", "2", "--max-candidates", "16", "--deadline-min", "0.000001"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((output / "stats.json").exists())


if __name__ == "__main__":
    unittest.main()

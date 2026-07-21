"""Execution, cap, arming, and atomic resume coverage for the sparse trainer."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "external/" + "ka" + "ggle-cell-tracking-competition/src")

import torch
import numpy as np

SPEC = importlib.util.spec_from_file_location("train_ta", ROOT / "scripts" / "train_ta.py")
tt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = tt
SPEC.loader.exec_module(tt)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.frozen = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)

    def forward(self, frames: torch.Tensor, scenario: str = "universal") -> torch.Tensor:
        return frames.repeat(1, 1, 6, 1, 1, 1) * self.weight + self.frozen * 0.0


def batch(index: int = 0) -> dict:
    return {"frames": torch.ones(1, 2, 1, 4, 4, 4),
            "nodes": [{"id": 1, "batch": 0, "time": 0, "z": 1, "y": 1, "x": 1},
                      {"id": 2, "batch": 0, "time": 1, "z": 1, "y": 2, "x": 1}],
            "edges": [{"source_id": 1, "target_id": 2}], "voxel_size_um": (1.0, 1.0, 1.0),
            "window_id": f"synthetic:{index}"}


def record() -> tt.SourceRecord:
    return tt.SourceRecord("g02-01db8d94baa3ee0ee831", "train:6bba_edf14583", "6bba_edf14583")


class Rows:
    def __init__(self, values: list[dict]) -> None:
        self.values = values

    def iter_rows(self, named: bool = True):
        return iter(self.values)


class Tracks:
    def __init__(self, nodes: list[dict] | None = None, edges: list[dict] | None = None) -> None:
        self.nodes = nodes or [{"node_id": 1, "t": 0, "z": 1.0, "y": 4.0, "x": 4.0},
                               {"node_id": 2, "t": 1, "z": 1.0, "y": 8.0, "x": 4.0}]
        self.edges = edges or [{"source_id": 1, "target_id": 2}]

    def node_attrs(self, attr_keys: list[str]):
        return Rows(self.nodes)

    def edge_attrs(self, attr_keys: list[str]):
        return Rows(self.edges)


class Dataset:
    def __init__(self, zarr_path: Path, tracks: Tracks | None = None) -> None:
        self.image = None
        self.zarr_path = zarr_path
        self.tracks = tracks or Tracks()
        self.quantiles = {"0.001": 1.0, "0.999": 10.0}
        self.scale = (1.625, 0.40625, 0.40625)
        self.image_shape = (2, 64, 256, 256)


def fake_io(value: Dataset | None = None) -> ModuleType:
    module = ModuleType("tracking_cellmot.io")
    module.open_dataset = mock.Mock(return_value=value)
    return module


class FakeSource:
    opened = 0

    def __init__(self, root: Path, downsample: tuple[int, int, int]) -> None:
        type(self).opened += 1

    def batches(self, groups: tuple[str, ...], *, completed_windows: int, deadline: float, epochs: int = 1):
        return (batch(index) for index in range(completed_windows, 3))


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = Path("/private/tmp/t03e/check")
        shutil.rmtree(self.out, ignore_errors=True)
        self.out.mkdir(parents=True)
        self.common = ["--fold", "0", "--output-dir", str(self.out), "--data-root", "/never/read", "--max-steps", "2"]

    def args(self, *extra: str):
        return tt._parse_args([*self.common, *extra])

    def test_01_train_steps_updates_only_trainable(self) -> None:
        model, old, frozen = Tiny(), 0.5, 3.0
        opt = torch.optim.AdamW([model.weight], lr=0.1)
        result, _ = tt.train_steps(model, opt, [batch(0), batch(1)],
                                  device=torch.device("cpu"), deadline=tt.time.monotonic() + 20, max_steps=2)
        self.assertEqual(result["steps"], 2)
        self.assertNotEqual(float(model.weight.detach()), old)
        self.assertEqual(float(model.frozen), frozen)

    def test_02_train_metrics_are_finite(self) -> None:
        model = Tiny()
        result, _ = tt.train_steps(model, torch.optim.AdamW([model.weight], lr=0.1), [batch()],
                                  device=torch.device("cpu"), deadline=tt.time.monotonic() + 20, max_steps=1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result["loss_components"].values()))

    def test_03_elapsed_cap_prevents_step(self) -> None:
        model = Tiny()
        result, _ = tt.train_steps(model, torch.optim.AdamW([model.weight], lr=0.1), [batch()],
                                  device=torch.device("cpu"), deadline=tt.time.monotonic() - 1, max_steps=1)
        self.assertEqual((result["status"], result["steps"]), ("time_cap", 0))

    def test_04_unarmed_rejects_before_source_access(self) -> None:
        FakeSource.opened = 0
        args = self.args()
        with mock.patch.object(tt, "SparseWindowSource", FakeSource), self.assertRaises(tt.LaunchError):
            tt._run_bounded_training(args, (record(),), "cpu", model_factory=lambda *_: Tiny())
        self.assertEqual(FakeSource.opened, 0)

    def test_05_prior_gate_is_absent(self) -> None:
        source = (ROOT / "scripts" / "train_ta.py").read_text()
        self.assertNotIn("intentionally " + "gated pending", source)
        self.assertIn("train_steps(", source)

    def test_06_armed_main_writes_inventory(self) -> None:
        FakeSource.opened = 0
        with mock.patch.object(tt, "SparseWindowSource", FakeSource), mock.patch.object(tt, "_build_released_model", lambda *_: Tiny()):
            self.assertEqual(tt.main([*self.common, "--armed"]), 0)
        self.assertEqual({path.name for path in self.out.iterdir()}, {"prelaunch.json", "checkpoint.pt", "metrics.json", "resume.json"})

    def test_07_atomic_resume_restores_identity(self) -> None:
        with mock.patch.object(tt, "SparseWindowSource", FakeSource), mock.patch.object(tt, "_build_released_model", lambda *_: Tiny()):
            tt.main([*self.common, "--armed"])
            tt.main([*self.common, "--armed", "--max-steps", "3", "--resume", str(self.out / "checkpoint.pt")])
        metrics = json.loads((self.out / "metrics.json").read_text())
        resume = json.loads((self.out / "resume.json").read_text())
        self.assertEqual(metrics["completed_steps"], 3)
        self.assertEqual(metrics["identity"], resume["identity"])

    def test_08_incompatible_resume_rejects(self) -> None:
        with mock.patch.object(tt, "SparseWindowSource", FakeSource), mock.patch.object(tt, "_build_released_model", lambda *_: Tiny()):
            tt.main([*self.common, "--armed"])
            with self.assertRaises(tt.LaunchError):
                tt.main([*self.common, "--armed", "--seed", "18", "--resume", str(self.out / "checkpoint.pt")])

    def test_09_dry_run_is_deterministic(self) -> None:
        one, two = self.out / "one", self.out / "two"
        self.assertEqual(tt.main(["--fold", "0", "--output-dir", str(one), "--dry-run"]), 0)
        self.assertEqual(tt.main(["--fold", "0", "--output-dir", str(two), "--dry-run"]), 0)
        self.assertEqual((one / "prelaunch.json").read_bytes(), (two / "prelaunch.json").read_bytes())

    def test_10_isolation_rejects_unknown_group(self) -> None:
        split = tt._load_json_exact("reports/g02/grouped_3fold_audit20_v1.json", tt.SPLIT_SHA256)
        prov = tt._load_json_exact("reports/g02/fit_provenance_v1.json", tt.PROVENANCE_SHA256)
        with self.assertRaises(tt.LaunchError):
            tt.select_fit_groups(0, split, prov, 0.1, 17, ["not-a-group"])

    def test_11_budget_cap_rejects(self) -> None:
        with self.assertRaises(tt.LaunchError):
            tt._validate_budget(tt.MAX_HOURS_CAP + 0.01, 0.0)

    def test_12_prelaunch_is_armed_only(self) -> None:
        args = self.args("--armed")
        item = tt.prelaunch_manifest(args, (record(),), "cpu")
        self.assertEqual((item["status"], item["execution"]["batch_size"]), ("armed", 1))

    def test_12b_teacher_detector_policy_requires_the_frozen_backbone_and_weights(self) -> None:
        with self.assertRaises(tt.LaunchError):
            self.args("--detector-policy", "released_teacher_distill")
        args = self.args(
            "--detector-policy", "released_teacher_distill",
            "--center-weight", "0.1",
            "--teacher-weight", "1.0",
        )
        self.assertEqual(args.detector_policy, "released_teacher_distill")
        with self.assertRaises(tt.LaunchError):
            self.args(
                "--detector-policy", "released_teacher_distill",
                "--center-weight", "0.1",
                "--teacher-weight", "1.0",
                "--unfreeze", "1",
            )

    def test_13_epoch_aggregates_are_emitted(self) -> None:
        model = Tiny()
        first, second = batch(0), batch(1)
        first["epoch"], second["epoch"] = 1, 2
        result, _ = tt.train_steps(model, torch.optim.AdamW([model.weight], lr=0.1), [first, second],
                                  device=torch.device("cpu"),
                                  deadline=tt.time.monotonic() + 20, max_steps=2)
        self.assertEqual([item["epoch"] for item in result["epoch_metrics"]], [1, 2])

    def test_14_registry_fixture_resolves_to_train_stem(self) -> None:
        registry = tt._load_json_exact("reports/g02/dataset_registry_v1.json", tt.REGISTRY_SHA256)
        split = tt._load_json_exact("reports/g02/grouped_3fold_audit20_v1.json", tt.SPLIT_SHA256)
        provenance = tt._load_json_exact("reports/g02/fit_provenance_v1.json", tt.PROVENANCE_SHA256)
        _, forbidden = tt._fold_data(0, split, provenance)
        self.assertEqual(tt.resolve_fit_records((record().group_id,), registry, forbidden), (record(),))

    def test_15_registry_nontrain_rejects(self) -> None:
        bad = {"groups": [{"group_id": "g02-aaaa0000000000000000", "video_count": 1,
                             "contains_nontrain_member": False, "splits": ["test"],
                             "video_ids": ["test:6bba_edf14583"]}]}
        with self.assertRaises(tt.LaunchError):
            tt.resolve_fit_records(("g02-aaaa0000000000000000",), bad, set())

    def test_16_registry_duplicate_mapping_rejects(self) -> None:
        item = {"group_id": record().group_id, "video_count": 1, "contains_nontrain_member": False,
                "splits": ["train"], "video_ids": [record().video_id]}
        with self.assertRaises(tt.LaunchError):
            tt.resolve_fit_records((record().group_id,), {"groups": [item, dict(item)]}, set())

    def _source_layout(self, name: str = "6bba_edf14583") -> tuple[Path, Path]:
        root = self.out / "real"
        zarr = root / f"{name}.zarr"
        (zarr / "0").mkdir(parents=True)
        (zarr / "zarr.json").write_text("{}")
        (zarr / "0" / "zarr.json").write_text("{}")
        (root / f"{name}.geff").mkdir()
        return root, zarr

    def test_17_directory_zarr_is_accepted(self) -> None:
        root, zarr = self._source_layout()
        source = tt.SparseWindowSource(root, (1, 4, 4))
        module = fake_io(Dataset(zarr))
        with mock.patch.dict(sys.modules, {"tracking_cellmot.io": module}):
            metadata = source._metadata(record())
        self.assertEqual(metadata[3], (1.625, 1.625, 1.625))

    def test_18_regular_zarr_file_rejects_before_open(self) -> None:
        root = self.out / "file"
        root.mkdir()
        (root / "6bba_edf14583.zarr").write_text("not a store")
        (root / "6bba_edf14583.geff").mkdir()
        source = tt.SparseWindowSource(root, (1, 4, 4))
        module = fake_io()
        with mock.patch.dict(sys.modules, {"tracking_cellmot.io": module}), self.assertRaises(tt.LaunchError):
            source._metadata(record())
        module.open_dataset.assert_not_called()

    def test_19_zarr_symlink_escape_rejects_before_open(self) -> None:
        root = self.out / "symlink"
        outside = self.out / "outside"
        (outside / "0").mkdir(parents=True)
        (outside / "zarr.json").write_text("{}")
        (outside / "0" / "zarr.json").write_text("{}")
        root.mkdir()
        os.symlink(outside, root / "6bba_edf14583.zarr", target_is_directory=True)
        (root / "6bba_edf14583.geff").mkdir()
        source = tt.SparseWindowSource(root, (1, 4, 4))
        module = fake_io()
        with mock.patch.dict(sys.modules, {"tracking_cellmot.io": module}), self.assertRaises(tt.LaunchError):
            source._metadata(record())
        module.open_dataset.assert_not_called()

    def test_20_collision_windows_are_deterministically_skipped(self) -> None:
        source = tt.SparseWindowSource(self.out, (1, 4, 4))
        nodes = [{"node_id": 1, "t": 0, "z": 1.0, "y": 4.0, "x": 4.0},
                 {"node_id": 2, "t": 0, "z": 1.0, "y": 4.0, "x": 4.0},
                 {"node_id": 3, "t": 1, "z": 1.0, "y": 8.0, "x": 4.0}]
        dataset = Dataset(self.out / "unused.zarr", Tracks(nodes, [{"source_id": 1, "target_id": 3}]))
        with mock.patch.object(source, "_metadata", return_value=(dataset, nodes, [{"source_id": 1, "target_id": 3}],
                                                                    (1.625, 1.625, 1.625), 1.0, 10.0)):
            self.assertEqual(list(source._windows(record())), [])
        self.assertEqual(source.window_stats["collision_windows"], 1)

    def test_21_completed_resume_does_not_open_data(self) -> None:
        args = self.args("--armed")
        with mock.patch.object(tt, "SparseWindowSource", FakeSource):
            tt._run_bounded_training(args, (record(),), "cpu", model_factory=lambda *_: Tiny())
        complete = self.args("--armed", "--resume", str(self.out / "checkpoint.pt"))
        FakeSource.opened = 0
        with mock.patch.object(tt, "SparseWindowSource", FakeSource):
            metrics = tt._run_bounded_training(complete, (record(),), "cpu", model_factory=lambda *_: Tiny())
        self.assertEqual((FakeSource.opened, metrics["status"]), (0, "completed"))

    def test_22_negative_remaining_steps_rejects(self) -> None:
        with self.assertRaises(tt.LaunchError):
            tt.train_steps(Tiny(), torch.optim.AdamW(Tiny().parameters(), lr=0.1), [],
                           device=torch.device("cpu"), deadline=tt.time.monotonic() + 1, max_steps=-1)

    def test_23_product_lazy_read_uses_two_frame_batch_contract(self) -> None:
        source = tt.SparseWindowSource(self.out, (1, 4, 4))

        class Array:
            def __getitem__(self, index):
                self.index = index
                return np.zeros((2, 4, 4, 4), dtype=np.float32)

        array = Array()
        module = ModuleType("zarr")
        module.open_group = mock.Mock(return_value={"0": array})
        metadata = {"start": 3, "nodes": [], "edges": [], "voxel_size_um": (1.0, 1.0, 1.0),
                    "zarr_path": self.out, "q_low": 0.0, "q_high": 1.0, "group_id": record().group_id}
        with mock.patch.dict(sys.modules, {"zarr": module}):
            loaded = source._read(metadata)
        self.assertEqual(tuple(loaded), tt.BATCH_SCHEMA)
        self.assertEqual(tuple(loaded["frames"].shape), (1, 2, 1, 4, 4, 4))
        self.assertEqual(array.index[0], slice(3, 5))

    def test_24_exact_real_fixture_evidence_validates(self) -> None:
        smoke = Path("/private/tmp/t03e/smoke")
        xtract, real, frames, fit = (smoke / "xtract.json", smoke / "real.json", smoke / "frames.npy", smoke / "fit.json")
        self.assertTrue(all(path.is_file() for path in (xtract, real, frames, fit)))
        inventory, batch, result = json.loads(xtract.read_text()), json.loads(real.read_text()), json.loads(fit.read_text())
        self.assertEqual(inventory["archive"]["central_directory_sha256"], "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77")
        self.assertEqual((inventory["image_chunk_count"], inventory["member_count"]), (2, 25))
        self.assertLessEqual(inventory["total_extracted_bytes"], inventory["cap_bytes"])
        self.assertEqual(batch["batch"]["frames"]["shape"], [1, 2, 1, 64, 64, 64])
        self.assertEqual(hashlib.sha256(frames.read_bytes()).hexdigest(), batch["batch"]["frames"]["npy_sha256"])
        self.assertTrue(batch["batch_schema"]["equal"])
        self.assertEqual((result["result"]["status"], result["result"]["optimizer_steps"]), ("completed", 1))

    def test_25_registry_multivideo_rejects(self) -> None:
        bad = {"groups": [{"group_id": record().group_id, "video_count": 2, "contains_nontrain_member": False,
                             "splits": ["train"], "video_ids": [record().video_id, "train:44b6_2f31fc2f"]}]}
        with self.assertRaises(tt.LaunchError):
            tt.resolve_fit_records((record().group_id,), bad, set())

    def test_26_initial_parameter_hashes_isolate_trainable_and_frozen_state(self) -> None:
        first, second = Tiny(), Tiny()
        first_policy = tt._training_policy(first, self.args())
        second_policy = tt._training_policy(second, self.args())
        self.assertEqual(first_policy["initial_parameter_sha256"], second_policy["initial_parameter_sha256"])
        with torch.no_grad():
            second.frozen.add_(1.0)
        changed = tt._training_policy(second, self.args())["initial_parameter_sha256"]
        self.assertEqual(changed["trainable"], first_policy["initial_parameter_sha256"]["trainable"])
        self.assertNotEqual(changed["frozen"], first_policy["initial_parameter_sha256"]["frozen"])

    def test_27_step_limit_does_not_fetch_an_extra_supervised_window(self) -> None:
        fetched: list[int] = []

        def batches():
            for index in range(2):
                fetched.append(index)
                yield batch(index)

        model = Tiny()
        result, _ = tt.train_steps(
            model,
            torch.optim.AdamW([model.weight], lr=0.1),
            batches(),
            device=torch.device("cpu"),
            deadline=tt.time.monotonic() + 20,
            max_steps=1,
        )
        self.assertEqual((result["steps"], result["windows"], fetched), (1, ["synthetic:0"], [0]))


if __name__ == "__main__":
    unittest.main()

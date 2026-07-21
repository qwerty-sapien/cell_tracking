"""Contracts for fold-safe alpha-pseudolink temporal adaptation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "kaggle-cell-tracking-competition" / "src"))

import torch
from tracking_cellmot.models.temporal_unet import TemporalUNet3D

from celltrack_alpha import temporal_pretrain as tp
from celltrack_alpha.t_adapter import load_released_core
import train_ta as supervised

SPEC = importlib.util.spec_from_file_location("pretrain_temporal", ROOT / "scripts" / "pretrain_temporal.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

WEIGHTS = Path("/private/tmp/ckpt-v1/edge_predictor_best.pth")
CONFIG = Path("/private/tmp/ckpt-v1/config.json")
ALPHA_ROOT = ROOT / "kaggle_kernel" / "bh9o7" / "bh9out"
RECORD = supervised.SourceRecord(
    "g02-f2dd7ac03d54b37c10d2", "train:6bba_e5e44988", "6bba_e5e44988"
)
EXACT_IDENTITY = "3ec77fa0d7f8a6c9db383b2b871250ae887ca8ef0a1c91ef824d7583b11b8e68"
MANIFEST_HASHES = {
    "split_sha256": supervised.SPLIT_SHA256,
    "provenance_sha256": supervised.PROVENANCE_SHA256,
    "registry_sha256": supervised.REGISTRY_SHA256,
}


def released_backbone() -> TemporalUNet3D:
    backbone = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
    load_released_core(backbone, WEIGHTS, CONFIG)
    return backbone


def dummy_artifact(frame_count: int = 5) -> tp.AlphaPseudoLinkArtifact:
    link = tp.PseudoLink(1, 2, (1, 1, 1), (1, 1, 2), 0.99, 1.0)
    motif = tp.DivisionMotif(1, (2, 3), (1, 1, 1), ((1, 1, 2), (1, 2, 1)), 0.98)
    return tp.AlphaPseudoLinkArtifact(
        group_id=RECORD.group_id,
        video_id=RECORD.video_id,
        stem=RECORD.stem,
        exact_identity_sha256=EXACT_IDENTITY,
        cache_sha256="1" * 64,
        fragment_sha256="2" * 64,
        status_sha256="3" * 64,
        feature_sha256="4" * 64,
        transaction_sha256="5" * 64,
        run_contract_sha256="6" * 64,
        alpha_weight_sha256="7" * 64,
        windows=(tp.PseudoLinkWindow(0, (link,), (motif,)),),
        summary={
            "frame_count": frame_count,
            "candidate_count": 3,
            "selected_edge_count": 2,
            "candidate_backed_edge_count": 2,
            "eligible_migration_count": 1,
            "retained_migration_count": 1,
            "candidate_backed_division_count": 1,
            "retained_division_count": 1,
            "retained_window_count": 1,
            "maximum_links_per_window": 256,
            "maximum_divisions_per_window": 16,
        },
    )


def args(output_dir: Path, *, armed: bool = False) -> object:
    mode = "--armed" if armed else "--dry-run"
    return runner._parse_args(
        [
            "--fold",
            "0",
            "--output-dir",
            str(output_dir),
            "--alpha-artifact-root",
            str(ALPHA_ROOT),
            "--max-steps",
            "2",
            "--checkpoint-every-steps",
            "1",
            mode,
        ]
    )


class _ZeroAnchor:
    def release_anchor_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)


def sparse_inputs() -> dict[str, torch.Tensor]:
    return {
        "link_batch_indices": torch.zeros(1, dtype=torch.long),
        "source_zyx": torch.tensor([[0, 0, 0]], dtype=torch.long),
        "target_zyx": torch.tensor([[0, 0, 1]], dtype=torch.long),
        "link_weights": torch.ones(1),
    }


class TemporalPretrainTest(unittest.TestCase):
    def test_01_only_released_temporal_parameters_are_trainable(self) -> None:
        backbone = released_backbone()
        before = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
        model = tp.ReleasedTemporalStudent(backbone)
        names = {name for name, value in model.student.named_parameters() if value.requires_grad}
        self.assertEqual(names, set(tp.temporal_parameter_names(model.student)))
        self.assertEqual(sum(value.numel() for value in model.student.parameters() if value.requires_grad), 83_072)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.student.state_dict().items()))
        self.assertEqual(float(model.release_anchor_loss().detach()), 0.0)
        self.assertTrue(
            all(
                not module.training
                for module in model.student.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
        )

    def test_02_migration_ranks_link_above_two_stale_coordinates(self) -> None:
        good = torch.zeros(1, 2, 2, 1, 1, 2, requires_grad=True)
        with torch.no_grad():
            good[0, 0, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
            good[0, 1, :, 0, 0, 1] = torch.tensor([1.0, 0.0])
            good[0, 1, :, 0, 0, 0] = torch.tensor([0.0, 1.0])
            good[0, 0, :, 0, 0, 1] = torch.tensor([0.0, 1.0])
        bad = good.detach().clone()
        bad[0, 1, :, 0, 0, 1] = torch.tensor([0.0, 1.0])
        good_loss = tp.sparse_migration_loss(_ZeroAnchor(), good, **sparse_inputs(), margin=0.2, release_anchor_weight=0.0)
        bad_loss = tp.sparse_migration_loss(_ZeroAnchor(), bad, **sparse_inputs(), margin=0.2, release_anchor_weight=0.0)
        self.assertEqual(float(good_loss["migration"].detach()), 0.0)
        self.assertGreater(
            float(bad_loss["migration"].detach()), float(good_loss["migration"].detach())
        )
        self.assertEqual(float(good_loss["positive_similarity"].detach()), 1.0)
        self.assertEqual(float(good_loss["stale_similarity"].detach()), 0.0)
        with self.assertRaises(tp.TemporalPretrainError):
            tp.sparse_migration_loss(
                _ZeroAnchor(),
                good,
                **{**sparse_inputs(), "target_zyx": torch.tensor([[0, 0, 0]])},
                margin=0.2,
                release_anchor_weight=0.0,
            )

    def test_03_division_motif_is_optional_and_accepts_empty_windows(self) -> None:
        features = torch.zeros(1, 2, 2, 1, 2, 3, requires_grad=True)
        with torch.no_grad():
            features[0, 0, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
            features[0, 1, :, 0, 0, 1] = torch.tensor([1.0, 0.0])
            features[0, 1, :, 0, 0, 0] = torch.tensor([0.0, 1.0])
            features[0, 0, :, 0, 0, 1] = torch.tensor([0.0, 1.0])
            features[0, 1, :, 0, 1, 0] = torch.tensor([1.0, 0.0])
        empty = tp.sparse_migration_loss(
            _ZeroAnchor(),
            features,
            **sparse_inputs(),
            margin=0.1,
            release_anchor_weight=0.0,
            division_batch_indices=torch.empty(0, dtype=torch.long),
            division_zyx=torch.empty((0, 3, 3), dtype=torch.long),
            division_weights=torch.empty(0),
            division_weight=1.0,
        )
        self.assertEqual(
            (float(empty["division"].detach()), float(empty["division_count"].detach())), (0.0, 0.0)
        )
        enabled = tp.sparse_migration_loss(
            _ZeroAnchor(),
            features,
            **sparse_inputs(),
            margin=0.1,
            release_anchor_weight=0.0,
            division_batch_indices=torch.zeros(1, dtype=torch.long),
            division_zyx=torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 0]]]),
            division_weights=torch.ones(1),
            division_weight=1.0,
            sibling_cosine_ceiling=0.5,
        )
        self.assertAlmostEqual(float(enabled["division"].detach()), 0.5, places=6)
        self.assertEqual(float(enabled["division_count"].detach()), 1.0)

    def test_04_real_alpha_bundle_is_sparse_fit_source_and_digest_bound(self) -> None:
        artifact = tp.load_alpha_pseudo_link_artifact(
            ALPHA_ROOT,
            group_id=RECORD.group_id,
            video_id=RECORD.video_id,
            stem=RECORD.stem,
            expected_exact_identity_sha256=EXACT_IDENTITY,
            min_migration_um=0.5,
            min_link_confidence=0.95,
            max_links_per_window=256,
            max_divisions_per_window=16,
        )
        self.assertEqual(artifact.summary["retained_migration_count"], 4_447)
        self.assertEqual(artifact.summary["retained_window_count"], 99)
        self.assertEqual(artifact.summary["retained_division_count"], 16)
        self.assertTrue(all(link.source_zyx != link.target_zyx for window in artifact.windows for link in window.links))
        with self.assertRaises(tp.TemporalPretrainError):
            tp.load_alpha_pseudo_link_artifact(
                ALPHA_ROOT,
                group_id=RECORD.group_id,
                video_id=RECORD.video_id,
                stem=RECORD.stem,
                expected_exact_identity_sha256="0" * 64,
                min_migration_um=0.5,
                min_link_confidence=0.95,
                max_links_per_window=256,
                max_divisions_per_window=16,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slug = "train__6bba_e5e44988"
            for directory, suffix in (
                ("cache", ".bhc"),
                ("cache", ".json"),
                ("cache", ".intent.json"),
                ("fragments", ".csv"),
                ("features", ".json"),
                ("status", ".json"),
                ("transactions", ".json"),
            ):
                (root / directory).mkdir(exist_ok=True)
                shutil.copy2(ALPHA_ROOT / directory / f"{slug}{suffix}", root / directory / f"{slug}{suffix}")
            with (root / "fragments" / f"{slug}.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(tp.TemporalPretrainError, "digest mismatch"):
                tp.load_alpha_pseudo_link_artifact(
                    root,
                    group_id=RECORD.group_id,
                    video_id=RECORD.video_id,
                    stem=RECORD.stem,
                    expected_exact_identity_sha256=EXACT_IDENTITY,
                    min_migration_um=0.5,
                    min_link_confidence=0.95,
                    max_links_per_window=256,
                    max_divisions_per_window=16,
                )

    def test_05_checkpoint_is_fold_artifact_and_frozen_core_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parsed = args(output)
            artifact = dummy_artifact()
            identity = runner._identity(parsed, (RECORD,), (artifact,))
            backbone = released_backbone()
            path = output / "checkpoint.pt"
            payload = {
                "schema": tp.PRETRAIN_SCHEMA,
                "identity": identity,
                "backbone": backbone.state_dict(),
                "optimizer": {},
                "progress": {"completed_steps": 1, "completed_windows": 1, "elapsed_seconds": 1.0, "status": "running"},
                "metrics": {"loss": {"total": 1.0}},
            }
            torch.save(payload, path)
            restored = tp.load_temporal_pretrain_checkpoint(
                released_backbone(),
                path,
                expected_fold=0,
                expected_manifest_hashes=MANIFEST_HASHES,
                allowed_fit_groups={RECORD.group_id},
                allowed_fit_video_ids={RECORD.video_id},
            )
            self.assertEqual((restored.fold, restored.completed_steps, restored.source_groups), (0, 1, (RECORD.group_id,)))
            changed = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
            spatial = next(name for name, value in changed.items() if name.startswith("encoder_blocks.") and value.is_floating_point())
            changed[spatial] += 1.0
            payload["backbone"] = changed
            bad = output / "changed-spatial.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(tp.TemporalPretrainError, "frozen released spatial core"):
                tp.load_temporal_pretrain_checkpoint(
                    released_backbone(),
                    bad,
                    expected_fold=0,
                    expected_manifest_hashes=MANIFEST_HASHES,
                    allowed_fit_groups={RECORD.group_id},
                    allowed_fit_video_ids={RECORD.video_id},
                )
            payload["schema"] = "celltrack-alpha.temporal-image-pretrain.v1"
            old = output / "old-objective.pt"
            torch.save(payload, old)
            with self.assertRaisesRegex(tp.TemporalPretrainError, "schema mismatch"):
                tp.load_temporal_pretrain_checkpoint(
                    released_backbone(), old, expected_fold=0, expected_manifest_hashes=MANIFEST_HASHES
                )

    def test_06_resume_restores_only_the_student_and_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parsed = args(output)
            identity = runner._identity(parsed, (RECORD,), (dummy_artifact(),))
            model = tp.ReleasedTemporalStudent(released_backbone())
            optimizer = torch.optim.AdamW(
                (value for value in model.student.parameters() if value.requires_grad), lr=parsed.lr
            )
            state = {name: value.detach().clone() for name, value in model.student.state_dict().items()}
            temporal = next(name for name in state if name.startswith("temporal_blocks.") and state[name].is_floating_point())
            state[temporal] += 1.0
            path = output / "resume.pt"
            torch.save(
                {
                    "schema": tp.PRETRAIN_SCHEMA,
                    "identity": identity,
                    "backbone": state,
                    "optimizer": optimizer.state_dict(),
                    "progress": {"completed_steps": 1, "completed_windows": 1, "elapsed_seconds": 2.0, "status": "running"},
                    "metrics": {},
                },
                path,
            )
            self.assertEqual(runner._load_resume(path, identity, model, optimizer), (1, 2.0))
            self.assertTrue(torch.equal(model.student.state_dict()[temporal], state[temporal]))
            changed_identity = dict(identity)
            changed_identity["alpha_artifacts"] = [dict(identity["alpha_artifacts"][0], cache_sha256="8" * 64)]
            with self.assertRaisesRegex(supervised.LaunchError, "resume identity"):
                runner._load_resume(path, changed_identity, model, optimizer)
            bad_payload = torch.load(path, map_location="cpu", weights_only=True)
            spatial = next(
                name
                for name, value in bad_payload["backbone"].items()
                if name.startswith("encoder_blocks.") and value.is_floating_point()
            )
            bad_payload["backbone"][spatial] += 1.0
            bad_path = output / "bad-spatial-resume.pt"
            torch.save(bad_payload, bad_path)
            with self.assertRaisesRegex(supervised.LaunchError, "frozen released spatial core"):
                runner._load_resume(bad_path, identity, model, optimizer)

    def test_07_image_source_opens_no_tracks_and_only_declared_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zarr_path = root / f"{RECORD.stem}.zarr"
            (zarr_path / "0").mkdir(parents=True)
            (zarr_path / "zarr.json").write_text("{}")
            (zarr_path / "0" / "zarr.json").write_text("{}")
            dataset = SimpleNamespace(
                zarr_path=zarr_path,
                image=None,
                tracks=None,
                image_shape=(5, 8, 8, 8),
                quantiles={"0.001": 1.0, "0.999": 10.0},
            )
            module = ModuleType("tracking_cellmot.io")
            module.open_dataset = mock.Mock(return_value=dataset)
            source = runner.ImageWindowSource(root, (1, 4, 4))
            with mock.patch.dict(sys.modules, {"tracking_cellmot.io": module}):
                source._metadata(dummy_artifact(frame_count=5))
            self.assertFalse(module.open_dataset.call_args.kwargs["require_tracks"])
            self.assertFalse(module.open_dataset.call_args.kwargs["load_image"])
            text = (ROOT / "scripts" / "pretrain_temporal.py").read_text().lower()
            self.assertNotIn("require_tracks=true", text)
            self.assertNotIn("node_attrs(", text)
            self.assertNotIn(".geff", text)

    def test_08_prelaunch_is_offline_sparse_and_nonpromotional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = runner.prelaunch_manifest(
                args(Path(temporary)), (RECORD,), (dummy_artifact(),), "cpu"
            )
        self.assertTrue(manifest["source_isolation"]["fit_only"])
        self.assertFalse(manifest["source_isolation"]["labels_used"])
        self.assertEqual(manifest["memory_contract"]["dense_nxn_allocations"], 0)
        self.assertEqual(manifest["memory_contract"]["objective_extra_work"], "O((K+D)*C)")
        self.assertEqual(manifest["memory_contract"]["teacher_forwards"], 0)
        self.assertFalse(manifest["tracking"]["network_calls"])
        self.assertFalse(manifest["tracking"]["kaggle_calls"])
        self.assertFalse(manifest["evidence_policy"]["improvement_claimed"])
        self.assertIn("equal-budget", manifest["evidence_policy"]["promotion_requires"])

    def test_09_outputs_are_resumable_local_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parsed = args(output)
            identity = runner._identity(parsed, (RECORD,), (dummy_artifact(),))
            model = tp.ReleasedTemporalStudent(released_backbone())
            optimizer = torch.optim.AdamW(
                (value for value in model.student.parameters() if value.requires_grad), lr=parsed.lr
            )
            result = {
                "status": "running",
                "steps": 1,
                "windows": ["w:0"],
                "per_step": [],
                "loss": {key: 1.0 for key in runner.LOSS_KEYS},
            }
            metrics = runner._write_outputs(
                output,
                identity,
                model,
                optimizer,
                result,
                prior_steps=0,
                prior_elapsed=0.0,
                started=time.monotonic(),
                tracker={"backend": "local", "network_calls": False},
                source_stats={"read": 1},
            )
            expected = {"checkpoint.pt", "metrics.json", "resume.json", "wandb-summary.json", "wandb-artifact.json"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            artifact = json.loads((output / "wandb-artifact.json").read_text())
            self.assertEqual((artifact["type"], artifact["aliases"]), ("model", ["latest"]))
            self.assertFalse(artifact["metadata"]["improvement_claimed"])
            self.assertEqual(metrics["progress"]["completed_steps"], 1)

    def test_10_released_backbone_executes_sparse_temporal_backward(self) -> None:
        torch.manual_seed(17)
        model = tp.ReleasedTemporalStudent(released_backbone())
        frames = torch.rand((1, 2, 1, 8, 16, 16), dtype=torch.float32)
        features = model(frames)
        losses = tp.sparse_migration_loss(
            model,
            features,
            link_batch_indices=torch.zeros(1, dtype=torch.long),
            source_zyx=torch.tensor([[2, 3, 3]]),
            target_zyx=torch.tensor([[3, 4, 4]]),
            link_weights=torch.ones(1),
            margin=2.0,
            release_anchor_weight=1e-3,
        )
        losses["total"].backward()
        self.assertEqual(tuple(features.shape), (1, 2, 32, 8, 16, 16))
        self.assertTrue(any(parameter.grad is not None for parameter in model.student.parameters() if parameter.requires_grad))
        self.assertTrue(all(parameter.grad is None for parameter in model.student.parameters() if not parameter.requires_grad))

    def test_11_sources_have_no_pair_matrix_or_rejected_teacher_objective(self) -> None:
        source = (ROOT / "src" / "celltrack_alpha" / "temporal_pretrain.py").read_text().lower()
        runner_source = (ROOT / "scripts" / "pretrain_temporal.py").read_text().lower()
        for forbidden in (
            "torch.cdist",
            "pair_matrix =",
            "transition_matrix =",
            "np.zeros((n, n",
            "masked_temporal_loss",
            "complementary_block_mask",
            "released-feature-masked-temporal-distillation",
        ):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, runner_source)

    def test_12_step_limit_does_not_fetch_an_extra_window(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.student = torch.nn.Module()
                self.student.weight = torch.nn.Parameter(torch.tensor([1.0, 0.5]))

            def forward(self, frames: torch.Tensor) -> torch.Tensor:
                first = frames[:, :, 0] * self.student.weight[0]
                second = frames[:, :, 0].square() * self.student.weight[1]
                return torch.stack((first, second), dim=2)

            def release_anchor_loss(self) -> torch.Tensor:
                return self.student.weight.square().mean() * 0.0

        fetched: list[int] = []

        def batches() -> object:
            for index in range(2):
                fetched.append(index)
                frames = torch.tensor([[[[[[1.0, 2.0]]]], [[[[3.0, 4.0]]]]]])
                yield {
                    "frames": frames,
                    **sparse_inputs(),
                    "division_batch_indices": torch.empty(0, dtype=torch.long),
                    "division_zyx": torch.empty((0, 3, 3), dtype=torch.long),
                    "division_weights": torch.empty(0),
                    "window_id": f"window:{index}",
                }

        model = TinyModel()
        optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-3)
        result = runner.train_steps(
            model,
            optimizer,
            batches(),
            device=torch.device("cpu"),
            deadline=time.monotonic() + 10.0,
            max_steps=1,
            global_step=0,
            migration_margin=0.2,
            release_anchor_weight=0.0,
            division_weight=0.0,
            sibling_cosine_ceiling=0.9,
        )
        self.assertEqual((result["steps"], result["windows"], fetched), (1, ["window:0"], [0]))


if __name__ == "__main__":
    unittest.main()

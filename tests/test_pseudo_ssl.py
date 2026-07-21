"""Focused contracts for the bounded G-02 pseudo-label SSL milestone."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from celltrack_alpha import pseudo_ssl as ssl
import train_pseudo_ssl as runner


EXPECTED_HASHES = {
    "split_sha256": runner.SPLIT_SHA256,
    "provenance_sha256": runner.PROVENANCE_SHA256,
    "registry_sha256": runner.REGISTRY_SHA256,
}


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.spatial = nn.Linear(2, 2, bias=False)


class _CompactModel(nn.Module):
    def __init__(self, *, disallowed: bool = False) -> None:
        super().__init__()
        self.detect_head = nn.Linear(2, 1, bias=False)
        self.aux_head = nn.Linear(2, 2, bias=False)
        self.backbone = _Backbone()
        self.other = nn.Linear(2, 2, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.detect_head, self.aux_head, self.backbone.temporal_blocks):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if disallowed:
            self.other.weight.requires_grad = True


def _all_open_curricula() -> dict[str, ssl.CurriculumSpec]:
    return {
        "detection": ssl.CurriculumSpec(0.5, 0.5, 0.5, 0.5, 1.0, 1.0),
        "continuation": ssl.CurriculumSpec(0.5, 0.5, 0.5, 0.5, 1.0, 1.0),
        "division": ssl.CurriculumSpec(0.9, 0.9, 0.1, 0.1, 1.0, 1.0),
    }


class PseudoSSLTest(unittest.TestCase):
    def test_01_ema_contains_only_permitted_trainable_parameters(self) -> None:
        model = _CompactModel()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
        ema = ssl.EMATeacherState(model, decay=0.5)
        self.assertEqual(set(ema.shadow), set(ssl.permitted_trainable_names(model)))
        self.assertNotIn("backbone.spatial.weight", ema.shadow)
        self.assertNotIn("other.weight", ema.shadow)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.fill_(3.0)
        ema.update(model)
        self.assertTrue(all(torch.equal(value, torch.full_like(value, 2.0)) for value in ema.shadow.values()))
        frozen_before = model.other.weight.detach().clone()
        with ema.average_parameters(model):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(
                all(
                    torch.equal(dict(model.named_parameters())[name], torch.full_like(value, 2.0))
                    for name, value in ema.shadow.items()
                )
            )
            self.assertTrue(torch.equal(model.other.weight, frozen_before))
        self.assertTrue(
            all(
                torch.equal(parameter, torch.full_like(parameter, 3.0))
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        state = ema.state_dict()
        self.assertEqual(set(state["shadow"]), set(ema.names))
        with self.assertRaisesRegex(ssl.PseudoSSLError, "disallowed"):
            ssl.EMATeacherState(_CompactModel(disallowed=True))

    def test_02_curriculum_relaxes_coverage_and_reconciles_rejections(self) -> None:
        policy = ssl.AdaptiveThresholdCurriculum(ramp_steps=10)
        confidence = torch.tensor([0.99, 0.96, 0.90, 0.80, 0.60])
        uncertainty = torch.tensor([0.01, 0.04, 0.10, 0.20, 0.40])
        early = policy.select("detection", confidence, uncertainty, step=0)
        late = policy.select("detection", confidence, uncertainty, step=10)
        self.assertGreaterEqual(late.audit.accepted, early.audit.accepted)
        self.assertEqual(
            late.audit.total,
            late.audit.accepted + sum(late.audit.rejected.values()),
        )
        closed = policy.select("division", confidence[:2], uncertainty[:2], step=10)
        self.assertEqual(closed.audit.rejected, {"gate_closed": 2})
        ledger = ssl.AcceptanceLedger()
        ledger.add("train:44b6_00000000", late.audit)
        self.assertFalse(ledger.as_dict()["dense_nxn_association"])

    def test_03_fit_scope_rejects_held_out_audit_and_wrong_identity(self) -> None:
        scope = ssl.FitOnlyScope.from_files(
            0,
            ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
            ROOT / "reports/g02/fit_provenance_v1.json",
            ROOT / "reports/g02/dataset_registry_v1.json",
            expected_hashes=EXPECTED_HASHES,
        )
        fit_video = sorted(scope.fit_video_ids)[0]
        fit_group = scope.video_to_group[fit_video]
        scope.validate(fit_group, fit_video, scope.exact_image_identity[fit_video])
        with self.assertRaisesRegex(ssl.PseudoSSLError, "fit-only"):
            scope.validate(fit_group, fit_video, "0" * 64)
        held_video = sorted(scope.forbidden_video_ids)[0]
        with self.assertRaisesRegex(ssl.PseudoSSLError, "fit-only"):
            scope.validate(scope.video_to_group[held_video], held_video, scope.exact_image_identity[held_video])
        provenance = ssl.BatchProvenance(fit_group, fit_video, scope.exact_image_identity[fit_video], "a" * 64)
        provenance.validate(scope)

    def test_04_flip_consistency_moves_points_and_physical_motion(self) -> None:
        graph = ssl.SparsePseudoGraph(
            video_id="train:44b6_00000000",
            shape_zyx=(1, 1, 3),
            voxel_size_um=(1.0, 1.0, 1.0),
            nodes=(
                ssl.PseudoNode(0, 0, 0, (0, 0, 0), 0.99, 0.01, 0.01),
                ssl.PseudoNode(1, 0, 1, (0, 0, 1), 0.99, 0.01, 0.01),
            ),
            edges=(ssl.PseudoEdge(0, 1, 0.99, 0.01, 0.0),),
            divisions=(),
            neighbor_cap=2,
            edge_radius_um=8.0,
        )
        selection = ssl.SelectedPseudoGraph(
            graph,
            torch.tensor([True, True]),
            torch.tensor([True]),
            torch.empty(0, dtype=torch.bool),
            {},
        )
        transform = ssl.FlipTransform(graph.shape_zyx, (False, False, True))
        good = torch.zeros(1, 2, 6, 1, 1, 3, requires_grad=True)
        with torch.no_grad():
            good[0, 0, 0, 0, 0, 2] = 6.0
            good[0, 1, 0, 0, 0, 1] = 6.0
            good[0, 1, 3, 0, 0, 1] = 1.0
        bad = good.detach().clone().requires_grad_(True)
        with torch.no_grad():
            bad[0, 1, 3, 0, 0, 1] = -1.0
        good_loss = ssl.sparse_pseudo_losses(good, selection, transform)
        bad_loss = ssl.sparse_pseudo_losses(bad, selection, transform)
        self.assertLess(float(good_loss["motion"].detach()), float(bad_loss["motion"].detach()))
        self.assertLess(float(good_loss["total"].detach()), float(bad_loss["total"].detach()))
        good_loss["total"].backward()
        self.assertIsNotNone(good.grad)

    def test_05_candidate_extraction_uses_physical_kdtree_not_dense_pairs(self) -> None:
        output = torch.full((1, 2, 6, 1, 3, 5), -10.0)
        output[0, 0, 0, 0, 1, 1] = 8.0
        output[0, 1, 0, 0, 1, 2] = 8.0
        output[0, 1, 1:4, 0, 1, 2] = torch.tensor([0.0, 0.0, -0.40625])
        with mock.patch.object(torch, "cdist", side_effect=AssertionError("dense pair allocation")):
            graph = ssl.extract_sparse_pseudo_graph(
                output,
                video_id="train:44b6_00000000",
                voxel_size_um=(1.625, 0.40625, 0.40625),
                neighbor_cap=2,
                edge_radius_um=8.0,
            )
        ssl.validate_physical_graph(graph)
        self.assertFalse(graph.full_frame_dense_matrix)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertLessEqual(len(graph.edges), len([node for node in graph.nodes if node.time == 1]) * graph.neighbor_cap)

    def test_06_divisions_require_gate_and_two_selected_sparse_edges(self) -> None:
        graph = ssl.SparsePseudoGraph(
            video_id="train:6bba_00000000",
            shape_zyx=(1, 3, 3),
            voxel_size_um=(1.0, 1.0, 1.0),
            nodes=(
                ssl.PseudoNode(0, 0, 0, (0, 1, 1), 0.99, 0.01, 0.99),
                ssl.PseudoNode(1, 0, 1, (0, 1, 0), 0.99, 0.01, 0.01),
                ssl.PseudoNode(2, 0, 1, (0, 1, 2), 0.99, 0.01, 0.01),
            ),
            edges=(
                ssl.PseudoEdge(0, 1, 0.99, 0.01, 0.1),
                ssl.PseudoEdge(0, 2, 0.99, 0.01, 0.1),
            ),
            divisions=(ssl.PseudoDivision(0, (1, 2), 0.99, 0.01),),
            neighbor_cap=2,
            edge_radius_um=8.0,
        )
        policy = ssl.AdaptiveThresholdCurriculum(_all_open_curricula(), ramp_steps=1)
        closed = ssl.select_pseudo_graph(graph, policy, step=1, division_gate_open=False)
        opened = ssl.select_pseudo_graph(graph, policy, step=1, division_gate_open=True)
        self.assertEqual(int(closed.division_mask.sum()), 0)
        self.assertEqual(closed.audits["division"].rejected, {"gate_closed": 1})
        self.assertEqual(int(opened.edge_mask.sum()), 2)
        self.assertEqual(int(opened.division_mask.sum()), 1)

    def test_07_cli_bounds_and_division_prerequisite(self) -> None:
        with self.assertRaisesRegex(runner.LaunchError, "exactly one"):
            runner.parse_args(["--fold", "0", "--output-dir", "/tmp/x", "--data-root", "/tmp/y"])
        with self.assertRaisesRegex(runner.LaunchError, "keeps division closed"):
            runner.parse_args(
                [
                    "--fold",
                    "0",
                    "--output-dir",
                    "/tmp/x",
                    "--data-root",
                    "/tmp/y",
                    "--dry-run",
                    "--division-gate-open",
                ]
            )
        with self.assertRaisesRegex(runner.LaunchError, "keeps division closed"):
            runner.parse_args(
                [
                    "--fold",
                    "0",
                    "--output-dir",
                    "/tmp/x",
                    "--data-root",
                    "/tmp/y",
                    "--dry-run",
                    "--division-gate-open",
                    "--edge-only-grouped-cv",
                    "0.91",
                ]
            )
        parsed = runner.parse_args(
            [
                "--fold",
                "0",
                "--output-dir",
                "/tmp/x",
                "--data-root",
                "/tmp/y",
                "--dry-run",
            ]
        )
        self.assertEqual(parsed.maximum_hours, 1.75)
        self.assertFalse(parsed.division_gate_open)


if __name__ == "__main__":
    unittest.main()

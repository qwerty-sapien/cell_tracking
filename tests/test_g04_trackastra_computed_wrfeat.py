from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/g04_trackastra_computed_wrfeat.py"
SPEC = importlib.util.spec_from_file_location("g04_trackastra_computed_wrfeat", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
g04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = g04
SPEC.loader.exec_module(g04)


class Frame:
    def __init__(self, frame_index, node_ids, coords_zyx):
        self.frame_index = frame_index
        self.node_ids = node_ids
        self.coords_zyx = coords_zyx


class G04ComputedWrfeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = g04._numpy()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="g04-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(
        self,
        *,
        provenance=g04.PROVENANCE_LABEL,
        authentic_masks=False,
        organizer_provided=False,
        feature_names=g04.FEATURE_NAMES,
        matched=None,
        features=None,
        coords=None,
    ):
        np = self.np
        identity = "synthetic:fixture"
        node_ids = np.asarray([11, 12, 13], dtype=np.int64)
        timepoints = np.asarray([0, 1, 1], dtype=np.int64)
        if coords is None:
            coords = np.asarray([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
        if features is None:
            features = np.zeros((3, 12), dtype=np.float32)
            features[:, 0] = [2.0, 3.0, 4.0]
            features[:, 1] = [0.2, 0.3, 0.4]
        if matched is None:
            matched = np.ones(3, dtype=np.bool_)
        artifact = self.root / "video.npz"
        g04.atomic_npz(
            artifact,
            node_ids=node_ids,
            timepoints=timepoints,
            coords_zyx=coords,
            features=features,
            matched=matched,
            feature_names=np.asarray(feature_names),
            feature_schema=np.asarray(g04.FEATURE_SCHEMA),
            provenance_label=np.asarray(provenance),
            authentic_masks=np.asarray(authentic_masks),
            organizer_provided=np.asarray(organizer_provided),
        )
        manifest = {
            "schema": g04.FEATURE_MANIFEST_SCHEMA,
            "provenance_label": provenance,
            "authentic_masks": authentic_masks,
            "organizer_provided": organizer_provided,
            "feature_schema": g04.FEATURE_SCHEMA,
            "feature_names": list(feature_names),
            "identities": [
                {"identity": identity, "path": "video.npz", "sha256": g04.sha256(artifact)}
            ],
        }
        manifest_path = self.root / "manifest.json"
        g04.atomic_json(manifest_path, manifest)
        return identity, node_ids, timepoints, coords, features, manifest_path

    def test_valid_contract_loads_and_aligns(self):
        identity, node_ids, timepoints, coords, features, manifest_path = self.write_fixture()
        manifest = g04.load_feature_manifest(manifest_path)
        video = g04.load_video_features(manifest_path, manifest, identity)
        frames = [Frame(0, node_ids[:1], coords[:1]), Frame(1, node_ids[1:], coords[1:])]
        aligned = g04.align_features(g04.detections_to_table(frames), video)
        self.assertTrue(self.np.array_equal(aligned, features))
        self.assertEqual(tuple(manifest["feature_names"]), g04.FEATURE_NAMES)

    def test_authentic_mask_claim_fails_closed(self):
        _identity, *_rest, manifest_path = self.write_fixture(
            provenance="H3_AUTHENTIC_MASKS", authentic_masks=True, organizer_provided=True
        )
        with self.assertRaisesRegex(g04.ContractError, "COMPUTED_SEGMENTATIONS"):
            g04.load_feature_manifest(manifest_path)

    def test_channel_reorder_fails_closed(self):
        names = list(g04.FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        _identity, *_rest, manifest_path = self.write_fixture(feature_names=tuple(names))
        with self.assertRaisesRegex(g04.ContractError, "12-channel order"):
            g04.load_feature_manifest(manifest_path)

    def test_unmatched_detection_fails_instead_of_zero_fill(self):
        np = self.np
        identity, *_rest, manifest_path = self.write_fixture(
            matched=np.asarray([True, False, True], dtype=np.bool_)
        )
        manifest = g04.load_feature_manifest(manifest_path)
        with self.assertRaisesRegex(g04.ContractError, "may not be zero-filled"):
            g04.load_video_features(manifest_path, manifest, identity)

    def test_all_zero_row_fails_closed(self):
        np = self.np
        features = np.zeros((3, 12), dtype=np.float32)
        features[0, 0] = 2.0
        features[2, 0] = 3.0
        identity, *_rest, manifest_path = self.write_fixture(features=features)
        manifest = g04.load_feature_manifest(manifest_path)
        with self.assertRaisesRegex(g04.ContractError, "all-zero"):
            g04.load_video_features(manifest_path, manifest, identity)

    def test_coordinate_mismatch_fails_closed(self):
        identity, node_ids, _timepoints, coords, _features, manifest_path = self.write_fixture()
        manifest = g04.load_feature_manifest(manifest_path)
        video = g04.load_video_features(manifest_path, manifest, identity)
        changed = coords.copy()
        changed[2, 2] += 1.0
        frames = [Frame(0, node_ids[:1], changed[:1]), Frame(1, node_ids[1:], changed[1:])]
        with self.assertRaisesRegex(g04.ContractError, "coordinate mismatch"):
            g04.align_features(g04.detections_to_table(frames), video)

    def test_divisions_enabled_decoder_caps_outdegree_at_two(self):
        np = self.np
        edges = np.asarray(
            [
                [1, 2, 0.99, 1.0],
                [1, 3, 0.98, 1.0],
                [1, 4, 0.97, 1.0],
                [5, 3, 0.96, 1.0],
                [5, 6, 0.49, 1.0],
            ],
            dtype=np.float64,
        )
        selected, counts = g04.solve_divisions(edges, threshold=0.5)
        self.assertEqual(selected, ((1, 2), (1, 3)))
        self.assertEqual(counts["forks"], 1)
        self.assertEqual(Counter(source for source, _target in selected)[1], 2)

    def test_registration_changes_candidate_view_only(self):
        np = self.np
        table = g04.DetectionTable(
            node_ids=np.asarray([1, 2], dtype=np.int64),
            timepoints=np.asarray([0, 1], dtype=np.int64),
            coords_zyx=np.asarray([[4, 20, 30], [4, 44, 30]], dtype=np.float32),
        )
        original = table.coords_zyx.copy()
        plan = g04.RegistrationPlan(
            identity="train:fixture",
            cumulative_shift_raw_zyx=np.asarray([[0, 0, 0], [0, -24, 0]]),
            pool_breaking_frames=(0,),
            census_sha256="a" * 64,
        )
        views = g04.registered_table_views(table, plan)
        np.testing.assert_array_equal(table.coords_zyx, original)
        np.testing.assert_array_equal(
            views.candidate_zyx,
            np.asarray([[4, 20, 30], [4, 20, 30]], dtype=np.float32),
        )
        self.assertEqual(
            views.receipt["coordinate_contract"]["applications_per_video"], 1
        )

    def test_cv57_population_entries_schema_is_accepted(self):
        identities = [f"train:44b6_{index:08x}" for index in range(21)] + [
            f"train:6bba_{index:08x}" for index in range(36)
        ]
        path = self.root / "cv57.json"
        g04.atomic_json(path, {"entries": [{"identity": item} for item in identities]})
        self.assertEqual(g04.load_population(path, "cv57"), tuple(identities))

    def test_synthetic_smoke_injects_exact_tensor_and_emits_fork(self):
        output = self.root / "smoke.json"
        result = g04.main(["synthetic-smoke", "--output", str(output)])
        self.assertEqual(result, 0)
        report = json.loads(output.read_text(encoding="ascii"))
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["feature_tensor_injected_exactly"])
        self.assertTrue(report["decoder"]["divisions"])
        self.assertEqual(report["decoder"]["forks"], 1)
        self.assertEqual(report["selected_edges"], [[1, 2], [1, 3]])
        self.assertFalse(report["authentic_masks"])


if __name__ == "__main__":
    unittest.main()

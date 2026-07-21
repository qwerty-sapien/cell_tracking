from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


# Load the focused NumPy-only module without importing the package facade,
# whose full pipeline intentionally requires optional SciPy dependencies.
SOURCE = Path(__file__).resolve().parents[1] / "src" / "celltrack_alpha" / "bh_route.py"
SPEC = importlib.util.spec_from_file_location("celltrack_alpha.bh_route", SOURCE)
assert SPEC is not None and SPEC.loader is not None
BHR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BHR
SPEC.loader.exec_module(BHR)

FEATURE_NAMES = BHR.FEATURE_NAMES
FEATURE_SCHEMA_VERSION = BHR.FEATURE_SCHEMA_VERSION
ModelSerializationError = BHR.ModelSerializationError
ProvenanceError = BHR.ProvenanceError
ScenarioRouter = BHR.ScenarioRouter
construct_route_label = BHR.construct_route_label
extract_features = BHR.extract_features


def _features(signal: float = 0.0) -> dict[str, float]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(
        {
            "is_volume": 1.0,
            "log_depth_planes": 3.0,
            "log_xy_area": 10.0,
            "z_to_xy_spacing": 2.0,
            "log_frame_count": 4.0,
            "detections_q10": signal,
            "detections_q50": signal,
            "detections_q90": signal,
            "detection_confidence_q10": 0.8,
            "detection_confidence_q50": 0.9,
            "detection_confidence_q90": 0.98,
        }
    )
    return values


def _separated_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(6):
        records.append(
            {
                "identity": f"train:sparse-{index}",
                "group": f"video-{index // 2}",
                "features": _features(4.0 + index),
                "label": "sparse_route",
            }
        )
        records.append(
            {
                "identity": f"train:dense-{index}",
                "group": f"video-{3 + index // 2}",
                "features": _features(80.0 + index),
                "label": "dense_route",
            }
        )
    return records


class FeatureSchemaTests(unittest.TestCase):
    def test_plain_summary_extracts_finite_versioned_features_without_identity(self) -> None:
        summary = {
            "dataset_id": "ignored-one",
            "frame_shape_zyx": [12, 64, 80],
            "voxel_size_um": [2.0, 0.5, 0.5],
            "frame_count": 4,
            "detections_per_frame": [0, 4, 6, 10],
            "nearest_neighbor_um": [3, 5, 7, 9],
            "unmatched_at_6um": [1, 0, 1, 0],
            "unmatched_at_10um": [0, 0, 1, 0],
            "assignment_margins": [0.02, 0.1, 0.2],
            "detection_confidences": [0.7, 0.8, 0.9],
            "border_flags": [1, 0, 0, 0],
            "depth_edge_flags": [1, 0, 1, 0],
            "gap_lengths": [1, 1, 2, 3],
            "division_motif_count": 2,
            "division_motif_opportunities": 20,
            "detection_intensities": [20, 40, 80],
        }
        first = extract_features(summary)
        second = extract_features({**summary, "dataset_id": "ignored-two", "prefix": "also-ignored"})
        self.assertEqual(tuple(first), FEATURE_NAMES)
        self.assertEqual(first, second)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "celltrack-alpha.bh-route-features.v1")
        self.assertEqual(first["is_volume"], 1.0)
        self.assertEqual(first["empty_frame_fraction"], 0.25)
        self.assertEqual(first["gap_demand_fraction"], 0.5)


class LabelConstructionTests(unittest.TestCase):
    def test_exact_score_ties_use_stable_route_name_order(self) -> None:
        forward = {"zeta": 0.8, "alpha": 0.8}
        reverse = {"alpha": 0.8, "zeta": 0.8}
        self.assertEqual(construct_route_label(forward), "alpha")
        self.assertEqual(construct_route_label(reverse), "alpha")

    def test_runtime_penalty_can_change_the_label(self) -> None:
        candidates = {
            "accurate": {"official_score": 0.90, "runtime_seconds": 10.0},
            "fast": {"official_score": 0.88, "runtime_seconds": 1.0},
        }
        self.assertEqual(construct_route_label(candidates), "accurate")
        self.assertEqual(construct_route_label(candidates, runtime_penalty=0.01), "fast")


class ProvenanceTests(unittest.TestCase):
    def test_audit_and_non_fit_identities_are_rejected(self) -> None:
        router = ScenarioRouter(default_route="safe")
        audit = [{"identity": "audit:clip", "features": _features(), "label": "route"}]
        with self.assertRaisesRegex(ProvenanceError, "non-training identity"):
            router.fit(audit, fit_members={"train:clip"})

        omitted = [{"identity": "train:other", "features": _features(), "label": "route"}]
        with self.assertRaisesRegex(ProvenanceError, "not a declared fit member"):
            router.fit(omitted, fit_members={"train:clip"})

        valid = [{"identity": "train:clip", "features": _features(), "label": "route"}]
        with self.assertRaisesRegex(ProvenanceError, "non-training fit member"):
            router.fit(valid, fit_members={"train:clip", "test:held-out"})


class RouterTests(unittest.TestCase):
    def test_shallow_tree_separates_synthetic_sparse_and_dense_scenarios(self) -> None:
        records = _separated_records()
        members = {str(record["identity"]) for record in records}
        router = ScenarioRouter(default_route="safe").fit(records, fit_members=members)
        sparse = router.predict(_features(7.0))
        dense = router.predict(_features(82.0))
        self.assertEqual(sparse.route, "sparse_route")
        self.assertEqual(dense.route, "dense_route")
        self.assertGreater(sparse.confidence, 0.8)
        self.assertGreater(dense.margin, 0.6)
        self.assertFalse(sparse.used_default)

    def test_low_confidence_and_ood_predictions_fall_back(self) -> None:
        records = []
        for index in range(8):
            records.append(
                {
                    "identity": f"train:ambiguous-{index}",
                    "features": _features(10.0),
                    "label": "left" if index % 2 == 0 else "right",
                }
            )
        members = {str(record["identity"]) for record in records}
        router = ScenarioRouter(default_route="safe").fit(records, fit_members=members)
        weak = router.predict(_features(10.0))
        self.assertEqual(weak.route, "safe")
        self.assertTrue(weak.used_default)
        self.assertFalse(weak.ood)
        self.assertEqual(weak.confidence, 0.5)
        self.assertEqual(weak.margin, 0.0)

        separated = _separated_records()
        fitted = ScenarioRouter(default_route="safe").fit(
            separated,
            fit_members={str(record["identity"]) for record in separated},
        )
        ood = fitted.predict(_features(1000.0))
        self.assertEqual(ood.route, "safe")
        self.assertTrue(ood.ood)

    def test_group_aware_oof_is_deterministic_and_group_disjoint(self) -> None:
        records = _separated_records()
        members = {str(record["identity"]) for record in records}
        first_router = ScenarioRouter(default_route="safe")
        second_router = ScenarioRouter(default_route="safe")
        first = first_router.fit_oof(records, fit_members=members, n_splits=3)
        second = second_router.fit_oof(list(reversed(records)), fit_members=members, n_splits=3)
        first_folds = dict(first.fold_by_group)
        second_folds = dict(second.fold_by_group)
        self.assertEqual(first_folds, second_folds)
        for record, fold in zip(records, first.folds):
            self.assertEqual(fold, first_folds[str(record["group"])])
        self.assertEqual(first_router.to_json(), second_router.to_json())


class SerializationTests(unittest.TestCase):
    def test_ascii_json_round_trip_is_deterministic_and_tamper_evident(self) -> None:
        records = _separated_records()
        members = {str(record["identity"]) for record in records}
        router = ScenarioRouter(default_route="safe").fit(records, fit_members=members)
        encoded = router.to_json()
        self.assertTrue(encoded.isascii())
        restored = ScenarioRouter.from_json(encoded)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(restored.predict(_features(7.0)).route, "sparse_route")

        tampered = encoded.replace('"max_depth":3', '"max_depth":4', 1)
        with self.assertRaisesRegex(ModelSerializationError, "SHA-256 mismatch"):
            ScenarioRouter.from_json(tampered)

    def test_rehashed_incompatible_schema_is_still_rejected(self) -> None:
        records = _separated_records()
        members = {str(record["identity"]) for record in records}
        payload = json.loads(
            ScenarioRouter(default_route="safe").fit(records, fit_members=members).to_json()
        )
        payload["feature_schema"]["version"] = "incompatible.v9"
        core = {key: value for key, value in payload.items() if key != "sha256"}
        canonical = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()
        rehashed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self.assertRaisesRegex(ModelSerializationError, "feature schema mismatch"):
            ScenarioRouter.from_json(rehashed)


if __name__ == "__main__":
    unittest.main()

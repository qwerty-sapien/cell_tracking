from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from celltrack_alpha import u0_t4canon as u0  # noqa: E402
import replay_u0_t4canon as replay  # noqa: E402


CONTRACT = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches/u0_t4canon"
    / "results/u0_t4canon_contract_draft.json"
)
BASE_REPORT = Path(
    "/private/tmp/trackastra-h1-full-cache-output/reports/g02"
    "/trackastra_h1_apncc_base/report.json"
)
FEATURE_REPORT = ROOT / "reports/g02/trackastra_h1_feature_cache/report.json"
FIT_MANIFEST = (
    ROOT
    / "kaggle_kernel/trackastra-h1-feature-cache-prep/cache_dataset_manifest.json"
)


def _payload_pair() -> tuple[dict, dict]:
    challengers = np.asarray([[1, 2]], dtype=np.int64)
    conflicts = np.asarray([[1, 3]], dtype=np.int64)
    offsets = np.asarray([0, 1], dtype=np.int64)
    transition_offsets = np.asarray([0, 1], dtype=np.int64)
    transition_target_t = np.asarray([1], dtype=np.int64)
    confidence = np.asarray([0.7], dtype=np.float32)
    probability = np.asarray([0.8], dtype=np.float32)
    rows = np.zeros((1, 48), dtype=np.float32)
    common = {
        "identity": "train:44b6_fixture",
        "contract_sha256": (
            "4da06d487ba1787f91467a01862e4f7471daa16244674f5171960774da06334b"
        ),
        "selected_raw_edges": 10,
        "challengers": challengers,
        "conflicts": conflicts,
        "proposal_offsets": offsets,
        "transition_offsets": transition_offsets,
        "transition_target_t": transition_target_t,
        "geometry_confidences": confidence,
        "geometry_probabilities": probability,
    }
    base = {
        **common,
        "schema": u0.BASE_SCHEMA,
        "implementation_sha256": (
            "71ce694a6375020b43982447c2c2d1ae4fd9ba088093a1c7e571e5d55877ea5c"
        ),
        "feature_names": [f"apncc_context_{index:02d}" for index in range(48)],
        "row_features": rows,
    }
    feature = {
        **common,
        "schema": u0.FEATURE_SCHEMA,
        "implementation_sha256": (
            "63ebbe51a2de369c6745604dcabf1ba662cac4566453aacbd7695ada98330564"
        ),
        "feature_names": [
            *[f"apncc_context_{index:02d}" for index in range(48)],
            "trackastra_margin",
        ],
        "row_features": np.column_stack((rows, np.ones(1, dtype=np.float32))),
    }
    return base, feature


class U0T4CanonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = u0.load_contract(
            CONTRACT, u0.sha256_path(CONTRACT), ROOT
        )

    def test_contract_has_exactly_twelve_invariants(self) -> None:
        self.assertEqual(len(self.contract["invariants"]), 12)
        self.assertEqual(
            len({row["id"] for row in self.contract["invariants"]}), 12
        )
        self.assertEqual(
            self.contract["old_h1_n1"]["status"],
            "DEAD_STRUCTURAL_M5_T4_DRIFT",
        )

    def test_positive_36_69_authenticated_coverage(self) -> None:
        identities, base, feature = u0.load_authenticated_records(
            self.contract, FIT_MANIFEST, BASE_REPORT, FEATURE_REPORT
        )
        self.assertEqual(len(identities), 105)
        self.assertEqual(len(base), 105)
        self.assertEqual(len(feature), 105)
        self.assertEqual(
            sum(value.startswith("train:44b6_") for value in identities), 36
        )
        self.assertEqual(
            sum(value.startswith("train:6bba_") for value in identities), 69
        )

    def test_payload_pair_and_changed_row(self) -> None:
        base, feature = _payload_pair()
        u0.validate_payload_pair(
            "train:44b6_fixture", base, feature, self.contract
        )
        changed = copy.deepcopy(feature)
        changed["row_features"][0, 0] = 1.0
        with self.assertRaisesRegex(u0.U0T4CanonError, "feature rows drifted"):
            u0.validate_payload_pair(
                "train:44b6_fixture", base, changed, self.contract
            )

    def test_changed_id_and_schema_fail_closed(self) -> None:
        base, feature = _payload_pair()
        changed_id = copy.deepcopy(feature)
        changed_id["identity"] = "train:44b6_other"
        with self.assertRaisesRegex(
            u0.U0T4CanonError, "schema or identity"
        ):
            u0.validate_payload_pair(
                "train:44b6_fixture", base, changed_id, self.contract
            )
        changed_schema = copy.deepcopy(feature)
        changed_schema["schema"] = "wrong"
        with self.assertRaisesRegex(
            u0.U0T4CanonError, "schema or identity"
        ):
            u0.validate_payload_pair(
                "train:44b6_fixture", base, changed_schema, self.contract
            )

    def test_changed_fold_fails_closed(self) -> None:
        identity = "train:44b6_fixture"
        expected = u0.fold_for_identity(identity)
        with self.assertRaisesRegex(u0.U0T4CanonError, "fold assignment"):
            u0.validate_fold_assignment(identity, (expected + 1) % 5)

    def test_changed_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(b"fixture")
            wrong = hashlib.sha256(b"other").hexdigest()
            with self.assertRaisesRegex(u0.U0T4CanonError, "SHA-256"):
                u0.require_sha256(path, wrong, "fixture")

    def test_changed_scorer_pin_fails_closed(self) -> None:
        with self.assertRaisesRegex(u0.U0T4CanonError, "scorer identity"):
            u0.validate_scorer_pin(self.contract, "0" * 64)

    def test_forbidden_path_fails_closed(self) -> None:
        blocked = (
            Path("/tmp")
            / self.contract["forbidden_path_tokens"][0]
            / "payload.pt"
        )
        with self.assertRaisesRegex(
            u0.U0T4CanonError, "outside the fit-only scope"
        ):
            u0.validate_decision_path(blocked, self.contract)

    def test_m5_decision_argument_cannot_be_supplied(self) -> None:
        namespace = argparse.Namespace()
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            u0.run_replay(
                root=ROOT,
                contract_path=CONTRACT,
                contract_sha256=u0.sha256_path(CONTRACT),
                fit_manifest_path=FIT_MANIFEST,
                base_report_path=BASE_REPORT,
                feature_report_path=FEATURE_REPORT,
                implementation_paths=(),
                m5_decision_input=namespace,
            )

    def test_label_free_video_remains_a_valid_empty_group(self) -> None:
        _base, feature = _payload_pair()
        prepared = {
            "labeled_challenger_pairs": np.asarray([[9, 10]], dtype=np.int64),
            "labeled_labels": np.asarray([1], dtype=np.int8),
        }
        known = u0.challenger_labels(
            prepared["labeled_challenger_pairs"],
            prepared["labeled_labels"],
        )
        with self.assertRaisesRegex(
            u0.TrackastraLinearError, "no prepared labels"
        ):
            u0.labeled_subset(feature, known)
        empty = u0.proposal_features(feature)[:0]
        self.assertEqual(empty.shape, (0, 49))

    def test_no_admissible_threshold_is_not_weakened(self) -> None:
        head = u0.LinearHead(
            mean=np.zeros(49, dtype=np.float64),
            scale=np.ones(49, dtype=np.float64),
            weights=np.zeros(49, dtype=np.float64),
            intercept=0.0,
            ridge_lambda=0.01,
        )
        scores = head.score(np.zeros((100, 49), dtype=np.float64))
        labels = np.asarray([0, 1] * 50, dtype=np.int8)
        with self.assertRaisesRegex(
            u0.TrackastraLinearError, "no calibration threshold"
        ):
            u0.select_threshold(
                scores,
                labels,
                minimum_accepted=100,
                minimum_precision=0.95,
            )

    def test_executable_cli_has_no_split_or_scorer_argument(self) -> None:
        destinations = {action.dest for action in replay.parser()._actions}
        self.assertEqual(
            destinations,
            {
                "help",
                "contract",
                "contract_sha256",
                "fit_manifest",
                "base_report",
                "feature_report",
                "output",
            },
        )


if __name__ == "__main__":
    unittest.main()

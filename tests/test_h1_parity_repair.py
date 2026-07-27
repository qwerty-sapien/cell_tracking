from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "reports/g02/screening/h1_parity_repair_contract.json"
)


class H1ParityRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_t4_is_canonical_and_m5_is_not_a_gate(self) -> None:
        platform = self.contract["platform_contract"]
        self.assertEqual(platform["canonical_platform"], "Tesla T4")
        self.assertEqual(
            platform["reproducibility_gate"], "T4-to-T4 array_equal"
        )
        self.assertFalse(platform["m5_is_a_pass_fail_gate"])

    def test_evidence_pins_match(self) -> None:
        for record in self.contract["pinned_evidence"].values():
            path = ROOT / record["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, record["sha256"])

    def test_t4_witnesses_are_bit_exact(self) -> None:
        witnesses = self.contract["t4_repeatability_witnesses"]
        self.assertEqual(len(witnesses), 2)
        for witness in witnesses:
            self.assertTrue(witness["array_equal"])
            self.assertEqual(
                witness["run1_sha256"], witness["run2_sha256"]
            )

    def test_every_float_drift_is_classified_and_covered(self) -> None:
        classification = self.contract["classification"]
        self.assertEqual(classification["verdict"], "BENIGN_FP")
        self.assertFalse(
            classification["safety_catch"]["structured_drift_detected"]
        )
        self.assertEqual(
            classification["safety_catch"]["nan_or_inf_count"], 0
        )
        geometry = classification["float_geometry"]
        neural = classification["neural_columns"]["drifting_columns"]
        self.assertEqual(len(geometry), 2)
        self.assertEqual(len(neural), 10)
        self.assertEqual(
            len(
                classification["neural_columns"]["exact_column_indices"]
            ),
            39,
        )
        for record in [*geometry, *neural]:
            self.assertTrue(record["within_tolerance"])
            self.assertGreater(record["maximum_absolute_difference"], 0)
            self.assertGreaterEqual(record["atol"], 0)
            self.assertGreaterEqual(record["rtol"], 0)

    def test_integer_geometry_is_exact(self) -> None:
        integer = self.contract["classification"]["integer_geometry"]
        self.assertEqual(integer["rule"], "EXACT")
        self.assertEqual(integer["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()

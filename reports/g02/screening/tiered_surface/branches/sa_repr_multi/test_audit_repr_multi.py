from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("audit_repr_multi", HERE / "audit_repr_multi.py")
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReprMultiAuditTests(unittest.TestCase):
    def test_all_pinned_inputs_authenticate(self) -> None:
        rows = AUDIT.authenticate_pins()
        self.assertTrue(rows)
        self.assertTrue(all(row["pass"] for row in rows.values()))

    def test_cellect_identity_authenticates(self) -> None:
        result = AUDIT.authenticate_cellect()
        self.assertTrue(result["authenticated"])
        self.assertEqual(
            result["queue_disposition"], "EXCLUDED_ALREADY_DISPATCHED_R7_SERIAL_LANE"
        )

    def test_label_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = root / "population.npz"
            request = root / "request.npz"
            endpoint = {
                "source_id": np.asarray([1], dtype=np.int64),
                "target_id": np.asarray([2], dtype=np.int64),
                "source_t": np.asarray([0], dtype=np.int16),
                "target_t": np.asarray([1], dtype=np.int16),
                "source_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
                "target_zyx": np.asarray([[0, 0, 1]], dtype=np.int16),
            }
            np.savez(
                population,
                **endpoint,
                b0=np.asarray([[0.0]], dtype=np.float32),
                oracle_delta_S=np.asarray([1.0]),
                oracle_label=np.asarray([0], dtype=np.uint8),
                fold=np.asarray([0], dtype=np.int8),
                fp_budget=np.asarray([0], dtype=np.int64),
            )
            np.savez(request, **endpoint)
            row = {
                "count": 1,
                "beneficial": 0,
                "harmful": 1,
                "fp_budget": 0,
                "population_sha256": AUDIT.sha256(population),
                "request_sha256": AUDIT.sha256(request),
            }
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.validate_shard(population, request, row)

    def test_request_endpoint_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = root / "population.npz"
            request = root / "request.npz"
            endpoint = {
                "source_id": np.asarray([1], dtype=np.int64),
                "target_id": np.asarray([2], dtype=np.int64),
                "source_t": np.asarray([0], dtype=np.int16),
                "target_t": np.asarray([1], dtype=np.int16),
                "source_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
                "target_zyx": np.asarray([[0, 0, 1]], dtype=np.int16),
            }
            np.savez(
                population,
                **endpoint,
                b0=np.asarray([[0.0]], dtype=np.float32),
                oracle_delta_S=np.asarray([-1.0]),
                oracle_label=np.asarray([0], dtype=np.uint8),
                fold=np.asarray([0], dtype=np.int8),
                fp_budget=np.asarray([0], dtype=np.int64),
            )
            wrong = dict(endpoint)
            wrong["target_id"] = np.asarray([3], dtype=np.int64)
            np.savez(request, **wrong)
            row = {
                "count": 1,
                "beneficial": 0,
                "harmful": 1,
                "fp_budget": 0,
                "population_sha256": AUDIT.sha256(population),
                "request_sha256": AUDIT.sha256(request),
            }
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.validate_shard(population, request, row)

    def test_queue_has_no_duplicate_r7(self) -> None:
        queue = AUDIT.queue_report({}, {"aggregate_sha256": "a" * 64})
        self.assertEqual(queue["ready_sample_count"], 0)
        self.assertEqual(len(queue["authenticated_but_excluded"]), 1)
        self.assertFalse(queue["authorization_requested"])


if __name__ == "__main__":
    unittest.main()

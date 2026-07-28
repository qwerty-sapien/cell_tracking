#!/usr/bin/env python3
"""Focused tests for fail-closed endpoint return and audit helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import audit_reverse_supplier as module


class ReverseSupplierTests(unittest.TestCase):
    def test_endpoint_return_indexes_separate_reverse_matrix(self) -> None:
        reverse = [
            [0.0, 0.1, 0.2],
            [0.3, 0.0, 0.4],
            [0.5, 0.6, 0.0],
        ]
        index = {10: 0, 11: 1, 12: 2}
        self.assertEqual(module.endpoint_return(reverse, index, 10, 12), 0.5)
        self.assertNotEqual(module.endpoint_return(reverse, index, 10, 12), 0.2)

    def test_endpoint_return_rejects_missing_identity(self) -> None:
        with self.assertRaises(module.AuditError):
            module.endpoint_return([[0.0]], {10: 0}, 10, 99)

    def test_receipt_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x"
            path.write_bytes(b"x")
            with self.assertRaises(module.AuditError):
                module.receipt(path, "0" * 64)

    def test_checkpoint_hashes_are_distinct_pins(self) -> None:
        self.assertEqual(len(module.CHECKPOINT_MEMBER_SHA256), 64)
        self.assertEqual(len(module.CHECKPOINT_ARCHIVE_SHA256), 64)
        self.assertNotEqual(
            module.CHECKPOINT_MEMBER_SHA256,
            module.CHECKPOINT_ARCHIVE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()

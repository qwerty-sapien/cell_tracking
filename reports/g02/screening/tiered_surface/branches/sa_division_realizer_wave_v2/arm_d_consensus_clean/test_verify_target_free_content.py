#!/usr/bin/env python3
"""Focused tests for the read-only clean B/D target-free audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_target_free_content import AuditError, audit, atomic_json, event_key


class TargetFreeContentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = audit()

    def test_clean_b_content_and_upstream_authentication_pass(self) -> None:
        clean_b = self.value["content_validation"]["clean_b"]
        self.assertEqual(clean_b["status"], "pass")
        self.assertEqual(clean_b["videos"], 57)
        self.assertEqual(clean_b["selected"], 48)
        self.assertEqual(clean_b["by_family"], {"44b6": 19, "6bba": 29})
        self.assertEqual(clean_b["authenticated_upstream_inputs"], 57)
        self.assertTrue(clean_b["operational_action_fields_equal_source"])

    def test_clean_d_content_is_exact_empty_intersection(self) -> None:
        clean_d = self.value["content_validation"]["clean_d"]
        self.assertEqual(clean_d["component_counts"], {"A": 23, "B": 48, "C": 22})
        self.assertEqual(clean_d["intersection_count"], 0)
        self.assertEqual(clean_d["selected_count"], 0)
        self.assertTrue(clean_d["payload_matches_frozen_builder_semantics"])

    def test_frozen_build_provenance_fails_closed(self) -> None:
        provenance = self.value["frozen_build_provenance"]
        self.assertEqual(provenance["status"], "fail_closed")
        self.assertFalse(provenance["hashes_match"])
        self.assertFalse(provenance["retroactive_freeze_created"])
        self.assertFalse(provenance["binding_consensus_allowed"])
        self.assertNotEqual(
            provenance["frozen_original_d_preregistration_sha256"],
            provenance["clean_d_preregistration_sha256"],
        )

    def test_event_key_is_canonical_in_branch_order(self) -> None:
        left = event_key("train:44b6_267148e4", 20390, 20636, 20642)
        right = event_key("train:44b6_267148e4", 20390, 20642, 20636)
        self.assertEqual(left, right)
        self.assertEqual(
            left,
            "8889f5134c1f547f624c290ba1b70b698ba219fd95b9f2cc1f1714fb54136c7d",
        )

    def test_receipt_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            atomic_json(output, self.value)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["binding_status"],
                "INVALID_NONBINDING",
            )
            with self.assertRaisesRegex(AuditError, "refusing to overwrite"):
                atomic_json(output, self.value)

    def test_artifact_hashes_are_sha256(self) -> None:
        for digest in self.value["artifact_hashes"].values():
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused mechanical tests for the SA-REPR supplier boundary."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_supplier_provenance.py"
SPEC = importlib.util.spec_from_file_location("sa_repr_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "cellect_support_smoke.py"
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "sa_repr_cellect_support", SUPPORT_PATH
)
assert SUPPORT_SPEC and SUPPORT_SPEC.loader
SUPPORT = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)


class SupplierAuditTests(unittest.TestCase):
    def test_canonical_sha_is_key_order_independent(self) -> None:
        self.assertEqual(
            AUDIT.canonical_sha({"a": 1, "b": 2}),
            AUDIT.canonical_sha({"b": 2, "a": 1}),
        )

    def test_mask_token_requires_a_path_token(self) -> None:
        self.assertTrue(AUDIT.token_hit("train/x/masks/0.npy"))
        self.assertTrue(AUDIT.token_hit("train/x/wrfeat.npy"))
        self.assertFalse(AUDIT.token_hit("train/x/unmasked_intensity/0"))
        self.assertFalse(AUDIT.token_hit("train/x/labelscores.csv"))

    def test_weight_classification_is_fail_closed(self) -> None:
        self.assertEqual(
            AUDIT.classify_weight(Path("/tmp/dinov2_vits14_pretrain.pth")),
            "target_dinov2_vits14",
        )
        self.assertEqual(
            AUDIT.classify_weight(Path("/tmp/trackastra/model.pt")),
            "excluded_trackastra_association",
        )

    def test_permutation_is_exact_repeat(self) -> None:
        witness = AUDIT.control_witness()
        self.assertTrue(all(witness["checks"].values()))
        self.assertEqual(
            witness["permuted_sha256"], witness["repeat_sha256"]
        )

    def test_singleton_group_is_supported_without_mutation(self) -> None:
        rows = [
            {
                "video_id": "train:singleton",
                "frame": 0,
                "row_id": 1,
                "xy": [1, 2],
                "xz": [3, 4],
                "yz": [5, 6],
            }
        ]
        self.assertEqual(AUDIT.permute_nonxy(rows, 2719), rows)

    def test_pin_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}\n", encoding="ascii")
            prereg = {
                "pinned_inputs": {
                    "x": {"path": str(path), "sha256": "0" * 64}
                }
            }
            with self.assertRaisesRegex(RuntimeError, "frozen input drift"):
                AUDIT.assert_pins(prereg)

    def test_output_json_roundtrip_is_ascii(self) -> None:
        witness = AUDIT.control_witness()
        payload = json.dumps(witness, sort_keys=True, ensure_ascii=True)
        self.assertEqual(json.loads(payload), witness)

    def test_network_guard_fails_closed(self) -> None:
        with self.assertRaisesRegex(SUPPORT.NetworkForbidden, "forbidden"):
            SUPPORT._forbid_network()

    def test_terminal_artifacts_are_internally_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        support = json.loads(
            (root / "cellect_support_audit.json").read_text(encoding="ascii")
        )
        frozen = json.loads(
            (root / "FROZEN_SUPPLIER_CONTRACT_v1.json").read_text(
                encoding="ascii"
            )
        )
        state = json.loads(
            (root / "state_fragment.json").read_text(encoding="ascii")
        )
        self.assertEqual(support["status"], "PASS")
        self.assertTrue(support["determinism"]["byte_identical"])
        self.assertFalse(support["data_access"]["cv57"])
        self.assertFalse(support["feature_cache_produced"])
        self.assertEqual(
            frozen["support_evidence"]["audit_sha256"],
            hashlib.sha256(
                (root / "cellect_support_audit.json").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(frozen["phase1_role"], "supplier_only")
        self.assertFalse(frozen["phase1_absences"]["gpu_request"])
        self.assertFalse(frozen["phase1_absences"]["cv57"])
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(
            state["forbidden_scope"],
            {
                "cv57_used": False,
                "submission_action": False,
                "runpod_action": False,
            },
        )
        self.assertEqual(
            state["family_execution"],
            {
                "44b6_ran": False,
                "6bba_ran": False,
                "cv57_ran": False,
                "graph_run_count": 0,
                "family_44b6_gate": None,
            },
        )
        self.assertEqual(
            {node["id"] for node in state["nodes"]}, {"R2", "R4", "R7"}
        )
        self.assertEqual(
            {node["tier"] for node in state["nodes"]}, {"PARK_B"}
        )


if __name__ == "__main__":
    unittest.main()

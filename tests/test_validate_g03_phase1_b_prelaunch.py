#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_g03_phase1_b_prelaunch as guard  # noqa: E402


def write_handoff(root: Path) -> None:
    root.mkdir(parents=True)
    manifest = {
        "schema": guard.HANDOFF_SCHEMA,
        "handoff_id": "test-handoff",
        "producer_kernel": "owner/a",
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "state": {},
    }
    (root / "state").mkdir()
    (root / "state/checkpoint.pt").write_bytes(b"checkpoint")
    (root / "MANIFEST.json").write_bytes(guard.canonical_json_bytes(manifest))
    records = []
    for path in sorted([root / "MANIFEST.json", root / "state/checkpoint.pt"]):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": guard.sha256_file(path),
            }
        )
    tree = guard.sha256_bytes(guard.canonical_json_bytes(records))
    (root / "INVENTORY.json").write_bytes(
        guard.canonical_json_bytes(
            {"schema": guard.HANDOFF_SCHEMA, "handoff_id": "test-handoff", "files": records, "tree_sha256": tree}
        )
    )
    (root / "READY.json").write_bytes(
        guard.canonical_json_bytes(
            {"schema": guard.HANDOFF_SCHEMA, "handoff_id": "test-handoff", "tree_sha256": tree}
        )
    )


class HandoffTests(unittest.TestCase):
    def test_exact_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "handoff"
            write_handoff(root)
            result = guard.verify_handoff(root)
            self.assertEqual(result["handoff_id"], "test-handoff")
            self.assertEqual(len(result["files"]), 2)

    def test_changed_payload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "handoff"
            write_handoff(root)
            (root / "state/checkpoint.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(guard.ValidationError, "size drifted"):
                guard.verify_handoff(root)

    def test_extra_payload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "handoff"
            write_handoff(root)
            (root / "undeclared.bin").write_bytes(b"x")
            with self.assertRaisesRegex(guard.ValidationError, "missing, extra"):
                guard.verify_handoff(root)

    def test_dataset_root_is_exact_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_handoff(root / "handoff")
            (root / "dataset-metadata.json").write_text(
                json.dumps({"id": "owner/handoff", "title": "handoff", "isPrivate": True})
            )
            self.assertEqual(
                guard.verify_dataset_root(root, "owner/handoff")["handoff_id"],
                "test-handoff",
            )
            (root / "extra.txt").write_text("no")
            with self.assertRaisesRegex(guard.ValidationError, "root members drifted"):
                guard.verify_dataset_root(root, "owner/handoff")

    def test_downloaded_runtime_root_without_metadata_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_handoff(root / "handoff")
            result = guard.verify_dataset_root(root, "owner/handoff")
            self.assertIsNone(result["dataset_metadata_sha256"])


class TerminalTests(unittest.TestCase):
    def make_terminal(self, root: Path) -> SimpleNamespace:
        output = root / "download/g03_phase1_b_output"
        output.mkdir(parents=True)
        graph = {
            "schema": "celltrack-alpha.g03-phase1-44b6-structural-graph-delta.v1",
            "scope": {"family": "44b6", "identity_count": 2, "manifest_sha256": "c" * 64},
            "checkpoint_state_digest": "d" * 64,
            "official_frozen_graph_objective_evaluated": False,
            "selection_authorized": False,
            "delta": {"adjusted_edge_jaccard": 0.0, "raw_edge_jaccard": 0.0, "node_recall": 0.0},
            "records": [
                {"identity": "train:44b6_a", "control_edge_sha256": "e" * 64, "candidate_edge_sha256": "e" * 64},
                {"identity": "train:44b6_b", "control_edge_sha256": "f" * 64, "candidate_edge_sha256": "f" * 64},
            ],
            "verdict": "STRUCTURAL_NOOP_PASS",
        }
        graph_path = output / "GRAPH_EMISSION.json"
        graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
        receipt = {
            "schema": "celltrack-alpha.g03-phase1-b-receipt.v1",
            "status": "PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED",
            "contract_sha256": "a" * 64,
            "handoff": {"tree_sha256": "1" * 64, "manifest_sha256": "2" * 64},
            "resumed_global_step": 7,
            "final_state_digest": "d" * 64,
            "graph_emission_sha256": guard.sha256_file(graph_path),
            "swap_guard": {"growth_bytes": 0},
            "selection_authorized": False,
        }
        (output / "B_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        contract = {
            "kernels": {"b": {"id": "owner/b", "embedded_contract_sha256": "a" * 64}},
            "limits": {"sustained_swapout_limit_bytes": 100},
            "held_fit_emission": {
                "family": "44b6",
                "identity_count": 2,
                "manifest_sha256": "c" * 64,
                "expected_structural_delta": graph["delta"],
            },
        }
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps(contract))
        validation = {
            "handoff": {"tree_sha256": "1" * 64, "manifest_sha256": "2" * 64},
            "receipt": {"global_step": 7},
        }
        validation_path = root / "a-validation.json"
        validation_path.write_text(json.dumps(validation))
        return SimpleNamespace(
            contract=contract_path,
            a_validation=validation_path,
            output_root=root / "download",
        )

    def test_terminal_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.make_terminal(Path(raw))
            result = guard.validate_terminal(args)
            self.assertEqual(result["status"], "PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED")
            self.assertFalse(result["selection_authorized"])

    def test_terminal_nonzero_structural_delta_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.make_terminal(Path(raw))
            graph_path = args.output_root / "g03_phase1_b_output/GRAPH_EMISSION.json"
            graph = json.loads(graph_path.read_text())
            graph["delta"]["adjusted_edge_jaccard"] = 0.1
            graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
            receipt_path = args.output_root / "g03_phase1_b_output/B_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["graph_emission_sha256"] = guard.sha256_file(graph_path)
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(guard.ValidationError, "structural delta drifted"):
                guard.validate_terminal(args)


if __name__ == "__main__":
    unittest.main()

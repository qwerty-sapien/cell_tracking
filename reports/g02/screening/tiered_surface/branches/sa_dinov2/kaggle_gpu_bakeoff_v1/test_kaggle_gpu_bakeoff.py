from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "kaggle_kernel_template.py"
PARTIAL = HERE / "PARTIAL_REUSE_DECISION.json"


def load_template():
    specification = importlib.util.spec_from_file_location("gpu_template", TEMPLATE)
    if specification is None or specification.loader is None:
        raise RuntimeError("template import failed")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class DINOv2KaggleGPUContractTests(unittest.TestCase):
    def test_parent_step0_begins_with_full_inventory(self):
        tree = ast.parse(TEMPLATE.read_text())
        parent = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parent_main"
        )
        first = parent.body[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertEqual(ast.unparse(first.value.func), "dump_mount_inventory")

    def test_no_heavy_top_level_import(self):
        tree = ast.parse(TEMPLATE.read_text())
        imports = {
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue({"numpy", "torch", "zarr"}.isdisjoint(imports))

    def test_hash_resolution_requires_exactly_one(self):
        module = load_template()
        rows = [{"path": "/tmp/a", "bytes": 1, "sha256": "a" * 64}]
        self.assertEqual(
            module.unique_by_hash(rows, "a" * 64, "fixture"), Path("/tmp/a")
        )
        with self.assertRaises(module.RuntimeContractError):
            module.unique_by_hash([], "a" * 64, "fixture")
        with self.assertRaises(module.RuntimeContractError):
            module.unique_by_hash(rows * 2, "a" * 64, "fixture")

    def test_partial_rows_are_preoutcome_discarded(self):
        decision = json.loads(PARTIAL.read_text())
        self.assertEqual(decision["status"], "DISCARD_AND_RECOMPUTE_ALL")
        self.assertEqual(decision["quarantined_action_rows"], 724_634)
        self.assertEqual(decision["reused_action_rows"], 0)
        self.assertFalse(decision["scientific_outcome_consumed"])

    def test_hard_and_soft_limits_are_literal(self):
        source = TEMPLATE.read_text()
        self.assertIn('config["limits"]["hard_seconds"]', source)
        self.assertIn('config["limits"]["soft_seconds"]', source)
        self.assertIn("subprocess.TimeoutExpired", source)


if __name__ == "__main__":
    unittest.main()

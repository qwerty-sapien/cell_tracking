from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v32_submission_bisect", ROOT / "scripts/v32_submission_bisect.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_row(*, row_id: int, row_type: str = "node", y: int = 0) -> dict[str, int | str]:
    return {
        "id": row_id,
        "dataset": "test_a",
        "row_type": row_type,
        "node_id": row_id,
        "t": 0,
        "z": 0,
        "y": y,
        "x": 0,
        "source_id": -1 if row_type == "node" else 0,
        "target_id": -1 if row_type == "node" else 1,
    }


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODULE.baseline.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class V32SubmissionBisectTests(unittest.TestCase):
    def test_stage_order_is_cumulative_and_hashing_is_last(self) -> None:
        self.assertEqual(MODULE.STAGES[0], "pristine_control")
        self.assertEqual(MODULE.STAGES[-1], "input_hashing_last")
        self.assertEqual(MODULE.stage_settings("telemetry_jsonl_no_input_hash")["replacement_count"], 0)
        self.assertEqual(MODULE.stage_settings("patch_exact_mount_discovery")["replacement_count"], 1)
        self.assertEqual(
            MODULE.stage_settings("patch_skip_zero_weight_secondary_inference")["replacement_count"],
            4,
        )
        self.assertFalse(MODULE.stage_settings("patch_exact_mount_discovery")["hash_input_contents"])
        self.assertTrue(MODULE.stage_settings("input_hashing_last")["hash_input_contents"])

    def test_no_hash_telemetry_and_step0_boundary_compile(self) -> None:
        telemetry = MODULE.baseline.runtime_logging_source(hash_input_contents=False)
        self.assertIn("HASH_INPUT_CONTENTS = False", telemetry)
        self.assertIn("content_hashing=HASH_INPUT_CONTENTS", telemetry)
        compile(telemetry, "telemetry_no_hash.py", "exec")
        step0 = MODULE.step0_without_telemetry_source(
            expected_test_count=4, local_gate={"gate_sha256": "a" * 64}
        )
        self.assertIn("STEP0 = run_step0()", step0)
        self.assertIn("recursive_input_content_hashing\": False", step0)
        compile(step0, "step0_no_hash.py", "exec")

    def test_pristine_control_copies_notebook_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory) / "package"
            result = MODULE.build_stage(
                SimpleNamespace(
                    stage="pristine_control",
                    kernel_id="onebitbrain/g02-v32-bisect-pristine-control-test",
                    kernel_dir=package_dir,
                    expected_test_count=4,
                    score_receipt=Path("unused"),
                    taxonomy_receipt=Path("unused"),
                    terminal_authentication=Path("unused"),
                )
            )
            control = package_dir / "public_v32_pristine.ipynb"
            self.assertEqual(MODULE.sha256(control), MODULE.baseline.PUBLIC_NOTEBOOK_SHA256)
            self.assertEqual(control.read_bytes(), MODULE.baseline.PUBLIC_NOTEBOOK.read_bytes())
            self.assertTrue(result["bisection"]["source_byte_exact"])

    def test_csv_comparison_checks_order_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine = root / "pristine.csv"
            candidate = root / "candidate.csv"
            rows = [valid_row(row_id=0), valid_row(row_id=1, row_type="edge")]
            write_csv(pristine, rows)
            write_csv(candidate, rows)
            equal = MODULE.compare_csv(
                SimpleNamespace(
                    pristine_csv=pristine, candidate_csv=candidate, output=root / "equal.json"
                )
            )
            self.assertEqual(equal["status"], "PASS")
            self.assertEqual(equal["pristine"]["node_rows"], 1)
            self.assertEqual(equal["pristine"]["edge_rows"], 1)

            write_csv(candidate, [valid_row(row_id=0, y=1), valid_row(row_id=1, row_type="edge")])
            changed = MODULE.compare_csv(
                SimpleNamespace(
                    pristine_csv=pristine, candidate_csv=candidate, output=root / "changed.json"
                )
            )
            self.assertEqual(changed["status"], "DIFF")
            self.assertFalse(changed["comparison"]["full_ordered_content_identical"])


if __name__ == "__main__":
    unittest.main()

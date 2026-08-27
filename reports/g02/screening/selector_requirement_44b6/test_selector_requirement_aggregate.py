from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from reports.g02.screening.selector_requirement_44b6 import (
    selector_requirement as base,
)
from reports.g02.screening.selector_requirement_44b6 import (
    selector_requirement_aggregate as aggregate,
)


class AggregateClassValidationTests(unittest.TestCase):
    def test_accepts_explicit_one_class_degenerate_population(self) -> None:
        payload = {
            "schema": "celltrack-alpha.selector-requirement-44b6-class.v1",
            "status": "COMPLETE",
            "action_class": "restore_topology",
            "population": {
                "actions": 50,
                "beneficial": 0,
                "harmful": 50,
                "videos": 36,
            },
            "oracle": {
                "action_class": "restore_topology",
                "support_grid": base.support_grid(50),
            },
            "noise": {
                "status": "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE",
                "seed_runs": [],
                "aggregate": None,
                "terminal_compiler_replay": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "CLASS_RESULT.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            observed = aggregate.load_class(path, "restore_topology")
        self.assertEqual(observed["population"]["beneficial"], 0)

    def test_rejects_false_degenerate_status(self) -> None:
        payload = {
            "schema": "celltrack-alpha.selector-requirement-44b6-class.v1",
            "status": "COMPLETE",
            "action_class": "restore_topology",
            "population": {
                "actions": 50,
                "beneficial": 1,
                "harmful": 49,
                "videos": 36,
            },
            "oracle": {
                "action_class": "restore_topology",
                "support_grid": base.support_grid(50),
            },
            "noise": {
                "status": "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE",
                "seed_runs": [],
                "aggregate": None,
                "terminal_compiler_replay": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "CLASS_RESULT.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(aggregate.base.DiagnosticError):
                aggregate.load_class(path, "restore_topology")


if __name__ == "__main__":
    unittest.main()

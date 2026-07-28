from pathlib import Path
import tempfile
import unittest

import audit_assoc_cycle as module


class AssocCycleAuditTests(unittest.TestCase):
    def test_expected_contract_has_five_frozen_inputs(self) -> None:
        self.assertEqual(len(module.EXPECTED), 5)
        self.assertTrue(all(len(value) == 64 for value in module.EXPECTED.values()))

    def test_sha256_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"cycle")
            self.assertEqual(module.sha256(path), module.sha256(path))

    def test_missing_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(module.AuditError):
                module.audit(Path(directory))

    def test_forward_only_rule_is_explicit(self) -> None:
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("synthetic_reverse_forbidden", source)
        self.assertIn("PARK_NOT_READY_MISSING_AUTHENTIC_REVERSE", source)


if __name__ == "__main__":
    unittest.main()

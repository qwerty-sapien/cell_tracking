from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("validate_dinov2_vits14.py")
SPEC = importlib.util.spec_from_file_location("validate_dinov2_vits14", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class FakeTensor:
    pass


class FakeTorch:
    Tensor = FakeTensor


class ValidatorUnitTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"dinov2")
            self.assertEqual(
                validator.sha256_file(path),
                hashlib.sha256(b"dinov2").hexdigest(),
            )

    def test_tensor_mapping_accepts_string_tensor_pairs(self):
        value = {"a": FakeTensor(), "b": FakeTensor()}
        self.assertIs(validator.require_tensor_mapping(value, FakeTorch), value)

    def test_tensor_mapping_rejects_non_tensor_value(self):
        with self.assertRaisesRegex(validator.ValidationError, "non-tensor"):
            validator.require_tensor_mapping({"a": 1}, FakeTorch)

    def test_tensor_mapping_rejects_non_string_key(self):
        with self.assertRaisesRegex(validator.ValidationError, "non-string"):
            validator.require_tensor_mapping({1: FakeTensor()}, FakeTorch)

    def test_weight_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad.pth"
            checkpoint.write_bytes(b"not-the-pinned-weight")
            with self.assertRaises(validator.ValidationError):
                validator.validate_weight_bytes(checkpoint)

    def test_source_revision_mismatch_fails_closed(self):
        values = iter(
            [
                "0" * 40,
                validator.EXPECTED_SOURCE_TREE,
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "LICENSE").write_bytes(b"wrong")
            with mock.patch.object(validator, "git_value", side_effect=lambda *_: next(values)):
                with self.assertRaises(validator.ValidationError):
                    validator.validate_source(source)


if __name__ == "__main__":
    unittest.main()

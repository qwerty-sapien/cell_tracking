from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import serial_launch


class SerialLaunchTests(unittest.TestCase):
    def test_authoritative_gpu_quota_parser(self):
        result = serial_launch.parse_gpu_quota(json.dumps([
            {"resource": "GPU", "used": "4.53h", "remaining": "25.47h", "total": "30.00h", "refreshAt": "2026-08-01T00:00:00"},
            {"resource": "TPU", "used": "0.00h", "remaining": "20.00h", "total": "20.00h"},
        ]))
        self.assertEqual(result["used_h"], 4.53)
        self.assertEqual(result["remaining_h"], 25.47)
        self.assertEqual(result["total_h"], 30.0)

    def test_exact_version_proof_rejects_nonfresh_or_non_v1_pushes(self):
        with self.assertRaises(serial_launch.LaunchError):
            serial_launch.assert_exactly_one_version([{"ref": serial_launch.KERNEL_ID}], 1)
        with self.assertRaises(serial_launch.LaunchError):
            serial_launch.assert_exactly_one_version([], 2)

    def test_shared_lock_rejects_a_second_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock"
            with serial_launch.exclusive_lock(lock):
                with self.assertRaises(serial_launch.LaunchError):
                    with serial_launch.exclusive_lock(lock):
                        self.fail("second owner must not acquire the launch lock")

    def test_not_found_kernel_list_means_zero_versions(self):
        self.assertEqual(serial_launch.parse_kernel_rows("Not found\n"), [])

    def test_inflight_filter_includes_the_apncc_transport_lane(self):
        with mock.patch.object(serial_launch, "checked") as checked:
            checked.return_value = type("Result", (), {"stdout": "running", "stderr": ""})()
            observed = serial_launch.inflight_campaign([{"ref": serial_launch.KERNEL_ID}])
        self.assertEqual(observed[0]["ref"], serial_launch.KERNEL_ID)


if __name__ == "__main__":
    unittest.main()

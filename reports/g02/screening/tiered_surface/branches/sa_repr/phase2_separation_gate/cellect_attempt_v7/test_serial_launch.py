from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from reports.g02.screening.tiered_surface.branches.sa_repr.phase2_separation_gate.cellect_attempt_v7 import serial_launch


class SerialLaunchTests(unittest.TestCase):
    def test_parse_authoritative_gpu_quota(self):
        quota = serial_launch.parse_gpu_quota(
            json.dumps(
                [
                    {
                        "resource": "GPU",
                        "used": "4.53h",
                        "remaining": "25.47h",
                        "total": "30.00h",
                        "refreshAt": "2026-08-01T00:00:00",
                    },
                    {
                        "resource": "TPU",
                        "used": "0.00h",
                        "remaining": "20.00h",
                        "total": "20.00h",
                    },
                ]
            )
        )
        self.assertEqual(quota["used_h"], 4.53)
        self.assertEqual(quota["remaining_h"], 25.47)
        self.assertEqual(quota["total_h"], 30.0)

    def test_exclusive_lock_rejects_second_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "serial.lock"
            with serial_launch.exclusive_lock(lock):
                with self.assertRaisesRegex(
                    serial_launch.LaunchError, "another Kaggle launch"
                ):
                    with serial_launch.exclusive_lock(lock):
                        self.fail("second lock holder must not enter")

    def test_target_must_be_fresh_and_post_count_increases_exactly_one(self):
        pre = [{"ref": "onebitbrain/unrelated"}]
        post = [{"ref": serial_launch.KERNEL_ID}]
        receipt = serial_launch.assert_version_delta(pre, post, 1)
        self.assertEqual(receipt["version_count_delta"], 1)
        with self.assertRaises(serial_launch.LaunchError):
            serial_launch.assert_version_delta(post, post, 2)

    def test_running_campaign_kernel_is_detected(self):
        rows = [
            {
                "ref": "onebitbrain/g02-other",
                "lastRunTime": serial_launch.now()[:10] + "T01:02:03",
            }
        ]

        def running(_command):
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout='Kernel has status "running"', stderr=""
            )

        self.assertEqual(
            serial_launch.recent_campaign_inflight(rows, running)[0]["ref"],
            "onebitbrain/g02-other",
        )

    def test_running_apncc_shared_lane_is_detected(self):
        rows = [{
            "ref": "onebitbrain/bhap-transport-smoke-v2",
            "lastRunTime": "2026-07-30T23:59:59",
        }]

        def running(_command):
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout='Kernel has status "running"', stderr=""
            )

        self.assertEqual(
            serial_launch.recent_campaign_inflight(rows, running)[0]["ref"],
            "onebitbrain/bhap-transport-smoke-v2",
        )

    def test_precommitted_transport_stop_rule_is_machine_classified(self):
        self.assertEqual(
            serial_launch.classify_terminal(
                "ERROR",
                "Step0InputError: offline wheel SHA-256 mismatch",
                output_downloaded=False,
            ),
            "STOP_RULE_TRANSPORT_FAILURE_PARK_B_NO_SEVENTH_RUN",
        )
        self.assertEqual(
            serial_launch.classify_terminal(
                "COMPLETE", "ok", output_downloaded=True
            ),
            "FEATURES_COMPLETE_PENDING_FROZEN_GATE",
        )
        self.assertEqual(
            serial_launch.classify_terminal(
                "ERROR",
                "FeatureError: authenticated projection exceeds five hours",
                output_downloaded=False,
            ),
            "INVALID_NONTRANSPORT_NO_VERDICT_AUTHORIZATION_CONSUMED",
        )

    def test_budget_is_derived_from_quota_and_postrun_debit_is_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(
                json.dumps(
                    {
                        "active_authorizations": [],
                        "debits": [],
                        "used_kernel_h": 2.88,
                        "remaining_kernel_h": 27.12,
                        "authenticated_capacity_basis": {},
                    }
                )
            )
            quota = {
                "used_h": 4.53,
                "remaining_h": 25.47,
                "total_h": 30.0,
                "queried_utc": serial_launch.now(),
                "source": "fixture",
            }
            observed = serial_launch.reconcile_budget(
                path,
                quota,
                phase="postrun",
                debit_id="ONE",
                debit_h=0.25,
            )
            self.assertEqual(observed["used_kernel_h"], 4.53)
            self.assertEqual(observed["remaining_kernel_h"], 25.47)
            self.assertEqual(observed["accounting_authority"]["local_file_role"],
                             "derived mirror plus named-run audit trail")
            with self.assertRaisesRegex(serial_launch.LaunchError, "duplicate"):
                serial_launch.reconcile_budget(
                    path,
                    quota,
                    phase="postrun",
                    debit_id="ONE",
                    debit_h=0.25,
                )


if __name__ == "__main__":
    unittest.main()

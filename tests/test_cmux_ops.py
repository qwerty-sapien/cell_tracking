#!/usr/bin/env python3
"""Tests for scripts/cmux_ops.py. No network: a fake `kaggle` executable is used throughout."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cmux_ops  # noqa: E402


def make_kernel_package_and_predicate(
    tmp: Path,
    kernel_id: str = "o/k",
    *,
    expected_initial_kernel_version: int | None = None,
) -> tuple[Path, Path]:
    package = tmp / "kernel"
    package.mkdir()
    code = package / "script.py"
    code.write_text("print('ok')\n")
    metadata = package / "kernel-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": kernel_id.split("/", 1)[-1],
                "code_file": code.name,
            }
        )
    )
    predicate = tmp / "kernel-predicate.json"
    predicate_payload = {
                "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
                "status": "PASS",
                "eligible": True,
                "phase": "inline",
                "failures": [],
                "kernel_id": kernel_id,
                "kernel_dir": str(package.resolve()),
                "metadata_sha256": cmux_ops._sha256_file(metadata),
                "code_file": str(code),
                "code_sha256": cmux_ops._sha256_file(code),
                "contract_sha256": "a" * 64,
                "registry_sha256": "b" * 64,
            }
    if expected_initial_kernel_version is not None:
        predicate_payload["expected_initial_kernel_version"] = (
            expected_initial_kernel_version
        )
    predicate.write_text(json.dumps(predicate_payload))
    return package, predicate


def make_fake_kaggle(
    tmp: Path,
    *,
    stdout: str = "",
    stderr: str = "",
    rc: int = 0,
    status_stdout: str | None = None,
    status_stderr: str | None = None,
    status_rc: int | None = None,
) -> str:
    """Write a fake `kaggle` that records argv and emits fixed output."""
    argv_log = tmp / "argv.log"
    script = tmp / "fake_kaggle"
    if status_stdout is None:
        status_stdout = (
            'has status "KernelWorkerStatus.COMPLETE"\n'
            if stdout.lstrip().startswith(("{", "["))
            else stdout
        )
    if status_stderr is None:
        status_stderr = stderr
    if status_rc is None:
        status_rc = rc
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json, pathlib\n"
        f"pathlib.Path({str(argv_log)!r}).open('a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "is_status = sys.argv[1:3] == ['kernels', 'status']\n"
        f"sys.stdout.write({status_stdout!r} if is_status else {stdout!r})\n"
        f"sys.stderr.write({status_stderr!r} if is_status else {stderr!r})\n"
        f"sys.exit({status_rc} if is_status else {rc})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def make_submission_predicate(tmp: Path, artifact: Path) -> Path:
    predicate = tmp / "submission-predicate.json"
    predicate.write_text(
        json.dumps(
            {
                "schema": "celltrack-alpha.test-submission-predicate.v1",
                "status": "PASS",
                "eligible": True,
                "competition": cmux_ops.DEFAULT_COMPETITION,
                "submission_mode": "direct_file",
                "revoked": False,
                "consumed": False,
                "failures": [],
                "artifact_path": str(artifact.resolve()),
                "artifact_sha256": cmux_ops._sha256_file(artifact),
                "artifact_bytes": artifact.stat().st_size,
            }
        )
    )
    return predicate


def make_gpu_ledger(tmp: Path, *, active: list[dict] | None = None) -> Path:
    path = tmp / "shared-gpu-ledger.json"
    path.write_text(
        json.dumps(
            {
                "schema": cmux_ops.GPU_LEDGER_SCHEMA,
                "capacity_h": 30.0,
                "refresh_at": "2026-08-08T00:00:00",
                "active_reservations": active or [],
                "events": [],
                "latest_authoritative_quota": None,
                "updated_utc": "2026-08-03T00:00:00Z",
            }
        )
    )
    return path


class TestCommandPlan(unittest.TestCase):
    def test_unknown_op_refused(self):
        with self.assertRaises(cmux_ops.UnknownOp):
            cmux_ops.run("kernels_delete", kaggle="/bin/true")

    def test_read_only_inventory_ops_are_bounded(self):
        plan = cmux_ops.command_plan("kaggle", dataset_id="o/d")
        self.assertEqual(
            plan["dataset_files"],
            ["kaggle", "datasets", "files", "o/d", "--page-size", "200", "--format", "json"],
        )
        self.assertEqual(
            plan["dataset_search_mine"],
            ["kaggle", "datasets", "list", "--mine", "--search", "d", "--format", "json"],
        )
        self.assertEqual(
            plan["kernel_list_mine"],
            ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--format", "json"],
        )
        self.assertEqual(
            plan["dataset_create"],
            ["kaggle", "datasets", "create", "-p", "", "--dir-mode", "zip"],
        )

    def test_kernel_output_can_bind_an_exact_version(self):
        plan = cmux_ops.command_plan(
            "kaggle", kernel_id="owner/kernel", kernel_version=7, path="/output"
        )
        self.assertEqual(
            plan["kernel_output"],
            ["kaggle", "kernels", "output", "owner/kernel/7", "-p", "/output"],
        )
        self.assertEqual(
            plan["kernel_status"],
            ["kaggle", "kernels", "status", "owner/kernel/7"],
        )
        self.assertEqual(
            plan["kernel_logs"],
            ["kaggle", "kernels", "logs", "owner/kernel/7"],
        )
        self.assertNotIn("kernel_logs", cmux_ops.MUTATING_OPS)

    def test_zero_exit_mutation_error_text_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake = make_fake_kaggle(
                tmp,
                stdout="Skipping folder: handoff\nDataset creation error: Please upload at least one file\n",
                rc=0,
            )
            receipt = cmux_ops.run(
                "dataset_create",
                kaggle=fake,
                path="/p",
                lock_path=tmp / "mutation.lock",
            )
            self.assertEqual(receipt["process_returncode"], 0)
            self.assertEqual(receipt["returncode"], 1)
            self.assertTrue(receipt["semantic_failure"])
            self.assertIn("Dataset creation error", receipt["semantic_failure_reason"])

    def test_plan_is_argv_lists_never_shell_strings(self):
        plan = cmux_ops.command_plan("kaggle", dataset_id="o/d", kernel_id="o/k", path="/p")
        for op, argv in plan.items():
            self.assertIsInstance(argv, list, op)
            for word in argv:
                self.assertIsInstance(word, str, op)
                # No shell metacharacters smuggled into a single word.
                self.assertNotIn("&&", word, op)
                self.assertNotIn(";", word, op)
                self.assertNotIn("|", word, op)

    def test_plan_surface_is_exactly_sixteen_ops(self):
        self.assertEqual(len(cmux_ops.command_plan()), 16)

    def test_submission_monitor_is_read_only_and_json_bounded(self):
        plan = cmux_ops.command_plan(
            "kaggle", competition="biohub-cell-tracking-during-development"
        )
        self.assertEqual(
            plan["competition_submissions"],
            [
                "kaggle",
                "competitions",
                "submissions",
                "biohub-cell-tracking-during-development",
                "--page-size",
                "50",
                "--format",
                "json",
            ],
        )
        self.assertNotIn("competition_submissions", cmux_ops.MUTATING_OPS)

    def test_kernel_pull_is_read_only_and_bounded(self):
        plan = cmux_ops.command_plan("kaggle", kernel_id="o/k/1", path="/p")
        self.assertEqual(
            plan["kernel_pull"],
            ["kaggle", "kernels", "pull", "o/k/1", "-p", "/p", "--metadata"],
        )
        self.assertNotIn("kernel_pull", cmux_ops.MUTATING_OPS)

    def test_no_interpolation_of_caller_strings_into_command_words(self):
        evil = "x; rm -rf /"
        plan = cmux_ops.command_plan("kaggle", dataset_id=evil)
        # The value appears only as its own argv element, never concatenated.
        self.assertIn(evil, plan["dataset_status"])
        self.assertEqual(plan["dataset_status"].count(evil), 1)


class TestScrubbing(unittest.TestCase):
    def test_secrets_removed(self):
        for probe in (
            "api_key: abc123def456",
            "Authorization: Bearer zzz",
            "token=supersecretvalue",
            "password = hunter2hunter2",
        ):
            self.assertIn("[REDACTED]", cmux_ops.scrub(probe), probe)

    def test_long_opaque_run_removed(self):
        self.assertIn("[REDACTED]", cmux_ops.scrub("k " + "A" * 48))

    def test_receipt_never_contains_a_token(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            secret = "aaaaBBBBccccDDDDeeeeFFFFgggg1111"
            fake = make_fake_kaggle(tmp, stdout=f"api_key: {secret}\n")
            r = cmux_ops.run("quota", kaggle=fake, receipt_dir=tmp / "receipts")
            body = Path(r["receipt_path"]).read_text()
            self.assertNotIn(secret, body)
            self.assertIn("[REDACTED]", body)
            # The hash of the true output is still recorded, so evidence is not lost.
            self.assertTrue(r["stdout_sha256"])


class TestKaggleAuthentication(unittest.TestCase):
    def test_fake_executable_never_receives_or_reads_credentials(self):
        self.assertIsNone(cmux_ops._authenticated_kaggle_environment("/tmp/fake_kaggle"))

    def test_real_cli_uses_only_a_valid_access_token(self):
        with tempfile.TemporaryDirectory() as td:
            credentials = Path(td) / "credentials.json"
            token = "a" * 40
            credentials.write_text(json.dumps({"access_token": token}))
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                cmux_ops, "_kaggle_credentials_path", return_value=credentials
            ), mock.patch.object(cmux_ops.shutil, "which", return_value="/opt/bin/kaggle"):
                env = cmux_ops._authenticated_kaggle_environment("/opt/bin/kaggle")
        self.assertEqual(env["KAGGLE_API_TOKEN"], token)

    def test_arbitrary_executable_named_kaggle_never_receives_token(self):
        with mock.patch.object(cmux_ops.shutil, "which", return_value="/trusted/kaggle"):
            self.assertIsNone(cmux_ops._authenticated_kaggle_environment("/tmp/kaggle"))

    def test_real_cli_refuses_missing_or_malformed_credential(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.json"
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                cmux_ops, "_kaggle_credentials_path", return_value=missing
            ), mock.patch.object(
                cmux_ops.shutil, "which", return_value="/opt/bin/kaggle"
            ), self.assertRaisesRegex(cmux_ops.CmuxOpsError, "unavailable"):
                cmux_ops._authenticated_kaggle_environment("kaggle")
            malformed = Path(td) / "malformed.json"
            malformed.write_text(json.dumps({"access_token": "short"}))
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                cmux_ops, "_kaggle_credentials_path", return_value=malformed
            ), mock.patch.object(
                cmux_ops.shutil, "which", return_value="/opt/bin/kaggle"
            ), self.assertRaisesRegex(cmux_ops.CmuxOpsError, "malformed"):
                cmux_ops._authenticated_kaggle_environment("kaggle")


class TestSerialLock(unittest.TestCase):
    def test_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            with cmux_ops.serial_lock(p, node="a"):
                with self.assertRaises(cmux_ops.LockHeld):
                    cmux_ops.serial_lock(p, node="b").acquire()

    def test_released_on_exception(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            with self.assertRaises(ValueError):
                with cmux_ops.serial_lock(p, node="a"):
                    raise ValueError("boom")
            self.assertTrue(p.exists())
            cmux_ops.serial_lock(p, node="b").acquire().release()
            self.assertTrue(p.exists())

    def test_stale_lock_not_broken_while_run_in_flight(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            old = cmux_ops.serial_lock(p, node="old").acquire()
            p.write_text(json.dumps({"acquired_monotonic": 0.0}))  # ancient
            with self.assertRaises(cmux_ops.LockHeld):
                cmux_ops.serial_lock(
                    p, node="new", stale_after_s=1.0,
                    status_probe=lambda: 'status "KernelWorkerStatus.RUNNING"',
                ).acquire()
            old.release()

    def test_age_and_terminal_probe_never_break_a_live_os_lock(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            old = cmux_ops.serial_lock(p, node="old").acquire()
            p.write_text(json.dumps({"acquired_monotonic": 0.0}))
            with self.assertRaises(cmux_ops.LockHeld):
                cmux_ops.serial_lock(
                    p, node="new", stale_after_s=1.0,
                    status_probe=lambda: 'status "KernelWorkerStatus.COMPLETE"',
                ).acquire()
            old.release()
            lk = cmux_ops.serial_lock(p, node="new").acquire()
            self.assertIsNone(lk.broke_stale)
            lk.release()

    def test_stale_lock_not_broken_without_probe_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            old = cmux_ops.serial_lock(p, node="old").acquire()
            p.write_text(json.dumps({"acquired_monotonic": 0.0}))
            with self.assertRaises(cmux_ops.LockHeld):
                cmux_ops.serial_lock(p, node="new", stale_after_s=1.0).acquire()
            old.release()

    def test_stale_lock_not_broken_by_unparseable_or_failed_probe(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            old = cmux_ops.serial_lock(p, node="old").acquire()
            p.write_text(json.dumps({"acquired_monotonic": 0.0}))
            with self.assertRaisesRegex(cmux_ops.LockHeld, "refusing to wait"):
                cmux_ops.serial_lock(
                    p, node="new", stale_after_s=1.0,
                    status_probe=lambda: "HTTP 503 upstream error",
                ).acquire()
            self.assertTrue(p.exists())
            old.release()

    def test_release_never_unlinks_a_replacement_owner(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            owner = cmux_ops.serial_lock(p, node="old").acquire()
            with self.assertRaises(cmux_ops.LockHeld):
                cmux_ops.serial_lock(p, node="new").acquire()
            owner.release()
            replacement = cmux_ops.serial_lock(p, node="new").acquire()
            self.assertTrue(p.exists())
            replacement.release()
            self.assertTrue(p.exists())

    def test_unlocked_unparseable_anchor_does_not_block(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lock"
            p.write_text("not json at all")
            lock = cmux_ops.serial_lock(
                p, stale_after_s=1.0, status_probe=lambda: "COMPLETE"
            ).acquire()
            self.assertEqual(json.loads(p.read_text())["pid"], os.getpid())
            lock.release()


class TestSubmissionPredicate(unittest.TestCase):
    def test_refused_without_receipt(self):
        with self.assertRaises(cmux_ops.PredicateNotSatisfied):
            cmux_ops.run("competition_submit", kaggle="/bin/true", csv="s.csv")

    def test_refused_when_ineligible(self):
        with tempfile.TemporaryDirectory() as td:
            rec = Path(td) / "pred.json"
            rec.write_text(json.dumps({"eligible": False}))
            with self.assertRaises(cmux_ops.PredicateNotSatisfied):
                cmux_ops.run(
                    "competition_submit", kaggle="/bin/true", csv="s.csv",
                    submission_predicate_receipt=rec,
                )

    def test_refused_when_receipt_unreadable(self):
        with self.assertRaises(cmux_ops.PredicateNotSatisfied):
            cmux_ops.run(
                "competition_submit", kaggle="/bin/true", csv="s.csv",
                submission_predicate_receipt="/nonexistent/pred.json",
            )

    def test_allowed_when_eligible(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            artifact = tmp / "submission.csv"
            artifact.write_text("row_type,dataset,node_id,t,z,y,x,source_id,target_id\n")
            rec = make_submission_predicate(tmp, artifact)
            fake = make_fake_kaggle(tmp, stdout="submitted\n")
            with mock.patch.object(
                cmux_ops, "DEFAULT_SUBMISSION_CLAIM_DIR", tmp / "claims"
            ):
                r = cmux_ops.run(
                    "competition_submit", kaggle=fake, csv=str(artifact), message="m",
                    submission_predicate_receipt=rec,
                    lock_path=tmp / "mutation.lock",
                )
            self.assertEqual(r["returncode"], 0)
            self.assertEqual(r["submission_predicate"]["artifact_sha256"], cmux_ops._sha256_file(artifact))

            with mock.patch.object(
                cmux_ops, "DEFAULT_SUBMISSION_CLAIM_DIR", tmp / "claims"
            ), self.assertRaisesRegex(cmux_ops.PredicateNotSatisfied, "already consumed"):
                cmux_ops.run(
                    "competition_submit", kaggle=fake, csv=str(artifact), message="m",
                    submission_predicate_receipt=rec,
                    lock_path=tmp / "different.lock",
                    receipt_dir=tmp / "different-receipts",
                )

    def test_bare_eligible_flag_is_not_authority(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rec = tmp / "pred.json"
            rec.write_text(json.dumps({"eligible": True}))
            with self.assertRaises(cmux_ops.PredicateNotSatisfied):
                cmux_ops.run(
                    "competition_submit",
                    kaggle="/bin/true",
                    csv=str(tmp / "submission.csv"),
                    submission_predicate_receipt=rec,
                    lock_path=tmp / "mutation.lock",
                )


class TestPushRace(unittest.TestCase):
    def test_push_refused_without_fresh_prelaunch_receipt(self):
        with self.assertRaises(cmux_ops.KernelPredicateNotSatisfied):
            cmux_ops.run("kernel_push", kaggle="/bin/true", kernel_id="o/k", path="/p")

    def test_push_refused_while_running(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            fake = make_fake_kaggle(tmp, stdout='has status "KernelWorkerStatus.RUNNING"\n')
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaises(cmux_ops.PushRefused):
                    cmux_ops.run(
                        "kernel_push",
                        kaggle=fake,
                        kernel_id="o/k",
                        path=package,
                        kernel_push_predicate_receipt=predicate,
                        _held_lock=held,
                    )

    def test_push_refused_while_queued(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            fake = make_fake_kaggle(tmp, stdout='has status "KernelWorkerStatus.QUEUED"\n')
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaises(cmux_ops.PushRefused):
                    cmux_ops.run(
                        "kernel_push",
                        kaggle=fake,
                        kernel_id="o/k",
                        path=package,
                        kernel_push_predicate_receipt=predicate,
                        _held_lock=held,
                    )

    def test_push_allowed_when_complete(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            fake = make_fake_kaggle(tmp, stdout='has status "KernelWorkerStatus.COMPLETE"\n')
            with cmux_ops.serial_lock(tmp / "lock") as held:
                r = cmux_ops.run(
                    "kernel_push",
                    kaggle=fake,
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                    _held_lock=held,
                )
            self.assertEqual(r["returncode"], 0)
            self.assertEqual(r["kernel_push_predicate"]["kernel_id"], "o/k")

    def test_explicit_fresh_v1_push_accepts_exact_absence_and_records_guard(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(
                tmp, expected_initial_kernel_version=1
            )
            fake = make_fake_kaggle(
                tmp,
                stdout="Kernel version 1 successfully pushed.\n",
                status_stdout="",
                status_stderr=(
                    "Cannot access kernel 'o/k' (Permission 'kernels.get' was denied). "
                    "The most likely cause is a wrong kernel slug.\n"
                ),
                status_rc=1,
            )
            with cmux_ops.serial_lock(tmp / "lock") as held:
                result = cmux_ops.run(
                    "kernel_push",
                    kaggle=fake,
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                    receipt_dir=tmp / "receipts",
                    _held_lock=held,
                )
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(
                result["kernel_push_status_guard"]["disposition"],
                "FRESH_SLUG_ABSENT_EXPECT_VERSION_1",
            )
            calls = (tmp / "argv.log").read_text().strip().splitlines()
            self.assertEqual(len(calls), 2)

    def test_explicit_fresh_v1_push_refuses_an_existing_terminal_slug(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(
                tmp, expected_initial_kernel_version=1
            )
            fake = make_fake_kaggle(
                tmp, status_stdout='has status "KernelWorkerStatus.COMPLETE"\n'
            )
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaisesRegex(cmux_ops.PushRefused, "fresh exact Version 1"):
                    cmux_ops.run(
                        "kernel_push",
                        kaggle=fake,
                        kernel_id="o/k",
                        path=package,
                        kernel_push_predicate_receipt=predicate,
                        _held_lock=held,
                    )
            calls = (tmp / "argv.log").read_text().strip().splitlines()
            self.assertEqual(len(calls), 1)

    def test_push_refuses_failed_or_unparseable_latest_status(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            fake = make_fake_kaggle(tmp, stdout="not a kernel status\n")
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaisesRegex(cmux_ops.PushRefused, "unparseable"):
                    cmux_ops.run(
                        "kernel_push", kaggle=fake, kernel_id="o/k", path=package,
                        kernel_version=1,
                        kernel_push_predicate_receipt=predicate, _held_lock=held,
                    )
            status_argv = json.loads((tmp / "argv.log").read_text().splitlines()[0])
            self.assertEqual(status_argv[2], "o/k")

    def test_push_refuses_incomplete_as_not_an_exact_complete_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            fake = make_fake_kaggle(
                tmp, stdout='has status "KernelWorkerStatus.INCOMPLETE"\n'
            )
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaisesRegex(cmux_ops.PushRefused, "unparseable"):
                    cmux_ops.run(
                        "kernel_push", kaggle=fake, kernel_id="o/k", path=package,
                        kernel_push_predicate_receipt=predicate, _held_lock=held,
                    )

    def test_gpu_quota_rejects_nonfinite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(cmux_ops.CmuxOpsError):
                cmux_ops.parse_authoritative_gpu_quota(
                    {"used_h": 1.0, "remaining_h": value, "total_h": 30.0}
                )

    def test_direct_valid_push_is_refused_without_shared_budget_bracket(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            with self.assertRaisesRegex(cmux_ops.BudgetLedgerError, "direct kernel_push"):
                cmux_ops.run(
                    "kernel_push",
                    kaggle="/bin/true",
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                )

    def test_push_refused_when_code_changes_after_prelaunch(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            (package / "script.py").write_text("print('changed')\n")
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaisesRegex(
                    cmux_ops.KernelPredicateNotSatisfied, "code changed after prelaunch"
                ):
                    cmux_ops.run(
                        "kernel_push",
                        kaggle="/bin/true",
                        kernel_id="o/k",
                        path=package,
                        kernel_push_predicate_receipt=predicate,
                        _held_lock=held,
                    )

    def test_push_refused_when_predicate_names_another_kernel(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            payload = json.loads(predicate.read_text())
            payload["kernel_id"] = "o/other"
            predicate.write_text(json.dumps(payload))
            with cmux_ops.serial_lock(tmp / "lock") as held:
                with self.assertRaisesRegex(
                    cmux_ops.KernelPredicateNotSatisfied, "kernel ID drift"
                ):
                    cmux_ops.run(
                        "kernel_push",
                        kaggle="/bin/true",
                        kernel_id="o/k",
                        path=package,
                        kernel_push_predicate_receipt=predicate,
                        _held_lock=held,
                    )


class TestBracketedLaunch(unittest.TestCase):
    def test_aborts_when_pre_query_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake = make_fake_kaggle(tmp, stdout="not json", rc=1)
            with self.assertRaises(cmux_ops.CmuxOpsError):
                cmux_ops.bracketed_launch(
                    "kernel_push", kaggle=fake, kernel_id="o/k", path="/p",
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                )

    def test_reports_debit_delta(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Every invocation returns the same quota, so the delta is 0.0 -- the point is
            # that the delta comes from the bracket, not from our own timing.
            fake = make_fake_kaggle(tmp, stdout=json.dumps({"used_h": 5.06, "remaining_h": 24.94}))
            out = cmux_ops.bracketed_launch(
                "dataset_version", kaggle=fake, path="/p", message="m",
                lock=cmux_ops.serial_lock(tmp / "lock"),
            )
            self.assertEqual(out["authoritative_debit_h"], 0.0)
            self.assertEqual(
                out["accounting_basis"],
                "owner-authenticated Kaggle pre/post quota difference",
            )

    def test_parses_the_real_gpu_quota_row(self):
        value = cmux_ops.parse_authoritative_gpu_quota(
            [
                {"resource": "CPU", "used": "0h", "remaining": "30h", "total": "30h"},
                {"resource": "GPU", "used": "5.43h", "remaining": "24.57h", "total": "30h", "refreshAt": "x"},
            ]
        )
        self.assertEqual(value, {"used_h": 5.43, "remaining_h": 24.57, "total_h": 30.0, "refresh_at": "x"})

    def test_aborts_before_mutation_on_ambiguous_gpu_quota(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake = make_fake_kaggle(tmp, stdout=json.dumps([
                {"resource": "GPU", "used": "1h", "remaining": "29h", "total": "30h"},
                {"resource": "GPU", "used": "1h", "remaining": "29h", "total": "30h"},
            ]))
            with self.assertRaises(cmux_ops.CmuxOpsError):
                cmux_ops.bracketed_launch(
                    "kernel_push", kaggle=fake, kernel_id="o/k", path="/p",
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                )
            calls = (tmp / "argv.log").read_text().strip().splitlines()
            self.assertEqual(len(calls), 1, "malformed pre-quota must prevent mutation")

    def test_kernel_push_requires_worst_case_gpu_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            ledger = make_gpu_ledger(tmp)
            fake = make_fake_kaggle(
                tmp,
                stdout=json.dumps(
                    {
                        "used_h": 1.37,
                        "remaining_h": 28.63,
                        "total_h": 30.0,
                        "refresh_at": "2026-08-08T00:00:00",
                    }
                ),
            )
            with self.assertRaisesRegex(cmux_ops.BudgetLedgerError, "positive"):
                cmux_ops.bracketed_launch(
                    "kernel_push",
                    kaggle=fake,
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                    gpu_ledger_path=ledger,
                )
            self.assertEqual(json.loads(ledger.read_text())["active_reservations"], [])

    def test_kernel_push_rejects_nan_gpu_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            ledger = make_gpu_ledger(tmp)
            fake = make_fake_kaggle(
                tmp,
                stdout=json.dumps(
                    {
                        "used_h": 1.0,
                        "remaining_h": 29.0,
                        "total_h": 30.0,
                        "refresh_at": "2026-08-08T00:00:00",
                    }
                ),
            )
            with self.assertRaisesRegex(cmux_ops.BudgetLedgerError, "positive"):
                cmux_ops.bracketed_launch(
                    "kernel_push", kaggle=fake, kernel_id="o/k", path=package,
                    kernel_push_predicate_receipt=predicate,
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                    gpu_reservation_h=float("nan"), gpu_ledger_path=ledger,
                )

    def test_kernel_push_reserves_in_single_shared_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            ledger = make_gpu_ledger(tmp)
            fake = make_fake_kaggle(
                tmp,
                stdout=json.dumps(
                    {
                        "used_h": 1.37,
                        "remaining_h": 28.63,
                        "total_h": 30.0,
                        "refresh_at": "2026-08-08T00:00:00",
                    }
                ),
            )
            out = cmux_ops.bracketed_launch(
                "kernel_push",
                kaggle=fake,
                kernel_id="o/k",
                path=package,
                kernel_push_predicate_receipt=predicate,
                lock=cmux_ops.serial_lock(tmp / "lock"),
                gpu_reservation_h=3.0,
                authenticated_projection_h=3.2,
                gpu_ledger_path=ledger,
                receipt_dir=tmp / "receipts",
            )
            payload = json.loads(ledger.read_text())
            self.assertEqual(out["gpu_reservation_h"], 3.0)
            self.assertEqual(out["projection_gate"]["disposition"], "AUTO_APPROVED_LT_8H")
            self.assertEqual(len(payload["active_reservations"]), 1)
            self.assertEqual(payload["active_reservations"][0]["kernel_id"], "o/k")
            self.assertEqual(payload["active_reservations"][0]["status"], "LAUNCHED_ACTIVE")

    def test_concurrent_reservations_cannot_jointly_overshoot(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            ledger = make_gpu_ledger(
                tmp,
                active=[
                    {
                        "kernel_id": "o/already-running",
                        "max_gpu_h": 27.0,
                        "status": "LAUNCHED_ACTIVE",
                    }
                ],
            )
            fake = make_fake_kaggle(
                tmp,
                stdout=json.dumps(
                    {
                        "used_h": 1.37,
                        "remaining_h": 28.63,
                        "total_h": 30.0,
                        "refresh_at": "2026-08-08T00:00:00",
                    }
                ),
            )
            with self.assertRaises(cmux_ops.InsufficientGpuQuota):
                cmux_ops.bracketed_launch(
                    "kernel_push",
                    kaggle=fake,
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                    gpu_reservation_h=3.0,
                    authenticated_projection_h=3.2,
                    gpu_ledger_path=ledger,
                )
            calls = (tmp / "argv.log").read_text().strip().splitlines()
            self.assertEqual(len(calls), 1, "capacity failure must precede status and push")

    def test_kernel_push_requires_projection_even_with_budget_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            package, predicate = make_kernel_package_and_predicate(tmp)
            ledger = make_gpu_ledger(tmp)
            fake = make_fake_kaggle(
                tmp,
                stdout=json.dumps(
                    {
                        "used_h": 1.0,
                        "remaining_h": 29.0,
                        "total_h": 30.0,
                        "refresh_at": "2026-08-08T00:00:00",
                    }
                ),
            )
            with self.assertRaisesRegex(cmux_ops.ProjectionApprovalRequired, "projection"):
                cmux_ops.bracketed_launch(
                    "kernel_push",
                    kaggle=fake,
                    kernel_id="o/k",
                    path=package,
                    kernel_push_predicate_receipt=predicate,
                    lock=cmux_ops.serial_lock(tmp / "lock"),
                    gpu_reservation_h=3.0,
                    gpu_ledger_path=ledger,
                )
            self.assertEqual(json.loads(ledger.read_text())["active_reservations"], [])

    def test_terminal_reconciliation_releases_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ledger = make_gpu_ledger(
                tmp,
                active=[
                    {
                        "kernel_id": "o/done",
                        "max_gpu_h": 3.0,
                        "status": "LAUNCHED_ACTIVE",
                    }
                ],
            )
            terminal = tmp / "terminal.json"
            terminal.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.cmux-op-receipt.v1",
                        "op": "kernel_status",
                        "argv": ["kaggle", "kernels", "status", "o/done"],
                        "returncode": 0,
                        "stdout_scrubbed": "KernelWorkerStatus.COMPLETE",
                    }
                )
            )
            cmux_ops.reconcile_gpu_reservation(
                "o/done",
                terminal_status="COMPLETE",
                quota={
                    "used_h": 2.0,
                    "remaining_h": 28.0,
                    "total_h": 30.0,
                    "refresh_at": "2026-08-08T00:00:00",
                },
                ledger_path=ledger,
                lock_path=tmp / "lock",
                terminal_receipt=str(terminal),
            )
            payload = json.loads(ledger.read_text())
            self.assertEqual(payload["active_reservations"], [])
            self.assertEqual(payload["events"][-1]["terminal_status"], "COMPLETE")

    def test_active_error_reconciliation_requires_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ledger = make_gpu_ledger(
                tmp,
                active=[
                    {
                        "kernel_id": "o/active",
                        "max_gpu_h": 3.0,
                        "status": "LAUNCHED_ACTIVE",
                    }
                ],
            )
            with self.assertRaisesRegex(
                cmux_ops.BudgetLedgerError, "kernel-bound terminal evidence"
            ):
                cmux_ops.reconcile_gpu_reservation(
                    "o/active", terminal_status="ERROR",
                    ledger_path=ledger, lock_path=tmp / "lock",
                )
            self.assertEqual(
                json.loads(ledger.read_text())["active_reservations"][0]["kernel_id"],
                "o/active",
            )

    def test_failed_push_receipt_must_bind_the_reserved_kernel(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ledger = make_gpu_ledger(
                tmp,
                active=[
                    {
                        "kernel_id": "o/failed",
                        "max_gpu_h": 3.0,
                        "status": "LAUNCH_FAILED",
                    }
                ],
            )
            receipt = tmp / "failed-push.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.cmux-op-receipt.v1",
                        "op": "kernel_push",
                        "argv": ["kaggle", "kernels", "push", "-p", "/tmp/package"],
                        "returncode": 1,
                        "kernel_push_predicate": {"kernel_id": "o/other"},
                    }
                )
            )
            with self.assertRaisesRegex(
                cmux_ops.BudgetLedgerError, "does not prove this kernel"
            ):
                cmux_ops.reconcile_gpu_reservation(
                    "o/failed", terminal_status="ERROR",
                    terminal_receipt=str(receipt), ledger_path=ledger,
                    lock_path=tmp / "lock",
                )
            malformed = json.loads(receipt.read_text())
            malformed["kernel_push_predicate"]["kernel_id"] = "o/failed"
            malformed.pop("returncode")
            receipt.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(
                cmux_ops.BudgetLedgerError, "does not prove this kernel"
            ):
                cmux_ops.reconcile_gpu_reservation(
                    "o/failed", terminal_status="ERROR",
                    terminal_receipt=str(receipt), ledger_path=ledger,
                    lock_path=tmp / "lock",
                )

    def test_terminal_status_receipt_rejects_substring_matches(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ledger = make_gpu_ledger(
                tmp,
                active=[
                    {
                        "kernel_id": "o/done",
                        "max_gpu_h": 3.0,
                        "status": "LAUNCHED_ACTIVE",
                    }
                ],
            )
            receipt = tmp / "ambiguous-status.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.cmux-op-receipt.v1",
                        "op": "kernel_status",
                        "argv": ["kaggle", "kernels", "status", "o/done"],
                        "returncode": 0,
                        "stdout_scrubbed": "KernelWorkerStatus.INCOMPLETE",
                    }
                )
            )
            with self.assertRaisesRegex(
                cmux_ops.BudgetLedgerError, "does not prove this kernel"
            ):
                cmux_ops.reconcile_gpu_reservation(
                    "o/done", terminal_status="COMPLETE",
                    terminal_receipt=str(receipt), ledger_path=ledger,
                    lock_path=tmp / "lock",
                )


class TestNoRetry(unittest.TestCase):
    def test_failure_returns_receipt_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake = make_fake_kaggle(tmp, stderr="403 Forbidden\n", rc=1)
            r = cmux_ops.run("dataset_status", kaggle=fake, dataset_id="o/d")
            self.assertEqual(r["returncode"], 1)
            self.assertIn("403", r["stderr_scrubbed"])
            calls = (tmp / "argv.log").read_text().strip().splitlines()
            self.assertEqual(len(calls), 1, "a failed op must not be retried")


if __name__ == "__main__":
    unittest.main(verbosity=2)

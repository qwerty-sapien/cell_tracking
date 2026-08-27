#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from content_addressed_step0 import (
    DiscoveryError,
    DiscoveryPins,
    discover_requests_by_content,
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_fixture(
    root: Path,
    *,
    empty: bool = False,
    duplicate_manifest: bool = False,
    duplicate_payload: bool = False,
) -> DiscoveryPins:
    population_sha = "3" * 64
    payload = b"request-npz-fixture"
    payload_sha = digest(payload)
    payload_path = root / "arbitrary" / "payload.data"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(payload)
    if duplicate_payload:
        duplicate = root / "other" / "same-content.bin"
        duplicate.parent.mkdir()
        duplicate.write_bytes(payload)

    requests = (
        []
        if empty
        else [
            {
                "action_class": "delete",
                "identity": "train:44b6_fixture",
                "action_count": 7,
                "bytes": len(payload),
                "path": "this-name-is-never-used.npz",
                "sha256": payload_sha,
            }
        ]
    )
    request_manifest = {
        "schema": "fixture",
        "status": "FROZEN_COMPLETE",
        "population_manifest_sha256": population_sha,
        "request_count": 0 if empty else 7,
        "request_file_count": 0 if empty else 1,
        "requests": requests,
    }
    request_payload = json.dumps(
        request_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    request_sha = digest(request_payload)
    request_path = root / "renamed" / "manifest.payload"
    request_path.parent.mkdir()
    request_path.write_bytes(request_payload)

    files = [
        {
            "role": "request_manifest",
            "path": "unused-request-name.json",
            "bytes": len(request_payload),
            "sha256": request_sha,
        }
    ]
    if not empty:
        files.append(
            {
                "role": "feature_request",
                "path": "unused-payload-name.npz",
                "bytes": len(payload),
                "sha256": payload_sha,
                "action_class": "delete",
                "identity": "train:44b6_fixture",
                "action_count": 7,
            }
        )
    bundle = {
        "status": "LOCAL_CONSOLIDATED_BYTE_EXACT",
        "files": files,
    }
    bundle_payload = json.dumps(
        bundle, sort_keys=True, separators=(",", ":")
    ).encode()
    bundle_sha = digest(bundle_payload)
    bundle_path = root / "object-store" / "opaque.bin"
    bundle_path.parent.mkdir()
    bundle_path.write_bytes(bundle_payload)
    if duplicate_manifest:
        duplicate = root / "duplicate" / "another.bin"
        duplicate.parent.mkdir()
        duplicate.write_bytes(bundle_payload)
    return DiscoveryPins(
        bundle_manifest_sha256=bundle_sha,
        request_manifest_sha256=request_sha,
        population_manifest_sha256=population_sha,
        request_file_count=1,
        request_count=7,
    )


class ContentAddressedStep0Test(unittest.TestCase):
    def test_positive_ignores_all_names_and_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root)
            result = discover_requests_by_content(root, pins)
            self.assertEqual(result["status"], "PASS_UNIQUE_CONTENT_DISCOVERY")
            self.assertEqual(result["request_file_count"], 1)
            self.assertEqual(result["request_count"], 7)
            self.assertEqual(result["request_records"][0]["action_class"], "delete")
            self.assertEqual(
                result["request_records"][0]["identity"], "train:44b6_fixture"
            )
            self.assertFalse(result["heavy_work_started"])

    def test_missing_bundle_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root)
            pins = DiscoveryPins(
                bundle_manifest_sha256="0" * 64,
                request_manifest_sha256=pins.request_manifest_sha256,
                population_manifest_sha256=pins.population_manifest_sha256,
                request_file_count=pins.request_file_count,
                request_count=pins.request_count,
            )
            with self.assertRaisesRegex(DiscoveryError, "found 0"):
                discover_requests_by_content(root, pins)

    def test_empty_request_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root, empty=True)
            with self.assertRaisesRegex(DiscoveryError, "empty or has invalid"):
                discover_requests_by_content(root, pins)

    def test_duplicate_bundle_manifest_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root, duplicate_manifest=True)
            with self.assertRaisesRegex(DiscoveryError, "found 2"):
                discover_requests_by_content(root, pins)

    def test_wrong_request_manifest_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root)
            pins = DiscoveryPins(
                bundle_manifest_sha256=pins.bundle_manifest_sha256,
                request_manifest_sha256="f" * 64,
                population_manifest_sha256=pins.population_manifest_sha256,
                request_file_count=pins.request_file_count,
                request_count=pins.request_count,
            )
            with self.assertRaisesRegex(DiscoveryError, "disagrees"):
                discover_requests_by_content(root, pins)

    def test_duplicate_request_payload_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root, duplicate_payload=True)
            with self.assertRaisesRegex(DiscoveryError, "found 2"):
                discover_requests_by_content(root, pins)

    def test_wrong_payload_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = write_fixture(root)
            request_path = root / "renamed" / "manifest.payload"
            request_manifest = json.loads(request_path.read_text())
            request_manifest["requests"][0]["sha256"] = "e" * 64
            altered_request = json.dumps(
                request_manifest, sort_keys=True, separators=(",", ":")
            ).encode()
            request_path.write_bytes(altered_request)
            altered_request_sha = digest(altered_request)

            bundle_path = root / "object-store" / "opaque.bin"
            bundle = json.loads(bundle_path.read_text())
            bundle["files"][0]["bytes"] = len(altered_request)
            bundle["files"][0]["sha256"] = altered_request_sha
            bundle["files"][1]["sha256"] = "e" * 64
            altered_bundle = json.dumps(
                bundle, sort_keys=True, separators=(",", ":")
            ).encode()
            bundle_path.write_bytes(altered_bundle)
            altered_pins = DiscoveryPins(
                bundle_manifest_sha256=digest(altered_bundle),
                request_manifest_sha256=altered_request_sha,
                population_manifest_sha256=pins.population_manifest_sha256,
                request_file_count=1,
                request_count=7,
            )
            with self.assertRaisesRegex(DiscoveryError, "found 0"):
                discover_requests_by_content(root, altered_pins)


if __name__ == "__main__":
    unittest.main()

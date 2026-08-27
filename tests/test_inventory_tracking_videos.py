#!/usr/bin/env python3
"""Adversarial tests for the canonical tracking-video inventory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/inventory_tracking_videos.py"
SPEC = importlib.util.spec_from_file_location("inventory_tracking_videos", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTrackingVideosTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalogue = self.root / "reports/g04/fixture"
        (self.catalogue / "videos").mkdir(parents=True)
        (self.catalogue / "metadata").mkdir()
        self.prediction_dir = self.root / "predictions"
        self.prediction_dir.mkdir()
        self.identity = "train:44b6_fixture"
        self.stem = "train__44b6_fixture"
        self.prediction = self.prediction_dir / f"{self.stem}.csv"
        self.prediction.write_text("source,target\n1,2\n", encoding="utf-8")
        self.video = self.catalogue / f"videos/{self.stem}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg is required")
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=2:d=1",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.video),
            ],
            check=True,
        )
        prediction_sha = inventory.sha256(self.prediction)
        video_sha = inventory.sha256(self.video)
        self.sidecar = self.catalogue / f"metadata/{self.stem}.json"
        self.sidecar.write_text(
            json.dumps(
                {
                    "schema": inventory.SIDECAR_SCHEMA,
                    "identity": self.identity,
                    "video": f"videos/{self.video.name}",
                    "video_sha256": video_sha,
                    "video_bytes": self.video.stat().st_size,
                    "prediction_fragment": str(self.prediction.relative_to(self.root)),
                    "prediction_fragment_sha256": prediction_sha,
                    "frames": 2,
                    "encoded_duration_seconds": 1.0,
                    "score": 0.5,
                    "rank": 1,
                }
            )
        )
        fragment_set = hashlib.sha256(
            f"{self.identity}\0{prediction_sha}\n".encode("utf-8")
        ).hexdigest()
        self.manifest = self.catalogue / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": "celltrack-alpha.track-visualizer-bundle.v1",
                    "status": "complete",
                    "generated_at": "2026-08-14T00:00:00Z",
                    "rendered_video_count": 1,
                    "indexed_video_count": 1,
                    "sources": {
                        "predictions": {
                            "path": str(self.prediction_dir.relative_to(self.root)),
                            "fragment_count": 1,
                            "fragment_set_sha256": fragment_set,
                            "fragments": [
                                {
                                    "identity": self.identity,
                                    "path": str(self.prediction.relative_to(self.root)),
                                    "bytes": self.prediction.stat().st_size,
                                    "sha256": prediction_sha,
                                }
                            ],
                        }
                    },
                    "videos": [
                        {
                            "identity": self.identity,
                            "rendered": True,
                            "video": f"videos/{self.video.name}",
                            "metadata": f"metadata/{self.sidecar.name}",
                        }
                    ],
                }
            )
        )
        (self.catalogue / "index.html").write_text(
            f'<a href="videos/{self.video.name}">video</a>\n', encoding="utf-8"
        )
        self.patches = [
            mock.patch.object(inventory, "ROOT", self.root),
            mock.patch.object(inventory, "REPORT", self.root / "inventory.json"),
            mock.patch.object(
                inventory,
                "CATALOGUES",
                (
                    (
                        "fixture",
                        self.catalogue,
                        "fixture_role",
                        str(self.prediction_dir.relative_to(self.root)),
                        "fixture_candidate",
                        False,
                    ),
                ),
            ),
            mock.patch.object(
                inventory,
                "CATALOGUE_MANIFEST_SCHEMAS",
                {"fixture": "celltrack-alpha.track-visualizer-bundle.v1"},
            ),
            mock.patch.object(
                inventory,
                "CANDIDATE_CLASSIFICATION",
                {
                    "fixture_candidate": {
                        "local_execution": False,
                        "prediction_sources": {
                            str(self.prediction_dir.relative_to(self.root))
                        },
                    }
                },
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def mutate_manifest(self, mutation) -> None:
        payload = json.loads(self.manifest.read_text())
        mutation(payload)
        self.manifest.write_text(json.dumps(payload))

    def reseal_identity(self, identity: str) -> None:
        manifest = json.loads(self.manifest.read_text())
        prediction_sha = manifest["sources"]["predictions"]["fragments"][0][
            "sha256"
        ]
        manifest["videos"][0]["identity"] = identity
        manifest["sources"]["predictions"]["fragments"][0]["identity"] = identity
        manifest["sources"]["predictions"]["fragment_set_sha256"] = hashlib.sha256(
            f"{identity}\0{prediction_sha}\n".encode("utf-8")
        ).hexdigest()
        self.manifest.write_text(json.dumps(manifest))
        sidecar = json.loads(self.sidecar.read_text())
        sidecar["identity"] = identity
        self.sidecar.write_text(json.dumps(sidecar))

    def test_exact_fixture_builds(self) -> None:
        result = inventory.build_inventory()
        self.assertEqual(result["physical_video_count"], 1)
        self.assertEqual(result["distinct_identity_count"], 1)

    def test_manifest_schema_drift_is_rejected(self) -> None:
        self.mutate_manifest(lambda value: value.update(schema="invented"))
        with self.assertRaisesRegex(inventory.InventoryError, "manifest schema"):
            inventory.build_inventory()

    def test_index_video_set_drift_is_rejected(self) -> None:
        (self.catalogue / "index.html").write_text("<html></html>\n")
        with self.assertRaisesRegex(inventory.InventoryError, "index/video exact-set"):
            inventory.build_inventory()

    def test_extra_sidecar_is_rejected(self) -> None:
        (self.catalogue / "metadata/extra.json").write_text("{}\n")
        with self.assertRaisesRegex(inventory.InventoryError, "exact-set drift"):
            inventory.build_inventory()

    def test_fragment_set_drift_is_rejected(self) -> None:
        self.mutate_manifest(
            lambda value: value["sources"]["predictions"].update(
                fragment_set_sha256="0" * 64
            )
        )
        with self.assertRaisesRegex(inventory.InventoryError, "fragment-set hash drift"):
            inventory.build_inventory()

    def test_box_like_fake_video_is_rejected_by_ffprobe(self) -> None:
        self.video.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00moovmdat")
        payload = json.loads(self.sidecar.read_text())
        payload["video_sha256"] = inventory.sha256(self.video)
        payload["video_bytes"] = self.video.stat().st_size
        self.sidecar.write_text(json.dumps(payload))
        with self.assertRaisesRegex(inventory.InventoryError, "ffprobe|video stream"):
            inventory.build_inventory()

    def test_local_execution_classification_drift_is_rejected(self) -> None:
        inventory.CANDIDATE_CLASSIFICATION["fixture_candidate"][
            "local_execution"
        ] = True
        with self.assertRaisesRegex(inventory.InventoryError, "classification drift"):
            inventory.build_inventory()

    def test_resealed_malformed_identities_are_rejected(self) -> None:
        original_manifest = self.manifest.read_bytes()
        original_sidecar = self.sidecar.read_bytes()
        for identity in (
            "train:",
            ":x",
            "train:../x",
            "train:x/y",
            "train:x\\y",
            "train:\x01x",
        ):
            with self.subTest(identity=repr(identity)):
                self.manifest.write_bytes(original_manifest)
                self.sidecar.write_bytes(original_sidecar)
                self.reseal_identity(identity)
                with self.assertRaisesRegex(
                    inventory.InventoryError, "must be split:dataset"
                ):
                    inventory.build_inventory()


if __name__ == "__main__":
    unittest.main()

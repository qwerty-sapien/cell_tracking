"""Private E4 package stays compact and fit-only."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_e4


class E4PackageTests(unittest.TestCase):
    def test_compact_parent_and_edge_registry_are_hash_bound(self) -> None:
        split = build_e4._load_exact(build_e4.SPLIT, "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769")
        provenance = build_e4._load_exact(build_e4.PROVENANCE, "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f")
        parent = build_e4._parent_projection(build_e4.REGISTRY, split, provenance)
        edge = build_e4._edge_registry(build_e4.REGISTRY, parent)
        parent_payload = json.loads(parent.decode("ascii"))
        edge_payload = json.loads(edge.decode("ascii"))
        self.assertEqual(parent_payload["schema"], "celltrack-alpha.v45-fast-projection.v1")
        self.assertEqual(parent_payload["requested_groups"], [])
        self.assertEqual(len(parent_payload["selected_sources"]), build_e4.PARENT_SOURCE_COUNT)
        self.assertEqual(edge_payload["artifact"], "celltrack-alpha.edgefit-registry-projection.v1")
        self.assertEqual(edge_payload["parent_projection_sha256"], build_e4._sha(parent))
        self.assertEqual(len(edge_payload["records"]), build_e4.PARENT_SOURCE_COUNT)
        self.assertEqual({row["video_id"] for row in edge_payload["records"]}, {row["video_id"] for row in parent_payload["selected_sources"]})

    def test_builds_under_kaggle_source_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "e4pkg"
            result = build_e4.build(output)
            self.assertLess(result["aggregate_bytes"], build_e4.SOURCE_LIMIT)
            self.assertTrue((output / build_e4.ENTRY).is_file())
            self.assertTrue((output / build_e4.METADATA).is_file())
            metadata = json.loads((output / build_e4.METADATA).read_text(encoding="ascii"))
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")


if __name__ == "__main__":
    unittest.main()

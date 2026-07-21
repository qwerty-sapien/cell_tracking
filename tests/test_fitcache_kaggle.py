from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts import build_fitcache_kaggle as fitcache


class FitCachePackageTests(unittest.TestCase):
    def test_metadata_pins_t4_accelerator(self) -> None:
        metadata = json.loads(fitcache._metadata("bh-fc00"))
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(
            metadata["machine_shape"], fitcache.pkgbh9.KAGGLE_ACCELERATOR_ID
        )

    def test_balanced_shards_are_complete_disjoint_and_byte_balanced(self) -> None:
        identities = tuple(f"train:v{index}" for index in range(7))
        sizes = {identity: index + 1 for index, identity in enumerate(identities)}
        shards = fitcache.balanced_shards(identities, sizes, 2)

        self.assertEqual(set().union(*(set(shard) for shard in shards)), set(identities))
        self.assertFalse(set(shards[0]) & set(shards[1]))
        totals = [sum(sizes[identity] for identity in shard) for shard in shards]
        self.assertLessEqual(max(totals) - min(totals), max(sizes.values()))

    def test_launcher_replacement_is_fit_only_and_submission_free(self) -> None:
        source = (
            "import json, signal, time\n"
            "def main():\n"
            " return 99\n"
            'if __name__=="__main__": raise SystemExit(main())\n'
        )
        identities = ("train:first", "train:second")
        digest = fitcache.sha256_bytes(fitcache.canonical_bytes(list(identities)))
        transformed = fitcache.transform_launcher(source, identities, digest)

        compile(transformed, "fitc.py", "exec")
        self.assertIn('"--mode","cv"', transformed)
        self.assertIn('"--split","train"', transformed)
        self.assertIn('"--graph-variant","a0"', transformed)
        self.assertIn('"competition_submission_performed":False', transformed)
        self.assertNotIn("return 99", transformed)
        encoded = transformed.split("FIT_IDS=json.loads(", 1)[1].split(")\n", 1)[0]
        self.assertEqual(json.loads(json.loads(encoded)), list(identities))

    def test_built_plan_covers_exact_fold_fit_partition_when_present(self) -> None:
        plan_path = fitcache.OUTPUT_ROOT / "plan.json"
        if not plan_path.is_file():
            self.skipTest("fit-cache package has not been built")
        payload = json.loads(plan_path.read_text(encoding="ascii"))
        plan = payload["plan"]
        fit = set(plan["fit_video_ids"])
        existing = set(plan["already_complete_video_ids"])
        shards = [set(row["video_ids"]) for row in plan["shards"]]

        self.assertEqual(len(fit), 117)
        self.assertEqual(existing | set().union(*shards), fit)
        self.assertFalse(existing & set().union(*shards))
        self.assertFalse(shards[0] & shards[1])
        self.assertGreater(plan["fit_compressed_bytes"], 50_000_000_000)
        self.assertTrue(plan["data_boundary"]["train_fit_only"])
        self.assertFalse(plan["data_boundary"]["competition_submission_performed"])


if __name__ == "__main__":
    unittest.main()

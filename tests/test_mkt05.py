import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class T05UploadTests(unittest.TestCase):
    def test_private_notebook_upload_has_only_attested_inputs(self):
        module = load("mkt05_test", ROOT / "scripts" / "mkt05.py")
        module.check()
        metadata = json.loads((ROOT / "kaggle_kernel" / "t05pkg" / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "onebitbrain/biohub-t05d")
        self.assertEqual(metadata["code_file"], "t05.ipynb")
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(set(metadata["dataset_sources"]), {"thibautgoldsborough/cellmot-baseline-artifacts", "pilkwang/biohub-tracking-support-pack-50ep-v1"})
        self.assertEqual({path.name for path in (ROOT / "kaggle_kernel" / "t05pkg").iterdir()}, {"kernel-metadata.json", "t05.ipynb", "t05push.json"})


if __name__ == "__main__":
    unittest.main()

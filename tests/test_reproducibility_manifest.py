import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_general_model_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("build_general_model_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproducibilityManifestTest(unittest.TestCase):
    def test_secret_and_cache_paths_are_excluded(self):
        module = load_manifest_module()
        self.assertTrue(module.is_secret_or_cache(Path(".env")))
        self.assertTrue(module.is_secret_or_cache(Path("gurobi.lic")))
        self.assertTrue(module.is_secret_or_cache(Path("src/celltrack_alpha/__pycache__/x.pyc")))
        self.assertFalse(module.is_secret_or_cache(Path("src/celltrack_alpha/pipeline.py")))

    def test_default_manifest_does_not_include_its_own_output(self):
        module = load_manifest_module()
        files = module.iter_files(["reports/general_model_*"], {"reports/general_model_reproducibility_manifest.json"})
        rels = [module.normalized_relpath(path) for path in files]
        self.assertNotIn("reports/general_model_reproducibility_manifest.json", rels)


if __name__ == "__main__":
    unittest.main()

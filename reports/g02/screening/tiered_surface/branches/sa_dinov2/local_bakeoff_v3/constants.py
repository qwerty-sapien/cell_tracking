"""Frozen paths and identities for resumable one-shard DINOv2 execution."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
DINO = ROOT / "reports/g02/screening/tiered_surface/branches/sa_dinov2"
GATE = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
V2 = DINO / "local_bakeoff_v2"

AMENDMENT = GATE / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
SEALED_B0 = GATE / "run_v3/B0_SEALED.json"
POPULATION_MANIFEST = GATE / "run_v1/population_manifest.json"
POPULATION_ROOT = GATE / "run_v1"
MATERIALIZER = DINO / "materialize_dinov2_xy_separation.py"
FEATURIZER = DINO / "dinov2_xy_action_featurizer.py"
EVALUATOR = GATE / "evaluate_cellect_separation.py"
REQUIREMENT_SURFACE = (
    ROOT / "reports/g02/screening/selector_requirement_44b6/run_v6/"
    "requirement_surface.json"
)
STAGING_CONTRACT = DINO / "R5_XY_BETA_STAGING_CONTRACT.json"
V2_AGGREGATOR = V2 / "aggregate_evaluate.py"
V2_RESOURCE_GUARD = V2 / "resource_guard.py"

SOURCE = Path("/private/tmp/g02_dinov2_source_7764ea0")
CHECKPOINT = Path(
    "/private/tmp/g02_dinov2_assets/"
    "sha256-b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9/"
    "dinov2_vits14_pretrain.pth"
)
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"

PINS = {
    AMENDMENT: "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    SEALED_B0: "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    POPULATION_MANIFEST: "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    MATERIALIZER: "8476d96eb23fb55b3630b2b819407f322d88ca9408325fe825912943f46a9953",
    FEATURIZER: "f93a202a7e662859f92442a4a6932ad371af513c65627838f2ebfd7bfecb9b49",
    EVALUATOR: "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
    REQUIREMENT_SURFACE: "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b",
    STAGING_CONTRACT: "c8e3df7739a3691b6ad9c93234ecc8db9696992c126de3ff343def38a93d5610",
    CHECKPOINT: "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}

CLASSES = ("delete", "occupied_swap")
EXPECTED_FILES = {"delete": 41, "occupied_swap": 36}
EXPECTED_ACTIONS = {"delete": 1_345_497, "occupied_swap": 45_961}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins() -> dict[str, str]:
    observed = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input identity drift: {path}: {actual}")
        observed[str(path)] = actual
    return observed

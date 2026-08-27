"""Frozen identities and paths for the resumable local CELLECT bake-off."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
GATE = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
DINO_V2 = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_dinov2/local_bakeoff_v2"
)
DAG2 = ROOT / "reports/g02/screening/selector_requirement_44b6/run_v6"
ASSET_ROOT = Path("/private/tmp/cellect-r7-v6-cmux-roundtrip.20260729/download")

AMENDMENT = GATE / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
SEALED_B0 = GATE / "run_v3/B0_SEALED.json"
POPULATION_MANIFEST = GATE / "run_v1/population_manifest.json"
POPULATION_ROOT = GATE / "run_v1"
EVALUATOR = GATE / "evaluate_cellect_separation.py"
SOURCE_FEATURIZER = GATE / "kaggle_cellect_action_features.py"
REQUIREMENT_SURFACE = DAG2 / "requirement_surface.json"
RESOURCE_GUARD = DINO_V2 / "resource_guard.py"

REQUEST_MANIFEST = ASSET_ROOT / "action_request_manifest.json"
REQUEST_ROOT = ASSET_ROOT
SAFE_LOADER = ASSET_ROOT / "cellect_model.py"
CHECKPOINT = (
    ASSET_ROOT
    / "cellect_uext_x3rd_149_0_4p6540"
    / "cellect_uext_x3rd_149_0_4p6540.pth"
)
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
RAW_ARCHIVE_BYTES = 87_393_127_165

PINS = {
    AMENDMENT: "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    SEALED_B0: "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    POPULATION_MANIFEST: "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    EVALUATOR: "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
    SOURCE_FEATURIZER: "3c8962ab3c92b14b09c7c3ff20c0042e13690428e504a6fdfc3b1b288547c9a3",
    REQUIREMENT_SURFACE: "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b",
    REQUEST_MANIFEST: "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83",
    SAFE_LOADER: "2394797f4cb680cd45136c83792df67a3ee8915b6925994a6135d4425998b0c0",
    CHECKPOINT: "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5",
}

CLASSES = ("delete", "occupied_swap")
EXPECTED_FILES = {"delete": 41, "occupied_swap": 36}
EXPECTED_ACTIONS = {"delete": 1_345_497, "occupied_swap": 45_961}
EXPECTED_BENEFICIAL = {"delete": 1_118, "occupied_swap": 134}
REQUIREMENTS = {
    "delete": {
        "mean_auc": 0.85,
        "mean_retention": 0.02039355992844365,
        "robust_p05_auc": 0.90,
        "robust_p05_retention": 0.044946332737030414,
    },
    "occupied_swap": {
        "mean_auc": 0.90,
        "mean_retention": 0.06791044776119404,
        "robust_p05_auc": 0.95,
        "robust_p05_retention": 0.14738805970149255,
    },
}

SUPPLIER_ID = "CELLECT-U-EXT-X3RD-149.0-4.6540"
SUPPLIER_SHA256 = "8d34445eafa19dcd135b59747c6ccb4314b0e2ca19f28e80fcfa752bab9d3664"
TILE_ZYX = (16, 64, 64)
DEFAULT_BATCH = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input identity drift: {path}: {actual}")
        observed[str(path)] = actual
    if CHECKPOINT.stat().st_size != 10_197_221:
        raise RuntimeError("CELLECT checkpoint byte-size drift")
    if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise RuntimeError("raw competition archive byte-size drift")
    return observed

from __future__ import annotations

from pathlib import Path

from scripts.audit_v32_native_multiframe_seam import audit_paths

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "repo/scripts/train_unet_transformer.py"
)
PREDICT = ROOT / (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "repo/scripts/predict_unet_transformer.py"
)
CACHE = ROOT / "src/celltrack_alpha/v32_preilp_cache.py"


def test_exact_v32_source_has_observation_distinct_multiframe_seam() -> None:
    result = audit_paths(TRAIN, PREDICT, CACHE)

    assert result["verdict"] == "GO"
    assert result["predicate_pass_count"] == result["predicate_count"] == 11
    assert result["baseline"]["unet_out_channels"] == 32
    assert result["baseline"]["window_size"] == 2
    assert result["predicates"]["baseline_edges_are_adjacent_only"] is True
    assert result["predicates"]["cache_omits_node_features"] is True
    assert result["predicates"]["cache_omits_raw_logits"] is True
    assert result["scientific_model_executed"] is False
    assert result["protected_outcomes_opened"] is False


def test_source_drift_fails_closed(tmp_path: Path) -> None:
    altered = tmp_path / "predict.py"
    altered.write_text(
        PREDICT.read_text().replace('"window_size": 2', '"window_size": 5')
    )

    result = audit_paths(TRAIN, altered, CACHE)

    assert result["verdict"] == "PARK_DEPENDENCY"
    assert result["predicates"]["exact_source_hashes"] is False
    assert result["predicates"]["baseline_window_size_2"] is False

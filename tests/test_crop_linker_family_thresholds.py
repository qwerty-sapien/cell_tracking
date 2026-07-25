from __future__ import annotations

import numpy as np

from scripts import audit_crop_linker_family_thresholds as audit
from scripts import fit_crop_linker_pilot as pilot


def _examples() -> pilot.Examples:
    return pilot.Examples(
        source=np.zeros((8, 16, 32, 32), dtype=np.float16),
        target=np.zeros((8, 16, 32, 32), dtype=np.float16),
        context=np.zeros((8, 48), dtype=np.float32),
        labels=np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8),
        video_index=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        identities=("train:44b6_alpha", "train:6bba_beta"),
        families=("44b6", "6bba"),
    )


def test_family_rows_and_combined_metrics_do_not_mix_thresholds() -> None:
    examples = _examples()
    scored = pilot.score_records(
        examples,
        np.asarray([0.9, 0.8, 0.7, 0.1, 0.6, 0.5, 0.4, 0.2]),
        selected_edges={"train:44b6_alpha": 100, "train:6bba_beta": 100},
    )
    assert len(audit.family_rows(scored, "44b6")) == 1
    metrics = audit.combined_metrics(
        scored, {"44b6": 0.75, "6bba": 0.45}
    )
    assert metrics["correct_rescues"] == 4
    assert metrics["harmful_swaps"] == 0
    assert metrics["precision"] == 1.0


def test_non_admissible_threshold_is_json_safe() -> None:
    assert audit.report_thresholds(
        {"44b6": float("inf"), "6bba": 0.5}
    ) == {"44b6": None, "6bba": 0.5}

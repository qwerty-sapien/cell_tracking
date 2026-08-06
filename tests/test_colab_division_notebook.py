from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(
    "reports/g02/screening/tiered_surface/branches/"
    "division_temporal_colab_v1/COLAB_HARNESS.ipynb"
)


def test_notebook_is_exactly_three_manual_resume_cells() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert len(cells) == 3
    assert all(cell["cell_type"] == "code" for cell in cells)
    sources = ["".join(cell["source"]) for cell in cells]
    assert sources[0].startswith("# CELL 1 — MOUNT / INSTALL / MANIFEST-BOUND RESUME")
    assert sources[1].startswith("# CELL 2 — TRAIN / TEN-MINUTE DURABILITY CADENCE")
    assert sources[2].startswith(
        "# CELL 3 — FINAL CHECKPOINT / FULL JSONL / DEVICE-SPECIFIC LEDGER"
    )
    for index, source in enumerate(sources, start=1):
        compile(source, f"COLAB_HARNESS.ipynb:cell-{index}", "exec")


def test_notebook_has_strict_resume_cadence_and_device_ledger() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    first, train, final = ["".join(cell["source"]) for cell in notebook["cells"]]
    assert "drive.mount" in first
    assert "pip', 'install'" in first
    assert "STORE.verify_latest()" in first
    assert "pull_latest_verified" in first
    assert "expected_manifest_sha256=EXPECTED_MANIFEST_SHA" in first
    assert "adjudicate_resume_runtime" in first
    assert "allow_gpu_transition=True" in first
    assert "runtime_continuity" in first
    assert "restore_training_state" in first
    assert "append_colab_session_start" in first
    assert "reconcile_interrupted_colab_sessions" in first
    assert "assigned_gpu_model()" in first
    assert "checkpoint_interval_seconds=600.0" in first
    assert "run_colab_session" in train
    assert "maximum_uncheckpointed_seconds" in train
    assert "> 600.0" in train
    assert "push_checkpoint" in final
    assert "full_jsonl_sha256" in final
    assert "append_colab_session" in final
    assert "colab_hours_by_gpu_model" in final


def test_notebook_contains_no_free_tier_session_automation() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8").lower()
    forbidden = (
        "selenium",
        "chromedriver",
        "pyautogui",
        "javascript",
        "keep_alive",
        "keep-alive",
        "multi-account",
        "session cycling",
    )
    assert not any(token in source for token in forbidden)

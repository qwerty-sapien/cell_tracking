from __future__ import annotations

import csv
from pathlib import Path

from scripts import build_dispersion_ladder_observation_matrix as ladder


FIELDS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)


def _fragment(
    root: Path,
    *,
    dataset: str,
    edge: tuple[int, int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"train__{dataset}.csv"
    rows = [
        [0, dataset, "node", 1, 0, 1, 2, 3, "", ""],
        [1, dataset, "node", 2, 1, 1, 2, 4, "", ""],
        [2, dataset, "node", 3, 1, 1, 2, 5, "", ""],
        [3, dataset, "edge", "", "", "", "", "", edge[0], edge[1]],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def test_streaming_comparison_preserves_phase0_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for index in range(57):
        dataset = f"6bba_{index:08x}"
        _fragment(left, dataset=dataset, edge=(1, 2))
        _fragment(
            right,
            dataset=dataset,
            edge=(1, 2) if index else (1, 3),
        )
    result = ladder.compare_fragment_dirs(left, right)
    assert result["aggregate"]["dataset_count"] == 57
    assert result["aggregate"]["nodes"]["jaccard"] == 1.0
    assert result["aggregate"]["edges_raw"]["shared"] == 56
    assert result["aggregate"]["edges_raw"]["a_only"] == 1
    assert result["aggregate"]["edges_raw"]["b_only"] == 1
    assert result["observation_class"]["label"] == "DISTINCT"


def test_fragment_index_rejects_incomplete_cv57(tmp_path: Path) -> None:
    _fragment(tmp_path, dataset="6bba_only", edge=(1, 2))
    try:
        ladder.fragment_files(tmp_path)
    except ladder.DispersionMatrixError as exc:
        assert "expected 57" in str(exc)
    else:
        raise AssertionError("incomplete fragment set was accepted")

#!/usr/bin/env python3
"""Build the single-file Kaggle script from the shared baseline source."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src/celltrack_baseline/rule_based.py"
OUT_PATH = ROOT / "kaggle_kernel/rule_based_submission.py"


HEADER = '''#!/usr/bin/env python3
"""Self-contained Kaggle free-compute rule-based submission script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_import(import_name: str, package_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        package = package_name or import_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


ensure_import("zarr")
ensure_import("scipy")

'''


MAIN = r'''

import zarr  # noqa: E402


COMP_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/biohub-cell-tracking-during-development")

OUTPUT_DIR = Path("/kaggle/working")
MODE = os.environ.get("MODE", "submit")


BASE_CONFIG = BaselineConfig(
    sigmas_um=(0.8, 1.2, 1.8),
    dog_ratio=1.7,
    threshold_quantile=0.99955,
    min_score=0.015,
    min_distance_um=2.8,
    max_points_per_t=180,
    max_link_um=8.0,
    polarity="bright",
    output_transform="identity",
)


def dataset_names(split: str) -> list[str]:
    split_dir = COMP_DIR / split
    return sorted(path.stem for path in split_dir.glob("*.zarr"))


def open_dataset(split: str, dataset: str):
    group = zarr.open_group(str(COMP_DIR / split / f"{dataset}.zarr"), mode="r")
    return group, group["0"]


def run_submit() -> None:
    config = BASE_CONFIG
    print("Submission config:")
    print(config_to_json(config))
    predictions = []
    for dataset in dataset_names("test"):
        group, array = open_dataset("test", dataset)
        attrs = dict(group.attrs)
        nodes, edges = predict_dataset_array(dataset, array, attrs, config, progress=True)
        predictions.append((dataset, nodes, edges))
        print(f"{dataset}: nodes={len(nodes)} edges={len(edges)}")
    rows = submission_rows(predictions)
    validation = validate_submission_rows(rows)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise RuntimeError("Submission validation failed")
    write_submission(rows, OUTPUT_DIR / "submission.csv")
    print("Wrote /kaggle/working/submission.csv")


def run_cv() -> None:
    datasets = dataset_names("train")[: int(os.environ.get("CV_DATASETS", "12"))]
    max_timepoints = int(os.environ.get("CV_TIMEPOINTS", "100"))
    threshold_grid = [0.9993, 0.99955, 0.99975]
    link_grid = [7.0, 8.0, 10.0]
    transform_grid = ["identity", "xflip", "yflip"]
    rows = []
    for threshold in threshold_grid:
        for link_um in link_grid:
            for output_transform in transform_grid:
                config = with_timepoint_limit(
                    BaselineConfig(
                        sigmas_um=BASE_CONFIG.sigmas_um,
                        dog_ratio=BASE_CONFIG.dog_ratio,
                        threshold_quantile=threshold,
                        min_score=BASE_CONFIG.min_score,
                        min_distance_um=BASE_CONFIG.min_distance_um,
                        max_points_per_t=BASE_CONFIG.max_points_per_t,
                        max_link_um=link_um,
                        polarity=BASE_CONFIG.polarity,
                        output_transform=output_transform,
                    ),
                    max_timepoints,
                )
                print("CV config:")
                print(config_to_json(config))
                fold_metrics = []
                for dataset in datasets:
                    group, array = open_dataset("train", dataset)
                    attrs = dict(group.attrs)
                    nodes, edges = predict_dataset_array(dataset, array, attrs, config, progress=False)
                    gt = read_geff_arrays(str(COMP_DIR), "train", dataset)
                    metrics = approximate_sparse_metrics(nodes, edges, gt, parse_scale(attrs))
                    metrics.update(
                        {
                            "dataset": dataset,
                            "threshold_quantile": threshold,
                            "max_link_um": link_um,
                            "output_transform": output_transform,
                        }
                    )
                    fold_metrics.append(metrics)
                    rows.append(metrics)
                    print(
                        f"{dataset}: adj={metrics['adj_edge_jaccard']:.4f} "
                        f"edge={metrics['edge_jaccard']:.4f} recall={metrics['node_recall']:.4f} "
                        f"pred_nodes={metrics['num_pred_nodes']:.0f}"
                    )
                adj = [m["adj_edge_jaccard"] for m in fold_metrics if m["adj_edge_jaccard"] == m["adj_edge_jaccard"]]
                recall = [m["node_recall"] for m in fold_metrics if m["node_recall"] == m["node_recall"]]
                print(
                    f"MEAN threshold={threshold} link={link_um} transform={output_transform}: "
                    f"adj={float(np.mean(adj)) if adj else float('nan'):.4f} "
                    f"recall={float(np.mean(recall)) if recall else float('nan'):.4f}"
                )
    import pandas as pd

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
    print("Wrote /kaggle/working/cv_results.csv")


if __name__ == "__main__":
    if MODE == "cv":
        run_cv()
    else:
        run_submit()
'''


def main() -> int:
    core = CORE_PATH.read_text(encoding="utf-8")
    filtered = "\n".join(
        line for line in core.splitlines()
        if line.strip() != "from __future__ import annotations"
    )
    OUT_PATH.write_text(HEADER + filtered + MAIN + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

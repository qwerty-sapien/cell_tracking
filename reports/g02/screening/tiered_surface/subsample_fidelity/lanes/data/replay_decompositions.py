#!/usr/bin/env python3
"""Recover per-video scorer statistics from already-banked common-36 fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings
import zipfile


ROOT = Path("/Users/agent/Desktop/cell_tracking")
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import zarr  # noqa: E402
from zarr.storage import ZipStore  # noqa: E402

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


TOLERANCE = 1e-12
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
RAW_ARCHIVE_BYTES = 87_393_127_165
RAW_CENTRAL_SHA256 = "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77"
RAW_ENTRY_COUNT = 24_886

SOURCE_HASHES = {
    "src/celltrack_alpha/official_eval.py": (
        "22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6"
    ),
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py": (
        "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444"
    ),
    (
        "external/kaggle-cell-tracking-competition/src/tracking_cellmot/"
        "division_metrics.py"
    ): "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
}

REPORT_SPECS = {
    "h1": {
        "path": "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json",
        "sha256": "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
    },
    "c1": {
        "path": (
            "reports/g02/screening/top50_parallel_dags/branches/"
            "c1_ta_hoct_lite/outputs/c1_n3_44b6_graph.json"
        ),
        "sha256": "6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10",
    },
    "delete": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_delete/"
            "graph_result.json"
        ),
        "sha256": "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a",
    },
    "swap": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_swap/"
            "result_44b6_postrepair_v1/report.json"
        ),
        "sha256": "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864",
    },
    "nodetopo": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/"
            "RESULT_44B6.json"
        ),
        "sha256": "fb9202586242f506a7cd8437d780bc93ba1fcfe5de957e22de8d6a2e73f94834",
    },
}


class ReplayError(RuntimeError):
    """The decomposition replay failed an immutable evidence check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root is not an object: {path}")
    return value


def central_directory_identity(path: Path) -> tuple[str, int]:
    with zipfile.ZipFile(path) as archive:
        rows = [
            {
                "compressed_size": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "filename": info.filename,
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
    encoded = (
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest(), len(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dict(official_summary(rows))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        result[key] = int(sum(int(float(row[key])) for row in rows))
    return result


def compare_summary(
    label: str,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "score",
        "adj_edge_jaccard",
        "edge_jaccard",
        "node_recall",
        "division_tp",
        "division_fp",
        "division_fn",
    )
    errors = {
        field: abs(float(observed[field]) - float(expected[field])) for field in fields
    }
    if any(error > TOLERANCE for error in errors.values()):
        raise ReplayError(f"{label}: aggregate replay mismatch: {errors}")
    return {
        "label": label,
        "pass": True,
        "tolerance": TOLERANCE,
        "absolute_errors": errors,
        "observed": observed,
        "expected": {field: expected[field] for field in fields},
    }


def fragment_map_from_files(
    rows: list[dict[str, Any]], expected_ids: tuple[str, ...], label: str
) -> dict[str, dict[str, Any]]:
    by_id = {str(row["identity"]): dict(row) for row in rows}
    if set(by_id) != set(expected_ids) or len(by_id) != len(rows):
        raise ReplayError(f"{label}: fragment universe mismatch")
    return by_id


def authenticate_fragment(
    record: dict[str, Any], identity: str, label: str
) -> dict[str, Any]:
    path = Path(str(record.get("path") or record.get("fragment")))
    if not path.is_absolute():
        path = ROOT / path
    expected = str(record["sha256"] if "sha256" in record else record["fragment_sha256"])
    observed = sha256_file(path)
    if observed != expected:
        raise ReplayError(f"{label} {identity}: fragment hash mismatch")
    return {
        "candidate": label,
        "identity": identity,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def score_fragment(
    path: Path,
    identity: str,
    gt: dict[str, Any],
    scale: tuple[float, float, float],
) -> dict[str, Any]:
    dataset = identity.split(":", 1)[1]
    graphs = read_submission(path)
    graph = graphs.get(dataset)
    if graph is None or len(graphs) != 1:
        raise ReplayError(f"{identity}: fragment dataset mismatch: {path}")
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return dict(official_metric_row(nodes, edges, gt, scale))


def direct_rows(
    records: list[dict[str, Any]], metric_key: str
) -> dict[str, dict[str, Any]]:
    return {
        str(record["identity"]): {
            "identity": str(record["identity"]),
            **dict(record[metric_key]),
        }
        for record in records
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ReplayError(f"refusing to overwrite: {args.output}")
    started = time.perf_counter()

    source_receipts = {}
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise ReplayError(f"scorer source drift: {relative}")
        source_receipts[relative] = observed

    reports = {}
    report_receipts = {}
    for label, spec in REPORT_SPECS.items():
        path = ROOT / spec["path"]
        observed = sha256_file(path)
        if observed != spec["sha256"]:
            raise ReplayError(f"terminal report drift: {label}")
        reports[label] = load_json(path)
        report_receipts[label] = {
            "path": spec["path"],
            "sha256": observed,
        }

    if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise ReplayError("raw archive byte size mismatch")
    central_sha, entry_count = central_directory_identity(RAW_ARCHIVE)
    if central_sha != RAW_CENTRAL_SHA256 or entry_count != RAW_ENTRY_COUNT:
        raise ReplayError("raw archive central directory mismatch")

    nodetopo_records = reports["nodetopo"]["per_video"]
    common_ids = tuple(sorted(str(row["identity"]) for row in nodetopo_records))
    if (
        len(common_ids) != 36
        or len(set(common_ids)) != 36
        or any(not identity.startswith("train:44b6_") for identity in common_ids)
    ):
        raise ReplayError("common held-44b6 universe mismatch")
    universe_sha = hashlib.sha256(
        ("\n".join(common_ids) + "\n").encode("ascii")
    ).hexdigest()

    fragment_maps = {
        "APNCC": fragment_map_from_files(
            reports["h1"]["graph"]["control"]["input_provenance"]["files"],
            common_ids,
            "APNCC",
        ),
        "C1": fragment_map_from_files(
            reports["c1"]["graph"]["c1"]["input_provenance"]["files"],
            common_ids,
            "C1",
        ),
        "H1-ZWRFEAT": fragment_map_from_files(
            reports["h1"]["graph"]["candidate"]["input_provenance"]["files"],
            common_ids,
            "H1-ZWRFEAT",
        ),
        "SA-SWAP": fragment_map_from_files(
            [
                {
                    "identity": row["identity"],
                    "fragment": row["alpha"]["fragment"],
                    "fragment_sha256": row["alpha"]["fragment_sha256"],
                }
                for row in reports["swap"]["graph"]["videos"]
            ],
            common_ids,
            "SA-SWAP",
        ),
    }
    fragment_receipts = []
    for candidate, mapping in fragment_maps.items():
        for identity in common_ids:
            fragment_receipts.append(
                authenticate_fragment(mapping[identity], identity, candidate)
            )

    candidate_rows: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in fragment_maps
    }
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for index, identity in enumerate(common_ids, start=1):
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            for candidate, mapping in fragment_maps.items():
                raw_path = Path(
                    str(
                        mapping[identity].get("path")
                        or mapping[identity].get("fragment")
                    )
                )
                path = raw_path if raw_path.is_absolute() else ROOT / raw_path
                candidate_rows[candidate][identity] = {
                    "identity": identity,
                    **score_fragment(path, identity, gt, scale),
                }
            print(f"[{index:02d}/36] replayed {identity}", flush=True)
            del gt, group

    cg0_rows = direct_rows(nodetopo_records, "baseline_metrics")
    candidate_rows["NODETOPO"] = direct_rows(
        nodetopo_records, "candidate_metrics"
    )

    delete_family = reports["delete"]["family_results"]["44b6"]
    delete_control_all = direct_rows(delete_family["videos"], "baseline_metrics")
    delete_candidate_all = direct_rows(delete_family["videos"], "candidate_metrics")
    if not set(common_ids) < set(delete_control_all) or len(delete_control_all) != 41:
        raise ReplayError("SA-DELETE 41-to-36 universe relation mismatch")
    extra_delete_ids = sorted(set(delete_control_all) - set(common_ids))
    if len(extra_delete_ids) != 5:
        raise ReplayError("SA-DELETE extra-video count mismatch")
    candidate_rows["SA-DELETE"] = {
        identity: delete_candidate_all[identity] for identity in common_ids
    }
    for identity in common_ids:
        for field in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "adj_edge_jaccard",
            "node_recall",
        ):
            if abs(
                float(delete_control_all[identity][field])
                - float(cg0_rows[identity][field])
            ) > TOLERANCE:
                raise ReplayError(f"CG0 mismatch in SA-DELETE row: {identity}/{field}")

    aggregate_checks = []
    aggregate_checks.append(
        compare_summary(
            "APNCC",
            summarise([candidate_rows["APNCC"][identity] for identity in common_ids]),
            reports["h1"]["graph"]["control"]["summary"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "C1",
            summarise([candidate_rows["C1"][identity] for identity in common_ids]),
            reports["c1"]["graph"]["c1"]["summary"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "H1-ZWRFEAT",
            summarise(
                [candidate_rows["H1-ZWRFEAT"][identity] for identity in common_ids]
            ),
            reports["h1"]["graph"]["candidate"]["summary"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "SA-SWAP",
            summarise([candidate_rows["SA-SWAP"][identity] for identity in common_ids]),
            reports["swap"]["graph"]["alpha"]["summary"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "CG0",
            summarise([cg0_rows[identity] for identity in common_ids]),
            reports["nodetopo"]["baseline"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "NODETOPO",
            summarise(
                [candidate_rows["NODETOPO"][identity] for identity in common_ids]
            ),
            reports["nodetopo"]["candidate"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "SA-DELETE-historical-41-control",
            summarise([delete_control_all[key] for key in sorted(delete_control_all)]),
            delete_family["control"],
        )
    )
    aggregate_checks.append(
        compare_summary(
            "SA-DELETE-historical-41-candidate",
            summarise(
                [delete_candidate_all[key] for key in sorted(delete_candidate_all)]
            ),
            delete_family["candidate"],
        )
    )

    cg0_summary = summarise([cg0_rows[identity] for identity in common_ids])
    standardized = {}
    for name in ("APNCC", "C1", "H1-ZWRFEAT", "SA-DELETE", "SA-SWAP", "NODETOPO"):
        rows = [candidate_rows[name][identity] for identity in common_ids]
        summary = summarise(rows)
        standardized[name] = {
            "summary": summary,
            "delta_vs_cg0": float(summary["score"]) - float(cg0_summary["score"]),
            "rows": rows,
        }

    payload = {
        "schema": "g02.subsample-fidelity.common36-decomposition.v1",
        "status": "PASS",
        "amendment": {
            "path": (
                "reports/g02/screening/tiered_surface/subsample_fidelity/"
                "PREREGISTRATION_AMENDMENT_V1.md"
            ),
            "sha256": (
                "3c497948189b9518c624e3c008cbfa021bc0072a7067a9ebd93e08163b9673fc"
            ),
        },
        "universe": {
            "family": "44b6",
            "video_count": 36,
            "video_ids": list(common_ids),
            "sha256": universe_sha,
        },
        "reference": {
            "name": "CG0",
            "summary": cg0_summary,
            "rows": [cg0_rows[identity] for identity in common_ids],
        },
        "candidates": standardized,
        "historical_scope_diagnostics": {
            "SA-DELETE": {
                "reported_video_count": 41,
                "common_video_count": 36,
                "excluded_video_ids": extra_delete_ids,
                "reported_delta": delete_family["delta"]["score"],
            },
            "APNCC": {
                "older_cv57_delta_not_used": True,
                "common36_source": REPORT_SPECS["h1"]["path"],
            },
        },
        "authentication": {
            "reports": report_receipts,
            "scorer_source_hashes": source_receipts,
            "raw_archive": {
                "path": str(RAW_ARCHIVE),
                "bytes": RAW_ARCHIVE.stat().st_size,
                "central_directory_sha256": central_sha,
                "entry_count": entry_count,
                "full_sha256": (
                    "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
                ),
                "full_sha256_status": "banked P0 receipt; not rehashed",
            },
            "fragment_count_replayed": len(fragment_receipts),
            "fragments": fragment_receipts,
            "aggregate_checks": aggregate_checks,
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_only": True,
            "model_inference": False,
            "fit": False,
            "graph_construction": False,
            "new_scientific_outcome": False,
            "family": "44b6",
            "second_family_opened": False,
            "cv57_opened": False,
            "gpu": False,
            "kaggle": False,
            "runpod": False,
            "budget_debit": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

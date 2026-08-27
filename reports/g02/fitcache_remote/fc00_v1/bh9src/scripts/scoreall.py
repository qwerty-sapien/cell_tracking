#!/usr/bin/env python3
"""Stream full-video predictions through the frozen official competition scorer."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SUBMISSION_COLUMNS,
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    read_submission,
    sha256_file,
    validate_graphs,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import (  # noqa: E402
    list_zip_datasets,
    parse_scale,
    read_geff_arrays,
)


IDENTITY_RE = re.compile(r"^(train|test):([A-Za-z0-9][A-Za-z0-9_.-]*)$")
FRAGMENT_RE = re.compile(r"^(train|test)__([A-Za-z0-9][A-Za-z0-9_.-]*)\.csv$")
SCORER_SOURCES = (
    "src/celltrack_alpha/official_eval.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py",
)


class ScoreAllError(ValueError):
    """An input cannot be scored without violating the scoring contract."""


class FragmentSetError(ScoreAllError):
    """The fragment files do not exactly cover the selected train identities."""

    def __init__(self, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        parts = []
        for key in (
            "missing_relevant_fragments",
            "extra_relevant_fragments",
            "duplicate_relevant_fragments",
            "invalid_fragment_paths",
        ):
            value = self.diagnostics.get(key)
            if value:
                parts.append(f"{key}={value}")
        super().__init__("fragment identity validation failed: " + "; ".join(parts))


@dataclass(frozen=True)
class Selection:
    subset: str
    requested_video_ids: tuple[str, ...]
    selected_video_ids: tuple[str, ...]
    max_datasets: int | None
    excluded_train_video_ids: tuple[str, ...]

    @property
    def truncated(self) -> bool:
        return len(self.selected_video_ids) < len(self.requested_video_ids)

    @property
    def non_acceptance(self) -> bool:
        return self.max_datasets is not None

    def as_dict(self) -> dict[str, Any]:
        if self.non_acceptance:
            run_label = "truncated_non_acceptance_smoke_test"
        elif self.subset == "all":
            run_label = "full_acceptance_candidate"
        else:
            run_label = "subset_diagnostic"
        return {
            "subset": self.subset,
            "requested_count": len(self.requested_video_ids),
            "selected_count": len(self.selected_video_ids),
            "requested_video_ids": list(self.requested_video_ids),
            "selected_video_ids": list(self.selected_video_ids),
            "excluded_train_video_ids_added_for_all": list(self.excluded_train_video_ids),
            "max_datasets": self.max_datasets,
            "truncated": self.truncated,
            "non_acceptance": self.non_acceptance,
            "acceptance_eligible": self.subset == "all" and not self.non_acceptance,
            "run_label": run_label,
        }


@dataclass(frozen=True)
class FragmentManifest:
    ordered: tuple[tuple[str, Path], ...]
    diagnostics: Mapping[str, Any]


def parse_identity(value: str, *, implied_split: str | None = None) -> tuple[str, str, str]:
    """Return ``(split, dataset, normalized_identity)`` without prefix ambiguity."""

    raw = str(value)
    match = IDENTITY_RE.fullmatch(raw)
    if match is not None:
        split, dataset = match.groups()
        return split, dataset, f"{split}:{dataset}"
    if implied_split is not None and ":" not in raw:
        match = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw)
        if match is not None and implied_split in {"train", "test"}:
            return implied_split, raw, f"{implied_split}:{raw}"
    raise ScoreAllError(f"invalid or ambiguous dataset identity {value!r}")


def _unique_ids(values: Any, location: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ScoreAllError(f"{location} must be a list")
    normalized: list[str] = []
    for value in values:
        split, _, identity = parse_identity(str(value))
        if split != "train":
            raise ScoreAllError(f"{location} contains non-train identity {identity!r}")
        normalized.append(identity)
    if len(normalized) != len(set(normalized)):
        raise ScoreAllError(f"{location} contains duplicate video identities")
    return tuple(normalized)


def _fold_objects(split: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    folds = split.get("folds")
    if isinstance(folds, Mapping):
        try:
            result = tuple(folds[str(index)] for index in range(3))
        except KeyError as exc:
            raise ScoreAllError("split folds must have keys '0', '1', and '2'") from exc
    elif isinstance(folds, list) and len(folds) == 3:
        result = tuple(folds)
    else:
        raise ScoreAllError("split folds must contain exactly three folds")
    if not all(isinstance(fold, Mapping) for fold in result):
        raise ScoreAllError("every split fold must be an object")
    return result


def select_video_ids(
    split: Mapping[str, Any],
    subset: str,
    *,
    archive_train_video_ids: Iterable[str] | None = None,
    max_datasets: int | None = None,
) -> Selection:
    """Validate the frozen split and select exact train identities."""

    if subset not in {"all", "audit", "fold0", "fold1", "fold2"}:
        raise ScoreAllError(f"unsupported subset {subset!r}")
    audit_obj = split.get("audit")
    if not isinstance(audit_obj, Mapping):
        raise ScoreAllError("split audit must be an object")
    audit = _unique_ids(audit_obj.get("video_ids"), "audit.video_ids")
    if int(audit_obj.get("video_count", -1)) != len(audit):
        raise ScoreAllError("audit video_count does not match audit.video_ids")

    fold_ids: list[tuple[str, ...]] = []
    for index, fold in enumerate(_fold_objects(split)):
        ids = _unique_ids(fold.get("held_out_video_ids"), f"folds.{index}.held_out_video_ids")
        if int(fold.get("held_out_video_count", -1)) != len(ids):
            raise ScoreAllError(f"fold {index} held_out_video_count mismatch")
        fold_ids.append(ids)

    audit_set = set(audit)
    remainder: set[str] = set()
    for index, ids in enumerate(fold_ids):
        overlap = remainder.intersection(ids)
        if overlap:
            raise ScoreAllError(f"fold {index} overlaps another fold: {sorted(overlap)}")
        remainder.update(ids)
    if audit_set.intersection(remainder):
        raise ScoreAllError("audit identities overlap held-out fold identities")

    counts = split.get("counts")
    if not isinstance(counts, Mapping):
        raise ScoreAllError("split counts must be an object")
    if int(counts.get("audit_videos", -1)) != len(audit):
        raise ScoreAllError("counts.audit_videos mismatch")
    if int(counts.get("remainder_videos", -1)) != len(remainder):
        raise ScoreAllError("counts.remainder_videos mismatch")

    core = audit_set | remainder
    archive_ids: set[str] | None = None
    excluded: set[str] = set()
    if archive_train_video_ids is not None:
        archive_ids = set()
        for value in archive_train_video_ids:
            split_name, _, identity = parse_identity(str(value), implied_split="train")
            if split_name != "train":
                raise ScoreAllError(f"archive train set contains {identity!r}")
            archive_ids.add(identity)
        if not core.issubset(archive_ids):
            raise ScoreAllError(
                f"archive is missing split train identities: {sorted(core - archive_ids)}"
            )
        excluded = archive_ids - core
        expected_excluded = int(counts.get("excluded_nontrain_or_mixed_groups", 0))
        if len(excluded) != expected_excluded:
            raise ScoreAllError(
                "archive/split excluded-train count mismatch: "
                f"expected {expected_excluded}, got {len(excluded)}"
            )

    if subset == "audit":
        requested = set(audit)
    elif subset.startswith("fold"):
        requested = set(fold_ids[int(subset[-1])])
    else:
        if archive_ids is None:
            expected_excluded = int(counts.get("excluded_nontrain_or_mixed_groups", 0))
            if expected_excluded:
                raise ScoreAllError(
                    "subset 'all' requires archive train identities to recover excluded train videos"
                )
        requested = set(archive_ids if archive_ids is not None else core)

    if archive_ids is not None and not requested.issubset(archive_ids):
        raise ScoreAllError("selected identities are absent from the archive train set")
    requested_ids = tuple(sorted(requested))
    if max_datasets is not None:
        if max_datasets <= 0:
            raise ScoreAllError("--max-datasets must be a positive integer")
        selected_ids = requested_ids[:max_datasets]
    else:
        selected_ids = requested_ids
    return Selection(
        subset=subset,
        requested_video_ids=requested_ids,
        selected_video_ids=selected_ids,
        max_datasets=max_datasets,
        excluded_train_video_ids=tuple(sorted(excluded)) if subset == "all" else (),
    )


def resolve_fragment_paths(spec: str) -> list[Path]:
    raw = Path(spec).expanduser()
    if raw.is_dir():
        paths = [path for path in raw.iterdir() if path.is_file() and path.suffix.lower() == ".csv"]
    elif glob.has_magic(spec):
        paths = [Path(value) for value in glob.glob(spec, recursive=True)]
        if any(not path.is_file() or path.suffix.lower() != ".csv" for path in paths):
            raise ScoreAllError("fragment glob must resolve only to CSV files")
    else:
        raise ScoreAllError("--frags must be a directory or a direct glob pattern")
    paths = sorted(paths, key=lambda path: str(path.resolve()))
    if not paths:
        raise ScoreAllError(f"fragment input matched no CSV files: {spec!r}")
    return paths


def validate_fragment_set(paths: Sequence[Path], selected_video_ids: Iterable[str]) -> FragmentManifest:
    expected: dict[str, str] = {}
    for identity in selected_video_ids:
        split, dataset, normalized = parse_identity(identity)
        if split != "train":
            raise ScoreAllError(f"selected identity is not train: {normalized}")
        expected[dataset] = normalized

    by_dataset: dict[str, list[Path]] = {}
    ignored_test: list[str] = []
    invalid: list[str] = []
    for path in paths:
        match = FRAGMENT_RE.fullmatch(path.name)
        if match is None:
            invalid.append(str(path))
            continue
        split, dataset = match.groups()
        if split == "test":
            ignored_test.append(str(path))
            continue
        by_dataset.setdefault(dataset, []).append(path)

    duplicates = {
        f"train:{dataset}": [str(path) for path in dataset_paths]
        for dataset, dataset_paths in sorted(by_dataset.items())
        if len(dataset_paths) != 1
    }
    actual = set(by_dataset)
    missing = [f"train__{dataset}.csv" for dataset in sorted(set(expected) - actual)]
    extra = [f"train__{dataset}.csv" for dataset in sorted(actual - set(expected))]
    diagnostics = {
        "expected_fragment_count": len(expected),
        "discovered_csv_count": len(paths),
        "train_fragment_count": sum(len(value) for value in by_dataset.values()),
        "ignored_test_fragments": ignored_test,
        "missing_relevant_fragments": missing,
        "extra_relevant_fragments": extra,
        "duplicate_relevant_fragments": duplicates,
        "invalid_fragment_paths": invalid,
    }
    if missing or extra or duplicates or invalid:
        raise FragmentSetError(diagnostics)
    ordered = tuple(
        (expected[dataset], by_dataset[dataset][0]) for dataset in sorted(expected)
    )
    return FragmentManifest(ordered, diagnostics)


def sha256_with_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def combined_fragment_sha256(records: Iterable[tuple[str, str]]) -> str:
    """Hash logical identity/content pairs, independent of paths and discovery order."""

    digest = hashlib.sha256()
    normalized = sorted((str(identity), str(file_hash)) for identity, file_hash in records)
    for identity, file_hash in normalized:
        parse_identity(identity)
        if not re.fullmatch(r"[0-9a-f]{64}", file_hash):
            raise ScoreAllError(f"invalid fragment SHA-256 for {identity}")
        payload = json.dumps(
            {"identity": identity, "sha256": file_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _new_graph(dataset: str) -> SubmissionGraph:
    return SubmissionGraph(dataset=dataset, nodes={}, edges=[])


def _add_aggregate_row(graph: SubmissionGraph, row: Mapping[str, str], location: str) -> None:
    try:
        row_type = row["row_type"]
        if row_type == "node":
            node = SubmissionNode(
                node_id=int(row["node_id"]),
                t=int(row["t"]),
                z=int(row["z"]),
                y=int(row["y"]),
                x=int(row["x"]),
            )
            if node.node_id in graph.nodes:
                raise ScoreAllError(f"{location}: duplicate node ID {node.node_id}")
            graph.nodes[node.node_id] = node
        elif row_type == "edge":
            graph.edges.append(SubmissionEdge(int(row["source_id"]), int(row["target_id"])))
        else:
            raise ScoreAllError(f"{location}: unexpected row_type {row_type!r}")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ScoreAllError):
            raise
        raise ScoreAllError(f"{location}: malformed integer field") from exc


def iter_aggregate_graphs(
    path: Path,
) -> Iterator[tuple[str, str, str, SubmissionGraph, int]]:
    """Yield one validated contiguous graph at a time from an aggregate CSV."""

    seen: set[str] = set()
    current_identity: str | None = None
    current_split = ""
    current_dataset = ""
    graph: SubmissionGraph | None = None
    row_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
                raise ScoreAllError(
                    f"unexpected aggregate header in {path}: {reader.fieldnames}"
                )
            for row in reader:
                location = f"{path}: row {reader.line_num}"
                if None in row or any(row.get(name) is None for name in SUBMISSION_COLUMNS):
                    raise ScoreAllError(f"{location}: malformed column count")
                split, dataset, identity = parse_identity(row["dataset"])
                if identity != current_identity:
                    if current_identity is not None:
                        assert graph is not None
                        try:
                            validate_graphs({current_dataset: graph})
                        except ValueError as exc:
                            raise ScoreAllError(
                                f"invalid aggregate graph {current_identity}: {exc}"
                            ) from exc
                        seen.add(current_identity)
                        yield current_identity, current_split, current_dataset, graph, row_count
                        graph = None
                    if identity in seen:
                        raise ScoreAllError(
                            f"aggregate dataset group recurs out of order: {identity}"
                        )
                    current_identity = identity
                    current_split = split
                    current_dataset = dataset
                    graph = _new_graph(dataset)
                    row_count = 0
                assert graph is not None
                _add_aggregate_row(graph, row, location)
                row_count += 1
            if current_identity is not None:
                assert graph is not None
                try:
                    validate_graphs({current_dataset: graph})
                except ValueError as exc:
                    raise ScoreAllError(
                        f"invalid aggregate graph {current_identity}: {exc}"
                    ) from exc
                seen.add(current_identity)
                yield current_identity, current_split, current_dataset, graph, row_count
                graph = None
    except UnicodeDecodeError as exc:
        raise ScoreAllError(f"aggregate CSV is not valid UTF-8: {path}") from exc
    except csv.Error as exc:
        raise ScoreAllError(f"malformed aggregate CSV {path}: {exc}") from exc


def _fragment_graph(path: Path, expected_identity: str) -> SubmissionGraph:
    try:
        graphs = read_submission(path)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise ScoreAllError(f"invalid submission fragment {path}: {exc}") from exc
    if len(graphs) != 1:
        raise ScoreAllError(f"fragment {path} must contain exactly one dataset graph")
    raw_identity, graph = next(iter(graphs.items()))
    split, _, identity = parse_identity(raw_identity, implied_split="train")
    graph_split, _, graph_identity = parse_identity(graph.dataset, implied_split="train")
    if split != "train" or graph_split != "train":
        raise ScoreAllError(f"test graph is forbidden in train fragment {path}")
    if identity != expected_identity or graph_identity != expected_identity:
        raise ScoreAllError(
            f"fragment graph identity mismatch for {path}: "
            f"expected {expected_identity}, got key={identity}, graph={graph_identity}"
        )
    return graph


def _prediction_rows(dataset: str, graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    nodes = [
        {
            "dataset": dataset,
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": edge.source_id, "target_id": edge.target_id}
        for edge in graph.edges
    ]
    return nodes, edges


def score_graph(
    store: Any,
    identity: str,
    graph: SubmissionGraph,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    split, dataset, normalized = parse_identity(identity)
    if split != "train":
        raise ScoreAllError(f"refusing to score non-train graph {normalized}")
    started = time.perf_counter()
    group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
    gt = read_geff_arrays(store, "train", dataset)
    scale = parse_scale(dict(group.attrs))
    nodes, edges = _prediction_rows(dataset, graph)
    metrics = official_metric_row(nodes, edges, gt, scale)
    row = {
        "identity": normalized,
        "dataset": dataset,
        "source": dict(source),
        "prediction_node_rows": len(nodes),
        "prediction_edge_rows": len(edges),
        "ground_truth_nodes": int(len(gt["nodes"]["node_id"])),
        "ground_truth_edges": int(len(gt["edges"])),
        "scale_zyx": [float(value) for value in scale],
        "full_video": True,
        "cropped": False,
        **metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del nodes, edges, gt, group, graph
    return row


def score_fragment_input(
    store: Any,
    manifest: FragmentManifest,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance_files: list[dict[str, Any]] = []
    hash_records: list[tuple[str, str]] = []
    for identity, path in manifest.ordered:
        file_hash, size = sha256_with_size(path)
        graph = _fragment_graph(path, identity)
        row = score_graph(
            store,
            identity,
            graph,
            source={"mode": "fragment", "path": str(path), "sha256": file_hash, "bytes": size},
        )
        rows.append(row)
        provenance_files.append(
            {"identity": identity, "path": str(path), "sha256": file_hash, "bytes": size}
        )
        hash_records.append((identity, file_hash))
        del graph
    provenance = {
        "mode": "fragments",
        "file_count": len(provenance_files),
        "files": provenance_files,
        "aggregate_sha256": combined_fragment_sha256(hash_records),
    }
    return rows, provenance, dict(manifest.diagnostics)


def score_aggregate_input(
    store: Any,
    path: Path,
    selected_video_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected = set(selected_video_ids)
    seen_selected: set[str] = set()
    seen_train: list[str] = []
    seen_test: list[str] = []
    rows: list[dict[str, Any]] = []
    aggregate_hash, aggregate_bytes = sha256_with_size(path)
    group_count = 0
    for identity, split, _, graph, row_count in iter_aggregate_graphs(path):
        group_count += 1
        if split == "test":
            seen_test.append(identity)
        else:
            seen_train.append(identity)
            if identity in selected:
                rows.append(
                    score_graph(
                        store,
                        identity,
                        graph,
                        source={
                            "mode": "aggregate",
                            "path": str(path),
                            "aggregate_sha256": aggregate_hash,
                            "group_row_count": row_count,
                        },
                    )
                )
                seen_selected.add(identity)
        del graph
    missing = sorted(selected - seen_selected)
    if missing:
        raise ScoreAllError(f"aggregate is missing selected train dataset groups: {missing}")
    coverage = {
        "selected_video_ids_seen": sorted(seen_selected),
        "missing_selected_video_ids": missing,
        "train_dataset_groups": len(seen_train),
        "test_dataset_groups_skipped": len(seen_test),
        "skipped_test_video_ids": seen_test,
        "unselected_train_video_ids_skipped": sorted(set(seen_train) - selected),
        "missing_relevant_fragments": [],
        "extra_relevant_fragments": [],
    }
    provenance = {
        "mode": "aggregate",
        "path": str(path),
        "sha256": aggregate_hash,
        "bytes": aggregate_bytes,
        "dataset_group_count": group_count,
        "train_dataset_group_count": len(seen_train),
        "test_dataset_group_count": len(seen_test),
    }
    return rows, provenance, coverage


def verify_scorer_sources(split: Mapping[str, Any]) -> dict[str, str]:
    expected = split.get("source_hashes")
    if not isinstance(expected, Mapping):
        raise ScoreAllError("split does not attest scorer source hashes")
    observed: dict[str, str] = {}
    for relative in SCORER_SOURCES:
        path = ROOT / relative
        expected_hash = expected.get(relative)
        if not isinstance(expected_hash, str):
            raise ScoreAllError(f"split is missing scorer source hash for {relative}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ScoreAllError(
                f"frozen scorer source hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual}"
            )
        observed[relative] = actual
    return observed


def archive_train_video_ids(zip_path: Path) -> tuple[str, ...]:
    datasets = list_zip_datasets(zip_path, "train")
    if not datasets:
        raise ScoreAllError(f"archive contains no train datasets: {zip_path}")
    return tuple(f"train:{dataset}" for dataset in datasets)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.zip.is_file():
        raise ScoreAllError(f"competition archive does not exist: {args.zip}")
    if not args.split.is_file():
        raise ScoreAllError(f"split JSON does not exist: {args.split}")
    if args.submission is not None and not args.submission.is_file():
        raise ScoreAllError(f"aggregate submission does not exist: {args.submission}")

    try:
        split = json.loads(args.split.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreAllError(f"cannot read split JSON: {args.split}") from exc
    if not isinstance(split, Mapping):
        raise ScoreAllError("split JSON must be an object")

    split_hash = sha256_file(args.split)
    archive_hash = sha256_file(args.zip)
    scorer_hashes = verify_scorer_sources(split)
    archive_ids = archive_train_video_ids(args.zip)
    selection = select_video_ids(
        split,
        args.subset,
        archive_train_video_ids=archive_ids,
        max_datasets=args.max_datasets,
    )

    manifest: FragmentManifest | None = None
    if args.frags is not None:
        manifest = validate_fragment_set(
            resolve_fragment_paths(args.frags), selection.selected_video_ids
        )

    store = ZipStore(args.zip, mode="r")
    try:
        if manifest is not None:
            rows, input_provenance, coverage = score_fragment_input(store, manifest)
        else:
            assert args.submission is not None
            rows, input_provenance, coverage = score_aggregate_input(
                store, args.submission, selection.selected_video_ids
            )
    finally:
        store.close()

    if [row["identity"] for row in rows] != list(selection.selected_video_ids):
        by_identity = {row["identity"]: row for row in rows}
        if set(by_identity) != set(selection.selected_video_ids):
            raise ScoreAllError("scored identities do not exactly match the selection")
        rows = [by_identity[identity] for identity in selection.selected_video_ids]

    summary = official_summary(rows)
    streaming_contract = {
        "bounded_memory": True,
        "aggregate_submission_loaded": False,
        "prediction_loading": "one_dataset_graph_at_a_time",
        "max_prediction_dataset_graphs_resident": 1,
        "ground_truth_loading": "one_dataset_at_a_time",
        "max_ground_truth_datasets_resident": 1,
        "zipstore_open_count": 1,
        "zipstore_reused_across_datasets": True,
        "image_frames_read": False,
        "frames_cropped": False,
        "full_video_geff_scoring": True,
        "aggregate_requires_contiguous_dataset_groups": True,
        "test_groups_scored": False,
    }
    result = {
        "schema_version": 1,
        "kind": "full_video_frozen_official_score",
        "status": "completed",
        "selection": selection.as_dict(),
        "selected_video_ids": list(selection.selected_video_ids),
        "summary": summary,
        "samples": rows,
        "split": {
            "path": str(args.split),
            "sha256": split_hash,
            "artifact": split.get("artifact"),
        },
        "archive": {
            "path": str(args.zip),
            "sha256": archive_hash,
            "train_video_count": len(archive_ids),
        },
        "frozen_scorer_source_hashes": scorer_hashes,
        "input_provenance": input_provenance,
        "coverage": coverage,
        "missing_relevant_fragments": coverage.get("missing_relevant_fragments", []),
        "extra_relevant_fragments": coverage.get("extra_relevant_fragments", []),
        "streaming_contract": streaming_contract,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frags", help="directory or quoted glob of train fragments")
    inputs.add_argument("--submission", type=Path, help="contiguous aggregate submission CSV")
    parser.add_argument("--zip", type=Path, required=True, help="competition archive")
    parser.add_argument("--split", type=Path, required=True, help="frozen grouped split JSON")
    parser.add_argument("--output", type=Path, required=True, help="output score JSON")
    parser.add_argument(
        "--subset",
        choices=("all", "audit", "fold0", "fold1", "fold2"),
        default="all",
    )
    parser.add_argument("--max-datasets", type=positive_int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except ScoreAllError as exc:
        print(f"scoreall: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": result["summary"],
                "selection": result["selection"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

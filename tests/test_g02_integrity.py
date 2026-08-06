from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.g02_integrity import (  # noqa: E402
    DEFAULT_SCALE_ZYX,
    IntegrityError,
    MAX_TEST_VIDEOS,
    MAX_TOTAL_VIDEOS,
    MAX_TRAIN_VIDEOS,
    MAX_VIDEOS,
    SubmissionIntegrityError,
    assert_fit_sources_allowed,
    build_dataset_registry,
    build_fit_provenance,
    build_grouped_split_manifest,
    freeze_integrity_outputs,
    score_submission_rows,
    validate_submission_rows,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_metric_row,
    prediction_to_graph,
)
from celltrack_alpha.scorer_parity import (  # noqa: E402
    DIVISION_EDGES,
    DIVISION_NODES,
    SCALE_ZYX,
    _gt,
    _pred_edges,
    _pred_nodes,
)
from tracking_cellmot.division_metrics import evaluate_divisions  # noqa: E402


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_video(
    archive: zipfile.ZipFile,
    dataset: str,
    chunks: list[bytes],
    *,
    split: str = "train",
    node_count: int = 64,
    edge_count: int = 63,
) -> None:
    prefix = f"{split}/{dataset}"
    archive.writestr(f"{prefix}.zarr/zarr.json", _json_bytes({"zarr_format": 3}))
    archive.writestr(
        f"{prefix}.zarr/0/zarr.json",
        _json_bytes({"zarr_format": 3, "shape": [4, 8, 16, 16], "data_type": "uint8"}),
    )
    for index, chunk in enumerate(chunks):
        archive.writestr(f"{prefix}.zarr/0/c/{index}/0/0/0", chunk)
    archive.writestr(
        f"{prefix}.geff/nodes/ids/zarr.json",
        _json_bytes({"zarr_format": 3, "shape": [node_count]}),
    )
    archive.writestr(
        f"{prefix}.geff/edges/ids/zarr.json",
        _json_bytes({"zarr_format": 3, "shape": [edge_count, 2]}),
    )
    archive.writestr(
        f"{prefix}.geff/nodes/props/t/values/zarr.json",
        _json_bytes({"zarr_format": 3, "shape": [node_count]}),
    )


def _write_fixture_zip(path: Path, count: int = 24) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(count):
            value = bytes([index]) * 256
            _write_video(archive, f"44b6_{index:03d}", [value, value, value], node_count=64 + index)
    return path


def _write_split_fixture_zip(path: Path, *, train_count: int, test_count: int) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for split, count in (("train", train_count), ("test", test_count)):
            for index in range(count):
                value = bytes([index % 256]) * 64
                _write_video(
                    archive,
                    f"{split}_{index:03d}",
                    [value, value, value],
                    split=split,
                )
    return path


def _generated_image_members(archive_path: Path, dataset: str) -> list[zipfile.ZipInfo]:
    """Read exact generated image member names from the fixture central directory."""

    prefix = f"train/{dataset}.zarr/0/c/"
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info for info in archive.infolist() if info.filename.startswith(prefix)
        ]
    if not members:
        raise AssertionError(f"fixture has no generated image members for {dataset}")
    return members


class _TrackedImageStream:
    def __init__(self, handle: object, read_sizes: list[int]) -> None:
        self._handle = handle
        self._read_sizes = read_sizes

    def __enter__(self) -> "_TrackedImageStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()  # type: ignore[union-attr]

    def read(self, size: int = -1) -> bytes:
        self._read_sizes.append(size)
        return self._handle.read(size)  # type: ignore[union-attr]


class _FixedImageStream:
    def __init__(self, payload: bytes, read_sizes: list[int]) -> None:
        self._payload = payload
        self._read_sizes = read_sizes

    def __enter__(self) -> "_FixedImageStream":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self._read_sizes.append(size)
        return self._payload if size < 0 else self._payload[:size]


class _RecordingZipFile:
    """ZIP proxy that distinguishes direct reads from bounded image streams."""

    source_zip_file = zipfile.ZipFile
    stream_overrides: dict[str, bytes | BaseException] = {}
    instances: list["_RecordingZipFile"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._archive = self.source_zip_file(*args, **kwargs)
        self.direct_read_calls: list[str] = []
        self.stream_open_calls: list[str] = []
        self.stream_read_sizes: list[int] = []
        self.instances.append(self)

    def __enter__(self) -> "_RecordingZipFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._archive.close()

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self._archive.infolist()

    def read(self, info: zipfile.ZipInfo) -> bytes:
        self.direct_read_calls.append(info.filename)
        return self._archive.read(info)

    def open(self, info: zipfile.ZipInfo, *args: object, **kwargs: object) -> object:
        self.stream_open_calls.append(info.filename)
        outcome = self.stream_overrides.get(info.filename)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bytes):
            return _FixedImageStream(outcome, self.stream_read_sizes)
        return _TrackedImageStream(
            self._archive.open(info, *args, **kwargs), self.stream_read_sizes
        )


def _line_gt() -> dict[str, object]:
    return {
        "nodes": {
            "node_id": np.asarray([1, 2], dtype=np.int64),
            "t": np.asarray([0, 1], dtype=np.int64),
            "z": np.asarray([0.0, 0.0]),
            "y": np.asarray([0.0, 0.0]),
            "x": np.asarray([0.0, 0.0]),
        },
        "edges": np.asarray([[1, 2]], dtype=np.int64),
        "estimated_number_of_nodes": 2.0,
    }


class G02IntegrityTests(unittest.TestCase):
    def tearDown(self) -> None:
        _RecordingZipFile.stream_overrides = {}
        _RecordingZipFile.instances = []

    def _valid_rows(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        nodes = [
            {"dataset": "a", "node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            {"dataset": "a", "node_id": 2, "t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"dataset": "b", "node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        ]
        edges = [{"dataset": "a", "source_id": 1, "target_id": 2}]
        return nodes, edges

    def test_strict_submission_validation_rejects_malformed_rows(self) -> None:
        nodes, edges = self._valid_rows()
        validate_submission_rows(nodes, edges, {"a", "b"})
        malformed: list[tuple[str, list[dict[str, object]], list[dict[str, object]]]] = []
        missing = [dict(row) for row in nodes]
        del missing[0]["x"]
        malformed.append(("missing_coordinate", missing, edges))
        wrong_type = [dict(row) for row in nodes]
        wrong_type[0]["node_id"] = 1.0
        malformed.append(("wrong_node_type", wrong_type, edges))
        duplicate_node = [dict(row) for row in nodes]
        duplicate_node[1]["node_id"] = 1
        malformed.append(("duplicate_node", duplicate_node, edges))
        missing_endpoint = [dict(row) for row in edges]
        missing_endpoint[0]["target_id"] = 99
        malformed.append(("missing_endpoint", nodes, missing_endpoint))
        self_edge = [dict(row) for row in edges]
        self_edge[0]["target_id"] = 1
        malformed.append(("self_edge", nodes, self_edge))
        duplicate_edge = [dict(row) for row in edges] * 2
        malformed.append(("duplicate_edge", nodes, duplicate_edge))
        backward = [dict(row) for row in edges]
        backward[0] = {"dataset": "a", "source_id": 2, "target_id": 1}
        malformed.append(("backward_time", nodes, backward))
        missing_dataset = [dict(row) for row in nodes if row["dataset"] != "b"]
        malformed.append(("missing_dataset", missing_dataset, edges))
        unexpected = [dict(row) for row in nodes]
        unexpected[0]["untrusted_note"] = "must_not_be_ignored"
        malformed.append(("unexpected_column", unexpected, edges))
        for name, bad_nodes, bad_edges in malformed:
            with self.subTest(name=name):
                with self.assertRaises(SubmissionIntegrityError):
                    validate_submission_rows(bad_nodes, bad_edges, {"a", "b"})
        with self.assertRaises(SubmissionIntegrityError):
            score_submission_rows(
                [{"dataset": "a", "node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0}],
                [{"dataset": "a", "source_id": 1, "target_id": 1}],
                {"a"},
                {},
            )

    def test_official_wrapper_has_exact_7um_boundary_and_deterministic_tie(self) -> None:
        gt = _line_gt()
        boundary_x = 7.0 / DEFAULT_SCALE_ZYX[2]
        boundary_nodes = [
            {"dataset": "boundary", "node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            {"dataset": "boundary", "node_id": 2, "t": 1, "z": 0.0, "y": 0.0, "x": boundary_x},
        ]
        row = official_metric_row(
            boundary_nodes,
            [{"source_id": 1, "target_id": 2}],
            gt,
            SCALE_ZYX,
        )
        self.assertEqual(row["edge_tp"], 1.0)
        beyond_nodes = [dict(item) for item in boundary_nodes]
        beyond_nodes[1]["x"] = (7.0001 / DEFAULT_SCALE_ZYX[2])
        beyond = official_metric_row(
            beyond_nodes,
            [{"source_id": 1, "target_id": 2}],
            gt,
            SCALE_ZYX,
        )
        self.assertEqual(beyond["edge_tp"], 0.0)
        tie_nodes = [
            {"dataset": "tie", "node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            {"dataset": "tie", "node_id": 2, "t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            {"dataset": "tie", "node_id": 3, "t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        ]
        tie_edges = [{"source_id": 1, "target_id": 2}, {"source_id": 1, "target_id": 3}]
        first = official_metric_row(tie_nodes, tie_edges, gt, SCALE_ZYX)
        second = official_metric_row(tie_nodes, tie_edges, gt, SCALE_ZYX)
        self.assertEqual(first["edge_tp"], 1.0)
        self.assertEqual(first["edge_fp"], 1.0)
        self.assertEqual(first, second)

    def test_division_connected_component_cases_call_local_official_code(self) -> None:
        gt = _gt(DIVISION_NODES, DIVISION_EDGES, 6.0)
        gt_graph = geff_arrays_to_graph(gt)
        cases = {
            "self": (DIVISION_EDGES, (1, 0, 0)),
            "missing_fork": ([(1, 2), (2, 3), (3, 5), (4, 6)], (0, 1, 0)),
            # The extra 1->3 edge makes node 3 a shared child. The organizer
            # rejects both resulting malformed forks: one FN and two FP.
            "extra_fork": ([*DIVISION_EDGES, (1, 3)], (0, 1, 2)),
        }
        for name, (edges, expected) in cases.items():
            with self.subTest(name=name):
                pred = prediction_to_graph(_pred_nodes(DIVISION_NODES), _pred_edges(edges))
                counts = evaluate_divisions(pred, gt_graph, scale=SCALE_ZYX, max_distance=7.0)
                self.assertEqual((counts.tp, counts.fn, counts.fp), expected)

    def test_registry_is_deterministic_and_unions_exact_and_near_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "duplicates.zip"
            exact = [b"e" * 256] * 3
            near_left = [b"n" * 256] * 3
            near_right = [b"n" * 256, b"n" * 256, b"n" * 255 + b"m"]
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                _write_video(archive, "44b6_exact_a", exact)
                _write_video(archive, "44b6_exact_b", exact)
                _write_video(archive, "44b6_near_a", near_left)
                _write_video(archive, "44b6_near_b", near_right)
            first = build_dataset_registry(archive_path)
            second = build_dataset_registry(archive_path)
            self.assertEqual(first, second)
            record_by_dataset = {record["dataset"]: record for record in first["records"]}
            self.assertEqual(
                record_by_dataset["44b6_exact_a"]["group_id"],
                record_by_dataset["44b6_exact_b"]["group_id"],
            )
            self.assertEqual(
                record_by_dataset["44b6_near_a"]["group_id"],
                record_by_dataset["44b6_near_b"]["group_id"],
            )
            self.assertTrue(first["bounded_reads"]["full_video_extraction"] is False)
            self.assertLessEqual(
                first["counts"]["selected_signature_chunks"],
                4 * 3,
            )
            self.assertTrue(
                any(
                    check["state"] == "distinct"
                    for check in first["near_duplicate_checks"]
                )
            )

    def test_registry_accepts_explicit_train_test_and_total_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = _write_split_fixture_zip(
                Path(temporary) / "official-counts.zip",
                train_count=MAX_TRAIN_VIDEOS,
                test_count=MAX_TEST_VIDEOS,
            )
            registry = build_dataset_registry(archive_path)
            self.assertEqual(MAX_VIDEOS, MAX_TRAIN_VIDEOS)
            self.assertEqual(registry["counts"]["train_videos"], MAX_TRAIN_VIDEOS)
            self.assertEqual(registry["counts"]["test_videos"], MAX_TEST_VIDEOS)
            self.assertEqual(registry["counts"]["videos"], MAX_TOTAL_VIDEOS)
            for name, expected in (
                ("max_train_videos", MAX_TRAIN_VIDEOS),
                ("max_test_videos", MAX_TEST_VIDEOS),
                ("max_total_videos", MAX_TOTAL_VIDEOS),
            ):
                self.assertEqual(registry["parameters"][name], expected)
                self.assertEqual(registry["bounded_reads"][name], expected)
            self.assertEqual(
                registry["bounded_reads"]["maximum_videos"], MAX_TOTAL_VIDEOS
            )
            self.assertEqual(
                registry["bounded_reads"]["maximum_videos_scope"],
                "total videos across train and test",
            )

    def test_registry_rejects_200_training_videos_before_payload_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = _write_split_fixture_zip(
                Path(temporary) / "too-many-train.zip",
                train_count=MAX_TRAIN_VIDEOS + 1,
                test_count=0,
            )
            with self.assertRaisesRegex(
                IntegrityError, r"200 training videos; G-02 is bounded to 199 training videos"
            ):
                build_dataset_registry(archive_path)

    def test_registry_rejects_5_test_videos_before_payload_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = _write_split_fixture_zip(
                Path(temporary) / "too-many-test.zip",
                train_count=0,
                test_count=MAX_TEST_VIDEOS + 1,
            )
            with self.assertRaisesRegex(
                IntegrityError, r"5 test videos; G-02 is bounded to 4 test videos"
            ):
                build_dataset_registry(archive_path)

    def test_registry_rejects_204_total_videos_with_split_specific_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = _write_split_fixture_zip(
                Path(temporary) / "too-many-total.zip",
                train_count=MAX_TRAIN_VIDEOS,
                test_count=MAX_TEST_VIDEOS + 1,
            )
            with self.assertRaisesRegex(
                IntegrityError, r"5 test videos; G-02 is bounded to 4 test videos"
            ):
                build_dataset_registry(archive_path)

    def test_signature_streaming_uses_bounded_prefixes_and_full_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "prefixes.zip"
            prefix_payloads = [bytes([index]) * 64 for index in range(5)]
            full_payload = b"f" * 16
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                _write_video(archive, "44b6_prefix", prefix_payloads)
                _write_video(archive, "44b6_full", [full_payload])
            prefix_infos = _generated_image_members(archive_path, "44b6_prefix")
            full_infos = _generated_image_members(archive_path, "44b6_full")
            self.assertEqual(len(prefix_infos), len(prefix_payloads))
            self.assertEqual(len(full_infos), 1)
            self.assertEqual(
                len({info.filename for info in prefix_infos}), len(prefix_payloads)
            )
            with mock.patch(
                "celltrack_alpha.g02_integrity.zipfile.ZipFile", _RecordingZipFile
            ):
                registry = build_dataset_registry(
                    archive_path,
                    signature_chunks_per_video=3,
                    max_uncompressed_chunk_bytes=32,
                )
            recorder = _RecordingZipFile.instances[0]
            records = {record["dataset"]: record for record in registry["records"]}
            prefix_chunks = records["44b6_prefix"]["signature"]["chunks"]
            full_chunk = records["44b6_full"]["signature"]["chunks"][0]

            self.assertEqual(records["44b6_prefix"]["signature"]["status"], "complete")
            self.assertEqual(
                [chunk["member_path"] for chunk in prefix_chunks],
                [
                    prefix_infos[index].filename.removeprefix(
                        "train/44b6_prefix.zarr/"
                    )
                    for index in (0, 2, 4)
                ],
            )
            for index, chunk in zip((0, 2, 4), prefix_chunks, strict=True):
                self.assertEqual(chunk["sampling_mode"], "prefix")
                self.assertTrue(chunk["truncated"])
                self.assertEqual(chunk["sampled_bytes"], 32)
                self.assertEqual(chunk["declared_uncompressed_size"], 64)
                self.assertEqual(
                    chunk["signature"]["sha256"],
                    hashlib.sha256(prefix_payloads[index][:32]).hexdigest(),
                )
            self.assertEqual(
                prefix_chunks[0]["central_directory_crc32"],
                f"{zlib.crc32(prefix_payloads[0]) & 0xFFFFFFFF:08x}",
            )
            self.assertEqual(full_chunk["sampling_mode"], "full")
            self.assertFalse(full_chunk["truncated"])
            self.assertEqual(full_chunk["sampled_bytes"], len(full_payload))
            self.assertEqual(full_chunk["signature"]["length"], len(full_payload))
            self.assertEqual(recorder.stream_read_sizes, [32, 32, 32, 32])
            self.assertTrue(all("/c/" not in name for name in recorder.direct_read_calls))

            sampled = [*prefix_chunks, full_chunk]
            total_sampled = sum(int(chunk["sampled_bytes"]) for chunk in sampled)
            self.assertEqual(registry["bounded_reads"]["actual_image_chunk_bytes_read"], total_sampled)
            self.assertEqual(total_sampled, 112)
            self.assertLessEqual(
                total_sampled, registry["bounded_reads"]["maximum_image_chunk_bytes"]
            )
            self.assertTrue(
                all(
                    int(chunk["sampled_bytes"])
                    <= int(chunk["declared_uncompressed_size"])
                    for chunk in sampled
                )
            )

    def test_full_chunk_short_stream_remains_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "short-full.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                _write_video(archive, "44b6_short", [b"s" * 16])
                _write_video(archive, "44b6_valid", [b"v" * 16])
            short_infos = _generated_image_members(archive_path, "44b6_short")
            self.assertEqual(len(short_infos), 1)
            _RecordingZipFile.stream_overrides = {
                short_infos[0].filename: b"s" * 15
            }
            with mock.patch(
                "celltrack_alpha.g02_integrity.zipfile.ZipFile", _RecordingZipFile
            ):
                registry = build_dataset_registry(
                    archive_path,
                    signature_chunks_per_video=1,
                    max_uncompressed_chunk_bytes=32,
                )
            records = {record["dataset"]: record for record in registry["records"]}
            short_chunk = records["44b6_short"]["signature"]["selected_chunks"][0]
            self.assertEqual(records["44b6_short"]["signature"]["status"], "unresolved")
            self.assertEqual(short_chunk["state"], "unreadable")
            self.assertEqual(short_chunk["error"], "IntegrityError")
            self.assertEqual(registry["grouping_status"], "blocked")
            self.assertEqual(registry["bounded_reads"]["actual_image_chunk_bytes_read"], 31)
            with self.assertRaises(IntegrityError):
                build_grouped_split_manifest(registry)

    def test_empty_and_unreadable_streams_remain_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unreadable.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                _write_video(archive, "44b6_empty", [b""])
                _write_video(archive, "44b6_error", [b"e" * 16])
            empty_infos = _generated_image_members(archive_path, "44b6_empty")
            error_infos = _generated_image_members(archive_path, "44b6_error")
            self.assertEqual(len(empty_infos), 1)
            self.assertEqual(len(error_infos), 1)
            _RecordingZipFile.stream_overrides = {
                empty_infos[0].filename: b"",
                error_infos[0].filename: OSError("unreadable"),
            }
            with mock.patch(
                "celltrack_alpha.g02_integrity.zipfile.ZipFile", _RecordingZipFile
            ):
                registry = build_dataset_registry(
                    archive_path,
                    signature_chunks_per_video=1,
                    max_uncompressed_chunk_bytes=32,
                )
            records = {record["dataset"]: record for record in registry["records"]}
            empty_chunk = records["44b6_empty"]["signature"]["selected_chunks"][0]
            error_chunk = records["44b6_error"]["signature"]["selected_chunks"][0]
            self.assertEqual(records["44b6_empty"]["signature"]["status"], "unresolved")
            self.assertEqual(records["44b6_error"]["signature"]["status"], "unresolved")
            self.assertEqual(empty_chunk["state"], "unreadable")
            self.assertEqual(empty_chunk["error"], "IntegrityError")
            self.assertEqual(error_chunk["state"], "unreadable")
            self.assertEqual(error_chunk["error"], "OSError")
            self.assertEqual(registry["bounded_reads"]["actual_image_chunk_bytes_read"], 0)
            self.assertEqual(registry["grouping_status"], "blocked")
            with self.assertRaises(IntegrityError):
                build_grouped_split_manifest(registry)

    def test_grouped_split_has_exact_audit_and_no_fit_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = build_dataset_registry(_write_fixture_zip(root / "fixture.zip"))
            split = build_grouped_split_manifest(registry, seed="fixed-seed")
            provenance = build_fit_provenance(split)
            self.assertEqual(split["audit"]["video_count"], 20)
            audit_groups = set(split["audit"]["group_ids"])
            self.assertEqual(len(split["folds"]), 3)
            for fold_id, fold in split["folds"].items():
                held_out = set(fold["held_out_group_ids"])
                fit = set(fold["fit_group_ids"])
                self.assertFalse(audit_groups & held_out)
                self.assertFalse(audit_groups & fit)
                self.assertFalse(held_out & fit)
                for role, sources in provenance["folds"][fold_id]["fit_sources"].items():
                    assert_fit_sources_allowed(provenance, fold_id, role, sources)
                    with self.assertRaises(IntegrityError):
                        assert_fit_sources_allowed(provenance, fold_id, role, audit_groups)

    def test_exact_audit_requirement_and_prefix_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            impossible = root / "impossible.zip"
            with zipfile.ZipFile(impossible, "w", compression=zipfile.ZIP_STORED) as archive:
                shared = [b"q" * 64] * 3
                for index in range(21):
                    _write_video(archive, f"44b6_shared_{index}", shared)
                for index in range(3):
                    value = bytes([index]) * 64
                    _write_video(archive, f"44b6_single_{index}", [value] * 3)
            with self.assertRaises(IntegrityError):
                build_grouped_split_manifest(build_dataset_registry(impossible))

            unresolved = root / "unresolved.zip"
            with zipfile.ZipFile(unresolved, "w", compression=zipfile.ZIP_STORED) as archive:
                _write_video(archive, "44b6_large", [b"z" * 128])
                _write_video(archive, "44b6_small", [b"z" * 16])
            registry = build_dataset_registry(
                unresolved,
                signature_chunks_per_video=1,
                max_uncompressed_chunk_bytes=32,
            )
            records = {record["dataset"]: record for record in registry["records"]}
            large_chunk = records["44b6_large"]["signature"]["chunks"][0]
            self.assertEqual(registry["grouping_status"], "ready")
            self.assertEqual(
                registry["counts"]["unresolved_candidate_duplicate_checks"], 0
            )
            self.assertEqual(records["44b6_large"]["signature"]["status"], "complete")
            self.assertEqual(large_chunk["sampling_mode"], "prefix")
            self.assertTrue(large_chunk["truncated"])
            self.assertEqual(large_chunk["sampled_bytes"], 32)
            self.assertEqual(large_chunk["declared_uncompressed_size"], 128)

    def test_freeze_outputs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = _write_fixture_zip(root / "fixture.zip")
            first = root / "first"
            second = root / "second"
            freeze_integrity_outputs(archive_path, first, seed="byte-stable")
            freeze_integrity_outputs(archive_path, second, seed="byte-stable")
            filenames = (
                "scorer_parity_v1.json",
                "dataset_registry_v1.json",
                "grouped_3fold_audit20_v1.json",
                "fit_provenance_v1.json",
            )
            for filename in filenames:
                with self.subTest(filename=filename):
                    self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()

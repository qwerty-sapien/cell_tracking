from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from celltrack_alpha import v32_preilp_cache as cache

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BUILDER = _load_script(
    "build_v32_preilp_cache_kernel", "scripts/build_v32_preilp_cache_kernel.py"
)
VALIDATOR = _load_script(
    "validate_v32_preilp_cache", "scripts/validate_v32_preilp_cache.py"
)

SHA = "a" * 64
OPEN_PATHS = (
    "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/44b6_cf8fed6b.zarr/0",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
)
OPEN_AUDIT_SHA = cache.sha256_bytes(
    b"".join(
        cache.canonical_json_bytes(
            {"schema": cache.OPEN_AUDIT_SCHEMA, "event": "input_open", "path": path}
        )
        for path in OPEN_PATHS
    )
)
RAW_ZARR_METADATA = [{"name": ".zarray", "bytes": 2, "sha256": SHA}]
RAW_ZARR_METADATA_SHA = cache.sha256_bytes(
    cache.canonical_json_bytes(RAW_ZARR_METADATA)
)


def node_rows() -> list[dict]:
    return [
        {"node_id": 10, "t": 0, "z": 1, "y": 2, "x": 3},
        {"node_id": 11, "t": 0, "z": 1, "y": 8, "x": 3},
        {"node_id": 20, "t": 1, "z": 1, "y": 3, "x": 3},
        # Selected isolated node: must survive the post-ILP node layer.
        {"node_id": 21, "t": 1, "z": 1, "y": 9, "x": 3},
        {"node_id": 30, "t": 2, "z": 1, "y": 4, "x": 3},
    ]


def complete_arrays() -> dict[str, np.ndarray]:
    pre_nodes = cache.make_nodes(node_rows())
    pre_edges = cache.make_pre_edges(
        [
            {"source_id": 10, "target_id": 20, "edge_prob": 0.9, "edge_dist": 1.0},
            {"source_id": 11, "target_id": 21, "edge_prob": 0.8, "edge_dist": 1.0},
            {"source_id": 20, "target_id": 30, "edge_prob": 0.7, "edge_dist": 1.0},
        ]
    )
    post_nodes = cache.make_nodes(node_rows())
    post_edges = cache.make_post_edges(
        [
            {"source_id": 10, "target_id": 20, "edge_prob": 0.9},
            {"source_id": 20, "target_id": 30, "edge_prob": 0.7},
        ]
    )
    final_nodes = cache.make_nodes(node_rows())
    final_edges = cache.make_final_edges(
        [
            {"source_id": 10, "target_id": 20},
            {"source_id": 20, "target_id": 30},
        ]
    )
    return {
        "pre_ilp_nodes": pre_nodes,
        "pre_ilp_edges": pre_edges,
        "post_ilp_nodes": post_nodes,
        "post_ilp_edges": post_edges,
        "frame_pairs": cache.make_frame_pairs(pre_nodes, pre_edges, frame_count=3),
        "base_final_nodes": final_nodes,
        "base_final_edges": final_edges,
    }


def metadata(dataset: str = "44b6_cf8fed6b", *, audit_sha256: str = SHA) -> dict:
    return {
        "schema": cache.DATASET_SCHEMA,
        "dataset": dataset,
        "identity": f"train:{dataset}",
        "population_role": "canary",
        "frame_count": 3,
        "voxel_scale_um": list(cache.VOXEL_SCALE_UM),
        "raw_archive_sha256": cache.IMMUTABLE_POPULATION_HASHES["raw_archive_sha256"],
        "raw_zarr_metadata_sha256": RAW_ZARR_METADATA_SHA,
        "effective_config_sha256": SHA,
        "opened_path_audit_sha256": audit_sha256,
        "image_access_mode": "raw_zarr_images_only",
        "label_free": True,
        "evaluate": False,
    }


def producer() -> dict:
    value = {
        "schema": cache.PRODUCER_SCHEMA,
        "public_notebook_sha256": SHA,
        "builder_sha256": SHA,
        "package_source_sha256": SHA,
        "source_freeze_sha256": SHA,
        "base_predictor_sha256": SHA,
        "runtime_predictor_sha256": SHA,
        "terminal_receipt_sha256": SHA,
        "terminal_inventory_sha256": SHA,
        "terminal_open_audit_sha256": SHA,
        "terminal_submission_sha256": cache.sha256_bytes(
            fragment().replace(b"\n", b"\r\n")
        ),
        "runtime_split_sha256": cache.expected_runtime_split_sha256(
            ["train:44b6_cf8fed6b"]
        ),
        "primary_manifest_sha256": SHA,
        "primary_weight_sha256": SHA,
        "primary_config_sha256": SHA,
        "secondary_manifest_sha256": SHA,
        "secondary_weight_sha256": SHA,
        "secondary_config_sha256": SHA,
        "effective_environment_sha256": SHA,
        "label_free": True,
        "evaluate": False,
        "m3_mounted": False,
    }
    value.update(cache.IMMUTABLE_PRODUCER_HASHES)
    value["terminal_open_audit_sha256"] = cache.sha256_bytes(
        b"".join(
            cache.canonical_json_bytes(event) for event in runtime_open_events()
        )
    )
    return value


def population(dataset: str = "44b6_cf8fed6b") -> dict:
    value = {
        "schema": cache.POPULATION_SCHEMA,
        "role": "canary",
        "identities": [f"train:{dataset}"],
        "population_file_sha256": SHA,
        "population_self_sha256": SHA,
        "raw_archive_sha256": SHA,
    }
    value.update(cache.IMMUTABLE_POPULATION_HASHES)
    return value


def fragment(dataset: str = "44b6_cf8fed6b") -> bytes:
    return (
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        f"0,{dataset},node,10,0,1,2,3,-1,-1\n"
        f"1,{dataset},node,11,0,1,8,3,-1,-1\n"
        f"2,{dataset},node,20,1,1,3,3,-1,-1\n"
        f"3,{dataset},node,21,1,1,9,3,-1,-1\n"
        f"4,{dataset},node,30,2,1,4,3,-1,-1\n"
        f"5,{dataset},edge,-1,-1,-1,-1,-1,10,20\n"
        f"6,{dataset},edge,-1,-1,-1,-1,-1,20,30\n"
    ).encode()


class FakeRustworkxGraph:
    def __init__(self, nodes: dict[int, dict], edges: list[tuple[int, int, dict]]):
        self._nodes = nodes
        self._edges = edges

    def node_indices(self):
        raise AssertionError("direct exporter must not materialize node indices")

    def __getitem__(self, node_id: int):
        return self._nodes[node_id]

    def weighted_edge_list(self):
        return list(self._edges)


class FakeTracksGraph:
    def __init__(
        self,
        nodes: dict[int, dict],
        node_ids: list[int],
        edges: list[tuple[int, int, dict]],
    ):
        self.rx_graph = FakeRustworkxGraph(nodes, edges)
        self._node_ids = node_ids

    def node_ids(self):
        raise AssertionError("direct exporter must not call graph.node_ids()")

    def num_nodes(self):
        return len(self.rx_graph._nodes)

    def num_edges(self):
        return len(self.rx_graph._edges)


def runtime_open_events(*, config_sha256: str | None = None) -> list[dict]:
    metadata = RAW_ZARR_METADATA
    common = {
        "schema": cache.OPEN_AUDIT_SCHEMA,
        "event": "input_open",
        "shard": "single",
    }
    split_payload = cache.expected_runtime_split_bytes(["train:44b6_cf8fed6b"])
    config_sha = config_sha256 or cache.IMMUTABLE_PRODUCER_HASHES[
        "primary_config_sha256"
    ]
    return [
        {
            **common,
            "path": "/kaggle/input/competition/train/44b6_cf8fed6b.zarr/0",
            "kind": "zarr_array",
            "metadata": metadata,
            "metadata_sha256": cache.sha256_bytes(cache.canonical_json_bytes(metadata)),
        },
        {
            **common,
            "path": "/kaggle/input/primary/edge_predictor_best.pth",
            "kind": "file",
            "bytes": 1,
            "sha256": cache.IMMUTABLE_PRODUCER_HASHES["primary_weight_sha256"],
        },
        {
            **common,
            "path": "/kaggle/input/primary/config.json",
            "kind": "file",
            "bytes": 1,
            "sha256": config_sha,
        },
        {
            **common,
            "path": "/kaggle/input/secondary/edge_predictor_best.pth",
            "kind": "file",
            "bytes": 1,
            "sha256": cache.IMMUTABLE_PRODUCER_HASHES["secondary_weight_sha256"],
        },
        {
            **common,
            "path": "/kaggle/input/secondary/config.json",
            "kind": "file",
            "bytes": 1,
            "sha256": config_sha,
        },
        {
            **common,
            "path": "kaggle_test_splits_50ep.json",
            "kind": "file",
            "bytes": len(split_payload),
            "sha256": cache.sha256_bytes(split_payload),
        },
    ]


def write_runtime_open_audit(path: Path, *, config_sha256: str | None = None) -> str:
    payload = b"".join(
        cache.canonical_json_bytes(event)
        for event in runtime_open_events(config_sha256=config_sha256)
    )
    return cache.atomic_write_bytes(path, payload)


def write_sealed(root: Path, dataset: str = "44b6_cf8fed6b") -> dict:
    audit_sha = write_runtime_open_audit(root / cache.OPEN_AUDIT_NAME)
    entry = cache.write_dataset_cache(
        root,
        dataset=dataset,
        metadata=metadata(dataset, audit_sha256=audit_sha),
        arrays=complete_arrays(),
        base_submission_fragment=fragment(dataset),
    )
    bound_producer = producer()
    bound_producer["terminal_open_audit_sha256"] = audit_sha
    return cache.write_cache_manifest(
        root,
        producer=bound_producer,
        population=population(dataset),
        dataset_entries=[entry],
        open_audit_sha256=audit_sha,
    )


class V32PreILPCacheTests(unittest.TestCase):
    def test_complete_cache_round_trip_and_mmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written = write_sealed(root)
            result = cache.validate_cache(
                root,
                expected_identities=["train:44b6_cf8fed6b"],
                expected_producer={
                    **producer(),
                    "terminal_open_audit_sha256": written["open_audit"]["sha256"],
                },
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["dataset_count"], 1)
            self.assertEqual(result["manifest_sha256"], written["manifest_sha256"])
            _, arrays = cache.load_dataset_arrays(root, "44b6_cf8fed6b")
            self.assertIsInstance(arrays["pre_ilp_nodes"], np.memmap)
            self.assertIn(21, arrays["post_ilp_nodes"]["node_id"])

    def test_two_independent_writes_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            write_sealed(Path(first))
            write_sealed(Path(second))
            first_files = {
                path.relative_to(first): cache.sha256_file(path)
                for path in Path(first).rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second): cache.sha256_file(path)
                for path in Path(second).rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_forged_immutable_producer_and_population_hashes_are_rejected(self) -> None:
        forged_producer = producer()
        forged_producer["public_notebook_sha256"] = SHA
        with self.assertRaisesRegex(cache.V32CacheError, "immutable binding"):
            cache.validate_producer_bindings(forged_producer)
        forged_population = population()
        forged_population["raw_archive_sha256"] = SHA
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_sha = cache.write_open_audit(
                root / cache.OPEN_AUDIT_NAME, ["/raw/image.zarr/0"]
            )
            entry = cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(audit_sha256=audit_sha),
                arrays=complete_arrays(),
                base_submission_fragment=fragment(),
            )
            with self.assertRaisesRegex(cache.V32CacheError, "immutable binding"):
                cache.write_cache_manifest(
                    root,
                    producer=producer(),
                    population=forged_population,
                    dataset_entries=[entry],
                    open_audit_sha256=audit_sha,
                )

    def test_writer_refuses_dataset_and_manifest_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = write_sealed(root)
            with self.assertRaisesRegex(cache.V32CacheError, "overwrite dataset"):
                cache.write_dataset_cache(
                    root,
                    dataset="44b6_cf8fed6b",
                    metadata=metadata(),
                    arrays=complete_arrays(),
                    base_submission_fragment=fragment(),
                )
            with self.assertRaisesRegex(cache.V32CacheError, "overwrite"):
                cache.write_cache_manifest(
                    root,
                    producer=producer(),
                    population=population(),
                    dataset_entries=sealed["datasets"],
                    open_audit_sha256=sealed["open_audit"]["sha256"],
                )

    def test_missing_top_manifest_is_not_a_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(),
                arrays=complete_arrays(),
                base_submission_fragment=fragment(),
            )
            with self.assertRaisesRegex(
                cache.V32CacheError, "cannot load|root file-set drift"
            ):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_container_corruption_and_manifest_tampering_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            path = root / "datasets/44b6_cf8fed6b/PRE_ILP_NODES.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"X")
            with self.assertRaisesRegex(cache.V32CacheError, "hash mismatch"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_truncated_array_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            path = root / "datasets/44b6_cf8fed6b/POST_ILP_EDGES.npy"
            path.write_bytes(path.read_bytes()[:32])
            with self.assertRaises(cache.V32CacheError):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_label_ranker_and_scorer_paths_are_rejected(self) -> None:
        protected = (
            "/data/labels/video.geff",
            "/repo/scorer/official.py",
            "/input/local_association_ranker/model.pt",
            "/x/ground_truth/table.csv",
            "/protected/gt/video.csv",
            "/protected/B0_OUTCOMES.json",
            "/mounted/M3/model.bin",
        )
        for path in protected:
            with self.subTest(path=path), self.assertRaises(cache.V32CacheError):
                cache.assert_label_free_open_paths([path])

    def test_open_audit_event_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            audit = root / cache.OPEN_AUDIT_NAME
            audit.write_text(
                json.dumps(
                    {
                        "schema": cache.OPEN_AUDIT_SCHEMA,
                        "event": "input_open",
                        "path": "/data/labels/a.geff",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / cache.TOP_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["open_audit"]["sha256"] = cache.sha256_file(audit)
            manifest_path.write_bytes(cache.canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                cache.V32CacheError, "sealed open audit|protected input"
            ):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_dataset_open_audit_cross_binding_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_sha = cache.write_open_audit(
                root / cache.OPEN_AUDIT_NAME, ["/raw/image.zarr/0"]
            )
            entry = cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(audit_sha256="b" * 64),
                arrays=complete_arrays(),
                base_submission_fragment=fragment(),
            )
            with self.assertRaisesRegex(cache.V32CacheError, "sealed open audit"):
                cache.write_cache_manifest(
                    root,
                    producer=producer(),
                    population=population(),
                    dataset_entries=[entry],
                    open_audit_sha256=audit_sha,
                )

    def test_duplicate_nodes_edges_dangling_and_nonconsecutive_fail(self) -> None:
        cases = []
        arrays = complete_arrays()
        duplicate_node = {key: value.copy() for key, value in arrays.items()}
        duplicate_node["pre_ilp_nodes"][1]["node_id"] = 10
        cases.append(("duplicate node", duplicate_node))
        duplicate_edge = {key: value.copy() for key, value in arrays.items()}
        duplicate_edge["pre_ilp_edges"][1]["source_id"] = 10
        duplicate_edge["pre_ilp_edges"][1]["target_id"] = 20
        cases.append(("duplicate edge", duplicate_edge))
        dangling = {key: value.copy() for key, value in arrays.items()}
        dangling["post_ilp_edges"][0]["source_id"] = 999
        cases.append(("dangling", dangling))
        nonconsecutive = {key: value.copy() for key, value in arrays.items()}
        nonconsecutive["post_ilp_edges"][0]["target_id"] = 30
        cases.append(("nonconsecutive", nonconsecutive))
        for label, value in cases:
            with self.subTest(label=label), self.assertRaises(cache.V32CacheError):
                cache.validate_dataset_arrays(value, frame_count=3)

    def test_nonfinite_probability_position_and_wrong_micron_projection_fail(self) -> None:
        mutations = []
        arrays = complete_arrays()
        bad_prob = {key: value.copy() for key, value in arrays.items()}
        bad_prob["pre_ilp_edges"][0]["edge_prob"] = np.nan
        mutations.append(bad_prob)
        bad_position = {key: value.copy() for key, value in arrays.items()}
        bad_position["pre_ilp_nodes"][0]["z_vox"] = np.inf
        mutations.append(bad_position)
        bad_scale = {key: value.copy() for key, value in arrays.items()}
        bad_scale["post_ilp_nodes"][0]["z_um"] += 1.0
        mutations.append(bad_scale)
        for value in mutations:
            with self.assertRaises(cache.V32CacheError):
                cache.validate_dataset_arrays(value, frame_count=3)

    def test_post_ilp_must_be_exact_subset_with_probability_identity(self) -> None:
        absent = complete_arrays()
        absent["post_ilp_edges"][0]["source_id"] = 11
        absent["post_ilp_edges"][0]["target_id"] = 20
        with self.assertRaisesRegex(cache.V32CacheError, "absent"):
            cache.validate_dataset_arrays(absent, frame_count=3)
        drift = complete_arrays()
        drift["post_ilp_edges"][0]["edge_prob"] = 0.91
        with self.assertRaisesRegex(cache.V32CacheError, "probability drift"):
            cache.validate_dataset_arrays(drift, frame_count=3)

        node_drift = complete_arrays()
        node_drift["post_ilp_nodes"][0]["x_vox"] += 100.0
        node_drift["post_ilp_nodes"][0]["x_um"] = (
            node_drift["post_ilp_nodes"][0]["x_vox"] * cache.VOXEL_SCALE_UM[2]
        )
        with self.assertRaisesRegex(cache.V32CacheError, "node attributes drift"):
            cache.validate_dataset_arrays(node_drift, frame_count=3)

    def test_post_ilp_selected_isolated_node_is_preserved(self) -> None:
        arrays = complete_arrays()
        result = cache.validate_dataset_arrays(arrays, frame_count=3)
        self.assertEqual(result["post_ilp_nodes"], 5)
        self.assertNotIn(21, arrays["post_ilp_edges"]["source_id"])
        self.assertNotIn(21, arrays["post_ilp_edges"]["target_id"])

    def test_frame_pair_empty_case_is_explicit(self) -> None:
        nodes = cache.make_nodes(
            [
                {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
                {"node_id": 2, "t": 2, "z": 0, "y": 0, "x": 0},
            ]
        )
        edges = cache.make_pre_edges([])
        pairs = cache.make_frame_pairs(nodes, edges, frame_count=3)
        self.assertEqual(pairs["skipped_empty"].tolist(), [1, 1])
        self.assertEqual(pairs["candidate_count"].tolist(), [0, 0])

    def test_frame_pair_candidate_slice_tampering_fails(self) -> None:
        arrays = complete_arrays()
        arrays["frame_pairs"][1]["candidate_start"] = 0
        with self.assertRaisesRegex(cache.V32CacheError, "not contiguous"):
            cache.validate_dataset_arrays(arrays, frame_count=3)

    def test_dtype_and_pickle_surface_fail_closed(self) -> None:
        arrays = complete_arrays()
        arrays["pre_ilp_nodes"] = arrays["pre_ilp_nodes"].astype(
            [(name, ">f8") if name == "z_vox" else (name, dtype)
             for name, dtype in cache.NODE_DTYPE.descr]
        )
        with self.assertRaisesRegex(cache.V32CacheError, "dtype drift"):
            cache.validate_dataset_arrays(arrays, frame_count=3)

    def test_population_order_extra_and_missing_dataset_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_sha = cache.write_open_audit(root / cache.OPEN_AUDIT_NAME, ["/image.zarr/0"])
            entry = cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(),
                arrays=complete_arrays(),
                base_submission_fragment=fragment(),
            )
            wrong = population()
            wrong["identities"] = ["train:other"]
            with self.assertRaisesRegex(
                cache.V32CacheError, "canary population|ordered population"
            ):
                cache.write_cache_manifest(
                    root,
                    producer=producer(),
                    population=wrong,
                    dataset_entries=[entry],
                    open_audit_sha256=audit_sha,
                )

    def test_extra_dataset_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            (root / "datasets/unlisted").mkdir()
            with self.assertRaisesRegex(cache.V32CacheError, "extra dataset"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_sealed_root_and_dataset_inventories_reject_untracked_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            (root / "B0_OUTCOMES.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(cache.V32CacheError, "root file-set drift"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sealed(root)
            dataset_dir = root / "datasets/44b6_cf8fed6b"
            (dataset_dir / "M3_RANKER.bin").write_bytes(b"forbidden")
            with self.assertRaisesRegex(cache.V32CacheError, "file-set drift"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_sealed_dataset_rejects_symlinked_manifested_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            root = outer / "cache"
            root.mkdir()
            write_sealed(root)
            dataset_dir = root / "datasets/44b6_cf8fed6b"
            target = dataset_dir / "PRE_ILP_NODES.npy"
            backing = outer / "backing.npy"
            target.replace(backing)
            target.symlink_to(backing)
            with self.assertRaisesRegex(cache.V32CacheError, "non-regular"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=producer(),
                )

    def test_coherent_base_final_mutation_cannot_rebind_terminal_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_sha = cache.write_open_audit(
                root / cache.OPEN_AUDIT_NAME, OPEN_PATHS
            )
            arrays = complete_arrays()
            arrays["base_final_edges"] = arrays["base_final_edges"][:1].copy()
            altered_fragment = b"\n".join(fragment().splitlines()[:-1]) + b"\n"
            entry = cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(audit_sha256=audit_sha),
                arrays=arrays,
                base_submission_fragment=altered_fragment,
            )
            expected = producer()
            expected["terminal_open_audit_sha256"] = audit_sha
            with self.assertRaisesRegex(
                cache.V32CacheError, "terminal submission"
            ):
                cache.write_cache_manifest(
                    root,
                    producer=expected,
                    population=population(),
                    dataset_entries=[entry],
                    open_audit_sha256=audit_sha,
                )

    def test_dataset_provenance_is_cross_bound_to_top_level_authorities(self) -> None:
        cases = (
            ("population_role", "cv57", "population role"),
            ("raw_archive_sha256", "b" * 64, "raw archive"),
            ("effective_config_sha256", "b" * 64, "effective config"),
        )
        for key, replacement, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                audit_sha = write_runtime_open_audit(root / cache.OPEN_AUDIT_NAME)
                dataset_metadata = metadata(audit_sha256=audit_sha)
                dataset_metadata[key] = replacement
                entry = cache.write_dataset_cache(
                    root,
                    dataset="44b6_cf8fed6b",
                    metadata=dataset_metadata,
                    arrays=complete_arrays(),
                    base_submission_fragment=fragment(),
                )
                expected = producer()
                expected["terminal_open_audit_sha256"] = audit_sha
                with self.assertRaisesRegex(cache.V32CacheError, message):
                    cache.write_cache_manifest(
                        root,
                        producer=expected,
                        population=population(),
                        dataset_entries=[entry],
                        open_audit_sha256=audit_sha,
                    )

    def test_raw_zarr_metadata_must_match_actual_open_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = runtime_open_events()
            events[0] = {**events[0], "metadata_sha256": "b" * 64}
            audit_path = root / cache.OPEN_AUDIT_NAME
            audit_sha = cache.atomic_write_bytes(
                audit_path,
                b"".join(cache.canonical_json_bytes(event) for event in events),
            )
            entry = cache.write_dataset_cache(
                root,
                dataset="44b6_cf8fed6b",
                metadata=metadata(audit_sha256=audit_sha),
                arrays=complete_arrays(),
                base_submission_fragment=fragment(),
            )
            expected = producer()
            expected["terminal_open_audit_sha256"] = audit_sha
            cache.write_cache_manifest(
                root,
                producer=expected,
                population=population(),
                dataset_entries=[entry],
                open_audit_sha256=audit_sha,
            )
            with self.assertRaisesRegex(cache.V32CacheError, "raw Zarr metadata"):
                cache.validate_cache(
                    root,
                    expected_identities=["train:44b6_cf8fed6b"],
                    expected_producer=expected,
                )

    def test_unsafe_dataset_names_fail(self) -> None:
        for value in ("../escape", "a/b", "", ".hidden"):
            with self.subTest(value=value), self.assertRaises(cache.V32CacheError):
                cache.write_dataset_cache(
                    Path("/tmp/unused"),
                    dataset=value,
                    metadata=metadata(value),
                    arrays=complete_arrays(),
                    base_submission_fragment=b"",
                )

    def test_submission_fragment_parser_keeps_only_requested_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            path.write_bytes(
                fragment()
                + b"7,other,node,1,0,0,0,0,-1,-1\n"
            )
            nodes, edges, payload = cache.parse_submission_fragment(
                path, dataset="44b6_cf8fed6b"
            )
            self.assertEqual(len(nodes), 5)
            self.assertEqual(len(edges), 2)
            self.assertNotIn(b",other,", payload)

    def test_submission_parser_rejects_unknown_row_type_and_bad_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            path.write_text("dataset,row_type\n44b6_cf8fed6b,node\n", encoding="utf-8")
            with self.assertRaisesRegex(cache.V32CacheError, "columns drift"):
                cache.parse_submission_fragment(path, dataset="44b6_cf8fed6b")
            path.write_bytes(
                fragment().replace(b",node,10,", b",mystery,10,", 1)
            )
            with self.assertRaisesRegex(cache.V32CacheError, "unknown row_type"):
                cache.parse_submission_fragment(path, dataset="44b6_cf8fed6b")

    def test_partial_validator_accepts_only_unsealed_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            root = outer / "cache"
            target = root / "datasets/44b6_cf8fed6b"
            target.mkdir(parents=True)
            arrays = complete_arrays()
            stage_specs = {
                "pre_ilp": {
                    "nodes": ("PRE_ILP_NODES.npy", arrays["pre_ilp_nodes"]),
                    "edges": ("PRE_ILP_EDGES.npy", arrays["pre_ilp_edges"]),
                    "frame_pairs": ("FRAME_PAIRS.npy", arrays["frame_pairs"]),
                },
                "post_ilp": {
                    "nodes": ("POST_ILP_NODES.npy", arrays["post_ilp_nodes"]),
                    "edges": ("POST_ILP_EDGES.npy", arrays["post_ilp_edges"]),
                },
            }
            for stage, values in stage_specs.items():
                records = {}
                for key, (filename, value) in values.items():
                    np.save(target / filename, value, allow_pickle=False)
                    records[key] = {
                        "path": filename,
                        "sha256": cache.sha256_file(target / filename),
                        "bytes": (target / filename).stat().st_size,
                        "rows": len(value),
                        "shape": list(value.shape),
                        "dtype": value.dtype.descr,
                    }
                receipt = {
                    "schema": VALIDATOR.PARTIAL_SCHEMA,
                    "status": "PASS_UNSEALED",
                    "dataset": "44b6_cf8fed6b",
                    "identity": "train:44b6_cf8fed6b",
                    "stage": stage,
                    "frame_count": 3,
                    "label_free": True,
                    "evaluate": False,
                    "m3_mounted": False,
                    "source_zarr": "/kaggle/input/competition/train/44b6_cf8fed6b.zarr",
                    "files": records,
                }
                (target / f"{stage.upper()}_RECEIPT.json").write_text(json.dumps(receipt))
            write_runtime_open_audit(root / cache.OPEN_AUDIT_NAME)
            inventory_files = [
                {
                    "path": cache.OPEN_AUDIT_NAME,
                    "bytes": (root / cache.OPEN_AUDIT_NAME).stat().st_size,
                    "sha256": cache.sha256_file(root / cache.OPEN_AUDIT_NAME),
                },
                *[
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": cache.sha256_file(path),
                }
                for path in sorted(target.iterdir())
                ],
            ]
            terminal_receipt = outer / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
            terminal_receipt.write_text(
                json.dumps(
                    {
                        "schema": VALIDATOR.TERMINAL_RUN_SCHEMA,
                        "status": "PASS",
                        "variant": "public_v32",
                        "selected_datasets": ["44b6_cf8fed6b"],
                        "request_count": 1,
                        "public_script_version_id": 337501010,
                        "public_notebook_sha256": cache.IMMUTABLE_PRODUCER_HASHES[
                            "public_notebook_sha256"
                        ],
                        "primary_weight_sha256": cache.IMMUTABLE_PRODUCER_HASHES[
                            "primary_weight_sha256"
                        ],
                        "secondary_weight_sha256": cache.IMMUTABLE_PRODUCER_HASHES[
                            "secondary_weight_sha256"
                        ],
                        "weight_refinement": False,
                        "substrate_only": True,
                        "weight_refinement_minimum_local_score": 0.916,
                        "submission": {
                            "path": "/kaggle/working/submission.csv",
                            "sha256": SHA,
                            "rows": 7,
                        },
                        "run_stats": {"path": "/kaggle/working/run_stats.csv", "sha256": SHA},
                        "run_log": {"path": "/kaggle/working/run_log.jsonl", "sha256": SHA},
                        "nodes": 5,
                        "edges": 2,
                        "prediction_seconds": 1.0,
                        "cuda_device_count": 1,
                        "cuda_device_name": "synthetic-test-gpu",
                        "runtime_predict_script_sha256": SHA,
                        "v32_preilp_cache": {
                            "schema": VALIDATOR.TERMINAL_INVENTORY_SCHEMA,
                            "status": "PASS_UNSEALED",
                            "identities": ["train:44b6_cf8fed6b"],
                            "files": inventory_files,
                            "inventory_sha256": cache.sha256_bytes(
                                cache.canonical_json_bytes(inventory_files)
                            ),
                            "scientific_use_open": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            expected_partial_producer = producer()
            expected_partial_producer["terminal_receipt_sha256"] = cache.sha256_file(
                terminal_receipt
            )
            expected_partial_producer["terminal_inventory_sha256"] = cache.sha256_bytes(
                cache.canonical_json_bytes(inventory_files)
            )
            expected_partial_producer["terminal_open_audit_sha256"] = cache.sha256_file(
                root / cache.OPEN_AUDIT_NAME
            )
            expected_partial_producer["terminal_submission_sha256"] = SHA
            result = VALIDATOR.validate_partial(
                root,
                ["train:44b6_cf8fed6b"],
                terminal_receipt=terminal_receipt,
                expected_producer=expected_partial_producer,
            )
            self.assertEqual(result["status"], "PASS_UNSEALED")
            self.assertFalse(result["scientific_use_open"])
            backing_dataset = outer / "backing_dataset"
            target.replace(backing_dataset)
            target.symlink_to(backing_dataset, target_is_directory=True)
            with self.assertRaisesRegex(cache.V32CacheError, "symlink"):
                VALIDATOR.validate_partial(
                    root,
                    ["train:44b6_cf8fed6b"],
                    terminal_receipt=terminal_receipt,
                    expected_producer=expected_partial_producer,
                )
            target.unlink()
            backing_dataset.replace(target)
            valid_terminal = json.loads(terminal_receipt.read_text())
            mismatched_submission = copy.deepcopy(valid_terminal)
            mismatched_submission["submission"]["sha256"] = "b" * 64
            terminal_receipt.write_text(
                json.dumps(mismatched_submission), encoding="utf-8"
            )
            mismatched_expected = dict(expected_partial_producer)
            mismatched_expected["terminal_receipt_sha256"] = cache.sha256_file(
                terminal_receipt
            )
            with self.assertRaisesRegex(
                cache.V32CacheError, "terminal submission"
            ):
                VALIDATOR.validate_partial(
                    root,
                    ["train:44b6_cf8fed6b"],
                    terminal_receipt=terminal_receipt,
                    expected_producer=mismatched_expected,
                )

            for label, events, expected_message in (
                (
                    "config",
                    runtime_open_events(config_sha256="b" * 64),
                    "runtime config hashes drift",
                ),
                (
                    "split",
                    [
                        {**event, **({"sha256": "b" * 64} if index == 5 else {})}
                        for index, event in enumerate(runtime_open_events())
                    ],
                    "runtime split hash drift",
                ),
            ):
                with self.subTest(label=label):
                    audit_path = root / cache.OPEN_AUDIT_NAME
                    audit_path.write_bytes(
                        b"".join(cache.canonical_json_bytes(event) for event in events)
                    )
                    mutated_inventory = [
                        {
                            "path": cache.OPEN_AUDIT_NAME,
                            "bytes": audit_path.stat().st_size,
                            "sha256": cache.sha256_file(audit_path),
                        },
                        *inventory_files[1:],
                    ]
                    mutated_terminal = copy.deepcopy(valid_terminal)
                    mutated_terminal["v32_preilp_cache"]["files"] = mutated_inventory
                    mutated_terminal["v32_preilp_cache"]["inventory_sha256"] = (
                        cache.sha256_bytes(
                            cache.canonical_json_bytes(mutated_inventory)
                        )
                    )
                    terminal_receipt.write_text(
                        json.dumps(mutated_terminal), encoding="utf-8"
                    )
                    mutated_expected = dict(expected_partial_producer)
                    mutated_expected["terminal_open_audit_sha256"] = cache.sha256_file(
                        audit_path
                    )
                    mutated_expected["terminal_inventory_sha256"] = (
                        mutated_terminal["v32_preilp_cache"]["inventory_sha256"]
                    )
                    mutated_expected["terminal_receipt_sha256"] = cache.sha256_file(
                        terminal_receipt
                    )
                    with self.assertRaisesRegex(
                        cache.V32CacheError, expected_message
                    ):
                        VALIDATOR.validate_partial(
                            root,
                            ["train:44b6_cf8fed6b"],
                            terminal_receipt=terminal_receipt,
                            expected_producer=mutated_expected,
                        )
            with self.assertRaisesRegex(cache.V32CacheError, "unsafe dataset"):
                VALIDATOR.validate_partial(
                    root,
                    ["train:../escape"],
                    terminal_receipt=terminal_receipt,
                    expected_producer=expected_partial_producer,
                )
            forged_terminal = json.loads(terminal_receipt.read_text())
            forged_terminal["status"] = "FAIL"
            terminal_receipt.write_text(json.dumps(forged_terminal), encoding="utf-8")
            forged_expected = dict(expected_partial_producer)
            forged_expected["terminal_receipt_sha256"] = cache.sha256_file(
                terminal_receipt
            )
            with self.assertRaisesRegex(cache.V32CacheError, "outer terminal"):
                VALIDATOR.validate_partial(
                    root,
                    ["train:44b6_cf8fed6b"],
                    terminal_receipt=terminal_receipt,
                    expected_producer=forged_expected,
                )


class V32CacheKernelBuilderTests(unittest.TestCase):
    def test_preflight_control_selects_identities_without_video_slice_alias(self) -> None:
        namespace = {"os": os}
        with mock.patch.dict(
            os.environ,
            {
                "KAGGLE_PREFLIGHT_MODE": "stratified_both",
                "KAGGLE_PREFLIGHT_UNITS": "4",
            },
            clear=True,
        ):
            exec(BUILDER.preflight_control_source(), namespace)
            self.assertEqual(namespace["SELECTED_DATASETS"], [
                "44b6_d754aa59", "44b6_18ced818"
            ])
            self.assertEqual(namespace["SHIP_PREFLIGHT_UNITS"], 4)
            self.assertNotIn("BIOHUB_SLICE", os.environ)
        with mock.patch.dict(
            os.environ,
            {
                "KAGGLE_PREFLIGHT_MODE": "tiny_sparse",
                "KAGGLE_PREFLIGHT_UNITS": "2",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, r"\[3, 100\]"):
                exec(BUILDER.preflight_control_source(), {"os": os})

    def test_dependency_install_restart_is_one_shot_and_source_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "f2_v32_parameterized_child.py"
            source_path.write_text("# exact child fixture\n", encoding="utf-8")
            namespace = {
                "__file__": str(source_path),
                "hashlib": BUILDER.hashlib,
                "os": os,
                "Path": Path,
                "sys": __import__("sys"),
            }
            with mock.patch.dict(
                os.environ, {"BIOHUB_V32_DEPENDENCY_REEXEC_V1": ""}, clear=False
            ):
                exec(BUILDER.dependency_restart_source(), namespace)
                with (
                    mock.patch.object(os, "execve", side_effect=OSError("exec fixture")) as reexec,
                    self.assertRaisesRegex(OSError, "exec fixture"),
                ):
                    namespace["_v32_restart_after_dependency_install"]()
            executable, argv, environment = reexec.call_args.args
            self.assertEqual(executable, namespace["sys"].executable)
            self.assertEqual(argv[1], str(source_path.resolve()))
            token = environment["BIOHUB_V32_DEPENDENCY_REEXEC_V1"]
            self.assertEqual(token, namespace["_v32_dependency_reexec_token"]())

            authenticated = dict(namespace)
            with mock.patch.dict(
                os.environ,
                {"BIOHUB_V32_DEPENDENCY_REEXEC_V1": token},
                clear=False,
            ):
                exec(BUILDER.dependency_restart_source(), authenticated)
            self.assertTrue(authenticated["_V32_DEPENDENCY_REEXEC_AUTHENTICATED"])
            with self.assertRaisesRegex(ImportError, "second interpreter restart"):
                authenticated["_v32_restart_after_dependency_install"]()
            with mock.patch.dict(
                os.environ,
                {"BIOHUB_V32_DEPENDENCY_REEXEC_V1": "0" * 64},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                    exec(BUILDER.dependency_restart_source(), dict(namespace))

    def test_predictor_instrumentation_has_exact_pre_and_post_seams(self) -> None:
        base_path = (
            ROOT / "external/Tracking_Support/repo/scripts/predict_unet_transformer.py"
        )
        source = base_path.read_text(encoding="utf-8")
        instrumented = BUILDER.instrument_predictor_source(source)
        self.assertIn(BUILDER.INSTRUMENTATION_MARKER, instrumented)
        pre_call = (
            '_v32_export_graph(name, ds_path, weights_path, splits_file, "pre_ilp", graph)'
        )
        post_call = (
            '_v32_export_graph(name, ds_path, weights_path, splits_file, "post_ilp", graph)'
        )
        self.assertEqual(instrumented.count(pre_call), 1)
        self.assertEqual(instrumented.count(post_call), 1)
        self.assertLess(
            instrumented.index(pre_call),
            instrumented.index("solver.solve(graph)"),
        )
        self.assertIn(
            'os.environ.get("BIOHUB_V32_CPU_PREFLIGHT_SKIP_ILP", "") == "1"',
            instrumented,
        )
        self.assertIn(
            'and bool(os.environ.get("KAGGLE_PREFLIGHT_MODE", "").strip())',
            instrumented,
        )
        self.assertIn("and not torch.cuda.is_available()", instrumented)
        self.assertIn("if cfg.use_ilp and not _v32_skip_ilp", instrumented)
        self.assertGreater(
            instrumented.index(post_call),
            instrumented.index("solver.solve(graph)"),
        )
        self.assertIn("_v32_record_input_opens(opened)", instrumented)
        self.assertIn("source_zarr = _v32_source_zarr_path(ds_path)", instrumented)
        self.assertIn('base.parent / f"{base.stem}.zarr"', instrumented)
        self.assertIn("_v32_fcntl.flock", instrumented)
        self.assertIn('"kind": "file"', instrumented)
        self.assertIn('"metadata_sha256"', instrumented)
        self.assertIn('os.environ.get("BIOHUB_GPU_SHARD", "single")', instrumented)
        self.assertIn('"node_id": int(node_id)', instrumented)
        self.assertIn("V32 rustworkx node construction id drift", instrumented)
        self.assertNotIn("edge_logits_pair.cpu", instrumented)
        self.assertNotIn("graph.node_attrs()", instrumented)
        self.assertNotIn("graph.edge_attrs()", instrumented)
        self.assertNotIn("graph.node_ids()", instrumented)
        self.assertNotIn("node_indices()", instrumented)
        self.assertIn("graph.rx_graph.weighted_edge_list()", instrumented)

    def test_rustworkx_export_is_direct_mapped_stable_and_polars_free(self) -> None:
        namespace = {"np": np, "Path": Path, "os": os, "json": json}
        exec(BUILDER.predictor_export_helpers_source(), namespace)
        graph = FakeTracksGraph(
            nodes={
                1: {"t": 1, "z": 2.0, "y": 3.0, "x": 4.0, "node_id": 90},
                0: {"t": 0, "z": 1.0, "y": 2.0, "x": 3.0, "node_id": 30},
            },
            node_ids=[90, 30],
            edges=[
                (0, 1, {"edge_id": 8, "edge_prob": 0.8, "edge_dist": 2.5}),
                (0, 1, {"edge_id": 2, "edge_prob": 0.6, "edge_dist": 1.5}),
            ],
        )

        nodes = namespace["_v32_nodes"](graph)
        pre_edges = namespace["_v32_edges"](graph, "pre_ilp")
        post_edges = namespace["_v32_edges"](graph, "post_ilp")

        self.assertEqual(nodes["node_id"].tolist(), [30, 90])
        self.assertEqual(nodes["ordinal"].tolist(), [0, 1])
        self.assertEqual(pre_edges["ordinal"].tolist(), [0, 1])
        self.assertEqual(pre_edges["source_id"].tolist(), [30, 30])
        self.assertEqual(pre_edges["target_id"].tolist(), [90, 90])
        self.assertEqual(pre_edges["edge_prob"].tolist(), [0.6, 0.8])
        self.assertEqual(pre_edges["edge_dist_grid"].tolist(), [1.5, 2.5])
        self.assertEqual(post_edges["source_id"].tolist(), [30, 30])
        exporter = BUILDER.predictor_export_helpers_source()
        self.assertNotIn("node_attrs", exporter)
        self.assertNotIn("edge_attrs", exporter)
        self.assertNotIn("graph.node_ids()", exporter)
        self.assertNotIn("node_indices()", exporter)

    def test_rustworkx_export_rejects_key_mapping_and_count_drift(self) -> None:
        namespace = {"np": np, "Path": Path, "os": os, "json": json}
        exec(BUILDER.predictor_export_helpers_source(), namespace)
        cases = (
            (
                FakeTracksGraph(
                    nodes={
                        1: {
                            "node_id": 10,
                            "t": 0,
                            "z": 1.0,
                            "y": 2.0,
                            "x": 3.0,
                        }
                    },
                    node_ids=[10],
                    edges=[],
                ),
                namespace["_v32_nodes"],
                "local node id drift",
            ),
            (
                FakeTracksGraph(
                    nodes={0: {"t": 0, "z": 1.0, "y": 2.0}},
                    node_ids=[10],
                    edges=[],
                ),
                namespace["_v32_nodes"],
                "missing keys",
            ),
            (
                FakeTracksGraph(
                    nodes={
                        0: {
                            "node_id": "11",
                            "t": 0,
                            "z": 1.0,
                            "y": 2.0,
                            "x": 3.0,
                        }
                    },
                    node_ids=[10],
                    edges=[],
                ),
                namespace["_v32_nodes"],
                "node_id is not an integer",
            ),
            (
                FakeTracksGraph(
                    nodes={
                        0: {
                            "node_id": 10,
                            "t": 0,
                            "z": 1.0,
                            "y": 2.0,
                            "x": 3.0,
                        }
                    },
                    node_ids=[10],
                    edges=[
                        (0, 4, {"edge_id": 0, "edge_prob": 0.5, "edge_dist": 1.0})
                    ],
                ),
                lambda graph: namespace["_v32_edges"](graph, "pre_ilp"),
                "endpoint is absent",
            ),
            (
                FakeTracksGraph(
                    nodes={
                        0: {
                            "node_id": 10,
                            "t": 0,
                            "z": 1.0,
                            "y": 2.0,
                            "x": 3.0,
                        },
                        1: {
                            "node_id": 20,
                            "t": 1,
                            "z": 1.0,
                            "y": 2.0,
                            "x": 4.0,
                        },
                    },
                    node_ids=[10, 20],
                    edges=[
                        (
                            0,
                            1,
                            {
                                "edge_id": 0,
                                "edge_prob": 0.5,
                                "edge_dist": 1.0,
                                "source_id": 99,
                            },
                        )
                    ],
                ),
                lambda graph: namespace["_v32_edges"](graph, "pre_ilp"),
                "source_id mapping drift",
            ),
        )
        for graph, operation, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                RuntimeError, expected
            ):
                operation(graph)

    def test_predictor_instrumentation_refuses_reapplication_and_anchor_drift(self) -> None:
        base = (
            ROOT / "external/Tracking_Support/repo/scripts/predict_unet_transformer.py"
        ).read_text()
        instrumented = BUILDER.instrument_predictor_source(base)
        with self.assertRaisesRegex(BUILDER.CacheBuildError, "already"):
            BUILDER.instrument_predictor_source(instrumented)
        with self.assertRaisesRegex(BUILDER.CacheBuildError, "expected one anchor"):
            BUILDER.instrument_predictor_source(base.replace("graph = build_graph", "graph = make_graph"))

    def test_runtime_patch_uses_formatted_post_ilp_geff_anchor(self) -> None:
        patch = BUILDER.kernel_runtime_patch_source()
        formatted_anchor = 'save_graph(graph, output_dir / f"{name}.geff")'
        literal_anchor = 'save_graph(graph, output_dir / f"{{name}}.geff")'
        self.assertIn(formatted_anchor, patch)
        self.assertNotIn(literal_anchor, patch)
        self.assertIn('_v32_cache_source.count(_v32_cache_post_old) != 1', patch)
        compile(patch, "v32_runtime_cache_patch.py", "exec")

    def test_population_selection_is_exact_and_44b6_first(self) -> None:
        manifest = json.loads(BUILDER.POPULATION_DEFAULT.read_text())
        held44 = BUILDER.select_identities(manifest, role="held_fit44b6")
        held6 = BUILDER.select_identities(manifest, role="held_fit6bba")
        cv57 = BUILDER.select_identities(manifest, role="cv57")
        f2_six = BUILDER.select_identities(manifest, role="f2_six")
        self.assertEqual((len(held44), len(held6), len(cv57)), (36, 69, 57))
        self.assertEqual(f2_six, BUILDER.F2_SIX_IDENTITIES)
        self.assertTrue(all(value.startswith("train:44b6_") for value in held44))
        self.assertTrue(all(value.startswith("train:6bba_") for value in held6))
        canary = BUILDER.select_identities(
            manifest, role="canary", canary_identity="train:44b6_cf8fed6b"
        )
        self.assertEqual(canary, ("train:44b6_cf8fed6b",))

    def test_cv57_source_never_reads_competition_test_namespace(self) -> None:
        manifest = json.loads(BUILDER.POPULATION_DEFAULT.read_text())
        source, freeze = BUILDER.build_instrumented_source(
            population_manifest=manifest,
            role="cv57",
        )
        self.assertEqual(len(freeze["identities"]), 57)
        self.assertEqual(freeze["role"], "cv57")
        self.assertIn("_v32_materialize_selected_train()\nTEST_DIR = SELECTED_TRAIN", source)
        self.assertNotIn('COMP_DIR / "test"', source)
        self.assertNotIn('/competitions/biohub-cell-tracking-during-development/test', source)
        self.assertNotIn('inventory_input("competition_test"', source)
        self.assertNotIn("COMPETITION_TEST_CANDIDATES", source)
        self.assertNotIn('"competition_test_root"', source)

    def test_canary_source_build_is_compile_only_label_free_and_m3_absent(self) -> None:
        manifest = json.loads(BUILDER.POPULATION_DEFAULT.read_text())
        source, freeze = BUILDER.build_instrumented_source(
            population_manifest=manifest,
            role="canary",
            canary_identity="train:44b6_cf8fed6b",
        )
        self.assertEqual(freeze["status"], "PASS_BUILD_ONLY")
        self.assertFalse(freeze["launch_authorized_by_this_manifest"])
        self.assertTrue(freeze["label_free"])
        self.assertFalse(freeze["m3_mounted"])
        self.assertEqual(
            freeze["expected_uninstrumented_runtime_predictor_sha256"],
            "64734adad551aec6ff7fa805ea24288fef3faab369cbc729f6bbe5fa52529dc1",
        )
        self.assertEqual(
            freeze["effective_v32_motion_relink"],
            {
                "tight_um": 6.0,
                "relaxed_um": 10.0,
                "velocity_weight": 0.5,
                "raw_distance_weight": 0.05,
                "learned_bonus": 1.0,
                "max_frame_nodes": 2600,
                "output_edge_max_um": 14.0,
            },
        )
        self.assertEqual(freeze["primary_config_sha256"], freeze["secondary_config_sha256"])
        self.assertEqual(
            freeze["runtime_split_sha256"],
            cache.expected_runtime_split_sha256(["train:44b6_cf8fed6b"]),
        )
        self.assertIn(BUILDER.INSTRUMENTATION_MARKER, source)
        self.assertIn(BUILDER.TERMINAL_INVENTORY_MARKER, source)
        self.assertIn("_v32_cache_uninstrumented_sha256", source)
        self.assertIn(BUILDER.EXPECTED_V32_RUNTIME_PREDICTOR_SHA256, source)
        self.assertIn('"numpy": ("numpy", "numpy==2.4.6")', source)
        self.assertIn('"packaging": ("packaging", "packaging")', source)
        self.assertIn('"scipy": ("scipy", "scipy")', source)
        self.assertIn('"numba": ("numba", "numba==0.65.1")', source)
        self.assertIn('"llvmlite": ("llvmlite", "llvmlite==0.47.0")', source)
        self.assertIn('"numba": "0.65.1"', source)
        self.assertIn('"llvmlite": "0.47.0"', source)
        self.assertIn(
            '{"polars", "zarr", "numpy", "numba", "llvmlite"}', source
        )
        self.assertIn("V32_DEPENDENCY_REEXEC_REQUESTED", source)
        self.assertIn("V32_DEPENDENCY_REEXEC_AUTHENTICATED", source)
        self.assertIn("os.execve(sys.executable, argv, environment)", source)
        self.assertNotIn("purge_imported_modules(missing)", source)
        self.assertIn("_V32_FULL_SUPPORT_INSTALL_PLAN", source)
        self.assertIn(BUILDER.EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256, source)
        self.assertIn('"--force-reinstall", *[str(path) for path in wheels]', source)
        self.assertIn("dependency drift remained after the exact full-support restart", source)
        self.assertEqual(freeze["full_support_wheel_count"], 62)
        self.assertEqual(
            freeze["full_support_install_plan_sha256"],
            BUILDER.EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256,
        )
        self.assertIn("def display(_value) -> None:", source)
        self.assertIn("V32_STANDALONE_DISPLAY_SINK", source)
        self.assertLess(
            source.index("def display(_value) -> None:"),
            source.index("display(pd.read_csv(SUBMISSION_PATH, nrows=8))"),
        )
        self.assertIn("def make_owner_writable_tree(root: Path)", source)
        self.assertIn("make_owner_writable_tree(REPO_DIR)", source)
        self.assertIn("path.chmod(mode | 0o200", source)
        self.assertIn("_v32_materialize_selected_train()\nTEST_DIR = SELECTED_TRAIN", source)
        self.assertNotIn('COMP_DIR / "test"', source)
        self.assertNotIn('/competitions/biohub-cell-tracking-during-development/test', source)
        self.assertLess(
            source.index("_v32_materialize_selected_train()\nTEST_DIR = SELECTED_TRAIN"),
            source.index("V32_HEAVY_WORK_BEGIN"),
        )
        # Regression for alpha-funnel Kaggle v1 (kernel version 343584465):
        # that ad-hoc package retained the hidden-test Step 0, which bound the
        # four competition-test stems before later switching inference to the
        # 57 selected-train stems.  The full model run then failed only at the
        # terminal population-consistency guard after roughly two hours.
        self.assertIn(
            '"schema": "celltrack-alpha.public-temporalunet-step0.v1"', source
        )
        self.assertIn('"selected_datasets": SELECTED_DATASETS', source)
        self.assertNotIn("public-temporalunet-hidden-step0.v1", source)
        self.assertNotIn(
            "dynamic hidden inventory drifted between step 0 and prediction", source
        )
        self.assertNotIn('inventory_input("competition_test"', source)
        self.assertNotIn("COMPETITION_TEST_CANDIDATES", source)
        self.assertNotIn('"competition_test_root"', source)
        self.assertIn("SUBMISSION_PATH.unlink()", source)
        self.assertIn("V32_CACHE_NO_SUBMISSION_CONFIRMED", source)
        self.assertIn("V32_FINAL_GRAPH_ROWS_PRESERVED", source)
        self.assertIn('WORKING_DIR / "V32_FINAL_GRAPH_ROWS.csv"', source)
        self.assertLess(
            source.index("V32_FINAL_GRAPH_ROWS_PRESERVED"),
            source.index("SUBMISSION_PATH.unlink()"),
        )
        self.assertIn('"completed_units": sum(_v32_terminal_frame_counts.values())', source)
        self.assertIn('"observed_frames_by_identity": _v32_terminal_frame_counts', source)
        self.assertIn('"execution_device": (', source)
        self.assertIn('array_metadata["shape"][0] = SHIP_PREFLIGHT_UNITS', source)
        self.assertIn('shutil.copytree(', source)
        self.assertNotIn('os.environ["BIOHUB_SLICE"]', source)
        self.assertNotIn('inventory_input("competition_train", COMPETITION_TRAIN)', source)
        self.assertIn('f"competition_train_selected:{_v32_selected_dataset}"', source)
        self.assertIn('elif SHIP_PREFLIGHT_MODE:', source)
        self.assertIn('"CPU_SHIP_PREFLIGHT"', source)
        self.assertIn(
            'os.environ["BIOHUB_V32_CPU_PREFLIGHT_SKIP_ILP"] = "1"',
            source,
        )
        self.assertIn("V32_CPU_PREFLIGHT_ILP_SKIPPED", source)
        self.assertIn("if cfg.use_ilp and not _v32_skip_ilp", source)
        self.assertIn("KAGGLE_PREFLIGHT_UNITS must be in [3, 100]", source)
        self.assertGreater(
            source.index("SUBMISSION_PATH.unlink()"),
            source.index(
                '_atomic_json(WORKING_DIR / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", _receipt)'
            ),
        )
        self.assertLess(
            source.index(BUILDER.TERMINAL_INVENTORY_MARKER),
            source.index(
                '_atomic_json(WORKING_DIR / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", _receipt)'
            ),
        )
        self.assertIn('"inventory_sha256": hashlib.sha256', source)
        self.assertIn('"scientific_use_open": False', source)
        self.assertNotIn("local_association_ranker", source)
        self.assertNotIn("evaluate=True", source)

    def test_full_support_install_plan_is_exact_complete_and_digest_bound(self) -> None:
        rows = BUILDER.full_support_install_plan()
        self.assertEqual(len(rows), BUILDER.EXPECTED_FULL_SUPPORT_WHEEL_COUNT)
        self.assertEqual(
            BUILDER.canonical_sha256(rows),
            BUILDER.EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256,
        )
        names = [row["filename"] for row in rows]
        self.assertEqual(names, sorted(names))
        for required in (
            "numpy-2.4.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "numba-0.65.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "llvmlite-0.47.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl",
        ):
            self.assertIn(required, names)

    def test_generated_bootstrap_refreshes_present_but_version_drifted_jit(self) -> None:
        manifest = json.loads(BUILDER.POPULATION_DEFAULT.read_text())
        source, _ = BUILDER.build_instrumented_source(
            population_manifest=manifest,
            role="canary",
            canary_identity="train:44b6_cf8fed6b",
        )
        tree = ast.parse(source)
        refresh_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "packages_requiring_refresh"
        )
        namespace = {
            "module_missing": lambda _name: True,
            "polars_runtime_ready": lambda: True,
        }
        executable = ast.fix_missing_locations(
            ast.Module(body=[refresh_function], type_ignores=[])
        )
        exec(compile(executable, "<bootstrap-refresh-test>", "exec"), namespace)

        exact_versions = {
            "numpy": "2.4.6",
            "numba": "0.65.1",
            "llvmlite": "0.47.0",
        }
        with mock.patch(
            "importlib.metadata.version", side_effect=exact_versions.__getitem__
        ):
            self.assertEqual(namespace["packages_requiring_refresh"](), [])

        drifted_versions = dict(exact_versions, numba="0.61.2", llvmlite="0.44.0")
        with mock.patch(
            "importlib.metadata.version", side_effect=drifted_versions.__getitem__
        ):
            self.assertEqual(
                namespace["packages_requiring_refresh"](), ["numba", "llvmlite"]
            )

    def test_builder_rejects_nonheld_canary_and_wrong_counts(self) -> None:
        manifest = json.loads(BUILDER.POPULATION_DEFAULT.read_text())
        with self.assertRaises(BUILDER.CacheBuildError):
            BUILDER.select_identities(
                manifest, role="canary", canary_identity="train:44b6_not_present"
            )
        other_held = next(
            identity
            for identity in manifest["populations"]["held_fit105"]
            if identity != BUILDER.CANARY_IDENTITY
        )
        with self.assertRaisesRegex(BUILDER.CacheBuildError, "frozen"):
            BUILDER.select_identities(
                manifest, role="canary", canary_identity=other_held
            )
        bad = copy.deepcopy(manifest)
        removed = next(
            identity
            for identity in bad["populations"]["held_fit105"]
            if identity.startswith("train:6bba_")
        )
        bad["populations"]["held_fit105"].remove(removed)
        with self.assertRaisesRegex(BUILDER.CacheBuildError, "count"):
            BUILDER.select_identities(bad, role="held_fit6bba")

        tampered = copy.deepcopy(manifest)
        tampered["purpose"] = "tampered"
        with self.assertRaisesRegex(BUILDER.CacheBuildError, "differs from the frozen"):
            BUILDER.build_instrumented_source(
                population_manifest=tampered,
                role="canary",
                canary_identity="train:44b6_cf8fed6b",
            )


if __name__ == "__main__":
    unittest.main()

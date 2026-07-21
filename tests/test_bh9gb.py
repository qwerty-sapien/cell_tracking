from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bh9gb  # noqa: E402


def make_zip(path: Path, *, label_bytes: int = 1, include_b: bool = True) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as handle:
        for dataset, payload in (("a", b"a" * 7), ("b", b"b" * 11)):
            if dataset == "b" and not include_b:
                continue
            handle.writestr(f"train/{dataset}.zarr/zarr.json", b"{}")
            handle.writestr(f"train/{dataset}.zarr/0/zarr.json", b"{}")
            handle.writestr(f"train/{dataset}.zarr/0/c/0/0/0/0", payload)
        handle.writestr("train/a.geff/zarr.json", b"g" * label_bytes)
        handle.writestr("train/a.geff/nodes/ids/c/0", b"secret" * label_bytes)


def make_directory(root: Path, *, include_b_image: bool = True, include_b_label: bool = False) -> None:
    for dataset in ("a", "b"):
        if dataset == "b" and not include_b_image:
            continue
        image = root / "train" / f"{dataset}.zarr" / "0"
        image.mkdir(parents=True, exist_ok=True)
        (image / "zarr.json").write_text("{}", encoding="ascii")
        if dataset == "a" or include_b_label:
            label = root / "train" / f"{dataset}.geff"
            label.mkdir(parents=True, exist_ok=True)
            (label / "zarr.json").write_text("{}", encoding="ascii")


def directory_inputs(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    competition = root / "competition"
    make_directory(competition)
    records = [
        {
            "split": "train",
            "dataset": dataset,
            "video_id": f"train:{dataset}",
            "exact_identity_sha256": hashlib.sha256(dataset.encode("ascii")).hexdigest(),
            "compact_metadata": {"image_shape": [100, 2, 3, 4]},
        }
        for dataset in ("a", "b")
    ]
    registry = root / "reg.json"
    registry.write_text(
        json.dumps(
            {
                "artifact": "dataset_registry_v1",
                "counts": {"train_videos": 2, "videos": 2},
                "source": {"zip_central_directory_sha256": "c" * 64, "zip_entry_count": 4},
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    source_hashes = {relative: bh9gb.sha256_file(ROOT / relative) for relative in bh9gb.SCORER_FILES}
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "dataset_registry_sha256": bh9gb.sha256_file(registry),
                "source_hashes": source_hashes,
            }
        ),
        encoding="utf-8",
    )
    inventory = {
        "compressed_bytes": {"train:a": 5_000_000_000, "train:b": 5_000_000_000},
        "central_sha256": "c" * 64,
        "entry_count": 4,
    }
    selection = bh9gb.select_bytes(("train:a", "train:b"), inventory["compressed_bytes"])
    manifest = bh9gb.build_frozen_manifest(
        inventory,
        selection,
        registry_sha256=bh9gb.sha256_file(registry),
        split_sha256=bh9gb.sha256_file(split),
    )
    frozen = root / "bh9m.json"
    frozen.write_bytes(bh9gb.canonical_bytes(manifest, newline=True))
    return competition, registry, split, manifest


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_identities() -> tuple[str, ...]:
    return tuple(f"train:v{index:02d}" for index in range(bh9gb.FROZEN_VIDEO_COUNT))


def a0_validation() -> dict[str, bool]:
    return {name: True for name in bh9gb.A0_EVIDENCE_GATES}


def evidence_runner() -> dict[str, object]:
    weight = "c" * 64
    return {
        "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
        "a0_validation": a0_validation(),
        "a0_audit": {
            "scope": "run",
            "video_count": bh9gb.FROZEN_VIDEO_COUNT,
            "frame_count": bh9gb.FROZEN_FRAME_COUNT,
            "pair_count": bh9gb.FROZEN_ADJACENT_PAIR_COUNT,
            "workers": {"worker_count": 2, "scorer_instances": 2},
        },
        "per_video": [{} for _ in range(bh9gb.FROZEN_VIDEO_COUNT)],
        "worker_audits": [{}, {}],
        "checkpoint_weight_identity": {
            "preflight_weight_sha256": weight,
            "frozen_cache_weight_sha256": weight,
            "a0_audit_weight_sha256": weight,
            "matched": True,
        },
    }


def video_a0_audit(*, cache_reused: bool = False) -> dict[str, object]:
    association = {
        "source_count": 4,
        "target_count": 4,
        "candidate_count": 4,
        "component_count": 4,
        "attention_component_count": 3,
        "fallback_component_count": 1,
        "node_budget_fallback_component_count": 1,
        "attention_budget_fallback_component_count": 0,
        "unsupported_fallback_component_count": 0,
        "attention_candidate_count": 3,
        "fallback_candidate_count": 1,
        "attention_component_fraction": 0.75,
        "fallback_component_fraction": 0.25,
        "attention_candidate_fraction": 0.75,
        "fallback_candidate_fraction": 0.25,
        "max_component_source_nodes": 1,
        "max_component_target_nodes": 1,
        "max_component_nodes": 2,
        "max_component_candidates": 1,
        "max_component_attention_cells": 1,
        "batch_count": 1,
        "max_batch_components": 3,
        "max_padded_source_nodes": 1,
        "max_padded_target_nodes": 1,
        "peak_padding_nodes": 0,
        "peak_attention_cells": 3,
        "neighbor_cap": 8,
        "max_component_nodes_budget": 64,
        "attention_cell_budget": 65_536,
        "pair_chunk_size": 256,
        "peak_pair_chunk": 4,
    }
    checkpoint = {
        "weight_sha256": "c" * 64,
        "architecture_sha256": "d" * 64,
        "frozen_subset_sha256": "e" * 64,
        "factory": "A0CheckpointScorer.from_loaded_model",
        "full_frame_dense_matrix": False,
    }
    return {
        "schema": bh9gb.run_bh.A0_AUDIT_SCHEMA,
        "scope": "video",
        "video_count": 1,
        "frame_count": bh9gb.FROZEN_FRAMES_PER_VIDEO,
        "pair_count": bh9gb.FROZEN_FRAMES_PER_VIDEO - 1,
        "checkpoint": checkpoint,
        "cache": {
            "record_types": ["FrameDet", "PairContext"],
            "detection_count": bh9gb.FROZEN_FRAMES_PER_VIDEO,
            "context_count": bh9gb.FROZEN_FRAMES_PER_VIDEO - 1,
            "complete_adjacent_coverage": True,
            "reused": cache_reused,
        },
        "gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": 10.0,
            "neighbor_cap": 8,
            "scale_zyx": [1.625, 0.40625, 0.40625],
            "candidate_count": 4,
            "peak_radius_neighbors": 4,
            "peak_retained_neighbors": 4,
            "full_frame_dense_matrix": False,
        },
        "probabilities": {
            "scope": "original-candidate-coo-only",
            "candidate_count": 4,
            "unique_pair_count": 4,
            "finite_unit_interval": True,
        },
        "association": association,
        "graph": {
            "variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
            "learned_probability_count": 4,
            "full_frame_dense_matrix": False,
        },
    }


def runner_fixture(root: Path, identities: tuple[str, ...]) -> dict[str, object]:
    workers: list[dict[str, object]] = [
        {"ok": True, "rank": rank, "device": bh9gb.DEVICES[rank], "results": []}
        for rank in range(2)
    ]
    for index, identity in enumerate(identities):
        rank = index % 2
        slug = identity.replace(":", "__")
        fragment = root / "fragments" / f"{slug}.csv"
        feature = root / "features" / f"{slug}.json"
        status = root / "status" / f"{slug}.json"
        transaction = root / "transactions" / f"{slug}.json"
        cache = root / "cache" / f"{slug}.bhc"
        cache_meta = root / "cache" / f"{slug}.json"
        for path in (fragment, feature, status, transaction, cache, cache_meta):
            path.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_text("fragment\n", encoding="utf-8")
        cache.write_bytes(f"cache:{identity}".encode("ascii"))
        cache_core = {
            "schema": bh9gb.run_bh.CACHE_META_SCHEMA,
            "data_exact_identity_sha256": hashlib.sha256(identity.encode("ascii")).hexdigest(),
            "model": {
                "config_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "weight_sha256": "c" * 64,
                "pair_batch_size": 1,
                "view_batch_size": 1,
            },
            "inference": {"frame_count": bh9gb.FROZEN_FRAMES_PER_VIDEO},
        }
        cache_contract = {
            **cache_core,
            "sha256": bh9gb.run_bh.canonical_sha256(cache_core),
        }
        run_core = {
            "schema": bh9gb.run_bh.RUNNER_SCHEMA,
            "inference_sha256": cache_contract["sha256"],
            "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
            "graph_config": bh9gb.run_bh.GRAPH_VARIANTS[bh9gb.run_bh.A0_GRAPH_VARIANT],
        }
        run_contract = {**run_core, "sha256": bh9gb.run_bh.canonical_sha256(run_core)}
        audit = video_a0_audit()
        feature.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
                    "run_contract": run_contract,
                    "a0_audit": audit,
                }
            ),
            encoding="utf-8",
        )
        cache_meta.write_text(
            json.dumps(
                {
                    "schema": bh9gb.run_bh.CACHE_META_SCHEMA,
                    "contract": cache_contract,
                    "cache_path": str(cache),
                    "cache_sha256": file_sha(cache),
                }
            ),
            encoding="utf-8",
        )
        transaction.write_text(
            json.dumps(
                {
                    "schema": bh9gb.run_bh.TRANSACTION_SCHEMA,
                    "identity": identity,
                    "run_contract": run_contract,
                    "artifacts": {
                        "fragment": str(fragment),
                        "feature": str(feature),
                        "status": str(status),
                    },
                }
            ),
            encoding="utf-8",
        )
        status.write_text(
            json.dumps(
                {
                    "ok": True,
                    "identity": identity,
                    "run_contract": run_contract,
                    "fragment_sha256": file_sha(fragment),
                    "feature_sha256": file_sha(feature),
                    "a0_audit": audit,
                }
            ),
            encoding="utf-8",
        )
        result = {
            "identity": identity,
            "seconds": 1.0,
            "nodes": 2,
            "edges": 1,
            "rows": 3,
            "resumed": False,
            "image_support_projection": {
                "boundary": "post-smoothing/public-output",
                "scale_zyx": [1.625, 0.40625, 0.40625],
                "limit_um": 3.2,
                "epsilon_um": 1.0e-12,
                "projected_nodes": rank,
                "max_correction_um": 0.40625 * rank,
                "node_count": 2,
                "edge_count": 1,
                "node_count_preserved": True,
                "topology_preserved": True,
            },
            "fragment": str(fragment),
            "feature_summary": str(feature),
            "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
            "cache_reused": False,
            "cache_contract_sha256": cache_contract["sha256"],
            "run_contract_sha256": run_contract["sha256"],
            "a0_audit": audit,
        }
        workers[rank]["results"].append(result)  # type: ignore[union-attr]
    for worker in workers:
        audit = bh9gb.run_bh.aggregate_a0_audits(
            [item["a0_audit"] for item in worker["results"]],  # type: ignore[index]
            scope="worker",
        )
        audit["worker"] = {
            "rank": worker["rank"],
            "device": worker["device"],
            "scorer_instances": 1,
        }
        worker["a0_audit"] = audit
    submission = root / bh9gb.SUBMISSION
    submission.write_text("submission\n", encoding="utf-8")
    report = {
        "ok": True,
        "mode": "cv",
        "elapsed_seconds": 2.0,
        "devices": list(bh9gb.DEVICES),
        "model_loads": 2,
        "whole_video_shards": [
            [item["identity"] for item in worker["results"]]  # type: ignore[index]
            for worker in workers
        ],
        "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
        "requested_identities": sorted(identities),
        "view_execution": {
            "mode": "exact-sequential",
            "pair_batch_size": 1,
            "view_batch_size": 1,
            "exact_code_path_control": True,
        },
        "preflight": {
            "ok": True,
            "cuda_count": 2,
            "checkpoint": {"hashes": {"weight_sha256": "c" * 64}},
        },
        "workers": workers,
        "submission": {
            "datasets": len(identities),
            "sha256": file_sha(submission),
        },
    }
    run_audit = bh9gb.run_bh.aggregate_a0_audits(
        [item["a0_audit"] for worker in workers for item in worker["results"]],  # type: ignore[index]
        scope="run",
    )
    run_audit["workers"] = {"worker_count": 2, "scorer_instances": 2}
    report["a0_audit"] = run_audit
    (root / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class SelectionTests(unittest.TestCase):
    def test_frozen_seed_order_and_first_crossing(self) -> None:
        identities = ("train:a", "train:b", "train:c", "train:d")
        self.assertEqual(
            bh9gb.seeded_order(identities),
            ("train:c", "train:d", "train:a", "train:b"),
        )
        selected = bh9gb.select_bytes(
            identities,
            {identity: 4 for identity in identities},
            target=9,
        )
        self.assertEqual(selected["selected_video_ids"], ["train:c", "train:d", "train:a"])
        self.assertEqual(selected["selected_compressed_bytes"], 12)
        self.assertEqual(selected["selected"][-2]["cumulative_bytes"], 8)
        self.assertEqual(selected["count_fraction"], 0.75)

    def test_geff_bytes_never_change_selection_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.zip"
            second = root / "second.zip"
            make_zip(first, label_bytes=1)
            make_zip(second, label_bytes=1000)
            identities = ("train:a", "train:b")
            opened: list[str] = []

            def forbidden(*args: object, **kwargs: object) -> object:
                opened.append("opened")
                raise AssertionError("selection must not open ZIP members")

            with mock.patch.object(zipfile.ZipFile, "open", new=forbidden):
                inv1 = bh9gb.image_inventory(first, identities)
                inv2 = bh9gb.image_inventory(second, identities)
            self.assertEqual(inv1["compressed_bytes"], inv2["compressed_bytes"])
            self.assertEqual(inv1["train_image_bytes"], inv2["train_image_bytes"])
            self.assertNotEqual(inv1["central_sha256"], inv2["central_sha256"])
            self.assertEqual(opened, [])

    def test_inventory_fails_for_missing_extra_or_duplicate_image_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            missing = root / "miss.zip"
            make_zip(missing, include_b=False)
            with self.assertRaisesRegex(bh9gb.Bh9Error, "missing"):
                bh9gb.image_inventory(missing, ("train:a", "train:b"))
            complete = root / "full.zip"
            make_zip(complete)
            with self.assertRaisesRegex(bh9gb.Bh9Error, "extra"):
                bh9gb.image_inventory(complete, ("train:a",))

            duplicate = root / "dupe.zip"
            make_zip(duplicate)
            with zipfile.ZipFile(duplicate, mode="a") as handle:
                handle.writestr("train/a.zarr/0/zarr.json", b"{}")
            with self.assertRaisesRegex(bh9gb.Bh9Error, "duplicate image ZIP member"):
                bh9gb.image_inventory(duplicate, ("train:a", "train:b"))

    def test_labels_are_validated_only_for_frozen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "data.zip"
            make_zip(archive)
            report = bh9gb.validate_labels(archive, ("train:a",))
            self.assertTrue(report["validated_after_selection"])
            self.assertEqual(report["selected_geff_root_count"], 1)
            with self.assertRaisesRegex(bh9gb.Bh9Error, "exactly once"):
                bh9gb.validate_labels(archive, ("train:b",))

    def test_selection_rejects_partial_or_insufficient_inventory(self) -> None:
        with self.assertRaisesRegex(bh9gb.Bh9Error, "exactly cover"):
            bh9gb.select_bytes(("train:a", "train:b"), {"train:a": 10}, target=5)
        with self.assertRaisesRegex(bh9gb.Bh9Error, "below"):
            bh9gb.select_bytes(
                ("train:a", "train:b"),
                {"train:a": 2, "train:b": 2},
                target=5,
            )

    def test_directory_mode_freezes_bytes_before_selected_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(bh9gb, "TRAIN_COUNT", 2):
            root = Path(raw)
            competition, registry, split, manifest = directory_inputs(root)
            frozen = root / "bh9m.json"
            inventory, hashes, identities = bh9gb.attest_inputs(
                competition, registry, split, frozen
            )
            self.assertEqual(identities, ("train:a", "train:b"))
            self.assertFalse(inventory["labels_inspected"])
            self.assertEqual(inventory["central_sha256"], "c" * 64)
            self.assertEqual(
                inventory["frozen_selection_sha256"],
                manifest["identity"]["selection"]["selection_sha256"],
            )
            self.assertIn("frozen_manifest_sha256", hashes)
            selected = bh9gb.select_bytes(identities, inventory["compressed_bytes"])
            with self.assertRaisesRegex(bh9gb.Bh9Error, "image/GEFF"):
                bh9gb.validate_selected_paths(competition, selected["selected_video_ids"])
            make_directory(competition, include_b_label=True)
            validated = bh9gb.validate_selected_paths(
                competition, selected["selected_video_ids"]
            )
            self.assertTrue(validated["directory_structured_paths"])

    def test_directory_mode_rejects_extra_identity_and_mutated_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(bh9gb, "TRAIN_COUNT", 2):
            root = Path(raw)
            competition, registry, split, manifest = directory_inputs(root)
            extra = competition / "train" / "extra.zarr"
            extra.mkdir()
            with self.assertRaisesRegex(bh9gb.Bh9Error, "extra"):
                bh9gb.attest_inputs(competition, registry, split, root / "bh9m.json")
            extra.rmdir()
            manifest["identity"]["selection"]["selected_video_ids"].reverse()
            manifest["identity_sha256"] = bh9gb.canonical_sha(manifest["identity"])
            (root / "bh9m.json").write_bytes(bh9gb.canonical_bytes(manifest, newline=True))
            with self.assertRaisesRegex(bh9gb.Bh9Error, "selection was altered"):
                bh9gb.attest_inputs(competition, registry, split, root / "bh9m.json")

    @unittest.skipUnless(bh9gb.ARCHIVE.is_file(), "local competition archive is absent")
    def test_real_archive_selection_is_frozen(self) -> None:
        inventory, hashes, identities = bh9gb.attest_inputs(
            bh9gb.ARCHIVE,
            bh9gb.REGISTRY,
            bh9gb.SPLIT,
        )
        selected = bh9gb.select_bytes(identities, inventory["compressed_bytes"])
        self.assertEqual(len(identities), 199)
        self.assertEqual(selected["selected_count"], 22)
        self.assertEqual(selected["selected_compressed_bytes"], 9_337_408_312)
        self.assertEqual(
            selected["selection_sha256"],
            "43687468e9f51d9d8e5795eb91bc081a645a3384fd599ebb6f56cb7129ba67f1",
        )
        self.assertEqual(
            hashes["archive_central_sha256"],
            "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77",
        )


class RunnerContractTests(unittest.TestCase):
    def test_exact_dual_gpu_transactional_a0_report_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identities = frozen_identities()
            report = runner_fixture(root, identities)
            validated = bh9gb.validate_runner(report, identities, root)
            self.assertEqual(validated["graph_variant"], bh9gb.run_bh.A0_GRAPH_VARIANT)
            self.assertEqual(validated["a0_audit"]["video_count"], 22)
            self.assertEqual(validated["a0_audit"]["pair_count"], 2_178)
            self.assertEqual(validated["a0_audit"]["workers"]["scorer_instances"], 2)
            self.assertTrue(all(validated["a0_validation"].values()))
            self.assertTrue(validated["no_dense_nxn"])
            self.assertTrue(validated["transactional_per_video"])
            self.assertEqual(len(validated["per_video"]), 22)
            self.assertEqual(
                [
                    row["image_support_projection"]["projected_nodes"]
                    for row in validated["per_video"][:2]
                ],
                [0, 1],
            )

    def test_runner_fails_for_wrong_gpu_coverage_or_a0_audit_contract(self) -> None:
        identities = frozen_identities()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            report["devices"] = ["cuda:0"]
            with self.assertRaisesRegex(bh9gb.Bh9Error, "dual-GPU"):
                bh9gb.validate_runner(report, identities, root)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            report["whole_video_shards"] = [["train:a"], ["train:a"]]
            with self.assertRaisesRegex(bh9gb.Bh9Error, "dual-GPU"):
                bh9gb.validate_runner(report, identities, root)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            status = root / "status/train__v00.json"
            payload = json.loads(status.read_text(encoding="utf-8"))
            payload["run_contract"]["graph_config"]["learned_edge_probabilities"] = True
            status.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(bh9gb.Bh9Error, "non-A0"):
                bh9gb.validate_runner(report, identities, root)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            projection = report["workers"][0]["results"][0]["image_support_projection"]
            projection["max_correction_um"] = 3.200000001
            with self.assertRaisesRegex(bh9gb.Bh9Error, "image-support projection"):
                bh9gb.validate_runner(report, identities, root)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            report["a0_audit"]["workers"]["scorer_instances"] = 3
            with self.assertRaisesRegex(bh9gb.Bh9Error, "exactly two scorer"):
                bh9gb.validate_runner(report, identities, root)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = runner_fixture(root, identities)
            report["workers"][0]["results"][0]["a0_audit"]["association"][
                "fallback_candidate_count"
            ] = 2
            with self.assertRaisesRegex(bh9gb.Bh9Error, "does not balance"):
                bh9gb.validate_runner(report, identities, root)

    def test_runner_cli_is_locked_to_50_minute_dual_gpu_cv(self) -> None:
        args = argparse.Namespace(
            archive=Path("a.zip"),
            registry=Path("r.json"),
            support=Path("support"),
            out_dir=Path("bh9out"),
        )
        parsed = bh9gb.runner_args(args, ("train:a",), bh9gb.DEADLINE_SECONDS)
        self.assertEqual(parsed.mode, "cv")
        self.assertEqual(parsed.split, "train")
        self.assertEqual(parsed.devices, "cuda:0,cuda:1")
        self.assertEqual(parsed.deadline_min, 50.0)
        self.assertEqual(parsed.pair_batch_size, 1)
        self.assertEqual(parsed.view_batch_size, 1)
        self.assertEqual(parsed.graph_variant, bh9gb.run_bh.A0_GRAPH_VARIANT)


class EvidenceTests(unittest.TestCase):
    def test_score_gate_and_json_nonfinite_encoding(self) -> None:
        args = argparse.Namespace(archive=Path("archive.zip"))
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "archive.zip"
            archive.write_bytes(b"zip")
            args.archive = archive
            evidence = bh9gb.build_evidence(
                args,
                {"selected_count": 1},
                {"archive_central_sha256": "a" * 64},
                {"validated_after_selection": True},
                evidence_runner(),
                {"summary": {"score": 0.899}, "samples": [{"division_jaccard": float("nan")}]},
                10.0,
            )
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["gates"]["score_strictly_above_0_90"])
        self.assertEqual(evidence["runtime_attestation"]["level"], 6)
        self.assertTrue(evidence["runtime_attestation"]["passed"])
        self.assertIsNone(evidence["official_score"]["samples"][0]["division_jaccard"])
        bh9gb.canonical_bytes(evidence)

    def test_score_gate_is_strict_and_level_6_runtime_gate_is_conjunctive(self) -> None:
        args = argparse.Namespace(archive=Path("archive.zip"))
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "archive.zip"
            archive.write_bytes(b"zip")
            args.archive = archive
            common = (
                args,
                {"selected_count": 22, "selected_compressed_bytes": 9_337_408_312},
                {"archive_central_sha256": "a" * 64},
                {"validated_after_selection": True},
                evidence_runner(),
            )
            equal = bh9gb.build_evidence(
                *common, {"summary": {"score": 0.90}, "samples": []}, 3000.0
            )
            above = bh9gb.build_evidence(
                *common, {"summary": {"score": 0.900001}, "samples": []}, 3000.0
            )
            late = bh9gb.build_evidence(
                *common, {"summary": {"score": 0.900001}, "samples": []}, 3000.001
            )
        self.assertFalse(equal["passed"])
        self.assertFalse(equal["gates"]["score_strictly_above_0_90"])
        self.assertTrue(equal["gates"]["deadline_met"])
        self.assertTrue(above["passed"])
        self.assertTrue(above["runtime_attestation"]["passed"])
        self.assertFalse(late["passed"])
        self.assertFalse(late["runtime_attestation"]["passed"])

    def test_success_evidence_fails_closed_without_complete_a0_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "archive.zip"
            archive.write_bytes(b"zip")
            evidence = bh9gb.build_evidence(
                argparse.Namespace(archive=archive),
                {"selected_count": 22},
                {"archive_central_sha256": "a" * 64},
                {"validated_after_selection": True},
                {"per_video": []},
                {"summary": {"score": 0.900001}, "samples": []},
                3000.0,
            )
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["gates"]["graph_variant_a0"])

    def test_hard_deadline_interrupts_and_created_names_are_short(self) -> None:
        started = time.monotonic()
        with self.assertRaises(bh9gb.Bh9Deadline):
            with bh9gb.HardLimit(0.05):
                time.sleep(2)
        self.assertLess(time.monotonic() - started, 1.0)
        for path in (
            Path(bh9gb.__file__),
            ROOT / "tests/test_bh9gb.py",
            bh9gb.OUT_DIR,
            bh9gb.EVIDENCE,
            bh9gb.FAILURE,
        ):
            self.assertLess(len(path.name), 15)

    def test_atomic_evidence_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bh9.json"
            bh9gb.run_bh.atomic_create_json(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                bh9gb.run_bh.atomic_create_json(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "first"})

    def test_directory_scoring_uses_read_only_localstore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fragments = root / "frags"
            fragments.mkdir()
            (fragments / "train__a.csv").write_text("x", encoding="ascii")
            observed: dict[str, object] = {}

            class LocalStore:
                def __init__(self, path: Path, *, read_only: bool) -> None:
                    observed.update(path=path, read_only=read_only)

            class ZipStore:
                pass

            scoreall = types.ModuleType("scoreall")
            scoreall.validate_fragment_set = lambda paths, selected: object()
            scoreall.score_fragment_input = lambda store, manifest: (
                [
                    {
                        "identity": "train:a",
                        "full_video": True,
                        "cropped": False,
                        "ground_truth_nodes": 1,
                        "adj_edge_jaccard": 1.0,
                    }
                ],
                {},
                {},
            )
            scoreall.official_summary = lambda rows: {"n": 1, "n_adj": 1, "score": 1.0}
            zarr = types.ModuleType("zarr")
            storage = types.ModuleType("zarr.storage")
            storage.LocalStore = LocalStore
            storage.ZipStore = ZipStore
            with mock.patch.dict(
                sys.modules,
                {"scoreall": scoreall, "zarr": zarr, "zarr.storage": storage},
            ):
                result = bh9gb.score_selected(root, fragments, ("train:a",))
            self.assertTrue(observed["read_only"])
            self.assertEqual(result["zarr_store"], "LocalStore(read_only=True)")


if __name__ == "__main__":
    unittest.main()

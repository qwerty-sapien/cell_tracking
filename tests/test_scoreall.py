from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import sys
import tempfile
import unittest
import weakref
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scoreall  # noqa: E402


def node_row(identity: str, node_id: int = 1, t: int = 0) -> dict[str, object]:
    return {
        "dataset": identity,
        "row_type": "node",
        "node_id": node_id,
        "t": t,
        "z": 0,
        "y": node_id,
        "x": 0,
        "source_id": -1,
        "target_id": -1,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scoreall.SUBMISSION_COLUMNS)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            writer.writerow({"id": row_id, **row})


def graph(dataset: str) -> scoreall.SubmissionGraph:
    return scoreall.SubmissionGraph(
        dataset,
        {1: scoreall.SubmissionNode(1, 0, 0, 0, 0)},
        [],
    )


def synthetic_split() -> dict[str, object]:
    return {
        "audit": {"video_count": 1, "video_ids": ["train:audit"]},
        "folds": {
            "0": {"held_out_video_count": 1, "held_out_video_ids": ["train:a"]},
            "1": {"held_out_video_count": 1, "held_out_video_ids": ["train:b"]},
            "2": {"held_out_video_count": 1, "held_out_video_ids": ["train:c"]},
        },
        "counts": {
            "audit_videos": 1,
            "remainder_videos": 3,
            "excluded_nontrain_or_mixed_groups": 1,
        },
    }


class ScoreAllTests(unittest.TestCase):
    def test_frozen_split_selection_counts(self) -> None:
        split = json.loads(
            (ROOT / "reports/g02/grouped_3fold_audit20_v1.json").read_text(encoding="utf-8")
        )
        core = set(split["audit"]["video_ids"])
        for fold in split["folds"].values():
            core.update(fold["held_out_video_ids"])
        archive = core | {f"train:excluded{i}" for i in range(4)}

        self.assertEqual(
            len(scoreall.select_video_ids(split, "all", archive_train_video_ids=archive).selected_video_ids),
            199,
        )
        self.assertEqual(
            len(scoreall.select_video_ids(split, "audit", archive_train_video_ids=archive).selected_video_ids),
            20,
        )
        self.assertEqual(
            [
                len(
                    scoreall.select_video_ids(
                        split, f"fold{index}", archive_train_video_ids=archive
                    ).selected_video_ids
                )
                for index in range(3)
            ],
            [58, 59, 58],
        )

    def test_duplicate_and_missing_fragments_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            one = root / "one" / "train__a.csv"
            duplicate = root / "two" / "train__a.csv"
            write_csv(one, [node_row("train:a")])
            write_csv(duplicate, [node_row("train:a")])
            with self.assertRaises(scoreall.FragmentSetError) as caught:
                scoreall.validate_fragment_set(
                    [one, duplicate], ["train:a", "train:b"]
                )
        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["missing_relevant_fragments"], ["train__b.csv"])
        self.assertIn("train:a", diagnostics["duplicate_relevant_fragments"])

    def test_test_fragments_are_ignored_and_cannot_cover_train(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train__a.csv"
            test = root / "test__a.csv"
            write_csv(train, [node_row("train:a")])
            write_csv(test, [node_row("test:a")])
            manifest = scoreall.validate_fragment_set([test, train], ["train:a"])
            self.assertEqual([item[0] for item in manifest.ordered], ["train:a"])
            self.assertEqual(manifest.diagnostics["ignored_test_fragments"], [str(test)])
            with self.assertRaises(scoreall.FragmentSetError):
                scoreall.validate_fragment_set([test], ["train:a"])

    def test_fragment_directory_and_direct_glob_resolve_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train__a.csv"
            test = root / "test__a.csv"
            write_csv(train, [node_row("train:a")])
            write_csv(test, [node_row("test:a")])
            directory_paths = scoreall.resolve_fragment_paths(str(root))
            glob_paths = scoreall.resolve_fragment_paths(str(root / "train__*.csv"))
        self.assertEqual(directory_paths, [test, train])
        self.assertEqual(glob_paths, [train])

    def test_fragment_graph_normalizes_prefix_and_rejects_test(self) -> None:
        path = Path("train__a.csv")
        with mock.patch.object(scoreall, "read_submission", return_value={"train:a": graph("train:a")}):
            loaded = scoreall._fragment_graph(path, "train:a")
        self.assertEqual(len(loaded.nodes), 1)
        with mock.patch.object(scoreall, "read_submission", return_value={"test:a": graph("test:a")}):
            with self.assertRaisesRegex(scoreall.ScoreAllError, "test graph"):
                scoreall._fragment_graph(path, "train:a")

    def test_combined_fragment_hash_is_deterministic_and_content_sensitive(self) -> None:
        a = hashlib.sha256(b"a").hexdigest()
        b = hashlib.sha256(b"b").hexdigest()
        first = scoreall.combined_fragment_sha256([("train:a", a), ("train:b", b)])
        second = scoreall.combined_fragment_sha256([("train:b", b), ("train:a", a)])
        changed = scoreall.combined_fragment_sha256([("train:a", b), ("train:b", b)])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_subset_and_truncation_labels(self) -> None:
        split = synthetic_split()
        archive = ["train:a", "train:b", "train:c", "train:audit", "train:excluded"]
        full = scoreall.select_video_ids(split, "fold0", archive_train_video_ids=archive)
        smoke = scoreall.select_video_ids(
            split, "all", archive_train_video_ids=archive, max_datasets=2
        )
        self.assertEqual(full.as_dict()["run_label"], "subset_diagnostic")
        self.assertFalse(full.as_dict()["non_acceptance"])
        self.assertTrue(smoke.truncated)
        self.assertTrue(smoke.as_dict()["non_acceptance"])
        self.assertFalse(smoke.as_dict()["acceptance_eligible"])
        self.assertEqual(smoke.as_dict()["run_label"], "truncated_non_acceptance_smoke_test")

    def test_fragment_scoring_releases_each_graph_before_reading_next(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ordered = []
            for identity in ("train:a", "train:b", "train:c"):
                path = root / f"train__{identity.split(':')[1]}.csv"
                path.write_text(identity, encoding="utf-8")
                ordered.append((identity, path))
            manifest = scoreall.FragmentManifest(tuple(ordered), {})
            previous: list[weakref.ReferenceType[scoreall.SubmissionGraph]] = []

            def reader(path: Path) -> dict[str, scoreall.SubmissionGraph]:
                gc.collect()
                if previous:
                    self.assertIsNone(previous[-1]())
                identity = path.read_text(encoding="utf-8")
                value = graph(identity)
                previous.append(weakref.ref(value))
                return {identity: value}

            def fake_score(store: object, identity: str, value: scoreall.SubmissionGraph, **kwargs: object) -> dict[str, object]:
                return {"identity": identity, "edge_tp": 1.0}

            with mock.patch.object(scoreall, "read_submission", new=reader), mock.patch.object(
                scoreall, "score_graph", new=fake_score
            ):
                rows, provenance, _ = scoreall.score_fragment_input(object(), manifest)
        self.assertEqual([row["identity"] for row in rows], ["train:a", "train:b", "train:c"])
        self.assertEqual(provenance["file_count"], 3)

    def test_aggregate_streams_sequentially_skips_test_and_normalizes_prefix(self) -> None:
        rows = [node_row("test:x"), node_row("train:a"), node_row("train:b")]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "all.csv"
            write_csv(path, rows)
            previous: list[weakref.ReferenceType[scoreall.SubmissionGraph]] = []

            def fake_score(store: object, identity: str, value: scoreall.SubmissionGraph, **kwargs: object) -> dict[str, object]:
                gc.collect()
                if previous:
                    self.assertIsNone(previous[-1]())
                previous.append(weakref.ref(value))
                return {"identity": identity, "edge_tp": 1.0}

            with mock.patch.object(scoreall, "score_graph", new=fake_score):
                scored, provenance, coverage = scoreall.score_aggregate_input(
                    object(), path, ["train:a", "train:b"]
                )
        self.assertEqual([row["identity"] for row in scored], ["train:a", "train:b"])
        self.assertEqual(coverage["test_dataset_groups_skipped"], 1)
        self.assertEqual(coverage["skipped_test_video_ids"], ["test:x"])
        self.assertEqual(provenance["dataset_group_count"], 3)
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_aggregate_rejects_interleaved_dataset_recurrence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.csv"
            write_csv(path, [node_row("train:a"), node_row("train:b"), node_row("train:a", 2)])
            with self.assertRaisesRegex(scoreall.ScoreAllError, "recurs out of order"):
                list(scoreall.iter_aggregate_graphs(path))

    def test_aggregate_requires_exact_selected_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "partial.csv"
            write_csv(path, [node_row("train:a"), node_row("train:extra")])
            with mock.patch.object(
                scoreall,
                "score_graph",
                side_effect=lambda store, identity, value, **kwargs: {
                    "identity": identity,
                    "edge_tp": 1.0,
                },
            ):
                with self.assertRaisesRegex(scoreall.ScoreAllError, "missing selected"):
                    scoreall.score_aggregate_input(
                        object(), path, ["train:a", "train:missing"]
                    )

    def test_cli_inputs_are_mutually_exclusive(self) -> None:
        common = ["--zip", "x.zip", "--split", "s.json", "--output", "o.json"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            scoreall.parse_args(["--frags", "f", "--submission", "a.csv", *common])


if __name__ == "__main__":
    unittest.main()

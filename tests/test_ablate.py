from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ablate  # noqa: E402


def node_row(dataset: str, node_id: int, t: int, z: int, y: int, x: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "row_type": "node",
        "node_id": node_id,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
        "source_id": -1,
        "target_id": -1,
    }


def edge_row(dataset: str, source: int, target: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "row_type": "edge",
        "node_id": -1,
        "t": -1,
        "z": -1,
        "y": -1,
        "x": -1,
        "source_id": source,
        "target_id": target,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ablate.SUBMISSION_COLUMNS)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            writer.writerow({"id": row_id, **row})


def graph(dataset: str = "a") -> ablate.SubmissionGraph:
    return ablate.SubmissionGraph(
        dataset,
        {
            1: ablate.SubmissionNode(1, 0, 0, 0, 0),
            2: ablate.SubmissionNode(2, 1, 0, 3, 0),
            3: ablate.SubmissionNode(3, 1, 0, 12, 0),
            4: ablate.SubmissionNode(4, 0, 0, 20, 0),
            5: ablate.SubmissionNode(5, 1, 0, 32, 0),
        },
        [
            ablate.SubmissionEdge(1, 2),
            ablate.SubmissionEdge(1, 3),
            ablate.SubmissionEdge(4, 5),
        ],
    )


class CsvTests(unittest.TestCase):
    def test_structured_parse_maps_plain_ids_without_rewriting_source(self) -> None:
        rows = [
            node_row("a", 1, 0, 0, 0, 0),
            node_row("a", 2, 1, 0, 1, 0),
            edge_row("a", 1, 2),
            node_row("b", 1, 0, 0, 0, 0),
        ]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "sub.csv"
            write_csv(path, rows)
            before = path.read_bytes()
            graphs, count = ablate.load_csv(path, ("a", "b"))
            after = path.read_bytes()
        self.assertEqual(count, 4)
        self.assertEqual(sorted(graphs), ["a", "b"])
        self.assertEqual(ablate.train_id("a"), "train:a")
        self.assertEqual(before, after)

    def test_parser_rejects_header_and_frozen_id_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "sub.csv"
            path.write_text("dataset,id\na,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ablate.AblateError, "header"):
                ablate.load_csv(path, ("a", "b"))
            write_csv(path, [node_row("a", 1, 0, 0, 0, 0)])
            with self.assertRaisesRegex(ablate.AblateError, "coverage"):
                ablate.load_csv(path, ("a", "b"))


class PartTests(unittest.TestCase):
    def test_hash_partition_is_balanced_disjoint_and_stable(self) -> None:
        first = ablate.hash_part()
        second = ablate.hash_part(tuple(reversed(ablate.FROZEN_IDS)))
        self.assertEqual(first, second)
        tune = {key for key, row in first.items() if row["part"] == "tune"}
        evaluate = {key for key, row in first.items() if row["part"] == "eval"}
        self.assertEqual((len(tune), len(evaluate)), (11, 11))
        self.assertFalse(tune & evaluate)
        self.assertEqual(tune | evaluate, {ablate.train_id(x) for x in ablate.FROZEN_IDS})
        self.assertEqual(
            tune,
            {
                "train:44b6_1d530831",
                "train:44b6_949adeb1",
                "train:44b6_a2bb48bb",
                "train:44b6_c15fded2",
                "train:44b6_c771cb04",
                "train:6bba_12665c0e",
                "train:6bba_3abfe10a",
                "train:6bba_705ec2c9",
                "train:6bba_8b7818bf",
                "train:6bba_aeee7805",
                "train:6bba_d5eae175",
            },
        )
        self.assertTrue(all(len(row["hash"]) == 64 for row in first.values()))

    def test_candidates_are_predeclared_global_and_short(self) -> None:
        names = [cand.name for cand in ablate.CANDS]
        self.assertEqual(names, ["base", "edge10", "edge8", "edge6", "edge4", "div10", "div8"])
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(len(name) < 15 for name in names))
        self.assertTrue(
            all(
                cand.as_dict()["node_policy"] == "preserve_all"
                for cand in ablate.CANDS
            )
        )


class TransformTests(unittest.TestCase):
    def test_physical_edge_cutoff_is_global_and_preserves_nodes(self) -> None:
        source = graph()
        result, stats = ablate.apply_cand(source, ablate.CANDS[3], (1.0, 1.0, 1.0))
        self.assertEqual(result.nodes, source.nodes)
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in result.edges],
            [(1, 2)],
        )
        self.assertEqual(stats["removed_nodes"], 0)
        self.assertEqual(stats["removed_edges"], 2)

    def test_division_pruning_drops_only_farther_daughter(self) -> None:
        source = graph()
        result, stats = ablate.apply_cand(source, ablate.CANDS[-1], (1.0, 1.0, 1.0))
        self.assertEqual(result.nodes, source.nodes)
        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in result.edges],
            [(1, 2), (4, 5)],
        )
        self.assertEqual(stats["before_divisions"], 1)
        self.assertEqual(stats["after_divisions"], 0)
        self.assertEqual(stats["removed_edges"], 1)


class ScoreTests(unittest.TestCase):
    def test_official_scorer_is_called_for_every_candidate_and_tune_precedes_eval(self) -> None:
        graphs = {"a": graph("a"), "b": graph("b")}
        parts = {
            "train:a": {"hash": "a" * 64, "part": "tune"},
            "train:b": {"hash": "b" * 64, "part": "eval"},
        }
        calls: list[tuple[str, str]] = []

        def fake_score(
            store: object,
            identity: str,
            value: ablate.SubmissionGraph,
            *,
            source: dict[str, object],
        ) -> dict[str, object]:
            calls.append((str(source["candidate"]), identity))
            score = 1.0 if source["candidate"] == "edge8" else 0.5
            return {"identity": identity, "score": score, "elapsed_seconds": 9.0}

        def fake_summary(rows: list[dict[str, object]]) -> dict[str, object]:
            return {"n": len(rows), "score": sum(float(row["score"]) for row in rows) / len(rows)}

        with mock.patch.object(ablate, "_scale", return_value=(1.0, 1.0, 1.0)), mock.patch.object(
            ablate.scoreall, "score_graph", side_effect=fake_score
        ), mock.patch.object(ablate.scoreall, "official_summary", side_effect=fake_summary):
            records, selected = ablate.score_all(object(), graphs, parts, "c" * 64)

        self.assertEqual(selected, "edge8")
        self.assertEqual(len(calls), len(ablate.CANDS) * 2)
        self.assertTrue(all(identity == "train:a" for _, identity in calls[: len(ablate.CANDS)]))
        self.assertTrue(all(identity == "train:b" for _, identity in calls[len(ablate.CANDS) :]))
        self.assertEqual(len(records), len(ablate.CANDS))
        self.assertNotIn("elapsed_seconds", records[0]["samples"][0])

    def test_selection_ignores_eval_and_uses_declaration_order_for_ties(self) -> None:
        records = {
            cand.name: {
                "summary": {
                    "tune": {"score": 0.5},
                    "eval": {"score": float(index)},
                }
            }
            for index, cand in enumerate(ablate.CANDS)
        }
        self.assertEqual(ablate.pick_best(records), "base")
        records["edge6"]["summary"]["tune"]["score"] = 0.6
        self.assertEqual(ablate.pick_best(records), "edge6")

    def test_json_is_strict_and_deterministic(self) -> None:
        value = {"z": math.nan, "a": [math.inf, 1.0]}
        first = ablate.json_text(value)
        second = ablate.json_text(value)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), {"a": [None, 1.0], "z": None})
        self.assertNotIn("NaN", first)
        self.assertNotIn("Infinity", first)


if __name__ == "__main__":
    unittest.main()

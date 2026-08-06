from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("a7_phase_template", HERE / "phase_template.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Node:
    def __init__(self, node_id: int, t: int) -> None:
        self.node_id = node_id
        self.t = t


class Edge:
    def __init__(self, source: int, target: int) -> None:
        self.source_id = source
        self.target_id = target


class Graph:
    def __init__(self) -> None:
        self.nodes = {value: Node(value, value) for value in range(6)}
        self.edges = [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(3, 5)]


def test_phase_evidence_is_deterministic_and_subvoxel_finite() -> None:
    source = np.zeros((33, 33, 33), dtype=np.float32)
    source[16, 16, 16] = 1.0
    destination = np.roll(source, shift=(1, -2, 3), axis=(0, 1, 2))
    first = MODULE.phase_evidence(source, destination)
    second = MODULE.phase_evidence(source, destination)
    assert first == second
    assert np.isfinite(
        [first.peak, first.second_peak, first.peak_ratio, first.shift_norm_um]
    ).all()
    assert np.allclose(np.abs(first.shift_zyx_um), np.asarray((1, 2, 3)) * 0.40625)


def test_physical_crop_uses_trilinear_border_replication() -> None:
    image = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    crops = MODULE.sample_crops(image, np.asarray([[0, 0, 0]]), (4.0, 4.0, 4.0))
    assert crops.shape == (1, 33, 33, 33)
    assert crops[0, 0, 0, 0] == image[0, 0, 0]
    assert np.isfinite(crops).all()


def test_tracklet_walk_stops_on_division() -> None:
    graph = Graph()
    incoming, outgoing = MODULE.graph_incidence(graph)
    assert MODULE.source_tracklet(graph, incoming, 3) == (1, 2, 3)
    assert MODULE.destination_tracklet(graph, outgoing, 2) == (2, 3)


def test_strict_selection_owns_all_touched_nodes() -> None:
    selected = MODULE.strict_select(
        [0, 1, 2],
        video=np.asarray([0, 0, 0]),
        source=np.asarray([1, 4, 7]),
        target=np.asarray([2, 5, 8]),
        conflicts=(((0, 2),), ((2, 3),), ((6, 8),)),
        budget=3,
        per_video_cap=3,
    )
    assert selected.tolist() == [0, 2]


def test_periodic_hann_has_frozen_geometry() -> None:
    window = MODULE.periodic_hann()
    assert window.shape == (33, 33, 33)
    assert window.dtype == np.float64
    assert window[0, 0, 0] == 0.0
    assert float(window.max()) < 1.0

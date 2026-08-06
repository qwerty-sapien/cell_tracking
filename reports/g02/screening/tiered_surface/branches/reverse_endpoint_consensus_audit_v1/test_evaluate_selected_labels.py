from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "evaluate_selected_labels", HERE / "evaluate_selected_labels.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_zero_harmful_gate_opens():
    result = MODULE.decide(np.ones(4, dtype=np.uint8), np.ones(4))
    assert result["graph_gate_open"] is True


def test_one_harmful_closes_graph():
    result = MODULE.decide(np.asarray([1, 1, 1, 0]), np.asarray([1, 1, 1, -1]))
    assert result["graph_gate_open"] is False
    assert result["checks"]["harmful_zero"] is False


def test_too_little_support_closes_graph():
    result = MODULE.decide(np.ones(3, dtype=np.uint8), np.ones(3))
    assert result["graph_gate_open"] is False
    assert result["checks"]["edit_count_4_to_64"] is False

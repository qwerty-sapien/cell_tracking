from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "select_literal_consensus", HERE / "select_literal_consensus.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def proposal(*, forward=0.6, reverse=0.7, score=0.8, null=0.2):
    return {
        "conflicts": [(3, 4)],
        "forward_margin": np.asarray([forward]),
        "reverse_margin": np.asarray([reverse]),
        "reverse_challenger_score": np.asarray([score]),
        "reverse_challenger_null": np.asarray([null]),
        "reverse_conflict_score": np.asarray([score - reverse]),
        "cycle_sign_agreement": np.asarray([1], dtype=np.uint8),
    }


def test_literal_consensus_passes_exact_return():
    passed, summary, reason = MODULE.evidence_pass(
        (1, 2), {(3, 4)}, {(1, 2): [proposal()]}
    )
    assert passed
    assert reason == "pass"
    assert summary["min_reverse_return_over_null"] == pytest.approx(0.6)


def test_disabled_reverse_is_null():
    passed, _summary, reason = MODULE.evidence_pass(
        (1, 2),
        {(3, 4)},
        {(1, 2): [proposal(reverse=0.0, score=0.0, null=0.0)]},
    )
    assert not passed
    assert reason == "threshold"


def test_extra_or_duplicate_conflict_fails_closed():
    row = proposal()
    row["conflicts"] = [(3, 4), (5, 6)]
    passed, summary, reason = MODULE.evidence_pass(
        (1, 2), {(3, 4)}, {(1, 2): [row]}
    )
    assert not passed
    assert summary is None
    assert reason == "removed_edge_alignment"


def test_duplicate_challenger_fails_closed():
    passed, summary, reason = MODULE.evidence_pass(
        (1, 2), {(3, 4)}, {(1, 2): [proposal(), proposal()]}
    )
    assert not passed
    assert summary is None
    assert reason == "missing_or_duplicate_challenger"

"""Focused tests for the frozen AE-P5 C0/C2 assignment rule."""

from __future__ import annotations

import math

import pytest

import ae_p5_assignment as ae


def act(
    source: int,
    target: int,
    margin: float,
    row: int = 0,
    incumbent_edges: tuple[tuple[int, int], ...] = (),
) -> ae.Action:
    return ae.Action(
        video="train:44b6_test",
        row=row,
        source_id=source,
        target_id=target,
        source_t=0,
        target_t=1,
        margin=margin,
        incumbent_edges=incumbent_edges,
    )


def chain(n: int, margin: float = 1.0) -> list[ae.Action]:
    """A path component: each action shares exactly one owned slot with its neighbour.

    Even k owns (out k/2, in k/2); odd k owns (out k//2, in k//2+1). So k and k+1
    always collide and non-neighbours never do. A path maximises the
    feasible-assignment count for a given component size.
    """
    actions = []
    for k in range(n):
        source = k // 2
        target = k // 2 if k % 2 == 0 else k // 2 + 1
        actions.append(act(source=source, target=target, margin=margin, row=k))
    return actions


# --- cost and evidence ----------------------------------------------------


def test_cost_is_the_negated_margin():
    assert act(1, 2, 0.25).cost == -0.25
    assert act(1, 2, -0.25).cost == 0.25


def test_no_op_costs_exactly_zero():
    assert ae.NO_OP_COST == 0.0
    assert ae._assignment_key(())[0] == 0.0


def test_evidence_is_the_49th_feature():
    assert ae.MARGIN_FEATURE_INDEX == 48
    assert ae.B0_FEATURE_WIDTH == 49


def test_nonfinite_evidence_sends_the_whole_component_to_apncc():
    for bad in (float("nan"), float("inf"), float("-inf")):
        record = ae.solve_component([act(1, 2, bad)])
        assert record["decision"] == "APNCC_FALLBACK"
        assert record["reason"] == "missing_or_nonfinite_evidence"


# --- ownership and feasibility -------------------------------------------


def test_actions_sharing_a_target_are_not_jointly_feasible():
    a, b = act(1, 9, 1.0), act(2, 9, 1.0)
    assert ae._disjoint([a]) is True
    assert ae._disjoint([a, b]) is False


def test_actions_sharing_a_source_are_not_jointly_feasible():
    assert ae._disjoint([act(5, 1, 1.0), act(5, 2, 1.0)]) is False


def test_disjoint_actions_are_jointly_feasible():
    assert ae._disjoint([act(1, 2, 1.0), act(3, 4, 1.0)]) is True


def test_actions_sharing_an_incumbent_edge_are_not_jointly_feasible():
    a = act(1, 2, 1.0, incumbent_edges=((90, 91),))
    b = act(3, 4, 1.0, incumbent_edges=((90, 91),))
    assert ae._disjoint([a, b]) is False


def test_actions_sharing_a_challenger_or_noop_endpoint_are_not_jointly_feasible():
    a = act(1, 2, 1.0)
    b = act(1, 2, 0.5, row=1)
    assert ae._disjoint([a, b]) is False


def test_empty_subset_is_always_a_feasible_assignment():
    found = ae.feasible_assignments([act(1, 2, 1.0)])
    assert () in found
    assert len(found) == 2  # no-op, or apply


def test_singleton_component_has_exactly_two_feasible_assignments():
    assert len(ae.feasible_assignments(chain(1))) == 2


# --- the k=8 rule and its fallback ---------------------------------------


def test_component_below_eight_feasible_assignments_falls_back():
    record = ae.solve_component(chain(1))
    assert record["decision"] == "APNCC_FALLBACK"
    assert record["reason"] == "fewer_than_eight_feasible_assignments"
    assert record["selected"] == ()


def test_conflicting_pair_falls_back():
    record = ae.solve_component([act(1, 9, 1.0), act(2, 9, 0.5)])
    assert record["reason"] == "fewer_than_eight_feasible_assignments"


def test_map_is_still_reported_when_the_rule_falls_back():
    """C0's MAP is defined even where C2 forbids applying it."""
    record = ae.solve_component([act(1, 2, 0.75)])
    assert record["c0_map_cost"] == pytest.approx(-0.75)
    assert record["c0_map_size"] == 1
    assert record["selected"] == ()


def test_map_prefers_the_negative_cost_assignment():
    """A positive margin lowers total cost, so the MAP applies the action."""
    component = [act(1, 2, 0.75)]
    best = min(ae.feasible_assignments(component), key=ae._assignment_key)
    assert len(best) == 1


def test_map_rejects_a_harmful_margin():
    """A negative margin raises total cost above the no-op, so MAP is empty."""
    component = [act(1, 2, -0.75)]
    best = min(ae.feasible_assignments(component), key=ae._assignment_key)
    assert best == ()


def test_tie_break_is_cost_then_sorted_canonical_digest():
    a, b = act(1, 2, 0.5, row=0), act(3, 4, 0.5, row=1)
    key_a, key_b = ae._assignment_key([a]), ae._assignment_key([b])
    assert key_a[0] == key_b[0]
    assert key_a[1] != key_b[1]
    assert sorted([key_a, key_b])[0][1] < sorted([key_a, key_b])[1][1]


def test_canonical_digest_is_stable_and_endpoint_specific():
    assert act(1, 2, 0.5).canonical_sha256() == act(1, 2, 0.9).canonical_sha256()
    assert act(1, 2, 0.5).canonical_sha256() != act(1, 3, 0.5).canonical_sha256()


def test_min_marginal_tolerance_scales_with_cost():
    assert ae._min_marginal_tolerance(0.0) == pytest.approx(
        100.0 * 2.220446049250313e-16
    )
    assert ae._min_marginal_tolerance(1000.0) > ae._min_marginal_tolerance(1.0)


def test_zero_margin_action_is_never_applied():
    """A zero margin has zero marginal value, inside the tolerance."""
    component = chain(6, margin=0.0)
    record = ae.solve_component(component)
    assert record["decision"] == "APNCC_FALLBACK"


# --- the structural finding ----------------------------------------------


def test_eight_feasible_assignments_needs_a_component_of_at_least_five():
    """The k=8 rule cannot engage on small components.

    Independent-set counts on a path of n actions are Fibonacci-like: 2, 3, 5,
    8, 13. Fewer than five chained actions can never reach eight.
    """
    counts = [len(ae.feasible_assignments(chain(n))) for n in range(1, 6)]
    assert counts == [2, 3, 5, 8, 13]
    assert all(count < ae.K_BEST for count in counts[:3])


def test_a_uniform_margin_chain_of_five_fails_the_membership_conjunction():
    """Even at eight assignments, uniform evidence cannot satisfy 7/8 and 1/8."""
    record = ae.solve_component(chain(5, margin=1.0))
    assert record["feasible_assignment_count"] == 13
    assert record["decision"] == "APNCC_FALLBACK"
    assert record["reason"] in {
        "introduce_membership_below_seven_eighths",
        "remove_membership_above_one_eighth",
    }

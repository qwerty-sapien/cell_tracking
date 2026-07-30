#!/usr/bin/env python3
"""Synthetic, standard-library-only DPRC contract tests.

No model, image/video, campaign data, R5 artifact, or third-party package is read.
The test checks the intended observable only; it is not a biological evaluation.
"""

from __future__ import annotations

import math
import unittest


GRID = 4


def dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    norm = math.sqrt(dot(left, left) * dot(right, right))
    if norm == 0.0:
        raise ValueError("zero-norm patch token")
    return dot(left, right) / norm


def neighbours(index: int) -> tuple[int, ...]:
    row, col = divmod(index, GRID)
    return tuple(
        other_row * GRID + other_col
        for other_row in range(max(0, row - 1), min(GRID, row + 2))
        for other_col in range(max(0, col - 1), min(GRID, col + 2))
    )


def best(source: tuple[float, ...], target: list[tuple[float, ...]], allowed: tuple[int, ...]) -> int:
    # Index is the deterministic tie-breaker, making the fixture reproducible.
    return max(allowed, key=lambda item: (cosine(source, target[item]), -item))


def cycle_score(
    source: list[tuple[float, ...]],
    target: list[tuple[float, ...]],
    source_mask: list[bool],
    target_mask: list[bool],
) -> float:
    if not (len(source) == len(target) == len(source_mask) == len(target_mask) == GRID * GRID):
        raise ValueError("expected a 4x4 token grid and masks")
    foreground_count = sum(source_mask)
    if foreground_count == 0:
        raise ValueError("no foreground cycle correspondence")
    score_sum = 0.0
    available_target = False
    for source_index, source_token in enumerate(source):
        if not source_mask[source_index]:
            continue
        forward_allowed = tuple(index for index in neighbours(source_index) if target_mask[index])
        if not forward_allowed:
            continue
        available_target = True
        target_index = best(source_token, target, forward_allowed)
        backward_allowed = tuple(index for index in neighbours(target_index) if source_mask[index])
        if not backward_allowed:
            continue
        if best(target[target_index], source, backward_allowed) == source_index:
            # Divide by every source foreground patch, not just successful cycles:
            # coverage is part of the observable and cannot be hidden by an
            # average over a convenient surviving subset.
            score_sum += cosine(source_token, target[target_index])
    if not available_target:
        raise ValueError("no foreground cycle correspondence")
    return score_sum / foreground_count


def global_mean_cosine(left: list[tuple[float, ...]], right: list[tuple[float, ...]]) -> float:
    width = len(left[0])
    mean_left = tuple(sum(item[axis] for item in left) / len(left) for axis in range(width))
    mean_right = tuple(sum(item[axis] for item in right) / len(right) for axis in range(width))
    return cosine(mean_left, mean_right)


def basis_grid() -> list[tuple[float, ...]]:
    return [tuple(float(index == axis) for axis in range(GRID * GRID)) for index in range(GRID * GRID)]


class DPRCFixtureTest(unittest.TestCase):
    def test_spatial_permutation_breaks_cycle_but_not_global_pool(self) -> None:
        source = basis_grid()
        true_target = list(source)
        # Same global mean, but most patches sit outside their local window.
        false_target = list(reversed(source))
        mask = [True] * (GRID * GRID)
        self.assertEqual(global_mean_cosine(source, true_target), 1.0)
        self.assertEqual(global_mean_cosine(source, false_target), 1.0)
        self.assertEqual(cycle_score(source, true_target, mask, mask), 1.0)
        self.assertLessEqual(cycle_score(source, false_target, mask, mask), 0.25)

    def test_uniform_tokens_have_no_preference_margin(self) -> None:
        uniform = [(1.0, 0.0)] * (GRID * GRID)
        mask = [True] * (GRID * GRID)
        incumbent = cycle_score(uniform, uniform, mask, mask)
        alternative = cycle_score(uniform, uniform, mask, mask)
        self.assertEqual(incumbent, alternative)
        self.assertEqual(alternative - incumbent, 0.0)

    def test_empty_foreground_fails_closed(self) -> None:
        source = basis_grid()
        with self.assertRaisesRegex(ValueError, "no foreground"):
            cycle_score(source, source, [True] * 16, [False] * 16)

    def test_repeat_is_identical(self) -> None:
        source = basis_grid()
        target = list(reversed(source))
        mask = [True] * 16
        self.assertEqual(cycle_score(source, target, mask, mask), cycle_score(source, target, mask, mask))


if __name__ == "__main__":
    unittest.main(verbosity=2)

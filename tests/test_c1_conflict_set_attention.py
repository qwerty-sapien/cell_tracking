from __future__ import annotations

import unittest

import numpy as np
import torch

from celltrack_alpha.c1_conflict_set_attention import (
    ComponentExample,
    ConflictSetAttention,
    abstention_scores,
    conflict_components,
    fold_index,
    fit_conflict_head,
    nested_partitions,
    parameter_count,
    predict_components,
    seed_everything,
)


class ConflictSetAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything()
        self.model = ConflictSetAttention().eval()

    def test_parameter_ceiling_and_exact_count(self) -> None:
        self.assertEqual(parameter_count(self.model), 20_097)

    def test_permutation_equivariance(self) -> None:
        values = torch.arange(4 * 49, dtype=torch.float32).reshape(1, 4, 49)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        with torch.no_grad():
            original, original_no_link = self.model(values)
            permuted, permuted_no_link = self.model(values[:, permutation])
        torch.testing.assert_close(original, permuted[:, inverse])
        torch.testing.assert_close(original_no_link, permuted_no_link)

    def test_padding_mask_invariance(self) -> None:
        values = torch.randn(1, 3, 49)
        padded = torch.cat((values, torch.randn(1, 4, 49)), dim=1)
        valid = torch.tensor([[True, True, True, False, False, False, False]])
        with torch.no_grad():
            plain, plain_no_link = self.model(values)
            masked, masked_no_link = self.model(padded, valid)
        torch.testing.assert_close(plain, masked[:, :3])
        torch.testing.assert_close(plain_no_link, masked_no_link)
        self.assertTrue(torch.equal(masked[:, 3:], torch.zeros(1, 4)))

    def test_components_do_not_communicate(self) -> None:
        first = torch.randn(1, 3, 49)
        second = torch.randn(1, 2, 49)
        changed = second + 1000
        with torch.no_grad():
            first_score, first_no_link = self.model(first)
            self.model(second)
            changed_first_score, changed_first_no_link = self.model(first)
            self.model(changed)
        torch.testing.assert_close(first_score, changed_first_score)
        torch.testing.assert_close(first_no_link, changed_first_no_link)

    def test_explicit_no_link_changes_abstention_score(self) -> None:
        proposal = torch.tensor([[2.0, -1.0]])
        no_link = torch.tensor([0.5])
        self.assertTrue(
            torch.equal(
                abstention_scores(proposal, no_link),
                torch.tensor([[1.5, -1.5]]),
            )
        )

    def test_conflict_component_construction(self) -> None:
        payload = {
            "challengers": np.asarray(
                [[1, 10], [2, 10], [7, 12], [20, 30]], dtype=np.int64
            ),
            "conflicts": np.asarray(
                [[1, 11], [3, 10], [7, 13], [20, 31]], dtype=np.int64
            ),
            "proposal_offsets": np.arange(5, dtype=np.int64),
            "transition_offsets": np.asarray([0, 3, 4], dtype=np.int64),
        }
        components = conflict_components(payload)
        self.assertEqual(
            [value.tolist() for value in components], [[0, 1], [2], [3]]
        )

    def test_folds_are_stable_and_video_disjoint(self) -> None:
        identities = tuple(f"train:44b6_{index:08x}" for index in range(100))
        self.assertEqual(fold_index(identities[0]), fold_index(identities[0]))
        partitions = nested_partitions(identities)
        held = []
        for fit_ids, calibration_ids, held_ids in partitions:
            self.assertFalse(set(fit_ids) & set(calibration_ids))
            self.assertFalse(set(fit_ids) & set(held_ids))
            self.assertFalse(set(calibration_ids) & set(held_ids))
            held.extend(held_ids)
        self.assertEqual(sorted(held), sorted(identities))

    def test_small_fit_is_byte_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        examples = []
        for ordinal in range(6):
            count = 2 + ordinal % 3
            examples.append(
                ComponentExample(
                    identity=f"train:44b6_{ordinal:08x}",
                    ordinal=ordinal,
                    features=rng.normal(size=(count, 49)).astype(np.float32),
                    labels=np.asarray(
                        [1 if index == ordinal % count else 0 for index in range(count)],
                        dtype=np.float32,
                    ),
                    proposal_indices=np.arange(count),
                )
            )
        first = fit_conflict_head(examples, epochs=1, component_batch_size=3)
        second = fit_conflict_head(examples, epochs=1, component_batch_size=3)
        first_bytes = b"".join(
            value.tobytes() for value in predict_components(first, examples)
        )
        second_bytes = b"".join(
            value.tobytes() for value in predict_components(second, examples)
        )
        self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()

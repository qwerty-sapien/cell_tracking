from __future__ import annotations

import torch

from celltrack_alpha.crop_linker import (
    SiameseCropLinker,
    calibration_metrics,
    fit_temperature,
    normalize_crops,
)


def test_crop_normalization_is_per_example_and_flat_safe() -> None:
    crops = torch.stack(
        (
            torch.arange(16 * 32 * 32, dtype=torch.float32).reshape(
                1, 16, 32, 32
            ),
            torch.ones((1, 16, 32, 32), dtype=torch.float32),
        )
    )
    normalized = normalize_crops(crops)
    assert abs(float(normalized[0].mean())) < 1.0e-6
    assert torch.isclose(
        torch.linalg.vector_norm(normalized[0]), torch.tensor(1.0)
    )
    assert torch.count_nonzero(normalized[1]) == 0


def test_siamese_crop_linker_shapes_and_shared_encoder() -> None:
    torch.manual_seed(4)
    model = SiameseCropLinker(embedding_width=8, hidden=16)
    source = torch.randn((2, 1, 16, 32, 32))
    target = torch.randn((2, 1, 16, 32, 32))
    context = torch.randn((2, 48))
    logits = model(source, target, context)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_temperature_calibration_improves_overconfident_logits() -> None:
    logits = torch.tensor([8.0, 6.0, 4.0, -4.0, -6.0, -8.0])
    # Deliberately include label noise so temperature > 1 is useful.
    labels = torch.tensor([1, 1, 0, 1, 0, 0])
    fitted = fit_temperature(logits, labels, steps=300)
    assert fitted.temperature > 1.0
    assert (
        fitted.after["negative_log_likelihood"]
        < fitted.before["negative_log_likelihood"]
    )
    assert fitted.after["brier_score"] < fitted.before["brier_score"]
    observed = calibration_metrics(fitted.apply(logits), labels)
    assert observed == fitted.after

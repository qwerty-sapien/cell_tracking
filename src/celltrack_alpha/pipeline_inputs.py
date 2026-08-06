"""Validation and normalization for ``predict_dataset_window`` inputs.

The tracker deliberately accepts an already-loaded array-like object so callers
can retain their own backing store.  These helpers establish the deterministic
window contract before the execution pipeline starts consuming that object.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np


def validate_array_window(
    dataset: str,
    array: Any,
    start: int,
    end: int | None,
) -> tuple[int, int]:
    """Validate an array-backed T/Z/Y/X prediction window."""

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    try:
        shape = tuple(int(value) for value in array.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("array must expose a four-dimensional T/Z/Y/X shape") from exc
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(
            f"array must have a non-empty four-dimensional T/Z/Y/X shape, got {shape!r}"
        )
    dtype = getattr(array, "dtype", None)
    if dtype is not None:
        try:
            resolved_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"array dtype is not supported: {dtype!r}") from exc
        if not np.issubdtype(resolved_dtype, np.number) or np.issubdtype(
            resolved_dtype, np.complexfloating
        ):
            raise ValueError(f"array dtype must be real numeric, got {resolved_dtype}")
    if isinstance(start, bool) or not isinstance(start, Integral) or int(start) < 0:
        raise ValueError(f"start must be a non-negative integer, got {start!r}")
    resolved_start = int(start)
    if end is None:
        resolved_end = shape[0]
    elif isinstance(end, bool) or not isinstance(end, Integral):
        raise ValueError(f"end must be an integer or None, got {end!r}")
    else:
        resolved_end = int(end)
    if resolved_end <= resolved_start or resolved_end > shape[0]:
        raise ValueError(
            f"window must satisfy 0 <= start < end <= {shape[0]}, got "
            f"start={start!r}, end={end!r}"
        )
    return resolved_start, resolved_end


def validate_output_offset_override(
    output_offset_override: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """Normalize the optional Z/Y/X output offset without changing its contract."""

    if output_offset_override is None:
        return None
    if isinstance(output_offset_override, (str, bytes)) or not isinstance(
        output_offset_override, Iterable
    ):
        raise ValueError("output_offset_override must contain exactly three integers")
    offset_values = tuple(output_offset_override)
    if len(offset_values) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in offset_values
    ):
        raise ValueError("output_offset_override must contain exactly three integers")
    return tuple(int(value) for value in offset_values)

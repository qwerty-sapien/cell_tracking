"""Repository-wide pytest lifecycle dispositions."""

from __future__ import annotations

import pytest

from retired_test_dispositions import RETIRED_XFAILS


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply only reviewed, exact-node strict XFAIL dispositions."""

    matched: set[str] = set()
    for item in items:
        reason = RETIRED_XFAILS.get(item.nodeid)
        if reason is None:
            continue
        item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
        matched.add(item.nodeid)
    stale = sorted(set(RETIRED_XFAILS) - matched)
    # Focused invocations intentionally collect only a subset. The canonical
    # active-suite collection is >1,000 items and must cover every disposition.
    if len(items) >= 1_000 and stale:
        raise pytest.UsageError(
            "retired-test disposition registry contains uncollected node IDs: "
            + ", ".join(stale)
        )

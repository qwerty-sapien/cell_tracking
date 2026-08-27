#!/usr/bin/env python3
"""Infrastructure-only adapter for the legacy compact-bank monitor attribute."""

from __future__ import annotations

import score_winner_once as target


def main() -> int:
    if hasattr(target.CompositeGuard, "active_monitor"):
        raise RuntimeError("compatibility alias is unexpectedly already present")
    target.CompositeGuard.active_monitor = property(lambda self: self.monitor)
    return target.main()


if __name__ == "__main__":
    raise SystemExit(main())

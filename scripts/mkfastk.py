#!/usr/bin/env python3
"""Embed the deterministic fast zipapp in the single Kaggle code file."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kfast/run.py"
PACKAGE = ROOT / "kaggle_kernel/fastsrc.zip"
LIMIT = 1_000_000


def main() -> int:
    package = PACKAGE.read_bytes()
    encoded = base64.b85encode(package).decode("ascii")
    source = SOURCE.read_text(encoding="ascii")
    updated, count = re.subn(r'^EMBED = "[^"]*"$', f'EMBED = "{encoded}"', source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("run.py must contain exactly one EMBED assignment")
    payload = updated.encode("ascii")
    if len(payload) >= LIMIT:
        raise ValueError(f"embedded Kaggle source exceeds {LIMIT} bytes")
    SOURCE.write_bytes(payload)
    print(f"bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the single-file Kaggle kernel without changing scientific logic."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


STEP0_IMPORT = "from kaggle_step0_preflight import run_cellect_step0\n"
FUTURE_IMPORT = "from __future__ import annotations\n"


class KernelBuildError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build(kernel_path: Path, step0_path: Path) -> bytes:
    kernel = kernel_path.read_text()
    step0 = step0_path.read_text()
    if kernel.count(STEP0_IMPORT) != 1:
        raise KernelBuildError("kernel step-0 import marker is absent or duplicated")
    if step0.count(FUTURE_IMPORT) != 1:
        raise KernelBuildError("step-0 future-import marker is absent or duplicated")
    step0_tail = step0.split(FUTURE_IMPORT, 1)[1].lstrip("\n")
    combined = kernel.replace(
        STEP0_IMPORT,
        (
            "# BEGIN INLINE CONTENT-ADDRESSED STEP 0\n"
            + step0_tail
            + "\n# END INLINE CONTENT-ADDRESSED STEP 0\n"
        ),
    )
    if STEP0_IMPORT in combined:
        raise KernelBuildError("self-contained kernel retains a sibling import")
    compile(combined, str(kernel_path), "exec")
    return combined.encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.kernel, args.step0)
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise KernelBuildError(f"refusing to overwrite drifted output: {args.output}")
    else:
        args.output.write_bytes(payload)
    print(
        f"{args.output} bytes={len(payload)} sha256={sha256_bytes(payload)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

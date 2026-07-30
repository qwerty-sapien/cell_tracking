#!/usr/bin/env python3
"""Run one fresh-process, scorer-free U0-T4CANON replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import u0_t4canon  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--contract-sha256", required=True)
    value.add_argument("--fit-manifest", type=Path, required=True)
    value.add_argument("--base-report", type=Path, required=True)
    value.add_argument("--feature-report", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    paths = (
        ROOT / "src/celltrack_alpha/u0_t4canon.py",
        ROOT / "src/celltrack_alpha/trackastra_linear.py",
        ROOT / "src/celltrack_alpha/appearance_swap.py",
        ROOT / "scripts/replay_u0_t4canon.py",
    )
    result = u0_t4canon.run_replay(
        root=ROOT,
        contract_path=args.contract.resolve(),
        contract_sha256=args.contract_sha256,
        fit_manifest_path=args.fit_manifest.resolve(),
        base_report_path=args.base_report.resolve(),
        feature_report_path=args.feature_report.resolve(),
        implementation_paths=paths,
    )
    u0_t4canon.write_replay(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "video_count": result["scope"]["video_count"],
                "disabled_challenger_exact_apncc": result["graphs"][
                    "disabled_challenger_exact_apncc"
                ],
                "output": str(args.output.resolve()),
                "sha256": u0_t4canon.sha256_path(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

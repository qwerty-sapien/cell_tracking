#!/usr/bin/env python3
"""Restore one frozen model stack for a guarded memory-causality diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
from build_trackastra_full_pool_feature_cache import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_CONTRACT_SHA256,
    load_contract,
)
from evaluate_trackastra_zero_wrfeat_signal import _load_model  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
)
from trackastra_checkpoint_loader import load_mmap_meta  # noqa: E402


class StackRestoreDiagnosticError(RuntimeError):
    """The isolated stack restoration failed closed."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack", choices=("trackastra", "v45"), required=True
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint-zip", type=Path)
    parser.add_argument("--checkpoint-member", type=Path)
    parser.add_argument(
        "--trackastra-loader",
        choices=("conventional", "mmap-meta"),
        default="conventional",
    )
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--hold-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.hold_seconds < 0.0:
        raise StackRestoreDiagnosticError(
            "hold-seconds must be nonnegative"
        )
    contract = load_contract(
        args.contract.resolve(), args.contract_sha256
    )
    started = time.perf_counter()
    if args.stack == "trackastra":
        if args.source_root is None or args.checkpoint_zip is None:
            raise StackRestoreDiagnosticError(
                "Trackastra restoration requires source and checkpoint"
            )
        if args.trackastra_loader == "mmap-meta":
            if args.checkpoint_member is None:
                raise StackRestoreDiagnosticError(
                    "mmap-meta restoration requires checkpoint-member"
                )
            model = load_mmap_meta(
                source_root=args.source_root.resolve(),
                checkpoint_zip=args.checkpoint_zip.resolve(),
                checkpoint_member=args.checkpoint_member.resolve(),
                contract=contract,
            )
        else:
            model = _load_model(
                args.source_root.resolve(),
                args.checkpoint_zip.resolve(),
                contract,
            )
        restored = {
            "stack": "trackastra",
            "loader": args.trackastra_loader,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    else:
        base_model, bundle = bh_model.load_v45_model(
            args.support_root.resolve(), device="cpu"
        )
        scorer, edge_record = run_bh.load_sparse_edge_scorer(
            args.edge_artifact.resolve(),
            sweep._sha256(args.edge_artifact.resolve()),
            args.edge_auxiliary.resolve(),
            base_model,
            bundle,
            edge_module=bh_edge,
        )
        if scorer is None or edge_record is None:
            raise StackRestoreDiagnosticError(
                "V45 edge scorer did not restore"
            )
        restored = {
            "stack": "v45",
            "edge_artifact_sha256": sweep._sha256(
                args.edge_artifact.resolve()
            ),
        }
    load_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                **restored,
                "load_seconds": load_seconds,
                "hold_seconds": args.hold_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    time.sleep(args.hold_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze deterministic G-02 scorer and split-integrity manifests from a ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.g02_integrity import freeze_integrity_outputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--audit-videos", type=int, default=20)
    parser.add_argument("--signature-chunks-per-video", type=int, default=3)
    parser.add_argument("--max-uncompressed-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--seed", default="g02-split-v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = freeze_integrity_outputs(
        args.zip_path,
        args.output_dir,
        fold_count=args.fold_count,
        audit_videos=args.audit_videos,
        signature_chunks_per_video=args.signature_chunks_per_video,
        max_uncompressed_chunk_bytes=args.max_uncompressed_chunk_bytes,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


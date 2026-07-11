#!/usr/bin/env python3
"""Stream lightweight CTC image/track diagnostics from ZIP archives."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from inspect_zip_archives import parse_ctc_track_file


RAW_TIF_RE = re.compile(r"/(\d{2})/[^/]+\.tiff?$", re.IGNORECASE)


def percentile_stats(values: np.ndarray) -> dict[str, float | int]:
    values = values.astype(np.float64, copy=False)
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def read_tiff_probe(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with zf.open(member) as fh:
        payload = fh.read()
    with Image.open(io.BytesIO(payload)) as img:
        n_frames = getattr(img, "n_frames", 1)
        frame_indices = sorted({0, max(0, n_frames // 2), max(0, n_frames - 1)})
        frame_stats = []
        preview_arrays = []
        for frame_index in frame_indices[:3]:
            img.seek(frame_index)
            arr = np.asarray(img)
            frame_stats.append(
                {
                    "frame_index": int(frame_index),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    **percentile_stats(arr),
                }
            )
            preview_arrays.append(arr)
        return {
            "member": member,
            "mode": img.mode,
            "size_xy": list(img.size),
            "n_frames": int(n_frames),
            "sampled_frames": frame_stats,
            "_preview_arrays": preview_arrays,
        }


def normalize_preview(arr: np.ndarray) -> Image.Image:
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def save_preview(probe: dict[str, Any], output_path: Path) -> None:
    arrays = probe.pop("_preview_arrays", [])
    if not arrays:
        return
    images = [normalize_preview(arr) for arr in arrays]
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    canvas = Image.new("L", (width, height), 0)
    x = 0
    for img in images:
        canvas.paste(ImageOps.contain(img, (img.width, height)), (x, 0))
        x += img.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def track_stats(zf: zipfile.ZipFile) -> dict[str, Any]:
    out = {}
    for info in zf.infolist():
        if Path(info.filename).name != "man_track.txt":
            continue
        with zf.open(info) as fh:
            text = fh.read().decode("utf-8", errors="replace")
        out[info.filename] = parse_ctc_track_file(text)
    return out


def analyze_archive(path: Path, plots_dir: Path, max_members_per_sequence: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "valid_zip": False}
    with zipfile.ZipFile(path) as zf:
        result["valid_zip"] = True
        raw_by_sequence: dict[str, list[str]] = defaultdict(list)
        for name in zf.namelist():
            if "_GT/" in name or "_ST/" in name:
                continue
            match = RAW_TIF_RE.search(name)
            if match:
                raw_by_sequence[match.group(1)].append(name)

        image_probes = {}
        stem = path.stem.replace("+", "plus")
        for sequence, members in sorted(raw_by_sequence.items()):
            selected = sorted(members)[:max_members_per_sequence]
            image_probes[sequence] = []
            for member in selected:
                probe = read_tiff_probe(zf, member)
                safe_member = Path(member).stem
                preview_path = plots_dir / f"{stem}_{sequence}_{safe_member}.png"
                save_preview(probe, preview_path)
                probe["preview_png"] = str(preview_path)
                image_probes[sequence].append(probe)

        result["raw_sequence_count"] = len(raw_by_sequence)
        result["raw_tif_counts_by_sequence"] = {
            sequence: len(members) for sequence, members in sorted(raw_by_sequence.items())
        }
        result["image_probes"] = image_probes
        result["track_files"] = track_stats(zf)
    return result


def write_markdown(results: list[dict[str, Any]], output: Path) -> None:
    lines = ["# CTC Streaming Analysis", ""]
    for result in results:
        lines.append(f"## {result['path']}")
        lines.append("")
        lines.append(f"- Valid ZIP: {result['valid_zip']}")
        lines.append(f"- Raw sequences: {result.get('raw_sequence_count')}")
        lines.append(f"- Raw TIFF counts: `{result.get('raw_tif_counts_by_sequence')}`")
        lines.append("")
        lines.append("### Track Summary")
        lines.append("```json")
        lines.append(json.dumps(result.get("track_files", {}), indent=2)[:8000])
        lines.append("```")
        lines.append("")
        lines.append("### Image Probes")
        for sequence, probes in result.get("image_probes", {}).items():
            lines.append(f"#### Sequence {sequence}")
            for probe in probes:
                compact = {
                    key: value
                    for key, value in probe.items()
                    if key not in {"sampled_frames"}
                }
                lines.append(f"- `{probe['member']}`")
                lines.append(f"  - metadata: `{compact}`")
                lines.append("  - sampled frame stats:")
                lines.append("```json")
                lines.append(json.dumps(probe["sampled_frames"], indent=2)[:4000])
                lines.append("```")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--plots-dir", type=Path, default=Path("reports/plots"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/ctc_streaming_analysis.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/ctc_streaming_analysis.md"))
    parser.add_argument("--max-members-per-sequence", type=int, default=2)
    args = parser.parse_args()

    results = [
        analyze_archive(path, args.plots_dir, args.max_members_per_sequence)
        for path in args.archives
    ]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, args.markdown_output)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


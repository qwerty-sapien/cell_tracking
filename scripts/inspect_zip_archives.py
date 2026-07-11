#!/usr/bin/env python3
"""Inspect Kaggle/CTC ZIP archives without extracting them."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SMALL_READ_LIMIT = 2_000_000


@dataclass
class ArchiveSummary:
    path: str
    exists: bool
    valid_zip: bool
    size_bytes: int | None = None
    entry_count: int | None = None
    compressed_bytes: int | None = None
    uncompressed_bytes: int | None = None
    top_level: dict[str, int] | None = None
    suffix_counts: dict[str, int] | None = None
    kaggle: dict[str, Any] | None = None
    ctc: dict[str, Any] | None = None
    sample_entries: list[str] | None = None
    error: str | None = None


def human_bytes(n: int | None) -> str:
    if n is None:
        return "NA"
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def suffix_for(name: str) -> str:
    path = Path(name)
    if name.endswith("/"):
        return "<dir>"
    if name.endswith(".zarr/.zattrs"):
        return ".zattrs"
    if name.endswith(".zarr/.zarray"):
        return ".zarray"
    suffixes = path.suffixes
    if suffixes[-2:] == [".ome", ".zarr"]:
        return ".ome.zarr"
    return suffixes[-1].lower() if suffixes else "<none>"


def top_level_for(name: str) -> str:
    stripped = name.strip("/")
    return stripped.split("/", 1)[0] if stripped else "<root>"


def read_text_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str | None:
    if info.file_size > SMALL_READ_LIMIT:
        return None
    with zf.open(info) as fh:
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def inspect_kaggle(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> dict[str, Any]:
    names = [i.filename for i in infos]
    zarr_markers = [n for n in names if ".zarr/" in n or n.endswith(".zarr")]
    geff_entries = [n for n in names if n.lower().endswith(".geff") or ".geff/" in n]
    csv_entries = [n for n in names if n.lower().endswith(".csv")]
    json_entries = [
        i for i in infos
        if (
            i.filename.endswith("zarr.json")
            or i.filename.endswith(".zattrs")
            or i.filename.endswith(".zarray")
        )
        and i.file_size <= SMALL_READ_LIMIT
    ]

    sample_submission_headers: dict[str, list[str]] = {}
    for info in infos:
        base = Path(info.filename).name.lower()
        if base == "sample_submission.csv" or base.startswith("sample_submission"):
            text = read_text_member(zf, info)
            if text:
                reader = csv.reader(io.StringIO(text))
                try:
                    sample_submission_headers[info.filename] = next(reader)
                except StopIteration:
                    sample_submission_headers[info.filename] = []

    zarr_metadata: dict[str, Any] = {}
    for info in json_entries[:100]:
        text = read_text_member(zf, info)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        summary = {}
        for key in ("shape", "chunks", "dtype", "dimension_names", "axes", "zarr_format"):
            if key in parsed:
                summary[key] = parsed[key]
        if summary:
            zarr_metadata[info.filename] = summary

    dataset_names = set()
    for name in names:
        parts = name.split("/")
        for part in parts:
            if part.endswith(".zarr") or part.endswith(".ome.zarr"):
                dataset_names.add(part)

    return {
        "zarr_entry_count": len(zarr_markers),
        "geff_entry_count": len(geff_entries),
        "csv_entries": csv_entries[:50],
        "sample_submission_headers": sample_submission_headers,
        "zarr_datasets": sorted(dataset_names)[:100],
        "zarr_metadata_sample": zarr_metadata,
    }


def parse_ctc_track_file(text: str) -> dict[str, Any]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            label, start, end, parent = map(int, parts[:4])
        except ValueError:
            continue
        rows.append((label, start, end, parent))

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    starts = []
    ends = []
    for label, start, end, parent in rows:
        starts.append(start)
        ends.append(end)
        if parent > 0:
            children_by_parent[parent].append(label)
    divisions = {p: c for p, c in children_by_parent.items() if len(c) >= 2}

    return {
        "track_count": len(rows),
        "division_parent_count": len(divisions),
        "min_t": min(starts) if starts else None,
        "max_t": max(ends) if ends else None,
        "parented_track_count": sum(1 for row in rows if row[3] > 0),
        "example_divisions": dict(list(divisions.items())[:5]),
    }


def inspect_ctc(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> dict[str, Any]:
    names = [i.filename for i in infos]
    tif_entries = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
    track_infos = [
        i for i in infos
        if Path(i.filename).name in {"man_track.txt", "res_track.txt"}
        and i.file_size <= SMALL_READ_LIMIT
    ]
    mask_tifs = [
        n for n in tif_entries
        if re.search(r"_(GT|ST)/(TRA|SEG)/", n) or "/TRA/" in n or "/SEG/" in n
    ]
    raw_tifs = [n for n in tif_entries if n not in set(mask_tifs)]

    track_files = {}
    for info in track_infos:
        text = read_text_member(zf, info)
        if text:
            track_files[info.filename] = parse_ctc_track_file(text)

    sequence_counts = Counter()
    for name in raw_tifs:
        parts = name.split("/")
        for part in parts:
            if re.fullmatch(r"\d{2}", part):
                sequence_counts[part] += 1
                break

    return {
        "tif_entry_count": len(tif_entries),
        "raw_tif_entry_count": len(raw_tifs),
        "mask_tif_entry_count": len(mask_tifs),
        "raw_tifs_by_sequence": dict(sorted(sequence_counts.items())),
        "track_files": track_files,
        "sample_tif_entries": tif_entries[:20],
    }


def inspect_archive(path: Path) -> ArchiveSummary:
    summary = ArchiveSummary(path=str(path), exists=path.exists(), valid_zip=False)
    if not path.exists():
        summary.error = "file does not exist"
        return summary
    summary.size_bytes = path.stat().st_size
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            summary.valid_zip = True
            summary.entry_count = len(infos)
            summary.compressed_bytes = sum(i.compress_size for i in infos)
            summary.uncompressed_bytes = sum(i.file_size for i in infos)
            summary.top_level = dict(Counter(top_level_for(i.filename) for i in infos).most_common(50))
            summary.suffix_counts = dict(Counter(suffix_for(i.filename) for i in infos).most_common(50))
            names = [i.filename for i in infos]
            summary.sample_entries = names[:30]
            if any(".zarr" in n or n.lower().endswith(".geff") for n in names):
                summary.kaggle = inspect_kaggle(zf, infos)
            if any(n.lower().endswith((".tif", ".tiff")) for n in names) or any(
                Path(n).name in {"man_track.txt", "res_track.txt"} for n in names
            ):
                summary.ctc = inspect_ctc(zf, infos)
    except zipfile.BadZipFile as exc:
        summary.error = f"not a complete/valid ZIP: {exc}"
    except Exception as exc:  # noqa: BLE001 - keep batch inspection resilient.
        summary.error = f"{type(exc).__name__}: {exc}"
    return summary


def write_markdown(summaries: list[ArchiveSummary], output: Path) -> None:
    lines = ["# ZIP Archive Inspection", ""]
    for summary in summaries:
        lines.append(f"## {summary.path}")
        lines.append("")
        lines.append(f"- Exists: {summary.exists}")
        lines.append(f"- Valid ZIP: {summary.valid_zip}")
        lines.append(f"- File size: {human_bytes(summary.size_bytes)}")
        if summary.error:
            lines.append(f"- Error: `{summary.error}`")
        if summary.valid_zip:
            lines.append(f"- Entries: {summary.entry_count}")
            lines.append(f"- Compressed bytes in ZIP: {human_bytes(summary.compressed_bytes)}")
            lines.append(f"- Uncompressed bytes in ZIP: {human_bytes(summary.uncompressed_bytes)}")
            lines.append(f"- Top-level entries: `{summary.top_level}`")
            lines.append(f"- Suffix counts: `{summary.suffix_counts}`")
            if summary.kaggle:
                lines.append("")
                lines.append("### Kaggle-like content")
                lines.append(f"- Zarr entries: {summary.kaggle.get('zarr_entry_count')}")
                lines.append(f"- GEFF entries: {summary.kaggle.get('geff_entry_count')}")
                lines.append(f"- CSV entries: `{summary.kaggle.get('csv_entries')}`")
                lines.append(
                    f"- Sample submission headers: `{summary.kaggle.get('sample_submission_headers')}`"
                )
                lines.append(f"- Zarr datasets sample: `{summary.kaggle.get('zarr_datasets')}`")
                metadata = summary.kaggle.get("zarr_metadata_sample")
                if metadata:
                    lines.append("- Zarr metadata sample:")
                    lines.append("```json")
                    lines.append(json.dumps(metadata, indent=2)[:8000])
                    lines.append("```")
            if summary.ctc:
                lines.append("")
                lines.append("### CTC-like content")
                lines.append(f"- TIFF entries: {summary.ctc.get('tif_entry_count')}")
                lines.append(f"- Raw TIFF entries: {summary.ctc.get('raw_tif_entry_count')}")
                lines.append(f"- Mask TIFF entries: {summary.ctc.get('mask_tif_entry_count')}")
                lines.append(f"- Raw TIFFs by sequence: `{summary.ctc.get('raw_tifs_by_sequence')}`")
                lines.append("- Track files:")
                lines.append("```json")
                lines.append(json.dumps(summary.ctc.get("track_files"), indent=2)[:8000])
                lines.append("```")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    summaries = [inspect_archive(path) for path in args.archives]
    payload = [asdict(summary) for summary in summaries]

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_output:
        write_markdown(summaries, args.markdown_output)
    if not args.json_output and not args.markdown_output:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


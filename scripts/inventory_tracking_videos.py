#!/usr/bin/env python3
"""Build or verify the canonical inventory of rendered tracking MP4s."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/g04/TRACKING_VIDEO_INVENTORY.json"
CATALOGUES = (
    (
        "track_visualizer_v1",
        ROOT / "reports/g04/track_visualizer_v1",
        "complete_first_render",
        "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments",
        "selected_graph_appearance_direct_cv57_v1",
        False,
    ),
    (
        "post_scoring_full_catalog_20260804",
        ROOT / "reports/g04/post_scoring_visual_review/full_catalog_20260804",
        "canonical_complete_review_catalogue",
        "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments",
        "selected_graph_appearance_direct_cv57_v1",
        False,
    ),
    (
        "post_scoring_latest",
        ROOT / "reports/g04/post_scoring_visual_review/latest",
        "curated_nine_video_error_review",
        "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments",
        "selected_graph_appearance_direct_cv57_v1",
        False,
    ),
    (
        "v32_exact_visual_review_20260814",
        ROOT / "reports/g04/v32_exact_visual_review_20260814",
        "canonical_exact_frozen_v32_review_catalogue",
        "reports/g04/v32_exact_visual_source_20260814/remote_output/submission.csv",
        "public_temporalunet_v32_exact_frozen_output",
        False,
    ),
)
CATALOGUE_MANIFEST_SCHEMAS = {
    "track_visualizer_v1": "celltrack-alpha.track-visualizer-bundle.v1",
    "post_scoring_full_catalog_20260804": "celltrack-alpha.post-scoring-visual-review.v2",
    "post_scoring_latest": "celltrack-alpha.post-scoring-visual-review.v2",
    "v32_exact_visual_review_20260814": "celltrack-alpha.post-scoring-visual-review.v2",
}
SIDECAR_SCHEMA = "celltrack-alpha.track-visualizer-video.v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*:[A-Za-z0-9][A-Za-z0-9_-]*")
INDEX_VIDEO = re.compile(r'''(?:href|src)=["'](videos/[^"']+\.mp4)["']''')
CANDIDATE_CLASSIFICATION = {
    "selected_graph_appearance_direct_cv57_v1": {
        "local_execution": False,
        "prediction_sources": {
            "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments"
        },
    },
    "public_temporalunet_v32_exact_frozen_output": {
        "local_execution": False,
        "prediction_sources": {
            "reports/g04/v32_exact_visual_source_20260814/remote_output/submission.csv"
        },
    },
}


class InventoryError(RuntimeError):
    """The physical video catalogue and its provenance records disagree."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InventoryError(f"{path} must contain a JSON object")
    return value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise InventoryError(f"path escapes repository root: {path}") from exc


def _sha_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise InventoryError(f"{label} must be a lowercase SHA-256")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTITY.fullmatch(value) is None:
        raise InventoryError(
            f"{label} must be split:dataset with nonempty alphanumeric/_/- tokens"
        )
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise InventoryError(f"{label} must be a nonempty non-symbolic-link regular file")
    relative(path)
    return path


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise InventoryError("ffprobe is required to authenticate tracking videos")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=codec_type,nb_frames,nb_read_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InventoryError(f"{path}: ffprobe could not run: {exc}") from exc
    if result.returncode != 0:
        raise InventoryError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InventoryError(f"{path}: malformed ffprobe output") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise InventoryError(f"{path}: no video stream")
    frame_counts: list[int] = []
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            frame_counts.append(
                int(stream.get("nb_read_frames", stream.get("nb_frames")))
            )
        except (TypeError, ValueError):
            continue
    if not frame_counts or max(frame_counts) <= 0:
        raise InventoryError(f"{path}: zero decodable video frames")
    try:
        duration = float(payload.get("format", {}).get("duration"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise InventoryError(f"{path}: ffprobe duration is malformed") from exc
    if duration <= 0:
        raise InventoryError(f"{path}: ffprobe duration must be positive")
    return {"frames": max(frame_counts), "duration_seconds": duration}


def _prediction_source(
    manifest: dict[str, Any], expected_prediction_source: str
) -> tuple[str, str]:
    predictions = manifest.get("sources", {}).get("predictions")
    if not isinstance(predictions, dict):
        raise InventoryError("manifest lacks sources.predictions")
    raw_path = predictions.get("path")
    fragment_set_sha256 = _sha_field(
        predictions.get("fragment_set_sha256"), "prediction fragment-set hash"
    )
    if not isinstance(raw_path, str):
        raise InventoryError("prediction source path/hash is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if relative(path) != expected_prediction_source:
        raise InventoryError(f"unexpected prediction source: {path}")
    if path.is_symlink() or not path.exists():
        raise InventoryError(f"prediction source is absent or symbolic: {path}")
    return relative(path), fragment_set_sha256


def _fragment_bindings(
    *,
    name: str,
    manifest: dict[str, Any],
    expected_identities: set[str],
) -> tuple[dict[str, tuple[str, str]], str]:
    predictions = manifest.get("sources", {}).get("predictions", {})
    fragments = predictions.get("fragments")
    if not isinstance(fragments, list):
        raise InventoryError(f"{name}: manifest lacks prediction fragments")
    if predictions.get("fragment_count") != len(fragments):
        raise InventoryError(f"{name}: prediction fragment count drift")
    bindings: dict[str, tuple[str, str]] = {}
    digest = hashlib.sha256()
    for index, item in enumerate(fragments):
        if not isinstance(item, dict):
            raise InventoryError(f"{name}: prediction fragment {index} is malformed")
        identity = _identity(
            item.get("identity"), f"{name}: prediction fragment {index} identity"
        )
        raw_path = item.get("path")
        expected_hash = _sha_field(
            item.get("sha256"), f"{name}: prediction fragment {index} hash"
        )
        if not isinstance(raw_path, str):
            raise InventoryError(f"{name}: prediction fragment {index} binding is malformed")
        if identity in bindings:
            raise InventoryError(f"{name}: duplicate prediction fragment identity {identity}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        _regular(path, f"{name}: prediction fragment {index}")
        if item.get("bytes") != path.stat().st_size or sha256(path) != expected_hash:
            raise InventoryError(f"{name}: prediction fragment {index} byte/hash drift")
        normalized = relative(path)
        bindings[identity] = (normalized, expected_hash)
        digest.update(f"{identity}\0{expected_hash}\n".encode("utf-8"))
    if set(bindings) != expected_identities:
        raise InventoryError(f"{name}: prediction fragment identity set drift")
    observed_set = digest.hexdigest()
    expected_set = _sha_field(
        predictions.get("fragment_set_sha256"), f"{name}: prediction fragment-set hash"
    )
    if observed_set != expected_set:
        raise InventoryError(f"{name}: prediction fragment-set hash drift")
    return bindings, observed_set


def catalogue_record(
    name: str,
    directory: Path,
    role: str,
    expected_prediction_source: str,
    candidate: str,
    local_execution: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classification = CANDIDATE_CLASSIFICATION.get(candidate)
    if not isinstance(classification, dict):
        raise InventoryError(f"{name}: candidate classification is not centrally registered")
    if classification.get("local_execution") is not local_execution:
        raise InventoryError(f"{name}: local-execution classification drift")
    registered_sources = classification.get("prediction_sources")
    if not isinstance(registered_sources, set) or expected_prediction_source not in registered_sources:
        raise InventoryError(f"{name}: prediction source classification drift")
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    expected_schema = CATALOGUE_MANIFEST_SCHEMAS.get(name)
    if expected_schema is None or manifest.get("schema") != expected_schema:
        raise InventoryError(f"{name}: unrecognized or drifted manifest schema")
    if manifest.get("status") != "complete":
        raise InventoryError(f"{name}: manifest status is not complete")
    prediction_source, fragment_set_sha256 = _prediction_source(
        manifest, expected_prediction_source
    )
    video_dir = directory / "videos"
    metadata_dir = directory / "metadata"
    videos = sorted(video_dir.glob("*.mp4"))
    if len(videos) != int(manifest.get("rendered_video_count", -1)):
        raise InventoryError(f"{name}: rendered video count disagrees with manifest")
    if len(videos) != int(manifest.get("indexed_video_count", -1)):
        raise InventoryError(f"{name}: indexed video count disagrees with manifest")
    manifest_videos = manifest.get("videos")
    if not isinstance(manifest_videos, list):
        raise InventoryError(f"{name}: manifest videos are malformed")
    expected_video_paths: set[str] = set()
    expected_metadata_paths: set[str] = set()
    expected_identities: set[str] = set()
    for item in manifest_videos:
        if not isinstance(item, dict) or item.get("rendered") is not True:
            raise InventoryError(f"{name}: manifest video record is malformed")
        identity = _identity(item.get("identity"), f"{name}: manifest video identity")
        video_path = item.get("video")
        metadata_path_value = item.get("metadata")
        if not all(isinstance(value, str) for value in (video_path, metadata_path_value)):
            raise InventoryError(f"{name}: manifest video binding is malformed")
        expected_identities.add(identity)
        expected_video_paths.add(video_path)
        expected_metadata_paths.add(metadata_path_value)
    physical_video_paths = {f"videos/{path.name}" for path in videos}
    metadata_files = sorted(metadata_dir.glob("*.json"))
    physical_metadata_paths = {f"metadata/{path.name}" for path in metadata_files}
    if (
        len(expected_identities) != len(manifest_videos)
        or expected_video_paths != physical_video_paths
        or expected_metadata_paths != physical_metadata_paths
    ):
        raise InventoryError(f"{name}: manifest/video/sidecar exact-set drift")
    all_metadata_entries = {path.resolve() for path in metadata_dir.iterdir()}
    if all_metadata_entries != {path.resolve() for path in metadata_files}:
        raise InventoryError(f"{name}: metadata directory contains undeclared entries")
    fragment_bindings, observed_fragment_set = _fragment_bindings(
        name=name, manifest=manifest, expected_identities=expected_identities
    )
    if observed_fragment_set != fragment_set_sha256:
        raise InventoryError(f"{name}: prediction fragment-set binding drift")
    index_path = _regular(directory / "index.html", f"{name}: index")
    index_videos = set(INDEX_VIDEO.findall(index_path.read_text(encoding="utf-8")))
    if index_videos != physical_video_paths:
        raise InventoryError(f"{name}: index/video exact-set drift")

    records: list[dict[str, Any]] = []
    for video in videos:
        _regular(video, f"{video}: MP4")
        media = probe_video(video)
        sidecar_path = metadata_dir / f"{video.stem}.json"
        sidecar = load_json(sidecar_path)
        if sidecar.get("schema") != SIDECAR_SCHEMA:
            raise InventoryError(f"{sidecar_path}: sidecar schema drift")
        video_sha256 = sha256(video)
        if sidecar.get("video") != f"videos/{video.name}":
            raise InventoryError(f"{sidecar_path}: video path drift")
        if sidecar.get("video_sha256") != video_sha256:
            raise InventoryError(f"{sidecar_path}: MP4 hash drift")
        if sidecar.get("video_bytes") != video.stat().st_size:
            raise InventoryError(f"{sidecar_path}: MP4 byte-count drift")
        identity = _identity(sidecar.get("identity"), f"{sidecar_path}: identity")
        if identity.replace(":", "__") != video.stem:
            raise InventoryError(f"{sidecar_path}: identity/filename drift")
        if identity not in expected_identities:
            raise InventoryError(f"{sidecar_path}: identity absent from manifest video set")
        if sidecar.get("frames") != media["frames"]:
            raise InventoryError(f"{sidecar_path}: ffprobe frame-count drift")
        declared_duration = sidecar.get("encoded_duration_seconds")
        if declared_duration is not None:
            try:
                duration_error = abs(float(declared_duration) - media["duration_seconds"])
            except (TypeError, ValueError) as exc:
                raise InventoryError(f"{sidecar_path}: encoded duration is malformed") from exc
            if duration_error > max(0.05, media["duration_seconds"] * 0.01):
                raise InventoryError(f"{sidecar_path}: ffprobe duration drift")
        prediction_path = sidecar.get("prediction_fragment")
        prediction_sha256 = sidecar.get("prediction_fragment_sha256")
        if not isinstance(prediction_path, str) or not isinstance(prediction_sha256, str):
            raise InventoryError(f"{sidecar_path}: prediction binding is malformed")
        bound_prediction = Path(prediction_path)
        if not bound_prediction.is_absolute():
            bound_prediction = ROOT / bound_prediction
        if not bound_prediction.is_file() or sha256(bound_prediction) != prediction_sha256:
            raise InventoryError(f"{sidecar_path}: prediction fragment hash drift")
        manifest_path_value, manifest_hash = fragment_bindings.get(identity, (None, None))
        if manifest_path_value is None:
            raise InventoryError(f"{sidecar_path}: fragment absent from manifest")
        manifest_prediction = Path(str(manifest_path_value))
        if not manifest_prediction.is_absolute():
            manifest_prediction = ROOT / manifest_prediction
        if manifest_prediction.resolve() != bound_prediction.resolve() or manifest_hash != prediction_sha256:
            raise InventoryError(f"{sidecar_path}: prediction/manifest binding drift")
        records.append(
            {
                "catalogue": name,
                "candidate": candidate,
                "local_execution": local_execution,
                "identity": identity,
                "family": identity.split(":", 1)[1].split("_", 1)[0],
                "path": relative(video),
                "bytes": video.stat().st_size,
                "sha256": video_sha256,
                "sidecar_path": relative(sidecar_path),
                "sidecar_sha256": sha256(sidecar_path),
                "prediction_fragment_path": relative(bound_prediction),
                "prediction_fragment_sha256": prediction_sha256,
                "frames": sidecar.get("frames"),
                "encoded_duration_seconds": sidecar.get("encoded_duration_seconds"),
                "score": sidecar.get("score"),
                "rank": sidecar.get("rank"),
            }
        )

    catalogue = {
        "name": name,
        "role": role,
        "candidate": candidate,
        "local_execution": local_execution,
        "directory": relative(directory),
        "index_path": relative(index_path),
        "manifest_path": relative(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_schema": manifest.get("schema"),
        "generated_at": manifest.get("generated_at"),
        "physical_video_count": len(videos),
        "identity_count": len({record["identity"] for record in records}),
        "family_counts": {
            family: sum(record["family"] == family for record in records)
            for family in sorted({record["family"] for record in records})
        },
        "prediction_source": prediction_source,
        "prediction_fragment_set_sha256": fragment_set_sha256,
    }
    return catalogue, records


def build_inventory() -> dict[str, Any]:
    catalogues: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    registered_roots = {directory.resolve() for _, directory, _, _, _, _ in CATALOGUES}
    for (
        name,
        directory,
        role,
        expected_prediction_source,
        candidate,
        local_execution,
    ) in CATALOGUES:
        catalogue, catalogue_videos = catalogue_record(
            name,
            directory,
            role,
            expected_prediction_source,
            candidate,
            local_execution,
        )
        catalogues.append(catalogue)
        records.extend(catalogue_videos)

    discovered = {
        path.resolve()
        for path in (ROOT / "reports/g04").rglob("*.mp4")
        if path.is_file()
    }
    registered = {Path(record["path"]).joinpath() for record in records}
    registered = {(ROOT / path).resolve() for path in registered}
    if discovered != registered:
        missing = sorted(relative(path) for path in discovered - registered)
        stale = sorted(relative(path) for path in registered - discovered)
        raise InventoryError(f"unregistered={missing}; stale={stale}")
    if not registered_roots:
        raise InventoryError("no catalogues registered")

    hashes: dict[str, list[str]] = {}
    for record in records:
        hashes.setdefault(str(record["sha256"]), []).append(str(record["path"]))
    duplicate_groups = [
        {"sha256": digest, "copies": sorted(paths)}
        for digest, paths in sorted(hashes.items())
        if len(paths) > 1
    ]
    identities = sorted({str(record["identity"]) for record in records})
    family_counts = {
        family: sum(identity.split(":", 1)[1].startswith(f"{family}_") for identity in identities)
        for family in sorted({identity.split(":", 1)[1].split("_", 1)[0] for identity in identities})
    }
    candidate_sources: dict[str, dict[str, Any]] = {}
    for catalogue in catalogues:
        candidate = str(catalogue["candidate"])
        summary = candidate_sources.setdefault(
            candidate,
            {
                "prediction_sources": [],
                "physical_video_count": 0,
                "local_execution": bool(catalogue["local_execution"]),
            },
        )
        if catalogue["prediction_source"] not in summary["prediction_sources"]:
            summary["prediction_sources"].append(catalogue["prediction_source"])
        summary["physical_video_count"] += int(catalogue["physical_video_count"])
        if bool(catalogue["local_execution"]) != summary["local_execution"]:
            raise InventoryError(f"candidate {candidate}: mixed local-execution classification")
    return {
        "schema": "celltrack-alpha.tracking-video-inventory.v2",
        "status": "PASS",
        "scope": "reports/g04/**/*.mp4",
        "physical_video_count": len(records),
        "distinct_identity_count": len(identities),
        "distinct_candidate_identity_count": len(
            {(str(record["candidate"]), str(record["identity"])) for record in records}
        ),
        "distinct_content_hash_count": len(hashes),
        "family_counts": family_counts,
        "catalogues": catalogues,
        "duplicate_content_groups": duplicate_groups,
        "source_classification": {
            "candidates": candidate_sources,
            "contains_public_temporalunet_v32_0_911": True,
            "warning": "The exact V32 catalogue diagnoses a recovered historical Kaggle output; it does not certify local execution of V32 or any changed candidate.",
        },
        "videos": records,
    }


def canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    group = value.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inventory = build_inventory()
    rendered = canonical(inventory)
    if args.write:
        REPORT.write_text(rendered, encoding="utf-8")
    elif args.check:
        if not REPORT.is_file() or REPORT.read_text(encoding="utf-8") != rendered:
            raise InventoryError(f"inventory report is stale: {REPORT}")
    else:
        print(rendered, end="")
    print(
        json.dumps(
            {
                "status": "PASS",
                "physical_video_count": inventory["physical_video_count"],
                "distinct_identity_count": inventory["distinct_identity_count"],
                "distinct_content_hash_count": inventory["distinct_content_hash_count"],
                "report": str(REPORT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

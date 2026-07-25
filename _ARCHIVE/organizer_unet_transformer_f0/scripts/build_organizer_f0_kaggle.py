#!/usr/bin/env python3
"""Build a private Kaggle package for one resumable organiser-model stage."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "kaggle-cell-tracking-competition"
SPLIT = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"
TRAINER = ROOT / "scripts" / "train_organizer_f0.py"
OUTPUT_ROOT = ROOT / "kaggle_kernel" / "organizer_f0"
SUPPORT_DATASET = "pilkwang/biohub-tracking-support-pack-50ep-v1"
COMPETITION = "biohub-cell-tracking-during-development"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_source(source: Path) -> str:
    """Return a deterministic zip payload suitable for Kaggle's single code file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def launcher(stage: int, previous_slug: str | None, source_payload: str) -> str:
    resume_block = "resume = None"
    resume_args = ""
    if stage > 0:
        resume_block = f'''candidates = []
for manifest in Path("/kaggle/input").rglob("run.json"):
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        continue
    if value.get("schema") == "celltrack-alpha.organizer-unet-transformer-f0.v1":
        checkpoint = manifest.with_name("checkpoint_last.pth")
        if checkpoint.is_file():
            candidates.append(checkpoint)
if len(candidates) != 1:
    raise RuntimeError("expected exactly one validated prior-stage checkpoint, got %d" % len(candidates))
resume = candidates[0]'''
        resume_args = ', "--resume-checkpoint", str(resume), "--require-resume"'

    return f'''#!/usr/bin/env python3
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path("/kaggle/working/organizer-src")
if ROOT.exists():
    shutil.rmtree(ROOT)
SOURCE_PAYLOAD = """{source_payload}"""
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_PAYLOAD)), mode="r") as archive:
    root = ROOT.resolve()
    for member in archive.infolist():
        destination = (ROOT / member.filename).resolve()
        if destination != root and root not in destination.parents:
            raise RuntimeError("embedded source archive contains an unsafe path")
    archive.extractall(ROOT)

wheel_roots = [path for path in Path("/kaggle/input").rglob("wheels") if path.is_dir()]
if len(wheel_roots) != 1:
    raise RuntimeError("expected one support wheelhouse, got %d" % len(wheel_roots))
subprocess.run([
    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--no-index", "--find-links", str(wheel_roots[0]),
    "tracksdata", "geff", "zarr", "polars", "tqdm",
], check=True)

{resume_block}
cmd = [
    sys.executable, str(ROOT / "scripts" / "train_organizer_f0.py"),
    "--data-dir", "/kaggle/input/competitions/biohub-cell-tracking-during-development/train",
    "--split-manifest", str(ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"),
    "--output-dir", "/kaggle/working/organizer_f0",
    "--target-epochs", "500", "--min-epochs", "300", "--patience-epochs", "40",
    "--eval-every", "5", "--min-delta", "0.0001", "--monitor-videos", "12",
    "--lr", "0.0001", "--batch-size", "8", "--num-workers", "4",
    "--seed", "17", "--max-hours", "7.5", "--epoch-safety-minutes", "100"
    {resume_args}
]
print(json.dumps({{"stage": {stage}, "previous_slug": {previous_slug!r}, "resume": str(resume) if resume else None}}, sort_keys=True), flush=True)
environment = dict(os.environ)
environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
raise SystemExit(subprocess.run(cmd, cwd=ROOT, env=environment).returncode)
'''


def build(stage: int, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if stage < 0:
        raise ValueError("stage must be nonnegative")
    slug = f"org-f0-{stage:02d}"
    previous_slug = f"org-f0-{stage - 1:02d}" if stage else None
    output = output_root / f"stage{stage:02d}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source = output / "source"
    (source / "external").mkdir(parents=True)
    shutil.copytree(
        UPSTREAM,
        source / "external" / "kaggle-cell-tracking-competition",
        ignore=shutil.ignore_patterns(".git", "assets", "tests", "visualize", "__pycache__", "*.pyc"),
    )
    (source / "scripts").mkdir(parents=True)
    shutil.copy2(TRAINER, source / "scripts" / TRAINER.name)
    split_target = source / "reports" / "g02" / SPLIT.name
    split_target.parent.mkdir(parents=True)
    shutil.copy2(SPLIT, split_target)

    launch_path = output / "launch.py"
    launch_source = launcher(stage, previous_slug, embedded_source(source))
    compile(launch_source, str(launch_path), "exec")
    launch_path.write_text(launch_source, encoding="ascii")
    metadata = {
        "id": f"onebitbrain/{slug}",
        "title": slug,
        "code_file": "launch.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [SUPPORT_DATASET],
    }
    if previous_slug is not None:
        metadata["kernel_sources"] = [f"onebitbrain/{previous_slug}"]
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    core = {
        "schema": "celltrack-alpha.organizer-f0-kaggle-package.v1",
        "stage": stage,
        "kernel_ref": f"onebitbrain/{slug}",
        "previous_kernel_ref": f"onebitbrain/{previous_slug}" if previous_slug else None,
        "private": True,
        "internet": False,
        "gpu": "T4x2-requested",
        "holdout_evaluation": False,
        "fit_only_training": True,
        "files": files,
    }
    report = {**core, "identity_sha256": hashlib.sha256(canonical_bytes(core)).hexdigest()}
    (output / "build.json").write_bytes(canonical_bytes(report) + b"\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args.stage, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

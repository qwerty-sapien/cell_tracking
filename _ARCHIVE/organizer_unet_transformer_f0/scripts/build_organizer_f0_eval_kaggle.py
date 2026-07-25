#!/usr/bin/env python3
"""Build the private, post-convergence organiser CV57 evaluation kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "kaggle-cell-tracking-competition"
TRAINER = ROOT / "scripts" / "train_organizer_f0.py"
EVALUATOR = ROOT / "scripts" / "evaluate_organizer_f0.py"
SPLIT = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"
PARITY = ROOT / "reports" / "g02" / "scorer_parity_v1.json"
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


def launcher() -> str:
    return '''#!/usr/bin/env python3
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/kaggle/working/organizer-eval-src")
SOURCE = Path(__file__).with_name("source")
if ROOT.exists():
    shutil.rmtree(ROOT)
shutil.copytree(SOURCE, ROOT)

wheel_roots = [path for path in Path("/kaggle/input").rglob("wheels") if path.is_dir()]
if len(wheel_roots) != 1:
    raise RuntimeError("expected one support wheelhouse, got %d" % len(wheel_roots))
subprocess.run([
    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--no-index", "--find-links", str(wheel_roots[0]),
    "tracksdata", "geff", "zarr", "polars", "tqdm",
], check=True)

candidates = []
for manifest in Path("/kaggle/input").rglob("run.json"):
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        continue
    if value.get("schema") == "celltrack-alpha.organizer-unet-transformer-f0.v1" and value.get("status") == "converged":
        candidates.append(manifest.parent)
if len(candidates) != 1:
    raise RuntimeError("expected exactly one converged organiser training output, got %d" % len(candidates))

cmd = [
    sys.executable, str(ROOT / "scripts" / "evaluate_organizer_f0.py"),
    "--data-dir", "/kaggle/input/competitions/biohub-cell-tracking-during-development/train",
    "--training-dir", str(candidates[0]),
    "--parity-ledger", str(ROOT / "reports" / "g02" / "scorer_parity_v1.json"),
    "--output-dir", "/kaggle/working/organizer_f0_cv57",
    "--unet-batch-size", "4",
]
raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)
'''


def build(output: Path, training_kernel_ref: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not training_kernel_ref.startswith("onebitbrain/org-f0-"):
        raise ValueError("training kernel ref is not an organiser fold-0 stage")
    source = output / "source"
    (source / "external").mkdir(parents=True)
    shutil.copytree(
        UPSTREAM,
        source / "external" / "kaggle-cell-tracking-competition",
        ignore=shutil.ignore_patterns(".git", "assets", "tests", "visualize", "__pycache__", "*.pyc"),
    )
    (source / "scripts").mkdir(parents=True)
    shutil.copy2(TRAINER, source / "scripts" / TRAINER.name)
    shutil.copy2(EVALUATOR, source / "scripts" / EVALUATOR.name)
    reports = source / "reports" / "g02"
    reports.mkdir(parents=True)
    shutil.copy2(SPLIT, reports / SPLIT.name)
    shutil.copy2(PARITY, reports / PARITY.name)

    launch_source = launcher()
    compile(launch_source, "launch.py", "exec")
    (output / "launch.py").write_text(launch_source, encoding="ascii")
    metadata = {
        "id": "onebitbrain/org-f0-cv57",
        "title": "org-f0-cv57",
        "code_file": "launch.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [SUPPORT_DATASET],
        "kernel_sources": [training_kernel_ref],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    files = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in output.rglob("*") if item.is_file())
    ]
    core = {
        "schema": "celltrack-alpha.organizer-f0-cv57-kaggle-package.v1",
        "kernel_ref": "onebitbrain/org-f0-cv57",
        "training_kernel_ref": training_kernel_ref,
        "private": True,
        "internet": False,
        "selection": "frozen_fold0_holdout57",
        "patched_scorer_required": True,
        "production_merge": False,
        "files": files,
    }
    report = {**core, "identity_sha256": hashlib.sha256(canonical_bytes(core)).hexdigest()}
    (output / "build.json").write_bytes(canonical_bytes(report) + b"\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-kernel-ref", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args.output, args.training_kernel_ref), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

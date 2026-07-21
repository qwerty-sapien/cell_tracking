#!/usr/bin/env python3
"""Run one hash-bound, private V45 D4-teacher student fit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
OUTPUT = WORK / "v45fout"
CONFIG = HERE / "v45fcfg.json"


class FastLaunchError(RuntimeError):
    pass


def _canon(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise FastLaunchError("invalid fast config") from exc
    if not isinstance(value, dict) or _canon(value) != Path(path).read_bytes().rstrip(b"\n"):
        raise FastLaunchError("fast config is not canonical")
    return value


def main():
    started = time.monotonic()
    diagnostic = {"schema": "v45f-private-run.v1", "network": "forbidden", "ok": False}
    try:
        bh45.MANIFEST = HERE / "v45fman.json"
        bh45.ATTACHMENT = HERE / "v45fatt.json"
        bh45.SOURCE_ARCHIVE = HERE / "v45fsrc.zip"
        bh45.SOURCE_ROOT = WORK / "v45fsrc"
        bh45.DIAGNOSTIC = WORK / "v45fdiag.json"
        package = bh45._verify_package()
        config = _load(CONFIG)
        diagnostic["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diagnostic)
        diagnostic["dependencies_final"] = bh45._install_missing(diagnostic["dependencies_before"], wheels, diagnostic)
        bh45._extract_source(package)
        competition = bh45._competition_root()
        train_root = competition / "train"
        if not train_root.is_dir():
            raise FastLaunchError("private training input lacks a train directory")
        environment = dict(os.environ)
        python_paths = [str(bh45.SOURCE_ROOT / "src"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "train_v45f.py"),
            "--fold", str(config["fold"]), "--fraction", str(config["fraction"]),
            "--lr", str(config["lr"]), "--seed", str(config["seed"]),
            "--epochs", str(config["epochs"]), "--max-steps", str(config["max_steps"]),
            "--maximum-hours", str(config["maximum_hours"]), "--maximum-charge", str(config["maximum_charge"]),
            "--support-root", str(support), "--data-root", str(train_root),
            "--split-manifest", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance-manifest", str(bh45.SOURCE_ROOT / "prov.json"),
            "--record-projection", str(bh45.SOURCE_ROOT / "fitrec.json"),
            "--output-dir", str(OUTPUT), "--armed",
        ]
        deadline = started + float(config["maximum_hours"]) * 3600.0 + 300.0
        diagnostic["runner"] = {"command": command, "deadline_seconds": deadline - started, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(command, environment, deadline)
        metrics = OUTPUT / "metrics.json"
        checkpoint = OUTPUT / "checkpoint.pt"
        if not metrics.is_file() or not checkpoint.is_file():
            raise FastLaunchError("bounded student fit did not emit metrics and checkpoint")
        result = _load(metrics)
        if result.get("completed_steps", 0) < 1 or result.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise FastLaunchError("bounded student artifact identity is invalid")
        diagnostic.update({"ok": True, "elapsed_seconds": time.monotonic() - started, "completed_steps": result["completed_steps"]})
        bh45._write_diagnostic(diagnostic)
        return 0
    except Exception as exc:
        diagnostic.update({"error": "%s: %s" % (type(exc).__name__, exc), "elapsed_seconds": time.monotonic() - started})
        bh45._write_diagnostic(diagnostic)
        print(diagnostic["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

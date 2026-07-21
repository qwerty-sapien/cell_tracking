#!/usr/bin/env python3
"""Run a bounded private parent fit followed by sparse edge calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import e4bh as bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
PARENT = WORK / "e4par"
EDGE = WORK / "e4out"
CONFIG = HERE / "e4cfg.json"


class E4LaunchError(RuntimeError):
    pass


def _canon(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise E4LaunchError("invalid private config") from exc
    if not isinstance(value, dict) or _canon(value) != Path(path).read_bytes().rstrip(b"\n"):
        raise E4LaunchError("private config is not canonical")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    diagnostic = {"schema": "edgefit-private-run.v1", "network": "forbidden", "ok": False}
    try:
        bh45.MANIFEST = HERE / "e4man.json"
        bh45.ATTACHMENT = HERE / "e4att.json"
        bh45.SOURCE_ARCHIVE = HERE / "e4src.zip"
        bh45.SOURCE_ROOT = WORK / "e4src"
        bh45.DIAGNOSTIC = WORK / "e4diag.json"
        package = bh45._verify_package()
        config = _load(CONFIG)
        if config.get("accelerator") != "NvidiaTeslaT4":
            raise E4LaunchError("private edge fit requires NvidiaTeslaT4")
        diagnostic["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diagnostic)
        diagnostic["dependencies_final"] = bh45._install_missing(diagnostic["dependencies_before"], wheels, diagnostic)
        bh45._extract_source(package)
        competition = bh45._competition_root()
        if not (competition / "train").is_dir():
            raise E4LaunchError("competition training labels are unavailable")
        environment = dict(os.environ)
        python_paths = [str(bh45.SOURCE_ROOT / "src"), str(bh45.SOURCE_ROOT / "scripts"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        deadline = started + float(config["maximum_hours"]) * 3600.0 + 300.0
        parent_command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "train_v45f.py"),
            "--fold", str(config["fold"]), "--fraction", str(config["fraction"]),
            "--lr", str(config["lr"]), "--seed", str(config["seed"]),
            "--epochs", "1", "--max-steps", str(config["max_steps"]),
            "--maximum-hours", str(config["maximum_hours"]), "--maximum-charge", str(config["maximum_charge"]),
            "--support-root", str(support), "--data-root", str(competition / "train"),
            "--split-manifest", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance-manifest", str(bh45.SOURCE_ROOT / "prov.json"),
            "--record-projection", str(bh45.SOURCE_ROOT / "fitrec.json"),
            "--output-dir", str(PARENT), "--armed",
        ]
        diagnostic["parent"] = {"command": parent_command, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(parent_command, environment, deadline)
        checkpoint, metrics = PARENT / "checkpoint.pt", PARENT / "metrics.json"
        if not checkpoint.is_file() or not metrics.is_file():
            raise E4LaunchError("parent fit omitted checkpoint or metrics")
        parent_metrics = _load(metrics)
        if parent_metrics.get("status") != "completed" or parent_metrics.get("completed_steps", 0) < 1:
            raise E4LaunchError("parent fit was not completed")
        edge_command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "edgefit.py"),
            "--input", str(competition), "--out-dir", str(EDGE), "--fold", str(config["fold"]),
            "--device", "cuda", "--support-root", str(support),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", _sha(checkpoint),
            "--metrics", str(metrics), "--metrics-sha256", _sha(metrics),
            "--registry", str(bh45.SOURCE_ROOT / "reg.json"),
            "--split", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance", str(bh45.SOURCE_ROOT / "prov.json"),
            "--val-groups", str(config["edge_val_groups"]),
            "--max-steps", str(config["edge_max_steps"]), "--seed", str(config["seed"]),
        ]
        diagnostic["edge"] = {"command": edge_command, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(edge_command, environment, deadline)
        edge_report, edge_artifact = EDGE / "report.json", EDGE / "edge.json"
        if not edge_report.is_file() or not edge_artifact.is_file():
            raise E4LaunchError("edge fit omitted report or edge artifact")
        edge_metrics = _load(edge_report)
        if edge_metrics.get("status") != "completed":
            raise E4LaunchError("edge fit was not completed")
        diagnostic.update({
            "ok": True,
            "elapsed_seconds": time.monotonic() - started,
            "parent_checkpoint_sha256": _sha(checkpoint),
            "edge_artifact_sha256": _sha(edge_artifact),
        })
        bh45._write_diagnostic(diagnostic)
        return 0
    except Exception as exc:
        diagnostic.update({"error": "%s: %s" % (type(exc).__name__, exc), "elapsed_seconds": time.monotonic() - started})
        bh45._write_diagnostic(diagnostic)
        print(diagnostic["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Attest a completed private T-05 Kaggle output and build scorer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "reports" / "g02" / "t05-trial.json"
OUT = ROOT / "reports" / "g02"
DEST_CKPT = OUT / "t05v1.pt"
DEST_MET = OUT / "t05m1.json"
DEST_ART = OUT / "t05v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def exact_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or canonical(value) != path.read_bytes():
        raise ValueError(f"noncanonical object: {path}")
    return value


def source_files(root: Path) -> tuple[Path, Path | None]:
    hits = sorted(root.rglob("checkpoint.pt"))
    if len(hits) != 1:
        raise ValueError("Kaggle output must contain exactly one checkpoint.pt")
    checkpoint = hits[0]
    metrics = checkpoint.with_name("metrics.json")
    return checkpoint, metrics if metrics.is_file() else None


def expected_identity() -> dict[str, Any]:
    trial = exact_json(TRIAL)
    candidate = trial.get("candidate")
    immutable = trial.get("immutable_inputs")
    if not isinstance(candidate, Mapping) or not isinstance(immutable, Mapping):
        raise ValueError("T-05 trial schema is invalid")
    return {
        "fold": candidate["fold"], "scenario": candidate["scenario"], "unfreeze": candidate["unfreeze"],
        "lr": candidate["learning_rate"], "fraction": candidate["fit_fraction"], "seed": candidate["seed"],
        "detector_policy": candidate["detector_policy"], "center_weight": candidate["center_weight"],
        "teacher_weight": candidate["teacher_weight"], "center_supervision": candidate["center_supervision"],
        "center_loss": candidate["center_loss"], "split_sha256": immutable["split_sha256"],
        "provenance_sha256": immutable["provenance_sha256"], "registry_sha256": immutable["registry_sha256"],
        "registry_projection_sha256": immutable["registry_projection_sha256"], "weights_sha256": immutable["checkpoint_sha256"],
        "config_sha256": immutable["checkpoint_config_sha256"],
    }


def verify(metrics: Mapping[str, Any], checkpoint: Path) -> None:
    identity = metrics.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != expected_identity() | {"selected_sources": identity.get("selected_sources")}:
        raise ValueError("metrics identity does not match the attested T-05 candidate")
    selected = identity.get("selected_sources")
    if not isinstance(selected, list) or not selected:
        raise ValueError("metrics lacks selected fit-only sources")
    if metrics.get("status") != "completed" or not isinstance(metrics.get("completed_steps"), int) or metrics["completed_steps"] < 1:
        raise ValueError("T-05 did not complete at least one training step")
    if metrics.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("metrics checkpoint hash mismatch")
    if not isinstance(metrics.get("loss_components"), Mapping) or "teacher" not in metrics["loss_components"]:
        raise ValueError("T-05 metrics do not attest teacher distillation")


def reconstructed_metrics(checkpoint: Path) -> dict[str, Any]:
    """Recover scoring metadata only from the trusted weights-only checkpoint state.

    Kaggle occasionally exports the large checkpoint but omits adjacent JSON files.
    The reconstructed record deliberately identifies that fact and includes only
    values persisted in the completed checkpoint itself.
    """
    try:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("cannot weights-only load the completed T-05 checkpoint") from exc
    required = ("identity", "completed_steps", "elapsed_seconds", "loss_components", "source_groups", "source_video_ids", "source_records", "execution", "status")
    if not isinstance(state, Mapping) or any(key not in state for key in required):
        raise ValueError("checkpoint state cannot reconstruct required metrics")
    metrics = {
        "identity": state["identity"],
        "checkpoint_sha256": sha256(checkpoint),
        "completed_steps": state["completed_steps"],
        "elapsed_seconds": state["elapsed_seconds"],
        "loss_components": state["loss_components"],
        "source_groups": state["source_groups"],
        "source_video_ids": state["source_video_ids"],
        "source_records": state["source_records"],
        "windows": [],
        "per_step": [],
        "epoch_metrics": [],
        "source_window_stats": {},
        "execution": state["execution"],
        "status": state["status"],
        "metrics_provenance": {"kind": "checkpoint-state-reconstruction-v1", "remote_metrics_missing": True},
    }
    verify(metrics, checkpoint)
    return metrics


def copy_exact(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite a distinct artifact: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build(output: Path) -> dict[str, Any]:
    checkpoint, metrics_path = source_files(output)
    if metrics_path is None:
        metrics = reconstructed_metrics(checkpoint)
        metrics_data = canonical(metrics)
    else:
        metrics = exact_json(metrics_path)
        verify(metrics, checkpoint)
        metrics_data = metrics_path.read_bytes()
    copy_exact(checkpoint, DEST_CKPT)
    if DEST_MET.exists() and DEST_MET.read_bytes() != metrics_data:
        raise ValueError("refusing to overwrite a distinct T-05 metrics artifact")
    if not DEST_MET.exists():
        DEST_MET.write_bytes(metrics_data)
    descriptor = {
        "schema_version": 1,
        "artifact": "checkpoint-score-artifact-v1",
        "model_id": "t05v1",
        "checkpoint": {"file": DEST_CKPT.name, "sha256": sha256(DEST_CKPT), "bytes": DEST_CKPT.stat().st_size},
        "metrics": {"file": DEST_MET.name, "sha256": sha256(DEST_MET)},
        "fold": 0,
        "split_sha256": expected_identity()["split_sha256"],
    }
    payload = canonical(descriptor)
    if DEST_ART.exists() and DEST_ART.read_bytes() != payload:
        raise ValueError("refusing to overwrite a distinct T-05 scorer descriptor")
    DEST_ART.write_bytes(payload)
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory retrieved from Kaggle kernels output")
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

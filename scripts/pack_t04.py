#!/usr/bin/env python3
"""Build and verify the non-destructive, private T-04 training package.

The package is deliberately code-self-contained but runtime-input-dependent: the
released checkpoint and offline Zarr wheel support pack are separate Kaggle
inputs, not hidden notebook payloads. This keeps the notebook source below
Kaggle's 1 MB limit while failing closed if either input is absent or altered.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "kaggle_kernel" / "train_nb.py"
PYTHON = ROOT / "kaggle_kernel" / "t04.py"
NOTEBOOK = ROOT / "kaggle_kernel" / "t04.ipynb"
META = ROOT / "kaggle_kernel" / "t04.meta.json"
PACK = ROOT / "kaggle_kernel" / "t04.pack.json"
TRIAL = ROOT / "reports" / "g02" / "t04-trial.json"
TEST = ROOT / "tests" / "test_tnb.py"
HISTORICAL = {
    ROOT / "reports" / "g02" / "trial-v1.json": "01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2",
    ROOT / "reports" / "g02" / "trial-v3.json": "9dbf157b6e59e7358baece018d3942126a45f67f7399095dd888401b5ea79f79",
    ROOT / "reports" / "g02" / "trial-v4.json": "dcd94b091ed965f924e3c4ab915fafabf2b8951a9de0b4e78e5d43f71fbc8691",
}
MAX_CODE_BYTES = 1_000_000
EXPECTED_CANDIDATE = {
    "batch_size": 1, "center_loss": "report-only-excluded", "center_supervision": "annotated-only",
    "division_enabled": False, "epochs": 1, "fit_fraction": 0.1, "fold": 0,
    "learning_rate": 1e-5, "max_steps": 1000, "released_detector_frozen": True,
    "scenario": "universal", "seed": 17, "unfreeze": 0,
}


def canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} is not an object")
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def historical_ok() -> None:
    for path, expected in HISTORICAL.items():
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"historical artifact drift: {path.relative_to(ROOT)}")


def trial_bytes() -> tuple[dict[str, Any], bytes]:
    trial = read_json(TRIAL)
    if trial.get("artifact") != "t04-private-trial" or trial.get("status") != "package_not_launched":
        raise ValueError("T-04 trial is not a not-launched private record")
    if trial.get("candidate") != EXPECTED_CANDIDATE:
        raise ValueError("T-04 trial candidate is not the fixed sparse adapter candidate")
    authorization = trial.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("private_only") is not True or authorization.get("scored_submission") is not False:
        raise ValueError("T-04 trial authorization is not private training only")
    contract = trial.get("package_contract")
    if contract != {
        "code_package_under_bytes": MAX_CODE_BYTES,
        "released_checkpoint_input_required": True,
        "runtime_self_contained": False,
        "support_input_required": True,
    }:
        raise ValueError("T-04 code/runtime package boundary is invalid")
    return trial, canonical(trial)


def t04_source(trial: bytes) -> bytes:
    source = TEMPLATE.read_text(encoding="utf-8")
    if "sparse_negative_weight" in source or "negative_weight" in source:
        raise ValueError("template retains invalid sparse-negative training")
    if source.count("warn_only=False") != 1:
        raise ValueError("template must fail closed on nondeterministic algorithms")
    encoded = base64.b85encode(trial)
    source, replacements = re.subn(
        r"^TRIAL_BYTES = base64\.b85decode\(b'[^']*'\)$",
        "TRIAL_BYTES = base64.b85decode(" + repr(encoded) + ")",
        source,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError("template trial payload replacement is ambiguous")
    source, replacements = re.subn(r"^TRIAL_SHA256 = '[0-9a-f]{64}'$", f"TRIAL_SHA256 = '{digest(trial)}'", source, flags=re.MULTILINE)
    if replacements != 1:
        raise ValueError("template trial digest replacement is ambiguous")
    source, replacements = re.subn(r"/kaggle/working/t03", "/kaggle/working/t04", source)
    if replacements != 2:
        raise ValueError("T-04 output path replacement is incomplete")
    source, replacements = re.subn(r"private Kaggle working/t03", "private Kaggle working/t04", source)
    if replacements != 1:
        raise ValueError("T-04 output diagnostic replacement is incomplete")
    ast.parse(source)
    result = source.encode("utf-8")
    if len(result) >= MAX_CODE_BYTES:
        raise ValueError("T-04 source exceeds Kaggle's 1 MB cap")
    return result


def wheel_specs(source: str) -> list[list[Any]]:
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ZARR_WHEELS" for target in node.targets)]
    if len(matches) != 1:
        raise ValueError("T-04 source must define one wheel specification")
    value = ast.literal_eval(matches[0].value)
    if not isinstance(value, tuple) or len(value) != 9:
        raise ValueError("T-04 support wheel count is invalid")
    parsed = [list(item) for item in value]
    if not all(len(item) == 3 and isinstance(item[0], str) and isinstance(item[1], int) and isinstance(item[2], str) and re.fullmatch(r"[0-9a-f]{64}", item[2]) for item in parsed):
        raise ValueError("T-04 support wheel specification is invalid")
    return parsed


def notebook_bytes(source: bytes) -> bytes:
    value = {
        "cells": [{"cell_type": "code", "execution_count": None, "id": "t04-entry", "metadata": {}, "outputs": [],
                   "source": source.decode("utf-8").splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "private_only": True,
            "runtime_self_contained": False,
            "support_input_required": True,
            "t04_script_sha256": digest(source),
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    result = canonical(value)
    if b"".join(item.encode("utf-8") for item in value["cells"][0]["source"]) != source or len(result) >= MAX_CODE_BYTES:
        raise ValueError("T-04 notebook is not an exact compact source container")
    return result


def metadata(trial: Mapping[str, Any], trial_data: bytes, source: bytes) -> bytes:
    value = {
        "artifact": "t04-meta-v1", "candidate": trial["candidate"], "entrypoint": "t04.py", "notebook": "t04.ipynb",
        "output_root": "/kaggle/working/t04", "private_only": True, "public_effect": False,
        "network": False, "submission": False, "runtime_self_contained": False, "support_input_required": True,
        "source": {"bytes": len(source), "sha256": digest(source)},
        "trial": {"path": "reports/g02/t04-trial.json", "sha256": digest(trial_data)},
        "checkpoint_input": {
            "roots": [
                "/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts",
                "/kaggle/input/cellmot-baseline-artifacts",
            ],
            "config_sha256": trial["immutable_inputs"]["checkpoint_config_sha256"],
            "weight_sha256": trial["immutable_inputs"]["checkpoint_sha256"],
        },
        "support_input": {"roots": ["/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1", "/kaggle/input/biohub-tracking-support-pack-50ep-v1"],
                          "wheel_count": 9, "wheels": wheel_specs(source.decode("utf-8"))},
    }
    return canonical(value)


def package_bytes(source: bytes, notebook: bytes, meta: bytes, trial: bytes) -> bytes:
    files = {PYTHON: source, NOTEBOOK: notebook, META: meta, TRIAL: trial, TEST: TEST.read_bytes(), Path(__file__): Path(__file__).read_bytes()}
    value = {
        "artifact": "t04-pack-v1", "code_package_bytes": len(source) + len(notebook), "code_package_under_bytes": MAX_CODE_BYTES,
        "historical_hashes": {path.relative_to(ROOT).as_posix(): value for path, value in HISTORICAL.items()},
        "notebook": {"runtime_self_contained": False, "source_cell_sha256": digest(source), "support_input_required": True},
        "package_files": {path.relative_to(ROOT).as_posix(): {"bytes": len(data), "sha256": digest(data)} for path, data in sorted(files.items(), key=lambda item: item[0].as_posix())},
        "runtime": {"fp32_only": True, "network": False, "private_only": True, "submission": False,
                    "released_checkpoint_input_required": True, "support_input_required": True},
    }
    if value["code_package_bytes"] >= MAX_CODE_BYTES:
        raise ValueError("T-04 code package exceeds Kaggle's 1 MB cap")
    return canonical(value)


def build() -> dict[Path, bytes]:
    historical_ok()
    trial, trial_data = trial_bytes()
    source = t04_source(trial_data)
    notebook = notebook_bytes(source)
    meta = metadata(trial, trial_data, source)
    pack = package_bytes(source, notebook, meta, trial_data)
    return {TRIAL: trial_data, PYTHON: source, NOTEBOOK: notebook, META: meta, PACK: pack}


def check() -> None:
    expected = build()
    for path, data in expected.items():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"T-04 package drift: {path.relative_to(ROOT)}")
    package = read_json(PACK)
    for name, identity in package["package_files"].items():
        path = ROOT / name
        data = path.read_bytes()
        if identity != {"bytes": len(data), "sha256": digest(data)}:
            raise ValueError(f"T-04 listed package identity drift: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        for path, data in build().items():
            atomic_write(path, data)
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

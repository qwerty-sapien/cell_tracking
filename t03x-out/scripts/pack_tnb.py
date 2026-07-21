#!/usr/bin/env python3
"""Deterministically draft the bounded T-03 ipykernel argv remediation."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import sys
import tempfile
import types
import zlib
from pathlib import Path


DRAFT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = DRAFT_ROOT.parent
INPUT_HASHES = {
    "scripts/pack_tnb.py": "92a36d6e9ae456555001c3459f630331c95bfe1b42822dca628a03b5985908ba",
    "kaggle_kernel/train_nb.py": "b467134582bad37b05c57074f2c9db1fa0e2d8915bd8d5629b62c5b1e596ed76",
    "kaggle_kernel/train_nb.ipynb": "20e42c465f4f189f430290010781432494550c07aa25bb83ee5f098259265bcf",
    "kaggle_kernel/tr-pack.json": "c991526c2bfe1a22c14353f44705575b897ce14fece17b6bb133fd3ef91d4c9b",
    "kaggle_kernel/tr-meta.json": "65f72e905e1538f7aca5e6494f07307ffb6c9c9ce7ac0c6758ad941039bfd2c2",
    "tests/test_tnb.py": "2b6e6e3f477c6021fc9d60ef2fad9bbdb4369faa385a764b1b88e9c577e7fa25",
    "reports/g02/trial-v1.json": "01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2",
    "reports/g02/run-v1.json": "b3f3a2da73e05bcfb49f6c8693960f4258d1ee8a0999234e68eca73dc959854e",
}
FULL_REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
ERROR = ("usage: colab_kernel_launcher.py [-h] [--dry-run] [--fold {0,1,2}] "
         "[--output-dir OUTPUT_DIR] [--resume RESUME] colab_kernel_launcher.py: "
         "error: unrecognized arguments: -f /tmp/tmpod6u8lyj.json "
         "--HistoryManager.hist_file=:memory:")
RUN_PATH = "/code/onebitbrain/biohub-t03-f0/edit/run/335881086"


def source_path(rel: str) -> Path:
    return SOURCE_ROOT / rel


def draft_path(rel: str) -> Path:
    return DRAFT_ROOT / rel


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one replacement for {old!r}")
    return source.replace(old, new)


def attest_inputs() -> None:
    for rel, expected in INPUT_HASHES.items():
        actual = sha256(source_path(rel).read_bytes())
        if actual != expected:
            raise ValueError(f"immutable input drift: {rel} expected {expected}, got {actual}")


def fixed_source() -> bytes:
    source = source_path("kaggle_kernel/train_nb.py").read_text(encoding="utf-8")
    source = replace_once(source, "if __name__ == '__main__': raise SystemExit(main())", "if __name__ == '__main__': raise SystemExit(main([]))")
    return source.encode("utf-8")


def fixed_notebook(source_bytes: bytes) -> bytes:
    old = read_json(source_path("kaggle_kernel/train_nb.ipynb"))
    cells = [cell for cell in old.get("cells", []) if cell.get("cell_type") == "code"]
    before = source_path("kaggle_kernel/train_nb.py").read_bytes()
    if len(cells) != 1 or "".join(cells[0].get("source", [])).encode("utf-8") != before:
        raise ValueError("pre-fix notebook is not one-cell byte-equivalent to source")
    metadata = dict(old.get("metadata", {}))
    if metadata.get("t03_script_sha256") != sha256(before):
        raise ValueError("pre-fix notebook metadata drift")
    metadata["t03_script_sha256"] = sha256(source_bytes)
    value = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": source_bytes.decode("utf-8").splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    result = canonical(value)
    if b"".join(part.encode("utf-8") for part in value["cells"][0]["source"]) != source_bytes:
        raise ValueError("fixed notebook source is not byte-identical")
    return result


def fixed_test() -> bytes:
    source = source_path("tests/test_tnb.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        "ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'kaggle_kernel'/'train_nb.py'",
        "ROOT=Path(__file__).resolve().parents[1]; INPUT_ROOT=ROOT.parent; P=ROOT/'kaggle_kernel'/'train_nb.py'\n"
        "def _artifact(name):\n p=ROOT/name; return p if p.exists() else INPUT_ROOT/name",
    )
    source = replace_once(
        source,
        "  raw=(ROOT/'reports'/'g02'/'dataset_registry_v1.json').read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),m.REGISTRY_SHA256); full=json.loads(raw)",
        "  raw=_artifact('reports/g02/dataset_registry_v1.json').read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),m.REGISTRY_SHA256); full=json.loads(raw)",
    )
    source = replace_once(
        source,
        "   data=(ROOT/name).read_bytes(); self.assertEqual((len(data),hashlib.sha256(data).hexdigest()),(info['bytes'],info['sha256']))",
        "   data=_artifact(name).read_bytes(); self.assertEqual((len(data),hashlib.sha256(data).hexdigest()),(info['bytes'],info['sha256']))",
    )
    marker = " def test_projection_semantic_equivalence(self):"
    regression = ''' def test_kernel_main_argv(self):
  tree=ast.parse(P.read_text()); calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main']
  empty=[n for n in calls if len(n.args)==1 and isinstance(n.args[0],ast.List) and not n.args[0].elts and not n.keywords]
  self.assertEqual(len(empty),1); self.assertFalse(any(not n.args and not n.keywords for n in calls))
  terminal=tree.body[-1]; self.assertIsInstance(terminal,ast.If); self.assertEqual(len(terminal.body),1); self.assertIsInstance(terminal.body[0],ast.Raise)
  outer=terminal.body[0].exc; self.assertIsInstance(outer,ast.Call); self.assertEqual(len(outer.args),1); self.assertIsInstance(outer.args[0],ast.Call)
  call=outer.args[0]; self.assertIsInstance(call.func,ast.Name); self.assertEqual(call.func.id,'main'); self.assertEqual(len(call.args),1); self.assertIsInstance(call.args[0],ast.List); self.assertFalse(call.args[0].elts)
'''
    source = replace_once(source, marker, regression + marker)
    return source.encode("utf-8")


def fixed_pack(source_bytes: bytes, notebook_bytes: bytes, test_bytes: bytes) -> bytes:
    package = read_json(source_path("kaggle_kernel/tr-pack.json"))
    files = {
        "kaggle_kernel/train_nb.py": source_bytes,
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": source_path("kaggle_kernel/tr-meta.json").read_bytes(),
        "reports/g02/trial-v1.json": source_path("reports/g02/trial-v1.json").read_bytes(),
        "tests/test_tnb.py": test_bytes,
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False,
                           "source_cell_sha256": sha256(source_bytes)}
    package["package_files"] = {name: {"bytes": len(data), "sha256": sha256(data)}
                                for name, data in sorted(files.items())}
    return canonical(package)


def fixed_ledger() -> bytes:
    record = read_json(source_path("reports/g02/run-v1.json"))
    record["browser_evidence"].update({"displayed_state": "Failed", "displayed_runtime_seconds": 25,
                                       "run_path": RUN_PATH})
    record["launch"].update({"status": "failed_before_training", "run_path": RUN_PATH})
    record["final"].update({"charge_usd": None, "displayed_runtime_seconds": 25, "error": ERROR,
                            "run_path": RUN_PATH, "status": "failed_before_training", "steps": 0})
    return canonical(record)


def build() -> dict[str, bytes]:
    attest_inputs()
    source = fixed_source()
    notebook = fixed_notebook(source)
    test = fixed_test()
    return {
        "kaggle_kernel/train_nb.py": source,
        "kaggle_kernel/train_nb.ipynb": notebook,
        "kaggle_kernel/tr-pack.json": fixed_pack(source, notebook, test),
        "tests/test_tnb.py": test,
        "reports/g02/run-v1.json": fixed_ledger(),
    }


def embedded_from_source(source: str) -> dict:
    module = ast.parse(source)
    values = [node.value for node in module.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "EMBEDDED" for target in node.targets)]
    if len(values) != 1:
        raise ValueError("source must define exactly one EMBEDDED literal")
    value = ast.literal_eval(values[0])
    if not isinstance(value, dict):
        raise ValueError("EMBEDDED is not a dictionary")
    return value


def unpack(entry: tuple[str, str, int]) -> dict:
    payload, expected, size = entry
    raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
    if len(raw) != size or sha256(raw) != expected:
        raise ValueError("embedded manifest identity is invalid")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("embedded manifest is not an object")
    return value


def fold_data(split: dict, provenance: dict, fold: int) -> tuple[tuple[str, ...], set[str]]:
    split_fold, provenance_fold = split["folds"][str(fold)], provenance["folds"][str(fold)]
    fit, held = set(split_fold["fit_group_ids"]), set(split_fold["held_out_group_ids"])
    audit, forbidden = set(split["audit"]["group_ids"]), set(provenance_fold["forbidden_fit_group_ids"])
    if not fit or fit & held or fit & audit or fit & forbidden or not held <= forbidden or not audit <= forbidden:
        raise ValueError("fit-source isolation invariant failed")
    return tuple(sorted(fit)), held | audit | forbidden


def semantic_projection(source: str) -> dict:
    embedded = embedded_from_source(source)
    if set(embedded) != {"registry_projection", "split", "provenance"}:
        raise ValueError("compact embedded-manifest set is invalid")
    projection, split, provenance = (unpack(embedded[name]) for name in ("registry_projection", "split", "provenance"))
    if projection.get("source_registry_sha256") != FULL_REGISTRY_SHA256:
        raise ValueError("projection full-registry anchor drift")
    full_bytes = source_path("reports/g02/dataset_registry_v1.json").read_bytes()
    if sha256(full_bytes) != FULL_REGISTRY_SHA256:
        raise ValueError("full registry input drift")
    full = json.loads(full_bytes)
    full_by_group = {item["group_id"]: item for item in full["groups"]}
    projected_by_group = {item["group_id"]: item for item in projection["groups"]}
    hashes: dict[str, str] = {}
    for fold in (0, 1, 2):
        fit, forbidden = fold_data(split, provenance, fold)
        for fraction in (0.1, 0.3, 1.0):
            ranked = sorted(fit, key=lambda group: sha256(f"17:{group}".encode("ascii")))
            selected = ranked[:max(1, math.ceil(len(ranked) * fraction))]
            full_records = [{"group_id": group, "video_id": full_by_group[group]["video_ids"][0],
                             "stem": full_by_group[group]["video_ids"][0].removeprefix("train:")}
                            for group in selected]
            projected_records = [{"group_id": group, "video_id": projected_by_group[group]["video_ids"][0],
                                  "stem": projected_by_group[group]["video_ids"][0].removeprefix("train:")}
                                 for group in selected]
            if full_records != projected_records or set(selected) & forbidden:
                raise ValueError(f"semantic resolution mismatch for fold {fold}, fraction {fraction}")
            hashes[f"fold-{fold}-fraction-{fraction}"] = sha256(canonical(projected_records))
    return {"projection_bytes": embedded["registry_projection"][2], "projection_group_count": len(projection["groups"]),
            "projection_sha256": sha256(canonical(projection)), "semantic_record_sha256": hashes}


def attest_entrypoint(source: str) -> None:
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "main"]
    empty = [node for node in calls if len(node.args) == 1 and isinstance(node.args[0], ast.List)
             and not node.args[0].elts and not node.keywords]
    if len(empty) != 1 or any(not node.args and not node.keywords for node in calls):
        raise ValueError("main argv AST regression")
    terminal = tree.body[-1]
    if not isinstance(terminal, ast.If) or len(terminal.body) != 1 or not isinstance(terminal.body[0], ast.Raise):
        raise ValueError("terminal main guard is invalid")
    outer = terminal.body[0].exc
    if not isinstance(outer, ast.Call) or len(outer.args) != 1 or not isinstance(outer.args[0], ast.Call):
        raise ValueError("terminal SystemExit call is invalid")
    call = outer.args[0]
    if not isinstance(call.func, ast.Name) or call.func.id != "main" or len(call.args) != 1:
        raise ValueError("terminal main argv is invalid")


def compact_check() -> dict:
    expected = build()
    for rel, data in expected.items():
        path = draft_path(rel)
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"draft identity is invalid for {rel}")
    source = expected["kaggle_kernel/train_nb.py"]
    if len(source) >= 1_000_000:
        raise ValueError("compact source exceeds Kaggle limit")
    notebook = read_json(draft_path("kaggle_kernel/train_nb.ipynb"))
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(cells) != 1 or "".join(cells[0].get("source", [])).encode("utf-8") != source:
        raise ValueError("notebook is not one-cell byte-equivalent to source")
    if notebook.get("metadata", {}).get("t03_script_sha256") != sha256(source):
        raise ValueError("notebook source metadata is invalid")
    attest_entrypoint(source.decode("utf-8"))
    package = read_json(draft_path("kaggle_kernel/tr-pack.json"))
    for name, record in package["package_files"].items():
        path = draft_path(name)
        data = path.read_bytes() if path.exists() else source_path(name).read_bytes()
        if record != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError(f"package identity is invalid for {name}")
    corrected = read_json(draft_path("reports/g02/run-v1.json"))
    before = read_json(source_path("reports/g02/run-v1.json"))
    if corrected["package"] != before["package"]:
        raise ValueError("Version-1 package hashes changed")
    final, browser = corrected["final"], corrected["browser_evidence"]
    if (corrected["launch"]["status"], final["status"], final["steps"], final["charge_usd"],
            final["displayed_runtime_seconds"], final["error"], final["run_path"],
            browser["displayed_runtime_seconds"], browser["run_path"]) != (
            "failed_before_training", "failed_before_training", 0, None, 25, ERROR, RUN_PATH, 25, RUN_PATH):
        raise ValueError("Version-1 ledger correction is invalid")
    if source.decode("utf-8").replace("main([])", "main()") != source_path("kaggle_kernel/train_nb.py").read_text(encoding="utf-8"):
        raise ValueError("source change is broader than argv isolation")
    semantic = semantic_projection(source.decode("utf-8"))
    return {"source_bytes": len(source), "notebook_code_bytes": len("".join(cells[0]["source"]).encode("utf-8")),
            **semantic}


def dry_run_check() -> dict:
    source = draft_path("kaggle_kernel/train_nb.py").read_text(encoding="utf-8")
    imports = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.checkpoint import checkpoint as _grad_ckpt\n"
    source = source.replace(imports, "")
    stub = ("import types\n"
            "class _Module: pass\n"
            "class _Cuda:\n @staticmethod\n def is_available(): return False\n"
            "torch=types.SimpleNamespace(cuda=_Cuda())\n"
            "nn=types.SimpleNamespace(Module=_Module)\n"
            "F=types.SimpleNamespace()\n"
            "def _grad_ckpt(*args, **kwargs): raise RuntimeError('not available in dry-run verification')\n")
    source = replace_once(source, "from __future__ import annotations\n", "from __future__ import annotations\n" + stub)
    module = types.ModuleType("tnb_dry_check")
    module.__file__ = str(draft_path("kaggle_kernel/train_nb.py"))
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    records = []
    with tempfile.TemporaryDirectory(prefix=".t03dry-", dir=DRAFT_ROOT) as temporary:
        root = Path(temporary)
        for fold in (0, 0, 1, 2):
            output = root / f"fold-{fold}-{len(records)}"
            if module.main(["--dry-run", "--fold", str(fold), "--output-dir", str(output)]) != 0:
                raise ValueError(f"dry-run returned nonzero for fold {fold}")
            payload = (output / "prelaunch.json").read_bytes()
            records.append({"fold": fold, "sha256": sha256(payload), "bytes": len(payload)})
    if records[0]["sha256"] != records[1]["sha256"] or records[0]["bytes"] != records[1]["bytes"]:
        raise ValueError("repeated fold-0 prelaunch dry-run is not deterministic")
    return {"mode": "torch-free prelaunch shim", "dry_runs": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run-check", action="store_true")
    args = parser.parse_args()
    if args.check and args.dry_run_check:
        raise ValueError("choose one verification mode")
    if args.check:
        print(json.dumps(compact_check(), sort_keys=True))
        return
    if args.dry_run_check:
        print(json.dumps(dry_run_check(), sort_keys=True))
        return
    outputs = build()
    for rel, data in outputs.items():
        draft_path(rel).write_bytes(data)
    print(json.dumps({rel: {"bytes": len(data), "sha256": sha256(data)} for rel, data in sorted(outputs.items())}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministically rebuild the compact, self-contained T-03 notebook package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import re
import sys
import tempfile
import types
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "kaggle_kernel" / "train_nb.py"
NOTEBOOK = ROOT / "kaggle_kernel" / "train_nb.ipynb"
META = ROOT / "kaggle_kernel" / "tr-meta.json"
PACK = ROOT / "kaggle_kernel" / "tr-pack.json"
TEST = ROOT / "tests" / "test_tnb.py"
TRIAL = ROOT / "reports" / "g02" / "trial-v1.json"
RUN = ROOT / "reports" / "g02" / "run-v1.json"
RUN2 = ROOT / "reports" / "g02" / "run-v2.json"
TRIAL3 = ROOT / "reports" / "g02" / "trial-v3.json"
RUN3 = ROOT / "reports" / "g02" / "run-v3.json"
REGISTRY = ROOT / "reports" / "g02" / "dataset_registry_v1.json"

FULL_REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
PROVENANCE_SHA256 = "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f"
INITIAL_HASHES = {
    PYTHON: "00b64aa97324a4fe5c0c7502237722608806ad0f5585d97ab4a063268d074925",
    NOTEBOOK: "cfd847d19ac95d58c79cba0dac14fad9d244f12903ccbe0d0a3aa24c2e73ba12",
    META: "e1d3a95f6f399904684db8d0809fd3748cfe46480470b81b760246b1f20c6c88",
    PACK: "481eaeb0842112205959115fb42f12e95a25501ada11ac6e2e35f8212ef1bd4e",
    TEST: "5e0e542fda8cde4118f097979de5dfdc3a543cf65dddc4b69d664a52c25986f5",
    TRIAL: "20f6667a699db8f14ee1c8d4d1a22eb8092b75a2a4e6cfe9e3d074f383fde01d",
    REGISTRY: FULL_REGISTRY_SHA256,
}
FIELDS = ("group_id", "video_count", "contains_nontrain_member", "video_ids", "splits")
KERNEL_INPUT_HASHES = {
    PYTHON: "b467134582bad37b05c57074f2c9db1fa0e2d8915bd8d5629b62c5b1e596ed76",
    NOTEBOOK: "20e42c465f4f189f430290010781432494550c07aa25bb83ee5f098259265bcf",
    META: "65f72e905e1538f7aca5e6494f07307ffb6c9c9ce7ac0c6758ad941039bfd2c2",
    PACK: "c991526c2bfe1a22c14353f44705575b897ce14fece17b6bb133fd3ef91d4c9b",
    TEST: "2b6e6e3f477c6021fc9d60ef2fad9bbdb4369faa385a764b1b88e9c577e7fa25",
    TRIAL: "01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2",
    RUN: "b3f3a2da73e05bcfb49f6c8693960f4258d1ee8a0999234e68eca73dc959854e",
}
ARMED_INPUT_HASHES = {
    PYTHON: "5bd22032cbe9edfba185ad4ca0d3270fda7f4f66e8143945f449508c2dcbde21",
    NOTEBOOK: "c37ecc0c05763d7d60a0673130ff8af20d15c57ee7417794002f2683ef629716",
    META: "65f72e905e1538f7aca5e6494f07307ffb6c9c9ce7ac0c6758ad941039bfd2c2",
    PACK: "d48368a4ff66d63f18a8a6ad41e5469147c626ecd463a6bab7d9ae6cd051500e",
    TEST: "17b294033d34125589fd2d2eb0746721669dd5cc3ebc6fbedf16f00b51615260",
    TRIAL: "01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2",
    RUN: "2e54e18f0d6ffd937c5092e07daf982203325de53a13c91e828f6a4ad5caba29",
}
ZARR_INPUT_HASHES = {
    PYTHON: "2c0a0afed31bf7a69679876164d1fb6ae7526516ce3259be1570137bf8372ddc",
    NOTEBOOK: "b1e486ad501ea7eeaf967735bc2c7282a76edecb4f1b796136a9f282541dae00",
    META: "65f72e905e1538f7aca5e6494f07307ffb6c9c9ce7ac0c6758ad941039bfd2c2",
    PACK: "2c6ccd09a09368b3155a4200bccea151a716aad38c038960aa2097eae2b3ed6d",
    TEST: "9dec0b75aca578e965a22384573aef43adaf6873d6a9c1523340bad26949e3ab",
    TRIAL: "01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2",
    RUN: "2e54e18f0d6ffd937c5092e07daf982203325de53a13c91e828f6a4ad5caba29",
    RUN2: "e3362383438527ad060b4d52b2a3043a06d9b2f8236160ceeea754332316b3d1",
    TRIAL3: "9dbf157b6e59e7358baece018d3942126a45f67f7399095dd888401b5ea79f79",
}
ENTRYPOINT_OLD = "if __name__ == '__main__': raise SystemExit(main())"
ENTRYPOINT_NEW = "if __name__ == '__main__': raise SystemExit(main([]))"
ARMED_OLD = "    a=p.parse_args(argv); a.scenario='universal';"
ARMED_NEW = "    a=p.parse_args(argv); a.armed=not a.dry_run; a.scenario='universal';"
KAGGLE_RUN_PATH = "/code/onebitbrain/biohub-t03-f0/edit/run/335881086"
KAGGLE_ERROR = (
    "usage: colab_kernel_launcher.py [-h] [--dry-run] [--fold {0,1,2}] "
    "[--output-dir OUTPUT_DIR] [--resume RESUME] colab_kernel_launcher.py: "
    "error: unrecognized arguments: -f /tmp/tmpod6u8lyj.json "
    "--HistoryManager.hist_file=:memory:"
)
KAGGLE_RUN2_PATH = "/code/onebitbrain/biohub-t03-f0/edit/run/335889356"
KAGGLE_ERROR2 = "AttributeError: 'Namespace' object has no attribute 'armed'"
KAGGLE_OUTPUT_ZIP_SHA256 = "4f06c093ad93d85337e4da3de20e3cb310d5be503723d3b442965f28d5b3a6fd"
KAGGLE_PRELAUNCH_SHA256 = "9d412a863e93cd06de2e39df407fffe041f8f741f92c4e3db201c1755060e53c"
KAGGLE_RUN3_PATH = "/code/onebitbrain/biohub-t03-f0/edit/run/335900089"
KAGGLE_ERROR3 = "LaunchError: zarr is required by Kaggle runtime"
FAILURE_LOG = "/Users/agent/.codex/attachments/029d852e-b85e-465e-bf19-6cca1e42c54c/pasted-text.txt"
FAILURE_LOG_BYTES = 5920
FAILURE_LOG_SHA256 = "0803adca1138fcd033efaefdf675049bf5b6abdf345337a4939447ee42cc2102"


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
        raise ValueError(f"expected exactly one replacement for {old[:48]!r}")
    return source.replace(old, new)


def attest_kernel_entrypoint(source: str) -> None:
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main"
    ]
    empty_argv = [
        node for node in calls
        if len(node.args) == 1
        and isinstance(node.args[0], ast.List)
        and not node.args[0].elts
        and not node.keywords
    ]
    if len(empty_argv) != 1 or any(not node.args and not node.keywords for node in calls):
        raise ValueError("kernel entrypoint must contain exactly one main([]) and no main() call")
    terminal = tree.body[-1]
    if not isinstance(terminal, ast.If) or len(terminal.body) != 1 or not isinstance(terminal.body[0], ast.Raise):
        raise ValueError("kernel entrypoint is not the terminal raise")
    system_exit = terminal.body[0].exc
    if not isinstance(system_exit, ast.Call) or len(system_exit.args) != 1 or system_exit.args[0] is not empty_argv[0]:
        raise ValueError("terminal SystemExit does not wrap the attested main([]) call")


def attest_armed_contract(source: str) -> None:
    if source.count(ARMED_NEW) != 1 or ARMED_OLD in source:
        raise ValueError("compact argument builder must derive armed exactly once from not dry_run")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_args"]
    if len(functions) != 1:
        raise ValueError("compact source must define exactly one _args function")
    assignments = [
        node for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "a"
            and target.attr == "armed"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("compact _args must assign a.armed exactly once")
    value = assignments[0].value
    if not (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.Not)
        and isinstance(value.operand, ast.Attribute)
        and isinstance(value.operand.value, ast.Name)
        and value.operand.value.id == "a"
        and value.operand.attr == "dry_run"
    ):
        raise ValueError("compact armed assignment is not logical negation of dry_run")


def version2_ledger() -> dict:
    return {
        "artifact": "t03-private-run-v2",
        "browser_evidence": {
            "active_events": 0,
            "displayed_accelerator": "GPU T4 x2",
            "displayed_runtime_seconds": 273,
            "displayed_state": "Failed",
            "displayed_version": 2,
            "internet_enabled": False,
            "notebook_url": "https://www.kaggle.com/code/onebitbrain/biohub-t03-f0/edit",
            "private": True,
            "run_path": KAGGLE_RUN2_PATH,
        },
        "budget": {
            "paid_trials_consumed": 2,
            "paid_trials_remaining": 0,
            "third_launch_authorized": False,
        },
        "candidate": {
            "batch_size": 1,
            "division_enabled": False,
            "epochs": 1,
            "fit_fraction": 0.1,
            "fold": 0,
            "learning_rate": 1e-5,
            "max_steps": 1000,
            "scenario": "universal",
            "seed": 17,
            "sparse_negative_weight": 0.001,
            "unfreeze": 0,
        },
        "effects": {
            "dataset_created": False,
            "network_enabled": False,
            "public": False,
            "scored_submission": False,
        },
        "final": {
            "charge_usd": None,
            "completed_at": None,
            "displayed_runtime_seconds": 273,
            "error": KAGGLE_ERROR2,
            "failure_phase": "_run_bounded_training armed gate before data/model/training",
            "hours": 273 / 3600,
            "metrics_sha256": None,
            "output_checkpoint_sha256": None,
            "run_path": KAGGLE_RUN2_PATH,
            "status": "failed_before_training",
            "steps": 0,
        },
        "launch": {
            "launched_at": None,
            "maximum_charge_usd": 0.3,
            "maximum_hours": 0.75,
            "platform": "Kaggle private notebook",
            "run_path": KAGGLE_RUN2_PATH,
            "status": "failed_before_training",
            "time_provenance": "exact launch/completion timestamps were not retained",
        },
        "output_evidence": {
            "archive_name": "results.zip",
            "archive_sha256": KAGGLE_OUTPUT_ZIP_SHA256,
            "members": [{
                "bytes": 2839,
                "path": "t03/prelaunch.json",
                "sha256": KAGGLE_PRELAUNCH_SHA256,
                "status": "armed",
                "zip_timestamp": "2026-07-17T08:10:00Z",
            }],
        },
        "package": {
            "notebook_bytes": 89549,
            "notebook_code_bytes": 83637,
            "notebook_sha256": ARMED_INPUT_HASHES[NOTEBOOK],
            "registry_projection_sha256": "ec92db4d5fd09e82e867bbf32c1d0d8100dd55a9cbea81c62263ddf03598c115",
            "source_bytes": 83637,
            "source_sha256": ARMED_INPUT_HASHES[PYTHON],
            "trial_sha256": ARMED_INPUT_HASHES[TRIAL],
        },
        "recorded_at": "2026-07-17T08:14:06Z",
        "schema_version": 1,
    }


def kernel_fix_test(source: str) -> str:
    source = replace_once(
        source,
        "import ast, hashlib, importlib.util, json, math, subprocess, sys, tempfile, unittest",
        "import ast, hashlib, importlib.util, json, math, sys, tempfile, unittest",
    )
    marker = " def test_projection_semantic_equivalence(self):"
    regression = ''' def test_kernel_main_argv(self):
  tree=ast.parse(P.read_text()); calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main']
  empty=[n for n in calls if len(n.args)==1 and isinstance(n.args[0],ast.List) and not n.args[0].elts and not n.keywords]
  self.assertEqual(len(empty),1); self.assertFalse(any(not n.args and not n.keywords for n in calls))
  terminal=tree.body[-1]; self.assertIsInstance(terminal,ast.If); self.assertEqual(len(terminal.body),1); self.assertIsInstance(terminal.body[0],ast.Raise)
  outer=terminal.body[0].exc; self.assertIsInstance(outer,ast.Call); self.assertEqual(len(outer.args),1); self.assertIs(outer.args[0],empty[0])
'''
    source = replace_once(source, marker, regression + marker)
    old_dry = ''' def test_dry_folds_and_repeat(self):
  out=[]
  for fold in (0,0,1,2):
   with tempfile.TemporaryDirectory() as d:
    q=Path(d); r=subprocess.run([sys.executable,str(P),'--dry-run','--fold',str(fold),'--output-dir',str(q)],capture_output=True,text=True,check=True); out.append((r.stdout,hashlib.sha256((q/'prelaunch.json').read_bytes()).hexdigest()))
  self.assertEqual(out[0],out[1])
'''
    new_dry = ''' def test_dry_folds_and_repeat(self):
  s=importlib.util.spec_from_file_location('tnb_dry',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_dry']=m; s.loader.exec_module(m); out=[]
  for fold in (0,0,1,2):
   with tempfile.TemporaryDirectory() as d:
    q=Path(d)
    with mock.patch('builtins.print') as emit: self.assertEqual(m.main(['--dry-run','--fold',str(fold),'--output-dir',str(q)]),0)
    out.append((emit.call_args.args[0],hashlib.sha256((q/'prelaunch.json').read_bytes()).hexdigest()))
  self.assertEqual(out[0],out[1])
'''
    return replace_once(source, old_dry, new_dry)


def kernel_fix_outputs() -> dict[Path, bytes]:
    for path, expected in KERNEL_INPUT_HASHES.items():
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"kernel-fix input drift: {path.relative_to(ROOT)} expected {expected}, got {actual}")

    source = replace_once(PYTHON.read_text(encoding="utf-8"), ENTRYPOINT_OLD, ENTRYPOINT_NEW)
    source_bytes = source.encode("utf-8")
    attest_kernel_entrypoint(source)

    old_notebook = read_json(NOTEBOOK)
    old_cells = [cell for cell in old_notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(old_cells) != 1 or "".join(old_cells[0].get("source", [])).encode("utf-8") != PYTHON.read_bytes():
        raise ValueError("pre-fix notebook is not one-cell byte-equivalent to source")
    metadata = dict(old_notebook.get("metadata", {}))
    if metadata.get("t03_script_sha256") != sha256(PYTHON.read_bytes()):
        raise ValueError("pre-fix notebook metadata is invalid")
    metadata["t03_script_sha256"] = sha256(source_bytes)
    notebook_value = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = canonical(notebook_value)
    if b"".join(part.encode("utf-8") for part in notebook_value["cells"][0]["source"]) != source_bytes:
        raise ValueError("fixed notebook cell does not equal source bytes")

    test_bytes = kernel_fix_test(TEST.read_text(encoding="utf-8")).encode("utf-8")
    package = read_json(PACK)
    package_files = {
        "kaggle_kernel/train_nb.py": source_bytes,
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": META.read_bytes(),
        "reports/g02/trial-v1.json": TRIAL.read_bytes(),
        "tests/test_tnb.py": test_bytes,
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False, "source_cell_sha256": sha256(source_bytes)}
    package["package_files"] = {
        name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(package_files.items())
    }

    ledger = read_json(RUN)
    ledger["browser_evidence"].update({
        "active_events": 0,
        "displayed_runtime_seconds": 25,
        "displayed_state": "Failed",
        "run_path": KAGGLE_RUN_PATH,
    })
    ledger["launch"].update({"run_path": KAGGLE_RUN_PATH, "status": "failed_before_training"})
    ledger["final"].update({
        "charge_usd": None,
        "completed_at": "2026-07-17T07:18:19Z",
        "displayed_runtime_seconds": 25,
        "error": KAGGLE_ERROR,
        "hours": 25 / 3600,
        "run_path": KAGGLE_RUN_PATH,
        "status": "failed_before_training",
        "steps": 0,
    })
    return {
        PYTHON: source_bytes,
        NOTEBOOK: notebook_bytes,
        PACK: canonical(package),
        TEST: test_bytes,
        RUN: canonical(ledger),
    }


def armed_fix_test(source: str) -> str:
    marker = " def test_projection_semantic_equivalence(self):"
    regression = ''' def test_compact_args_and_armed_training(self):
  s=importlib.util.spec_from_file_location('tnb_armed',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_armed']=m; s.loader.exec_module(m)
  self.assertTrue(m._args([]).armed); self.assertFalse(m._args(['--dry-run']).armed)
  torch=__import__('torch')
  class Tiny(torch.nn.Module):
   def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  b={'frames':torch.ones(1,2,1,4,4,4),'nodes':[{'id':1,'batch':0,'time':0,'z':1,'y':1,'x':1},{'id':2,'batch':0,'time':1,'z':1,'y':2,'x':1}],'edges':[{'source_id':1,'target_id':2}],'voxel_size_um':(1.,1.,1.),'window_id':'armed'}
  record=m.SourceRecord('g02-local-armed','train:6bba_edf14583','6bba_edf14583')
  with tempfile.TemporaryDirectory() as d:
   q=Path(d); a=m._args([]); a.output_dir=q; a.data_root=None; a.max_steps=1; a.maximum_hours=.01
   result=m._run_bounded_training(a,[record],'cpu',model_factory=lambda _a,_d:Tiny(),batch_factory=lambda _s,_d:[b])
   self.assertEqual((result['status'],result['completed_steps']),('completed',1))
   self.assertTrue(all((q/name).is_file() for name in ('checkpoint.pt','metrics.json','resume.json')))
'''
    source = replace_once(source, marker, regression + marker)
    old_names = "ROOT/'reports'/'g02'/'trial-v1.json')"
    new_names = "ROOT/'reports'/'g02'/'trial-v1.json',ROOT/'reports'/'g02'/'run-v2.json')"
    return replace_once(source, old_names, new_names)


def armed_fix_outputs() -> dict[Path, bytes]:
    for path, expected in ARMED_INPUT_HASHES.items():
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"armed-fix input drift: {path.relative_to(ROOT)} expected {expected}, got {actual}")
    if RUN2.exists():
        raise ValueError("armed-fix is one-shot and run-v2.json already exists")

    source = replace_once(PYTHON.read_text(encoding="utf-8"), ARMED_OLD, ARMED_NEW)
    source_bytes = source.encode("utf-8")
    attest_kernel_entrypoint(source)
    attest_armed_contract(source)

    old_notebook = read_json(NOTEBOOK)
    old_cells = [cell for cell in old_notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(old_cells) != 1 or "".join(old_cells[0].get("source", [])).encode("utf-8") != PYTHON.read_bytes():
        raise ValueError("pre-armed notebook is not one-cell byte-equivalent to source")
    metadata = dict(old_notebook.get("metadata", {}))
    if metadata.get("t03_script_sha256") != sha256(PYTHON.read_bytes()):
        raise ValueError("pre-armed notebook metadata is invalid")
    metadata["t03_script_sha256"] = sha256(source_bytes)
    notebook_value = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = canonical(notebook_value)
    if b"".join(part.encode("utf-8") for part in notebook_value["cells"][0]["source"]) != source_bytes:
        raise ValueError("armed notebook cell does not equal source bytes")

    test_bytes = armed_fix_test(TEST.read_text(encoding="utf-8")).encode("utf-8")
    package = read_json(PACK)
    package_files = {
        "kaggle_kernel/train_nb.py": source_bytes,
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": META.read_bytes(),
        "reports/g02/trial-v1.json": TRIAL.read_bytes(),
        "tests/test_tnb.py": test_bytes,
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False, "source_cell_sha256": sha256(source_bytes)}
    package["package_files"] = {
        name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(package_files.items())
    }
    package["runtime_contract"] = {
        "armed": "not dry_run",
        "kaggle_entrypoint": "main([])",
        "local_cpu_step_regression": True,
    }
    return {
        PYTHON: source_bytes,
        NOTEBOOK: notebook_bytes,
        PACK: canonical(package),
        TEST: test_bytes,
        RUN2: canonical(version2_ledger()),
    }


def zarr_bootstrap_code() -> str:
    return '''
ZARR_ROOTS = (
    Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1'),
    Path('/kaggle/input/biohub-tracking-support-pack-50ep-v1'),
)
ZARR_DEPS = Path('/kaggle/working/t03/deps-v1')
ZARR_WHEELS = (
    ('zarr-3.2.1-py3-none-any.whl', 319617, 'f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7'),
    ('donfig-0.8.1.post1-py3-none-any.whl', 21592, '2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d'),
    ('google_crc32c-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl', 33364, '14f87e04d613dfa218d6135e81b78272c3b904e2a7053b841481b38a7d901411'),
    ('numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 8879163, 'c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3'),
    ('deprecated-1.3.1-py2.py3-none-any.whl', 11298, '597bfef186b6f60181535a29fbe44865ce137a5079f295b479886c82729d5f3f'),
    ('wrapt-2.2.2-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl', 169828, '2e096c9d39a59b35b63c9aacfbbbec2088ff51ff1fc31051acc60a07f42f273a'),
    ('typing_extensions-4.15.0-py3-none-any.whl', 44614, 'f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548'),
    ('packaging-26.2-py3-none-any.whl', 100195, '5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e'),
    ('pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl', 807870, 'ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc'),
)

def _wheel_root(specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS,
                roots: Sequence[Path]=ZARR_ROOTS) -> Path:
    hits = []
    for base in roots:
        directory = Path(base) / 'wheels'
        if not directory.is_dir(): continue
        if all((directory / name).is_file() for name, _, _ in specs):
            resolved = directory.resolve(strict=True)
            if resolved not in hits: hits.append(resolved)
    if len(hits) != 1: raise LaunchError('support wheel search must yield one exact directory')
    root = hits[0]
    for name, size, digest in specs:
        path = root / name
        if path.stat().st_size != size or _sha_file(path) != digest:
            raise LaunchError('support wheel identity mismatch')
    return root

def _extract_wheels(root: Path, dest: Path,
                    specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS) -> None:
    identity = json.dumps({'schema_version':1,'wheels':[list(item) for item in specs]},sort_keys=True,separators=(',',':')) + '\\n'
    marker = hashlib.sha256(identity.encode('ascii')).hexdigest() + '\\n'
    ready = dest / '.ready'
    if ready.is_file() and ready.read_text('ascii') == marker: return
    temporary = dest.with_name(dest.name + '.tmp')
    if temporary.exists(): shutil.rmtree(temporary)
    if dest.exists(): shutil.rmtree(dest)
    temporary.mkdir(parents=True)
    try:
        for name, _, _ in specs:
            with zipfile.ZipFile(root / name) as archive:
                for info in archive.infolist():
                    parts = Path(info.filename).parts
                    mode = (info.external_attr >> 16) & 0o170000
                    if (not parts or info.filename.startswith(('/', '\\\\')) or '\\\\' in info.filename
                            or any(part in ('', '.', '..') for part in parts) or mode == 0o120000):
                        raise LaunchError('unsafe support wheel member')
                    target = temporary.joinpath(*parts)
                    try: target.resolve().relative_to(temporary.resolve())
                    except ValueError as exc: raise LaunchError('support wheel member escapes extraction root') from exc
                    if info.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open('wb') as output: shutil.copyfileobj(source, output)
        (temporary / '.ready').write_text(marker, encoding='ascii')
        temporary.replace(dest)
    except Exception as exc:
        if temporary.exists(): shutil.rmtree(temporary)
        if isinstance(exc, LaunchError): raise
        raise LaunchError('offline support wheel extraction failed') from exc

def _ensure_zarr(specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS,
                 roots: Sequence[Path]=ZARR_ROOTS, dest: Path=ZARR_DEPS,
                 importer: Any=None) -> Any:
    load = importlib.import_module if importer is None else importer
    try: return load('zarr')
    except ImportError: pass
    root = _wheel_root(specs, roots); _extract_wheels(root, Path(dest), specs)
    location = str(Path(dest))
    if location in sys.path: sys.path.remove(location)
    sys.path.insert(0, location)
    try: return load('zarr')
    except ImportError as exc: raise LaunchError('zarr import failed after verified offline bootstrap') from exc
'''


def version3_ledger() -> dict:
    return {
        "artifact": "t03-private-run-v3",
        "browser_evidence": {
            "displayed_accelerator": "GPU T4 x2",
            "displayed_runtime_seconds": 290.3,
            "displayed_state": "Failed",
            "displayed_version": 3,
            "internet_enabled": False,
            "notebook_url": "https://www.kaggle.com/code/onebitbrain/biohub-t03-f0/edit",
            "private": True,
            "run_path": KAGGLE_RUN3_PATH,
            "script_version_id": 335900089,
        },
        "budget": {
            "paid_trials_consumed": 3,
            "paid_trials_remaining": 0,
            "fourth_launch_authorized": False,
        },
        "candidate_changed": False,
        "effects": {
            "dataset_created": False,
            "network_enabled": False,
            "public": False,
            "scored_submission": False,
        },
        "failure_log": {
            "bytes": FAILURE_LOG_BYTES,
            "path": FAILURE_LOG,
            "sha256": FAILURE_LOG_SHA256,
        },
        "final": {
            "charge_usd": None,
            "displayed_runtime_seconds": 290.3,
            "error": KAGGLE_ERROR3,
            "failure_phase": "first fit-source metadata read before first batch or optimizer step",
            "run_path": KAGGLE_RUN3_PATH,
            "status": "failed_before_training",
            "steps": 0,
            "traceback_runtime_seconds": 283.2,
        },
        "output_evidence": {
            "members": [{
                "displayed_bytes": 2840,
                "exact_bytes_observed": False,
                "exact_sha256_observed": False,
                "path": "t03/prelaunch.json",
                "status": "armed",
            }],
        },
        "package": {
            "notebook_bytes": 89572,
            "notebook_sha256": ZARR_INPUT_HASHES[NOTEBOOK],
            "source_bytes": 83660,
            "source_sha256": ZARR_INPUT_HASHES[PYTHON],
            "trial_authorization_sha256": ZARR_INPUT_HASHES[TRIAL3],
        },
        "recorded_at": "2026-07-17T10:15:00Z",
        "remediation": {
            "checkpoint_or_candidate_change": False,
            "fourth_launch_performed": False,
            "required_action": "attach existing support-pack input and verify exact offline zarr wheels",
            "status": "prepared_not_kaggle_verified",
        },
        "schema_version": 1,
    }


def zarr_fix_test(source: str) -> str:
    source = replace_once(
        source,
        "import ast, hashlib, importlib.util, json, math, sys, tempfile, unittest",
        "import ast, hashlib, importlib.util, json, math, sys, tempfile, unittest, zipfile",
    )
    marker = " def test_package_identity_and_checkpoint_search(self):"
    regression = ''' def test_offline_zarr_bootstrap(self):
  s=importlib.util.spec_from_file_location('tnb_zarr',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_zarr']=m; s.loader.exec_module(m)
  def make(path,members):
   path.parent.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(path,'w') as archive:
    for name,data in members: archive.writestr(name,data)
   data=path.read_bytes(); return (path.name,len(data),hashlib.sha256(data).hexdigest())
  def missing(_): raise ModuleNotFoundError('zarr')
  with tempfile.TemporaryDirectory() as d:
   base=Path(d); support=base/'support'; dest=base/'deps'
   spec=make(support/'wheels'/'zarr-test.whl',[('zarr/__init__.py','VALUE=1\\n')])
   token=object()
   def importer(_):
    if not (dest/'zarr/__init__.py').is_file(): raise ModuleNotFoundError('zarr')
    return token
   self.assertIs(m._ensure_zarr((spec,),(support,),dest,importer),token); before=(dest/'.ready').read_bytes()
   self.assertIs(m._ensure_zarr((spec,),(support,),dest,importer),token); self.assertEqual((dest/'.ready').read_bytes(),before)
   with self.assertRaises(m.LaunchError): m._ensure_zarr((spec,),(base/'missing',),base/'x',missing)
   twin=base/'twin'; make(twin/'wheels'/spec[0],[('zarr/__init__.py','VALUE=1\\n')])
   with self.assertRaises(m.LaunchError): m._ensure_zarr((spec,),(support,twin),base/'y',missing)
   bad=base/'bad'; badspec=make(bad/'wheels'/'bad.whl',[('../escape','x')])
   with self.assertRaises(m.LaunchError): m._ensure_zarr((badspec,),(bad,),base/'z',missing)
   self.assertFalse((base/'escape').exists())
'''
    source = replace_once(source, marker, regression + marker)
    source = replace_once(
        source,
        "with mock.patch.object(m,'_sha_file',return_value=m.WEIGHTS_SHA256):\n     with self.assertRaises(m.LaunchError): m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256)",
        "with mock.patch.object(m,'_sha_file',return_value=m.WEIGHTS_SHA256):\n     with self.assertRaises(m.LaunchError): m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256)\n    weight=root/'locked'/'edge_predictor_best.pth'; weight.parent.mkdir(); weight.touch(); config=weight.with_name('config.json'); config.touch()\n    with config.open('r+b') as handle: handle.truncate(m.CONFIG_BYTES)\n    with mock.patch.object(m,'_sha_file',return_value=m.CONFIG_SHA256): self.assertEqual(m._sibling_config(weight),config)",
    )
    source = replace_once(
        source,
        "ROOT/'reports'/'g02'/'run-v2.json'):",
        "ROOT/'reports'/'g02'/'run-v2.json',ROOT/'reports'/'g02'/'run-v3.json'):",
    )
    return source


def zarr_fix_outputs() -> dict[Path, bytes]:
    for path, expected in ZARR_INPUT_HASHES.items():
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"zarr-fix input drift: {path.relative_to(ROOT)} expected {expected}, got {actual}")
    if RUN3.exists():
        raise ValueError("zarr-fix is one-shot and run-v3.json already exists")

    source = PYTHON.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "import argparse, base64, hashlib, json, math, os, random, re, tempfile, time, zlib",
        "import argparse, base64, hashlib, importlib, json, math, os, random, re, shutil, sys, tempfile, time, zipfile, zlib",
    )
    source = replace_once(source, "\n\nclass SparseWindowSource:", zarr_bootstrap_code() + "\n\nclass SparseWindowSource:")
    source = replace_once(
        source,
        "        try: import zarr\n        except ImportError as exc: raise LaunchError(\"zarr is required by Kaggle runtime\") from exc",
        "        zarr = _ensure_zarr()",
    )
    source = replace_once(
        source,
        "def _args(argv: Sequence[str] | None=None) -> argparse.Namespace:",
        "def _sibling_config(weights: Path) -> Path:\n    path = Path(weights).with_name('config.json')\n    if not path.is_file() or path.stat().st_size != CONFIG_BYTES or _sha_file(path) != CONFIG_SHA256: raise LaunchError('checkpoint sibling config identity mismatch')\n    return path\ndef _args(argv: Sequence[str] | None=None) -> argparse.Namespace:",
    )
    source = replace_once(
        source,
        "    args.weights_path=_find_input('edge_predictor_best.pth',8357783,WEIGHTS_SHA256); args.config_path=_find_input('config.json',165,CONFIG_SHA256)",
        "    args.weights_path=_find_input('edge_predictor_best.pth',WEIGHTS_BYTES,WEIGHTS_SHA256); args.config_path=_sibling_config(args.weights_path)",
    )
    source_bytes = source.encode("utf-8")
    if len(source_bytes) >= 1_000_000:
        raise ValueError("zarr-fixed source exceeds Kaggle limit")
    attest_kernel_entrypoint(source)
    attest_armed_contract(source)

    old_notebook = read_json(NOTEBOOK)
    old_cells = [cell for cell in old_notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(old_cells) != 1 or "".join(old_cells[0].get("source", [])).encode("utf-8") != PYTHON.read_bytes():
        raise ValueError("pre-zarr notebook is not one-cell byte-equivalent to source")
    metadata = dict(old_notebook.get("metadata", {}))
    metadata["t03_script_sha256"] = sha256(source_bytes)
    notebook_value = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = canonical(notebook_value)
    if b"".join(part.encode("utf-8") for part in notebook_value["cells"][0]["source"]) != source_bytes:
        raise ValueError("zarr-fixed notebook cell does not equal source bytes")

    test_bytes = zarr_fix_test(TEST.read_text(encoding="utf-8")).encode("utf-8")
    package = read_json(PACK)
    package_files = {
        "kaggle_kernel/train_nb.py": source_bytes,
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": META.read_bytes(),
        "reports/g02/trial-v1.json": TRIAL.read_bytes(),
        "tests/test_tnb.py": test_bytes,
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False, "source_cell_sha256": sha256(source_bytes)}
    package["package_files"] = {
        name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(package_files.items())
    }
    package["runtime_contract"] = {
        "armed": "not dry_run",
        "kaggle_entrypoint": "main([])",
        "local_cpu_step_regression": True,
        "offline_zarr_bootstrap": {
            "exact_wheel_count": 9,
            "extract_root": "/kaggle/working/t03/deps-v1",
            "install": False,
            "network": False,
            "support_pack_input_required": True,
        },
    }
    return {
        PYTHON: source_bytes,
        NOTEBOOK: notebook_bytes,
        PACK: canonical(package),
        TEST: test_bytes,
        RUN3: canonical(version3_ledger()),
    }


def runtime_contract_v4() -> dict:
    return {
        "armed": "not dry_run",
        "kaggle_entrypoint": "main([])",
        "local_cpu_step_regression": True,
        "mixed_precision": False,
        "source_coordinate_preflight": "all selected sources before model training",
        "offline_dependency_bootstrap": {
            "exact_distribution_count": 9,
            "exact_native_import_count": 4,
            "exact_wheel_count": 9,
            "extract_root": "/kaggle/working/t03/deps-v1",
            "installation": "manual_verified_wheel_extraction",
            "network": False,
            "pip": False,
            "preflight_before_data": True,
            "support_pack_input_required": True,
        },
    }


def sync_current_outputs() -> dict[Path, bytes]:
    source_bytes = PYTHON.read_bytes()
    source = source_bytes.decode("utf-8")
    if len(source_bytes) >= 1_000_000:
        raise ValueError("current source exceeds Kaggle limit")
    attest_kernel_entrypoint(source)
    attest_armed_contract(source)
    for marker in ("RUNTIME_IMPORTS = (", "def _validate_arg_contract(", "runtime=_runtime_preflight(args)"):
        if source.count(marker) != 1:
            raise ValueError(f"current preflight marker is invalid: {marker}")

    old_notebook = read_json(NOTEBOOK)
    metadata = dict(old_notebook.get("metadata", {}))
    metadata["t03_script_sha256"] = sha256(source_bytes)
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = canonical(notebook)
    if b"".join(part.encode("utf-8") for part in notebook["cells"][0]["source"]) != source_bytes:
        raise ValueError("current notebook cell does not equal source bytes")

    package = read_json(PACK)
    package_files = {
        "kaggle_kernel/train_nb.py": source_bytes,
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": META.read_bytes(),
        "reports/g02/trial-v1.json": TRIAL.read_bytes(),
        "tests/test_tnb.py": TEST.read_bytes(),
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False, "source_cell_sha256": sha256(source_bytes)}
    package["package_files"] = {
        name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(package_files.items())
    }
    package["runtime_contract"] = runtime_contract_v4()
    return {NOTEBOOK: notebook_bytes, PACK: canonical(package)}


def embedded_from_source(source: str) -> dict:
    module = ast.parse(source)
    assignments = [
        node.value for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EMBEDDED" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("source must define exactly one EMBEDDED literal")
    value = ast.literal_eval(assignments[0])
    if not isinstance(value, dict):
        raise ValueError("EMBEDDED is not a dict")
    return value


def unpack(entry: tuple[str, str, int]) -> dict:
    payload, expected, size = entry
    raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
    if len(raw) != size or sha256(raw) != expected:
        raise ValueError("embedded immutable manifest is invalid")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("embedded immutable manifest is not an object")
    return value


def pack(value: dict) -> tuple[str, str, int]:
    raw = canonical(value)
    return (base64.b85encode(zlib.compress(raw, level=9)).decode("ascii"), sha256(raw), len(raw))


def fold_data(split: dict, provenance: dict, fold: int) -> tuple[tuple[str, ...], set[str]]:
    split_fold = split["folds"][str(fold)]
    provenance_fold = provenance["folds"][str(fold)]
    fit, held = set(split_fold["fit_group_ids"]), set(split_fold["held_out_group_ids"])
    audit, forbidden = set(split["audit"]["group_ids"]), set(provenance_fold["forbidden_fit_group_ids"])
    if not fit or fit & held or fit & audit or fit & forbidden or not held <= forbidden or not audit <= forbidden:
        raise ValueError("fit-source isolation invariant failed while projecting registry")
    return tuple(sorted(fit)), held | audit | forbidden


def projection_from_full_registry(full_registry: dict, split: dict, provenance: dict) -> dict:
    groups = full_registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("full registry groups is not a list")
    fit_groups = sorted(set().union(*(set(fold_data(split, provenance, fold)[0]) for fold in (0, 1, 2))))
    wanted = set(fit_groups)
    indexed: dict[str, list[dict]] = {group_id: [] for group_id in fit_groups}
    for item in groups:
        if isinstance(item, dict) and item.get("group_id") in wanted:
            indexed[item["group_id"]].append(item)
    projected = []
    for group_id in fit_groups:
        matches = indexed[group_id]
        if len(matches) != 1 or set(FIELDS) - set(matches[0]):
            raise ValueError(f"full registry cannot deterministically project {group_id}")
        projected.append({field: matches[0][field] for field in FIELDS})
    return {
        "artifact": "runtime-registry-projection-v1",
        "source_registry_sha256": FULL_REGISTRY_SHA256,
        "groups": projected,
    }


def runtime_projection_code() -> str:
    return '''\n\nPROJECTION_FIELDS = ("group_id", "video_count", "contains_nontrain_member", "video_ids", "splits")\ndef runtime_registry_projection() -> dict[str, Any]:
    projection = embedded_manifest("registry_projection")
    if (projection.get("artifact") != "runtime-registry-projection-v1"
            or projection.get("source_registry_sha256") != REGISTRY_SHA256):
        raise LaunchError("runtime registry projection provenance mismatch")
    groups = projection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise LaunchError("runtime registry projection groups are invalid")
    normalized = []
    for item in groups:
        if not isinstance(item, Mapping) or set(item) != set(PROJECTION_FIELDS):
            raise LaunchError("runtime registry projection fields are invalid")
        if (not isinstance(item["group_id"], str) or not isinstance(item["video_count"], int)
                or not isinstance(item["contains_nontrain_member"], bool)
                or not isinstance(item["video_ids"], list) or not all(isinstance(value, str) for value in item["video_ids"])
                or not isinstance(item["splits"], list) or not all(isinstance(value, str) for value in item["splits"])):
            raise LaunchError("runtime registry projection values are invalid")
        normalized.append(dict(item))
    canonical = {"artifact": "runtime-registry-projection-v1", "source_registry_sha256": REGISTRY_SHA256, "groups": normalized}
    if _canonical_json(projection) != _canonical_json(canonical):
        raise LaunchError("runtime registry projection is not canonical")
    if hashlib.sha256(_canonical_json(canonical)).hexdigest() != REGISTRY_PROJECTION_SHA256:
        raise LaunchError("runtime registry projection hash mismatch")
    return canonical
'''


def revised_test(source: str) -> str:
    source = replace_once(source, "import ast, hashlib, importlib.util, json, subprocess, sys, tempfile, unittest", "import ast, hashlib, importlib.util, json, math, subprocess, sys, tempfile, unittest")
    old = ''' def test_ast_and_embedded(self):
  ast.parse(P.read_text()); s=importlib.util.spec_from_file_location('tnb',P); m=importlib.util.module_from_spec(s); sys.modules['tnb']=m; s.loader.exec_module(m)
  for n,h in [('registry','21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158'),('split','ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769'),('provenance','6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f')]:
   self.assertEqual(m.EMBEDDED[n][1],h); self.assertIsInstance(m.embedded_manifest(n),dict)
'''
    new = ''' def test_ast_and_embedded(self):
  ast.parse(P.read_text()); s=importlib.util.spec_from_file_location('tnb',P); m=importlib.util.module_from_spec(s); sys.modules['tnb']=m; s.loader.exec_module(m)
  for n,h in [('split','ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769'),('provenance','6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f')]:
   self.assertEqual(m.EMBEDDED[n][1],h); self.assertIsInstance(m.embedded_manifest(n),dict)
  projection=m.runtime_registry_projection(); self.assertEqual(projection['source_registry_sha256'],m.REGISTRY_SHA256); self.assertEqual(m.EMBEDDED['registry_projection'][1],m.REGISTRY_PROJECTION_SHA256)
 def test_projection_semantic_equivalence(self):
  s=importlib.util.spec_from_file_location('tnb_projection',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_projection']=m; s.loader.exec_module(m)
  raw=(ROOT/'reports'/'g02'/'dataset_registry_v1.json').read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),m.REGISTRY_SHA256); full=json.loads(raw)
  split,prov=m.embedded_manifest('split'),m.embedded_manifest('provenance'); projection=m.runtime_registry_projection(); fields=('group_id','video_count','contains_nontrain_member','video_ids','splits')
  union=sorted(set().union(*(set(m._fold_data(f,split,prov)[0]) for f in (0,1,2)))); wanted=set(union); indexed={item['group_id']:item for item in full['groups'] if item['group_id'] in wanted}
  rebuilt={'artifact':'runtime-registry-projection-v1','source_registry_sha256':m.REGISTRY_SHA256,'groups':[{key:indexed[group][key] for key in fields} for group in union]}
  self.assertEqual(m._canonical_json(projection),m._canonical_json(rebuilt)); self.assertEqual(hashlib.sha256(m._canonical_json(projection)).hexdigest(),m.REGISTRY_PROJECTION_SHA256)
  for fold in (0,1,2):
   _,forbidden=m._fold_data(fold,split,prov)
   for fraction in m.FRACTIONS:
    selected=m.select_fit_groups(fold,split,prov,fraction,17); self.assertEqual(m.resolve_fit_records(selected,projection,forbidden),m.resolve_fit_records(selected,full,forbidden))
'''
    return replace_once(source, old, new)


def build() -> dict[Path, bytes]:
    for path, expected in INITIAL_HASHES.items():
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"immutable input drift: {path.relative_to(ROOT)} expected {expected}, got {actual}")
    source = PYTHON.read_text(encoding="utf-8")
    embedded = embedded_from_source(source)
    if set(embedded) != {"registry", "split", "provenance"}:
        raise ValueError("unexpected initial embedded-manifest set")
    split, provenance = unpack(embedded["split"]), unpack(embedded["provenance"])
    if embedded["registry"][1] != FULL_REGISTRY_SHA256 or embedded["split"][1] != SPLIT_SHA256 or embedded["provenance"][1] != PROVENANCE_SHA256:
        raise ValueError("embedded immutable identity drift")
    projection = projection_from_full_registry(read_json(REGISTRY), split, provenance)
    projection_entry = pack(projection)
    projection_sha256 = projection_entry[1]
    compact_embedded = {"registry_projection": projection_entry, "split": embedded["split"], "provenance": embedded["provenance"]}
    embedded_line = next(line for line in source.splitlines() if line.startswith("EMBEDDED = "))
    source = replace_once(source, embedded_line, "EMBEDDED = " + repr(compact_embedded))
    source = replace_once(source, "    return value\n\ndef trial_record", "    return value" + runtime_projection_code() + "\n\ndef trial_record")
    source = replace_once(source, f'REGISTRY_SHA256 = "{FULL_REGISTRY_SHA256}"', f'REGISTRY_SHA256 = "{FULL_REGISTRY_SHA256}"\nREGISTRY_PROJECTION_SHA256 = "{projection_sha256}"')
    source = replace_once(source, '"split_sha256": SPLIT_SHA256, "provenance_sha256": PROVENANCE_SHA256, "registry_sha256": REGISTRY_SHA256,', '"split_sha256": SPLIT_SHA256, "provenance_sha256": PROVENANCE_SHA256, "registry_sha256": REGISTRY_SHA256, "registry_projection_sha256": REGISTRY_PROJECTION_SHA256,')
    source = replace_once(source, "if trial['immutable_inputs']['registry_sha256'] != REGISTRY_SHA256: raise LaunchError('trial registry identity drift')", "if (trial['immutable_inputs']['registry_sha256'] != REGISTRY_SHA256 or trial['immutable_inputs']['registry_projection_sha256'] != REGISTRY_PROJECTION_SHA256): raise LaunchError('trial registry identity drift')")
    source = replace_once(source, "split, provenance, registry = embedded_manifest('split'), embedded_manifest('provenance'), embedded_manifest('registry')", "split, provenance = embedded_manifest('split'), embedded_manifest('provenance'); registry = runtime_registry_projection()")

    trial = read_json(TRIAL)
    trial["immutable_inputs"]["registry_projection_sha256"] = projection_sha256
    trial["source_isolation"]["identity_source"] = "embedded runtime registry projection; full registry SHA-256 anchor"
    trial_bytes = canonical(trial)
    trial_sha256 = sha256(trial_bytes)
    source = re.sub(r"TRIAL_BYTES = base64\.b85decode\([^\n]+\)", "TRIAL_BYTES = base64.b85decode(" + repr(base64.b85encode(trial_bytes)) + ")", source, count=1)
    source = re.sub(r"TRIAL_SHA256 = '[0-9a-f]{64}'", f"TRIAL_SHA256 = '{trial_sha256}'", source, count=1)
    if source.count("REGISTRY_PROJECTION_SHA256") < 3:
        raise ValueError("projection identity did not bind source, prelaunch, and armed gate")
    if len(source.encode("utf-8")) >= 1_000_000:
        raise ValueError("compact source still exceeds Kaggle limit")

    test = revised_test(TEST.read_text(encoding="utf-8"))
    old_notebook = read_json(NOTEBOOK)
    metadata = dict(old_notebook.get("metadata", {}))
    metadata["t03_script_sha256"] = sha256(source.encode("utf-8"))
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = canonical(notebook)
    if b"".join(part.encode("utf-8") for part in notebook["cells"][0]["source"]) != source.encode("utf-8"):
        raise ValueError("notebook cell does not equal source bytes")

    meta = read_json(META)
    meta["trial_sha256"] = trial_sha256
    meta["registry_projection_sha256"] = projection_sha256
    meta["source_registry_sha256"] = FULL_REGISTRY_SHA256
    meta_bytes = canonical(meta)
    package_files = {
        "kaggle_kernel/train_nb.py": source.encode("utf-8"),
        "kaggle_kernel/train_nb.ipynb": notebook_bytes,
        "kaggle_kernel/tr-meta.json": meta_bytes,
        "reports/g02/trial-v1.json": trial_bytes,
        "tests/test_tnb.py": test.encode("utf-8"),
    }
    package = read_json(PACK)
    package["embedded_manifests"] = {
        "provenance": {"bytes": embedded["provenance"][2], "sha256": PROVENANCE_SHA256},
        "registry": {"bytes": REGISTRY.stat().st_size, "sha256": FULL_REGISTRY_SHA256},
        "registry_projection": {"bytes": projection_entry[2], "group_count": len(projection["groups"]), "sha256": projection_sha256, "source_registry_sha256": FULL_REGISTRY_SHA256},
        "split": {"bytes": embedded["split"][2], "sha256": SPLIT_SHA256},
    }
    package["notebook"] = {"self_contained": True, "sidecar_required": False, "source_cell_sha256": sha256(source.encode("utf-8"))}
    package["package_files"] = {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(package_files.items())}
    package["projection_binding"] = {"full_registry_sha256": FULL_REGISTRY_SHA256, "projection_sha256": projection_sha256, "projection_group_count": len(projection["groups"])}
    package_bytes = canonical(package)
    return {
        PYTHON: source.encode("utf-8"),
        NOTEBOOK: notebook_bytes,
        META: meta_bytes,
        PACK: package_bytes,
        TEST: test.encode("utf-8"),
        TRIAL: trial_bytes,
    }


def compact_check() -> dict:
    source_bytes = PYTHON.read_bytes()
    source = source_bytes.decode("utf-8")
    if len(source_bytes) >= 1_000_000:
        raise ValueError("compact source exceeds Kaggle limit")
    attest_kernel_entrypoint(source)
    embedded = embedded_from_source(source)
    if set(embedded) != {"registry_projection", "split", "provenance"}:
        raise ValueError("compact embedded-manifest set is invalid")
    projection, split, provenance = (unpack(embedded[name]) for name in ("registry_projection", "split", "provenance"))
    expected_projection = projection_from_full_registry(read_json(REGISTRY), split, provenance)
    if canonical(projection) != canonical(expected_projection):
        raise ValueError("runtime projection is not the deterministic full-registry projection")
    projection_sha256 = sha256(canonical(projection))
    constant = re.search(r'^REGISTRY_PROJECTION_SHA256 = "([0-9a-f]{64})"$', source, re.MULTILINE)
    if constant is None or constant.group(1) != projection_sha256 or embedded["registry_projection"][1] != projection_sha256:
        raise ValueError("runtime projection hash binding is invalid")
    trial = read_json(TRIAL)
    if trial["immutable_inputs"].get("registry_sha256") != FULL_REGISTRY_SHA256 or trial["immutable_inputs"].get("registry_projection_sha256") != projection_sha256:
        raise ValueError("trial projection binding is invalid")
    notebook = read_json(NOTEBOOK)
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(cells) != 1 or "".join(cells[0].get("source", [])).encode("utf-8") != source_bytes:
        raise ValueError("notebook is not one-cell byte-equivalent to source")
    if notebook.get("metadata", {}).get("t03_script_sha256") != sha256(source_bytes):
        raise ValueError("notebook source identity is invalid")
    meta, package = read_json(META), read_json(PACK)
    if meta.get("trial_sha256") != sha256(TRIAL.read_bytes()) or meta.get("registry_projection_sha256") != projection_sha256:
        raise ValueError("metadata identity is invalid")
    expected_files = {
        "kaggle_kernel/train_nb.py": PYTHON,
        "kaggle_kernel/train_nb.ipynb": NOTEBOOK,
        "kaggle_kernel/tr-meta.json": META,
        "reports/g02/trial-v1.json": TRIAL,
        "tests/test_tnb.py": TEST,
    }
    for name, path in expected_files.items():
        data, record = path.read_bytes(), package["package_files"][name]
        if record != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError(f"package identity is invalid for {name}")
    run = read_json(RUN)
    final, browser = run["final"], run["browser_evidence"]
    expected_v1 = {
        "notebook_sha256": KERNEL_INPUT_HASHES[NOTEBOOK],
        "source_sha256": KERNEL_INPUT_HASHES[PYTHON],
        "trial_sha256": KERNEL_INPUT_HASHES[TRIAL],
    }
    if any(run["package"].get(key) != value for key, value in expected_v1.items()):
        raise ValueError("Version-1 ledger package identity changed")
    if (
        run["launch"].get("status"),
        final.get("status"),
        final.get("steps"),
        final.get("charge_usd"),
        final.get("displayed_runtime_seconds"),
        final.get("error"),
        final.get("run_path"),
        browser.get("active_events"),
        browser.get("displayed_runtime_seconds"),
    ) != ("failed_before_training", "failed_before_training", 0, None, 25, KAGGLE_ERROR, KAGGLE_RUN_PATH, 0, 25):
        raise ValueError("Version-1 failure ledger is invalid")
    armed = ARMED_NEW in source
    if RUN2.exists() != armed:
        raise ValueError("Version-2 ledger presence does not match the armed runtime contract")
    if armed:
        attest_armed_contract(source)
        if RUN2.read_bytes() != canonical(version2_ledger()):
            raise ValueError("Version-2 failure ledger is invalid")
        runtime_contract = {
            "armed": "not dry_run",
            "kaggle_entrypoint": "main([])",
            "local_cpu_step_regression": True,
        }
        zarr_fixed = "ZARR_WHEELS = (" in source
        if RUN3.exists() != zarr_fixed:
            raise ValueError("Version-3 ledger presence does not match the zarr runtime contract")
        if zarr_fixed:
            runtime_contract = runtime_contract_v4() if "RUNTIME_IMPORTS = (" in source else runtime_contract
            if RUN3.read_bytes() != canonical(version3_ledger()):
                raise ValueError("Version-3 failure ledger is invalid")
        if package.get("runtime_contract") != runtime_contract:
            raise ValueError("package armed runtime contract is invalid")
    full = read_json(REGISTRY)
    full_by_group = {item["group_id"]: item for item in full["groups"]}
    projected_by_group = {item["group_id"]: item for item in projection["groups"]}
    semantic_hashes = {}
    for fold in (0, 1, 2):
        fit, forbidden = fold_data(split, provenance, fold)
        for fraction in (0.1, 0.3, 1.0):
            ranked = sorted(fit, key=lambda group: hashlib.sha256(f"17:{group}".encode("ascii")).hexdigest())
            selected = ranked[:max(1, math.ceil(len(ranked) * fraction))]
            full_records = [{"group_id": group, "video_id": full_by_group[group]["video_ids"][0], "stem": full_by_group[group]["video_ids"][0].removeprefix("train:")} for group in selected]
            projected_records = [{"group_id": group, "video_id": projected_by_group[group]["video_ids"][0], "stem": projected_by_group[group]["video_ids"][0].removeprefix("train:")} for group in selected]
            if full_records != projected_records or set(selected) & forbidden:
                raise ValueError(f"semantic resolution mismatch for fold {fold}, fraction {fraction}")
            semantic_hashes[f"fold-{fold}-fraction-{fraction}"] = sha256(canonical(projected_records))
    return {
        "armed_contract": armed,
        "source_bytes": len(source_bytes),
        "notebook_code_bytes": len("".join(cells[0]["source"]).encode("utf-8")),
        "projection_bytes": embedded["registry_projection"][2],
        "projection_group_count": len(projection["groups"]),
        "projection_sha256": projection_sha256,
        "source_registry_sha256": FULL_REGISTRY_SHA256,
        "semantic_record_sha256": semantic_hashes,
    }


def dry_run_check() -> dict:
    source = PYTHON.read_text(encoding="utf-8")
    stub = '''import types
class _Module: pass
class _Cuda:
    @staticmethod
    def is_available(): return False
torch = types.SimpleNamespace(cuda=_Cuda())
nn = types.SimpleNamespace(Module=_Module)
F = types.SimpleNamespace()
def _grad_ckpt(*args, **kwargs): raise RuntimeError("not available in dry-run verification")
'''
    imports = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.checkpoint import checkpoint as _grad_ckpt\n"
    short_imports = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    source = source.replace(imports, "").replace(short_imports, "")
    source = replace_once(source, "from __future__ import annotations\n", "from __future__ import annotations\n" + stub)
    module = types.ModuleType("tnb_dry_check")
    module.__file__ = str(PYTHON)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, str(PYTHON), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    records = []
    with tempfile.TemporaryDirectory(prefix=".t03dry-", dir=ROOT) as temporary:
        root = Path(temporary)
        for fold in (0, 0, 1, 2):
            output = root / f"fold-{fold}-{len(records)}"
            if module.main(["--dry-run", "--fold", str(fold), "--output-dir", str(output)]) != 0:
                raise ValueError(f"dry-run returned nonzero for fold {fold}")
            payload = (output / "prelaunch.json").read_bytes()
            records.append({"fold": fold, "sha256": sha256(payload), "bytes": len(payload)})
    if records[0] != {**records[1], "fold": 0}:
        raise ValueError("repeated fold-0 prelaunch dry-run is not deterministic")
    return {"mode": "torch-free prelaunch shim", "dry_runs": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run-check", action="store_true")
    parser.add_argument("--fix-kargs", action="store_true")
    parser.add_argument("--fix-armed", action="store_true")
    parser.add_argument("--fix-zarr", action="store_true")
    parser.add_argument("--sync-current", action="store_true")
    args = parser.parse_args()
    if sum((args.check, args.dry_run_check, args.fix_kargs, args.fix_armed, args.fix_zarr, args.sync_current)) > 1:
        raise ValueError("choose one verification mode")
    if args.check:
        print(json.dumps(compact_check(), sort_keys=True))
        return
    if args.dry_run_check:
        print(json.dumps(dry_run_check(), sort_keys=True))
        return
    outputs = sync_current_outputs() if args.sync_current else zarr_fix_outputs() if args.fix_zarr else armed_fix_outputs() if args.fix_armed else kernel_fix_outputs() if args.fix_kargs else build()
    for path, data in outputs.items():
        path.write_bytes(data)
    for path, data in outputs.items():
        if path.read_bytes() != data:
            raise RuntimeError(f"write verification failed: {path}")
    if args.fix_kargs or args.fix_armed or args.fix_zarr or args.sync_current:
        compact_check()
    print(json.dumps({str(path.relative_to(ROOT)): {"bytes": len(data), "sha256": sha256(data)} for path, data in sorted(outputs.items())}, sort_keys=True))


if __name__ == "__main__":
    main()

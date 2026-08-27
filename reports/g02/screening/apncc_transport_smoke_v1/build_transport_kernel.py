#!/usr/bin/env python3
"""Build the one-shot, transport-hardened APNCC-direct Kaggle kernel.

This builder intentionally changes only the outer transport.  It extracts the
historical banked runner payload from ``bhap-submit`` and makes the new wrapper
authenticate every declared support input before it lets that runner import a
single non-stdlib module.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BANKED_KERNEL = ROOT / "kaggle_kernel/bhap-submit"
BANKED_IDENTITY_SHA256 = "779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b"
SUPPORT_DATASET = "pilkwang/biohub-tracking-support-pack-50ep-v1"
SUPPORT_MOUNT = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
KERNEL_ID = "onebitbrain/bhap-transport-smoke-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def banked_payloads() -> dict[str, str]:
    source = (BANKED_KERNEL / "bh45.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BANKED_KERNEL / "bh45.py"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_PAYLOADS" for target in node.targets):
            continue
        payloads = ast.literal_eval(node.value)
        if not isinstance(payloads, dict) or set(payloads) != {
            "bh45.py", "bh45att.json", "bh45man.json", "bh45src.zip", "kernel-metadata.json"
        }:
            raise RuntimeError("historical APNCC payload inventory drift")
        if any(not isinstance(value, str) for value in payloads.values()):
            raise RuntimeError("historical APNCC payload encoding drift")
        return payloads
    raise RuntimeError("historical APNCC payload map is absent")


def expected_output() -> dict:
    run = load_json(ROOT / "kaggle_kernel/bhsubmitout_v3/run.json")
    submission = run.get("submission")
    if not isinstance(submission, dict):
        raise RuntimeError("historical APNCC run is missing submission evidence")
    fragments = []
    for path in sorted((ROOT / "kaggle_kernel/bhsubmitout_v3/fragments").glob("*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        fragments.append(
            {
                "filename": path.name,
                "rows": rows,
                "sha256": sha256(path),
            }
        )
    if len(fragments) != 4 or sum(item["rows"] for item in fragments) != int(submission["rows"]):
        raise RuntimeError("historical APNCC fragment evidence drift")
    return {
        "submission": {
            "rows": int(submission["rows"]),
            "edges": int(submission["edges"]),
            "nodes": int(submission["nodes"]),
            "sha256": str(submission["sha256"]),
        },
        "fragments": fragments,
    }


KERNEL_TEMPLATE = r'''#!/usr/bin/env python3
"""One-shot, offline APNCC-direct transport smoke; it never submits a result."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


WORK = Path("/kaggle/working")
PACKAGE = WORK / "bhap_banked_package"
STEP0_RECEIPT = WORK / "APNCC_STEP0.json"
TRANSPORT_RECEIPT = WORK / "APNCC_TRANSPORT_RECEIPT.json"
SUPPORT_MOUNT = Path("__SUPPORT_MOUNT__")
COMPETITION_MOUNTS = (
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
)
EXPECTED_INPUT_MANIFEST_SHA256 = "__BANKED_IDENTITY__"
EXPECTED_CHECKPOINT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
EXPECTED_POPULATION_SHA256 = "__SUBMISSION_SHA__"
EXPECTED_REQUEST_COUNT = __ROW_COUNT__
EXPECTED_REQUEST_FILE_COUNT = __FRAGMENT_COUNT__
EXPECTED_WHEEL_COUNT = 62
OFFLINE_INSTALL_POLICY = ("--no-index", "--find-links")
BANKED_PAYLOADS = __PAYLOADS__
EXPECTED_OUTPUT = __EXPECTED_OUTPUT__


class Step0InputError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key %r" % key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject_duplicates)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def wheel_name_is_canonical(name):
    if not isinstance(name, str) or not name.endswith(".whl") or "/" in name or "\\" in name:
        return False
    # PEP 427: distribution-version(-build)?-python-abi-platform.whl.
    parts = name[:-4].split("-")
    if len(parts) not in (5, 6):
        return False
    distribution, version = parts[0], parts[1]
    tags = parts[-3:]
    if not re.fullmatch(r"[A-Za-z0-9_]+", distribution):
        return False
    if not re.fullmatch(r"[A-Za-z0-9.!+_]+", version):
        return False
    if len(parts) == 6 and not re.fullmatch(r"[0-9][A-Za-z0-9_.]*", parts[2]):
        return False
    return all(re.fullmatch(r"[A-Za-z0-9_.]+", tag) for tag in tags)


def materialize_banked_package():
    if PACKAGE.exists():
        raise Step0InputError("refusing to overwrite an existing banked package directory")
    PACKAGE.mkdir(parents=True, exist_ok=False)
    for name, encoded in BANKED_PAYLOADS.items():
        path = PACKAGE / name
        path.write_bytes(base64.b64decode(encoded.encode("ascii"), validate=True))
    return PACKAGE


def verify_banked_package(package):
    manifest = load_json(package / "bh45man.json")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("identity_sha256") != hashlib.sha256(canonical(identity)).hexdigest():
        raise Step0InputError("banked package manifest identity SHA-256 mismatch")
    if manifest.get("identity_sha256") != EXPECTED_INPUT_MANIFEST_SHA256:
        raise Step0InputError("banked APNCC identity is not the requested 779aa8d3 package")
    observed_bytes = (package / "bh45man.json").stat().st_size
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise Step0InputError("banked package member inventory is absent")
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise Step0InputError("unsafe banked package member name")
        path = package / relative
        if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
            raise Step0InputError("banked package member SHA-256 mismatch: %s" % relative)
        observed_bytes += path.stat().st_size
    if observed_bytes != identity.get("package_bytes") or observed_bytes >= identity.get("budget_bytes", 0):
        raise Step0InputError("banked package byte budget mismatch")
    return load_json(package / "bh45att.json")


def source_tree_sha(root, paths):
    digest = hashlib.sha256()
    for relative in paths:
        path = root / relative
        if not path.is_file():
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def verify_support(attachment):
    identity = attachment.get("identity")
    if not isinstance(identity, dict) or attachment.get("identity_sha256") != hashlib.sha256(canonical(identity)).hexdigest():
        raise Step0InputError("support attachment identity SHA-256 mismatch")
    if identity.get("dataset_ref") != "pilkwang/biohub-tracking-support-pack-50ep-v1":
        raise Step0InputError("support dataset identity drift")
    if not SUPPORT_MOUNT.is_dir():
        raise Step0InputError("declared APNCC support mount is absent: %s" % SUPPORT_MOUNT)
    required = identity.get("required_files")
    source = identity.get("source_tree")
    wheelhouse = identity.get("wheelhouse")
    if not isinstance(required, list) or not isinstance(source, dict) or not isinstance(wheelhouse, dict):
        raise Step0InputError("support attachment inventory is malformed")
    for item in required:
        relative = item.get("path")
        path = SUPPORT_MOUNT / str(relative)
        if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
            raise Step0InputError("declared support input SHA-256 mismatch: %s" % relative)
    if source_tree_sha(SUPPORT_MOUNT, source.get("paths", [])) != source.get("sha256"):
        raise Step0InputError("declared support source-tree SHA-256 mismatch")
    wheel_dir = SUPPORT_MOUNT / str(wheelhouse.get("path"))
    expected = wheelhouse.get("files")
    if not wheel_dir.is_dir() or not isinstance(expected, list) or len(expected) != EXPECTED_WHEEL_COUNT:
        raise Step0InputError("declared canonical wheel inventory is malformed")
    expected_names = sorted(str(item.get("path")) for item in expected)
    actual_names = sorted(path.name for path in wheel_dir.iterdir() if path.is_file() and path.suffix == ".whl")
    if actual_names != expected_names or any(not wheel_name_is_canonical(name) for name in actual_names):
        raise Step0InputError("canonical PEP 427 wheel names drift")
    for item in expected:
        path = wheel_dir / str(item.get("path"))
        if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
            raise Step0InputError("declared offline wheel SHA-256 mismatch: %s" % item.get("path"))
    return wheel_dir


def verify_competition():
    available = [path for path in COMPETITION_MOUNTS if (path / "test").is_dir()]
    if len(available) != 1:
        raise Step0InputError("competition source must resolve at exactly one declared mount")
    root = available[0]
    names = sorted(item["filename"].removeprefix("test__").removesuffix(".csv") + ".zarr" for item in EXPECTED_OUTPUT["fragments"])
    observed = sorted(path.name for path in (root / "test").iterdir() if path.is_dir() and path.name.endswith(".zarr"))
    if observed != names:
        raise Step0InputError("declared test-store set/cardinality drift")
    return root


def run_step0():
    started = time.monotonic()
    try:
        package = materialize_banked_package()
        attachment = verify_banked_package(package)
        wheel_dir = verify_support(attachment)
        competition = verify_competition()
        elapsed = time.monotonic() - started
        if elapsed >= 5.0:
            raise Step0InputError("targeted step 0 exceeded five seconds: %.6f" % elapsed)
        result = {
            "schema": "celltrack-alpha.apncc-transport-step0.v1",
            "status": "PASS_TARGETED_DECLARED_INPUTS",
            "elapsed_seconds": elapsed,
            "dataset_source": "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "dataset_version": 1,
            "dataset_status": "ready",
            "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "population_manifest_sha256": EXPECTED_POPULATION_SHA256,
            "request_count": EXPECTED_REQUEST_COUNT,
            "request_file_count": EXPECTED_REQUEST_FILE_COUNT,
            "population_file_count": EXPECTED_REQUEST_FILE_COUNT,
            "wheel_count": EXPECTED_WHEEL_COUNT,
            "wheel_directory": str(wheel_dir),
            "competition_root": str(competition),
            "banked_package_root": str(package),
            "heavy_work_started": False,
        }
        write_json(STEP0_RECEIPT, result)
        print("APNCC_STEP0=" + json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        result = {
            "schema": "celltrack-alpha.apncc-transport-step0.v1",
            "status": "FAIL_CLOSED",
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "heavy_work_started": False,
        }
        WORK.mkdir(parents=True, exist_ok=True)
        write_json(STEP0_RECEIPT, result)
        print("APNCC_STEP0=" + json.dumps(result, sort_keys=True), flush=True)
        raise


STEP0 = run_step0()


def row_count(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def verify_output():
    observed = {"fragments": []}
    for expected in EXPECTED_OUTPUT["fragments"]:
        path = WORK / "fragments" / expected["filename"]
        actual = {"filename": expected["filename"], "present": path.is_file()}
        if path.is_file():
            actual.update({"rows": row_count(path), "sha256": sha256(path)})
        if actual.get("rows") != expected["rows"] or actual.get("sha256") != expected["sha256"]:
            raise RuntimeError("CV57-parity fragment identity mismatch: %s" % expected["filename"])
        observed["fragments"].append(actual)
    submission = WORK / "submission.csv"
    actual_submission = {
        "present": submission.is_file(),
        "rows": row_count(submission) if submission.is_file() else None,
        "sha256": sha256(submission) if submission.is_file() else None,
    }
    expected_submission = EXPECTED_OUTPUT["submission"]
    if actual_submission["rows"] != expected_submission["rows"] or actual_submission["sha256"] != expected_submission["sha256"]:
        raise RuntimeError("combined submission.csv parity mismatch")
    observed["submission"] = actual_submission
    observed["expected"] = expected_submission
    return observed


def main():
    receipt = {
        "schema": "celltrack-alpha.apncc-transport-terminal.v1",
        "banked_identity_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "submission_action": False,
        "step0": STEP0,
    }
    try:
        completed = subprocess.run([sys.executable, str(PACKAGE / "bh45.py")], cwd=PACKAGE, check=False)
        receipt["banked_runner_returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError("banked APNCC runner failed")
        receipt["output"] = verify_output()
        receipt["status"] = "PASS_VALID_SUBMISSION_TRANSPORT_ONLY"
        write_json(TRANSPORT_RECEIPT, receipt)
        print("APNCC_TRANSPORT=" + json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        receipt.update({"status": "FAIL", "error_type": type(exc).__name__, "error_message": str(exc)})
        write_json(TRANSPORT_RECEIPT, receipt)
        print("APNCC_TRANSPORT=" + json.dumps(receipt, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build() -> dict:
    payloads = banked_payloads()
    manifest = json.loads(base64.b64decode(payloads["bh45man.json"]).decode("ascii"))
    if manifest.get("identity_sha256") != BANKED_IDENTITY_SHA256:
        raise RuntimeError("requested banked APNCC identity is unavailable")
    output = expected_output()
    source = (
        KERNEL_TEMPLATE.replace("__SUPPORT_MOUNT__", SUPPORT_MOUNT)
        .replace("__BANKED_IDENTITY__", BANKED_IDENTITY_SHA256)
        .replace("__SUBMISSION_SHA__", output["submission"]["sha256"])
        .replace("__ROW_COUNT__", str(output["submission"]["rows"]))
        .replace("__FRAGMENT_COUNT__", str(len(output["fragments"])))
        .replace("__PAYLOADS__", repr(payloads))
        .replace("__EXPECTED_OUTPUT__", repr(output))
    )
    compile(source, "kaggle_apncc_transport.py", "exec")
    kernel_dir = HERE / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    code = kernel_dir / "kaggle_apncc_transport.py"
    code.write_text(source, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "bhap transport smoke v1",
        "code_file": code.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [SUPPORT_DATASET],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    contract = {
        "schema": "celltrack-alpha.apncc-transport-contract.v1",
        "kernel_id": KERNEL_ID,
        "remote_preflight_kernel_id": KERNEL_ID,
        "remote_preflight_mode": "inline_step0_before_heavy_work",
        "lane_owner": "APNCC_TRANSPORT_SMOKE_ONLY",
        "dataset_sources": [SUPPORT_DATASET],
        "exclusive_dataset_source": SUPPORT_DATASET,
        "dataset_version": 1,
        "expected_mount_root": SUPPORT_MOUNT,
        "input_manifest_sha256": BANKED_IDENTITY_SHA256,
        "checkpoint_sha256": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
        "population_manifest_sha256": output["submission"]["sha256"],
        "request_count": output["submission"]["rows"],
        "request_file_count": len(output["fragments"]),
        "population_file_count": len(output["fragments"]),
        "wheel_count": 62,
        "step0_limit_seconds": 5.0,
        "code_sha256": sha256(code),
        "kernel_properties": {
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
        },
        "banked_output": output,
    }
    (HERE / "CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    receipt = {
        "schema": "celltrack-alpha.apncc-transport-build.v1",
        "status": "PASS",
        "banked_identity_sha256": BANKED_IDENTITY_SHA256,
        "kernel_id": KERNEL_ID,
        "kernel_sha256": sha256(code),
        "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
        "contract_sha256": sha256(HERE / "CONTRACT.json"),
        "support_dataset": SUPPORT_DATASET,
        "expected_submission": output["submission"],
        "expected_fragments": output["fragments"],
    }
    (HERE / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))

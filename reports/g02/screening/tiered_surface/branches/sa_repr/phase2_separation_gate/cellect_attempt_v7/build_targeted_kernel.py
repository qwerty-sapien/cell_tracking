#!/usr/bin/env python3
"""Build the single-file targeted CELLECT v7 kernel and metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
LEGACY_BUILDER = PARENT / "cellect_bleed_stop_v1/build_selfcontained.py"
BEGIN = "# BEGIN INLINE CONTENT-ADDRESSED STEP 0"
END = "# END INLINE CONTENT-ADDRESSED STEP 0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def load_legacy():
    spec = importlib.util.spec_from_file_location("cellect_legacy_builder", LEGACY_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load legacy CELLECT builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_role(bundle: dict, role: str) -> dict:
    rows = [row for row in bundle["files"] if row.get("role") == role]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {role}, found {len(rows)}")
    return rows[0]


def enriched_contract(bundle: dict, requests: dict, roundtrip: dict) -> dict:
    legacy = load_legacy()
    contract = legacy.build_contract(bundle, requests, roundtrip)
    request_record = one_role(bundle, "request_manifest")
    contract["request_manifest"] = {
        "filename": Path(request_record["path"]).name,
        "bytes": request_record["bytes"],
        "sha256": request_record["sha256"],
    }
    contract["population"]["filename"] = Path(
        one_role(bundle, "population_manifest")["path"]
    ).name
    transport = bundle["checkpoint_transport"]
    contract["checkpoint"]["container_filename"] = Path(
        transport["container_path"]
    ).name
    for actual, embedded in zip(requests["requests"], contract["requests"], strict=True):
        embedded["path"] = actual["path"]
    for wheel in contract["wheels"]:
        try:
            parse_wheel_filename(str(wheel["filename"]))
        except InvalidWheelFilename as error:
            raise RuntimeError(
                f"non-PEP-427 wheel filename: {wheel['filename']}"
            ) from error
    return contract


def render_step0(contract: dict) -> str:
    source = (HERE / "step0_targeted.py").read_text(encoding="utf-8")
    source = source.replace(
        '"__EMBEDDED_ASSET_CONTRACT_JSON__"',
        repr(json.dumps(contract, sort_keys=True, separators=(",", ":"))),
        1,
    )
    source = source.replace(
        '"__EMBEDDED_ASSET_CONTRACT_SHA256__"',
        repr(canonical_sha256(contract)),
        1,
    )
    return f"{BEGIN}\n{source.rstrip()}\n{END}"


def replace_between(source: str, start_marker: str, end_marker: str, value: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    return source[:start] + value + source[end:]


def build(source_kernel: Path, bundle_path: Path, request_path: Path, roundtrip_path: Path, output: Path) -> dict:
    bundle = json.loads(bundle_path.read_text())
    requests = json.loads(request_path.read_text())
    roundtrip = json.loads(roundtrip_path.read_text())
    contract = enriched_contract(bundle, requests, roundtrip)
    generated = source_kernel.read_text(encoding="utf-8")
    generated = replace_between(generated, BEGIN, END, render_step0(contract))

    install_start = generated.index(
        "    subprocess.check_call(\n",
        generated.index('if importlib.util.find_spec("zarr") is None:'),
    )
    install_end = generated.index("\n\nimport zarr", install_start)
    install = '''    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(STEP0["wheel_directory"]),
            "--no-deps",
            "donfig==0.8.1.post1",
            "numcodecs==0.15.1",
            "zarr==3.2.1",
        ]
    )'''
    generated = generated[:install_start] + install + generated[install_end:]

    locate_start = generated.index("def locate_one(pattern: str) -> Path:")
    locate_end = generated.index("\n\ndef action_features", locate_start)
    locate = '''COMPETITION_ROOT = Path(os.environ.get(
    "CELLECT_COMPETITION_ROOT_OVERRIDE",
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
))


def locate_one(pattern: str) -> Path:
    if not pattern.endswith(".zarr") or Path(pattern).name != pattern:
        raise FeatureError(f"invalid declared competition store: {pattern}")
    candidate = COMPETITION_ROOT / "train" / pattern
    if not candidate.is_dir():
        raise FeatureError(f"declared competition store is absent: {candidate}")
    return candidate


def verify_request_record(record: dict) -> Path:
    path = Path(str(record["resolved_path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256(path) != str(record["sha256"])
    ):
        raise FeatureError(
            f"request payload SHA-256 mismatch: {record['action_class']} {record['identity']}"
        )
    return path'''
    generated = generated[:locate_start] + locate + generated[locate_end:]
    old = '        request_path = Path(str(record["resolved_path"]))'
    new = '        request_path = verify_request_record(record)'
    if generated.count(old) != 2:
        raise RuntimeError("request verification insertion count drift")
    generated = generated.replace(old, new)
    generated = generated.replace("ASSET_DATASET_VERSION = 6", "ASSET_DATASET_VERSION = 6")
    compile(generated, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")

    metadata = {
        "id": "onebitbrain/g02-cellect-r7-separation-v7-one-shot",
        "title": "G02 CELLECT R7 separation v7 one shot",
        "code_file": output.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["onebitbrain/g02-cellect-r7-published-assets-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output.parent / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return {
        "schema": "celltrack-alpha.cellect-targeted-kernel-build.v7",
        "status": "PASS",
        "kernel_sha256": sha256(output),
        "metadata_sha256": sha256(output.parent / "kernel-metadata.json"),
        "asset_contract_sha256": canonical_sha256(contract),
        "request_count": contract["request_count"],
        "request_file_count": contract["request_file_count"],
        "population_manifest_sha256": contract["population"]["sha256"],
        "checkpoint_member_sha256": contract["checkpoint"]["sha256"],
        "checkpoint_container_roundtrip_required": False,
        "wheel_filenames": [row["filename"] for row in contract["wheels"]],
        "single_code_file": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-kernel", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-roundtrip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(
        args.source_kernel,
        args.bundle_manifest,
        args.request_manifest,
        args.checkpoint_roundtrip,
        args.output,
    )
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

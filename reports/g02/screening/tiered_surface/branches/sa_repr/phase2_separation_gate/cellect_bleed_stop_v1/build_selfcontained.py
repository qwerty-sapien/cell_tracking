#!/usr/bin/env python3
"""Build the CELLECT v6 kernel with every runtime input pin embedded in code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


BEGIN = "# BEGIN INLINE CONTENT-ADDRESSED STEP 0"
END = "# END INLINE CONTENT-ADDRESSED STEP 0"
CALL_BEGIN = "# Step 0 deliberately authenticates every required input before NumPy, torch,"
HEAVY_IMPORT = "import numpy as np"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def one_role(bundle: dict[str, object], role: str) -> dict[str, object]:
    records = [
        row
        for row in bundle["files"]
        if isinstance(row, dict) and row.get("role") == role
    ]
    if len(records) != 1:
        raise ValueError(f"expected one {role}, found {len(records)}")
    return records[0]


def build_contract(
    bundle: dict[str, object],
    requests: dict[str, object],
    checkpoint_roundtrip: dict[str, object],
) -> dict[str, object]:
    if bundle.get("status") != "LOCAL_CONSOLIDATED_BYTE_EXACT":
        raise ValueError("bundle input is not frozen complete")
    if requests.get("status") != "FROZEN_COMPLETE":
        raise ValueError("request input is not frozen complete")
    request_rows = requests.get("requests")
    if not isinstance(request_rows, list) or len(request_rows) != 77:
        raise ValueError("expected 77 request records")

    bundle_requests = {
        row["sha256"]: row
        for row in bundle["files"]
        if isinstance(row, dict) and row.get("role") == "feature_request"
    }
    embedded_requests: list[dict[str, object]] = []
    total_actions = 0
    seen_semantics: set[tuple[str, str]] = set()
    for ordinal, row in enumerate(request_rows):
        digest = row["sha256"]
        bundle_row = bundle_requests.get(digest)
        if bundle_row is None:
            raise ValueError(f"request {ordinal} absent from bundle records")
        embedded = {
            "ordinal": ordinal,
            "action_class": row["action_class"],
            "identity": row["identity"],
            "action_count": row["action_count"],
            "bytes": row["bytes"],
            "sha256": digest,
        }
        for key in ("action_class", "identity", "action_count", "bytes"):
            if bundle_row.get(key) != embedded[key]:
                raise ValueError(f"request {ordinal} bundle drift: {key}")
        semantic = (str(embedded["action_class"]), str(embedded["identity"]))
        if semantic in seen_semantics:
            raise ValueError("duplicate request semantic identity")
        seen_semantics.add(semantic)
        total_actions += int(embedded["action_count"])
        embedded_requests.append(embedded)
    if set(bundle_requests) != {row["sha256"] for row in embedded_requests}:
        raise ValueError("bundle/request payload sets differ")
    if total_actions != requests["request_count"]:
        raise ValueError("request action total drift")

    wheels = [
        {
            "filename": Path(str(row["path"])).name,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in bundle["files"]
        if isinstance(row, dict) and row.get("role") == "offline_wheel"
    ]
    wheels.sort(key=lambda row: str(row["filename"]))
    if len(wheels) != 3:
        raise ValueError("expected three offline wheels")
    transport = checkpoint_roundtrip["checkpoint_transport"]
    return {
        "schema": "celltrack-alpha.cellect-embedded-runtime-inputs.v1",
        "dataset": "onebitbrain/g02-cellect-r7-published-assets-v1",
        "dataset_version": 6,
        "request_count": total_actions,
        "request_file_count": len(embedded_requests),
        "population": {
            key: one_role(bundle, "population_manifest")[key]
            for key in ("bytes", "sha256")
        },
        "safe_loader": {
            "filename": "cellect_model.py",
            **{
                key: one_role(bundle, "safe_loader")[key]
                for key in ("bytes", "sha256")
            },
        },
        "checkpoint": {
            "filename": transport["member"],
            "bytes": transport["member_bytes"],
            "sha256": transport["member_sha256"],
            "container_bytes": transport["container_bytes"],
            "container_sha256": transport["container_sha256"],
        },
        "wheels": wheels,
        "requests": embedded_requests,
    }


def render_step0(contract: dict[str, object]) -> str:
    contract_text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_sha = canonical_sha256(contract)
    return f'''{BEGIN}
import os
from collections import Counter, defaultdict

EMBEDDED_ASSET_CONTRACT = json.loads({contract_text!r})
EMBEDDED_ASSET_CONTRACT_SHA = "{contract_sha}"


class Step0InputError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _inventory(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise Step0InputError(f"input root is not a directory: {{root}}")
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            by_hash[sha256(candidate)].append(candidate)
    return dict(by_hash)


def _unique_content(
    by_hash: dict[str, list[Path]],
    record: dict[str, object],
    *,
    label: str,
) -> Path:
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise Step0InputError(f"{{label}} has invalid embedded pins")
    matches = by_hash.get(expected_sha256, [])
    if len(matches) != 1:
        raise Step0InputError(
            f"{{label}} content resolution expected one match, found {{len(matches)}}"
        )
    path = matches[0]
    if path.stat().st_size != expected_bytes:
        raise Step0InputError(f"{{label}} byte count drift")
    return path


def _materialize(
    source: Path,
    work_root: Path,
    relative: Path,
    record: dict[str, object],
) -> Path:
    target = work_root / "cellect_step0_materialized" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
        raise Step0InputError(f"materialized input drift: {{relative}}")
    return target


def _checkpoint(
    by_hash: dict[str, list[Path]],
    contract: dict[str, object],
    work_root: Path,
) -> Path:
    record = contract["checkpoint"]
    direct = by_hash.get(str(record["sha256"]), [])
    if len(direct) > 1:
        raise Step0InputError("checkpoint content resolution found duplicates")
    if len(direct) == 1:
        if direct[0].stat().st_size != record["bytes"]:
            raise Step0InputError("checkpoint byte count drift")
        return _materialize(
            direct[0], work_root, Path(str(record["filename"])), record
        )
    containers = by_hash.get(str(record["container_sha256"]), [])
    if len(containers) != 1:
        raise Step0InputError(
            f"checkpoint container resolution expected one match, found {{len(containers)}}"
        )
    if containers[0].stat().st_size != record["container_bytes"]:
        raise Step0InputError("checkpoint container byte count drift")
    matches: list[bytes] = []
    try:
        archive = tarfile.open(containers[0], mode="r:")
    except tarfile.TarError as error:
        raise Step0InputError("checkpoint container is not an uncompressed tar") from error
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise Step0InputError("checkpoint tar member is unreadable")
            payload = handle.read()
            if hashlib.sha256(payload).hexdigest() == record["sha256"]:
                if len(payload) != record["bytes"]:
                    raise Step0InputError("checkpoint tar member byte count drift")
                matches.append(payload)
    if len(matches) != 1:
        raise Step0InputError(
            f"checkpoint tar-member resolution expected one match, found {{len(matches)}}"
        )
    target = work_root / "cellect_step0_materialized" / str(record["filename"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(matches[0])
    if sha256(target) != record["sha256"]:
        raise Step0InputError("checkpoint changed during materialization")
    return target


def run_cellect_step0(input_root: Path, work_root: Path) -> dict[str, object]:
    """Resolve every required input from code-embedded bytes/SHA pins only."""
    started = time.monotonic()
    contract = EMBEDDED_ASSET_CONTRACT
    if _canonical_sha256(contract) != EMBEDDED_ASSET_CONTRACT_SHA:
        raise Step0InputError("embedded asset contract hash drift")
    requests = contract.get("requests")
    if (
        not isinstance(requests, list)
        or len(requests) != contract.get("request_file_count")
        or contract.get("request_file_count") != 77
        or contract.get("request_count") != 1391458
    ):
        raise Step0InputError("embedded frozen request census drift")
    by_hash = _inventory(input_root)

    population = _unique_content(
        by_hash, contract["population"], label="population"
    )
    safe_loader_source = _unique_content(
        by_hash, contract["safe_loader"], label="safe loader"
    )
    safe_loader = _materialize(
        safe_loader_source,
        work_root,
        Path(str(contract["safe_loader"]["filename"])),
        contract["safe_loader"],
    )
    checkpoint = _checkpoint(by_hash, contract, work_root)

    wheels: dict[str, str] = {{}}
    for record in contract["wheels"]:
        source = _unique_content(
            by_hash, record, label=f"offline wheel {{record['filename']}}"
        )
        wheel = _materialize(
            source, work_root, Path("wheels") / str(record["filename"]), record
        )
        wheels[str(record["filename"])] = str(wheel)

    resolved_requests: list[dict[str, object]] = []
    observed_actions = 0
    seen_semantics: set[tuple[str, str]] = set()
    class_files: Counter[str] = Counter()
    class_actions: Counter[str] = Counter()
    for ordinal, record in enumerate(requests):
        action_class = record.get("action_class")
        identity = record.get("identity")
        action_count = record.get("action_count")
        if action_class not in {{"delete", "occupied_swap"}}:
            raise Step0InputError("embedded request class drift")
        if not isinstance(identity, str) or not identity.startswith("train:44b6_"):
            raise Step0InputError("embedded request identity drift")
        if not isinstance(action_count, int) or action_count <= 0:
            raise Step0InputError("embedded request count drift")
        semantic = (action_class, identity)
        if semantic in seen_semantics:
            raise Step0InputError("embedded request semantic duplicate")
        seen_semantics.add(semantic)
        path = _unique_content(
            by_hash, record, label=f"request payload {{ordinal}}"
        )
        resolved_requests.append(
            {{
                "ordinal": ordinal,
                "action_class": action_class,
                "identity": identity,
                "action_count": action_count,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "resolved_path": str(path),
            }}
        )
        observed_actions += action_count
        class_files[action_class] += 1
        class_actions[action_class] += action_count
    if observed_actions != contract["request_count"]:
        raise Step0InputError("resolved request action total drift")
    if dict(class_files) != {{"delete": 41, "occupied_swap": 36}}:
        raise Step0InputError("resolved request class-file census drift")
    if dict(class_actions) != {{"delete": 1345497, "occupied_swap": 45961}}:
        raise Step0InputError("resolved request class-action census drift")

    result = {{
        "schema": "celltrack-alpha.cellect-embedded-step0.v1",
        "status": "PASS_SELF_CONTAINED_EXPANDED_MOUNT",
        "elapsed_seconds": time.monotonic() - started,
        "asset_contract_sha256": EMBEDDED_ASSET_CONTRACT_SHA,
        "inventory_file_count": sum(len(paths) for paths in by_hash.values()),
        "population_manifest_path": str(population),
        "population_manifest_sha256": contract["population"]["sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": contract["checkpoint"]["sha256"],
        "safe_loader_path": str(safe_loader),
        "safe_loader_sha256": contract["safe_loader"]["sha256"],
        "wheels": wheels,
        "request_file_count": len(resolved_requests),
        "request_count": observed_actions,
        "class_file_counts": dict(class_files),
        "class_action_counts": dict(class_actions),
        "request_records": resolved_requests,
        "external_manifest_files_required": 0,
        "heavy_work_started": False,
    }}
    print(json.dumps({{"step": 0, **result}}, sort_keys=True), flush=True)
    return result

{END}'''


def build_kernel(
    source_kernel: Path,
    bundle_manifest: Path,
    request_manifest: Path,
    checkpoint_roundtrip: Path,
    output: Path,
) -> dict[str, object]:
    source = source_kernel.read_text()
    bundle = json.loads(bundle_manifest.read_text())
    requests = json.loads(request_manifest.read_text())
    roundtrip = json.loads(checkpoint_roundtrip.read_text())
    contract = build_contract(bundle, requests, roundtrip)

    start = source.index(BEGIN)
    end = source.index(END, start) + len(END)
    generated = source[:start] + render_step0(contract) + source[end:]
    generated = re.sub(
        r'BUNDLE_MANIFEST_SHA = \(.*?\)\nREQUEST_MANIFEST_SHA = \(.*?\)\n',
        "",
        generated,
        count=1,
        flags=re.DOTALL,
    )
    generated = generated.replace(
        'CELLECT_INPUT_ROOT = Path(\n'
        '    "/kaggle/input/datasets/onebitbrain/g02-cellect-r7-published-assets-v1"\n'
        ')',
        'CELLECT_INPUT_ROOT = Path(os.environ.get(\n'
        '    "CELLECT_INPUT_ROOT_OVERRIDE",\n'
        '    "/kaggle/input/datasets/onebitbrain/g02-cellect-r7-published-assets-v1",\n'
        '))',
        1,
    )
    call_start = generated.index(CALL_BEGIN)
    heavy_start = generated.index(HEAVY_IMPORT, call_start)
    call = '''# Step 0 authenticates every required input from embedded pins before
# NumPy, torch, zarr, dependency installation, model loading, or inference.
STEP0 = (
    run_cellect_step0(
        CELLECT_INPUT_ROOT,
        Path(os.environ.get("CELLECT_WORK_ROOT_OVERRIDE", "/kaggle/working")),
    )
    if __name__ == "__main__"
    else None
)
RESOLVED_WEIGHT = Path(str(STEP0["checkpoint_path"])) if STEP0 is not None else None
if __name__ == "__main__" and os.environ.get("CELLECT_STEP0_ONLY") == "1":
    print(json.dumps({"step0_only": True, "status": STEP0["status"]}, sort_keys=True))
    raise SystemExit(0)

'''
    generated = generated[:call_start] + call + generated[heavy_start:]
    generated = generated.replace("ASSET_DATASET_VERSION = 4", "ASSET_DATASET_VERSION = 6")
    if "BUNDLE_MANIFEST_SHA" in generated or "expected_bundle_manifest" in generated:
        raise ValueError("generated kernel retains an external bundle-manifest pin")
    if "REQUEST_MANIFEST_SHA" in generated or "expected_request_manifest" in generated:
        raise ValueError("generated kernel retains an external request-manifest pin")
    compile(generated, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated)
    return {
        "schema": "celltrack-alpha.cellect-bleed-stop-build.v1",
        "status": "BUILT_SELF_CONTAINED",
        "kernel_sha256": sha256(output),
        "kernel_bytes": output.stat().st_size,
        "embedded_asset_contract_sha256": canonical_sha256(contract),
        "request_file_count": contract["request_file_count"],
        "request_count": contract["request_count"],
        "external_manifest_files_required": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-kernel", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-roundtrip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_kernel(
        args.source_kernel,
        args.bundle_manifest,
        args.request_manifest,
        args.checkpoint_roundtrip,
        args.output,
    )
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

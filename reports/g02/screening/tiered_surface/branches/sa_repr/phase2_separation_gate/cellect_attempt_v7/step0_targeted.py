"""Pure-stdlib CELLECT step 0 that touches declared asset paths only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


EMBEDDED_ASSET_CONTRACT = json.loads("__EMBEDDED_ASSET_CONTRACT_JSON__")
EMBEDDED_ASSET_CONTRACT_SHA256 = "__EMBEDDED_ASSET_CONTRACT_SHA256__"
EXPECTED_POPULATION_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)


class Step0InputError(RuntimeError):
    """A declared transport input failed authentication."""


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


def verified_declared(root: Path, record: dict, label: str) -> Path:
    filename = record.get("filename")
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise Step0InputError(f"{label} has invalid declared pins")
    path = root / filename
    if not path.is_file():
        raise Step0InputError(f"declared {label} is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise Step0InputError(f"declared {label} byte count drift")
    if sha256(path) != expected_sha256:
        raise Step0InputError(f"declared {label} SHA-256 mismatch")
    return path


def materialize_wheel(source: Path, work_root: Path, record: dict) -> Path:
    target = work_root / "cellect_step0_wheels" / str(record["filename"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if (
        target.name != record["filename"]
        or target.stat().st_size != record["bytes"]
        or sha256(target) != record["sha256"]
    ):
        raise Step0InputError(f"materialized wheel mismatch: {record['filename']}")
    return target


def resolve_checkpoint_member(
    root: Path, work_root: Path, record: dict
) -> tuple[Path, str]:
    """Accept raw uncompressed tar or Kaggle's expanded directory; hash the member."""
    filename = str(record["filename"])
    expanded_directory = Path(filename).stem
    direct_candidates = (
        root / filename,
        root / expanded_directory / filename,
    )
    present_direct = [path for path in direct_candidates if path.is_file()]
    container = root / str(record["container_filename"])
    if len(present_direct) > 1:
        raise Step0InputError("checkpoint member has duplicate direct representations")
    payload: bytes | None = None
    source_representation = None
    if present_direct:
        candidate = present_direct[0]
        payload = candidate.read_bytes()
        source_representation = "kaggle_expanded_uncompressed_tar"
    elif container.is_file():
        try:
            archive = tarfile.open(container, mode="r:")
        except tarfile.TarError as error:
            raise Step0InputError("checkpoint transport is not an uncompressed tar") from error
        matches: list[bytes] = []
        with archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name != filename:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise Step0InputError("checkpoint tar member is unreadable")
                matches.append(handle.read())
        if len(matches) != 1:
            raise Step0InputError(
                f"checkpoint tar member expected one match, found {len(matches)}"
            )
        payload = matches[0]
        source_representation = "raw_uncompressed_tar"
    else:
        raise Step0InputError("checkpoint member and uncompressed tar are both absent")

    if len(payload) != record["bytes"]:
        raise Step0InputError("checkpoint member byte count mismatch")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != record["sha256"]:
        raise Step0InputError(
            f"checkpoint member SHA-256 mismatch: expected {record['sha256']}, got {observed}"
        )
    destination = work_root / "cellect_checkpoint_step0" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    if sha256(destination) != record["sha256"]:
        raise Step0InputError("checkpoint member changed during materialization")
    return destination, source_representation


def request_records_from_manifest(root: Path, manifest: dict, contract: dict) -> list[dict]:
    requests = manifest.get("requests")
    request_count = manifest.get("request_count")
    request_file_count = manifest.get("request_file_count")
    if (
        not isinstance(request_count, int)
        or request_count <= 0
        or not isinstance(request_file_count, int)
        or request_file_count <= 0
        or not isinstance(requests, list)
        or len(requests) != request_file_count
        or request_file_count != contract["request_file_count"]
    ):
        raise Step0InputError("request manifest is empty or has invalid counts")
    if manifest.get("population_manifest_sha256") != EXPECTED_POPULATION_SHA256:
        raise Step0InputError("request manifest population SHA-256 drift")
    embedded = contract["requests"]
    if len(embedded) != len(requests):
        raise Step0InputError("embedded and declared request counts differ")
    result = []
    observed_actions = 0
    seen = set()
    for ordinal, (actual, expected) in enumerate(zip(requests, embedded, strict=True)):
        keys = ("action_class", "identity", "action_count", "bytes", "sha256", "path")
        if any(actual.get(key) != expected.get(key) for key in keys):
            raise Step0InputError(f"request manifest row {ordinal} disagrees with embedded pins")
        semantic = (actual["action_class"], actual["identity"])
        if semantic in seen or actual["action_count"] <= 0:
            raise Step0InputError("request semantic duplicate or nonpositive count")
        seen.add(semantic)
        observed_actions += actual["action_count"]
        result.append(
            {
                "ordinal": ordinal,
                "action_class": actual["action_class"],
                "identity": actual["identity"],
                "action_count": actual["action_count"],
                "bytes": actual["bytes"],
                "sha256": actual["sha256"],
                "resolved_path": str(root / actual["path"]),
            }
        )
    if observed_actions != request_count or request_count != contract["request_count"]:
        raise Step0InputError("request_count disagrees with request rows")
    return result


def run_cellect_step0(input_root: Path, work_root: Path) -> dict[str, object]:
    """Authenticate only checkpoint, manifests, and canonical wheels before heavy work."""
    started = time.monotonic()
    contract = EMBEDDED_ASSET_CONTRACT
    if canonical_sha256(contract) != EMBEDDED_ASSET_CONTRACT_SHA256:
        raise Step0InputError("embedded asset contract SHA-256 drift")
    if not input_root.is_dir():
        raise Step0InputError(f"declared CELLECT asset root is absent: {input_root}")

    request_manifest_path = verified_declared(
        input_root, contract["request_manifest"], "request manifest"
    )
    population_manifest_path = verified_declared(
        input_root, contract["population"], "population manifest"
    )
    if contract["population"]["sha256"] != EXPECTED_POPULATION_SHA256:
        raise Step0InputError("frozen population manifest SHA-256 drift")
    request_manifest = json.loads(request_manifest_path.read_text(encoding="ascii"))
    request_records = request_records_from_manifest(
        input_root, request_manifest, contract
    )

    wheels: dict[str, str] = {}
    for record in contract["wheels"]:
        source = verified_declared(
            input_root, record, f"offline wheel {record['filename']}"
        )
        target = materialize_wheel(source, work_root, record)
        wheels[str(record["filename"])] = str(target)

    checkpoint, checkpoint_representation = resolve_checkpoint_member(
        input_root, work_root, contract["checkpoint"]
    )
    elapsed = time.monotonic() - started
    if elapsed >= 5.0:
        raise Step0InputError(f"targeted step 0 exceeded five seconds: {elapsed:.6f}")
    result = {
        "schema": "celltrack-alpha.cellect-targeted-step0.v7",
        "status": "PASS_TARGETED_DECLARED_INPUTS",
        "elapsed_seconds": elapsed,
        "asset_contract_sha256": EMBEDDED_ASSET_CONTRACT_SHA256,
        "declared_paths_hashed": 6,
        "competition_paths_visited": 0,
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": contract["request_manifest"]["sha256"],
        "population_manifest_path": str(population_manifest_path),
        "population_manifest_sha256": EXPECTED_POPULATION_SHA256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": contract["checkpoint"]["sha256"],
        "checkpoint_representation": checkpoint_representation,
        "checkpoint_container_sha256_verified": False,
        "checkpoint_member_sha256_verified": True,
        "wheels": wheels,
        "wheel_directory": str(Path(next(iter(wheels.values()))).parent),
        "safe_loader_path": str(input_root / contract["safe_loader"]["filename"]),
        "request_file_count": len(request_records),
        "request_count": request_manifest["request_count"],
        "request_records": request_records,
        "request_payloads_hashed_in_step0": 0,
        "heavy_work_started": False,
    }
    log_record = {key: value for key, value in result.items() if key != "request_records"}
    print(json.dumps({"step": 0, **log_record}, sort_keys=True), flush=True)
    return result

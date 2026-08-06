#!/usr/bin/env python3
"""Build the deterministic uncompressed PEFT input dataset transport."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PEFT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1"
)
POPULATION_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/"
    "phase2_separation_gate/run_v1"
)
ASSIGNMENT_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/"
    "sa_assignment_separation_v1"
)
POPULATION_MANIFEST_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
ASSIGNMENT_MANIFEST_SHA256 = (
    "41c6d31378e8f7a1e5afbea8d05274c82dffe2100497c8d47e887dc83c3ecb2b"
)
GRAPH_RESULT_SHA256 = (
    "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a"
)
CHECKPOINT_AUTH_SHA256 = (
    "8a6d2f61a23bad83042dcb61e1fadacf9febd1450890fafcc57357fb9aa06633"
)
SEALED_B0_SHA256 = (
    "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2"
)
EXPECTED_ACTIONS = 1_345_497
EXPECTED_BENEFICIAL = 1_118
EXPECTED_WHEEL_NAME = "donfig-0.8.1.post1-py3-none-any.whl"
EXPECTED_WHEEL_SHA256 = (
    "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
)


class BuildError(RuntimeError):
    """The generated transport would not satisfy its frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def add_file(
    files: dict[str, tuple[Path, str]],
    relative: str,
    source: Path,
    role: str,
) -> None:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != relative
        or relative in files
        or not source.is_file()
    ):
        raise BuildError(f"invalid bundle member {relative}: {source}")
    files[relative] = (source, role)


def build(output: Path, wheel: Path) -> dict[str, Any]:
    if output.exists():
        raise BuildError(f"refusing to overwrite existing output directory: {output}")
    if (
        wheel.name != EXPECTED_WHEEL_NAME
        or not wheel.is_file()
        or sha256(wheel) != EXPECTED_WHEEL_SHA256
    ):
        raise BuildError("declared canonical wheel identity drifted")
    output.mkdir(parents=True)
    amendment = PEFT / "RUNTIME_MARGIN_AMENDMENT_20260731.json"
    population_manifest_path = POPULATION_ROOT / "population_manifest.json"
    assignment_manifest_path = ASSIGNMENT_ROOT / "FEATURE_MANIFEST_V2.json"
    graph_result = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_delete/"
        "graph_result.json"
    )
    checkpoint_auth = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "sa_reverse_authentic/CHECKPOINT_AUTHENTICATION.json"
    )
    sealed_b0 = POPULATION_ROOT.parent / "run_v3" / "B0_SEALED.json"
    preregistration = PEFT / "PREREGISTRATION.json"
    expected_controls = {
        population_manifest_path: POPULATION_MANIFEST_SHA256,
        assignment_manifest_path: ASSIGNMENT_MANIFEST_SHA256,
        graph_result: GRAPH_RESULT_SHA256,
        checkpoint_auth: CHECKPOINT_AUTH_SHA256,
        sealed_b0: SEALED_B0_SHA256,
    }
    for path, expected in expected_controls.items():
        observed = sha256(path)
        if observed != expected:
            raise BuildError(
                f"frozen control drifted: {path}; expected {expected}, got {observed}"
            )

    population_manifest = json.loads(population_manifest_path.read_text())
    assignment_manifest = json.loads(assignment_manifest_path.read_text())
    population_rows = population_manifest["classes"]["delete"]["videos"]
    feature_rows = {
        str(row["identity"]): row
        for row in assignment_manifest["feature_files"]
        if row["action_class"] == "delete"
    }
    if (
        sum(int(row["count"]) for row in population_rows) != EXPECTED_ACTIONS
        or sum(int(row["beneficial"]) for row in population_rows)
        != EXPECTED_BENEFICIAL
        or len(population_rows) != 41
        or len(feature_rows) != 41
    ):
        raise BuildError("frozen DELETE population census drifted")

    files: dict[str, tuple[Path, str]] = {}
    for relative, source, role in (
        ("amendment.json", amendment, "preregistration_amendment"),
        (
            "population_manifest.json",
            population_manifest_path,
            "population_manifest",
        ),
        (
            "assignment_feature_manifest.json",
            assignment_manifest_path,
            "frozen_representation_manifest",
        ),
        ("graph_result.json", graph_result, "frozen_graph_baselines"),
        (
            "checkpoint_authentication.json",
            checkpoint_auth,
            "frozen_backbone_authentication",
        ),
        ("sealed_b0.json", sealed_b0, "sealed_baseline"),
        ("original_preregistration.json", preregistration, "original_preregistration"),
        (
            f"wheels/{EXPECTED_WHEEL_NAME}",
            wheel,
            "canonical_offline_environment_wheel",
        ),
    ):
        add_file(files, relative, source, role)

    for row in population_rows:
        identity = str(row["identity"])
        stem = identity.replace(":", "__", 1)
        population_path = ROOT / str(row["population_path"])
        feature_path = ROOT / str(feature_rows[identity]["path"])
        if (
            sha256(population_path) != row["population_sha256"]
            or sha256(feature_path) != feature_rows[identity]["sha256"]
        ):
            raise BuildError(f"population/feature member drifted: {identity}")
        add_file(
            files,
            f"population/delete/{stem}.npz",
            population_path,
            "full_delete_population",
        )
        add_file(
            files,
            f"features/delete/{stem}.npz",
            feature_path,
            "frozen_assignment_representation",
        )

    members = [
        {
            "path": relative,
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
            "role": role,
        }
        for relative, (source, role) in sorted(files.items())
    ]
    manifest = {
        "schema": "celltrack-alpha.peft-delete-assets.v1",
        "status": "COMPLETE_DECLARED_INPUTS_ONLY",
        "dataset": "onebitbrain/g02-peft-delete-assets-v1",
        "transport": {
            "format": "uncompressed tar or Kaggle-expanded directory",
            "container_roundtrip_required": False,
            "verification": "member SHA-256",
        },
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "request_count": EXPECTED_ACTIONS,
        "request_file_count": len(population_rows),
        "population_file_count": len(population_rows),
        "wheel_count": 1,
        "competition_members": 0,
        "members": members,
    }
    manifest_bytes = canonical_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = output / "asset_manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    archive_path = output / "peft_bundle.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        manifest_info = tarfile.TarInfo("peft_bundle/asset_manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = 0
        manifest_info.uid = manifest_info.gid = 0
        manifest_info.uname = manifest_info.gname = ""
        manifest_info.mode = 0o644
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for relative, (source, _role) in sorted(files.items()):
            info = tarfile.TarInfo(f"peft_bundle/{relative}")
            info.size = source.stat().st_size
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with source.open("rb") as handle:
                archive.addfile(info, handle)

    metadata = {
        "title": "G02 PEFT DELETE assets v1",
        "id": "onebitbrain/g02-peft-delete-assets-v1",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (output / "dataset-metadata.json").write_bytes(canonical_bytes(metadata))
    receipt = {
        "schema": "celltrack-alpha.peft-delete-assets-build.v1",
        "status": "COMPLETE_LOCAL_NOT_UPLOADED",
        "output": str(output),
        "asset_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "bytes": len(manifest_bytes),
        },
        "transport": {
            "path": str(archive_path),
            "sha256": sha256(archive_path),
            "bytes": archive_path.stat().st_size,
            "compression": "none",
        },
        "member_count": len(members),
        "request_count": EXPECTED_ACTIONS,
        "dataset_mutations": 0,
        "kernel_launches": 0,
    }
    (output / "BUILD_RECEIPT.json").write_bytes(canonical_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.output.resolve(), args.wheel.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

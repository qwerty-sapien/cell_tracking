#!/usr/bin/env python3
"""Build the withdrawn selector CPU port as a member-addressed tar transport."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
POP_SHA = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
GZIP_POPULATION_PATH = (
    "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
    "generated/action_population.jsonl.gz"
)
TRANSPORT_ONLY = {
    HERE / "build_kaggle_cpu_bundle.py",
    HERE / "verify_kaggle_cpu_bundle_remote.py",
    HERE / "test_kaggle_selector_transport.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object JSON required: {path}")
    return value


def repo_path(value: str) -> Path:
    path = Path(value)
    original = Path("/Users/agent/Desktop/cell_tracking")
    if path.is_absolute():
        path = path.relative_to(original)
    return ROOT / path


def add(files: set[Path], path: Path) -> None:
    path = path.resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise RuntimeError(f"required bundle file missing: {path}")
    files.add(path)


def add_tar_member(archive: tarfile.TarFile, source: Path, name: str) -> None:
    """Add one deterministic regular member to an uncompressed POSIX tar."""
    member = Path(name)
    if (
        not name
        or not name.isascii()
        or member.is_absolute()
        or ".." in member.parts
        or not source.is_file()
    ):
        raise RuntimeError(f"unsafe tar member: {name!r}")
    info = tarfile.TarInfo(name)
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as payload:
        archive.addfile(info, payload)


def write_uncompressed_tar(
    archive_path: Path,
    entries: list[tuple[str, Path]],
    inventory_bytes: bytes,
) -> None:
    """Write the upload container; remote authentication is member-only."""
    with tarfile.open(
        archive_path, mode="w:", format=tarfile.PAX_FORMAT
    ) as archive:
        for name, source in entries:
            add_tar_member(archive, source, name)
        inventory = tarfile.TarInfo("selector_cpu_bundle_inventory.json")
        inventory.size = len(inventory_bytes)
        inventory.mode = 0o644
        inventory.mtime = 0
        inventory.uid = 0
        inventory.gid = 0
        inventory.uname = ""
        inventory.gname = ""
        import io

        archive.addfile(inventory, io.BytesIO(inventory_bytes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files: set[Path] = set()

    for directory in (
        HERE,
        ROOT / "src",
        ROOT / "external/Tracking_Support/repo/src",
        ROOT / "external/kaggle-cell-tracking-competition/src",
        ROOT / "external/Tracking_Support/wheels",
    ):
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and "kaggle_cpu_kernel" not in path.parts
                and path not in TRANSPORT_ONLY
            ):
                add(files, path)
    add(files, ROOT / "scripts/scoreall.py")

    population_path = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_repr/"
        "phase2_separation_gate/run_v1/population_manifest.json"
    )
    if sha256(population_path) != POP_SHA:
        raise RuntimeError("population manifest drifted")
    population = load(population_path)
    add(files, population_path)
    for action_class in ("delete", "occupied_swap"):
        for row in population["classes"][action_class]["videos"]:
            add(files, repo_path(str(row["population_path"])))

    fixed = (
        "reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json",
        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "reports/g02/screening/tiered_surface/branches/sa_swap/PREREGISTRATION.json",
        "reports/g02/screening/tiered_surface/branches/sa_swap/EXECUTION_REPAIR.json",
        "reports/g02/screening/tiered_surface/branches/sa_swap/N1_LOCALITY_EQUIVALENCE.json",
        "reports/g02/screening/tiered_surface/branches/sa_swap/result_44b6_postrepair_v1/report.json",
        "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo/evaluate_nodetopo_44b6_graph_gate.py",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo/graph_gate_44b6/report.json",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/PREREGISTRATION.json",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/generated/FALSE_NODE_STRESS_CURVE.json",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/generated/action_population.jsonl.gz",
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/generated/raw_universe/manifest.json",
        "reports/g02/fitcache_117/manifest.json",
        "reports/g02/grouped_3fold_audit20_v1.json",
        "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "reports/g02/screening/h1_existing_cache_feature_report.json",
        "reports/g02/screening/h1_existing_cache_run_manifest.json",
        "reports/g02/screening/h1_cache_attestation.json",
        "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json",
        "reports/g02/appearance_direct_swap_contract_v1.json"
    )
    for relative in fixed:
        add(files, ROOT / relative)

    p0 = load(ROOT / fixed[1])
    delete_root = repo_path(str(p0["control"]["fragment_root"]))
    for row in population["classes"]["delete"]["videos"]:
        add(files, delete_root / f"{str(row['identity']).replace(':', '__', 1)}.csv")

    apncc = load(ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json")
    for row in apncc["graph"]["control"]["input_provenance"]["files"]:
        add(files, repo_path(str(row["path"])))

    raw_manifest = load(
        ROOT
        / "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
        "generated/raw_universe/manifest.json"
    )
    baseline_root = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
    for row in raw_manifest["videos"]:
        add(
            files,
            baseline_root / f"{str(row['identity']).replace(':', '__', 1)}.csv",
        )

    inventory = []
    archive = output / "selector_cpu_bundle.tar"
    with tempfile.TemporaryDirectory(
        prefix="selector-transport-", dir=output
    ) as temporary:
        transport_root = Path(temporary)
        entries: list[tuple[str, Path]] = []
        for path in sorted(files):
            source_relative = path.relative_to(ROOT).as_posix()
            relative = source_relative
            transport_path = path
            transport = "byte_exact_regular_member"
            if source_relative == GZIP_POPULATION_PATH:
                relative = source_relative.removesuffix(".gz")
                transport_path = transport_root / "action_population.jsonl"
                with gzip.open(path, "rb") as source, transport_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                transport = "gzip_payload_as_plain_member"
            record = {
                "path": relative,
                "bytes": transport_path.stat().st_size,
                "sha256": sha256(transport_path),
                "transport": transport,
            }
            if source_relative != relative:
                record["source_path"] = source_relative
            inventory.append(record)
            entries.append((relative, transport_path))
        inventory_bytes = (
            json.dumps(
                {
                    "schema": "celltrack-alpha.selector-cpu-bundle-inventory.v2",
                    "verification": "member_sha256_only",
                    "files": inventory,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        write_uncompressed_tar(archive, entries, inventory_bytes)

    required_videos = sorted(
        {
            str(row["identity"])
            for action_class in ("delete", "occupied_swap")
            for row in population["classes"][action_class]["videos"]
        }
        | {str(row["identity"]) for row in raw_manifest["videos"]}
    )
    manifest = {
        "schema": "celltrack-alpha.selector-kaggle-cpu-bundle.v2",
        "status": "FROZEN_BEFORE_KAGGLE_CPU_OUTCOMES",
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "format": "uncompressed_tar",
            "compression": "none",
            "verification": "member_sha256_only",
            "member_count": len(inventory) + 1,
        },
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "population_manifest_sha256": POP_SHA,
        "population_action_count": sum(
            int(row["count"])
            for name in ("delete", "occupied_swap")
            for row in population["classes"][name]["videos"]
        ),
        "wheel_count": sum(row["path"].endswith(".whl") for row in inventory),
        "required_train_video_ids": required_videos,
        "competition_source": "biohub-cell-tracking-during-development",
        "enable_gpu": False,
        "ram_gib": 16,
    }
    manifest_path = output / "selector_cpu_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "G02 Selector Requirement CPU Inputs v1",
                "id": "onebitbrain/g02-selector-requirement-cpu-inputs-v1",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

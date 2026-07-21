#!/usr/bin/env python3
"""Build the alpha Kaggle package and its deterministic parity manifest.

The competition kernel must execute the promoted ``celltrack_alpha`` path.
``celltrack_baseline`` is copied only because alpha imports its image primitives
and CSV utilities; ``rule_based_submission.py`` is deliberately not an entry
point in the resulting package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel"
REPORT_PATH = ROOT / "reports/kaggle_g01/package_parity_manifest.json"
ENTRYPOINT = "alpha_submission.py"

# Only modules reachable from the submission entrypoint are packaged.  In
# particular, official local-evaluation helpers (and their optional packages)
# are not part of the Kaggle runtime surface.
PACKAGE_SOURCES = (
    "celltrack_alpha/__init__.py",
    "celltrack_alpha/types.py",
    "celltrack_alpha/proposals.py",
    "celltrack_alpha/pipeline.py",
    "celltrack_alpha/routing.py",
    "celltrack_alpha/scoring.py",
    "celltrack_alpha/solver.py",
    "celltrack_alpha/localization.py",
    "celltrack_baseline/__init__.py",
    "celltrack_baseline/rule_based.py",
)

BUNDLED_MODULE_ORDER = (
    ("celltrack_baseline", "celltrack_baseline/__init__.py", True),
    ("celltrack_baseline.rule_based", "celltrack_baseline/rule_based.py", False),
    ("celltrack_alpha.types", "celltrack_alpha/types.py", False),
    ("celltrack_alpha.localization", "celltrack_alpha/localization.py", False),
    ("celltrack_alpha.scoring", "celltrack_alpha/scoring.py", False),
    ("celltrack_alpha.solver", "celltrack_alpha/solver.py", False),
    ("celltrack_alpha.proposals", "celltrack_alpha/proposals.py", False),
    ("celltrack_alpha.routing", "celltrack_alpha/routing.py", False),
    ("celltrack_alpha.pipeline", "celltrack_alpha/pipeline.py", False),
    ("celltrack_alpha", "celltrack_alpha/__init__.py", True),
)

ALPHA_CONFIG = {
    "model_package": "celltrack_alpha",
    "route": "auto",
    "allow_division_route": False,
    "output_offset_override": None,
    "scoring_calibration": None,
    "calibration_policy": "No fold-specific calibration is applied to Kaggle test data.",
    "submission_split": "test",
}

KERNEL_ENTRYPOINT = r'''#!/usr/bin/env python3
"""Kaggle entrypoint for the current scenario-routed celltrack alpha."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

try:
    if os.environ.get("ALPHA_FORCE_FALLBACK_ZARR"):
        raise ModuleNotFoundError("forced fallback")
    import zarr
except ModuleNotFoundError:
    zarr = None

# __BUNDLED_MODULE_BOOTSTRAP__

from celltrack_alpha import predict_dataset_window
from celltrack_baseline.rule_based import submission_rows, validate_submission_rows, write_submission


ROOT = Path(__file__).resolve().parent
COMP_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/biohub-cell-tracking-during-development")
OUTPUT_DIR = Path("/kaggle/working")
CONFIG = __ALPHA_CONFIG_LITERAL__


def dataset_names() -> list[str]:
    return sorted(path.stem for path in (COMP_DIR / "test").glob("*.zarr"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decompress_blosc(data: bytes) -> bytes:
    try:
        import blosc2

        return blosc2.decompress(data)
    except ModuleNotFoundError:
        pass
    try:
        from numcodecs import blosc

        return blosc.decompress(data)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Kaggle image has neither zarr nor a Blosc decompressor; "
            "cannot read compressed competition Zarr without internet or vendored wheels."
        ) from exc


class _FallbackArray:
    def __init__(self, root: Path):
        self.root = root
        metadata = _load_json(root / "zarr.json")
        self.shape = tuple(int(v) for v in metadata["shape"])
        self.chunks = tuple(
            int(v)
            for v in metadata["chunk_grid"]["configuration"]["chunk_shape"]
        )
        dtype_name = str(metadata["data_type"])
        if dtype_name != "uint16":
            raise ValueError(f"fallback reader only supports uint16 arrays, got {dtype_name!r}")
        self.dtype = np.dtype("<u2")
        codecs = [codec.get("name") for codec in metadata.get("codecs", [])]
        if codecs != ["bytes", "blosc"]:
            raise ValueError(f"fallback reader expected bytes+blosc codecs, got {codecs!r}")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 1:
                raise TypeError("fallback reader supports integer timepoint indexing only")
            key = key[0]
        if not isinstance(key, (int, np.integer)):
            raise TypeError("fallback reader supports integer timepoint indexing only")
        t = int(key)
        if t < 0:
            t += self.shape[0]
        if t < 0 or t >= self.shape[0]:
            raise IndexError(t)
        chunk_path = self.root / "c" / str(t) / "0" / "0" / "0"
        raw = _decompress_blosc(chunk_path.read_bytes())
        volume = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks[1:4])
        return volume


class _FallbackGroup:
    def __init__(self, root: Path):
        self.root = root
        self.attrs = _load_json(root / "zarr.json").get("attributes", {})

    def __getitem__(self, key: str):
        if key != "0":
            raise KeyError(key)
        return _FallbackArray(self.root / "0")


def open_test_dataset(dataset: str):
    path = COMP_DIR / "test" / f"{dataset}.zarr"
    if zarr is not None:
        group = zarr.open_group(str(path), mode="r")
        return group, group["0"]
    group = _FallbackGroup(path)
    return group, group["0"]


def main() -> None:
    datasets = dataset_names()
    if not datasets:
        raise RuntimeError(f"No test .zarr datasets found beneath {COMP_DIR}")
    print(json.dumps({"alpha_submission_config": CONFIG, "datasets": datasets}, indent=2))
    predictions = []
    for dataset in datasets:
        group, array = open_test_dataset(dataset)
        result = predict_dataset_window(
            dataset=dataset,
            array=array,
            attrs=dict(group.attrs),
            start=0,
            end=int(array.shape[0]),
            forced_route=str(CONFIG["route"]),
            allow_division_route=bool(CONFIG["allow_division_route"]),
            output_offset_override=None,
            estimated_total_nodes=None,
            scoring_calibration=None,
            progress=True,
        )
        predictions.append((dataset, result.nodes, result.edges))
        print(f"{dataset}: route={result.route.label} nodes={len(result.nodes)} edges={len(result.edges)}")
    rows = submission_rows(predictions)
    validation = validate_submission_rows(rows)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise RuntimeError("Submission validation failed")
    write_submission(rows, OUTPUT_DIR / "submission.csv")
    print("Wrote /kaggle/working/submission.csv")


if __name__ == "__main__":
    main()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_module_bootstrap() -> str:
    """Return source that installs alpha modules from this single script."""

    entries: list[tuple[str, str, bool]] = []
    for module_name, relative, is_package in BUNDLED_MODULE_ORDER:
        source = (ROOT / "src" / relative).read_text(encoding="utf-8")
        entries.append((module_name, source, is_package))
    lines = [
        "def _install_bundled_modules() -> None:",
        "    bundled_sources = [",
    ]
    for module_name, source, is_package in entries:
        lines.append(f"        ({module_name!r}, {source!r}, {is_package!r}),")
    lines.extend(
        [
            "    ]",
            "    for module_name, source, is_package in bundled_sources:",
            "        if '.' in module_name:",
            "            package_name = module_name.split('.', 1)[0]",
            "            if package_name not in sys.modules:",
            "                package = types.ModuleType(package_name)",
            "                package.__file__ = f'<bundled {package_name}>'",
            "                package.__package__ = package_name",
            "                package.__path__ = []",
            "                sys.modules[package_name] = package",
            "        module = sys.modules.get(module_name)",
            "        if module is None:",
            "            module = types.ModuleType(module_name)",
            "            sys.modules[module_name] = module",
            "        module.__file__ = f'<bundled {module_name}>'",
            "        module.__package__ = module_name if is_package else module_name.rpartition('.')[0]",
            "        if is_package:",
            "            module.__path__ = []",
            "        exec(compile(source, module.__file__, 'exec'), module.__dict__)",
            "",
            "_install_bundled_modules()",
        ]
    )
    return "\n".join(lines)


def copy_source(relative: str) -> dict[str, str]:
    source = ROOT / "src" / relative
    destination = KERNEL_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    package_hash = sha256_file(destination)
    if source_hash != package_hash:
        raise RuntimeError(f"copy hash mismatch for {relative}")
    return {"path": relative, "source_sha256": source_hash, "package_sha256": package_hash}


def build_manifest() -> dict[str, object]:
    # Bytecode is host/interpreter-specific and does not contribute to the
    # package identity.  Remove stale caches so a push contains source only.
    for cache_dir in KERNEL_DIR.rglob("__pycache__"):
        shutil.rmtree(cache_dir)
    files = [copy_source(relative) for relative in PACKAGE_SOURCES]
    entrypoint_source = KERNEL_ENTRYPOINT.replace(
        "# __BUNDLED_MODULE_BOOTSTRAP__",
        build_module_bootstrap(),
    ).replace(
        "__ALPHA_CONFIG_LITERAL__",
        repr(ALPHA_CONFIG),
    )
    (KERNEL_DIR / ENTRYPOINT).write_text(entrypoint_source, encoding="utf-8")
    (KERNEL_DIR / "alpha_submission_config.json").write_text(
        json.dumps(ALPHA_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    old_baseline_entrypoint = KERNEL_DIR / "rule_based_submission.py"
    if old_baseline_entrypoint.exists():
        old_baseline_entrypoint.unlink()

    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = "onebitbrain/biohub-celltrack-alpha"
    metadata["title"] = "biohub-celltrack-alpha"
    metadata["code_file"] = ENTRYPOINT
    metadata["enable_gpu"] = False
    metadata["enable_internet"] = False
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    entrypoint_hash = sha256_file(KERNEL_DIR / ENTRYPOINT)
    config_hash = sha256_file(KERNEL_DIR / "alpha_submission_config.json")
    identity_payload = {
        "entrypoint": ENTRYPOINT,
        "entrypoint_sha256": entrypoint_hash,
        "config_sha256": config_hash,
        "files": files,
        "metadata_sha256": sha256_file(metadata_path),
        "bundled_modules": [module_name for module_name, _, _ in BUNDLED_MODULE_ORDER],
    }
    return {
        "schema_version": 1,
        "artifact_kind": "kaggle_alpha_package",
        "package_identity_sha256": canonical_sha256(identity_payload),
        "identity_inputs": identity_payload,
        "entrypoint": ENTRYPOINT,
        "entrypoint_import": "from celltrack_alpha import predict_dataset_window",
        "route_config": ALPHA_CONFIG,
        "baseline_dependency": {
            "package": "celltrack_baseline",
            "role": "shared image primitives and CSV schema utilities only",
            "not_submission_entrypoint": True,
        },
        "excluded_entrypoints": ["rule_based_submission.py"],
        "submission_rows_included": False,
        "row_independent_identity": True,
        "network_or_kaggle_actions": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    package_manifest_path = KERNEL_DIR / "package_manifest.json"
    package_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {package_manifest_path}")
    print(f"Wrote {args.report}")
    print(f"package_identity_sha256={manifest['package_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

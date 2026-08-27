#!/usr/bin/env python3
"""Freeze, package, merge, and score the public TemporalUNet3D baseline.

This harness never trains or mutates a checkpoint.  It derives a self-contained
Kaggle script from the immutable public Version-32 notebook and exposes three
observation-distinct inference modes plus one explicitly labelled graph-output
experiment:

``single_seed``
    Primary public checkpoint only.
``detection_blend``
    50/50 aligned detection logits; primary association logits only.
``public_v32``
    Exact Version-32 settings: 50/50 detection plus the uncertainty-gated
    secondary association view.
``public_v32_cap2``
    The exact Version-32 settings followed by an experimental outdegree-two
    cap that retains the two highest-ranked daughters under V32's own
    ``edge_sort_key``.  It is not part of the replication baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NOTEBOOK = (
    ROOT
    / "external/public_temporalunet_baseline/pilkwang_two_seed_v32/"
    "biohub-cell-tracking-two-seeds-logit-blend.ipynb"
)
PUBLIC_NOTEBOOK_SHA256 = "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d"
PRIMARY_MANIFEST_SHA256 = "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103"
PRIMARY_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
SECONDARY_MANIFEST_SHA256 = "7f01c0b2f9491606a1b543339e072230e25317d9c70fb9a18aea59e002a995eb"
SECONDARY_WEIGHT_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
PRIMARY_SOURCE = "pilkwang/biohub-tracking-support-pack-50ep-v1"
SECONDARY_SOURCE = "pilkwang/biohub-temporal-unet3d-seed314159-v1"
COMPETITION = "biohub-cell-tracking-during-development"
PRIMARY_ROOT = f"/kaggle/input/datasets/{PRIMARY_SOURCE}"
SECONDARY_ROOT = f"/kaggle/input/datasets/{SECONDARY_SOURCE}"
COMPETITION_TRAIN = f"/kaggle/input/competitions/{COMPETITION}/train"
COMPETITION_TEST = f"/kaggle/input/competitions/{COMPETITION}/test"
# Kaggle mounts competition data under either convention depending on how the
# kernel is attached; the hidden rerun is not guaranteed to use the same one as
# the public run.  Order is preference order, not an assertion of existence.
COMPETITION_TEST_CANDIDATES = (
    f"/kaggle/input/competitions/{COMPETITION}/test",
    f"/kaggle/input/{COMPETITION}/test",
)
POPULATION_SCHEMA = "celltrack-alpha.public-temporalunet-populations.v1"
MERGE_SCHEMA = "celltrack-alpha.public-temporalunet-merge.v1"
SCORE_SCHEMA = "celltrack-alpha.public-temporalunet-frozen-score.v1"
RUN_SCHEMA = "celltrack-alpha.public-temporalunet-kaggle-run.v1"
HIDDEN_PACKAGE_SCHEMA = "celltrack-alpha.public-temporalunet-hidden-package.v1"
HIDDEN_GATE_SCHEMA = "celltrack-alpha.public-temporalunet-hidden-gate.v1"
UNSHARDED_V2_PRELOGGING_KERNEL_ID = "onebitbrain/g02-public-temporalunet-v32-unsharded-v2"
UNSHARDED_V2_PRELOGGING_PACKAGE_SHA256 = "4b54dc76365e7cddf3b860bc48f58b03909be65ebd332853a98bf73aabecee3a"
VARIANTS = ("single_seed", "detection_blend", "public_v32")
CAP2_VARIANT = "public_v32_cap2"
GAP2_VARIANT = "public_v32_gap2"
BUILD_VARIANTS = (*VARIANTS, CAP2_VARIANT, GAP2_VARIANT)
CAP2_EXPECTED_PARENTS = (
    {
        "dataset": "44b6_a2bb48bb",
        "parent_id": 27740,
        "outdegree": 3,
        "daughters": [28445, 28466, 28467],
    },
    {
        "dataset": "6bba_ebdf3b34",
        "parent_id": 24468,
        "outdegree": 3,
        "daughters": [25150, 25152, 25165],
    },
)
CSV_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


class BaselineError(RuntimeError):
    """The immutable public-baseline contract was violated."""


def competition_faithful_graph_variant(variant: str) -> bool:
    """Return whether scoring must preserve V32's uncapped graph contract."""

    return variant in ("public_v32", GAP2_VARIANT)


def registered_v32_runtime_source(source: str) -> str:
    """Build the distinct post-authentication V32+registration replay source.

    This helper is deliberately not called by the immutable six-shard V32 base.
    Once that base authenticates, the registered arm must be packaged under a
    fresh identity and measured directly; the APNCC +0.005242 result is not an
    additive estimate for this topology.
    """

    source_root = ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from celltrack_alpha.temporalunet_registration import patch_predict_source

    try:
        return patch_predict_source(source)
    except ValueError as exc:
        raise BaselineError(f"cannot build registered V32 runtime source: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError(f"{label} must be a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BaselineError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_population_manifest(path: Path) -> dict[str, Any]:
    manifest = load_object(path, "population manifest")
    if manifest.get("schema") != POPULATION_SCHEMA:
        raise BaselineError("population manifest schema drift")
    populations = manifest.get("populations")
    if not isinstance(populations, dict) or set(populations) != {"held_fit105", "cv57"}:
        raise BaselineError("population manifest must contain held_fit105 and cv57")
    observed: dict[str, tuple[str, ...]] = {}
    for name in ("held_fit105", "cv57"):
        raw = populations[name]
        if not isinstance(raw, list):
            raise BaselineError(f"population {name} must be a list")
        identities = tuple(str(value) for value in raw)
        if not identities or len(identities) != len(set(identities)):
            raise BaselineError(f"population {name} is empty or contains duplicates")
        if any(not identity.startswith("train:") for identity in identities):
            raise BaselineError(f"population {name} contains a non-train identity")
        observed[name] = identities
    if len(observed["held_fit105"]) != 105 or len(observed["cv57"]) != 57:
        raise BaselineError("population cardinality is not 105/57")
    if set(observed["held_fit105"]) & set(observed["cv57"]):
        raise BaselineError("held_fit105 overlaps cv57")
    union = tuple(str(value) for value in manifest.get("union", []))
    expected_union = observed["held_fit105"] + observed["cv57"]
    if union != expected_union:
        raise BaselineError("population union ordering drift")
    recorded = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if recorded != canonical_sha256(payload):
        raise BaselineError("population manifest self-hash mismatch")
    return manifest


def freeze_populations(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BaselineError(f"refusing to overwrite {args.output}")
    selection = load_object(args.fit_selection, "fit105 selection")
    scope = selection.get("scope", {})
    fit105 = tuple(str(value) for value in scope.get("train_video_ids", []))
    if (
        selection.get("schema") != "celltrack-alpha.selected-graph-appearance-audit.v1"
        or scope.get("fit_only") is not True
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or len(fit105) != 105
        or len(set(fit105)) != 105
    ):
        raise BaselineError("fit105 source is not the frozen admissible 105-video scope")

    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    import score_bh_cv57  # noqa: PLC0415
    import scoreall  # noqa: PLC0415

    split = load_object(args.split, "frozen grouped split")
    cv57 = tuple(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.archive),
        )
    )
    if set(fit105) & set(cv57):
        raise BaselineError("fit105 unexpectedly overlaps exact CV57")
    payload: dict[str, Any] = {
        "schema": POPULATION_SCHEMA,
        "purpose": "immutable-weight public TemporalUNet3D held-fit105 and exact CV57 reproduction",
        "populations": {"held_fit105": list(fit105), "cv57": list(cv57)},
        "union": list(fit105 + cv57),
        "counts": {"held_fit105": 105, "cv57": 57, "union": 162},
        "substrate_only": True,
        "weight_refinement_minimum_local_score": 0.916,
        "sources": {
            "fit_selection": {"path": str(args.fit_selection), "sha256": sha256_file(args.fit_selection)},
            "split": {"path": str(args.split), "sha256": sha256_file(args.split)},
            "archive": {"path": str(args.archive), "sha256": sha256_file(args.archive)},
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    atomic_json(args.output, payload)
    return payload


def source_cells(notebook: Mapping[str, Any]) -> list[str]:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise BaselineError("public notebook has no cells")
    result: list[str] = []
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if not isinstance(source, list):
                raise BaselineError("public notebook code cell source is malformed")
            result.append("".join(str(value) for value in source))
    if len(result) != 6:
        raise BaselineError(f"expected six Version-32 code cells, found {len(result)}")
    return result


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise BaselineError(f"{label}: expected one replacement target, found {count}")
    return source.replace(old, new, 1)


def artifact_binding() -> dict[str, Any]:
    value = {
        "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
        "public_script_version_id": 337501010,
        "primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
        "primary_weight_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_manifest_sha256": SECONDARY_MANIFEST_SHA256,
        "secondary_weight_sha256": SECONDARY_WEIGHT_SHA256,
    }
    return {**value, "binding_sha256": canonical_sha256(value)}


def runtime_logging_source(*, hash_input_contents: bool = True) -> str:
    """Return the self-contained JSONL telemetry required in every new kernel."""

    source = r'''\
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

RUN_LOG_PATH = Path("/kaggle/working/RUN_LOG.jsonl")
RUN_STATE = {"stages": {}, "config": {}}
HASH_INPUT_CONTENTS = __HASH_INPUT_CONTENTS__


def _peak_rss_bytes():
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except Exception:
        return None


def _run_log(event, **fields):
    payload = {"schema": "celltrack-alpha.run-log.v1", "event": event,
               "utc": time.time(), "peak_rss_bytes": _peak_rss_bytes(), **fields}
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    print(line, flush=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return payload


def _run_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_config(config):
    RUN_STATE["config"] = config
    _run_log("config", config=config)


def inventory_input(label, root):
    root = Path(root)
    if not root.exists():
        _run_log("input_inventory", label=label, root=str(root), status="missing", files=[])
        return []
    files = []
    for directory, _dirs, names in os.walk(root):
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file():
                entry = {"path": str(path), "bytes": path.stat().st_size}
                if HASH_INPUT_CONTENTS:
                    entry["sha256"] = _run_sha256(path)
                files.append(entry)
    _run_log("input_inventory", label=label, root=str(root), status="present",
             content_hashing=HASH_INPUT_CONTENTS, files=files)
    return files


def log_lookup(sought, found):
    _run_log("lookup", sought=str(sought), found=str(found) if found else None,
             status="hit" if found else "miss")


def stage_start(name, records_in):
    RUN_STATE["stages"][name] = time.time()
    _run_log("stage_start", stage=name, records_in=records_in)


def stage_end(name, records_in, records_out):
    started = RUN_STATE["stages"].pop(name, None)
    _run_log("stage_end", stage=name, records_in=records_in, records_out=records_out,
             duration_seconds=None if started is None else time.time() - started)


def preprocess_counts(name, examined, modified, **extra):
    _run_log("preprocess", stage=name, examined=examined, modified=modified, **extra)


def _run_failure(exc_type, exc_value, tb):
    _run_log("failure", exception_type=exc_type.__name__, exception=str(exc_value),
             traceback="".join(traceback.format_exception(exc_type, exc_value, tb)), state=RUN_STATE)
    sys.__excepthook__(exc_type, exc_value, tb)


sys.excepthook = _run_failure
'''
    return source.replace("__HASH_INPUT_CONTENTS__", repr(hash_input_contents))


def step0_source(
    *,
    variant: str,
    identities: Sequence[str],
    population_sha256: str,
    global_population_sha256: str,
) -> str:
    datasets = [identity.split(":", 1)[1] for identity in identities]
    binding = artifact_binding()
    return f'''\
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

RUN_SCHEMA = {RUN_SCHEMA!r}
VARIANT = {variant!r}
SELECTED_DATASETS = {datasets!r}
REQUEST_COUNT = {len(datasets)}
REQUEST_FILE_COUNT = {len(datasets)}
POPULATION_FILE_COUNT = {len(datasets)}
WHEEL_COUNT = 60
POPULATION_MANIFEST_SHA256 = {population_sha256!r}
GLOBAL_POPULATION_MANIFEST_SHA256 = {global_population_sha256!r}
INPUT_MANIFEST_SHA256 = {binding['binding_sha256']!r}
PUBLIC_NOTEBOOK_SHA256 = {PUBLIC_NOTEBOOK_SHA256!r}
PUBLIC_SCRIPT_VERSION_ID = 337501010
PRIMARY_ROOT = Path({PRIMARY_ROOT!r})
SECONDARY_ROOT = Path({SECONDARY_ROOT!r})
COMPETITION_TRAIN = Path({COMPETITION_TRAIN!r})
PRIMARY_MANIFEST_SHA256 = {PRIMARY_MANIFEST_SHA256!r}
PRIMARY_WEIGHT_SHA256 = {PRIMARY_WEIGHT_SHA256!r}
SECONDARY_MANIFEST_SHA256 = {SECONDARY_MANIFEST_SHA256!r}
SECONDARY_WEIGHT_SHA256 = {SECONDARY_WEIGHT_SHA256!r}
WEIGHT_REFINEMENT_ALLOWED = False
WEIGHT_REFINEMENT_MINIMUM_LOCAL_SCORE = 0.916


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    temporary.replace(path)


def run_step0() -> dict:
    started = time.time()
    run_config({{
        "variant": VARIANT, "thresholds": {{}}, "radii": {{}}, "windows": {{}},
        "scales": {{}}, "conventions": {{"immutable_weights": True, "substrate_only": True}},
        "selected_datasets": SELECTED_DATASETS,
    }})
    inventory_input("primary_support_pack", PRIMARY_ROOT)
    inventory_input("secondary_support_pack", SECONDARY_ROOT)
    inventory_input("competition_train", COMPETITION_TRAIN)
    stage_start("step0", {{"requested_datasets": REQUEST_COUNT}})
    exact_files = {{
        PRIMARY_ROOT / "ARTIFACT_MANIFEST.json": PRIMARY_MANIFEST_SHA256,
        PRIMARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": PRIMARY_WEIGHT_SHA256,
        SECONDARY_ROOT / "ARTIFACT_MANIFEST.json": SECONDARY_MANIFEST_SHA256,
        SECONDARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": SECONDARY_WEIGHT_SHA256,
    }}
    observed = {{}}
    for path, expected in exact_files.items():
        if not path.is_file():
            log_lookup(path, None)
            raise FileNotFoundError(f"STEP0 missing exact public artifact: {{path}}")
        log_lookup(path, path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"STEP0 hash mismatch for {{path}}: {{actual}} != {{expected}}")
        observed[str(path)] = actual
    if len(SELECTED_DATASETS) != REQUEST_COUNT or len(set(SELECTED_DATASETS)) != REQUEST_COUNT:
        raise RuntimeError("STEP0 request count or uniqueness drift")
    missing = []
    for dataset in SELECTED_DATASETS:
        source = COMPETITION_TRAIN / f"{{dataset}}.zarr"
        if not source.is_dir() or not (source / "0").exists():
            log_lookup(source, None)
            missing.append(dataset)
        else:
            log_lookup(source, source)
    if missing:
        raise FileNotFoundError(f"STEP0 missing organizer raw datasets: {{missing[:10]}}")
    value = {{
        "schema": "celltrack-alpha.public-temporalunet-step0.v1",
        "status": "PASS",
        "variant": VARIANT,
        "request_count": REQUEST_COUNT,
        "request_file_count": REQUEST_FILE_COUNT,
        "population_file_count": POPULATION_FILE_COUNT,
        "wheel_count": WHEEL_COUNT,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "global_population_manifest_sha256": GLOBAL_POPULATION_MANIFEST_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "checkpoint_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_WEIGHT_SHA256,
        "dataset_source": {SECONDARY_SOURCE!r},
        "dataset_version": 1,
        "dataset_status": "ready",
        "enable_gpu": True,
        "heavy_work_started": False,
        "competition_path_visits": REQUEST_COUNT,
        "artifacts": observed,
        "weight_refinement_allowed": WEIGHT_REFINEMENT_ALLOWED,
        "elapsed_seconds": time.time() - started,
    }}
    _atomic_json(Path("/kaggle/working/PUBLIC_TEMPORALUNET_STEP0.json"), value)
    stage_end("step0", {{"requested_datasets": REQUEST_COUNT}}, {{"verified_artifacts": len(observed)}})
    return value


STEP0 = run_step0()

SELECTED_TRAIN = Path("/kaggle/working/public_temporalunet_selected_train")
if SELECTED_TRAIN.exists():
    shutil.rmtree(SELECTED_TRAIN)
SELECTED_TRAIN.mkdir(parents=True, exist_ok=False)
for dataset in SELECTED_DATASETS:
    (SELECTED_TRAIN / f"{{dataset}}.zarr").symlink_to(
        COMPETITION_TRAIN / f"{{dataset}}.zarr", target_is_directory=True
    )
os.environ["BIOHUB_TEST_DIR"] = str(SELECTED_TRAIN)
preprocess_counts("selected_train_symlink_view", len(SELECTED_DATASETS), 0,
                  created_symlinks=len(SELECTED_DATASETS), raw_inputs_modified=0)
'''


def hidden_step0_source(
    *, variant: str, expected_test_count: int, local_gate_sha256: str
) -> str:
    """Return the inline preflight for a fresh hidden-test inference run."""

    if variant != "public_v32":
        raise BaselineError("hidden submission packaging is only defined for public_v32")
    if expected_test_count <= 0:
        raise BaselineError("hidden test count must be positive")
    binding = artifact_binding()
    return f'''\
import hashlib
import json
import os
from pathlib import Path
import platform
import time

RUN_SCHEMA = {RUN_SCHEMA!r}
VARIANT = {variant!r}
# Public dry-run inventory size.  Retained for telemetry and for the prelaunch
# contract's exact-count declaration ONLY.  It is NOT a gate: Kaggle states the
# hidden rerun dataset "can be larger/smaller/different than the public dataset",
# and the hidden inventory is discovered dynamically below.
PUBLIC_TEST_COUNT = {expected_test_count}
REQUEST_COUNT = {expected_test_count}
REQUEST_FILE_COUNT = {expected_test_count}
POPULATION_FILE_COUNT = {expected_test_count}
WHEEL_COUNT = 60
POPULATION_MANIFEST_SHA256 = {local_gate_sha256!r}
GLOBAL_POPULATION_MANIFEST_SHA256 = {local_gate_sha256!r}
INPUT_MANIFEST_SHA256 = {binding['binding_sha256']!r}
PUBLIC_NOTEBOOK_SHA256 = {PUBLIC_NOTEBOOK_SHA256!r}
PUBLIC_SCRIPT_VERSION_ID = 337501010
PRIMARY_ROOT = Path({PRIMARY_ROOT!r})
SECONDARY_ROOT = Path({SECONDARY_ROOT!r})
COMPETITION_TEST_CANDIDATES = [Path(_p) for _p in {COMPETITION_TEST_CANDIDATES!r}]
# Preference-ordered default; run_step0() rebinds this to the root that actually
# exists on the machine before any other use.
COMPETITION_TEST = COMPETITION_TEST_CANDIDATES[0]
PRIMARY_MANIFEST_SHA256 = {PRIMARY_MANIFEST_SHA256!r}
PRIMARY_WEIGHT_SHA256 = {PRIMARY_WEIGHT_SHA256!r}
SECONDARY_MANIFEST_SHA256 = {SECONDARY_MANIFEST_SHA256!r}
SECONDARY_WEIGHT_SHA256 = {SECONDARY_WEIGHT_SHA256!r}
WEIGHT_REFINEMENT_ALLOWED = False
WEIGHT_REFINEMENT_MINIMUM_LOCAL_SCORE = 0.916
HIDDEN_TEST_COMPATIBLE = True
DYNAMIC_TEST_DISCOVERY = True
PUBLIC_OUTPUT_COPIED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    temporary.replace(path)


def run_step0() -> dict:
    started = time.time()
    run_config({{
        "variant": VARIANT,
        "thresholds": {{}},
        "radii": {{}},
        "windows": {{}},
        "scales": {{}},
        "conventions": {{
            "immutable_weights": True,
            "substrate_only": True,
            "hidden_test_compatible": True,
            "dynamic_test_discovery": True,
            "public_output_copied": False,
        }},
    }})
    inventory_input("primary_support_pack", PRIMARY_ROOT)
    inventory_input("secondary_support_pack", SECONDARY_ROOT)
    print("=" * 78, flush=True)
    print("STEP0 HIDDEN-RERUN CONTRACT | mount resolution", flush=True)
    print("=" * 78, flush=True)
    print(f"[step0] public dry-run inventory size (telemetry only): {{PUBLIC_TEST_COUNT}}", flush=True)
    print(f"[step0] candidate roots ({{len(COMPETITION_TEST_CANDIDATES)}}):", flush=True)
    resolved_test = None
    for candidate in COMPETITION_TEST_CANDIDATES:
        exists = candidate.is_dir()
        print(f"[step0]   {{'HIT ' if exists else 'miss'}} {{candidate}}", flush=True)
        if exists and resolved_test is None:
            resolved_test = candidate
        elif not exists:
            log_lookup(candidate, None)
    if resolved_test is None:
        print("[step0] FATAL: no candidate root exists on this machine.", flush=True)
        try:
            for entry in sorted(Path("/kaggle/input").iterdir()):
                print(f"[step0]   /kaggle/input contains: {{entry}}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[step0]   /kaggle/input unreadable: {{exc!r}}", flush=True)
        raise FileNotFoundError(
            "STEP0 hidden test root is unavailable under any known Kaggle mount: "
            + repr([str(candidate) for candidate in COMPETITION_TEST_CANDIDATES])
        )
    print(f"[step0] RESOLVED competition test root -> {{resolved_test}}", flush=True)
    globals()["COMPETITION_TEST"] = resolved_test
    log_lookup(resolved_test, resolved_test)
    inventory_input("competition_test", resolved_test)
    stage_start("step0", {{"public_test_datasets": PUBLIC_TEST_COUNT}})
    exact_files = {{
        PRIMARY_ROOT / "ARTIFACT_MANIFEST.json": PRIMARY_MANIFEST_SHA256,
        PRIMARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": PRIMARY_WEIGHT_SHA256,
        SECONDARY_ROOT / "ARTIFACT_MANIFEST.json": SECONDARY_MANIFEST_SHA256,
        SECONDARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": SECONDARY_WEIGHT_SHA256,
    }}
    observed = {{}}
    for path, expected in exact_files.items():
        if not path.is_file():
            log_lookup(path, None)
            raise FileNotFoundError(f"STEP0 missing exact public artifact: {{path}}")
        log_lookup(path, path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"STEP0 hash mismatch for {{path}}: {{actual}} != {{expected}}")
        observed[str(path)] = actual
    selected = sorted(
        path.name[:-5]
        for path in resolved_test.iterdir()
        if path.is_dir() and path.name.endswith(".zarr")
    )
    # The hidden inventory is accepted at ANY positive cardinality.  Only an
    # empty or self-inconsistent inventory is a genuine failure.
    print(f"[step0] discovered {{len(selected)}} *.zarr dataset(s) under {{resolved_test}}", flush=True)
    print(f"[step0] public dry run saw {{PUBLIC_TEST_COUNT}}; "
          f"a different hidden count is EXPECTED and accepted", flush=True)
    for _index, _dataset in enumerate(selected):
        print(f"[step0]   [{{_index:03d}}] {{_dataset}}", flush=True)
    if not selected:
        try:
            _listing = sorted(str(entry.name) for entry in resolved_test.iterdir())
        except Exception as exc:  # noqa: BLE001
            _listing = [f"<unreadable: {{exc!r}}>"]
        print(f"[step0] FATAL: no *.zarr datasets. Root contains: {{_listing[:50]}}", flush=True)
        raise RuntimeError(
            f"STEP0 hidden inventory is empty under {{resolved_test}}"
        )
    if len(set(selected)) != len(selected):
        _dupes = sorted({{name for name in selected if selected.count(name) > 1}})
        print(f"[step0] FATAL: duplicate identities: {{_dupes}}", flush=True)
        raise RuntimeError(
            "STEP0 hidden inventory contains duplicate identities: "
            f"{{len(selected)}} entries, {{len(set(selected))}} unique"
        )
    for dataset in selected:
        source = resolved_test / f"{{dataset}}.zarr"
        if not (source / "0").exists():
            log_lookup(source, None)
            raise FileNotFoundError(f"STEP0 malformed hidden dataset: {{source}}")
        log_lookup(source, source)
    globals()["SELECTED_DATASETS"] = selected
    value = {{
        "schema": "celltrack-alpha.public-temporalunet-hidden-step0.v1",
        "status": "PASS",
        "variant": VARIANT,
        "request_count": len(selected),
        "request_file_count": len(selected),
        "population_file_count": len(selected),
        "wheel_count": WHEEL_COUNT,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "global_population_manifest_sha256": GLOBAL_POPULATION_MANIFEST_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "checkpoint_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_WEIGHT_SHA256,
        "dataset_source": {SECONDARY_SOURCE!r},
        "dataset_version": 1,
        "dataset_status": "ready",
        "enable_gpu": True,
        "heavy_work_started": False,
        "competition_path_visits": len(selected),
        "competition_test_root": str(resolved_test),
        "public_test_count": PUBLIC_TEST_COUNT,
        "artifacts": observed,
        "selected_datasets": selected,
        "hidden_test_compatible": True,
        "dynamic_test_discovery": True,
        "public_identity_pinned": False,
        "public_output_copied": False,
        "weight_refinement_allowed": WEIGHT_REFINEMENT_ALLOWED,
        "elapsed_seconds": time.time() - started,
    }}
    _atomic_json(Path("/kaggle/working/PUBLIC_TEMPORALUNET_STEP0.json"), value)
    stage_end("step0", {{"public_test_datasets": PUBLIC_TEST_COUNT}},
              {{"verified_artifacts": len(observed), "discovered_datasets": len(selected)}})
    return value


STEP0 = run_step0()
# Propagate the root STEP0 actually RESOLVED, never a hard-coded one.  The
# notebook body prefers BIOHUB_TEST_DIR over its own candidate resolution, so
# writing an unverified path here would silently disable that fallback.
os.environ["BIOHUB_TEST_DIR"] = STEP0["competition_test_root"]
print("=" * 78, flush=True)
print(f"[step0] STATUS={{STEP0['status']}} "
      f"datasets={{len(STEP0['selected_datasets'])}} "
      f"root={{STEP0['competition_test_root']}}", flush=True)
print(f"[step0] BIOHUB_TEST_DIR -> {{os.environ['BIOHUB_TEST_DIR']}}", flush=True)
print(f"[step0] elapsed={{STEP0['elapsed_seconds']:.3f}}s -- proceeding to inference", flush=True)
print("=" * 78, flush=True)
preprocess_counts(
    "dynamic_hidden_test_inventory",
    len(SELECTED_DATASETS),
    0,
    public_identity_pinned=False,
    public_output_copied=False,
    raw_inputs_modified=0,
)
'''


def variant_override(variant: str) -> str:
    if variant == "single_seed":
        values = {
            "BIOHUB_SECONDARY_WEIGHTS": "",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.0",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.0",
            "BIOHUB_SECONDARY_LINK_MODE": "raw",
            "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
            "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
            "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        }
    elif variant == "detection_blend":
        values = {
            "BIOHUB_SECONDARY_WEIGHTS": "__KEEP_MATERIALIZED_SECONDARY__",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.0",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.5",
            "BIOHUB_SECONDARY_LINK_MODE": "raw",
            "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
            "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
            "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        }
    elif variant in ("public_v32", CAP2_VARIANT, GAP2_VARIANT):
        values = {
            "BIOHUB_SECONDARY_WEIGHTS": "__KEEP_MATERIALIZED_SECONDARY__",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.15",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.5",
            "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
            "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
            "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
            "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        }
    else:
        raise BaselineError(f"unknown variant {variant!r}")
    lines = ["# Frozen immutable-weight variant override."]
    for key, value in values.items():
        if value == "__KEEP_MATERIALIZED_SECONDARY__":
            lines.append(f'os.environ[{key!r}] = str(SECONDARY_WEIGHTS_PATH)')
        else:
            lines.append(f'os.environ[{key!r}] = {value!r}')
    lines.append(f"PUBLIC_TEMPORALUNET_VARIANT = {variant!r}")
    lines.append(f"PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO = {variant == CAP2_VARIANT!r}")
    if variant == CAP2_VARIANT:
        lines.append("os.environ['BIOHUB_EXPERIMENTAL_CAP_OUTDEGREE_TWO'] = '1'")
    return "\n".join(lines) + "\n"


def patch_gap2_experiment(source: str) -> str:
    """Flip the pristine notebook's already-built, currently-disabled
    2-frame gap-recovery mechanism (``recover_strict_gap2``, gated by
    ``BIOHUB_OUTPUT_GAP2_RECOVERY``) to enabled.

    Unlike ``patch_cap2_experiment`` this injects no new code -- the
    mechanism is already fully implemented in the pristine cell; only the
    flag it reads is flipped.
    """
    return replace_once(
        source,
        'os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"',
        'os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"',
        "enable existing gap2 recovery for the gap2 experimental variant",
    )


def patch_cap2_experiment(source: str) -> str:
    """Append an explicit cap after pristine V32 safe-division generation.

    The pristine ``add_safe_divisions_postlink`` implementation is left byte
    unchanged.  Only the separately labelled cap2 package receives this
    postlink stage, and every affected parent plus its full ranking is emitted
    through the persisted structured run log.
    """

    anchor = "\n    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:\n"
    inserted = r'''
    if PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO and edges:
        cap2_by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            cap2_by_source.setdefault(int(edge["source_id"]), []).append(edge)

        cap2_kept_ids: set[int] = set()
        cap2_parent_count = 0
        cap2_dropped_count = 0
        for source_id, source_edges in cap2_by_source.items():
            if len(source_edges) <= 2:
                cap2_kept_ids.update(id(edge) for edge in source_edges)
                continue
            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            retained = ranked[:2]
            dropped = ranked[2:]
            retained_ids = {id(edge) for edge in retained}
            cap2_kept_ids.update(retained_ids)
            cap2_parent_count += 1
            cap2_dropped_count += len(dropped)
            _run_log(
                "experimental_cap2_parent",
                dataset=dataset,
                parent_id=source_id,
                outdegree_before=len(ranked),
                daughters_before=[int(edge["target_id"]) for edge in source_edges],
                ranking=[
                    {
                        "target_id": int(edge["target_id"]),
                        "edge_prob": None if edge.get("edge_prob") is None else float(edge["edge_prob"]),
                        "distance_um": float(edge["distance_um"]),
                        "retained": id(edge) in retained_ids,
                    }
                    for edge in ranked
                ],
                retained_daughters=[int(edge["target_id"]) for edge in retained],
                dropped_daughters=[int(edge["target_id"]) for edge in dropped],
            )
        edges = [edge for edge in edges if id(edge) in cap2_kept_ids]
        stats["experimental_cap2_parent_count"] = cap2_parent_count
        stats["experimental_cap2_dropped_edges"] = cap2_dropped_count
        _run_log(
            "experimental_cap2_dataset_summary",
            dataset=dataset,
            affected_parents=cap2_parent_count,
            dropped_edges=cap2_dropped_count,
        )
'''
    return replace_once(
        source,
        anchor,
        inserted + anchor,
        "insert explicit postlink cap-to-two experiment",
    )


def patch_prediction_cell(source: str, *, patch_count: int = 3) -> str:
    """Apply an ordered prefix of the three prediction-cell replacements."""

    if not 0 <= patch_count <= 3:
        raise BaselineError("prediction patch count must be in 0..3")
    if patch_count >= 1:
        source = replace_once(
            source,
            'if not 0.0 < secondary_edge_weight < 1.0:',
            'if not 0.0 <= secondary_edge_weight < 1.0:',
            "allow exact detection-only association weight",
        )
    if patch_count >= 2:
        source = replace_once(
            source,
            'raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")',
            'raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be in [0, 1)")',
            "detection-only validation message",
        )
    if patch_count >= 3:
        source = replace_once(
            source,
            "\\n\\n            if secondary_model is not None:\\n"
            "                if secondary_unet_out is None:\\n",
            "\\n\\n            if secondary_model is not None and secondary_edge_weight > 0.0:\\n"
            "                if secondary_unet_out is None:\\n",
            "skip secondary association inference in the detection-only mode",
        )
    return source


def receipt_source(
    *, variant: str, identities: Sequence[str], population_sha256: str, global_sha256: str
) -> str:
    datasets = [identity.split(":", 1)[1] for identity in identities]
    return f'''\
# Atomic terminal receipt for local merge/scoring.
stage_end("prediction", {{"datasets": len(SELECTED_DATASETS)}},
          {{"nodes": globals().get("total_nodes"), "edges": globals().get("total_edges"),
            "submission_rows": globals().get("row_id")}})
_run_log("terminal", status="PASS", state=RUN_STATE)
_run_log_sha256 = _sha256(RUN_LOG_PATH)
_submission_sha256 = _sha256(SUBMISSION_PATH)
_run_stats_sha256 = _sha256(RUN_STATS_PATH)
_runtime_predict_sha256 = _sha256(REPO_DIR / "scripts/predict_unet_transformer.py")
_receipt = {{
    "schema": RUN_SCHEMA,
    "status": "PASS",
    "variant": {variant!r},
    "selected_datasets": {datasets!r},
    "request_count": {len(datasets)},
    "population_manifest_sha256": {population_sha256!r},
    "global_population_manifest_sha256": {global_sha256!r},
    "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
    "public_script_version_id": PUBLIC_SCRIPT_VERSION_ID,
    "primary_weight_sha256": PRIMARY_WEIGHT_SHA256,
    "secondary_weight_sha256": SECONDARY_WEIGHT_SHA256,
    "submission": {{"path": str(SUBMISSION_PATH), "sha256": _submission_sha256, "rows": row_id}},
    "run_stats": {{"path": str(RUN_STATS_PATH), "sha256": _run_stats_sha256}},
    "run_log": {{"path": str(RUN_LOG_PATH), "sha256": _run_log_sha256}},
    "runtime_predict_script_sha256": _runtime_predict_sha256,
    "nodes": total_nodes,
    "edges": total_edges,
    "prediction_seconds": predict_seconds,
    "cuda_device_count": _torch.cuda.device_count(),
    "cuda_device_name": _torch.cuda.get_device_name(0),
    "weight_refinement": False,
    "substrate_only": True,
    "weight_refinement_minimum_local_score": 0.916,
}}
_atomic_json(WORKING_DIR / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", _receipt)
print(json.dumps(_receipt, indent=2, sort_keys=True))
# Keep remote outputs bounded to the immutable receipt, submission, and stats.
# All hashes above are committed before scratch materialization is removed.
for _scratch_path in (REPO_DIR, SECONDARY_WEIGHTS_ROOT, SELECTED_TRAIN):
    if _scratch_path.exists():
        shutil.rmtree(_scratch_path)
'''


def validate_hidden_submission_gate(
    *, score_path: Path, taxonomy_path: Path, terminal_auth_path: Path
) -> dict[str, Any]:
    """Bind hidden packaging to the authenticated local replication result."""

    score = load_object(score_path, "V32 CV57 score")
    taxonomy = load_object(taxonomy_path, "V32 residual taxonomy")
    terminal = load_object(terminal_auth_path, "V32 terminal authentication")
    score_sha = sha256_file(score_path)
    taxonomy_sha = sha256_file(taxonomy_path)
    terminal_sha = sha256_file(terminal_auth_path)
    local_score = float(score.get("score", float("nan")))
    score_submission_sha = str(score.get("submission", {}).get("sha256", ""))
    terminal_submission_sha = str(
        terminal.get("validation", {}).get("submission_sha256", "")
    )
    taxonomy_validation = taxonomy.get("validation", {})
    taxonomy_summary_score = float(
        taxonomy.get("global", {}).get("baseline_summary", {}).get("score", float("nan"))
    )
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("status") != "PASS"
        or score.get("variant") != "public_v32"
        or score.get("population") != "cv57"
        or not math.isfinite(local_score)
        or local_score < 0.91
        or score.get("weight_refinement") is not False
    ):
        raise BaselineError("V32 score does not clear the immutable 0.91 submission gate")
    if (
        terminal.get("status") != "PASS"
        or terminal.get("kernel", {}).get("version") != 1
        or terminal.get("kernel", {}).get("terminal_status") != "COMPLETE"
        or terminal.get("validation", {}).get("graph_mutated") is not False
        or terminal.get("validation", {}).get("graph_policy")
        != "competition_faithful_no_outdegree_cap"
        or score_submission_sha != terminal_submission_sha
    ):
        raise BaselineError("V32 score is not bound to the authenticated Version 1 output")
    if (
        taxonomy.get("status") != "PASS"
        or taxonomy_validation.get("status") != "PASS"
        or taxonomy_validation.get("submission_graph_policy")
        != "competition_faithful_no_outdegree_cap"
        or taxonomy_validation.get("authoritative_replay", {}).get("mismatch_count") != 0
        or taxonomy_validation.get("authoritative_replay", {}).get("reference_sha256")
        != score_sha
        or not math.isclose(taxonomy_summary_score, local_score, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise BaselineError("V32 taxonomy is not an exact replay of the eligible score")
    payload: dict[str, Any] = {
        "schema": HIDDEN_GATE_SCHEMA,
        "status": "PASS",
        "eligible": True,
        "variant": "public_v32",
        "local_success_threshold": 0.91,
        "local_score": local_score,
        "reported_public_reference_range": [0.908, 0.911],
        "apncc_reference": 0.8815220699597315,
        "weight_refinement": False,
        "weight_refinement_gate": 0.916,
        "public_output_sha256": score_submission_sha,
        "public_output_copy_allowed": False,
        "hidden_test_dynamic_discovery_required": True,
        "graph_policy": "competition_faithful_no_outdegree_cap",
        "degree_three_parent_count": int(
            score.get("outdegree_diagnostics", {}).get("degree_three_parent_count", -1)
        ),
        "score_receipt": {"path": str(score_path), "sha256": score_sha},
        "taxonomy_receipt": {"path": str(taxonomy_path), "sha256": taxonomy_sha},
        "terminal_authentication": {
            "path": str(terminal_auth_path),
            "sha256": terminal_sha,
        },
        "g1_assumed_additive": False,
    }
    payload["gate_sha256"] = canonical_sha256(payload)
    return payload


def hidden_receipt_source(*, local_gate: Mapping[str, Any]) -> str:
    return f'''\
# Atomic terminal receipt for the separately generated hidden-test submission.
if sorted(SELECTED_DATASETS) != sorted(test_stems):
    raise RuntimeError("dynamic hidden inventory drifted between step 0 and prediction")
stage_end("prediction", {{"datasets": len(SELECTED_DATASETS)}},
          {{"nodes": globals().get("total_nodes"), "edges": globals().get("total_edges"),
            "submission_rows": globals().get("row_id")}})
_run_log("terminal", status="PASS", state=RUN_STATE)
_run_log_sha256 = _sha256(RUN_LOG_PATH)
_submission_sha256 = _sha256(SUBMISSION_PATH)
_run_stats_sha256 = _sha256(RUN_STATS_PATH)
_runtime_predict_sha256 = _sha256(REPO_DIR / "scripts/predict_unet_transformer.py")
_receipt = {{
    "schema": "celltrack-alpha.public-temporalunet-hidden-run.v1",
    "status": "PASS",
    "variant": "public_v32",
    "selected_datasets": list(test_stems),
    "request_count": len(test_stems),
    "population_manifest_sha256": {str(local_gate['gate_sha256'])!r},
    "global_population_manifest_sha256": {str(local_gate['gate_sha256'])!r},
    "local_submission_gate_sha256": {str(local_gate['gate_sha256'])!r},
    "local_cv57_score": {float(local_gate['local_score'])!r},
    "public_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
    "public_script_version_id": PUBLIC_SCRIPT_VERSION_ID,
    "primary_weight_sha256": PRIMARY_WEIGHT_SHA256,
    "secondary_weight_sha256": SECONDARY_WEIGHT_SHA256,
    "submission": {{"path": str(SUBMISSION_PATH), "sha256": _submission_sha256, "rows": row_id}},
    "run_stats": {{"path": str(RUN_STATS_PATH), "sha256": _run_stats_sha256}},
    "run_log": {{"path": str(RUN_LOG_PATH), "sha256": _run_log_sha256}},
    "runtime_predict_script_sha256": _runtime_predict_sha256,
    "nodes": total_nodes,
    "edges": total_edges,
    "prediction_seconds": predict_seconds,
    "cuda_device_count": _torch.cuda.device_count(),
    "cuda_device_name": _torch.cuda.get_device_name(0),
    "hidden_test_compatible": True,
    "dynamic_test_discovery": True,
    "public_identity_pinned": False,
    "fresh_submission_generation": True,
    "public_output_copied": False,
    "graph_policy": "competition_faithful_no_outdegree_cap",
    "weight_refinement": False,
    "substrate_only": True,
    "g1_applied": False,
    "weight_refinement_minimum_local_score": 0.916,
}}
_atomic_json(WORKING_DIR / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", _receipt)
print(json.dumps(_receipt, indent=2, sort_keys=True))
for _scratch_path in (REPO_DIR, SECONDARY_WEIGHTS_ROOT):
    if _scratch_path.exists():
        shutil.rmtree(_scratch_path)
'''


def build_hidden_script(
    *,
    expected_test_count: int,
    local_gate: Mapping[str, Any],
    replacement_count: int = 4,
    hash_input_contents: bool = True,
) -> str:
    """Build the hidden-run script with an ordered prefix of our four deltas.

    ``replacement_count`` is intentionally an ordered prefix: first the exact-mount
    discovery replacement, then the three prediction-cell replacements.  It is used
    only by the submission-repair bisection; the ordinary package path retains its
    historical four-replacement default.
    """

    if not 0 <= replacement_count <= 4:
        raise BaselineError("hidden replacement count must be in 0..4")
    if sha256_file(PUBLIC_NOTEBOOK) != PUBLIC_NOTEBOOK_SHA256:
        raise BaselineError("immutable public Version-32 notebook hash drift")
    notebook = load_object(PUBLIC_NOTEBOOK, "public Version-32 notebook")
    cells = source_cells(notebook)
    cells = [cell.replace("from __future__ import annotations\n\n", "") for cell in cells]
    if replacement_count >= 1:
        cells[3] = replace_once(
            cells[3],
            '    input_root = Path("/kaggle/input")\n    if input_root.exists():\n        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))\n\n',
            "",
            "remove broad secondary-artifact discovery in favor of exact mounts",
        )
    cells[4] = patch_prediction_cell(cells[4], patch_count=max(0, replacement_count - 1))
    cells[3] = cells[3] + "\n" + variant_override("public_v32")
    pieces = [
        runtime_logging_source(hash_input_contents=hash_input_contents),
        hidden_step0_source(
            variant="public_v32",
            expected_test_count=expected_test_count,
            local_gate_sha256=str(local_gate["gate_sha256"]),
        ),
    ]
    for index, cell in enumerate(cells):
        pieces.append(f"\n# ---- PUBLIC V32 CODE CELL {index} ----\n{cell}\n")
        if index == 3:
            pieces.append(
                "run_config({\"variant\": PUBLIC_TEMPORALUNET_VARIANT, "
                "\"thresholds\": {key: value for key, value in os.environ.items() if \"THRESHOLD\" in key}, "
                "\"radii\": {key: value for key, value in os.environ.items() if \"RADIUS\" in key}, "
                "\"windows\": {key: value for key, value in os.environ.items() if \"WINDOW\" in key}, "
                "\"scales\": {key: value for key, value in os.environ.items() if \"SCALE\" in key}, "
                "\"conventions\": {\"immutable_weights\": True, \"hidden_test_compatible\": True, "
                "\"dynamic_test_discovery\": True, \"public_output_copied\": False, "
                "\"effective_biohub_environment\": "
                "{key: value for key, value in os.environ.items() if key.startswith(\"BIOHUB_\")}}})\n"
                "stage_start(\"prediction\", {\"datasets\": len(SELECTED_DATASETS)})\n"
            )
    pieces.append(hidden_receipt_source(local_gate=local_gate))
    script = "".join(pieces)
    compile(script, "public_temporalunet_hidden_kernel.py", "exec")
    forbidden = (
        "optimizer.step(",
        "loss.backward(",
        "train_unet_transformer.py",
        "fine_tun",
        "SELECTED_TRAIN",
        "44b6_",
        "6bba_",
    )
    hits = [token for token in forbidden if token in script]
    if hits:
        raise BaselineError(f"hidden package contains a forbidden training/public-identity token: {hits}")
    return script


def build_hidden_package(args: argparse.Namespace) -> dict[str, Any]:
    local_gate = validate_hidden_submission_gate(
        score_path=args.score_receipt,
        taxonomy_path=args.taxonomy_receipt,
        terminal_auth_path=args.terminal_authentication,
    )
    script = build_hidden_script(
        expected_test_count=args.expected_test_count,
        local_gate=local_gate,
    )
    if args.kernel_dir.exists():
        raise BaselineError(f"refusing to overwrite package directory {args.kernel_dir}")
    args.kernel_dir.mkdir(parents=True)
    code_path = args.kernel_dir / "public_temporalunet_hidden_kernel.py"
    code_path.write_text(script, encoding="utf-8")
    metadata = {
        "id": args.kernel_id,
        "title": args.kernel_id.split("/", 1)[1],
        "code_file": code_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [PRIMARY_SOURCE, SECONDARY_SOURCE],
        "model_sources": [],
    }
    metadata_path = args.kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    binding = artifact_binding()
    contract = {
        "schema": "celltrack-alpha.public-temporalunet-hidden-prelaunch.v1",
        "authorization": {
            "source": "owner Prompt B local-success submission authorization",
            "action": "fresh hidden-test immutable public TemporalUNet3D V32 inference",
            "public_submission": True,
            "weight_refinement": False,
        },
        "kernel_id": args.kernel_id,
        "lane_owner": "PUBLIC_TEMPORALUNET_BASELINE_ADOPTION",
        "dataset_sources": metadata["dataset_sources"],
        "dataset_version": 1,
        "exclusive_dataset_source": SECONDARY_SOURCE,
        "expected_mount_root": SECONDARY_ROOT,
        "kernel_properties": {
            "enable_gpu": True,
            "enable_internet": False,
            "enable_tpu": False,
            "machine_shape": "NvidiaTeslaT4",
        },
        "code_sha256": sha256_file(code_path),
        "input_manifest_sha256": binding["binding_sha256"],
        "checkpoint_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_WEIGHT_SHA256,
        "population_manifest_sha256": local_gate["gate_sha256"],
        "population_file_count": args.expected_test_count,
        "request_count": args.expected_test_count,
        "request_file_count": args.expected_test_count,
        "wheel_count": 60,
        "hidden_test_compatible": True,
        "dynamic_test_discovery": True,
        "public_identity_pinned": False,
        "public_output_copied": False,
        "remote_preflight_mode": "inline_step0_before_heavy_work",
        "remote_preflight_kernel_id": args.kernel_id,
        "step0_competition_path_visits": args.expected_test_count,
        "step0_elapsed_is_hard_failure": False,
        "step0_limit_seconds": 900.0,
        "run_logging": {
            "schema": "celltrack-alpha.run-log.v1",
            "stdout_jsonl": True,
            "persisted_with_artifacts": True,
            "download_on_success": True,
            "inventory_before_resolution": True,
            "failure_traceback_with_state": True,
        },
        "runtime_governance": {
            "per_run_wall_clock_kill": False,
            "projection_gate_h": 8.0,
            "weekly_shared_ledger_cap_h": 30.0,
        },
        "local_submission_gate": local_gate,
        "substrate_only": True,
        "weight_refinement_minimum_local_score": 0.916,
    }
    contract_path = args.kernel_dir / "prelaunch-contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package = {
        "schema": HIDDEN_PACKAGE_SCHEMA,
        "variant": "public_v32",
        "kernel_id": args.kernel_id,
        "expected_test_count": args.expected_test_count,
        "hidden_test_compatible": True,
        "dynamic_test_discovery": True,
        "public_identity_pinned": False,
        "fresh_submission_generation": True,
        "public_output_copied": False,
        "local_submission_gate": local_gate,
        "artifacts": binding,
        "files": {
            code_path.name: sha256_file(code_path),
            metadata_path.name: sha256_file(metadata_path),
            contract_path.name: sha256_file(contract_path),
        },
        "weight_refinement": False,
        "g1_applied": False,
    }
    atomic_json(args.kernel_dir / "PACKAGE_MANIFEST.json", package)
    return package


def build_script(
    *, variant: str, identities: Sequence[str], population_sha256: str,
    global_population_sha256: str,
) -> str:
    if sha256_file(PUBLIC_NOTEBOOK) != PUBLIC_NOTEBOOK_SHA256:
        raise BaselineError("immutable public Version-32 notebook hash drift")
    notebook = load_object(PUBLIC_NOTEBOOK, "public Version-32 notebook")
    cells = source_cells(notebook)
    cells = [cell.replace("from __future__ import annotations\n\n", "") for cell in cells]
    cells[3] = replace_once(
        cells[3],
        '    input_root = Path("/kaggle/input")\n    if input_root.exists():\n        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))\n\n',
        "",
        "remove broad secondary-artifact discovery in favor of exact mounts",
    )
    cells[4] = patch_prediction_cell(cells[4])
    if variant == CAP2_VARIANT:
        cells[5] = patch_cap2_experiment(cells[5])
    if variant == GAP2_VARIANT:
        cells[1] = patch_gap2_experiment(cells[1])
    cells[3] = cells[3] + "\n" + variant_override(variant)
    pieces = [runtime_logging_source(),
        step0_source(
            variant=variant,
            identities=identities,
            population_sha256=population_sha256,
            global_population_sha256=global_population_sha256,
        )
    ]
    for index, cell in enumerate(cells):
        pieces.append(f"\n# ---- PUBLIC V32 CODE CELL {index} ----\n{cell}\n")
        if index == 3:
            pieces.append(
                "run_config({\"variant\": PUBLIC_TEMPORALUNET_VARIANT, "
                "\"thresholds\": {key: value for key, value in os.environ.items() if \"THRESHOLD\" in key}, "
                "\"radii\": {key: value for key, value in os.environ.items() if \"RADIUS\" in key}, "
                "\"windows\": {key: value for key, value in os.environ.items() if \"WINDOW\" in key}, "
                "\"scales\": {key: value for key, value in os.environ.items() if \"SCALE\" in key}, "
                "\"conventions\": {\"immutable_weights\": True, \"effective_biohub_environment\": "
                "{key: value for key, value in os.environ.items() if key.startswith(\"BIOHUB_\")}}})\n"
                "stage_start(\"prediction\", {\"datasets\": len(SELECTED_DATASETS)})\n"
            )
    pieces.append(
        receipt_source(
            variant=variant,
            identities=identities,
            population_sha256=population_sha256,
            global_sha256=global_population_sha256,
        )
    )
    script = "".join(pieces)
    compile(script, "public_temporalunet_kernel.py", "exec")
    forbidden = ("optimizer.step(", "loss.backward(", "train_unet_transformer.py", "fine_tun")
    hits = [token for token in forbidden if token in script]
    if hits:
        raise BaselineError(f"weight-refinement token present in package: {hits}")
    return script


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in BUILD_VARIANTS:
        raise BaselineError(f"unknown variant {args.variant}")
    manifest = validate_population_manifest(args.population_manifest)
    union = tuple(str(value) for value in manifest["union"])
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise BaselineError("invalid shard index/count")
    identities = union[args.shard_index :: args.shard_count]
    if not identities:
        raise BaselineError("selected shard is empty")
    shard_manifest = {
        "schema": "celltrack-alpha.public-temporalunet-shard.v1",
        "variant": args.variant,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "identities": list(identities),
        "global_population_manifest_sha256": manifest["manifest_sha256"],
    }
    shard_sha = canonical_sha256(shard_manifest)
    script = build_script(
        variant=args.variant,
        identities=identities,
        population_sha256=shard_sha,
        global_population_sha256=str(manifest["manifest_sha256"]),
    )
    if args.kernel_dir.exists():
        raise BaselineError(f"refusing to overwrite package directory {args.kernel_dir}")
    args.kernel_dir.mkdir(parents=True)
    code_path = args.kernel_dir / "public_temporalunet_kernel.py"
    code_path.write_text(script, encoding="utf-8")
    metadata = {
        "id": args.kernel_id,
        "title": args.kernel_id.split("/", 1)[1],
        "code_file": code_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [PRIMARY_SOURCE, SECONDARY_SOURCE],
        "model_sources": [],
    }
    metadata_path = args.kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    binding = artifact_binding()
    is_cap2 = args.variant == CAP2_VARIANT
    contract = {
        "schema": "celltrack-alpha.public-temporalunet-prelaunch.v1",
        "authorization": {
            "source": "owner-authorized public-baseline supersession and standing productive Kaggle authorization",
            "action": (
                "experimental immutable-weight public TemporalUNet3D outdegree-two cap"
                if is_cap2
                else "immutable-weight public TemporalUNet3D reproduction shard"
            ),
            "public_submission": False,
            "weight_refinement": False,
        },
        "kernel_id": args.kernel_id,
        "lane_owner": "PUBLIC_TEMPORALUNET_BASELINE_ADOPTION",
        "dataset_sources": metadata["dataset_sources"],
        "dataset_version": 1,
        "exclusive_dataset_source": SECONDARY_SOURCE,
        "expected_mount_root": SECONDARY_ROOT,
        "kernel_properties": {
            "enable_gpu": True,
            "enable_internet": False,
            "enable_tpu": False,
            "machine_shape": "NvidiaTeslaT4",
        },
        "code_sha256": sha256_file(code_path),
        "input_manifest_sha256": binding["binding_sha256"],
        "checkpoint_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_WEIGHT_SHA256,
        "population_manifest_sha256": shard_sha,
        "population_file_count": len(identities),
        "request_count": len(identities),
        "request_file_count": len(identities),
        "wheel_count": 60,
        "remote_preflight_mode": "inline_step0_before_heavy_work",
        "remote_preflight_kernel_id": args.kernel_id,
        "step0_competition_path_visits": len(identities),
        "step0_elapsed_is_hard_failure": False,
        "step0_limit_seconds": 900.0,
        "run_logging": {
            "schema": "celltrack-alpha.run-log.v1",
            "stdout_jsonl": True,
            "persisted_with_artifacts": True,
            "download_on_success": True,
            "inventory_before_resolution": True,
            "failure_traceback_with_state": True,
        },
        "runtime_governance": {
            "per_run_wall_clock_kill": False,
            "projection_gate_h": 8.0,
            "weekly_shared_ledger_cap_h": 30.0,
        },
        "substrate_only": True,
        "weight_refinement_minimum_local_score": 0.916,
        "experiment": (
            {
                "name": CAP2_VARIANT,
                "baseline": "public_v32",
                "stage": "postlink_after_pristine_safe_divisions",
                "rule": "for outdegree > 2 retain top two by edge_sort_key(edge_prob, -distance_um)",
                "safe_division_routine_patched": False,
                "expected_baseline_parents": list(CAP2_EXPECTED_PARENTS),
            }
            if is_cap2
            else None
        ),
    }
    contract_path = args.kernel_dir / "prelaunch-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    package = {
        "schema": "celltrack-alpha.public-temporalunet-package.v1",
        "variant": args.variant,
        "kernel_id": args.kernel_id,
        "shard": {**shard_manifest, "manifest_sha256": shard_sha},
        "artifacts": binding,
        "files": {
            code_path.name: sha256_file(code_path),
            metadata_path.name: sha256_file(metadata_path),
            contract_path.name: sha256_file(contract_path),
        },
        "weight_refinement": False,
        "experiment": contract["experiment"],
    }
    atomic_json(args.kernel_dir / "PACKAGE_MANIFEST.json", package)
    return package


def freeze_package_set(args: argparse.Namespace) -> dict[str, Any]:
    """Seal complete disjoint package/guard coverage for all three modes."""

    manifest = validate_population_manifest(args.population_manifest)
    expected = set(str(value) for value in manifest["union"])
    guards: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in args.prelaunch_receipt:
        guard = load_object(path, "prelaunch guard receipt")
        kernel_id = str(guard.get("kernel_id", ""))
        if not kernel_id or kernel_id in guards:
            raise BaselineError(f"missing or duplicate guard kernel id: {path}")
        guards[kernel_id] = (path, guard)

    packages: list[dict[str, Any]] = []
    coverage: dict[str, set[str]] = {variant: set() for variant in VARIANTS}
    shard_indexes: dict[str, set[int]] = {variant: set() for variant in VARIANTS}
    shard_counts: dict[str, set[int]] = {variant: set() for variant in VARIANTS}
    for directory in args.package_dir:
        package_path = directory / "PACKAGE_MANIFEST.json"
        package = load_object(package_path, "package manifest")
        variant = str(package.get("variant", ""))
        kernel_id = str(package.get("kernel_id", ""))
        shard = package.get("shard", {})
        if (
            package.get("schema") != "celltrack-alpha.public-temporalunet-package.v1"
            or variant not in VARIANTS
            or package.get("weight_refinement") is not False
            or not isinstance(shard, dict)
            or shard.get("global_population_manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise BaselineError(f"inadmissible package in set: {package_path}")
        identities = {str(value) for value in shard.get("identities", [])}
        if not identities or identities - expected or coverage[variant] & identities:
            raise BaselineError(f"bad or overlapping {variant} shard coverage: {package_path}")
        coverage[variant].update(identities)
        shard_indexes[variant].add(int(shard.get("shard_index", -1)))
        shard_counts[variant].add(int(shard.get("shard_count", -1)))
        guard_entry = guards.pop(kernel_id, None)
        if guard_entry is None:
            raise BaselineError(f"package has no unique prelaunch guard: {kernel_id}")
        guard_path, guard = guard_entry
        code_hash = package.get("files", {}).get("public_temporalunet_kernel.py")
        if (
            guard.get("schema") != "celltrack-alpha.kaggle-prelaunch-guard-result.v1"
            or guard.get("status") != "PASS"
            or guard.get("eligible") is not True
            or guard.get("phase") != "inline"
            or guard.get("failures") != []
            or guard.get("code_sha256") != code_hash
        ):
            raise BaselineError(f"inadmissible prelaunch guard: {guard_path}")
        packages.append(
            {
                "variant": variant,
                "kernel_id": kernel_id,
                "shard_index": int(shard["shard_index"]),
                "shard_count": int(shard["shard_count"]),
                "request_count": len(identities),
                "shard_manifest_sha256": shard.get("manifest_sha256"),
                "code_sha256": code_hash,
                "package_manifest": {
                    "path": str(package_path),
                    "sha256": sha256_file(package_path),
                },
                "prelaunch_guard": {
                    "path": str(guard_path),
                    "sha256": sha256_file(guard_path),
                },
            }
        )
    if guards:
        raise BaselineError(f"unused prelaunch guard receipts: {sorted(guards)[:5]}")
    for variant in VARIANTS:
        if coverage[variant] != expected:
            raise BaselineError(f"incomplete {variant} package-set coverage")
        if shard_counts[variant] != {6} or shard_indexes[variant] != set(range(6)):
            raise BaselineError(f"{variant} package set is not exact shards 0..5 of 6")
    packages.sort(key=lambda row: (VARIANTS.index(str(row["variant"])), int(row["shard_index"])))
    result = {
        "schema": "celltrack-alpha.public-temporalunet-package-set.v1",
        "status": "PASS",
        "population_manifest_sha256": manifest["manifest_sha256"],
        "variants": list(VARIANTS),
        "package_count": len(packages),
        "packages_per_variant": 6,
        "identities_per_variant": 162,
        "packages": packages,
        "substrate_only": True,
        "weight_refinement": False,
        "weight_refinement_minimum_local_score": 0.916,
    }
    atomic_json(args.output, result)
    return result


def dataset_from_row(row: Mapping[str, str]) -> str:
    dataset = str(row.get("dataset", ""))
    if not dataset:
        raise BaselineError("submission row has no dataset")
    return dataset


def outdegree_diagnostics(
    edges_by_dataset: Mapping[str, Iterable[tuple[int, int]]],
) -> dict[str, Any]:
    """Describe higher-degree parents without changing an external graph.

    The pinned public V32 notebook has no outdegree cap.  This diagnostic is
    intentionally separate from the strict APNCC graph contract: every
    parent above two is retained and recorded, including the daughters needed
    to make the faithful replication auditable.
    """

    parents: list[dict[str, Any]] = []
    histogram: dict[int, int] = defaultdict(int)
    exact_two = 0
    at_least_two = 0
    max_outdegree = 0
    for dataset in sorted(edges_by_dataset):
        daughters_by_parent: dict[int, list[int]] = defaultdict(list)
        for source_id, target_id in edges_by_dataset[dataset]:
            daughters_by_parent[int(source_id)].append(int(target_id))
        for parent_id in sorted(daughters_by_parent):
            daughters = sorted(daughters_by_parent[parent_id])
            degree = len(daughters)
            histogram[degree] += 1
            max_outdegree = max(max_outdegree, degree)
            exact_two += int(degree == 2)
            at_least_two += int(degree >= 2)
            if degree > 2:
                parents.append(
                    {
                        "dataset": dataset,
                        "parent_id": parent_id,
                        "outdegree": degree,
                        "daughters": daughters,
                    }
                )
    return {
        "policy": "competition_faithful_no_outdegree_cap",
        "outdegree_histogram": {str(degree): histogram[degree] for degree in sorted(histogram)},
        "degree_three_parent_count": sum(
            int(parent["outdegree"]) == 3 for parent in parents
        ),
        "outdegree_gt_two_parent_count": len(parents),
        "max_outdegree": max_outdegree,
        "dividing_nodes_exactly_two": exact_two,
        "branching_nodes_outdegree_at_least_two": at_least_two,
        "parents_with_outdegree_gt_two": parents,
    }


def validate_cap2_parent_log_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Authenticate the exact preregistered parents changed by cap2."""

    parent_records = [
        record for record in records
        if record.get("event") == "experimental_cap2_parent"
    ]
    observed = sorted(
        (
            str(record.get("dataset")),
            int(record.get("parent_id", -1)),
            int(record.get("outdegree_before", -1)),
            tuple(sorted(int(value) for value in record.get("daughters_before", []))),
        )
        for record in parent_records
    )
    expected = sorted(
        (
            str(parent["dataset"]),
            int(parent["parent_id"]),
            int(parent["outdegree"]),
            tuple(int(value) for value in parent["daughters"]),
        )
        for parent in CAP2_EXPECTED_PARENTS
    )
    if observed != expected:
        raise BaselineError(
            "cap2 affected-parent log differs from the preregistered V32 baseline parents"
        )
    for record in parent_records:
        ranking = record.get("ranking")
        retained = record.get("retained_daughters")
        dropped = record.get("dropped_daughters")
        outdegree_before = int(record.get("outdegree_before", -1))
        if (
            not isinstance(ranking, list)
            or len(ranking) != outdegree_before
            or any(not isinstance(item, dict) for item in ranking)
            or not isinstance(retained, list)
            or len(retained) != 2
            or not isinstance(dropped, list)
            or len(dropped) != outdegree_before - 2
        ):
            raise BaselineError("cap2 affected-parent ranking log is malformed")
        target_ids = [int(item["target_id"]) for item in ranking]
        rank_keys = [
            (
                0.0 if item.get("edge_prob") is None else float(item["edge_prob"]),
                -float(item["distance_um"]),
            )
            for item in ranking
        ]
        if (
            target_ids[:2] != [int(value) for value in retained]
            or target_ids[2:] != [int(value) for value in dropped]
            or [bool(item.get("retained")) for item in ranking]
            != [True, True, *([False] * (outdegree_before - 2))]
            or any(rank_keys[index] < rank_keys[index + 1] for index in range(len(rank_keys) - 1))
            or any(not all(math.isfinite(value) for value in key) for key in rank_keys)
        ):
            raise BaselineError("cap2 affected-parent ranking does not prove top-two retention")
    return parent_records


def validate_shard_output(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate one downloaded public V32 output without graph mutation."""

    package_path = args.package_dir / "PACKAGE_MANIFEST.json"
    package = load_object(package_path, "package manifest")
    shard = package.get("shard", {})
    if (
        package.get("schema") != "celltrack-alpha.public-temporalunet-package.v1"
        or package.get("weight_refinement") is not False
        or package.get("variant") not in BUILD_VARIANTS
        or not isinstance(shard, dict)
    ):
        raise BaselineError(f"inadmissible package manifest: {package_path}")
    for name, expected_hash in dict(package.get("files", {})).items():
        path = args.package_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise BaselineError(f"package file hash mismatch: {path}")

    expected_datasets = [
        str(identity).split(":", 1)[1] for identity in shard.get("identities", [])
    ]
    if not expected_datasets or len(expected_datasets) != len(set(expected_datasets)):
        raise BaselineError("package shard identities are empty or duplicated")
    expected_set = set(expected_datasets)
    step0_path = args.output_dir / "PUBLIC_TEMPORALUNET_STEP0.json"
    run_path = args.output_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
    submission_path = args.output_dir / "submission.csv"
    run_stats_path = args.output_dir / "run_stats.csv"
    run_log_path = args.output_dir / "RUN_LOG.jsonl"
    step0 = load_object(step0_path, "remote Step-0 receipt")
    run = load_object(run_path, "remote terminal receipt")
    common = (
        step0.get("status") == "PASS"
        and run.get("status") == "PASS"
        and step0.get("variant") == package["variant"]
        and run.get("variant") == package["variant"]
        and step0.get("population_manifest_sha256") == shard.get("manifest_sha256")
        and run.get("population_manifest_sha256") == shard.get("manifest_sha256")
        and step0.get("global_population_manifest_sha256")
        == shard.get("global_population_manifest_sha256")
        and run.get("global_population_manifest_sha256")
        == shard.get("global_population_manifest_sha256")
        and step0.get("checkpoint_sha256") == PRIMARY_WEIGHT_SHA256
        and step0.get("secondary_checkpoint_sha256") == SECONDARY_WEIGHT_SHA256
        and run.get("primary_weight_sha256") == PRIMARY_WEIGHT_SHA256
        and run.get("secondary_weight_sha256") == SECONDARY_WEIGHT_SHA256
        and run.get("public_notebook_sha256") == PUBLIC_NOTEBOOK_SHA256
        and run.get("public_script_version_id") == 337501010
        and step0.get("weight_refinement_allowed") is False
        and run.get("weight_refinement") is False
        and run.get("substrate_only") is True
        and finite(run.get("weight_refinement_minimum_local_score")) == 0.916
    )
    if not common:
        raise BaselineError(f"remote receipt binding failure: {args.output_dir}")
    if step0.get("request_count") != len(expected_datasets):
        raise BaselineError("Step-0 request count does not match package shard")
    if [str(value) for value in run.get("selected_datasets", [])] != expected_datasets:
        raise BaselineError("terminal selected-dataset order does not match package shard")
    if sha256_file(submission_path) != run.get("submission", {}).get("sha256"):
        raise BaselineError("downloaded shard submission hash mismatch")
    if sha256_file(run_stats_path) != run.get("run_stats", {}).get("sha256"):
        raise BaselineError("downloaded shard run-stats hash mismatch")
    run_log = run.get("run_log")
    legacy_prelogging_v2 = (
        package.get("kernel_id") == UNSHARDED_V2_PRELOGGING_KERNEL_ID
        and sha256_file(package_path) == UNSHARDED_V2_PRELOGGING_PACKAGE_SHA256
        and run_log is None
    )
    if legacy_prelogging_v2:
        logging = {
            "status": "PERSISTED_STRUCTURED_RUN_LOG_UNAVAILABLE_IN_IMMUTABLE_V2",
            "finding_receipt": str(
                ROOT
                / "reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/"
                "V32_UNSHARDED_V2_LOGGING_GAP_20260804.json"
            ),
        }
    else:
        if not isinstance(run_log, dict) or not run_log_path.is_file() or (
            sha256_file(run_log_path) != run_log.get("sha256")
        ):
            raise BaselineError("downloaded shard run-log hash mismatch")
        try:
            log_records = [
                json.loads(line)
                for line in run_log_path.read_text(encoding="utf-8").splitlines()
            ]
            log_events = [record.get("event") for record in log_records]
        except json.JSONDecodeError as exc:
            raise BaselineError(f"downloaded shard run log is not JSONL: {exc}") from exc
        required_log_events = {"config", "input_inventory", "lookup", "stage_end", "preprocess", "terminal"}
        if not required_log_events.issubset(log_events):
            raise BaselineError("downloaded shard run log is missing mandatory telemetry events")
        cap2_log_records: list[Mapping[str, Any]] = []
        if package["variant"] == CAP2_VARIANT:
            if "experimental_cap2_dataset_summary" not in log_events:
                raise BaselineError("cap2 run log is missing dataset summaries")
            cap2_log_records = validate_cap2_parent_log_records(log_records)
        logging = {
            "status": "PERSISTED_STRUCTURED_RUN_LOG_AUTHENTICATED",
            "path": str(run_log_path),
            "sha256": sha256_file(run_log_path),
            "events": sorted(required_log_events),
            "experimental_cap2_parent_records": (
                cap2_log_records if package["variant"] == CAP2_VARIANT else []
            ),
        }
    runtime_hash = str(run.get("runtime_predict_script_sha256", ""))
    if len(runtime_hash) != 64 or any(char not in "0123456789abcdef" for char in runtime_hash):
        raise BaselineError("runtime prediction source hash is malformed")
    unexpected_scratch = [
        str(path)
        for path in (
            args.output_dir / "tracking_repo",
            args.output_dir / "secondary_seed_weights",
            args.output_dir / "public_temporalunet_selected_train",
        )
        if path.exists()
    ]
    if unexpected_scratch:
        raise BaselineError(f"post-receipt scratch cleanup failed: {unexpected_scratch}")

    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from g04_division_hidden_postprocess import read_submission  # noqa: PLC0415

    # The pristine public notebook contains add_safe_divisions_postlink but no
    # outdegree cap/validator.  Faithful V32 replication must therefore not
    # reject or rewrite degree-three parents merely because APNCC is stricter.
    graphs, row_count = read_submission(submission_path, max_outdegree=None)
    if set(graphs) != expected_set:
        raise BaselineError("competition-faithfully parsed shard coverage differs from package")
    degree_diagnostics = outdegree_diagnostics(
        {dataset: graph["edges"] for dataset, graph in graphs.items()}
    )
    if package["variant"] == CAP2_VARIANT:
        if degree_diagnostics["outdegree_gt_two_parent_count"] != 0:
            raise BaselineError("cap2 output still contains an outdegree-greater-than-two parent")
        degree_diagnostics["policy"] = (
            "experimental_cap_outdegree_two_by_v32_edge_sort_key"
        )
    totals = {
        "datasets": len(graphs),
        "nodes": sum(len(graph["nodes"]) for graph in graphs.values()),
        "edges": sum(len(graph["edges"]) for graph in graphs.values()),
        "dividing_nodes": degree_diagnostics["dividing_nodes_exactly_two"],
    }
    if (
        row_count != int(run.get("submission", {}).get("rows", -1))
        or totals["nodes"] != int(run.get("nodes", -1))
        or totals["edges"] != int(run.get("edges", -1))
    ):
        raise BaselineError("strict graph totals differ from terminal receipt")
    result = {
        "schema": "celltrack-alpha.public-temporalunet-shard-terminal-validation.v1",
        "status": "PASS",
        "variant": package["variant"],
        "kernel_id": package["kernel_id"],
        "shard_index": shard.get("shard_index"),
        "shard_count": shard.get("shard_count"),
        "package_manifest": {"path": str(package_path), "sha256": sha256_file(package_path)},
        "step0": {"path": str(step0_path), "sha256": sha256_file(step0_path)},
        "run_receipt": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "submission": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
            "rows": row_count,
        },
        "graphs": totals,
        "outdegree_diagnostics": degree_diagnostics,
        "logging": logging,
        "runtime_predict_script_sha256": runtime_hash,
        "weight_refinement": False,
        "substrate_only": True,
        "weight_refinement_minimum_local_score": 0.916,
    }
    atomic_json(args.output, result)
    return result


def validate_hidden_output(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate a freshly generated dynamic hidden-test V32 output."""

    package_path = args.package_dir / "PACKAGE_MANIFEST.json"
    package = load_object(package_path, "hidden package manifest")
    if (
        package.get("schema") != HIDDEN_PACKAGE_SCHEMA
        or package.get("variant") != "public_v32"
        or package.get("hidden_test_compatible") is not True
        or package.get("dynamic_test_discovery") is not True
        or package.get("public_identity_pinned") is not False
        or package.get("fresh_submission_generation") is not True
        or package.get("public_output_copied") is not False
        or package.get("weight_refinement") is not False
        or package.get("g1_applied") is not False
    ):
        raise BaselineError("hidden package contract is inadmissible")
    for name, expected_hash in dict(package.get("files", {})).items():
        path = args.package_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise BaselineError(f"hidden package file hash mismatch: {path}")

    local_gate = package.get("local_submission_gate", {})
    if (
        not isinstance(local_gate, dict)
        or local_gate.get("schema") != HIDDEN_GATE_SCHEMA
        or local_gate.get("status") != "PASS"
        or local_gate.get("eligible") is not True
        or float(local_gate.get("local_score", 0.0)) < 0.91
        or local_gate.get("public_output_copy_allowed") is not False
        or local_gate.get("g1_assumed_additive") is not False
        or local_gate.get("gate_sha256")
        != canonical_sha256({key: value for key, value in local_gate.items() if key != "gate_sha256"})
    ):
        raise BaselineError("hidden package local submission gate drifted")

    step0_path = args.output_dir / "PUBLIC_TEMPORALUNET_STEP0.json"
    run_path = args.output_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
    submission_path = args.output_dir / "submission.csv"
    run_stats_path = args.output_dir / "run_stats.csv"
    run_log_path = args.output_dir / "RUN_LOG.jsonl"
    step0 = load_object(step0_path, "hidden Step-0 receipt")
    run = load_object(run_path, "hidden terminal receipt")
    selected = [str(value) for value in run.get("selected_datasets", [])]
    step0_selected = [str(value) for value in step0.get("selected_datasets", [])]
    selected_set = set(selected)
    # The hidden rerun inventory is discovered dynamically and may legitimately
    # differ in size from the public dry run ("the hidden dataset can be
    # larger/smaller/different").  Require internal self-consistency at any
    # positive cardinality, never a pinned count.
    discovered_count = len(selected)
    if (
        discovered_count <= 0
        or len(selected_set) != discovered_count
        or selected != step0_selected
    ):
        raise BaselineError("hidden dynamic dataset inventory is not self-consistent")
    common = (
        step0.get("schema") == "celltrack-alpha.public-temporalunet-hidden-step0.v1"
        and run.get("schema") == "celltrack-alpha.public-temporalunet-hidden-run.v1"
        and step0.get("status") == "PASS"
        and run.get("status") == "PASS"
        and step0.get("variant") == "public_v32"
        and run.get("variant") == "public_v32"
        and step0.get("request_count") == discovered_count
        and run.get("request_count") == discovered_count
        and step0.get("population_manifest_sha256") == local_gate.get("gate_sha256")
        and run.get("population_manifest_sha256") == local_gate.get("gate_sha256")
        and run.get("local_submission_gate_sha256") == local_gate.get("gate_sha256")
        and math.isclose(
            float(run.get("local_cv57_score", float("nan"))),
            float(local_gate.get("local_score")),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and step0.get("checkpoint_sha256") == PRIMARY_WEIGHT_SHA256
        and step0.get("secondary_checkpoint_sha256") == SECONDARY_WEIGHT_SHA256
        and run.get("primary_weight_sha256") == PRIMARY_WEIGHT_SHA256
        and run.get("secondary_weight_sha256") == SECONDARY_WEIGHT_SHA256
        and run.get("public_notebook_sha256") == PUBLIC_NOTEBOOK_SHA256
        and run.get("public_script_version_id") == 337501010
        and step0.get("hidden_test_compatible") is True
        and step0.get("dynamic_test_discovery") is True
        and step0.get("public_identity_pinned") is False
        and step0.get("public_output_copied") is False
        and run.get("hidden_test_compatible") is True
        and run.get("dynamic_test_discovery") is True
        and run.get("public_identity_pinned") is False
        and run.get("fresh_submission_generation") is True
        and run.get("public_output_copied") is False
        and run.get("graph_policy") == "competition_faithful_no_outdegree_cap"
        and run.get("weight_refinement") is False
        and run.get("substrate_only") is True
        and run.get("g1_applied") is False
    )
    if not common:
        raise BaselineError("hidden remote receipt binding failure")
    submission_sha = sha256_file(submission_path)
    if submission_sha != run.get("submission", {}).get("sha256"):
        raise BaselineError("hidden submission hash mismatch")
    if submission_sha == local_gate.get("public_output_sha256"):
        raise BaselineError("hidden output is a forbidden copy of the public output")
    if sha256_file(run_stats_path) != run.get("run_stats", {}).get("sha256"):
        raise BaselineError("hidden run-stats hash mismatch")
    if (
        not run_log_path.is_file()
        or sha256_file(run_log_path) != run.get("run_log", {}).get("sha256")
    ):
        raise BaselineError("hidden persisted run-log hash mismatch")
    try:
        log_rows = [
            json.loads(line)
            for line in run_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise BaselineError(f"hidden persisted run log is not JSONL: {exc}") from exc
    log_events = [row.get("event") for row in log_rows]
    required_log_events = {
        "config",
        "input_inventory",
        "lookup",
        "stage_start",
        "stage_end",
        "preprocess",
        "terminal",
    }
    if not required_log_events.issubset(log_events):
        raise BaselineError("hidden persisted run log lacks mandatory telemetry")
    if any(row.get("event") == "failure" for row in log_rows):
        raise BaselineError("hidden persisted run log contains a failure event")

    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from g04_division_hidden_postprocess import read_submission  # noqa: PLC0415

    graphs, row_count = read_submission(submission_path, max_outdegree=None)
    if set(graphs) != selected_set:
        raise BaselineError("hidden submission coverage differs from dynamic receipt")
    degree_diagnostics = outdegree_diagnostics(
        {dataset: graph["edges"] for dataset, graph in graphs.items()}
    )
    totals = {
        "datasets": len(graphs),
        "nodes": sum(len(graph["nodes"]) for graph in graphs.values()),
        "edges": sum(len(graph["edges"]) for graph in graphs.values()),
        "dividing_nodes": degree_diagnostics["dividing_nodes_exactly_two"],
    }
    if (
        row_count != int(run.get("submission", {}).get("rows", -1))
        or totals["nodes"] != int(run.get("nodes", -1))
        or totals["edges"] != int(run.get("edges", -1))
    ):
        raise BaselineError("hidden graph totals differ from terminal receipt")
    result = {
        "schema": "celltrack-alpha.public-temporalunet-hidden-terminal-validation.v1",
        "status": "PASS",
        "eligible_for_submission_predicate": True,
        "kernel_id": package["kernel_id"],
        "package_manifest": {"path": str(package_path), "sha256": sha256_file(package_path)},
        "local_submission_gate_sha256": local_gate["gate_sha256"],
        "step0": {"path": str(step0_path), "sha256": sha256_file(step0_path)},
        "run_receipt": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "submission": {"path": str(submission_path), "sha256": submission_sha, "rows": row_count},
        "graphs": totals,
        "dataset_discovery": {
            "datasets": selected,
            "dynamic_test_discovery": True,
            "public_identity_pinned": False,
            "inventory_exact_match": True,
        },
        "generation": {
            "fresh_submission_generation": True,
            "public_output_copied": False,
            "public_output_sha256": local_gate["public_output_sha256"],
        },
        "outdegree_diagnostics": degree_diagnostics,
        "logging": {
            "status": "PERSISTED_STRUCTURED_RUN_LOG_AUTHENTICATED",
            "path": str(run_log_path),
            "sha256": sha256_file(run_log_path),
            "events": sorted(required_log_events),
        },
        "runtime_predict_script_sha256": run.get("runtime_predict_script_sha256"),
        "weight_refinement": False,
        "g1_applied": False,
    }
    atomic_json(args.output, result)
    return result


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_population_manifest(args.population_manifest)
    expected = {identity.split(":", 1)[1] for identity in manifest["union"]}
    receipts: list[dict[str, Any]] = []
    sources: list[Path] = []
    seen: set[str] = set()
    for shard_dir in args.shard_dir:
        receipt_path = shard_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
        submission_path = shard_dir / "submission.csv"
        receipt = load_object(receipt_path, f"run receipt in {shard_dir}")
        if (
            receipt.get("schema") != RUN_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("variant") != args.variant
            or receipt.get("global_population_manifest_sha256") != manifest["manifest_sha256"]
            or receipt.get("weight_refinement") is not False
        ):
            raise BaselineError(f"inadmissible run receipt: {receipt_path}")
        if sha256_file(submission_path) != receipt.get("submission", {}).get("sha256"):
            raise BaselineError(f"submission hash mismatch: {submission_path}")
        selected = {str(value) for value in receipt.get("selected_datasets", [])}
        if not selected:
            raise BaselineError(f"empty selected-dataset receipt: {receipt_path}")
        if seen & selected:
            raise BaselineError(f"shard dataset overlap: {sorted(seen & selected)[:10]}")
        file_seen: set[str] = set()
        file_rows = 0
        file_nodes = 0
        file_edges = 0
        with submission_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_COLUMNS:
                raise BaselineError(f"bad submission header: {submission_path}")
            for row in reader:
                file_seen.add(dataset_from_row(row))
                file_rows += 1
                if row.get("row_type") == "node":
                    file_nodes += 1
                elif row.get("row_type") == "edge":
                    file_edges += 1
                else:
                    raise BaselineError(f"bad row type in {submission_path}")
        if file_seen != selected:
            raise BaselineError(
                f"shard receipt/CSV coverage mismatch in {shard_dir}: "
                f"missing={sorted(selected - file_seen)[:10]}, "
                f"extra={sorted(file_seen - selected)[:10]}"
            )
        if (
            file_rows != int(receipt.get("submission", {}).get("rows", -1))
            or file_nodes != int(receipt.get("nodes", -1))
            or file_edges != int(receipt.get("edges", -1))
        ):
            raise BaselineError(f"shard count/receipt mismatch: {shard_dir}")
        seen.update(selected)
        receipts.append(receipt)
        sources.append(submission_path)
    if seen != expected:
        raise BaselineError(
            f"merged shard coverage mismatch: missing={sorted(expected - seen)[:10]}, "
            f"extra={sorted(seen - expected)[:10]}"
        )
    if args.output.exists() or args.receipt.exists():
        raise BaselineError("refusing to overwrite merged output or receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise BaselineError(f"refusing to overwrite stale merge temporary: {temporary}")
    row_id = 0
    node_count = 0
    edge_count = 0
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for submission_path in sources:
                with submission_path.open(newline="", encoding="utf-8") as source:
                    reader = csv.DictReader(source)
                    if reader.fieldnames != CSV_COLUMNS:
                        raise BaselineError(f"bad submission header on merge pass: {submission_path}")
                    for row in reader:
                        row["id"] = str(row_id)
                        writer.writerow(row)
                        row_id += 1
                        node_count += row["row_type"] == "node"
                        edge_count += row["row_type"] == "edge"
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    result = {
        "schema": MERGE_SCHEMA,
        "status": "PASS",
        "variant": args.variant,
        "population_manifest_sha256": manifest["manifest_sha256"],
        "shard_count": len(receipts),
        "dataset_count": len(expected),
        "submission": {"path": str(args.output), "sha256": sha256_file(args.output), "rows": row_id},
        "nodes": node_count,
        "edges": edge_count,
        "weight_refinement": False,
    }
    atomic_json(args.receipt, result)
    return result


def merge_local_control(args: argparse.Namespace) -> dict[str, Any]:
    """Merge one exact local fragment per frozen identity for scorer validation."""
    manifest = validate_population_manifest(args.population_manifest)
    expected = tuple(str(value) for value in manifest["union"])
    paths: dict[str, Path] = {}
    for directory in args.fragment_dir:
        for path in directory.glob("*.csv"):
            identity = path.stem.replace("__", ":", 1)
            if identity in paths:
                raise BaselineError(f"duplicate local control fragment for {identity}")
            paths[identity] = path
    if set(paths) != set(expected):
        raise BaselineError(
            f"local control coverage mismatch: missing={sorted(set(expected) - set(paths))[:10]}, "
            f"extra={sorted(set(paths) - set(expected))[:10]}"
        )
    if args.output.exists() or args.receipt.exists():
        raise BaselineError("refusing to overwrite local-control merge output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    nodes = 0
    edges = 0
    fragment_hashes: dict[str, str] = {}
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for identity in expected:
            path = paths[identity]
            fragment_hashes[identity] = sha256_file(path)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != CSV_COLUMNS:
                    raise BaselineError(f"bad local fragment header: {path}")
                for row in reader:
                    if f"train:{dataset_from_row(row)}" != identity:
                        raise BaselineError(f"local fragment identity mismatch: {path}")
                    row["id"] = str(row_id)
                    writer.writerow(row)
                    row_id += 1
                    nodes += row["row_type"] == "node"
                    edges += row["row_type"] == "edge"
    result = {
        "schema": "celltrack-alpha.public-temporalunet-local-control-merge.v1",
        "status": "PASS",
        "name": args.name,
        "population_manifest_sha256": manifest["manifest_sha256"],
        "submission": {"path": str(args.output), "sha256": sha256_file(args.output), "rows": row_id},
        "nodes": nodes,
        "edges": edges,
        "fragment_set_sha256": canonical_sha256(fragment_hashes),
    }
    atomic_json(args.receipt, result)
    return result


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def component_labels(
    node_ids: Iterable[int], edges: Iterable[tuple[int, int]]
) -> dict[int, int]:
    """Return deterministic weak-component labels for one lineage graph."""

    parent = {int(value): int(value) for value in node_ids}

    def find(value: int) -> int:
        root = value
        while root != parent[root]:
            root = parent[root]
        while value != root:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    for raw_source, raw_target in edges:
        source = find(int(raw_source))
        target = find(int(raw_target))
        if source != target:
            if source > target:
                source, target = target, source
            parent[target] = source
    return {value: find(value) for value in parent}


def fragmentation_diagnostic(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, int | float | None]:
    """Separate matched-lineage fragmentation from detection and edge errors.

    A GT lineage component is fragmented when its officially distance-matched
    prediction nodes occupy more than one weak prediction component. Fully
    unmatched GT components are reported as detection misses, not fragmentation.
    Wrong links can reconnect components, so official edge FP/FN remains the
    authoritative ordinary-association diagnostic alongside this topology view.
    """

    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    import tracksdata as td  # noqa: PLC0415
    from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: PLC0415
    from tracking_cellmot.metrics import _evaluate  # noqa: PLC0415
    from tracksdata.metrics import DistanceMatching  # noqa: PLC0415
    from tracksdata.options import get_options, set_options  # noqa: PLC0415

    prediction = prediction_to_graph(nodes, edges)
    truth = geff_arrays_to_graph(gt)
    if prediction.num_nodes() > 0:
        if prediction.num_edges() == 0:
            previous = get_options().show_progress
            set_options(show_progress=False)
            try:
                with warnings.catch_warnings():
                    from scipy.sparse import SparseEfficiencyWarning  # noqa: PLC0415

                    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
                    prediction.match(
                        truth,
                        matching=DistanceMatching(max_distance=7.0, scale=scale_zyx),
                    )
            finally:
                set_options(show_progress=previous)
        else:
            _evaluate(prediction, truth, "jaccard", scale_zyx, 7.0)

    pred_node_ids = [
        int(value)
        for (value,) in prediction.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID]
        ).iter_rows()
    ]
    pred_edges = [
        (int(source), int(target))
        for source, target in prediction.edge_attrs()
        .select("source_id", "target_id")
        .iter_rows()
    ]
    gt_node_ids = [
        int(value)
        for (value,) in truth.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID]
        ).iter_rows()
    ]
    gt_edges = [
        (int(source), int(target))
        for source, target in truth.edge_attrs()
        .select("source_id", "target_id")
        .iter_rows()
    ]
    pred_components = component_labels(pred_node_ids, pred_edges)
    gt_components = component_labels(gt_node_ids, gt_edges)
    segments: dict[int, set[int]] = defaultdict(set)
    if prediction.num_nodes() > 0:
        matched = prediction.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        )
        for pred_id, gt_id in matched.iter_rows():
            if gt_id is None or int(gt_id) == -1:
                continue
            segments[gt_components[int(gt_id)]].add(pred_components[int(pred_id)])

    gt_component_roots = set(gt_components.values())
    segment_counts = [len(segments[root]) for root in sorted(gt_component_roots)]
    matched_counts = [count for count in segment_counts if count > 0]
    fragmented = sum(count > 1 for count in matched_counts)
    excess = sum(max(0, count - 1) for count in matched_counts)
    matched_count = len(matched_counts)
    return {
        "gt_lineage_components": len(gt_component_roots),
        "matched_gt_lineage_components": matched_count,
        "fully_unmatched_gt_lineage_components": len(gt_component_roots) - matched_count,
        "fragmented_matched_gt_lineage_components": fragmented,
        "excess_prediction_segments": excess,
        "matched_component_fragmentation_fraction": (
            fragmented / matched_count if matched_count else None
        ),
        "prediction_components": len(set(pred_components.values())),
    }


def score_submission(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_population_manifest(args.population_manifest)
    identities = tuple(str(value) for value in manifest["populations"][args.population])
    datasets = tuple(identity.split(":", 1)[1] for identity in identities)
    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    import score_bh_cv57  # noqa: PLC0415
    import scoreall  # noqa: PLC0415
    from celltrack_alpha.kaggle_calibration import read_submission  # noqa: PLC0415
    from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: PLC0415
    from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: PLC0415
    import zarr  # noqa: PLC0415
    from zarr.storage import ZipStore  # noqa: PLC0415

    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    competition_faithful_v32 = competition_faithful_graph_variant(args.variant)
    graphs = read_submission(
        args.submission,
        max_outdegree=None if competition_faithful_v32 else 2,
    )
    expected_union = {identity.split(":", 1)[1] for identity in manifest["union"]}
    if set(graphs) != expected_union:
        raise BaselineError("merged submission does not exactly cover the frozen 162-video union")
    rows: list[dict[str, Any]] = []
    fragmentation_rows: list[dict[str, Any]] = []
    store = ZipStore(args.archive, mode="r")
    try:
        for dataset in datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            graph = graphs[dataset]
            nodes = [
                {"dataset": dataset, "node_id": node.node_id, "t": node.t, "z": node.z, "y": node.y, "x": node.x}
                for node in graph.nodes.values()
            ]
            edges = [{"source_id": edge.source_id, "target_id": edge.target_id} for edge in graph.edges]
            scale = parse_scale(dict(group.attrs))
            rows.append({"identity": f"train:{dataset}", **official_metric_row(nodes, edges, gt, scale)})
            fragmentation_rows.append(
                {
                    "identity": f"train:{dataset}",
                    **fragmentation_diagnostic(nodes, edges, gt, scale),
                }
            )
    finally:
        store.close()
    summary = official_summary(rows)
    totals = {
        key: int(sum(int(float(row.get(key, 0))) for row in rows))
        for key in ("num_pred_nodes", "edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    }
    fragmentation_totals = {
        key: int(sum(int(row[key]) for row in fragmentation_rows))
        for key in (
            "gt_lineage_components",
            "matched_gt_lineage_components",
            "fully_unmatched_gt_lineage_components",
            "fragmented_matched_gt_lineage_components",
            "excess_prediction_segments",
            "prediction_components",
        )
    }
    matched_components = fragmentation_totals["matched_gt_lineage_components"]
    fragmentation_totals["matched_component_fragmentation_fraction"] = (
        fragmentation_totals["fragmented_matched_gt_lineage_components"]
        / matched_components
        if matched_components
        else None
    )
    score_outdegree = outdegree_diagnostics(
        {
            dataset: ((edge.source_id, edge.target_id) for edge in graph.edges)
            for dataset, graph in graphs.items()
        }
    )
    if args.variant == CAP2_VARIANT:
        score_outdegree["policy"] = "experimental_cap_outdegree_two_by_v32_edge_sort_key"
        score_outdegree["baseline_degree_three_parent_count"] = len(CAP2_EXPECTED_PARENTS)
    elif not competition_faithful_v32:
        score_outdegree["policy"] = "strict_max_outdegree_2"
    result = {
        "schema": SCORE_SCHEMA,
        "status": "PASS",
        "variant": args.variant,
        "population": args.population,
        "population_manifest_sha256": manifest["manifest_sha256"],
        "submission": {"path": str(args.submission), "sha256": sha256_file(args.submission)},
        "archive": {"path": str(args.archive), "sha256": sha256_file(args.archive)},
        "scorer": scorer,
        "outdegree_diagnostics": score_outdegree,
        "summary": summary,
        "score": finite(summary.get("score")),
        "decomposition": {
            "detection": {"num_pred_nodes": totals["num_pred_nodes"], "node_recall": finite(summary.get("node_recall"))},
            "ordinary_association": {key: totals[key] for key in ("edge_tp", "edge_fp", "edge_fn")},
            "track_fragmentation": {
                **fragmentation_totals,
                "definition": (
                    "officially matched GT lineage components split across more than one "
                    "weak prediction component; fully unmatched GT components are detection misses"
                ),
                "samples": fragmentation_rows,
            },
            "division": {key: totals[key] for key in ("division_tp", "division_fp", "division_fn")},
        },
        "samples": rows,
        "substrate_only": True,
        "weight_refinement": False,
        "weight_refinement_minimum_local_score": 0.916,
    }
    atomic_json(args.output, result)
    # The V32 CV57 score is the first point at which the complete immutable
    # graph and official per-video scorer rows coexist.  Run the generic
    # taxonomy immediately here, rather than relying on a separate watcher or
    # allowing a later mechanism to be funded against the APNCC taxonomy.
    # This branch is intentionally unreachable for shard scoring, held-fit,
    # and APNCC controls.
    if args.variant == "public_v32" and args.population == "cv57":
        taxonomy_root = ROOT / "reports/g04/lanes/cv57_residual_error_taxonomy_20260804"
        v32_output = taxonomy_root / "V32_RESIDUAL_ERROR_TAXONOMY.json"
        v32_report = taxonomy_root / "V32_REPORT.md"
        comparison = taxonomy_root / "APNCC_V32_SIDE_BY_SIDE.md"
        if any(path.exists() for path in (v32_output, v32_report, comparison)):
            raise BaselineError("refusing to overwrite an existing V32 taxonomy receipt")
        command = [
            sys.executable,
            str(taxonomy_root / "build_taxonomy.py"),
            "--control-score", str(args.output),
            "--populations", str(args.population_manifest),
            "--archive", str(args.archive),
            "--scorer-parity", str(args.scorer_parity),
            "--output", str(v32_output),
            "--report", str(v32_report),
            "--baseline-label", "immutable public TemporalUNet V32 CV57",
            "--comparison-with", str(taxonomy_root / "CV57_RESIDUAL_ERROR_TAXONOMY.json"),
            "--comparison-output", str(comparison),
        ]
        taxonomy_env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            env=taxonomy_env,
        )
        if completed.returncode:
            raise BaselineError(
                "V32 taxonomy harness failed after score publication: "
                + completed.stderr.strip()
            )
        result["taxonomy_harness"] = {
            "status": "PASS",
            "taxonomy": str(v32_output),
            "report": str(v32_report),
            "comparison": str(comparison),
        }
    return result


def summarize_division_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        key: int(sum(int(float(row.get(key, 0))) for row in rows))
        for key in ("division_tp", "division_fp", "division_fn", "predicted_dividing_nodes")
    }
    denominator = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    totals["division_jaccard"] = (
        totals["division_tp"] / denominator if denominator else None
    )
    totals["gt_division_count"] = totals["division_tp"] + totals["division_fn"]
    totals["video_count"] = len(rows)
    return totals


def score_shard_divisions(args: argparse.Namespace) -> dict[str, Any]:
    """Score one authenticated shard's native forks with the frozen official evaluator."""

    package_path = args.package_dir / "PACKAGE_MANIFEST.json"
    package = load_object(package_path, "package manifest")
    shard = package.get("shard", {})
    if (
        package.get("schema") != "celltrack-alpha.public-temporalunet-package.v1"
        or package.get("variant") != "public_v32"
        or package.get("weight_refinement") is not False
        or not isinstance(shard, dict)
    ):
        raise BaselineError("division diagnostic requires an immutable public_v32 package")
    run_path = args.output_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json"
    submission_path = args.output_dir / "submission.csv"
    run = load_object(run_path, "remote terminal receipt")
    expected_datasets = [
        str(identity).split(":", 1)[1] for identity in shard.get("identities", [])
    ]
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "PASS"
        or run.get("variant") != "public_v32"
        or run.get("population_manifest_sha256") != shard.get("manifest_sha256")
        or run.get("weight_refinement") is not False
        or [str(value) for value in run.get("selected_datasets", [])] != expected_datasets
        or sha256_file(submission_path) != run.get("submission", {}).get("sha256")
    ):
        raise BaselineError("shard division diagnostic receipt binding failed")

    manifest = validate_population_manifest(args.population_manifest)
    populations = {
        identity.split(":", 1)[1]: name
        for name, identities in manifest["populations"].items()
        for identity in identities
    }
    for location in (ROOT / "scripts", ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    import score_bh_cv57  # noqa: PLC0415
    from celltrack_alpha.kaggle_calibration import read_submission  # noqa: PLC0415
    from celltrack_alpha.official_eval import official_metric_row  # noqa: PLC0415
    from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: PLC0415
    import zarr  # noqa: PLC0415
    from zarr.storage import ZipStore  # noqa: PLC0415

    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    graphs = read_submission(submission_path)
    if set(graphs) != set(expected_datasets):
        raise BaselineError("shard submission coverage differs from frozen package")
    rows: list[dict[str, Any]] = []
    store = ZipStore(args.archive, mode="r")
    try:
        for dataset in expected_datasets:
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            graph = graphs[dataset]
            nodes = [
                {
                    "dataset": dataset,
                    "node_id": node.node_id,
                    "t": node.t,
                    "z": node.z,
                    "y": node.y,
                    "x": node.x,
                }
                for node in graph.nodes.values()
            ]
            edges = [
                {"source_id": edge.source_id, "target_id": edge.target_id}
                for edge in graph.edges
            ]
            outdegree: dict[int, int] = defaultdict(int)
            for edge in graph.edges:
                outdegree[int(edge.source_id)] += 1
            metrics = official_metric_row(nodes, edges, gt, parse_scale(dict(group.attrs)))
            rows.append(
                {
                    "identity": f"train:{dataset}",
                    "family": dataset.split("_", 1)[0],
                    "population": populations[dataset],
                    "predicted_dividing_nodes": sum(value == 2 for value in outdegree.values()),
                    "division_tp": int(metrics["division_tp"]),
                    "division_fp": int(metrics["division_fp"]),
                    "division_fn": int(metrics["division_fn"]),
                    "division_jaccard": finite(metrics["division_jaccard"]),
                }
            )
    finally:
        store.close()

    by_family = {
        family: summarize_division_rows([row for row in rows if row["family"] == family])
        for family in sorted({str(row["family"]) for row in rows})
    }
    by_population = {
        population: summarize_division_rows(
            [row for row in rows if row["population"] == population]
        )
        for population in sorted({str(row["population"]) for row in rows})
    }
    aggregate = summarize_division_rows(rows)
    result = {
        "schema": "celltrack-alpha.public-temporalunet-shard-native-division.v1",
        "status": "PASS",
        "variant": "public_v32",
        "shard_index": shard.get("shard_index"),
        "shard_count": shard.get("shard_count"),
        "package_manifest": {"path": str(package_path), "sha256": sha256_file(package_path)},
        "run_receipt": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "archive": {"path": str(args.archive), "sha256": sha256_file(args.archive)},
        "scorer": scorer,
        "aggregate": aggregate,
        "by_family": by_family,
        "by_population": by_population,
        "samples": rows,
        "interpretation": (
            "ADD_AND_REFINE_NATIVE_SIGNAL"
            if (
                aggregate["division_tp"] > 0
                and aggregate["division_fp"] > 0
                and aggregate["division_fn"] > 0
            )
            else "REFINE_NATIVE_SIGNAL"
            if aggregate["division_tp"] > 0 and aggregate["division_fp"] > 0
            else "ADD_MISSING_SIGNAL"
            if aggregate["division_fn"] > 0
            else "NATIVE_SIGNAL_SUFFICIENT"
        ),
        "diagnostic_only": True,
        "weight_refinement": False,
    }
    atomic_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze-populations")
    freeze.add_argument("--fit-selection", type=Path, default=ROOT / "reports/g02/selected_graph_appearance_audit_v1.json")
    freeze.add_argument("--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json")
    freeze.add_argument("--archive", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(func=freeze_populations)

    build = sub.add_parser("build-package")
    build.add_argument("--population-manifest", type=Path, required=True)
    build.add_argument("--variant", choices=BUILD_VARIANTS, required=True)
    build.add_argument("--shard-index", type=int, required=True)
    build.add_argument("--shard-count", type=int, required=True)
    build.add_argument("--kernel-id", required=True)
    build.add_argument("--kernel-dir", type=Path, required=True)
    build.set_defaults(func=build_package)

    hidden_build = sub.add_parser("build-hidden-package")
    hidden_build.add_argument("--score-receipt", type=Path, required=True)
    hidden_build.add_argument("--taxonomy-receipt", type=Path, required=True)
    hidden_build.add_argument("--terminal-authentication", type=Path, required=True)
    hidden_build.add_argument("--expected-test-count", type=int, required=True)
    hidden_build.add_argument("--kernel-id", required=True)
    hidden_build.add_argument("--kernel-dir", type=Path, required=True)
    hidden_build.set_defaults(func=build_hidden_package)

    package_set = sub.add_parser("freeze-package-set")
    package_set.add_argument("--population-manifest", type=Path, required=True)
    package_set.add_argument("--package-dir", type=Path, action="append", required=True)
    package_set.add_argument(
        "--prelaunch-receipt", type=Path, action="append", required=True
    )
    package_set.add_argument("--output", type=Path, required=True)
    package_set.set_defaults(func=freeze_package_set)

    merge = sub.add_parser("merge")
    merge.add_argument("--population-manifest", type=Path, required=True)
    merge.add_argument("--variant", choices=VARIANTS, required=True)
    merge.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--receipt", type=Path, required=True)
    merge.set_defaults(func=merge_shards)

    validate_shard = sub.add_parser("validate-shard")
    validate_shard.add_argument("--package-dir", type=Path, required=True)
    validate_shard.add_argument("--output-dir", type=Path, required=True)
    validate_shard.add_argument("--output", type=Path, required=True)
    validate_shard.set_defaults(func=validate_shard_output)

    validate_hidden = sub.add_parser("validate-hidden-output")
    validate_hidden.add_argument("--package-dir", type=Path, required=True)
    validate_hidden.add_argument("--output-dir", type=Path, required=True)
    validate_hidden.add_argument("--output", type=Path, required=True)
    validate_hidden.set_defaults(func=validate_hidden_output)

    local_merge = sub.add_parser("merge-local-control")
    local_merge.add_argument("--population-manifest", type=Path, required=True)
    local_merge.add_argument("--fragment-dir", type=Path, action="append", required=True)
    local_merge.add_argument("--name", required=True)
    local_merge.add_argument("--output", type=Path, required=True)
    local_merge.add_argument("--receipt", type=Path, required=True)
    local_merge.set_defaults(func=merge_local_control)

    score = sub.add_parser("score")
    score.add_argument("--population-manifest", type=Path, required=True)
    score.add_argument("--variant", choices=(*BUILD_VARIANTS, "apncc_control"), required=True)
    score.add_argument("--population", choices=("held_fit105", "cv57"), required=True)
    score.add_argument("--submission", type=Path, required=True)
    score.add_argument("--archive", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    score.add_argument("--scorer-parity", type=Path, default=ROOT / "reports/g02/scorer_parity_v2.json")
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(func=score_submission)

    shard_division = sub.add_parser("score-shard-divisions")
    shard_division.add_argument("--package-dir", type=Path, required=True)
    shard_division.add_argument("--output-dir", type=Path, required=True)
    shard_division.add_argument("--population-manifest", type=Path, required=True)
    shard_division.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    shard_division.add_argument(
        "--scorer-parity", type=Path, default=ROOT / "reports/g02/scorer_parity_v2.json"
    )
    shard_division.add_argument("--output", type=Path, required=True)
    shard_division.set_defaults(func=score_shard_divisions)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.func(args)
    except (BaselineError, OSError, ValueError) as exc:
        print(f"public_temporalunet_baseline: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

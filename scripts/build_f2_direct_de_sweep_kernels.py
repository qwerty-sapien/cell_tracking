#!/usr/bin/env python3
"""Build two self-contained, no-submission F2 D/E direct-sweep kernels.

Each kernel runs the exact six frozen F2 movies once per preregistered candidate.
The child V32 package is embedded byte-for-byte and is permitted to use CPU only
under the ship-certifier's bounded preflight environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V32_BUILDER_PATH = ROOT / "scripts/build_v32_preilp_cache_kernel.py"
CANDIDATES_PATH = (
    ROOT / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/F1_CANDIDATES.json"
)
SUBSAMPLE_PATH = (
    ROOT / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/F2_SUBSAMPLE.json"
)
V8_D_REJECTION_PATH = (
    ROOT
    / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
    "f2_d_v8_native_interactive_cert/downloaded_native_t4/extracted/"
    "F2_DIRECT_D_SWEEP_V8_NATIVE_CERTIFICATION.json"
)
EXPECTED_V8_D_REJECTION_SHA256 = (
    "7b48ca347954487d2e6991e3bf5872d86f951e58773d61fb1b4990ee7fd1496a"
)
V9_D_REJECTION_PATH = (
    ROOT
    / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
    "f2_d_v9_native_interactive_cert/downloaded_native_t4/extracted/"
    "F2_DIRECT_D_SWEEP_V9_NATIVE_CERTIFICATION.json"
)
EXPECTED_V9_D_REJECTION_SHA256 = (
    "5370c55e18d0ec25ec673258e44a9b8ef40e15ff75123c53ecd10362bcfc174c"
)

FAMILY_ENV = {
    "DET_THRESHOLD": "F2_DET_THRESHOLD",
    "SECONDARY_EDGE_WEIGHT": "F2_SECONDARY_EDGE_WEIGHT",
    "SECONDARY_DETECTION_WEIGHT": "F2_SECONDARY_DETECTION_WEIGHT",
    "DUAL_SEED_EDGE_THRESHOLD": "F2_DUAL_SEED_EDGE_THRESHOLD",
    "SECONDARY_LOW_MARGIN_MAX": "F2_SECONDARY_LOW_MARGIN_MAX",
}


class SweepBuildError(ValueError):
    """The direct-sweep package could not be frozen safely."""


def _load_v32_builder():
    spec = importlib.util.spec_from_file_location("_f2_v32_builder", V32_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise SweepBuildError(f"cannot import {V32_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def v8_rejection_binding() -> dict[str, Any]:
    if sha256_file(V8_D_REJECTION_PATH) != EXPECTED_V8_D_REJECTION_SHA256:
        raise SweepBuildError("exact D-v8 native rejection receipt hash drift")
    receipt = json.loads(V8_D_REJECTION_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema": "ship-kaggle-kernels.certification.v1",
        "status": "REJECTED",
        "eligible": False,
        "autonomous_launch_eligible": False,
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v8",
        "artifact_sha256": (
            "d742f51783e7638ff3a2dcbddeb017ad0382080450d1d08f30347ea3b5c91f7a"
        ),
        "profile_sha256": (
            "8398fade50ddd54723c36f67408dd3a207704ef12fa120d12d5ced2cf80aad28"
        ),
        "package_inventory_sha256": (
            "fbe577532b9c4d93578ff5a72e41010c307b311cb7c6dd08e3a059c7260a3ae5"
        ),
    }
    drift = [key for key, value in expected.items() if receipt.get(key) != value]
    if drift:
        raise SweepBuildError(f"exact D-v8 native rejection binding drift: {drift}")
    return {
        "path": V8_D_REJECTION_PATH.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_V8_D_REJECTION_SHA256,
        **expected,
    }


def v9_rejection_binding() -> dict[str, Any]:
    if sha256_file(V9_D_REJECTION_PATH) != EXPECTED_V9_D_REJECTION_SHA256:
        raise SweepBuildError("exact D-v9 native rejection receipt hash drift")
    receipt = json.loads(V9_D_REJECTION_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema": "ship-kaggle-kernels.certification.v1",
        "status": "REJECTED",
        "eligible": False,
        "autonomous_launch_eligible": False,
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v9",
        "artifact_sha256": (
            "3c81ca3b5c088f061e0f3835953556c0bd3aafeb36ed8bfe3cf864333d8b63e1"
        ),
        "profile_sha256": (
            "5d8061a80bff26b639ae84b2c9ce876ed08b0b28902cb7e4303ca729f028806f"
        ),
        "package_inventory_sha256": (
            "e2929647561e51de2607f53aad77a25def9641db9a3849d8f57427f1e2b1c61f"
        ),
    }
    drift = [key for key, value in expected.items() if receipt.get(key) != value]
    if drift:
        raise SweepBuildError(f"exact D-v9 native rejection binding drift: {drift}")
    if receipt.get("static", {}).get("status") != "PASS":
        raise SweepBuildError("D-v9 static certification was not PASS")
    if receipt.get("repeatability", {}).get("status") != "PASS":
        raise SweepBuildError("D-v9 repeatability was not PASS")
    if receipt.get("quick_debug", {}).get("status") != "PASS":
        raise SweepBuildError("D-v9 focused pre-push debug was not PASS")
    simulations = receipt.get("simulation_runs", [])
    if (
        len(simulations) != 8
        or any(run.get("status") != "PASS" or run.get("returncode") != 0 for run in simulations)
        or [run.get("workload_units") for run in simulations]
        != [3, 3, 6, 6, 12, 12, 108, 108]
    ):
        raise SweepBuildError("D-v9 successful simulation evidence drift")
    projection = receipt.get("runtime_projection", {})
    if projection.get("status") != "FAIL" or projection.get("failures") != [
        "timing spread too high for scenario tiny_sparse_first_3: 1.242241",
        "projected peak RSS exceeds the eligible memory threshold",
    ]:
        raise SweepBuildError("D-v9 projection-only rejection evidence drift")
    return {
        "path": V9_D_REJECTION_PATH.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_V9_D_REJECTION_SHA256,
        **expected,
        "static_status": "PASS",
        "repeatability_status": "PASS",
        "quick_debug_status": "PASS",
        "successful_simulations": 8,
        "projection_status": "FAIL",
    }


def _replace_exact(source: str, old: str, new: str, expected: int, label: str) -> str:
    count = source.count(old)
    if count != expected:
        raise SweepBuildError(f"{label}: expected {expected} anchors, found {count}")
    return source.replace(old, new)


def parameterize_child_source(source: str) -> str:
    replacements = (
        (
            'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"',
            'os.environ["BIOHUB_DET_THRESHOLD"] = os.environ.get('
            '"F2_DET_THRESHOLD", "0.96875")',
            1,
            "detection threshold",
        ),
        (
            'os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"',
            'os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = os.environ.get('
            '"F2_SECONDARY_EDGE_WEIGHT", "0.15")',
            1,
            "secondary edge weight bootstrap",
        ),
        (
            "os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'",
            'os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = os.environ.get('
            '"F2_SECONDARY_EDGE_WEIGHT", "0.15")',
            1,
            "secondary edge weight override",
        ),
        (
            'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.5"',
            'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = os.environ.get('
            '"F2_SECONDARY_DETECTION_WEIGHT", "0.5")',
            1,
            "secondary detection weight bootstrap",
        ),
        (
            "os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'] = '0.5'",
            'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = os.environ.get('
            '"F2_SECONDARY_DETECTION_WEIGHT", "0.5")',
            1,
            "secondary detection weight override",
        ),
        (
            'os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"',
            'os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = os.environ.get('
            '"F2_SECONDARY_LOW_MARGIN_MAX", "0.35")',
            1,
            "low-margin bootstrap",
        ),
        (
            "os.environ['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] = '0.35'",
            'os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = os.environ.get('
            '"F2_SECONDARY_LOW_MARGIN_MAX", "0.35")',
            1,
            "low-margin override",
        ),
        (
            'os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"',
            'os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = os.environ.get('
            '"F2_DUAL_SEED_EDGE_THRESHOLD", "0.48")',
            1,
            "dual-seed threshold bootstrap",
        ),
        (
            "os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'] = '0.48'",
            'os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = os.environ.get('
            '"F2_DUAL_SEED_EDGE_THRESHOLD", "0.48")',
            1,
            "dual-seed threshold override",
        ),
    )
    result = source
    for old, new, count, label in replacements:
        result = _replace_exact(result, old, new, count, label)
    compile(result, "f2_v32_parameterized_child.py", "exec")
    return result


def frozen_candidates(family: str) -> list[dict[str, Any]]:
    if family not in {"D", "E"}:
        raise SweepBuildError("family must be D or E")
    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    rows = [row for row in payload["candidates"] if row.get("family") == family]
    expected = 6 if family == "D" else 8
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise SweepBuildError(f"frozen family {family} count/uniqueness drift")
    for row in rows:
        delta = row.get("delta")
        if not isinstance(delta, dict) or len(delta) != 1:
            raise SweepBuildError(f"candidate {row.get('id')} must have one delta")
        key = next(iter(delta))
        if key not in FAMILY_ENV:
            raise SweepBuildError(f"unsupported F2 delta {key}")
    return rows


def runtime_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key, value = next(iter(row["delta"].items()))
        result.append(
            {
                "id": str(row["id"]),
                "family": str(row["family"]),
                "delta": {str(key): str(value)},
                "environment": {FAMILY_ENV[str(key)]: str(value)},
            }
        )
    result.append(
        {
            "id": "CONTROL_V32",
            "family": "CONTROL",
            "delta": {},
            "environment": {},
        }
    )
    return result


def orchestrator_source(
    *,
    child_source: str,
    candidates: Sequence[Mapping[str, Any]],
    family: str,
    bounded_sequential_envelope: bool = False,
) -> str:
    child_literal = repr(child_source)
    candidate_literal = repr(list(candidates))
    return textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        # F2 direct family-{family} sweep; generated, frozen, and no-submission.
        import hashlib
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        FAMILY = {family!r}
        CHILD_SOURCE = {child_literal}
        CANDIDATES = {candidate_literal}
        BOUNDED_SEQUENTIAL_ENVELOPE = {bounded_sequential_envelope!r}
        WORKING = Path("/kaggle/working")
        OUTPUT_ROOT = WORKING / f"f2_direct_{{FAMILY.lower()}}_sweep_v1"
        CHILD_PATH = WORKING / "f2_v32_parameterized_child.py"
        MAX_SECONDS = 7 * 60 * 60
        MOVED_OUTPUTS = (
            "v32_preilp_cache_v1",
            "V32_FINAL_GRAPH_ROWS.csv",
            "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json",
            "PUBLIC_TEMPORALUNET_STEP0.json",
            "V32_SHIP_PREFLIGHT_COMPLETION.json",
            "RUN_LOG.jsonl",
            "run_stats.csv",
        )


        def sha256_file(path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()


        def atomic_json(path, value):
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            temporary.replace(path)


        def relay_successful_child_streams(stdout_path, stderr_path):
            stdout_payload = stdout_path.read_bytes()
            stderr_payload = stderr_path.read_bytes()
            if b"\\x00" in stdout_payload:
                raise RuntimeError("successful child stdout contains a NUL byte")
            stdout_sha256 = hashlib.sha256(stdout_payload).hexdigest()
            stderr_sha256 = hashlib.sha256(stderr_payload).hexdigest()
            sys.stdout.write(
                f"F2_CHILD_STDOUT_RELAY_BEGIN bytes={{len(stdout_payload)}} "
                f"sha256={{stdout_sha256}}\\n"
            )
            sys.stdout.flush()
            sys.stdout.buffer.write(stdout_payload)
            if stdout_payload and not stdout_payload.endswith(b"\\n"):
                sys.stdout.buffer.write(b"\\n")
            sys.stdout.buffer.flush()
            sys.stdout.write(
                f"F2_CHILD_STDOUT_RELAY_END sha256={{stdout_sha256}}\\n"
                f"F2_CHILD_STDERR_RETAINED bytes={{len(stderr_payload)}} "
                f"sha256={{stderr_sha256}}\\n"
            )
            sys.stdout.flush()


        def inventory(root):
            rows = []
            pending = [root]
            while pending:
                directory = pending.pop()
                for path in sorted(directory.iterdir(), key=lambda value: value.name):
                    if path.is_symlink():
                        raise RuntimeError(f"sweep output contains symlink: {{path}}")
                    if path.is_dir():
                        pending.append(path)
                    elif path.is_file():
                        rows.append({{
                            "path": str(path.relative_to(root)),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }})
            rows.sort(key=lambda row: row["path"])
            payload = (json.dumps(
                rows, allow_nan=False, separators=(",", ":"), sort_keys=True
            ) + "\\n").encode("ascii")
            return rows, hashlib.sha256(payload).hexdigest()


        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
        OUTPUT_ROOT.mkdir(parents=True)
        CHILD_PATH.write_text(CHILD_SOURCE, encoding="utf-8")
        if os.environ.get("V32_RUNTIME_SAFETY_PROBE", "").strip():
            raise SystemExit(subprocess.run(
                [sys.executable, str(CHILD_PATH)],
                env=os.environ.copy(),
                check=False,
            ).returncode)
        preflight = bool(os.environ.get("KAGGLE_PREFLIGHT_MODE", "").strip())
        preflight_candidate_count_text = os.environ.get(
            "KAGGLE_PREFLIGHT_CANDIDATES", ""
        ).strip()
        if preflight:
            if BOUNDED_SEQUENTIAL_ENVELOPE:
                try:
                    preflight_candidate_count = int(preflight_candidate_count_text)
                except ValueError as exc:
                    raise RuntimeError(
                        "KAGGLE_PREFLIGHT_CANDIDATES must be an integer"
                    ) from exc
                if not 1 <= preflight_candidate_count <= len(CANDIDATES):
                    raise RuntimeError("KAGGLE_PREFLIGHT_CANDIDATES is out of range")
            else:
                if preflight_candidate_count_text:
                    raise RuntimeError(
                        "KAGGLE_PREFLIGHT_CANDIDATES requires the bounded sequential envelope"
                    )
                preflight_candidate_count = 1
            selected_candidates = CANDIDATES[:preflight_candidate_count]
        else:
            if preflight_candidate_count_text:
                raise RuntimeError(
                    "KAGGLE_PREFLIGHT_CANDIDATES requires KAGGLE_PREFLIGHT_MODE"
                )
            selected_candidates = CANDIDATES
        started = time.monotonic()
        completed = []

        for candidate in selected_candidates:
            elapsed = time.monotonic() - started
            remaining = MAX_SECONDS - elapsed
            if remaining <= 60:
                raise RuntimeError("F2 sweep wall-time guard reached before next candidate")
            candidate_root = OUTPUT_ROOT / candidate["id"]
            candidate_root.mkdir()
            atomic_json(candidate_root / "CANDIDATE_CONFIG.json", candidate)
            environment = os.environ.copy()
            environment.update(candidate["environment"])
            if BOUNDED_SEQUENTIAL_ENVELOPE:
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = "0"
            environment["F2_SWEEP_CANDIDATE_ID"] = candidate["id"]
            stdout_path = candidate_root / "child.stdout.log"
            stderr_path = candidate_root / "child.stderr.log"
            child_started = time.monotonic()
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.run(
                    [sys.executable, str(CHILD_PATH)],
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                    timeout=max(1, int(remaining)),
                )
            if process.returncode != 0:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                print(tail, file=sys.stderr)
                raise RuntimeError(
                    f"F2 candidate {{candidate['id']}} failed with {{process.returncode}}"
                )
            relay_successful_child_streams(stdout_path, stderr_path)
            if (WORKING / "submission.csv").exists():
                raise RuntimeError("F2 direct sweep child left submission.csv behind")
            for name in MOVED_OUTPUTS:
                source = WORKING / name
                if source.exists():
                    shutil.move(str(source), str(candidate_root / name))
            cache_root = candidate_root / "v32_preilp_cache_v1" / "datasets"
            if not cache_root.is_dir():
                raise RuntimeError(f"F2 candidate {{candidate['id']}} cache is absent")
            frame_counts = {{}}
            for dataset_root in sorted(cache_root.iterdir()):
                receipt = json.loads(
                    (dataset_root / "PRE_ILP_RECEIPT.json").read_text(encoding="utf-8")
                )
                frame_counts[dataset_root.name] = int(receipt["frame_count"])
            completed.append({{
                "candidate_id": candidate["id"],
                "delta": candidate["delta"],
                "completed_units": sum(frame_counts.values()),
                "frames_by_identity": frame_counts,
                "elapsed_seconds": time.monotonic() - child_started,
            }})
            print(f"F2_SWEEP_CANDIDATE_COMPLETE {{candidate['id']}}", flush=True)

        files, inventory_sha256 = inventory(OUTPUT_ROOT)
        completion = {{
            "schema": "celltrack-alpha.f2-direct-de-sweep-completion.v1",
            "status": "PASS",
            "family": FAMILY,
            "preflight": preflight,
            "candidate_ids": [row["candidate_id"] for row in completed],
            "completed_candidates": completed,
            "completed_units": sum(row["completed_units"] for row in completed),
            "output_inventory_sha256": inventory_sha256,
            "output_files": len(files),
            "submission_absent": not (WORKING / "submission.csv").exists(),
            "elapsed_seconds": time.monotonic() - started,
        }}
        atomic_json(WORKING / "F2_SWEEP_COMPLETION.json", completion)
        if not completion["submission_absent"]:
            raise RuntimeError("submission.csv exists at F2 sweep terminal publication")
        print("F2_DIRECT_SWEEP_COMPLETE", flush=True)
        '''
    )


def build_package(
    *,
    family: str,
    kernel_id: str,
    output_dir: Path,
    parent_rejection_version: str = "v8",
    bounded_sequential_envelope: bool = False,
) -> dict[str, Any]:
    if parent_rejection_version == "v8":
        parent_rejection = v8_rejection_binding()
    elif parent_rejection_version == "v9":
        parent_rejection = v9_rejection_binding()
    else:
        raise SweepBuildError("parent rejection version must be v8 or v9")
    builder = _load_v32_builder()
    population = json.loads(builder.POPULATION_DEFAULT.read_text(encoding="utf-8"))
    child_source, child_freeze = builder.build_instrumented_source(
        population_manifest=population,
        role="f2_six",
    )
    child_source = parameterize_child_source(child_source)
    rows = frozen_candidates(family)
    candidates = runtime_candidates(rows)
    subsample = json.loads(SUBSAMPLE_PATH.read_text(encoding="utf-8"))
    expected_identities = [value.removeprefix("train:") for value in subsample["subsample"]]
    observed_identities = [value.removeprefix("train:") for value in child_freeze["identities"]]
    if observed_identities != expected_identities:
        raise SweepBuildError("child population differs from frozen F2 subsample")
    source = orchestrator_source(
        child_source=child_source,
        candidates=candidates,
        family=family,
        bounded_sequential_envelope=bounded_sequential_envelope,
    )
    compile(source, f"f2_direct_{family.lower()}_sweep_kernel.py", "exec")
    if output_dir.exists():
        raise SweepBuildError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    code_name = f"f2_direct_{family.lower()}_sweep_kernel.py"
    code_path = output_dir / code_name
    code_path.write_text(source, encoding="utf-8")
    metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/", 1)[-1],
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "model_sources": [],
    }
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = {
        "schema": "celltrack-alpha.f2-direct-de-sweep-source-freeze.v1",
        "status": "PASS_BUILD_ONLY",
        "family": family,
        "kernel_id": kernel_id,
        f"parent_{parent_rejection_version}_native_rejection": parent_rejection,
        "candidate_source_sha256": sha256_file(CANDIDATES_PATH),
        "subsample_source_sha256": sha256_file(SUBSAMPLE_PATH),
        "candidate_ids": [row["id"] for row in candidates],
        "frozen_experimental_candidate_ids": [row["id"] for row in candidates[:-1]],
        "same_run_control_id": "CONTROL_V32",
        "candidates": candidates,
        "identities": list(child_freeze["identities"]),
        "child_source_sha256": sha256_bytes(child_source.encode("utf-8")),
        "child_freeze": child_freeze,
        "promotion_gate": {
            "minimum_adjusted_delta": 0.005,
            "minimum_raw_delta": 0.0,
            "minimum_node_recall_delta": 0.0,
            "screen_kill_adjusted_delta_lte": -0.002,
        },
        "label_free_kernel": True,
        "evaluate_in_kernel": False,
        "submission_written": False,
        "launch_authorized_by_this_manifest": False,
        "files": {
            code_name: sha256_file(code_path),
            "kernel-metadata.json": sha256_file(metadata_path),
        },
    }
    if bounded_sequential_envelope:
        predictor_path = (
            ROOT
            / "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/"
            "provider_primary/repo/scripts/predict_unet_transformer.py"
        )
        predictor_source = predictor_path.read_text(encoding="utf-8")
        required_predictor_anchors = (
            "for name in tqdm(test_names",
            "coords, edges = predict_video(",
            "save_graph(graph, output_dir / f\"{name}.geff\")",
        )
        if any(anchor not in predictor_source for anchor in required_predictor_anchors):
            raise SweepBuildError("sequential predictor source anchors drift")
        if any(
            token in predictor_source
            for token in ("multiprocessing", "ProcessPoolExecutor", "ThreadPoolExecutor")
        ):
            raise SweepBuildError("predictor source contains unbounded concurrency")
        freeze["runtime_envelope_proof"] = {
            "schema": "celltrack-alpha.f2-v10-bounded-sequential-envelope.v1",
            "status": "PASS_SOURCE_BOUND",
            "artifact_sha256": sha256_file(code_path),
            "child_source_sha256": freeze["child_source_sha256"],
            "predictor_source_sha256": sha256_file(predictor_path),
            "candidate_process_concurrency": 1,
            "visible_cuda_devices": ["0"],
            "predictor_dataset_iteration": "sequential_for_name",
            "production_frame_ceiling": 100,
            "preflight_frame_ceiling": 100,
            "candidate_outputs_accumulate_on_disk_only": True,
            "science_or_ilp_modified": False,
        }
    freeze_path = output_dir / "F2_SWEEP_SOURCE_FREEZE.json"
    freeze_path.write_bytes(canonical_bytes(freeze))
    return {**freeze, "source_freeze_sha256": sha256_file(freeze_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--family", choices=("D", "E"), required=True)
    value.add_argument("--kernel-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--parent-rejection-version", choices=("v8", "v9"), default="v8")
    value.add_argument("--bounded-sequential-envelope", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = build_package(
        family=args.family,
        kernel_id=args.kernel_id,
        output_dir=args.output_dir,
        parent_rejection_version=args.parent_rejection_version,
        bounded_sequential_envelope=args.bounded_sequential_envelope,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed authentication of the CELLECT closeout feature payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


AMENDMENT_SHA256 = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
SUPPLIER_SHA256 = "8d34445eafa19dcd135b59747c6ccb4314b0e2ca19f28e80fcfa752bab9d3664"
WEIGHT_SHA256 = "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
SAFE_LOADER_SHA256 = "2394797f4cb680cd45136c83792df67a3ee8915b6925994a6135d4425998b0c0"
EVALUATOR_SHA256 = "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
REQUEST_MANIFEST_SHA256 = "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
SEALED_B0_SHA256 = "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2"
EXPECTED_FILES = {"delete": 41, "occupied_swap": 36}
EXPECTED_ACTIONS = {"delete": 1_345_497, "occupied_swap": 45_961}
FEATURE_WIDTH = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--b0-sealed", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite {args.output}")

    pinned_inputs = {
        "request_manifest": (args.request_manifest, REQUEST_MANIFEST_SHA256),
        "population_manifest": (
            args.population_manifest,
            POPULATION_MANIFEST_SHA256,
        ),
        "evaluator": (args.evaluator, EVALUATOR_SHA256),
        "sealed_b0": (args.b0_sealed, SEALED_B0_SHA256),
    }
    observed_pins: dict[str, dict[str, str]] = {}
    for name, (path, expected) in pinned_inputs.items():
        observed = sha256(path)
        require(observed == expected, f"{name} hash mismatch: {observed}")
        observed_pins[name] = {
            "path": str(path.resolve()),
            "sha256": observed,
        }

    request_manifest = json.loads(args.request_manifest.read_text())
    require(
        request_manifest.get("schema")
        == "celltrack-alpha.cellect-action-request-manifest.v1",
        "request manifest schema mismatch",
    )
    require(
        request_manifest.get("status") == "FROZEN_COMPLETE",
        "request manifest is not frozen complete",
    )
    require(
        request_manifest.get("population_manifest_sha256")
        == POPULATION_MANIFEST_SHA256,
        "request/population binding mismatch",
    )
    require(request_manifest.get("request_file_count") == 77, "request file count mismatch")
    require(request_manifest.get("request_count") == 1_391_458, "request action count mismatch")
    population_manifest = json.loads(args.population_manifest.read_text())
    require(
        population_manifest.get("schema")
        == "celltrack-alpha.cellect-separation-population.v1",
        "population manifest schema mismatch",
    )
    require(population_manifest.get("status") == "COMPLETE", "population is not complete")
    require(
        population_manifest.get("totals", {}).get("delete", {}).get("actions")
        == EXPECTED_ACTIONS["delete"],
        "DELETE population total mismatch",
    )
    require(
        population_manifest.get("totals", {}).get("occupied_swap", {}).get("actions")
        == EXPECTED_ACTIONS["occupied_swap"],
        "SWAP population total mismatch",
    )

    population_by_key: dict[tuple[str, str], dict[str, object]] = {}
    expected_population_paths: set[Path] = set()
    for action_class in EXPECTED_FILES:
        videos = population_manifest["classes"][action_class]["videos"]
        require(
            len(videos) == EXPECTED_FILES[action_class],
            f"{action_class} population file count mismatch",
        )
        for row in videos:
            key = (action_class, str(row["identity"]))
            require(key not in population_by_key, f"duplicate population identity: {key}")
            path = (
                args.population_root
                / "population"
                / action_class
                / f"{str(row['identity']).replace(':', '__', 1)}.npz"
            )
            require(path.is_file(), f"missing population file: {path}")
            require(
                sha256(path) == row["population_sha256"],
                f"population hash mismatch: {path}",
            )
            count = int(row["count"])
            with np.load(path, allow_pickle=False) as loaded:
                require(
                    {"oracle_label", "fold", "b0", "fp_budget"}.issubset(loaded.files),
                    f"population keyset incomplete: {path}",
                )
                require(len(loaded["oracle_label"]) == count, f"population count mismatch: {path}")
                require(loaded["fold"].shape == (count,), f"fold shape mismatch: {path}")
                require(loaded["b0"].shape[0] == count, f"B0 row count mismatch: {path}")
                require(loaded["fp_budget"].size >= 1, f"FP budget missing: {path}")
            population_by_key[key] = row
            expected_population_paths.add(path.resolve())
    observed_population_paths = {
        path.resolve() for path in (args.population_root / "population").rglob("*.npz")
    }
    require(
        observed_population_paths == expected_population_paths,
        "population tree contains missing or unmanifested NPZ files",
    )

    manifest = json.loads(args.feature_manifest.read_text())
    require(
        manifest.get("schema") == "celltrack-alpha.cellect-action-features.v1",
        "feature manifest schema mismatch",
    )
    require(manifest.get("status") == "COMPLETE", "feature manifest is not complete")
    expected_manifest_pins = {
        "amendment_sha256": AMENDMENT_SHA256,
        "supplier_sha256": SUPPLIER_SHA256,
        "weight_sha256": WEIGHT_SHA256,
        "safe_loader_sha256": SAFE_LOADER_SHA256,
    }
    for key, expected in expected_manifest_pins.items():
        require(manifest.get(key) == expected, f"feature manifest {key} mismatch")
    require(manifest.get("full_pool_cache") is False, "full-pool cache flag mismatch")
    require(manifest.get("graph") is False, "graph flag mismatch")
    require(manifest.get("family_6bba") is False, "6bba flag mismatch")
    require(manifest.get("cv57") is False, "CV57 flag mismatch")

    expected_rows = request_manifest["requests"]
    actual_rows = manifest.get("files")
    require(isinstance(actual_rows, list), "feature file list missing")
    require(len(actual_rows) == len(expected_rows) == 77, "feature file count mismatch")

    class_files = {key: 0 for key in EXPECTED_FILES}
    class_actions = {key: 0 for key in EXPECTED_ACTIONS}
    authenticated_files: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    for ordinal, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
        action_class = expected["action_class"]
        identity = expected["identity"]
        count = int(expected["action_count"])
        output = f"{action_class}/{identity.replace(':', '__', 1)}.npz"
        require(actual.get("action_class") == action_class, f"class mismatch at {ordinal}")
        require(actual.get("identity") == identity, f"identity mismatch at {ordinal}")
        require(actual.get("count") == count, f"action count mismatch at {ordinal}")
        population_row = population_by_key.get((action_class, identity))
        require(population_row is not None, f"feature has no population row at {ordinal}")
        require(population_row["count"] == count, f"feature/population count mismatch at {ordinal}")
        require(
            population_row["request_sha256"] == expected["sha256"],
            f"request/population hash mismatch at {ordinal}",
        )
        require(
            actual.get("request_sha256") == expected["sha256"],
            f"request hash mismatch at {ordinal}",
        )
        require(actual.get("output") == output, f"output path mismatch at {ordinal}")
        feature_path = args.feature_root / output
        require(feature_path.is_file(), f"missing feature file: {feature_path}")
        observed_output_sha = sha256(feature_path)
        require(
            actual.get("output_sha256") == observed_output_sha,
            f"output hash mismatch: {feature_path}",
        )
        with np.load(feature_path, allow_pickle=False) as loaded:
            require(set(loaded.files) == {"a0", "a1"}, f"array keyset mismatch: {feature_path}")
            for arm in ("a0", "a1"):
                array = loaded[arm]
                require(
                    array.shape == (count, FEATURE_WIDTH),
                    f"{arm} shape mismatch: {feature_path} {array.shape}",
                )
                require(np.isfinite(array).all(), f"{arm} contains non-finite values: {feature_path}")
        class_files[action_class] += 1
        class_actions[action_class] += count
        expected_paths.add(feature_path.resolve())
        authenticated_files.append(
            {
                "ordinal": ordinal,
                "action_class": action_class,
                "identity": identity,
                "count": count,
                "path": str(feature_path.resolve()),
                "request_sha256": expected["sha256"],
                "output_sha256": observed_output_sha,
            }
        )

    require(class_files == EXPECTED_FILES, f"per-class file counts mismatch: {class_files}")
    require(class_actions == EXPECTED_ACTIONS, f"per-class action counts mismatch: {class_actions}")
    observed_paths = {path.resolve() for path in args.feature_root.rglob("*.npz")}
    require(
        observed_paths == expected_paths,
        "feature tree contains missing or unmanifested NPZ files",
    )

    report = {
        "schema": "celltrack-alpha.cellect-feature-payload-authentication.v1",
        "status": "PASS_AUTHENTICATED_FOR_A1_A0",
        "feature_root": str(args.feature_root.resolve()),
        "feature_manifest": {
            "path": str(args.feature_manifest.resolve()),
            "sha256": sha256(args.feature_manifest),
        },
        "pinned_inputs": observed_pins,
        "manifest_pins": expected_manifest_pins,
        "file_count": len(authenticated_files),
        "action_count": sum(class_actions.values()),
        "per_class_file_count": class_files,
        "per_class_action_count": class_actions,
        "files": authenticated_files,
        "evaluation_authorization": "A1_A0_MAY_RUN_ONLY_WITH_THIS_PASS_RECEIPT",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "files": 77, "actions": 1_391_458}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

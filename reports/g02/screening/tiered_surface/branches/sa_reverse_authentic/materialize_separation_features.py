#!/usr/bin/env python3
"""Map frozen reverse-cycle rows to the unchanged per-class separation interface."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POPULATION_MANIFEST_SHA = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
AMENDMENT_SHA = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
SEED_NAME = "TRACKASTRA-AUTHENTIC-REVERSE-44B6-V1"
FEATURE_NAMES = (
    "mean_reverse_edge_score",
    "mean_reverse_null_mass",
    "mean_role_oriented_forward_margin",
    "mean_role_oriented_reverse_margin",
    "mean_role_oriented_cycle_min_margin",
    "mean_cycle_abs_disagreement",
    "mean_cycle_sign_agreement",
    "supplier_edge_support",
)


class SeparationFeatureError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SeparationFeatureError(f"JSON object required: {path}")
    return value


def edge_evidence(supplier_path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Aggregate role-oriented label-free evidence for every observed frozen edge."""

    with np.load(supplier_path, allow_pickle=False) as loaded:
        z = {key: loaded[key] for key in loaded.files}
    required = {
        "challengers",
        "conflicts",
        "proposal_offsets",
        "forward_margin",
        "reverse_challenger_score",
        "reverse_conflict_score",
        "reverse_margin",
        "reverse_challenger_null",
        "reverse_conflict_null",
        "cycle_abs_disagreement",
        "cycle_sign_agreement",
    }
    if not required <= set(z):
        raise SeparationFeatureError("supplier payload lacks required arrays")
    challengers = np.asarray(z["challengers"], dtype=np.int64)
    conflicts = np.asarray(z["conflicts"], dtype=np.int64)
    offsets = np.asarray(z["proposal_offsets"], dtype=np.int64)
    forward = np.asarray(z["forward_margin"], dtype=np.float64)
    reverse = np.asarray(z["reverse_margin"], dtype=np.float64)
    disagreement = np.asarray(z["cycle_abs_disagreement"], dtype=np.float64)
    agreement = np.asarray(z["cycle_sign_agreement"], dtype=np.float64)
    challenger_score = np.asarray(z["reverse_challenger_score"], dtype=np.float64)
    conflict_score = np.asarray(z["reverse_conflict_score"], dtype=np.float64)
    challenger_null = np.asarray(z["reverse_challenger_null"], dtype=np.float64)
    conflict_null = np.asarray(z["reverse_conflict_null"], dtype=np.float64)
    if (
        challengers.shape != (len(offsets) - 1, 2)
        or conflicts.shape != (len(forward), 2)
        or int(offsets[-1]) != len(forward)
        or any(
            value.shape != (len(forward),)
            for value in (
                reverse,
                disagreement,
                agreement,
                challenger_score,
                conflict_score,
                challenger_null,
                conflict_null,
            )
        )
    ):
        raise SeparationFeatureError("supplier array alignment drifted")

    observations: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for proposal_index, pair in enumerate(challengers):
        start = int(offsets[proposal_index])
        stop = int(offsets[proposal_index + 1])
        for row in range(start, stop):
            observations[tuple(map(int, pair))].append(
                np.asarray(
                    (
                        challenger_score[row],
                        challenger_null[row],
                        forward[row],
                        reverse[row],
                        min(forward[row], reverse[row]),
                        disagreement[row],
                        agreement[row],
                    ),
                    dtype=np.float64,
                )
            )
    for row, pair in enumerate(conflicts):
        observations[tuple(map(int, pair))].append(
            np.asarray(
                (
                    conflict_score[row],
                    conflict_null[row],
                    -forward[row],
                    -reverse[row],
                    min(-forward[row], -reverse[row]),
                    disagreement[row],
                    agreement[row],
                ),
                dtype=np.float64,
            )
        )
    result = {}
    for edge, values in observations.items():
        mean = np.mean(np.stack(values), axis=0)
        result[edge] = np.concatenate((mean, np.asarray((1.0,)))).astype(
            np.float32
        )
    if any(value.shape != (8,) or not np.isfinite(value).all() for value in result.values()):
        raise SeparationFeatureError("aggregated edge evidence is malformed")
    return result


def permute_within_transition(
    values: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    source_t: np.ndarray,
    identity: str,
) -> np.ndarray:
    """Capacity-matched deterministic cyclic permutation within video/transition."""

    output = np.empty_like(values)
    for frame in np.unique(source_t):
        indices = np.flatnonzero(source_t == frame)
        order = sorted(
            indices.tolist(),
            key=lambda index: (
                hashlib.sha256(
                    (
                        f"{SEED_NAME}|{identity}|{int(frame)}|"
                        f"{int(source_ids[index])}|{int(target_ids[index])}"
                    ).encode("ascii")
                ).digest(),
                int(source_ids[index]),
                int(target_ids[index]),
                int(index),
            ),
        )
        if not order:
            continue
        rotated = order[1:] + order[:1]
        for destination, source in zip(order, rotated, strict=True):
            output[destination] = values[source]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplier-root", type=Path, required=True)
    parser.add_argument("--supplier-manifest-sha", required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    supplier_manifest_path = args.supplier_root / "manifest.json"
    if sha256(supplier_manifest_path) != args.supplier_manifest_sha:
        raise SystemExit("supplier manifest SHA mismatch")
    supplier_manifest = load_json(supplier_manifest_path)
    if (
        supplier_manifest.get("status") != "COMPLETE"
        or supplier_manifest.get("classification") != "NON-GT-IDENTIFIABLE"
        or supplier_manifest.get("totals", {}).get("video_count") != 36
        or supplier_manifest.get("determinism", {}).get("all_array_equal") is not True
    ):
        raise SystemExit("supplier is incomplete or nondeterministic")
    if sha256(args.population_manifest) != POPULATION_MANIFEST_SHA:
        raise SystemExit("population manifest SHA mismatch")
    population = load_json(args.population_manifest)
    supplier_files = {
        str(row["identity"]): row for row in supplier_manifest["files"]
    }
    args.output.mkdir(parents=True)
    files = []
    totals: dict[str, dict[str, int]] = {}
    for action_class in ("delete", "occupied_swap"):
        class_rows = population["classes"][action_class]["videos"]
        action_count = supported_count = 0
        for row in class_rows:
            identity = str(row["identity"])
            supplier_record = supplier_files.get(identity)
            supplier_path = (
                None
                if supplier_record is None
                else args.supplier_root / str(supplier_record["path"])
            )
            if (
                supplier_path is not None
                and sha256(supplier_path) != supplier_record["sha256"]
            ):
                raise SystemExit(f"supplier file SHA mismatch: {identity}")
            request_path = args.workspace_root / str(row["request_path"])
            if sha256(request_path) != row["request_sha256"]:
                raise SystemExit(f"request SHA mismatch: {identity} {action_class}")
            with np.load(request_path, allow_pickle=False) as loaded:
                request = {key: loaded[key] for key in loaded.files}
            count = int(row["count"])
            if any(
                np.asarray(request[key]).shape[0] != count
                for key in ("source_id", "target_id", "source_t", "target_t")
            ) or not np.array_equal(
                np.asarray(request["target_t"]), np.asarray(request["source_t"]) + 1
            ):
                raise SeparationFeatureError("action request alignment drifted")
            evidence = (
                {} if supplier_path is None else edge_evidence(supplier_path)
            )
            a1 = np.zeros((count, 8), dtype=np.float32)
            for index, edge in enumerate(
                zip(
                    np.asarray(request["source_id"], dtype=np.int64),
                    np.asarray(request["target_id"], dtype=np.int64),
                    strict=True,
                )
            ):
                observed = evidence.get(tuple(map(int, edge)))
                if observed is not None:
                    a1[index] = observed
            a0 = permute_within_transition(
                a1,
                np.asarray(request["source_id"], dtype=np.int64),
                np.asarray(request["target_id"], dtype=np.int64),
                np.asarray(request["source_t"], dtype=np.int64),
                identity,
            )
            if (
                a1.shape != (count, 8)
                or a0.shape != (count, 8)
                or not np.isfinite(a1).all()
                or not np.isfinite(a0).all()
                or int(a1[:, -1].sum()) != int(a0[:, -1].sum())
                or not np.allclose(
                    a1.sum(axis=0, dtype=np.float64),
                    a0.sum(axis=0, dtype=np.float64),
                    rtol=1e-7,
                    atol=1e-6,
                )
            ):
                raise SeparationFeatureError("feature/control integrity failure")
            destination = (
                args.output
                / action_class
                / f"{identity.replace(':', '__', 1)}.npz"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, a1=a1, a0=a0)
            supported = int(a1[:, -1].sum())
            action_count += count
            supported_count += supported
            files.append(
                {
                    "identity": identity,
                    "action_class": action_class,
                    "count": count,
                    "supported": supported,
                    "request_sha256": row["request_sha256"],
                    "supplier_sha256": (
                        None
                        if supplier_record is None
                        else supplier_record["sha256"]
                    ),
                    "supplier_identity_covered": supplier_record is not None,
                    "path": str(destination.relative_to(args.output)),
                    "sha256": sha256(destination),
                }
            )
        totals[action_class] = {
            "actions": action_count,
            "supported": supported_count,
            "unsupported_zero_evidence": action_count - supported_count,
        }
    manifest = {
        "schema": "g02.trackastra-reverse-separation-features.v1",
        "status": "COMPLETE",
        "classification": "NON-GT-IDENTIFIABLE",
        "amendment_sha256": AMENDMENT_SHA,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA,
        "supplier_manifest_sha256": args.supplier_manifest_sha,
        "feature_names": FEATURE_NAMES,
        "unsupported_action_contract": "seven evidence values exactly zero plus support=0",
        "A1": "authentic label-free reverse/cycle evidence",
        "A0": "capacity-matched deterministic cyclic within-video within-transition permutation",
        "disabled_cycle_beta": "all eight columns zero; algebraically identical to sealed B0",
        "totals": totals,
        "files": files,
        "labels_read": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
    }
    manifest_path = args.output / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit C1 fit-only identities and run deterministic non-scoring dry passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.c1_conflict_set_attention import (  # noqa: E402
    ConflictSetAttention,
    abstention_scores,
    component_examples,
    conflict_components,
    fold_index,
    nested_partitions,
    parameter_count,
    seed_everything,
)
from celltrack_alpha.trackastra_linear import (  # noqa: E402
    challenger_labels,
    labeled_subset,
    proposal_features,
)


FEATURE_REPORT = ROOT / "reports/g02/trackastra_h1_feature_cache/report.json"
PREPARE = (
    ROOT
    / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
PREPARED_ROOT = PREPARE.parent / "examples"


class C1AuditError(RuntimeError):
    """C1 audit failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise C1AuditError(f"expected JSON object: {path}")
    return value


def load_payload(record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(record["feature_path"])
    if sha256(path) != record["feature_sha256"]:
        raise C1AuditError(f"feature hash mismatch: {record['identity']}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("identity") != record["identity"]
        or value.get("contract_sha256") != record["contract_sha256"]
        or value.get("implementation_sha256")
        != record["implementation_sha256"]
    ):
        raise C1AuditError(f"feature identity mismatch: {record['identity']}")
    return value


def proposal_labels(identity: str, payload: dict[str, Any]) -> np.ndarray:
    path = PREPARED_ROOT / f"{identity.replace(':', '__', 1)}.pt"
    if not path.is_file():
        raise C1AuditError(f"fit label payload missing: {identity}")
    prepared = torch.load(path, map_location="cpu", weights_only=False)
    known = challenger_labels(
        np.asarray(prepared["labeled_challenger_pairs"]),
        np.asarray(prepared["labeled_labels"]),
    )
    challengers = np.asarray(payload["challengers"], dtype=np.int64)
    return np.asarray(
        [
            known.get((int(pair[0]), int(pair[1])), 0)
            for pair in challengers
        ],
        dtype=np.float32,
    )


def deterministic_dry(payload: dict[str, Any]) -> bytes:
    seed_everything()
    features = proposal_features(payload).astype(np.float32)
    components = conflict_components(payload)[:8]
    model = ConflictSetAttention().eval()
    chunks = []
    with torch.no_grad():
        for indices in components:
            values = torch.from_numpy(features[indices])[None]
            logits, no_link = model(values)
            chunks.append(
                abstention_scores(logits, no_link)
                .cpu()
                .numpy()
                .astype("<f4")
                .tobytes()
            )
    return b"".join(chunks)


def audit() -> dict[str, Any]:
    started = time.monotonic()
    report = load_json(FEATURE_REPORT)
    prepare = load_json(PREPARE)
    files = report.get("files")
    fit_ids = tuple(prepare.get("scope", {}).get("train_video_ids", ()))
    if (
        report.get("schema")
        != "celltrack-alpha.trackastra-full-pool-feature-cache.v2"
        or report.get("verdict") != "PASS_LABEL_FREE_FEATURE_CACHE"
        or not isinstance(files, list)
        or len(files) != 105
        or len(fit_ids) != 105
        or set(fit_ids) != {str(row["identity"]) for row in files}
    ):
        raise C1AuditError("feature and fit-only identity scopes disagree")

    identities = sorted(fit_ids)
    partitions = nested_partitions(identities)
    fold_counts = {
        str(fold): sum(fold_index(identity) == fold for identity in identities)
        for fold in range(5)
    }
    family_counts = {
        family: sum(f":{family}_" in identity for identity in identities)
        for family in ("44b6", "6bba")
    }
    row_count = 0
    proposal_count = 0
    component_count = 0
    maximum_component = 0
    labeled_proposals = 0
    positive_proposals = 0
    zero_label_videos = []
    checkpoint_hashes = {}
    first_payload = None
    for raw in files:
        record = dict(raw)
        payload = load_payload(record)
        identity = str(record["identity"])
        path = PREPARED_ROOT / f"{identity.replace(':', '__', 1)}.pt"
        checkpoint_hashes[identity] = {
            "feature_sha256": str(record["feature_sha256"]),
            "label_sha256": sha256(path),
        }
        prepared = torch.load(path, map_location="cpu", weights_only=False)
        known = challenger_labels(
            np.asarray(prepared["labeled_challenger_pairs"]),
            np.asarray(prepared["labeled_labels"]),
        )
        challengers = np.asarray(payload["challengers"], dtype=np.int64)
        matched_indices = np.asarray(
            [
                index
                for index, pair in enumerate(challengers)
                if (int(pair[0]), int(pair[1])) in known
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [
                known[
                    (
                        int(challengers[index, 0]),
                        int(challengers[index, 1]),
                    )
                ]
                for index in matched_indices
            ],
            dtype=np.int8,
        )
        if len(matched_indices) == 0:
            zero_label_videos.append(identity)
        elif proposal_features(payload)[matched_indices].shape != (
            len(labels),
            49,
        ):
            raise C1AuditError(f"labeled join failed: {identity}")
        components = conflict_components(payload)
        row_count += int(record["row_count"])
        proposal_count += int(record["proposal_count"])
        component_count += len(components)
        maximum_component = max(maximum_component, *(map(len, components)))
        labeled_proposals += len(matched_indices)
        positive_proposals += int(labels.sum())
        if first_payload is None:
            first_payload = payload

    if first_payload is None:
        raise C1AuditError("no feature payloads were audited")
    dry_one = deterministic_dry(first_payload)
    dry_two = deterministic_dry(first_payload)
    if dry_one != dry_two:
        raise C1AuditError("dry predictions are not byte-identical")
    model = ConflictSetAttention()
    return {
        "schema": "celltrack-alpha.c1-mechanism-manifest-audit.v1",
        "status": "PASS",
        "scope": {
            "fit_train105_only": True,
            "validation12_access": False,
            "cv57_access": False,
            "graph_scorer_called": False,
            "family_counts": family_counts,
            "fold_counts": fold_counts,
            "nested_partition_count": len(partitions),
        },
        "cache": {
            "feature_report_sha256": sha256(FEATURE_REPORT),
            "prepare_sha256": sha256(PREPARE),
            "video_count": len(identities),
            "proposal_count": proposal_count,
            "row_count": row_count,
            "component_count": component_count,
            "maximum_component_size": maximum_component,
            "labeled_proposal_count": labeled_proposals,
            "positive_proposal_count": positive_proposals,
            "zero_label_video_count": len(zero_label_videos),
            "zero_label_videos": zero_label_videos,
            "checkpoint_hashes": checkpoint_hashes,
        },
        "architecture": {
            "trainable_parameter_count": parameter_count(model),
            "body_frozen": True,
            "graph_topology_messages": False,
            "cross_component_messages": False,
            "explicit_no_link": True,
        },
        "dry_validation": {
            "repeat_count": 2,
            "byte_identical": True,
            "prediction_sha256": hashlib.sha256(dry_one).hexdigest(),
            "official_scorer_called": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise C1AuditError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

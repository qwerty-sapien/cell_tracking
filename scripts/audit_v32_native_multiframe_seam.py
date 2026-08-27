"""Authenticate the observation-distinct V32 multi-frame lineage seam.

This is a source/schema audit only. It never imports the model, opens image data,
loads weights, reads labels/scorer outputs, or executes scientific inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_TRAIN_SHA256 = (
    "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea"
)
EXPECTED_PREDICT_SHA256 = (
    "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
)
EXPECTED_CACHE_SHA256 = (
    "bd6fb773e4bb871a7429299921702c2f1f98c046465b21d83bbb22f2b50aac71"
)

DEFAULT_TRAIN = Path(
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "repo/scripts/train_unet_transformer.py"
)
DEFAULT_PREDICT = Path(
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "repo/scripts/predict_unet_transformer.py"
)
DEFAULT_CACHE = Path("src/celltrack_alpha/v32_preilp_cache.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_methods(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _literal_assignment(tree: ast.AST, name: str) -> Any:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            return ast.literal_eval(value.args[0])
        return ast.literal_eval(value)
    raise ValueError(f"assignment not found: {name}")


def _config_literal(tree: ast.AST, name: str) -> dict[str, Any]:
    value = _literal_assignment(tree, name)
    if not isinstance(value, dict):
        raise TypeError(f"{name} is not a literal dict")
    return value


def audit_paths(
    train_path: Path,
    predict_path: Path,
    cache_path: Path,
    *,
    expected_train_sha256: str = EXPECTED_TRAIN_SHA256,
    expected_predict_sha256: str = EXPECTED_PREDICT_SHA256,
    expected_cache_sha256: str = EXPECTED_CACHE_SHA256,
) -> dict[str, Any]:
    paths = {"train": train_path, "predict": predict_path, "cache": cache_path}
    for role, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"{role} source must be a regular non-symlink file: {path}"
            )

    sources = {role: path.read_text(encoding="utf-8") for role, path in paths.items()}
    trees = {
        role: ast.parse(source, filename=str(paths[role]))
        for role, source in sources.items()
    }
    observed_hashes = {role: _sha256(path) for role, path in paths.items()}
    expected_hashes = {
        "train": expected_train_sha256,
        "predict": expected_predict_sha256,
        "cache": expected_cache_sha256,
    }

    methods = _class_methods(trees["train"], "UNetNodeTransformer")
    config = _config_literal(trees["predict"], "_DEFAULT_CONFIG")
    node_fields = [row[0] for row in _literal_assignment(trees["cache"], "NODE_DTYPE")]
    edge_fields = [
        row[0] for row in _literal_assignment(trees["cache"], "PRE_EDGE_DTYPE")
    ]
    cached_fields = set(node_fields) | set(edge_fields)

    predicates = {
        "exact_source_hashes": observed_hashes == expected_hashes,
        "target_native_feature_width_32": config.get("unet_out_channels") == 32,
        "baseline_window_size_2": config.get("window_size") == 2,
        "encoder_exposes_dense_temporal_features": (
            "encode" in methods
            and "return unet_out, det_logits" in sources["train"]
            and "(B, W, C_feat, *spatial)" in sources["train"]
        ),
        "node_feature_indexing_exists": (
            "_index_features" in methods
            and "feat_maps[b, :, z, y, x].T" in sources["train"]
            and "model._index_features(" in sources["predict"]
        ),
        "pairwise_transformer_exists": (
            "predict_edges" in methods
            and "return self.transformer(" in sources["train"]
            and "edge_logits_pair = model.predict_edges(" in sources["predict"]
        ),
        "baseline_edges_are_adjacent_only": (
            "for f_idx in range(W - 1):" in sources["predict"]
            and "t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]"
            in sources["predict"]
            and "unet_out[:, f_idx + 1]" in sources["predict"]
        ),
        "raw_pair_logits_exist_before_activation": (
            "raw = edge_logits_pair[0]" in sources["predict"]
            and "torch.softmax(raw" in sources["predict"]
            and "torch.sigmoid(raw" in sources["predict"]
        ),
        "cache_omits_node_features": not any(
            "feat" in field.lower() for field in cached_fields
        ),
        "cache_omits_raw_logits": not any(
            "logit" in field.lower() for field in cached_fields
        ),
        "cache_retains_only_scalar_edge_probability": "edge_prob" in edge_fields,
    }
    passed = all(predicates.values())
    return {
        "schema": "celltrack-alpha.v32-native-multiframe-seam-audit.v1",
        "candidate_id": "V32-NATIVE-MULTIFRAME-LINEAGE-v1",
        "mode": "audit_only",
        "scientific_model_executed": False,
        "protected_outcomes_opened": False,
        "observed_sha256": observed_hashes,
        "expected_sha256": expected_hashes,
        "baseline": {
            "unet_out_channels": config.get("unet_out_channels"),
            "window_size": config.get("window_size"),
            "node_cache_fields": node_fields,
            "pre_edge_cache_fields": edge_fields,
        },
        "predicates": predicates,
        "predicate_count": len(predicates),
        "predicate_pass_count": sum(predicates.values()),
        "observation_distinct_seam": (
            "export the existing 32-D node features and raw pair logits, then apply a fixed "
            "five-frame setwise lineage model instead of the current adjacent-pair scorer"
        ),
        "verdict": "GO" if passed else "PARK_DEPENDENCY",
        "next_gate": (
            "freeze one five-frame architecture/configuration/seed and validate a separate "
            "fit-only exploratory plan; this audit is not efficacy evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--predict", type=Path, default=DEFAULT_PREDICT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    receipt = audit_paths(
        resolve(args.train), resolve(args.predict), resolve(args.cache)
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if receipt["verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

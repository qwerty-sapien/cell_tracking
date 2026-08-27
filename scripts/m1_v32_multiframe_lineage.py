"""Frozen local feasibility seam for V32 native five-frame lineage.

This module never discovers or opens competition data.  The exporter consumes
already-computed in-memory V32 node features and pair logits, and authenticates
the exact fit-only allowlist plus the exact source files that define those
values.  The model is one fixed feasibility realization, not a trained or
score-bearing result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # exporter/firewall remain usable without torch
    torch = None
    nn = None


CANDIDATE_ID = "V32-NATIVE-MULTIFRAME-LINEAGE-v1"
FIXED_SEED = 104729
EXPECTED_FIT_MANIFEST_SHA256 = (
    "c5009ec69dff95bdfad3ba81a29d8281d1ea47ccaae7e5715c39124c61f21c4b"
)
EXPECTED_SOURCE_SHA256 = {
    "train": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "predict": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "cache": "bd6fb773e4bb871a7429299921702c2f1f98c046465b21d83bbb22f2b50aac71",
    "transformer": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
}
SOURCE_PATHS = {
    "train": Path(
        "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
        "repo/scripts/train_unet_transformer.py"
    ),
    "predict": Path(
        "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
        "repo/scripts/predict_unet_transformer.py"
    ),
    "cache": Path("src/celltrack_alpha/v32_preilp_cache.py"),
    "transformer": Path(
        "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
        "repo/src/biohub_tracking/models/simple_node_transformer.py"
    ),
}
FIT_MANIFEST = Path(
    "dag_prompts/2026-08-20-alpha-funnel-v1/runs/manifests/fit-groups.json"
)
FORBIDDEN_TOKENS = (
    "6bba",
    "competition",
    "leaderboard",
    "submission",
    "scorer",
)


class M1ContractError(RuntimeError):
    """Fail-closed M1 source, schema, or firewall violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise M1ContractError(f"{label} must be a regular non-symlink file: {path}")


def _reject_forbidden_text(value: str, label: str) -> None:
    lowered = value.lower()
    hit = next((token for token in FORBIDDEN_TOKENS if token in lowered), None)
    if hit is not None:
        raise M1ContractError(f"{label} contains forbidden token {hit!r}")
    components = {part.lower() for part in Path(value).parts}
    if "test" in components or "tests" in components or "competition_test" in lowered:
        raise M1ContractError(f"{label} enters the forbidden test namespace")


@dataclass(frozen=True)
class FrozenConfig:
    candidate_id: str = CANDIDATE_ID
    seed: int = FIXED_SEED
    frame_count: int = 5
    node_feature_dim: int = 32
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_blocks: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.3
    pair_chunk_size: int = 32
    raw_logit_residual_scale: float = 0.01
    training_scope: str = "lineage_head_only_v32_encoder_frozen"
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 16
    checkpoint_policy: str = "fixed_final_epoch_50_no_selection"
    loss_contract: str = "inherited_v32_softmax_dim0_focal_bce"
    augmentation_policy: str = "none_on_exported_features"

    def validate(self) -> None:
        expected = FrozenConfig()
        if self != expected:
            raise M1ContractError("the M1 configuration is immutable; alternatives are denied")
        if self.hidden_dim % self.attention_heads:
            raise M1ContractError("hidden dimension must be divisible by attention heads")

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


def authenticate_contract(project_root: Path) -> dict[str, Any]:
    """Open only the exact fit manifest and exact source files, then hash them."""
    root = project_root.resolve()
    fit_path = root / FIT_MANIFEST
    _regular_file(fit_path, "fit manifest")
    if sha256_file(fit_path) != EXPECTED_FIT_MANIFEST_SHA256:
        raise M1ContractError("fit manifest SHA-256 drift")
    manifest = json.loads(fit_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups")
    if (
        manifest.get("schema") != "celltrack-alpha.fit-groups.v1"
        or manifest.get("family") != "44b6"
        or manifest.get("group_key") != "movie_identity"
        or manifest.get("count") != 21
        or not isinstance(groups, list)
        or len(groups) != 21
        or len(set(groups)) != 21
        or any(not isinstance(value, str) or not value.startswith("train:44b6_") for value in groups)
    ):
        raise M1ContractError("fit manifest semantic contract drift")

    observed: dict[str, str] = {}
    opened = [str(FIT_MANIFEST)]
    for role, relative in SOURCE_PATHS.items():
        path = root / relative
        _regular_file(path, f"{role} source")
        observed[role] = sha256_file(path)
        opened.append(str(relative))
    expected = dict(EXPECTED_SOURCE_SHA256)
    if observed != expected:
        raise M1ContractError(
            f"source SHA-256 drift: expected={expected!r} observed={observed!r}"
        )
    return {
        "fit_manifest_sha256": EXPECTED_FIT_MANIFEST_SHA256,
        "fit_identities": tuple(groups),
        "source_sha256": observed,
        "opened_paths": tuple(opened),
    }


def assert_selected_identity(identity: str, authenticated: dict[str, Any]) -> None:
    _reject_forbidden_text(identity, "identity")
    if identity not in authenticated["fit_identities"]:
        raise M1ContractError(f"identity is not in the exact selected-train allowlist: {identity}")


def assert_input_path_allowed(path: Path, authenticated: dict[str, Any]) -> str:
    """Validate a prospective raw-input path without opening or hashing it."""
    text = str(path)
    _reject_forbidden_text(text, "prospective input path")
    matches = [identity for identity in authenticated["fit_identities"] if identity.split(":", 1)[1] in text]
    if len(matches) != 1:
        raise M1ContractError("prospective input path must name exactly one selected identity")
    return matches[0]


def _as_array(value: Any, dtype: np.dtype[Any], shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise M1ContractError(f"{label} shape drift: got {array.shape}, tail must be {shape_tail}")
    if not np.isfinite(array).all():
        raise M1ContractError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _write_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def export_five_frame_window(
    *,
    project_root: Path,
    identity: str,
    frames: Sequence[int],
    node_ids: Sequence[Any],
    coords_um: Sequence[Any],
    node_features: Sequence[Any],
    raw_pair_logits: Sequence[Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Export one exact five-frame V32 feature/logit window deterministically.

    All scientific arrays are supplied in memory.  No image, label, graph, scorer,
    holdout, or test path is accepted or discovered by this function.
    """
    authenticated = authenticate_contract(project_root)
    assert_selected_identity(identity, authenticated)
    config = FrozenConfig()
    config.validate()
    frames = tuple(int(value) for value in frames)
    if len(frames) != config.frame_count or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise M1ContractError("export requires exactly five consecutive frames")
    if not (len(node_ids) == len(coords_um) == len(node_features) == 5):
        raise M1ContractError("five node-id, coordinate, and feature arrays are required")
    if len(raw_pair_logits) != 4:
        raise M1ContractError("exactly four adjacent raw-logit matrices are required")

    ids_by_frame: list[np.ndarray] = []
    coords_by_frame: list[np.ndarray] = []
    features_by_frame: list[np.ndarray] = []
    all_ids: list[int] = []
    for slot in range(5):
        ids = np.asarray(node_ids[slot], dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0:
            raise M1ContractError(f"node_ids[{slot}] must be a nonempty vector")
        coords = _as_array(coords_um[slot], np.dtype("float32"), (3,), f"coords_um[{slot}]")
        features = _as_array(
            node_features[slot], np.dtype("float32"), (32,), f"node_features[{slot}]"
        )
        if not (len(ids) == len(coords) == len(features)):
            raise M1ContractError(f"node field row counts disagree at frame slot {slot}")
        ids_by_frame.append(np.ascontiguousarray(ids))
        coords_by_frame.append(coords)
        features_by_frame.append(features)
        all_ids.extend(int(value) for value in ids)
    if len(set(all_ids)) != len(all_ids):
        raise M1ContractError("node_id values must be unique across the five-frame window")

    logits: list[np.ndarray] = []
    for slot, value in enumerate(raw_pair_logits):
        array = np.asarray(value, dtype=np.float32)
        expected_shape = (len(ids_by_frame[slot]), len(ids_by_frame[slot + 1]))
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise M1ContractError(
                f"raw_pair_logits[{slot}] must be finite with shape {expected_shape}, got {array.shape}"
            )
        logits.append(np.ascontiguousarray(array))

    output = output_dir.resolve()
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise M1ContractError("output directory must be absent or empty and non-symlink")
    output.mkdir(parents=True, exist_ok=True)
    node_counts = np.asarray([len(value) for value in ids_by_frame], dtype=np.int64)
    arrays = {
        "node_ids.npy": np.concatenate(ids_by_frame),
        "frame_slots.npy": np.concatenate(
            [np.full(len(value), slot, dtype=np.int8) for slot, value in enumerate(ids_by_frame)]
        ),
        "frames.npy": np.concatenate(
            [np.full(len(value), frames[slot], dtype=np.int64) for slot, value in enumerate(ids_by_frame)]
        ),
        "coords_um.npy": np.concatenate(coords_by_frame, axis=0),
        "node_features_f32.npy": np.concatenate(features_by_frame, axis=0),
        "node_counts.npy": node_counts,
        "raw_pair_logits_f32.npy": np.concatenate([value.reshape(-1) for value in logits]),
        "raw_pair_shapes.npy": np.asarray([value.shape for value in logits], dtype=np.int64),
    }
    file_records = {name: _write_npy(output / name, value) for name, value in arrays.items()}
    manifest = {
        "schema": "celltrack-alpha.m1-v32-five-frame-export.v1",
        "candidate_id": CANDIDATE_ID,
        "identity": identity,
        "selected_train_only": True,
        "frames": list(frames),
        "frame_count": 5,
        "node_feature_dim": 32,
        "node_counts": node_counts.tolist(),
        "raw_pair_shapes": [list(value.shape) for value in logits],
        "raw_values_are_pre_activation": True,
        "fit_manifest_sha256": authenticated["fit_manifest_sha256"],
        "source_sha256": authenticated["source_sha256"],
        "opened_paths": list(authenticated["opened_paths"]),
        "protected_outcomes_opened": False,
        "files": file_records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


if nn is not None:

    class FiveFrameLineageModel(nn.Module):
        """One fixed five-frame context model over V32 node features/raw logits.

        The four V32-sized attention blocks operate over five frame summaries,
        which makes global temporal context bounded in node count.  Adjacent pair
        scores are emitted in chunks and retain the exact raw V32 logit as a
        residual; no candidate threshold or graph decoder is defined here.
        """

        def __init__(self, config: FrozenConfig | None = None):
            super().__init__()
            self.config = config or FrozenConfig()
            self.config.validate()
            torch.manual_seed(self.config.seed)
            self.feature_proj = nn.Linear(self.config.node_feature_dim, self.config.hidden_dim)
            self.frame_embedding = nn.Parameter(
                torch.empty(self.config.frame_count, self.config.hidden_dim)
            )
            nn.init.normal_(self.frame_embedding, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_dim,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.feedforward_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.context = nn.TransformerEncoder(layer, self.config.transformer_blocks)
            pair_dim = self.config.hidden_dim * 3 + 4
            self.pair_mlp = nn.Sequential(
                nn.Linear(pair_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim // 2, 1),
            )

        def forward(
            self,
            node_features: "torch.Tensor",
            coords_um: "torch.Tensor",
            raw_pair_logits: "torch.Tensor",
            node_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            if node_features.ndim != 4 or node_features.shape[1] != 5 or node_features.shape[-1] != 32:
                raise M1ContractError("node_features must have shape (B,5,N,32)")
            if coords_um.shape != (*node_features.shape[:3], 3):
                raise M1ContractError("coords_um must have shape (B,5,N,3)")
            if node_mask.shape != node_features.shape[:3] or node_mask.dtype != torch.bool:
                raise M1ContractError("node_mask must be bool with shape (B,5,N)")
            batch, _, nodes, _ = node_features.shape
            if raw_pair_logits.shape != (batch, 4, nodes, nodes):
                raise M1ContractError("raw_pair_logits must have shape (B,4,N,N)")
            if not node_mask.any(dim=-1).all():
                raise M1ContractError("every frame needs at least one real node")
            if not all(torch.isfinite(value).all() for value in (node_features, coords_um, raw_pair_logits)):
                raise M1ContractError("model inputs must be finite")

            projected = self.feature_proj(node_features)
            weights = node_mask.unsqueeze(-1).to(projected.dtype)
            summaries = (projected * weights).sum(dim=2) / weights.sum(dim=2)
            summaries = summaries + self.frame_embedding.unsqueeze(0)
            context = self.context(summaries)
            outputs = []
            chunk = self.config.pair_chunk_size
            for transition in range(4):
                src = projected[:, transition]
                tgt = projected[:, transition + 1]
                pair_context = (context[:, transition] + context[:, transition + 1]) * 0.5
                transition_chunks = []
                for start in range(0, nodes, chunk):
                    stop = min(start + chunk, nodes)
                    src_chunk = src[:, start:stop]
                    src_coord = coords_um[:, transition, start:stop]
                    width = stop - start
                    src_expanded = src_chunk.unsqueeze(2).expand(-1, -1, nodes, -1)
                    tgt_expanded = tgt.unsqueeze(1).expand(-1, width, -1, -1)
                    ctx_expanded = pair_context[:, None, None, :].expand(-1, width, nodes, -1)
                    relative = (src_coord.unsqueeze(2) - coords_um[:, transition + 1].unsqueeze(1)) / 100.0
                    raw = raw_pair_logits[:, transition, start:stop].unsqueeze(-1)
                    correction = self.pair_mlp(
                        torch.cat([src_expanded, tgt_expanded, ctx_expanded, relative, raw], dim=-1)
                    ).squeeze(-1)
                    transition_chunks.append(
                        raw_pair_logits[:, transition, start:stop]
                        + self.config.raw_logit_residual_scale * correction
                    )
                score = torch.cat(transition_chunks, dim=1)
                valid = node_mask[:, transition].unsqueeze(2) & node_mask[:, transition + 1].unsqueeze(1)
                outputs.append(score.masked_fill(~valid, float("-inf")))
            return torch.stack(outputs, dim=1)


def build_frozen_model() -> Any:
    if torch is None or nn is None:
        raise M1ContractError("torch is required for the frozen model")
    config = FrozenConfig()
    config.validate()
    torch.manual_seed(config.seed)
    model = FiveFrameLineageModel(config)
    model.eval()
    return model


def synthetic_planted_lineage(nodes: int = 3) -> dict[str, Any]:
    if torch is None:
        raise M1ContractError("torch is required for the synthetic control")
    if not 2 <= nodes <= 64:
        raise M1ContractError("synthetic control nodes must be in [2,64]")
    features = torch.zeros(1, 5, nodes, 32, dtype=torch.float32)
    coords = torch.zeros(1, 5, nodes, 3, dtype=torch.float32)
    for frame in range(5):
        for node in range(nodes):
            features[0, frame, node, node % 16] = 1.0
            features[0, frame, node, 16] = frame / 4.0
            features[0, frame, node, 17] = node / max(nodes - 1, 1)
            coords[0, frame, node] = torch.tensor(
                [float(node), float(10 * node + frame), float(20 * node + 2 * frame)]
            )
    raw = torch.full((1, 4, nodes, nodes), -6.0, dtype=torch.float32)
    diagonal = torch.arange(nodes)
    raw[:, :, diagonal, diagonal] = 6.0
    return {
        "node_features": features,
        "coords_um": coords,
        "raw_pair_logits": raw,
        "node_mask": torch.ones(1, 5, nodes, dtype=torch.bool),
        "expected_parent": torch.arange(nodes).expand(1, 4, nodes),
    }


def _tensor_sha256(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def run_control_contract() -> dict[str, Any]:
    """Run one deterministic positive and three predeclared negative controls."""
    if torch is None:
        raise M1ContractError("torch is required for controls")
    model = build_frozen_model()
    fixture = synthetic_planted_lineage()
    with torch.inference_mode():
        positive = model(
            fixture["node_features"], fixture["coords_um"],
            fixture["raw_pair_logits"], fixture["node_mask"]
        )
        replay = model(
            fixture["node_features"], fixture["coords_um"],
            fixture["raw_pair_logits"], fixture["node_mask"]
        )
        temporal_features = fixture["node_features"].roll(1, dims=1)
        temporal = model(
            temporal_features, fixture["coords_um"],
            fixture["raw_pair_logits"], fixture["node_mask"]
        )
        permuted_logits = fixture["raw_pair_logits"].roll(1, dims=2)
        parent_permutation = model(
            fixture["node_features"], fixture["coords_um"],
            permuted_logits, fixture["node_mask"]
        )
        identity_features = fixture["node_features"].clone()
        identity_features[:, 2] = identity_features[:, 2].roll(1, dims=1)
        identity_break = model(
            identity_features, fixture["coords_um"],
            fixture["raw_pair_logits"], fixture["node_mask"]
        )
    expected = fixture["expected_parent"]
    positive_parent = positive.argmax(dim=2)
    permuted_parent = parent_permutation.argmax(dim=2)
    positive_recovery = float((positive_parent == expected).float().mean().item())
    permuted_recovery = float((permuted_parent == expected).float().mean().item())
    positive_sha = _tensor_sha256(positive)
    checks = {
        "deterministic_replay": torch.equal(positive, replay),
        "planted_lineage_positive_recovery_exact": positive_recovery == 1.0,
        "temporal_shift_is_output_distinct": _tensor_sha256(temporal) != positive_sha,
        "parent_permutation_is_output_distinct": _tensor_sha256(parent_permutation) != positive_sha,
        "parent_permutation_breaks_planted_recovery": permuted_recovery < positive_recovery,
        "identity_break_is_output_distinct": _tensor_sha256(identity_break) != positive_sha,
    }
    return {
        "schema": "celltrack-alpha.m1-v32-five-frame-controls.v1",
        "candidate_id": CANDIDATE_ID,
        "config_sha256": FrozenConfig().sha256,
        "seed": FIXED_SEED,
        "scientific_outcome_opened": False,
        "synthetic_only": True,
        "checks": checks,
        "positive_recovery_fraction": positive_recovery,
        "parent_permutation_recovery_fraction": permuted_recovery,
        "output_sha256": {
            "positive": positive_sha,
            "temporal_shift": _tensor_sha256(temporal),
            "parent_permutation": _tensor_sha256(parent_permutation),
            "identity_break": _tensor_sha256(identity_break),
        },
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def environment_receipt() -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__,
    }
    return {**versions, "sha256": canonical_json_sha256(versions)}


def benchmark_synthetic_forward(
    node_ladder: Sequence[int] = (8, 16, 32, 64), repeats: int = 2
) -> dict[str, Any]:
    """Outcome-free CPU ladder for the new head only.

    This deliberately does not claim whole-pipeline or remote eligibility: no
    authenticated per-movie frame/node census or local-to-T4 calibration is
    available to this local-only task.
    """
    if torch is None:
        raise M1ContractError("torch is required for runtime evidence")
    if tuple(node_ladder) != (8, 16, 32, 64) or repeats != 2:
        raise M1ContractError("the runtime ladder is frozen at (8,16,32,64), repeats=2")
    torch.set_num_threads(1)
    model = build_frozen_model()
    rows = []
    with torch.inference_mode():
        for nodes in node_ladder:
            fixture = synthetic_planted_lineage(nodes)
            elapsed = []
            for _ in range(repeats):
                started = time.perf_counter()
                output = model(
                    fixture["node_features"], fixture["coords_um"],
                    fixture["raw_pair_logits"], fixture["node_mask"]
                )
                elapsed.append(time.perf_counter() - started)
            rows.append(
                {
                    "nodes_per_frame": nodes,
                    "pair_scores": 4 * nodes * nodes,
                    "elapsed_seconds": elapsed,
                    "output_sha256": _tensor_sha256(output),
                }
            )
    largest_fraction = max(row["elapsed_seconds"][-1] / row["pair_scores"] for row in rows)
    return {
        "schema": "celltrack-alpha.m1-v32-five-frame-local-runtime.v1",
        "candidate_id": CANDIDATE_ID,
        "device": "cpu",
        "threads": 1,
        "rows": rows,
        "conservative_observed_seconds_per_pair_score": largest_fraction,
        "whole_pipeline_projection_hours": None,
        "remote_projection_hours": None,
        "remote_multiplier_floor_without_calibration": 1.5,
        "status": "PARK_RUNTIME_IDENTITY",
        "reason": (
            "head-only synthetic timings do not authenticate the full selected-train frame/node "
            "census, V32 encoder/export cost, or a matched local-to-T4 calibration"
        ),
    }


def model_identity(model: Any) -> dict[str, Any]:
    if torch is None:
        raise M1ContractError("torch is required for model identity")
    digest = hashlib.sha256()
    parameter_count = 0
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        parameter_count += value.numel()
    return {
        "state_sha256": digest.hexdigest(),
        "parameter_count": parameter_count,
        "config_sha256": FrozenConfig().sha256,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    authenticated = authenticate_contract(root)
    controls = run_control_contract()
    runtime = benchmark_synthetic_forward()
    model = build_frozen_model()
    payload = {
        "candidate_id": CANDIDATE_ID,
        "authenticated": {
            "fit_manifest_sha256": authenticated["fit_manifest_sha256"],
            "fit_identity_count": len(authenticated["fit_identities"]),
            "source_sha256": authenticated["source_sha256"],
            "opened_paths": list(authenticated["opened_paths"]),
        },
        "config": asdict(FrozenConfig()),
        "config_sha256": FrozenConfig().sha256,
        "environment": environment_receipt(),
        "model": model_identity(model),
        "controls": controls,
        "runtime": runtime,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if controls["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

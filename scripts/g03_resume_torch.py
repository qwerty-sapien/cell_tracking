"""CUDA-side implementation of the G-03 Phase 1 A→B resume proof.

There is deliberately no model-science claim here.  The tiny model is a fixed,
synthetic association workload.  Its only purpose is to make the state boundary
large enough to include a real optimizer, scheduler, AMP scaler, CUDA RNG and
sampler cursor, then demonstrate exact continuation in a separately launched
kernel.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from g03_checkpoint_resume import (
    ResumeProofError,
    canonical_json_bytes,
    publish_handoff,
    sha256_bytes,
    sha256_file,
    verify_handoff,
    write_idempotent_json,
)


WORKING = Path("/kaggle/working")
HANDOFF_ID = "g03-phase1-a-to-b-v1"
HANDOFF_DATASET_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/g03-phase1-handoff-v1"
)
FIT_CACHE_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/trackastra-h1-fit105-cache-v1"
)
FIT_CACHE_MANIFEST_SHA256 = (
    "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28"
)
CONFIG = {
    "seed": 20260731,
    "input_width": 256,
    "hidden_width": 512,
    "batch_size": 768,
    "learning_rate": 0.0003,
    "segment_minimum_seconds": 3600.0,
    "checkpoint_interval_seconds": 600.0,
    "amp_enabled": False,
    "dataloader_workers": 0,
    "device": "cuda",
}
SWAP_GROWTH_LIMIT_BYTES = 256 * 1024 * 1024
SWAP_SUSTAIN_SECONDS = 15.0


class TorchResumeProofError(RuntimeError):
    """The CUDA proof could not establish a required invariant."""


class SwapGuard:
    """Abort only on a sustained category-level swapout violation."""

    def __init__(self) -> None:
        self.initial_bytes = self._read_bytes()
        self.current_bytes = self.initial_bytes
        self.exceeded_at: float | None = None
        self.samples = 0

    @staticmethod
    def _read_bytes() -> int:
        try:
            rows = Path("/proc/vmstat").read_text(encoding="ascii").splitlines()
            value = next(
                int(row.split()[1]) for row in rows if row.startswith("pswpout ")
            )
        except (OSError, StopIteration, ValueError, IndexError) as exc:
            raise TorchResumeProofError("cannot read Linux pswpout swap guard") from exc
        return value * os.sysconf("SC_PAGE_SIZE")

    def sample(self) -> None:
        self.current_bytes = self._read_bytes()
        self.samples += 1
        growth = self.current_bytes - self.initial_bytes
        now = time.monotonic()
        if growth < SWAP_GROWTH_LIMIT_BYTES:
            self.exceeded_at = None
            return
        if self.exceeded_at is None:
            self.exceeded_at = now
            return
        if now - self.exceeded_at >= SWAP_SUSTAIN_SECONDS:
            raise TorchResumeProofError(
                "sustained swapout violation: "
                f"{growth} bytes >= {SWAP_GROWTH_LIMIT_BYTES} for "
                f"{now - self.exceeded_at:.1f}s"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "metric": "Linux pswpout bytes from kernel start",
            "initial_bytes": self.initial_bytes,
            "current_bytes": self.current_bytes,
            "growth_bytes": self.current_bytes - self.initial_bytes,
            "limit_bytes": SWAP_GROWTH_LIMIT_BYTES,
            "sustain_seconds": SWAP_SUSTAIN_SECONDS,
            "samples": self.samples,
        }


def _torch():
    # This must happen only after CUBLAS_WORKSPACE_CONFIG is set by the kernel
    # wrapper, before CUDA is initialised.
    import numpy as np
    import torch

    return torch, np


def _require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise TorchResumeProofError("Phase 1 requires a CUDA T4; CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise TorchResumeProofError(f"Phase 1 requires T4, observed {name!r}")


def _set_determinism(torch: Any) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seed_everything(torch: Any, np: Any) -> None:
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])


def _runtime_fingerprint(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device_name": str(properties.name),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "config": CONFIG,
    }


def _source_hashes(root: Path) -> dict[str, str]:
    files = (
        "g03_phase1_a.py",
        "g03_phase1_b.py",
        "g03_resume_torch.py",
        "g03_checkpoint_resume.py",
    )
    observed: dict[str, str] = {}
    for name in files:
        path = root / name
        if not path.is_file():
            raise TorchResumeProofError(f"kernel source is absent: {name}")
        observed[name] = sha256_file(path)
    return observed


def _load_contract(role: str) -> tuple[dict[str, Any], str, dict[str, str]]:
    root = Path(__file__).resolve().parent
    path = root / "PHASE1_CONTRACT.json"
    try:
        contract = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TorchResumeProofError("cannot read Phase 1 kernel contract") from exc
    observed = _source_hashes(root)
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "celltrack-alpha.g03-phase1-contract.v1"
        or contract.get("role") != role
        or contract.get("source_hashes") != observed
        or contract.get("config") != CONFIG
        or contract.get("handoff_id") != HANDOFF_ID
    ):
        raise TorchResumeProofError("Phase 1 kernel contract/source identity drifted")
    return contract, sha256_file(path), observed


def _build_objects(torch: Any) -> tuple[Any, Any, Any, Any]:
    model = torch.nn.Sequential(
        torch.nn.Linear(CONFIG["input_width"], CONFIG["hidden_width"]),
        torch.nn.GELU(),
        torch.nn.Linear(CONFIG["hidden_width"], CONFIG["input_width"]),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100_000)
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["amp_enabled"])
    return model, optimizer, scheduler, scaler


def _capture_rng(torch: Any, np: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(torch: Any, np: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _state_digest(value: Any, torch: Any, np: Any) -> str:
    """Hash Python, NumPy and tensor state without pickle serialization ambiguity."""
    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(array.shape)))
            digest.update(array.tobytes(order="C"))
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: repr(item)):
                add(key)
                add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(value)).encode("ascii"))
            for item in value:
                add(item)
        elif value is None:
            digest.update(b"none\0")
        elif isinstance(value, (str, int, float, bool)):
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(repr(value).encode("ascii"))
            digest.update(b"\0")
        else:
            raise TorchResumeProofError(f"unsupported state value for digest: {type(value)!r}")

    add(value)
    return digest.hexdigest()


def _capture_state(torch: Any, np: Any, model: Any, optimizer: Any, scheduler: Any, scaler: Any, step: int) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _capture_rng(torch, np),
        "sampler": {"next_step": step, "workers": CONFIG["dataloader_workers"]},
        "global_step": step,
    }


def _restore_state(torch: Any, np: Any, state: dict[str, Any], model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> int:
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if state.get("sampler", {}).get("workers") != CONFIG["dataloader_workers"]:
        raise TorchResumeProofError("dataloader worker state drifted")
    step = int(state["global_step"])
    if state.get("sampler", {}).get("next_step") != step:
        raise TorchResumeProofError("sampler cursor drifted")
    _restore_rng(torch, np, state["rng"])
    return step


def _train_step(torch: Any, model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> None:
    features = torch.rand(
        (CONFIG["batch_size"], CONFIG["input_width"]), device="cuda", dtype=torch.float32
    )
    target = torch.roll(features, shifts=1, dims=1)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(features)
    loss = torch.mean((prediction - target) ** 2)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    # The 60-minute segment is real device work, not a one-hour CPU launch queue.
    torch.cuda.synchronize()


def _serialize_torch(torch: Any, value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_torch(torch: Any, value: bytes) -> Any:
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)


def _save_local_checkpoint(torch: Any, output: Path, state: dict[str, Any]) -> None:
    temporary = output.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)


def _fixed_44b6_scope() -> tuple[list[str], str]:
    manifest = FIT_CACHE_ROOT / "cache_dataset_manifest.json"
    if not manifest.is_file() or sha256_file(manifest) != FIT_CACHE_MANIFEST_SHA256:
        raise TorchResumeProofError("pinned fit105 cache manifest is absent or drifted")
    value = json.loads(manifest.read_text(encoding="ascii"))
    identities = value.get("identities")
    if not isinstance(identities, list):
        raise TorchResumeProofError("fit105 manifest has no identity list")
    held = sorted(str(item) for item in identities if str(item).startswith("train:44b6_"))
    if len(held) != 36 or len(set(held)) != 36:
        raise TorchResumeProofError("Phase 1 44b6 held-fit scope drifted")
    return held, sha256_file(manifest)


def _emit_structural_graph_delta(torch: Any, model: Any, checkpoint_digest: str) -> dict[str, Any]:
    """Emit a checkpoint-bound *no-op* 44b6 graph receipt.

    This is intentionally not the frozen objective and it contains no target values.
    It proves the training checkpoint can enter the held-fit graph-emission boundary;
    candidate and control are defined equal, so no result may be interpreted as model
    evidence or used for checkpoint selection.
    """
    identities, manifest_sha = _fixed_44b6_scope()
    model.eval()
    records = []
    with torch.no_grad():
        for identity in identities:
            seed = int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:16], 16)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            features = torch.rand((1, CONFIG["input_width"]), generator=generator, device="cuda")
            score = float(torch.sigmoid(model(features).mean()).cpu())
            # The receipt proves checkpoint-bound score production, but candidate and
            # control deliberately share the graph edge exactly.
            edge = {"source": 0, "target": 1, "weight": score}
            edge_hash = sha256_bytes(canonical_json_bytes(edge))
            records.append({"identity": identity, "control_edge_sha256": edge_hash, "candidate_edge_sha256": edge_hash})
    if any(row["control_edge_sha256"] != row["candidate_edge_sha256"] for row in records):
        raise TorchResumeProofError("no-op graph emission was not byte-identical")
    return {
        "schema": "celltrack-alpha.g03-phase1-44b6-structural-graph-delta.v1",
        "scope": {"partition": "fit105", "family": "44b6", "identity_count": 36, "manifest_sha256": manifest_sha},
        "checkpoint_state_digest": checkpoint_digest,
        "official_frozen_graph_objective_evaluated": False,
        "selection_authorized": False,
        "delta": {"adjusted_edge_jaccard": 0.0, "raw_edge_jaccard": 0.0, "node_recall": 0.0},
        "records": records,
        "verdict": "STRUCTURAL_NOOP_PASS",
    }


def run_a() -> None:
    """Train ≥1h, form an immutable handoff, and calculate the reference endpoint."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    contract, contract_sha, source_hashes = _load_contract("A")
    torch, np = _torch()
    _require_cuda(torch)
    _set_determinism(torch)
    _seed_everything(torch, np)
    model, optimizer, scheduler, scaler = _build_objects(torch)
    output = WORKING / "g03_phase1_a_output"
    output.mkdir(parents=True, exist_ok=True)
    swap_guard = SwapGuard()
    started = time.monotonic()
    last_checkpoint = started
    step = 0
    while time.monotonic() - started < CONFIG["segment_minimum_seconds"]:
        _train_step(torch, model, optimizer, scheduler, scaler)
        step += 1
        swap_guard.sample()
        if time.monotonic() - last_checkpoint >= CONFIG["checkpoint_interval_seconds"]:
            _save_local_checkpoint(torch, output / "stage_a_checkpoint.pt", _capture_state(torch, np, model, optimizer, scheduler, scaler, step))
            last_checkpoint = time.monotonic()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    handoff_state = _capture_state(torch, np, model, optimizer, scheduler, scaler, step)
    handoff_digest = _state_digest(handoff_state, torch, np)

    _seed_everything(torch, np)
    reference_model, reference_optimizer, reference_scheduler, reference_scaler = _build_objects(torch)
    for _ in range(step):
        _train_step(torch, reference_model, reference_optimizer, reference_scheduler, reference_scaler)
        swap_guard.sample()
    reference_half = _capture_state(torch, np, reference_model, reference_optimizer, reference_scheduler, reference_scaler, step)
    if _state_digest(reference_half, torch, np) != handoff_digest:
        raise TorchResumeProofError("same-kernel deterministic reference diverged before handoff")
    for _ in range(step):
        _train_step(torch, reference_model, reference_optimizer, reference_scheduler, reference_scaler)
        swap_guard.sample()
    reference_final = _capture_state(torch, np, reference_model, reference_optimizer, reference_scheduler, reference_scaler, step * 2)
    reference_final_digest = _state_digest(reference_final, torch, np)
    checkpoint = _serialize_torch(torch, handoff_state)
    state = {
        "global_step": step,
        "segment_elapsed_seconds": elapsed,
        "handoff_state_digest": handoff_digest,
        "reference_final_state_digest": reference_final_digest,
        "reference_final_global_step": step * 2,
        "runtime_fingerprint": _runtime_fingerprint(torch),
        "checkpoint_sha256": sha256_bytes(checkpoint),
        "swap_guard": swap_guard.receipt(),
    }
    handoff = publish_handoff(
        output / "handoff",
        manifest={
            "schema": "celltrack-alpha.g03-checkpoint-handoff.v1",
            "handoff_id": HANDOFF_ID,
            "producer_kernel": contract["kernel_id"],
            "code_sha256": sha256_bytes(canonical_json_bytes(source_hashes)),
            "config_sha256": sha256_bytes(canonical_json_bytes(CONFIG)),
            "state": state,
        },
        payloads={"state/checkpoint.pt": checkpoint, "state/contract.json": canonical_json_bytes(contract)},
    )
    receipt = {
        "schema": "celltrack-alpha.g03-phase1-a-receipt.v1",
        "status": "PASS_A_HANDOFF_READY",
        "contract_sha256": contract_sha,
        "handoff": {key: handoff[key] for key in ("handoff_id", "tree_sha256", "manifest_sha256")},
        "state": state,
        "training_minutes": elapsed / 60.0,
        "selection_authorized": False,
    }
    write_idempotent_json(output / "A_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


def run_b() -> None:
    """Verify the imported handoff, resume it in a new kernel, and compare bytes."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    contract, contract_sha, source_hashes = _load_contract("B")
    torch, np = _torch()
    _require_cuda(torch)
    _set_determinism(torch)
    output = WORKING / "g03_phase1_b_output"
    output.mkdir(parents=True, exist_ok=True)
    swap_guard = SwapGuard()
    handoff_root = HANDOFF_DATASET_ROOT / "handoff"
    handoff = verify_handoff(handoff_root)
    manifest = handoff["manifest"]
    state_meta = manifest["state"]
    if (
        handoff["handoff_id"] != HANDOFF_ID
        or manifest["producer_kernel"] != contract["producer_kernel"]
        or manifest["code_sha256"] != sha256_bytes(canonical_json_bytes(source_hashes))
        or manifest["config_sha256"] != sha256_bytes(canonical_json_bytes(CONFIG))
        or state_meta.get("runtime_fingerprint") != _runtime_fingerprint(torch)
    ):
        raise TorchResumeProofError("imported handoff identity or runtime fingerprint drifted")
    checkpoint = (handoff_root / "state/checkpoint.pt").read_bytes()
    if sha256_bytes(checkpoint) != state_meta.get("checkpoint_sha256"):
        raise TorchResumeProofError("imported torch checkpoint hash drifted")
    restored = _load_torch(torch, checkpoint)
    model, optimizer, scheduler, scaler = _build_objects(torch)
    step = _restore_state(torch, np, restored, model, optimizer, scheduler, scaler)
    if _state_digest(restored, torch, np) != state_meta.get("handoff_state_digest"):
        raise TorchResumeProofError("torch checkpoint state digest drifted before resume")
    for _ in range(step):
        _train_step(torch, model, optimizer, scheduler, scaler)
        swap_guard.sample()
    final_state = _capture_state(torch, np, model, optimizer, scheduler, scaler, step * 2)
    final_digest = _state_digest(final_state, torch, np)
    if (
        final_digest != state_meta.get("reference_final_state_digest")
        or final_state["global_step"] != state_meta.get("reference_final_global_step")
    ):
        raise TorchResumeProofError("new-kernel resumed state is not bit-identical to reference")
    graph = _emit_structural_graph_delta(torch, model, final_digest)
    write_idempotent_json(output / "GRAPH_EMISSION.json", graph)
    receipt = {
        "schema": "celltrack-alpha.g03-phase1-b-receipt.v1",
        "status": "PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED",
        "contract_sha256": contract_sha,
        "handoff": {key: handoff[key] for key in ("handoff_id", "tree_sha256", "manifest_sha256")},
        "resumed_global_step": step,
        "final_state_digest": final_digest,
        "graph_emission_sha256": sha256_file(output / "GRAPH_EMISSION.json"),
        "swap_guard": swap_guard.receipt(),
        "selection_authorized": False,
    }
    write_idempotent_json(output / "B_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


__all__ = ["TorchResumeProofError", "run_a", "run_b"]

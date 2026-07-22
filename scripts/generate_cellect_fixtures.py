#!/usr/bin/env python3
"""DEV-ONLY: generate frozen CELLECT numeric fixtures from the reference tree.

This is the sole project script permitted to load the GPL reference source. It
is not part of the shipped ``src`` package or Kaggle package. Generated ``npz``
values and hashes are numeric verification data, not executable reference code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "external/CELLECT"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/cellect"
BACKBONE_FILENAME = "U-ext+-x3rd-149.0-4.6540.pth"
INTRA_FILENAME = "EN+-x3rd-149.0-4.6540.pth"
INTER_FILENAME = "EX+-x3rd-149.0-4.6540.pth"
SEED = 20251020
OPTIMIZER_SEED = 41116016
E2E_SEED = 28086
E2E_STEPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_module(path: Path, name: str, *, neutralize_cuda: bool = False) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    if not neutralize_cuda:
        spec.loader.exec_module(module)
        return module
    # ``recoloss.py`` imports pandas for unrelated evaluation helpers.  The
    # distance/loss fixtures do not execute those helpers, so avoid adding a
    # heavyweight dev dependency merely to import the reference definitions.
    created_pandas_stub = "pandas" not in sys.modules
    if created_pandas_stub:
        sys.modules["pandas"] = ModuleType("pandas")
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor  # type: ignore[method-assign]
    try:
        spec.loader.exec_module(module)
    finally:
        torch.Tensor.cuda = original_cuda  # type: ignore[method-assign]
        if created_pandas_stub:
            sys.modules.pop("pandas", None)
    return module


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"reference checkpoint is not a state dictionary: {path}")
    return value


def preprocess(value: torch.Tensor) -> torch.Tensor:
    minimum = value[value > 0].min()
    result = value.clone()
    result[result < minimum] = minimum
    return torch.log1p(result + 1900.0)


def tensor_stream_hash(values: Iterable[torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"<none>")
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_norms(model: torch.nn.Module) -> np.ndarray:
    return np.asarray(
        [float(parameter.detach().double().norm().item()) for parameter in model.parameters()],
        dtype=np.float64,
    )


def fixture_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    coefficients = (0.5, 0.25, 0.125, 0.75, 1.0)
    return sum(
        coefficient * output.square().mean()
        for coefficient, output in zip(coefficients, outputs, strict=True)
    )


def optimizer_fixture(
    module: ModuleType,
    checkpoint: Path,
    raw_input: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    model = module.UNet3D(2, 6)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(OPTIMIZER_SEED)
    loss = fixture_loss(tuple(model(preprocess(raw_input))))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    arrays = {
        "optimizer_input": raw_input.detach().cpu().numpy(),
        "optimizer_loss": np.asarray(loss.detach().item(), dtype=np.float64),
        "optimizer_gradient_norms": np.asarray(
            [
                float("nan") if gradient is None else float(gradient.detach().double().norm().item())
                for gradient in gradients
            ],
            dtype=np.float64,
        ),
    }
    hashes = {"optimizer_gradient_stream_sha256": tensor_stream_hash(gradients)}
    optimizer.step()
    arrays["optimizer_post_parameter_norms"] = parameter_norms(model)
    hashes["optimizer_post_parameter_stream_sha256"] = tensor_stream_hash(
        model.parameters()
    )
    return arrays, hashes


def e2e_fixture(
    module: ModuleType,
    checkpoint: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    model = module.UNet3D(2, 6)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False)
    input_generator = torch.Generator(device="cpu").manual_seed(E2E_SEED + 1)
    torch.manual_seed(E2E_SEED)
    curve = []
    for _step in range(E2E_STEPS):
        raw_input = torch.rand((1, 2, 16, 16, 2), generator=input_generator) * 2000.0
        optimizer.zero_grad(set_to_none=True)
        loss = fixture_loss(tuple(model(preprocess(raw_input))))
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("reference E2E loss became non-finite")
        loss.backward()
        optimizer.step()
        curve.append(float(loss.detach().item()))
    arrays = {
        "e2e_loss_curve": np.asarray(curve, dtype=np.float64),
        "e2e_final_parameter_norms": parameter_norms(model),
    }
    hashes = {
        "e2e_final_parameter_stream_sha256": tensor_stream_hash(model.parameters())
    }
    return arrays, hashes


def generate(reference_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = reference_root / "unetext3Dn_con7.py"
    losses = reference_root / "recoloss.py"
    backbone_path = reference_root / "model" / BACKBONE_FILENAME
    intra_path = reference_root / "model" / INTRA_FILENAME
    inter_path = reference_root / "model" / INTER_FILENAME
    for path in (source, losses, backbone_path, intra_path, inter_path):
        if not path.is_file():
            raise RuntimeError(f"missing reference fixture input: {path}")
    module = load_module(source, "cellect_fixture_reference_model")
    loss_module = load_module(
        losses,
        "cellect_fixture_reference_losses",
        neutralize_cuda=True,
    )
    arrays: dict[str, np.ndarray] = {}

    preprocess_input = torch.tensor(
        [[[[[0.0, -4.0, 2.0], [4.0, 8.0, 16.0]]]]],
        dtype=torch.float32,
    ).expand(1, 2, 1, 2, 3).clone()
    arrays["preprocess_input"] = preprocess_input.numpy()
    arrays["preprocess_expected"] = preprocess(preprocess_input).numpy()

    distance_x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, -1.0, 0.5]],
        dtype=torch.float32,
    )
    distance_y = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 1.0, -0.5]],
        dtype=torch.float32,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distance_expected = loss_module.euclidean_dist(distance_x, distance_y)
    arrays["distance_x"] = distance_x.numpy()
    arrays["distance_y"] = distance_y.numpy()
    arrays["distance_expected"] = distance_expected.numpy()

    mining_embedding = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    mining_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    mining_distance = loss_module.euclidean_dist(mining_embedding, mining_embedding)
    mined = loss_module.hard_example_mining(
        mining_distance,
        mining_labels,
        return_inds=True,
    )
    triplet = loss_module.TripletLoss(0.3)(mining_embedding, mining_labels)
    arrays["mining_embedding"] = mining_embedding.numpy()
    arrays["mining_labels"] = mining_labels.numpy()
    arrays["mining_distance"] = mining_distance.numpy()
    arrays["mining_hardest_positive"] = mined[0].numpy()
    arrays["mining_hardest_negative"] = mined[1].numpy()
    arrays["mining_positive_indices"] = mined[2].numpy()
    arrays["mining_negative_indices"] = mined[3].numpy()
    arrays["triplet_loss"] = np.asarray(triplet[0].item(), dtype=np.float64)
    arrays["triplet_hardest_positive"] = triplet[1].numpy()
    arrays["triplet_hardest_negative"] = triplet[2].numpy()

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    mlp_anchor = torch.rand((4, 1, 64), generator=generator) + 0.1
    mlp_candidates = torch.rand((4, 5, 64), generator=generator) + 0.1
    mlp_spatial = torch.rand((4, 5, 3), generator=generator)
    mlp_size = torch.rand((4, 5, 6), generator=generator)
    mlp_division = torch.rand((4, 5, 6), generator=generator)
    arrays.update(
        {
            "mlp_anchor": mlp_anchor.numpy(),
            "mlp_candidates": mlp_candidates.numpy(),
            "mlp_spatial": mlp_spatial.numpy(),
            "mlp_size": mlp_size.numpy(),
            "mlp_division": mlp_division.numpy(),
        }
    )
    for label, path, classes in (
        ("intra", intra_path, 6),
        ("inter", inter_path, 8),
    ):
        matcher = module.EXNet(64, classes)
        matcher.load_state_dict(load_state(path), strict=True)
        matcher.eval()
        with torch.no_grad():
            output = matcher(
                mlp_anchor,
                mlp_candidates,
                mlp_spatial,
                mlp_size,
                mlp_division,
            )
        arrays[f"{label}_mlp_output"] = output.numpy()

    backbone = module.UNet3D(2, 6)
    backbone.load_state_dict(load_state(backbone_path), strict=True)
    backbone.eval()
    forward_input = torch.rand((1, 2, 16, 16, 2), generator=generator) * 3000.0
    forward_preprocessed = preprocess(forward_input)
    with torch.no_grad():
        outputs = tuple(backbone(forward_preprocessed))
    arrays["forward_input"] = forward_input.numpy()
    arrays["forward_preprocessed"] = forward_preprocessed.numpy()
    for index, output in enumerate(outputs):
        arrays[f"forward_output_{index}"] = output.numpy()

    optimizer_input = torch.rand((1, 2, 16, 16, 2), generator=generator) * 2500.0
    optimizer_arrays, optimizer_hashes = optimizer_fixture(
        module,
        backbone_path,
        optimizer_input,
    )
    arrays.update(optimizer_arrays)
    e2e_arrays, e2e_hashes = e2e_fixture(module, backbone_path)
    arrays.update(e2e_arrays)

    head = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema": "celltrack-alpha.cellect-frozen-fixtures.v1",
        "dev_only_reference_import": True,
        "shipped_source_dependency": False,
        "seed": SEED,
        "optimizer_seed": OPTIMIZER_SEED,
        "e2e_seed": E2E_SEED,
        "e2e_steps": E2E_STEPS,
        "input_layout": "B,C,Y,X,Z",
        "reference": {
            "commit": head,
            "model_source_sha256": sha256_file(source),
            "loss_source_sha256": sha256_file(losses),
            "backbone_checkpoint_sha256": sha256_file(backbone_path),
            "intra_checkpoint_sha256": sha256_file(intra_path),
            "inter_checkpoint_sha256": sha256_file(inter_path),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "deprecated_addmm_finding": {
            "warning_observed": any("deprecated" in str(item.message).lower() for item in caught),
            "observed_semantics": "beta=1, alpha=-2; xx+yy-2*x@y.T",
            "reimplementation": "keyword torch.addmm followed by clamp(1e-12).sqrt()",
        },
        "fixtures": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": sha256_array(value),
            }
            for key, value in sorted(arrays.items())
        },
        "optimizer": optimizer_hashes,
        "e2e": e2e_hashes,
    }
    return arrays, manifest


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    fixture_path = output_dir / "components.npz"
    manifest_path = output_dir / "manifest.json"
    if not args.force and (fixture_path.exists() or manifest_path.exists()):
        raise RuntimeError(f"refusing to overwrite frozen CELLECT fixtures in {output_dir}")
    arrays, manifest = generate(args.reference_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture_path, **arrays)
    manifest["npz_sha256"] = sha256_file(fixture_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"fixture": str(fixture_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze exactly 1,000 Optuna-QMC CRW parameter tuples before labels are opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PREREG = HERE / "PREREGISTRATION.json"
OUTPUT = HERE / "PARAMETER_BANK.json"


class FreezeError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tuple_id(stratum: str, params: dict[str, float]) -> str:
    return hashlib.sha256(canonical({"stratum": stratum, "params": params})).hexdigest()


def generate() -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text(encoding="ascii"))
    if prereg.get("status") != "FROZEN_BEFORE_NEW_CARRIER_SUPERVISION_MATERIALIZATION":
        raise FreezeError("preregistration is not frozen")
    if sha256(ROOT / prereg["bindings"]["lmt"]["path"]) != prereg["bindings"]["lmt"]["sha256"]:
        raise FreezeError("LMT binding drifted")
    import optuna

    if optuna.__version__ != "4.9.0":
        raise FreezeError(f"Optuna version drifted: {optuna.__version__}")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    rows: list[dict[str, Any]] = []
    for stratum in prereg["parameter_search"]["strata"]:
        sampler = optuna.samplers.QMCSampler(
            qmc_type="sobol", scramble=True, seed=int(stratum["seed"])
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for _ in range(int(stratum["count"])):
            trial = study.ask()
            params = {
                name: float(
                    trial.suggest_float(
                        name,
                        float(spec["low"]),
                        float(spec["high"]),
                        log=bool(spec["log"]),
                    )
                )
                for name, spec in stratum["ranges"].items()
            }
            study.tell(trial, 0.0)
            rows.append(
                {
                    "trial_id": len(rows),
                    "stratum": str(stratum["name"]),
                    "tuple_id": tuple_id(str(stratum["name"]), params),
                    "params": params,
                }
            )
    expected = int(prereg["parameter_search"]["total_unique_combinations"])
    if len(rows) != expected or len({row["tuple_id"] for row in rows}) != expected:
        raise FreezeError("parameter tuple count or uniqueness drifted")
    payload = {
        "schema": "celltrack-alpha.crw-hpo1000-parameter-bank.v1",
        "status": "FROZEN_BEFORE_CARRIER_SUPERVISION",
        "preregistration_sha256": sha256(PREREG),
        "optuna_version": optuna.__version__,
        "sampler": "QMCSampler(qmc_type='sobol', scramble=True)",
        "total_unique_combinations": expected,
        "stratum_counts": {
            stratum["name"]: int(stratum["count"])
            for stratum in prereg["parameter_search"]["strata"]
        },
        "trials": rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        parser.error("--output must be the canonical parameter-bank path")
    if OUTPUT.exists():
        raise FreezeError(f"refusing to overwrite immutable output: {OUTPUT}")
    OUTPUT.write_bytes(canonical(generate()))
    print(json.dumps({"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

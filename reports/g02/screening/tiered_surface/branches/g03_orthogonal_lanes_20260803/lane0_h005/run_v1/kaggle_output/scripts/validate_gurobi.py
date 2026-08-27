#!/usr/bin/env python3
"""Validate that Gurobi credentials can solve a tiny MILP without logging secrets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/gurobi_validation.json"

SECRET_KEYS = {
    "GUROBI_API_KEY",
    "GRB_LICENSE_FILE",
    "WLSACCESSID",
    "WLSSECRET",
    "LICENSEID",
    "GUROBI_WLSACCESSID",
    "GUROBI_WLSSECRET",
    "GUROBI_LICENSEID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--install-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install missing Python packages with pip before importing them.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def load_env(path: Path) -> dict[str, Any]:
    loaded = False
    method = "none"
    if not path.exists():
        return {"path": str(path), "exists": False, "loaded": False, "method": method}

    try:
        from dotenv import load_dotenv

        loaded = bool(load_dotenv(path, override=False))
        method = "python-dotenv"
    except Exception:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True
        method = "fallback-parser"
    return {"path": str(path), "exists": True, "loaded": loaded, "method": method}


def ensure_import(module: str, package: str, install_missing: bool, timeout: int) -> tuple[Any | None, dict[str, Any]]:
    result: dict[str, Any] = {"module": module, "package": package, "installed": False, "success": False}
    try:
        imported = importlib.import_module(module)
        result["success"] = True
        return imported, result
    except ModuleNotFoundError:
        if not install_missing:
            result["failure_kind"] = "module_not_found"
            return None, result

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
        check=False,
    )
    result["installed"] = proc.returncode == 0
    if proc.returncode != 0:
        result["failure_kind"] = "pip_install_failed"
        result["pip_returncode"] = proc.returncode
        return None, result
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        result["failure_kind"] = type(exc).__name__
        return None, result
    result["success"] = True
    return imported, result


def credential_presence() -> dict[str, bool]:
    presence = {key: bool(os.environ.get(key)) for key in sorted(SECRET_KEYS)}
    license_file = os.environ.get("GRB_LICENSE_FILE")
    if license_file:
        presence["GRB_LICENSE_FILE_EXISTS"] = Path(license_file).expanduser().exists()
    return presence


def _normalized_mapping(raw: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
        mapping[norm] = str(value)
    return mapping


def parse_gurobi_api_key() -> dict[str, str]:
    """Best-effort parse for API-key values that encode WLS credentials."""

    raw = os.environ.get("GUROBI_API_KEY", "").strip()
    if not raw:
        return {}
    parsed: dict[str, Any] | None = None
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    if parsed is None and "=" in raw:
        parsed = {}
        for part in re.split(r"[;,\n]", raw):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key.strip()] = value.strip()
    if not isinstance(parsed, dict):
        return {}
    norm = _normalized_mapping(parsed)
    out: dict[str, str] = {}
    access = norm.get("wlsaccessid") or norm.get("accessid")
    secret = norm.get("wlssecret") or norm.get("secret")
    license_id = norm.get("licenseid")
    if access:
        out["WLSAccessID"] = access
    if secret:
        out["WLSSecret"] = secret
    if license_id:
        out["LicenseID"] = license_id
    return out


def gurobi_params_from_env() -> tuple[dict[str, str], str]:
    direct = {
        "WLSAccessID": os.environ.get("WLSACCESSID") or os.environ.get("GUROBI_WLSACCESSID"),
        "WLSSecret": os.environ.get("WLSSECRET") or os.environ.get("GUROBI_WLSSECRET"),
        "LicenseID": os.environ.get("LICENSEID") or os.environ.get("GUROBI_LICENSEID"),
    }
    if all(direct.values()):
        return {k: str(v) for k, v in direct.items() if v is not None}, "wls_env"
    parsed = parse_gurobi_api_key()
    if all(k in parsed for k in ("WLSAccessID", "WLSSecret", "LicenseID")):
        return parsed, "gurobi_api_key_encoded_wls"
    return {}, "default_license_discovery"


def tiny_milp(gp: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False}
    try:
        result["version"] = ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        result["version"] = None

    params, source = gurobi_params_from_env()
    result["credential_source"] = source
    result["explicit_wls_params"] = bool(params)
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        for key, value in params.items():
            if key == "LicenseID":
                try:
                    env.setParam(key, int(value))
                except ValueError:
                    env.setParam(key, value)
            else:
                env.setParam(key, value)
        env.start()
        model = gp.Model("celltrack_gurobi_smoke", env=env)
        model.setParam("OutputFlag", 0)
        x = model.addVar(vtype=gp.GRB.BINARY, name="x")
        y = model.addVar(vtype=gp.GRB.BINARY, name="y")
        model.addConstr(x + y <= 1, name="choose_one")
        model.setObjective(x + 2 * y, gp.GRB.MAXIMIZE)
        model.optimize()
        result["status_code"] = int(model.Status)
        result["objective"] = float(model.ObjVal) if model.Status == gp.GRB.OPTIMAL else None
        result["success"] = model.Status == gp.GRB.OPTIMAL and abs(float(model.ObjVal) - 2.0) < 1e-9
        model.dispose()
        env.dispose()
    except Exception as exc:
        result["failure_kind"] = type(exc).__name__
        result["success"] = False
    return result


def find_ultrack_cli() -> str | None:
    exe = shutil.which("ultrack")
    if exe:
        return exe
    candidate = Path(sys.executable).resolve().parent / "ultrack"
    if candidate.exists():
        return str(candidate)
    return None


def run_ultrack_check(install_missing: bool, timeout: int) -> dict[str, Any]:
    exe = find_ultrack_cli()
    result: dict[str, Any] = {"installed": bool(exe), "pip_installed": False, "success": False}
    if not exe and install_missing:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ultrack"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
        result["pip_installed"] = proc.returncode == 0
        result["pip_returncode"] = proc.returncode
        exe = find_ultrack_cli()
        result["installed"] = bool(exe)
    if not exe:
        result["skipped_reason"] = "ultrack_cli_not_found"
        return result
    proc = subprocess.run(
        [exe, "check_gurobi"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
        check=False,
    )
    result["returncode"] = proc.returncode
    result["success"] = proc.returncode == 0
    return result


def credential_hint(report: dict[str, Any]) -> str | None:
    presence = report["environment"]["credential_presence"]
    milp = report.get("gurobi_milp", {})
    if milp.get("success"):
        return None
    if presence.get("GUROBI_API_KEY") and not milp.get("explicit_wls_params"):
        return (
            "GUROBI_API_KEY alone was not recognized as a standard gurobipy license. "
            "Provide WLSAccessID/WLSSecret/LicenseID values, WLSACCESSID/WLSSECRET/LICENSEID "
            "environment variables, or a valid GRB_LICENSE_FILE/gurobi.lic."
        )
    if not any(presence.values()):
        return (
            "No Gurobi credential variables were detected. Provide WLSACCESSID, WLSSECRET, "
            "and LICENSEID, or set GRB_LICENSE_FILE to a valid gurobi.lic."
        )
    return "Gurobi credentials were detected, but the tiny MILP did not solve; validate license scope and format."


def main() -> int:
    args = parse_args()
    env_info = load_env(args.env_file)
    gp, import_result = ensure_import("gurobipy", "gurobipy", args.install_missing, args.timeout)
    milp_result = tiny_milp(gp) if gp is not None else {"success": False, "failure_kind": "gurobipy_unavailable"}
    ultrack_result = run_ultrack_check(args.install_missing, args.timeout)

    status = "fail"
    if milp_result.get("success") and ultrack_result.get("success"):
        status = "pass"
    elif milp_result.get("success"):
        status = "partial"

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "python": sys.version.split()[0],
        "environment": {
            **env_info,
            "credential_presence": credential_presence(),
        },
        "gurobipy_import": import_result,
        "gurobi_milp": milp_result,
        "ultrack_check_gurobi": ultrack_result,
    }
    hint = credential_hint(report)
    if hint:
        report["credential_hint"] = hint

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"status={status}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

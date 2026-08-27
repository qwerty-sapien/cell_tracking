#!/usr/bin/env python3
"""Bounded, receipted, serially-locked Kaggle operations for the owner's cmux session.

Doctrine (see dag_prompts/2026-07-31-gpu-window/codex/C0_CMUX_OPS_MODULE.md):

    Never type ad-hoc Kaggle commands into the authenticated cmux shell. Invoke this
    module, whose COMPLETE external command surface is enumerated by ``command_plan()``
    and which writes a receipt for every invocation.

An enumerated plan is reviewable before it runs and reproducible after. Free-form typing
into an authenticated session is how a one-run authorization became two byte-identical
kernel versions and a 0.74 GPU-h debit.

Campaign-agnostic on purpose: G-02 is archived, G-03 is live, and both need this. It
complements ``reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py``
(static/launch kernel validation) rather than duplicating it -- that module validates what
you are about to push; this one binds its exact PASS receipt to the bytes being pushed.

Hard rules enforced in code, not in comments:
  * whitelist-only execution, argv lists, never ``shell=True``
  * exclusive serial lock for every mutating op
  * ``kernel_push`` refuses without a fresh eligible prelaunch receipt whose metadata and
    code hashes still match the package directory
  * ``kernel_push`` refuses while a version is RUNNING/QUEUED
  * ``competition_submit`` refuses without an ``eligible: true`` predicate receipt
  * credential-scrubbed receipts for every invocation
  * NO retry logic, ever
  * autonomous external/resource limits are loaded from the repository ``LMT.md``
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import PurePath
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "BudgetLedgerError",
    "CmuxOpsError",
    "DatasetMetadataNotSatisfied",
    "InsufficientGpuQuota",
    "KernelPredicateNotSatisfied",
    "LockHeld",
    "ProjectionApprovalRequired",
    "PredicateNotSatisfied",
    "PushRefused",
    "command_plan",
    "run",
    "serial_lock",
    "bracketed_launch",
    "authenticated_kaggle_environment",
    "reconcile_gpu_reservation",
    "semantic_mutation_error",
    "scrub",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical_shared_repo_root(local_root: Path) -> Path:
    """Resolve the main checkout so linked worktrees share one lock and ledger."""
    override = os.environ.get("CELLTRACK_SHARED_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    marker = local_root / ".git"
    if not marker.is_file():
        return local_root
    try:
        line = marker.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir:"):
            return local_root
        git_dir = Path(line.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (local_root / git_dir).resolve()
        if git_dir.parent.name == "worktrees":
            return git_dir.parent.parent.parent.resolve()
    except OSError:
        pass
    return local_root


SHARED_REPO_ROOT = _canonical_shared_repo_root(REPO_ROOT)
DEFAULT_LOCK = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/.cmux_serial.lock"
)
DEFAULT_GPU_LEDGER = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/KAGGLE_SHARED_GPU_LEDGER.json"
)
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
DEFAULT_SUBMISSION_CLAIM_DIR = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/.submission_claims/direct"
)
DEFAULT_LMT = SHARED_REPO_ROOT / "LMT.md"
LMT_POLICY_BEGIN = "<!-- LMT_POLICY_JSON_BEGIN -->"
LMT_POLICY_END = "<!-- LMT_POLICY_JSON_END -->"
KAGGLE_PRELAUNCH_GUARD = (
    REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py"
)
# Visual/push integrity is anchored to the physical checkout containing this
# script.  Unlike locks and quota ledgers, it must never follow an environment
# override supplied by the caller.
VISUAL_GUARD_PROJECT_ROOT = REPO_ROOT.resolve()
CANONICAL_KAGGLE_PRELAUNCH_REGISTRY = (
    VISUAL_GUARD_PROJECT_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
)

_KERNEL_STATUS = re.compile(
    r"(?<![A-Za-z0-9_])KernelWorkerStatus\."
    r"(RUNNING|QUEUED|COMPLETE|ERROR|CANCELLED)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
M7R6_NATIVE_T4_CERT_KERNEL_ID = (
    "onebitbrain/m7r6-native-t4-whole-pipeline-cert-v1"
)
M7R8_NATIVE_T4_CERT_KERNEL_ID = (
    "onebitbrain/m7r8-native-t4-whole-pipeline-cert-v1"
)
M7R13_PROVIDER_CERT_KERNEL_ID = (
    "onebitbrain/m7r13-full21-v32-m6-provider-cert-v1"
)
M7R14_PROVIDER_CERT_KERNEL_ID = (
    "onebitbrain/m7r14-full21-v32-m6-provider-cert-v1"
)
M7R15_ACTION_SUPPLY_KERNEL_ID = (
    "onebitbrain/m7r15-full21-action-supply-census-v1"
)
M10R3_PROVIDER_CERT_KERNEL_ID = (
    "onebitbrain/m10r3-aniso3d-dualslot-provider-cert-v1"
)
M10R5_PROVIDER_CERT_KERNEL_ID = (
    "onebitbrain/m10r5-aniso3d-dualslot-provider-cert-v1"
)
D10_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID = (
    "onebitbrain/f2-d-v10-native-cert-current-v1"
)
D11_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID = (
    "onebitbrain/f2-d-v11-native-cert-current-v1"
)
D28_CLOSED_SCHEMA_NATIVE_CERT_KERNEL_ID = (
    "onebitbrain/f2-d28-native-cert-fresh-reconciliation-time-v1"
)
D19_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID = (
    "onebitbrain/f2-d-v19-native-cert-current-v1"
)
D16_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID = (
    "onebitbrain/f2-d-v16-native-cert-current-v1"
)
M12R8_PROVIDER_CERT_KERNEL_ID = (
    "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1"
)
M8R13_NATIVE_T4_EXPORTER_CERT_KERNEL_ID = (
    "onebitbrain/m8r13-native-t4-exporter-cert-v1"
)
M8R14_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID = (
    "onebitbrain/m8r14-postilp-target-feature-cert-v1"
)
M8R15_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID = (
    "onebitbrain/m8r15-postilp-target-feature-cert-v1"
)
M8R16_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID = (
    "onebitbrain/m8r16-postilp-target-feature-cert-v1"
)
M8R17_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID = (
    "onebitbrain/m8r17-postilp-target-feature-cert-v1"
)
M8R20_FULL21_TARGET_FEATURE_CERT_KERNEL_ID = (
    "onebitbrain/m8r20-full21-target-feature-export-v1"
)
REMOTE_NATIVE_CERTIFICATION_EXCEPTIONS = {
    M7R6_NATIVE_T4_CERT_KERNEL_ID: {
        "exception_id": "m7r6-native-t4-whole-pipeline-cert-v1",
        "record_sha256": "849ec6df0b68b7ba3ba84edb45b971341a916aab8f304d7027f7d1dc3486296a",
    },
    M7R8_NATIVE_T4_CERT_KERNEL_ID: {
        "exception_id": "m7r8-native-t4-whole-pipeline-cert-v1",
        "record_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M7R13_PROVIDER_CERT_KERNEL_ID: {
        "exception_id": "m7r13-full21-provider-cert-v1",
        "record_sha256": "2c34727c4b871fb10111e356e405b48f737b12833aee1027c11c2bb85993613d",
        "identities": [],
    },
    M7R14_PROVIDER_CERT_KERNEL_ID: {
        "exception_id": "m7r14-full21-raw-metadata-transport-v1",
        "record_sha256": "6109fc26d11dc7e89c59a46cc5d11a1e6c182d1f6799e5fbba4ee8600a7032c9",
        "identities": [],
    },
    M7R15_ACTION_SUPPLY_KERNEL_ID: {
        "exception_id": "m7r15-full21-one-look-action-supply-census-v1",
        "record_sha256": "fd6ce8cbd3167fc4a8e7a0f1de14fa6e675b5c689fc0b5868f949415eb0eac81",
        "identities": [],
    },
    M10R3_PROVIDER_CERT_KERNEL_ID: {
        "exception_id": "m10r3-self-orchestrating-provider-cert-v1",
        "record_sha256": "8cd1b1ab565a680906269c50f1edaa79d2ee170e5e53ff53a47f16ac4d99ca8c",
        "identities": [],
    },
    M10R5_PROVIDER_CERT_KERNEL_ID: {
        "exception_id": "m10r5-gather-native-t4-provider-cert-v1",
        "record_sha256": "152999fc4e80a1bedfc31bc091e9bf2f021f3e5d2d64c87241331f49976845e0",
        "identities": [],
    },
    D10_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID: {
        "exception_id": "d10-current-policy-native-cert-v1",
        "record_sha256": "a734580d2061a9169b81b509fe2d76017df4d0174638c93ccb9eba7c634be64b",
        "identities": [
            "train:44b6_cf2536e8",
            "train:44b6_3bb3690f",
            "train:44b6_aaf8b0ea",
            "train:44b6_c8e2a523",
            "train:44b6_87bba6c4",
            "train:44b6_996155de",
        ],
    },
    D11_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID: {
        "exception_id": "d11-current-policy-native-cert-v1",
        "record_sha256": "af5d30e9a8b03a8fe939ee83fc77c6514e0720cbb742cc8d2e8bb1c35efb0b1b",
        "identities": [
            "train:44b6_cf2536e8",
            "train:44b6_3bb3690f",
            "train:44b6_aaf8b0ea",
            "train:44b6_c8e2a523",
            "train:44b6_87bba6c4",
            "train:44b6_996155de",
        ],
    },
    D28_CLOSED_SCHEMA_NATIVE_CERT_KERNEL_ID: {
        "exception_id": "d28-fresh-reconciliation-time-central-release-v1",
        "record_sha256": "35ff73e0f0bc54ecd195b9c87f18dec27da3af6186a4a662eabbf94a369ba35f",
        "identities": [],
    },
    D19_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID: {
        "exception_id": "d19-current-policy-native-cert-v1",
        "record_sha256": "cf8720162a8ceda6d43c2788b532ca2d5fbaa3800eff0f1d53ec582c615a836f",
        "identities": ['train:44b6_cf2536e8', 'train:44b6_3bb3690f', 'train:44b6_aaf8b0ea', 'train:44b6_c8e2a523', 'train:44b6_87bba6c4', 'train:44b6_996155de'],
    },
    D16_CURRENT_POLICY_NATIVE_CERT_KERNEL_ID: {
        "exception_id": "d16-current-policy-native-cert-v1",
        "record_sha256": "809fb757fc871ed1772358a7b4d601119529cc493073ef30e253a3e01fa65059",
        "identities": ['train:44b6_cf2536e8', 'train:44b6_3bb3690f', 'train:44b6_aaf8b0ea', 'train:44b6_c8e2a523', 'train:44b6_87bba6c4', 'train:44b6_996155de'],
    },
    M12R8_PROVIDER_CERT_KERNEL_ID: {
        "exception_id": "m12r8-synthetic-control-visual-preflight-v1",
        "record_sha256": "2bbaba0877b4abb0e785f94b2318bc097ababdcc7ec6cb124020b4bca1400559",
        "identities": [],
    },
    M8R13_NATIVE_T4_EXPORTER_CERT_KERNEL_ID: {
        "exception_id": "m8r13-native-t4-exporter-cert-v1",
        "record_sha256": "659d2bd78554f9497271ba3539f2eb02bb986df9859020fc4d49e71ac76cdeb9",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M8R14_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID: {
        "exception_id": "m8r14-postilp-target-feature-cert-v1",
        "record_sha256": "8b2742427ad910cca005395d9dffebb0599a9830bb8fd788549f959f83914b62",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M8R15_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID: {
        "exception_id": "m8r15-postilp-target-feature-cert-v1",
        "record_sha256": "090759d804f3683212a7dddbab8a6a1127eec810b1c0d3781997a332677ea0d4",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M8R16_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID: {
        "exception_id": "m8r16-postilp-target-feature-cert-v1",
        "record_sha256": "93479a683b575b5d557de4ab5d7f8f4d5ad85e8efbf17a8d723a346707f25593",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M8R17_POSTILP_TARGET_FEATURE_CERT_KERNEL_ID: {
        "exception_id": "m8r17-postilp-target-feature-cert-v1",
        "record_sha256": "edb59a3155bb365a6b82b6a9a4d86af7aba2eb09e599409f0290312d20c53cd2",
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    },
    M8R20_FULL21_TARGET_FEATURE_CERT_KERNEL_ID: {
        "exception_id": "m8r20-full21-target-feature-provider-cert-v1",
        "record_sha256": "876719dab75d41b6ac8f589543d0b581e71de2531f1faefa5ca9394ef53786ee",
        "identities": [
            "train:44b6_24264f12",
            "train:44b6_267148e4",
            "train:44b6_33b596bf",
            "train:44b6_341df25f",
            "train:44b6_3bb3690f",
            "train:44b6_53f95252",
            "train:44b6_66f9292d",
            "train:44b6_71a4179f",
            "train:44b6_74d0c52e",
            "train:44b6_87bba6c4",
            "train:44b6_8cc6506c",
            "train:44b6_8f9ecab4",
            "train:44b6_996155de",
            "train:44b6_aaf8b0ea",
            "train:44b6_c8e2a523",
            "train:44b6_c96cfa10",
            "train:44b6_cf2536e8",
            "train:44b6_d2f34f90",
            "train:44b6_d5e7d891",
            "train:44b6_deabac95",
            "train:44b6_e57ff5c6",
        ],
    },
}


def _parse_kernel_status(text: str) -> str | None:
    """Return one exact Kaggle worker state, rejecting ambiguous prose."""

    matches = _KERNEL_STATUS.findall(text or "")
    if len(matches) != 1:
        return None
    return matches[0].upper()


def _status_kernel_ref_binds(kernel_ref: object, kernel_id: str) -> bool:
    """Accept Kaggle's exact slug or its exact ``slug/version`` status form.

    GPU reservations are keyed by the immutable kernel slug, while Kaggle's
    ``kernels status`` response is normally queried with the pushed version
    (for example ``owner/slug/1``).  A substring check would let a different
    slug release this reservation, so accept only the full slug or a single
    positive integer version suffix.
    """

    if not isinstance(kernel_ref, str) or not kernel_id:
        return False
    return re.fullmatch(
        re.escape(kernel_id) + r"(?:/[1-9][0-9]*)?", kernel_ref
    ) is not None


def _status_proves_fresh_kernel_absence(receipt: dict[str, Any]) -> bool:
    """Accept Kaggle's exact new-slug denial only for an explicit v1 predicate."""

    stderr = str(receipt.get("stderr_scrubbed") or "")
    return (
        receipt.get("returncode") == 1
        and receipt.get("timed_out") is False
        and receipt.get("semantic_failure") is False
        and not str(receipt.get("stdout_scrubbed") or "").strip()
        and "Cannot access kernel " in stderr
        and "Permission 'kernels.get' was denied" in stderr
        and "most likely cause is a wrong kernel slug" in stderr
    )

#: Ops that mutate external state and therefore require the serial lock.
MUTATING_OPS = frozenset(
    {"dataset_create", "dataset_version", "kernel_push", "competition_submit"}
)


class CmuxOpsError(RuntimeError):
    """Base class for every refusal raised by this module."""


class LockHeld(CmuxOpsError):
    """Another holder owns the serial lock."""


def _load_lmt_policy(path: Path | str = DEFAULT_LMT) -> dict[str, Any]:
    """Load the one machine-readable autonomy policy from LMT.md."""

    lmt_path = Path(path)
    try:
        source = lmt_path.read_text(encoding="utf-8")
        block = source.split(LMT_POLICY_BEGIN, 1)[1].split(LMT_POLICY_END, 1)[0].strip()
        if block.startswith("```json"):
            block = block[len("```json") :].strip()
        if block.endswith("```"):
            block = block[:-3].strip()
        policy = json.loads(block)
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise CmuxOpsError(f"LMT policy is absent or malformed: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("schema") != "cell-tracking.autonomy-limits.v1":
        raise CmuxOpsError("LMT policy schema mismatch")
    kaggle = policy.get("kaggle")
    if not isinstance(kaggle, dict):
        raise CmuxOpsError("LMT Kaggle policy is missing")
    for key in (
        "autonomous_projection_strictly_below_hours",
        "weekly_gpu_budget_hours",
    ):
        value = kaggle.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CmuxOpsError(f"LMT Kaggle policy {key} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise CmuxOpsError(f"LMT Kaggle policy {key} must be finite and positive")
    if kaggle.get("auto_approved_wall_clock_kill_seconds") is not None:
        raise CmuxOpsError("LMT auto-approved Kaggle runs must not have an agent wall-clock kill")
    return policy


class PredicateNotSatisfied(CmuxOpsError):
    """competition_submit was attempted without an eligible predicate receipt."""


class KernelPredicateNotSatisfied(CmuxOpsError):
    """kernel_push was attempted without a fresh eligible package receipt."""


class PushRefused(CmuxOpsError):
    """A kernel push was refused because a version is already in flight."""


class BudgetLedgerError(CmuxOpsError):
    """The single shared GPU ledger is absent, malformed, or unreconciled."""


class InsufficientGpuQuota(BudgetLedgerError):
    """Authenticated remaining quota cannot cover all live reservations."""


class ProjectionApprovalRequired(BudgetLedgerError):
    """The current LMT autonomous projection gate is unsatisfied."""


class DatasetMetadataNotSatisfied(CmuxOpsError):
    """A local Kaggle Dataset package has invalid creation metadata."""


class UnknownOp(CmuxOpsError):
    """The requested op is not in the bounded command plan."""


# --------------------------------------------------------------------------------------
# The bounded command plan -- the security boundary of this module.
# --------------------------------------------------------------------------------------


def command_plan(kaggle: str = "kaggle", **kw: Any) -> dict[str, list[str]]:
    """Return the COMPLETE set of external commands this module may ever execute.

    Every value is an argv list. Nothing here is ever passed through a shell, and no
    caller-supplied string is interpolated into a command word. Adding an entry is a
    reviewable change with a reason in the calling node's report -- not a convenience.
    """
    dataset_id = kw.get("dataset_id", "")
    kernel_id = kw.get("kernel_id", "")
    kernel_version = kw.get("kernel_version")
    if kernel_version is None:
        kernel_ref = kernel_id
    else:
        if (
            isinstance(kernel_version, bool)
            or not isinstance(kernel_version, int)
            or kernel_version <= 0
        ):
            raise ValueError("kernel_version must be a positive integer")
        kernel_ref = f"{kernel_id}/{kernel_version}"
    path = str(kw.get("path", ""))
    message = kw.get("message", "")
    competition = kw.get("competition", DEFAULT_COMPETITION)
    csv = str(kw.get("csv", ""))

    return {
        # --- read-only -------------------------------------------------------------
        "quota": [kaggle, "quota", "--format", "json"],
        "dataset_status": [kaggle, "datasets", "status", dataset_id, "--format", "json"],
        "dataset_search_mine": [
            kaggle, "datasets", "list", "--mine", "--search",
            dataset_id.split("/", 1)[-1], "--format", "json",
        ],
        "dataset_files": [
            kaggle, "datasets", "files", dataset_id,
            "--page-size", "200", "--format", "json",
        ],
        "dataset_download": [
            kaggle, "datasets", "download", "-d", dataset_id, "-p", path, "--unzip",
        ],
        "kernel_status": [kaggle, "kernels", "status", kernel_ref],
        "kernel_list_mine": [
            kaggle, "kernels", "list", "--mine", "--page-size", "100", "--format", "json",
        ],
        "kernel_files": [
            kaggle, "kernels", "files", kernel_ref,
            "--page-size", "200", "--format", "json",
        ],
        "kernel_pull": [
            kaggle, "kernels", "pull", kernel_ref, "-p", path, "--metadata",
        ],
        "kernel_output": [kaggle, "kernels", "output", kernel_ref, "-p", path],
        # Version-qualified logs are the only remaining read-only route for a
        # historical kernel traceback when ``output`` returns latest artifacts.
        "kernel_logs": [kaggle, "kernels", "logs", kernel_ref],
        "competition_submissions": [
            kaggle, "competitions", "submissions", competition,
            "--page-size", "50", "--format", "json",
        ],
        # --- mutating --------------------------------------------------------------
        "dataset_create": [
            kaggle, "datasets", "create", "-p", path, "--dir-mode", "zip",
        ],
        "dataset_version": [
            kaggle, "datasets", "version", "-p", path, "-m", message, "-r", "zip",
        ],
        "kernel_push": [kaggle, "kernels", "push", "-p", path],
        # --- irreversible ----------------------------------------------------------
        "competition_submit": [
            kaggle, "competitions", "submit", "-c", competition, "-f", csv, "-m", message,
        ],
    }


# --------------------------------------------------------------------------------------
# Credential scrubbing -- a receipt must never carry a secret.
# --------------------------------------------------------------------------------------

_SECRET_KEY = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization|bearer|cookie|"
    r"session|credential)\b\s*[:=]\s*\S+"
)
_LONG_OPAQUE = re.compile(r"\b[A-Za-z0-9+/_-]{32,}={0,2}\b")
_REDACTED = "[REDACTED]"
_HOURS = re.compile(r"^([0-9]+(?:\.[0-9]+)?)h$")


def scrub(text: str) -> str:
    """Remove anything that looks like a credential from *text*.

    Deliberately aggressive: a false positive costs a little readability in a receipt,
    a false negative writes the owner's token into a committed artifact.
    """
    if not text:
        return ""
    out = _SECRET_KEY.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)
    return _LONG_OPAQUE.sub(_REDACTED, out)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_KAGGLE_DATASET_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def _validate_dataset_create_metadata(
    package_dir: Path | str,
    *,
    expected_dataset_id: str = "",
) -> dict[str, Any]:
    """Validate Kaggle's required Dataset metadata before any external process.

    Kaggle accepts the create request only when ``dataset-metadata.json`` has a
    6..50-character title.  Checking that provider constraint here prevents a
    malformed package from consuming an external create attempt.  The required
    Dataset identity and privacy declaration are bound at the same boundary.
    """

    root = Path(package_dir)
    metadata_path = root / "dataset-metadata.json"
    if root.is_symlink() or not root.is_dir():
        raise DatasetMetadataNotSatisfied(
            "dataset_create path must be a non-symbolic-link directory"
        )
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise DatasetMetadataNotSatisfied(
            "dataset_create requires a regular dataset-metadata.json"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetMetadataNotSatisfied(
            f"dataset-metadata.json is unreadable or invalid JSON: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise DatasetMetadataNotSatisfied(
            "dataset-metadata.json must contain one JSON object"
        )

    title = metadata.get("title")
    if not isinstance(title, str) or title != title.strip():
        raise DatasetMetadataNotSatisfied(
            "Kaggle Dataset title must be a trimmed string"
        )
    if not 6 <= len(title) <= 50:
        raise DatasetMetadataNotSatisfied(
            f"Kaggle Dataset title length must be 6..50 characters; observed {len(title)}"
        )

    dataset_id = metadata.get("id")
    if not isinstance(dataset_id, str) or _KAGGLE_DATASET_ID.fullmatch(dataset_id) is None:
        raise DatasetMetadataNotSatisfied(
            "Kaggle Dataset id must be an owner/slug string"
        )
    if expected_dataset_id and dataset_id != expected_dataset_id:
        raise DatasetMetadataNotSatisfied(
            "dataset_create dataset_id does not match dataset-metadata.json id"
        )
    if type(metadata.get("isPrivate")) is not bool:
        raise DatasetMetadataNotSatisfied(
            "Kaggle Dataset metadata must declare boolean isPrivate"
        )

    return {
        "path": str(metadata_path.resolve()),
        "sha256": _sha256_file(metadata_path),
        "id": dataset_id,
        "title": title,
        "title_length": len(title),
        "is_private": metadata["isPrivate"],
    }


def _relative_regular_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise KernelPredicateNotSatisfied(
            f"{label} must be a nonempty repository-relative path"
        )
    unresolved = root / raw
    if unresolved.is_symlink():
        raise KernelPredicateNotSatisfied(f"{label} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise KernelPredicateNotSatisfied(f"{label} escapes project root") from exc
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise KernelPredicateNotSatisfied(f"{label} must be a nonempty regular file")
    return path


def _package_inventory(package: Path, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    resolved = package.resolve()
    try:
        relative_package = str(resolved.relative_to(root))
    except ValueError as exc:
        raise KernelPredicateNotSatisfied("kernel package escapes project root") from exc
    if package.is_symlink() or not resolved.is_dir():
        raise KernelPredicateNotSatisfied(
            "kernel package must be a non-symbolic-link directory"
        )
    files: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise KernelPredicateNotSatisfied(
                f"kernel package contains symbolic link: {path}"
            )
        if path.is_file():
            files.append(
                {
                    "path": str(path.resolve().relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not files:
        raise KernelPredicateNotSatisfied("kernel package contains no regular files")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": relative_package,
        "file_count": len(files),
        "files": files,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _rerun_kaggle_prelaunch_guard(
    *,
    kernel_dir: Path,
    contract_path: Path,
    registry_path: Path,
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
) -> dict[str, Any]:
    """Load the repository guard and reproduce its verdict from current bytes."""

    spec = importlib.util.spec_from_file_location(
        "celltrack_kaggle_prelaunch_guard_replay", KAGGLE_PRELAUNCH_GUARD
    )
    if spec is None or spec.loader is None:
        raise KernelPredicateNotSatisfied("real Kaggle prelaunch guard is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        result = module.check_kernel(
            kernel_dir,
            module.load_json(contract_path),
            module.load_json(registry_path),
            phase,
            expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )
    except Exception as exc:
        raise KernelPredicateNotSatisfied(
            f"real Kaggle prelaunch guard replay failed: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise KernelPredicateNotSatisfied("real Kaggle prelaunch guard returned no result")
    return result


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _kaggle_credentials_path() -> Path:
    """Return the one credential file the authenticated Kaggle CLI may use."""
    return Path.home() / ".kaggle" / "credentials.json"


def _authenticated_kaggle_environment(kaggle: str) -> dict[str, str] | None:
    """Supply the owner's API token to a real Kaggle CLI without exposing it.

    The current CLI requires ``KAGGLE_API_TOKEN`` but does not automatically
    consume the existing access-token credential file.  Test fixtures and
    arbitrary non-Kaggle executables receive no altered environment, which
    keeps the command-plan test surface credential-free.
    """
    if PurePath(kaggle).name not in {"kaggle", "kaggle.exe"}:
        return None
    candidate = Path(kaggle)
    if not candidate.is_absolute():
        resolved_text = shutil.which(kaggle)
        if resolved_text is None:
            return None
        candidate = Path(resolved_text)
    candidate = candidate.resolve()
    trusted: set[Path] = set()
    configured = os.environ.get("CELLTRACK_KAGGLE_EXECUTABLE")
    if configured:
        trusted.add(Path(configured).expanduser().resolve())
    discovered = shutil.which("kaggle")
    if discovered:
        trusted.add(Path(discovered).resolve())
    local_cli = REPO_ROOT / ".venv/bin/kaggle"
    if local_cli.is_file():
        trusted.add(local_cli.resolve())
    if candidate not in trusted:
        return None
    environment = os.environ.copy()
    if environment.get("KAGGLE_API_TOKEN"):
        return environment
    path = _kaggle_credentials_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        token = value["access_token"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CmuxOpsError(
            "authenticated Kaggle access token is unavailable; refusing external action"
        ) from exc
    if not isinstance(token, str) or len(token) < 32:
        raise CmuxOpsError(
            "authenticated Kaggle access token is malformed; refusing external action"
        )
    environment["KAGGLE_API_TOKEN"] = token
    return environment


def authenticated_kaggle_environment(kaggle: str) -> dict[str, str] | None:
    """Return the credential-safe environment for a bounded Kaggle command."""

    return _authenticated_kaggle_environment(kaggle)


# --------------------------------------------------------------------------------------
# Serial lock
# --------------------------------------------------------------------------------------


class serial_lock:
    """Exclusive lock guarding every mutating Kaggle op.

    The persistent path is only a diagnostic anchor. Ownership is an advisory OS lock
    held on an open descriptor, so process death releases it atomically and no supervisor
    ever deletes another supervisor's lock file. ``stale_after_s`` and ``status_probe``
    remain accepted for API compatibility but are intentionally not used to break locks;
    consequently they do not define an autonomous timeout.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_LOCK,
        *,
        node: str = "",
        target: str = "",
        stale_after_s: float | None = None,
        status_probe: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.node = node
        self.target = target
        self.stale_after_s = stale_after_s
        self.status_probe = status_probe
        self._fd: int | None = None
        self.broke_stale: dict[str, Any] | None = None

    def _payload(self) -> str:
        return json.dumps(
            {
                "pid": os.getpid(),
                "node": self.node,
                "target": self.target,
                "acquired_utc": _utc(),
                "acquired_monotonic": time.time(),
                "broke_stale": self.broke_stale,
            },
            indent=2,
        )

    def _try_create(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        body = self._payload().encode()
        os.ftruncate(fd, 0)
        os.write(fd, body)
        os.fsync(fd)
        self._fd = fd
        return True

    def acquire(self) -> "serial_lock":
        if self._try_create():
            return self
        raise LockHeld(
            f"serial lock held at {self.path}; refusing to wait, delete, or race"
        )

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self) -> "serial_lock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _write_receipt(receipt: dict[str, Any], receipt_dir: Path | str | None) -> dict[str, Any]:
    if receipt_dir is None:
        return receipt
    d = Path(receipt_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{receipt['op']}_{receipt['started_utc'].replace(':', '').replace('.', '')}.json"
    p = d / name
    p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_path"] = str(p)
    receipt["receipt_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return receipt


def _validate_submission_predicate(
    path: Path | str,
    *,
    competition: str,
    csv: str,
) -> dict[str, Any]:
    """Validate a direct-file permit against the exact competition and bytes.

    A bare ``eligible: true`` flag is not mutation authority. Code-output
    submissions use the stricter version-bound lifecycle module instead.
    """

    predicate_path = Path(path).resolve()
    try:
        payload = json.loads(predicate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredicateNotSatisfied(f"submission predicate is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredicateNotSatisfied("submission predicate must be a JSON object")

    failures: list[str] = []
    schema = payload.get("schema")
    if not isinstance(schema, str) or re.fullmatch(
        r"celltrack-alpha\.[a-z0-9-]+-submission-predicate\.v[1-9][0-9]*",
        schema,
    ) is None:
        failures.append("unsupported predicate schema")
    if payload.get("status") != "PASS" or payload.get("eligible") is not True:
        failures.append("predicate is not PASS/eligible")
    if payload.get("competition") != competition:
        failures.append("competition binding drift")
    if payload.get("submission_mode") != "direct_file":
        failures.append("submission mode is not direct_file")
    if payload.get("revoked") is not False or payload.get("consumed") is not False:
        failures.append("predicate is revoked or consumed")
    if payload.get("failures") != []:
        failures.append("predicate failures are nonempty")

    try:
        artifact_path = Path(str(payload.get("artifact_path", ""))).resolve()
    except (OSError, RuntimeError, ValueError):
        artifact_path = Path()
    if not csv or artifact_path != Path(csv).resolve():
        failures.append("artifact path binding drift")
    expected_sha = payload.get("artifact_sha256")
    expected_bytes = payload.get("artifact_bytes")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        failures.append("artifact SHA is invalid")
    try:
        actual_bytes = artifact_path.stat().st_size
        actual_sha = _sha256_file(artifact_path)
    except OSError as exc:
        failures.append(f"artifact unreadable: {exc}")
    else:
        if expected_bytes != actual_bytes:
            failures.append("artifact byte-size drift")
        if expected_sha != actual_sha:
            failures.append("artifact SHA drift")
    if failures:
        raise PredicateNotSatisfied("; ".join(failures))
    return {
        "predicate_path": str(predicate_path),
        "predicate_sha256": _sha256_file(predicate_path),
        "schema": schema,
        "competition": competition,
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_sha,
        "artifact_bytes": expected_bytes,
    }


def _claim_direct_submission(predicate: dict[str, Any]) -> dict[str, Any]:
    """Consume one direct-file predicate globally before external mutation."""

    predicate_sha = str(predicate["predicate_sha256"])
    claim_path = DEFAULT_SUBMISSION_CLAIM_DIR / f"{predicate_sha}.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "celltrack-alpha.direct-submission-claim.v1",
        "status": "CLAIMED_BEFORE_EXTERNAL_MUTATION",
        "claimed_utc": _utc(),
        "predicate_sha256": predicate_sha,
        "competition": predicate["competition"],
        "artifact_sha256": predicate["artifact_sha256"],
        "retry_allowed": False,
    }
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(claim, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PredicateNotSatisfied(
            f"submission predicate was already consumed at {claim_path}"
        ) from exc
    return {"claim_path": str(claim_path), **claim}


def _validate_kernel_push_predicate(
    receipt_path: Path | str,
    *,
    kernel_id: str,
    kernel_dir: Path | str,
) -> dict[str, Any]:
    """Bind an attested prelaunch PASS to the exact local package bytes.

    The former launch path could bypass the prelaunch guard entirely. A stale
    receipt is also unsafe, so this check re-hashes the metadata and declared
    code file immediately before any status query or push.
    """
    receipt_file = Path(receipt_path)
    try:
        predicate = json.loads(receipt_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KernelPredicateNotSatisfied(
            f"kernel push predicate receipt unreadable: {exc}"
        ) from exc
    if not isinstance(predicate, dict):
        raise KernelPredicateNotSatisfied("kernel push predicate must be a JSON object")

    raw_project_root = predicate.get("project_root")
    try:
        project_root = Path(str(raw_project_root)).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise KernelPredicateNotSatisfied("predicate project root is malformed") from exc
    if (
        not isinstance(raw_project_root, str)
        or not Path(raw_project_root).is_absolute()
        or not project_root.is_dir()
    ):
        raise KernelPredicateNotSatisfied(
            "predicate project root must be an absolute existing directory"
        )
    canonical_root = VISUAL_GUARD_PROJECT_ROOT
    if project_root != canonical_root:
        raise KernelPredicateNotSatisfied(
            "predicate project root is not the canonical shared repository root"
        )
    try:
        receipt_file.resolve().relative_to(project_root)
    except ValueError as exc:
        raise KernelPredicateNotSatisfied(
            "kernel push predicate receipt is not repository-relative"
        ) from exc
    if receipt_file.is_symlink() or not receipt_file.is_file():
        raise KernelPredicateNotSatisfied(
            "kernel push predicate receipt must be a non-symbolic-link regular file"
        )

    package = Path(kernel_dir).resolve()
    try:
        package.relative_to(project_root)
    except ValueError as exc:
        raise KernelPredicateNotSatisfied(
            "kernel package is not repository-relative"
        ) from exc
    metadata_path = package / "kernel-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KernelPredicateNotSatisfied(
            f"kernel metadata unreadable at {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise KernelPredicateNotSatisfied("kernel metadata must be a JSON object")
    code_path = package / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise KernelPredicateNotSatisfied(
            f"declared kernel code file is absent: {code_path}"
        )

    contract_path = _relative_regular_file(
        project_root, predicate.get("contract_path"), "prelaunch contract"
    )
    registry_path = _relative_regular_file(
        project_root, predicate.get("registry_path"), "prelaunch registry"
    )
    canonical_registry = CANONICAL_KAGGLE_PRELAUNCH_REGISTRY.resolve()
    if registry_path != canonical_registry:
        raise KernelPredicateNotSatisfied(
            "prelaunch registry is not the canonical repository registry"
        )
    expected_initial_version = predicate.get("expected_initial_kernel_version")
    fresh = _rerun_kaggle_prelaunch_guard(
        kernel_dir=package,
        contract_path=contract_path,
        registry_path=registry_path,
        phase=str(predicate.get("phase", "")),
        expected_initial_kernel_version=expected_initial_version,
        project_root=project_root,
    )

    failures: list[str] = []
    if predicate != fresh:
        failures.append(
            "stored predicate is fabricated or stale relative to the real guard replay"
        )
    if predicate.get("schema") != "celltrack-alpha.kaggle-prelaunch-guard-result.v1":
        failures.append("unexpected predicate schema")
    if predicate.get("status") != "PASS" or predicate.get("eligible") is not True:
        failures.append("predicate is not PASS and eligible:true")
    if predicate.get("phase") not in {"inline", "launch"}:
        failures.append("predicate phase is not push-eligible")
    if predicate.get("failures") != []:
        failures.append("predicate contains failures")
    if fresh.get("status") != "PASS" or fresh.get("eligible") is not True:
        failures.append("real guard replay is not PASS and eligible:true")
    model_execution = predicate.get("model_execution")
    visual = predicate.get("local_visual_preflight")
    if not isinstance(model_execution, bool):
        failures.append("predicate lacks model_execution classification")
    elif not isinstance(visual, dict):
        failures.append("predicate lacks local visual preflight result")
    elif model_execution:
        if visual.get("required") is not True:
            failures.append("model kernel local visual preflight is not required")
        elif visual.get("status") == "PASS":
            if not isinstance(visual.get("video_count"), int) or visual["video_count"] <= 0:
                failures.append("model kernel local visual preflight has no reviewed MP4")
            identities = visual.get("identities")
            if (
                not isinstance(identities, list)
                or not identities
                or not all(isinstance(identity, str) for identity in identities)
            ):
                failures.append("model kernel local visual identities are absent")
            for field in ("receipt_sha256",):
                value = visual.get(field)
                if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    failures.append(f"model kernel local visual {field} is malformed")
        elif visual.get("status") == "REMOTE_VISUAL_PENDING":
            # The guard only emits this state after replaying a centrally pinned
            # resource-stop/GPU/ELIGIBLE-certificate exception.  It remains
            # visibly pending so no caller can mistake it for local MP4 review.
            exception_id = visual.get("exception_id")
            record_sha = visual.get("record_sha256")
            identities = visual.get("identities")
            post_review = visual.get("post_remote_review")
            if not isinstance(exception_id, str) or not exception_id:
                failures.append("remote visual-only exception identity is absent")
            if not isinstance(record_sha, str) or re.fullmatch(r"[0-9a-f]{64}", record_sha) is None:
                failures.append("remote visual-only exception record hash is malformed")
            if not isinstance(identities, list) or len(identities) != 2 or len(set(identities)) != 2:
                failures.append("remote visual-only exception lacks exactly two fixed identities")
            if not isinstance(post_review, dict) or any(
                post_review.get(key) != expected
                for key, expected in {
                    "remote_mp4_retrieval_required": True,
                    "local_visual_review_required": True,
                    "score_or_promotion_permitted_before_review": False,
                    "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_REVIEW",
                }.items()
            ):
                failures.append("remote visual-only exception lacks the mandatory post-retrieval local-review lock")
        elif visual.get("status") == "REMOTE_CERTIFICATION_PENDING":
            exception_id = visual.get("exception_id")
            record_sha = visual.get("record_sha256")
            identities = visual.get("identities")
            post_certification = visual.get("post_remote_certification")
            expected_exception = REMOTE_NATIVE_CERTIFICATION_EXCEPTIONS.get(kernel_id)
            if expected_exception is None:
                failures.append(
                    "remote certification exception is not the exact M7R6 kernel, exact M7R8 successor, or exact M10R3 provider diagnostic"
                )
            if visual.get("certification_only") is not True:
                failures.append("remote certification exception is not certification-only")
            if (
                expected_exception is None
                or exception_id != expected_exception["exception_id"]
            ):
                failures.append("remote certification exception identity drift")
            if (
                expected_exception is None
                or record_sha != expected_exception["record_sha256"]
            ):
                failures.append("remote certification parent-rejection record drift")
            expected_identities = (
                expected_exception.get(
                    "identities", ["train:44b6_cf2536e8", "train:44b6_3bb3690f"]
                )
                if expected_exception is not None
                else None
            )
            if identities != expected_identities:
                failures.append("remote certification fixed-identity contract drift")
            if kernel_id in {
                M10R3_PROVIDER_CERT_KERNEL_ID,
                M10R5_PROVIDER_CERT_KERNEL_ID,
                M12R8_PROVIDER_CERT_KERNEL_ID,
            } and visual.get("target_input") is not False:
                failures.append("no-target provider certificate target-input firewall drift")
            if not isinstance(post_certification, dict) or any(
                post_certification.get(key) != expected
                for key, expected in {
                    "downloaded_output_zip_required": True,
                    "exact_local_seal_required": True,
                    "production_or_score_permitted_before_seal": False,
                    "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
                }.items()
            ):
                failures.append("remote certification lacks the mandatory downloaded-evidence seal lock")
        else:
            failures.append("model kernel lacks a passing local visual preflight")
    elif visual.get("required") is not False or visual.get("status") != "NOT_APPLICABLE":
        failures.append("non-model kernel exemption is not guard-attested")
    if (
        not kernel_id
        or metadata.get("id") != kernel_id
        or predicate.get("kernel_id") != kernel_id
    ):
        failures.append("kernel ID drift")
    if predicate.get("kernel_dir") != str(package):
        failures.append("kernel directory drift")
    observed_package = _package_inventory(package, project_root)
    if predicate.get("package") != observed_package:
        failures.append("kernel package inventory changed after prelaunch")
    if predicate.get("package_tree_sha256") != observed_package["tree_sha256"]:
        failures.append("kernel package tree changed after prelaunch")
    if predicate.get("metadata_sha256") != _sha256_file(metadata_path):
        failures.append("kernel metadata changed after prelaunch")
    if predicate.get("code_sha256") != _sha256_file(code_path):
        failures.append("kernel code changed after prelaunch")
    try:
        predicate_code_path = Path(str(predicate.get("code_file", ""))).resolve()
    except (OSError, RuntimeError, ValueError):
        predicate_code_path = Path()
    if predicate_code_path != code_path.resolve():
        failures.append("declared code path drift")
    for field, path in (
        ("contract_sha256", contract_path),
        ("registry_sha256", registry_path),
    ):
        value = predicate.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            failures.append(f"{field} missing or malformed")
        elif value != _sha256_file(path):
            failures.append(f"{field} changed after prelaunch")
    if isinstance(visual, dict):
        artifacts = visual.get("artifacts")
        if model_execution and (not isinstance(artifacts, list) or not artifacts):
            failures.append("model kernel local visual artifact bindings are absent")
        elif isinstance(artifacts, list):
            for index, artifact in enumerate(artifacts):
                if not isinstance(artifact, dict):
                    failures.append(f"local visual artifact {index} is malformed")
                    continue
                try:
                    artifact_path = _relative_regular_file(
                        project_root,
                        artifact.get("path"),
                        f"local visual artifact {index}",
                    )
                except KernelPredicateNotSatisfied as exc:
                    failures.append(str(exc))
                    continue
                expected_hash = artifact.get("sha256")
                if (
                    not isinstance(expected_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
                    or _sha256_file(artifact_path) != expected_hash
                ):
                    failures.append(f"local visual artifact {index} hash drift")
    if expected_initial_version not in {None, 1}:
        failures.append("expected_initial_kernel_version must be exactly 1 when present")
    if failures:
        raise KernelPredicateNotSatisfied(
            "kernel push predicate rejected: " + "; ".join(failures)
        )
    return {
        "receipt_path": str(receipt_file.resolve()),
        "receipt_sha256": _sha256_file(receipt_file),
        "kernel_id": kernel_id,
        "kernel_dir": str(package),
        "project_root": str(canonical_root),
        "metadata_sha256": _sha256_file(metadata_path),
        "code_sha256": _sha256_file(code_path),
        "contract_path": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256_file(registry_path),
        "package_tree_sha256": observed_package["tree_sha256"],
        "phase": predicate["phase"],
        "model_execution": model_execution,
        "local_visual_preflight": visual,
        "remote_visual_only_exception": (
            visual.get("exception_id")
            if isinstance(visual, dict) and visual.get("status") == "REMOTE_VISUAL_PENDING"
            else None
        ),
        "remote_certification_exception": (
            visual.get("exception_id")
            if isinstance(visual, dict)
            and visual.get("status") == "REMOTE_CERTIFICATION_PENDING"
            else None
        ),
        "expected_initial_kernel_version": expected_initial_version,
    }


_MUTATION_ERROR_TEXT = re.compile(
    r"(?im)^(?:dataset creation|dataset versioning|kernel push|competition submission) "
    r"error:\s*(.+)$"
)


def _semantic_mutation_error(op: str, stdout: str, stderr: str) -> str | None:
    """Detect Kaggle CLI mutations that print failure text but exit zero.

    Kaggle CLI 2.x can report ``Dataset creation error: ...`` on stdout while
    returning process status 0.  Treating that as a successful mutation can
    advance a launch into a missing or partial input carrier.
    """
    if op not in MUTATING_OPS:
        return None
    match = _MUTATION_ERROR_TEXT.search(f"{stdout}\n{stderr}")
    if match is not None:
        return f"Kaggle CLI reported {match.group(0).strip()}"
    return None


def semantic_mutation_error(op: str, stdout: str, stderr: str) -> str | None:
    """Public semantic-error classifier for bounded lifecycle modules."""

    return _semantic_mutation_error(op, stdout, stderr)


def run(
    op: str,
    *,
    kaggle: str = "kaggle",
    receipt_dir: Path | str | None = None,
    timeout: float | None = 1800.0,
    submission_predicate_receipt: Path | str | None = None,
    kernel_push_predicate_receipt: Path | str | None = None,
    lock_path: Path | str = DEFAULT_LOCK,
    _held_lock: serial_lock | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Execute one bounded op and return its receipt.

    Never retries. A failure returns its receipt with the real returncode; deciding whether
    a retry is legitimate is a registry/human decision (KAGGLE_FAILURE_REGISTRY.json),
    never a loop in here.
    """
    dataset_metadata = None
    if op == "dataset_create":
        dataset_metadata = _validate_dataset_create_metadata(
            kw.get("path", ""),
            expected_dataset_id=str(kw.get("dataset_id", "")),
        )
    plan = command_plan(kaggle=kaggle, **kw)
    if op not in plan:
        raise UnknownOp(f"{op!r} is not in the bounded command plan: {sorted(plan)}")

    if op == "kernel_push" and kernel_push_predicate_receipt is None:
        raise KernelPredicateNotSatisfied(
            "kernel_push requires kernel_push_predicate_receipt; there is no override"
        )
    if op == "kernel_push" and _held_lock is None:
        raise BudgetLedgerError(
            "direct kernel_push is forbidden; use bracketed_launch with a positive "
            "gpu_reservation_h so the shared weekly ledger is checked"
        )

    owned_lock: serial_lock | None = None
    if op in MUTATING_OPS:
        if _held_lock is not None:
            if _held_lock._fd is None:
                raise LockHeld("the supplied mutation lock is not held")
        else:
            owned_lock = serial_lock(
                lock_path,
                node=f"cmux_ops:{op}",
                target=str(
                    kw.get("kernel_id")
                    or kw.get("dataset_id")
                    or kw.get("competition", DEFAULT_COMPETITION)
                ),
            )
            owned_lock.acquire()

    submission_predicate = None
    submission_claim = None
    kernel_push_predicate = None
    kernel_push_status_guard = None
    try:
        # Predicate validation occurs under the same lock as the mutation,
        # closing the former validate-then-wait time-of-check/time-of-use gap.
        if op == "competition_submit":
            if submission_predicate_receipt is None:
                raise PredicateNotSatisfied(
                    "competition_submit requires submission_predicate_receipt; there is no override"
                )
            submission_predicate = _validate_submission_predicate(
                submission_predicate_receipt,
                competition=str(kw.get("competition", DEFAULT_COMPETITION)),
                csv=str(kw.get("csv", "")),
            )
            submission_claim = _claim_direct_submission(submission_predicate)

        if op == "kernel_push":
            if kernel_push_predicate_receipt is None:
                raise KernelPredicateNotSatisfied(
                    "kernel_push requires kernel_push_predicate_receipt; there is no override"
                )
            kernel_push_predicate = _validate_kernel_push_predicate(
                kernel_push_predicate_receipt,
                kernel_id=str(kw.get("kernel_id", "")),
                kernel_dir=str(kw.get("path", "")),
            )
            status_kw = dict(kw)
            status_kw["kernel_version"] = None
            status = run(
                "kernel_status",
                kaggle=kaggle,
                receipt_dir=receipt_dir,
                timeout=timeout,
                **status_kw,
            )
            parsed_status = _parse_kernel_status(
                str(status.get("stdout_scrubbed") or "")
            )
            expected_initial_version = kernel_push_predicate.get(
                "expected_initial_kernel_version"
            )
            if expected_initial_version == 1:
                if not _status_proves_fresh_kernel_absence(status):
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: predicate requires "
                        "a fresh exact Version 1, but the under-lock status did not prove "
                        "that the slug is absent"
                    )
                kernel_push_status_guard = {
                    "disposition": "FRESH_SLUG_ABSENT_EXPECT_VERSION_1",
                    "status_receipt": status.get("receipt_path"),
                    "status_receipt_sha256": status.get("receipt_sha256"),
                }
            else:
                if (
                    status.get("returncode") != 0
                    or status.get("timed_out") is True
                    or parsed_status is None
                ):
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: latest-version status "
                        "was unsuccessful or unparseable"
                    )
                if parsed_status in {"RUNNING", "QUEUED"}:
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: a version is already in flight "
                        f"({status.get('stdout_scrubbed', '').strip()!r}). This is the 0.74 GPU-h "
                        "launch-race guard."
                    )
                kernel_push_status_guard = {
                    "disposition": f"EXISTING_{parsed_status}",
                    "status_receipt": status.get("receipt_path"),
                    "status_receipt_sha256": status.get("receipt_sha256"),
                }

            # The under-lock status call is still an execution window.  Reproduce
            # the real guard a second time immediately before constructing the
            # mutating process so post-guard package/artifact tampering cannot race
            # the push.
            immediate = _validate_kernel_push_predicate(
                kernel_push_predicate_receipt,
                kernel_id=str(kw.get("kernel_id", "")),
                kernel_dir=str(kw.get("path", "")),
            )
            if immediate != kernel_push_predicate:
                raise KernelPredicateNotSatisfied(
                    "kernel push bindings changed during the under-lock status check"
                )

        if op == "dataset_create":
            immediate_dataset_metadata = _validate_dataset_create_metadata(
                kw.get("path", ""),
                expected_dataset_id=str(kw.get("dataset_id", "")),
            )
            if immediate_dataset_metadata != dataset_metadata:
                raise DatasetMetadataNotSatisfied(
                    "dataset metadata changed after initial preflight"
                )
            dataset_metadata = immediate_dataset_metadata

        argv = plan[op]
        started = _utc()
        t0 = time.time()
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, never shell
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env=_authenticated_kaggle_environment(kaggle),
            )
            process_rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
            semantic_error = _semantic_mutation_error(op, out, err)
            rc = 1 if process_rc == 0 and semantic_error else process_rc
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            rc, process_rc, timed_out, semantic_error = None, None, True, None
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        receipt = {
            "schema": "celltrack-alpha.cmux-op-receipt.v1",
            "op": op,
            "argv": argv,
            "mutating": op in MUTATING_OPS,
            "started_utc": started,
            "ended_utc": _utc(),
            "duration_s": round(time.time() - t0, 6),
            "returncode": rc,
            "process_returncode": process_rc,
            "semantic_failure": semantic_error is not None,
            "semantic_failure_reason": semantic_error,
            "timed_out": timed_out,
            "stdout_sha256": _sha256(out),
            "stderr_sha256": _sha256(err),
            "stdout_scrubbed": scrub(out),
            "stderr_scrubbed": scrub(err),
        }
        if kernel_push_predicate is not None:
            receipt["kernel_push_predicate"] = kernel_push_predicate
        if kernel_push_status_guard is not None:
            receipt["kernel_push_status_guard"] = kernel_push_status_guard
        if submission_predicate is not None:
            receipt["submission_predicate"] = submission_predicate
        if submission_claim is not None:
            receipt["submission_claim"] = submission_claim
        if dataset_metadata is not None:
            receipt["dataset_metadata_preflight"] = dataset_metadata
        return _write_receipt(receipt, receipt_dir)
    finally:
        if owned_lock is not None:
            owned_lock.release()


def parse_authoritative_gpu_quota(payload: Any) -> dict[str, Any]:
    """Normalize only an authenticated, unambiguous Kaggle GPU quota response.

    The real CLI emits a list of resource rows using values such as ``"5.43h"``.
    A test fixture may use the already-normalized mapping form.  Any other shape is
    an integrity failure, never a reason to run the mutating operation anyway.
    """
    if isinstance(payload, list):
        rows = [
            row for row in payload
            if isinstance(row, dict) and str(row.get("resource", "")).upper() == "GPU"
        ]
        if len(rows) != 1:
            raise CmuxOpsError(
                f"expected exactly one GPU quota row, found {len(rows)}"
            )
        row = rows[0]

        def hours(name: str) -> float:
            match = _HOURS.fullmatch(str(row.get(name, "")).strip())
            if match is None:
                raise CmuxOpsError(f"malformed GPU quota {name!r}: {row.get(name)!r}")
            return float(match.group(1))

        result = {
            "used_h": hours("used"),
            "remaining_h": hours("remaining"),
            "total_h": hours("total"),
            "refresh_at": row.get("refreshAt"),
        }
        if not all(
            math.isfinite(result[key]) and result[key] >= 0
            for key in ("used_h", "remaining_h", "total_h")
        ):
            raise CmuxOpsError("GPU quota contains a non-finite or negative balance")
        if round(result["used_h"] + result["remaining_h"], 6) != round(result["total_h"], 6):
            raise CmuxOpsError(f"GPU quota arithmetic drifted: {result}")
        return result
    if isinstance(payload, dict):
        try:
            used = float(payload["used_h"])
            remaining = float(payload["remaining_h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CmuxOpsError("normalized quota fixture is malformed") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (used, remaining)):
            raise CmuxOpsError(
                "normalized quota fixture contains a non-finite or negative balance"
            )
        result = {"used_h": used, "remaining_h": remaining}
        if "total_h" in payload:
            total = float(payload["total_h"])
            if not math.isfinite(total) or total < 0:
                raise CmuxOpsError(
                    "normalized quota fixture contains a non-finite or negative total"
                )
            if round(used + remaining, 6) != round(total, 6):
                raise CmuxOpsError("normalized quota arithmetic drifted")
            result["total_h"] = total
        if "refresh_at" in payload:
            result["refresh_at"] = payload["refresh_at"]
        return result
    raise CmuxOpsError("quota response is neither a Kaggle resource list nor a normalized mapping")


GPU_LEDGER_SCHEMA = "celltrack-alpha.kaggle-shared-gpu-ledger.v1"


def _read_gpu_ledger(path: Path | str) -> dict[str, Any]:
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetLedgerError(
            f"shared GPU ledger is unavailable or malformed at {ledger_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != GPU_LEDGER_SCHEMA:
        raise BudgetLedgerError("shared GPU ledger has an unexpected schema")
    if not isinstance(payload.get("active_reservations"), list):
        raise BudgetLedgerError("shared GPU ledger active_reservations is malformed")
    if not isinstance(payload.get("events"), list):
        raise BudgetLedgerError("shared GPU ledger events is malformed")
    try:
        capacity = float(payload.get("capacity_h"))
        reservation_hours = [
            float(item["max_gpu_h"])
            for item in payload["active_reservations"]
            if isinstance(item, dict)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetLedgerError("shared GPU ledger hour values are malformed") from exc
    if len(reservation_hours) != len(payload["active_reservations"]):
        raise BudgetLedgerError("shared GPU ledger contains a non-object reservation")
    if not math.isfinite(capacity) or capacity <= 0 or any(
        not math.isfinite(value) or value <= 0 for value in reservation_hours
    ):
        raise BudgetLedgerError("shared GPU ledger contains non-finite/non-positive hours")
    return payload


def _write_gpu_ledger(path: Path | str, payload: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ledger_path)
    finally:
        temporary.unlink(missing_ok=True)


def _authoritative_snapshot(quota: dict[str, Any], *, receipt: str | None = None) -> dict[str, Any]:
    weekly_cap = float(_load_lmt_policy()["kaggle"]["weekly_gpu_budget_hours"])
    snapshot = {
        "used_h": float(quota["used_h"]),
        "remaining_h": float(quota["remaining_h"]),
        "total_h": float(quota.get("total_h", weekly_cap)),
        "refresh_at": quota.get("refresh_at"),
        "recorded_utc": _utc(),
    }
    if receipt:
        receipt_path = Path(receipt)
        snapshot["receipt"] = str(receipt_path)
        if receipt_path.is_file():
            snapshot["receipt_sha256"] = _sha256_file(receipt_path)
    return snapshot


def _reserve_gpu_budget(
    *,
    ledger_path: Path | str,
    quota: dict[str, Any],
    kernel_id: str,
    max_gpu_h: float,
    quota_receipt: str | None,
) -> dict[str, Any]:
    if not kernel_id:
        raise BudgetLedgerError("GPU reservation requires a kernel_id")
    if not math.isfinite(max_gpu_h) or max_gpu_h <= 0:
        raise BudgetLedgerError("kernel_push requires a positive --gpu-reservation-h")
    ledger = _read_gpu_ledger(ledger_path)
    policy = _load_lmt_policy()
    policy_capacity = float(policy["kaggle"]["weekly_gpu_budget_hours"])
    capacity = float(ledger.get("capacity_h", 0.0))
    total = float(quota.get("total_h", capacity))
    if not all(
        math.isfinite(value) and value >= 0
        for value in (capacity, total, float(quota["remaining_h"]))
    ):
        raise BudgetLedgerError("quota/ledger hours must be finite and non-negative")
    if capacity != total:
        raise BudgetLedgerError(
            f"shared ledger capacity {capacity} differs from authenticated quota {total}"
        )
    if capacity != policy_capacity:
        raise BudgetLedgerError(
            f"shared ledger capacity {capacity} differs from LMT weekly cap {policy_capacity}"
        )
    refresh_at = quota.get("refresh_at")
    if ledger.get("refresh_at") != refresh_at:
        if ledger["active_reservations"]:
            raise BudgetLedgerError(
                "quota refresh boundary changed while reservations remain; reconcile first"
            )
        ledger["refresh_at"] = refresh_at
        ledger["events"].append(
            {"type": "weekly_rollover", "recorded_utc": _utc(), "refresh_at": refresh_at}
        )
    reservations = ledger["active_reservations"]
    if any(item.get("kernel_id") == kernel_id for item in reservations):
        raise BudgetLedgerError(f"kernel already has an active GPU reservation: {kernel_id}")
    reserved_h = sum(float(item["max_gpu_h"]) for item in reservations)
    required_h = reserved_h + max_gpu_h
    if required_h > float(quota["remaining_h"]) + 1e-9:
        raise InsufficientGpuQuota(
            f"authenticated remaining {quota['remaining_h']} h cannot cover existing "
            f"reservations {reserved_h} h plus new ceiling {max_gpu_h} h"
        )
    reservation = {
        "kernel_id": kernel_id,
        "max_gpu_h": max_gpu_h,
        "reserved_utc": _utc(),
        "status": "RESERVED_PRELAUNCH",
    }
    reservations.append(reservation)
    ledger["latest_authoritative_quota"] = _authoritative_snapshot(
        quota, receipt=quota_receipt
    )
    ledger["events"].append(
        {
            "type": "reserve",
            "recorded_utc": _utc(),
            "kernel_id": kernel_id,
            "max_gpu_h": max_gpu_h,
            "existing_reserved_h": reserved_h,
            "authenticated_remaining_h": float(quota["remaining_h"]),
        }
    )
    ledger["updated_utc"] = _utc()
    _write_gpu_ledger(ledger_path, ledger)
    return reservation


def _projection_gate(
    authenticated_projection_h: float | None,
    human_approval_receipt: Path | str | None,
) -> dict[str, Any]:
    """Enforce the LMT autonomous projection boundary without a runtime kill.

    The weekly ledger reservation remains independently mandatory.  This gate is
    intentionally based on the authenticated estimate, never on sharding or a
    synthetic per-process timeout.
    """

    policy = _load_lmt_policy()
    kaggle_policy = policy["kaggle"]
    threshold_h = float(kaggle_policy["autonomous_projection_strictly_below_hours"])
    if authenticated_projection_h is None:
        raise ProjectionApprovalRequired(
            "kernel_push requires authenticated_projection_h for the LMT gate"
        )
    projection = float(authenticated_projection_h)
    if not math.isfinite(projection) or projection <= 0:
        raise ProjectionApprovalRequired(
            "authenticated_projection_h must be a finite positive number"
        )
    result: dict[str, Any] = {
        "authenticated_projection_h": projection,
        "threshold_h": threshold_h,
        "runtime_wall_clock_kill": False,
        "lmt_path": str(DEFAULT_LMT.resolve()),
        "lmt_sha256": _sha256_file(DEFAULT_LMT),
    }
    if projection < threshold_h:
        result["disposition"] = "AUTO_APPROVED_UNDER_LMT_CAP"
        return result

    if human_approval_receipt is None:
        raise ProjectionApprovalRequired(
            f"projection >= LMT cap ({threshold_h:g} h) requires an explicit human approval receipt with a "
            "literature-cited pitch; do not shard to evade this gate"
        )
    path = Path(human_approval_receipt)
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionApprovalRequired(
            f"human approval receipt is unreadable: {exc}"
        ) from exc
    if not isinstance(approval, dict):
        raise ProjectionApprovalRequired("human approval receipt must be a JSON object")
    required_text = (
        "expected_effect_size",
        "compute_necessity",
        "failure_modes",
        "falsification_evidence",
        "approved_by",
    )
    missing = [key for key in required_text if not str(approval.get(key, "")).strip()]
    sources = approval.get("primary_sources")
    if (
        approval.get("status") != "APPROVED_BY_HUMAN"
        or not isinstance(sources, list)
        or not sources
        or missing
    ):
        raise ProjectionApprovalRequired(
            "human approval receipt must be APPROVED_BY_HUMAN and include primary_sources, "
            "effect size, compute necessity, failure modes, falsification evidence, and approver"
        )
    result.update(
        {
            "disposition": "HUMAN_APPROVED_AT_OR_ABOVE_LMT_CAP",
            "approval_receipt": str(path.resolve()),
            "approval_receipt_sha256": _sha256_file(path),
        }
    )
    return result


def reconcile_gpu_reservation(
    kernel_id: str,
    *,
    terminal_status: str,
    quota: dict[str, Any] | None = None,
    quota_receipt: str | None = None,
    terminal_receipt: str | None = None,
    ledger_path: Path | str = DEFAULT_GPU_LEDGER,
    lock_path: Path | str = DEFAULT_LOCK,
    _held_lock: serial_lock | None = None,
) -> dict[str, Any]:
    """Release one terminal reservation while holding the canonical mutation lock."""

    owned_lock: serial_lock | None = None
    if _held_lock is not None:
        if _held_lock._fd is None:
            raise LockHeld("the supplied reservation-reconcile lock is not held")
    else:
        owned_lock = serial_lock(
            lock_path, node="reconcile_gpu_reservation", target=kernel_id
        ).acquire()
    try:
        normalized_status = terminal_status.upper()
        if normalized_status not in {"COMPLETE", "ERROR", "CANCELLED"}:
            raise BudgetLedgerError(f"nonterminal reservation status: {terminal_status!r}")
        ledger = _read_gpu_ledger(ledger_path)
        matches = [
            item for item in ledger["active_reservations"]
            if item.get("kernel_id") == kernel_id
        ]
        if len(matches) != 1:
            raise BudgetLedgerError(
                f"expected one active reservation for {kernel_id}, found {len(matches)}"
            )
        reservation_status = str(matches[0].get("status", ""))
        internal_prelaunch_rollback = (
            normalized_status == "ERROR"
            and reservation_status == "RESERVED_PRELAUNCH"
            and _held_lock is not None
        )
        if not terminal_receipt and not internal_prelaunch_rollback:
            raise BudgetLedgerError(
                "terminal reservation release requires kernel-bound terminal evidence"
            )
        if terminal_receipt:
            terminal_path = Path(terminal_receipt)
            try:
                terminal_evidence = json.loads(terminal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BudgetLedgerError(f"terminal receipt is unreadable: {exc}") from exc
            if not isinstance(terminal_evidence, dict):
                raise BudgetLedgerError("terminal receipt must be a JSON object")
            evidence_op = terminal_evidence.get("op")
            evidence_argv = terminal_evidence.get("argv")
            parsed_evidence_status = _parse_kernel_status(
                str(terminal_evidence.get("stdout_scrubbed", ""))
            )
            exact_status = (
                terminal_evidence.get("schema")
                == "celltrack-alpha.cmux-op-receipt.v1"
                and evidence_op == "kernel_status"
                and terminal_evidence.get("returncode") == 0
                and isinstance(evidence_argv, list)
                and len(evidence_argv) >= 4
                and evidence_argv[1:3] == ["kernels", "status"]
                and _status_kernel_ref_binds(evidence_argv[3], kernel_id)
                and parsed_evidence_status == normalized_status
            )
            push_predicate = terminal_evidence.get("kernel_push_predicate")
            push_rc = terminal_evidence.get("returncode")
            push_failed = (
                (type(push_rc) is int and push_rc != 0)
                or (push_rc is None and terminal_evidence.get("timed_out") is True)
            )
            failed_push = (
                normalized_status == "ERROR"
                and terminal_evidence.get("schema")
                == "celltrack-alpha.cmux-op-receipt.v1"
                and evidence_op == "kernel_push"
                and isinstance(evidence_argv, list)
                and len(evidence_argv) >= 3
                and evidence_argv[1:3] == ["kernels", "push"]
                and push_failed
                and isinstance(push_predicate, dict)
                and push_predicate.get("kernel_id") == kernel_id
            )
            if not (exact_status or failed_push):
                raise BudgetLedgerError(
                    "terminal receipt does not prove this kernel and terminal state"
                )
        if quota is not None:
            quota = parse_authoritative_gpu_quota(quota)
        ledger["active_reservations"] = [
            item for item in ledger["active_reservations"]
            if item.get("kernel_id") != kernel_id
        ]
        event = {
            "type": "terminal_reconcile",
            "recorded_utc": _utc(),
            "kernel_id": kernel_id,
            "terminal_status": normalized_status,
            "released_max_gpu_h": float(matches[0]["max_gpu_h"]),
        }
        if terminal_receipt:
            terminal_path = Path(terminal_receipt)
            event["terminal_receipt"] = str(terminal_path)
            if terminal_path.is_file():
                event["terminal_receipt_sha256"] = _sha256_file(terminal_path)
        if quota is not None:
            ledger["latest_authoritative_quota"] = _authoritative_snapshot(
                quota, receipt=quota_receipt
            )
        ledger["events"].append(event)
        ledger["updated_utc"] = _utc()
        _write_gpu_ledger(ledger_path, ledger)
        return event
    finally:
        if owned_lock is not None:
            owned_lock.release()


def bracketed_launch(
    op: str,
    *,
    kaggle: str = "kaggle",
    receipt_dir: Path | str | None = None,
    lock: serial_lock | None = None,
    gpu_reservation_h: float | None = None,
    authenticated_projection_h: float | None = None,
    human_approval_receipt: Path | str | None = None,
    gpu_ledger_path: Path | str = DEFAULT_GPU_LEDGER,
    **kw: Any,
) -> dict[str, Any]:
    """Quota-bracket a mutating op: query, act, query. The delta is the authoritative debit.

    If either quota query fails, the op does not run. Timing your own elapsed seconds is not
    an acceptable substitute -- this campaign already reconciled a 1.65 h accounting drift
    and a 0.74 h launch race that way.
    """
    if op == "dataset_create":
        # Fail before even the read-only quota subprocess. ``run`` repeats this
        # check immediately before the mutation to retain a narrow TOCTOU guard.
        _validate_dataset_create_metadata(
            kw.get("path", ""),
            expected_dataset_id=str(kw.get("dataset_id", "")),
        )

    def _quota() -> tuple[dict[str, Any], dict[str, Any] | None]:
        r = run("quota", kaggle=kaggle, receipt_dir=receipt_dir, **kw)
        if r["returncode"] != 0:
            return r, None
        try:
            return r, parse_authoritative_gpu_quota(json.loads(r["stdout_scrubbed"]))
        except (json.JSONDecodeError, CmuxOpsError):
            return r, None

    owned = lock is None
    lock = lock or serial_lock(node=f"bracketed_launch:{op}", target=str(kw.get("kernel_id") or kw.get("dataset_id") or ""))
    acquired_here = owned or lock._fd is None
    if acquired_here:
        lock.acquire()
    try:
        # The serial lock spans the entire authoritative bracket.  Otherwise another
        # mutation could fall between the pre-reading, action, and post-reading.
        pre_receipt, pre = _quota()
        if pre is None:
            raise CmuxOpsError(
                "quota pre-query failed or was unparseable; refusing to launch without an "
                f"authoritative pre-reading (returncode={pre_receipt['returncode']})"
            )
        reservation = None
        projection_gate = None
        if op == "kernel_push":
            reservation = _reserve_gpu_budget(
                ledger_path=gpu_ledger_path,
                quota=pre,
                kernel_id=str(kw.get("kernel_id", "")),
                max_gpu_h=float(gpu_reservation_h or 0.0),
                quota_receipt=pre_receipt.get("receipt_path"),
            )
            try:
                projection_gate = _projection_gate(
                    authenticated_projection_h, human_approval_receipt
                )
            except Exception:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
                raise
        try:
            action = run(
                op,
                kaggle=kaggle,
                receipt_dir=receipt_dir,
                _held_lock=lock,
                **kw,
            )
        except Exception:
            if reservation is not None:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
            raise
        post_receipt, post = _quota()
        if reservation is not None:
            ledger = _read_gpu_ledger(gpu_ledger_path)
            for item in ledger["active_reservations"]:
                if item.get("kernel_id") == kw.get("kernel_id"):
                    item["status"] = (
                        "LAUNCHED_ACTIVE" if action.get("returncode") == 0
                        else "LAUNCH_FAILED"
                    )
                    item["launch_receipt"] = action.get("receipt_path")
                    if action.get("receipt_sha256"):
                        item["launch_receipt_sha256"] = action["receipt_sha256"]
            if post is not None:
                ledger["latest_authoritative_quota"] = _authoritative_snapshot(
                    post, receipt=post_receipt.get("receipt_path")
                )
            ledger["updated_utc"] = _utc()
            _write_gpu_ledger(gpu_ledger_path, ledger)
            if action.get("returncode") != 0:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    quota=post,
                    quota_receipt=post_receipt.get("receipt_path"),
                    terminal_receipt=action.get("receipt_path"),
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
    finally:
        if acquired_here:
            lock.release()

    delta = None if post is None else round(post["used_h"] - pre["used_h"], 6)

    return {
        "schema": "celltrack-alpha.cmux-bracketed-launch.v1",
        "op": op,
        "quota_pre": pre,
        "quota_post": post,
        "authoritative_debit_h": delta,
        "accounting_basis": "owner-authenticated Kaggle pre/post quota difference",
        "action_receipt": action,
        "quota_pre_receipt": pre_receipt,
        "quota_post_receipt": post_receipt,
        "gpu_reservation_h": gpu_reservation_h if op == "kernel_push" else None,
        "projection_gate": projection_gate if op == "kernel_push" else None,
        "gpu_ledger_path": str(gpu_ledger_path) if op == "kernel_push" else None,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("op", nargs="?", help="op name from command_plan()")
    ap.add_argument("--kaggle", default="kaggle")
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--dataset-id", dest="dataset_id", default="")
    ap.add_argument("--kernel-id", dest="kernel_id", default="")
    ap.add_argument("--kernel-version", dest="kernel_version", type=int, default=None)
    ap.add_argument("--path", default="")
    ap.add_argument("--message", default="")
    ap.add_argument("--competition", default=DEFAULT_COMPETITION)
    ap.add_argument("--csv", default="")
    ap.add_argument("--submission-predicate-receipt", default=None)
    ap.add_argument("--kernel-push-predicate-receipt", default=None)
    ap.add_argument("--bracketed", action="store_true", help="quota-bracket a mutating op")
    ap.add_argument(
        "--gpu-reservation-h",
        type=float,
        default=None,
        help="required worst-case GPU-hour reservation for a bracketed kernel_push",
    )
    ap.add_argument(
        "--authenticated-projection-h",
        type=float,
        default=None,
        help="authenticated whole-run projection for the repository LMT gate",
    )
    ap.add_argument(
        "--human-approval-receipt",
        default=None,
        help="required approval/pitch receipt at or above the LMT autonomous cap",
    )
    ap.add_argument("--gpu-ledger", default=str(DEFAULT_GPU_LEDGER))
    ap.add_argument("--print-plan", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    kw = dict(
        dataset_id=a.dataset_id, kernel_id=a.kernel_id, kernel_version=a.kernel_version, path=a.path,
        message=a.message, competition=a.competition, csv=a.csv,
    )
    if a.print_plan:
        print(json.dumps(command_plan(a.kaggle, **kw), indent=2))
        return 0
    if not a.op:
        ap.error("op is required unless --print-plan is given")

    extra = {
        "submission_predicate_receipt": a.submission_predicate_receipt,
        "kernel_push_predicate_receipt": a.kernel_push_predicate_receipt,
    }
    if a.bracketed:
        result = bracketed_launch(
            a.op,
            kaggle=a.kaggle,
            receipt_dir=a.receipt_dir,
            gpu_reservation_h=a.gpu_reservation_h,
            authenticated_projection_h=a.authenticated_projection_h,
            human_approval_receipt=a.human_approval_receipt,
            gpu_ledger_path=a.gpu_ledger,
            **extra,
            **kw,
        )
    else:
        result = run(a.op, kaggle=a.kaggle, receipt_dir=a.receipt_dir, **extra, **kw)
    print(json.dumps(result, indent=2, sort_keys=True))
    rc = result.get("returncode", result.get("action_receipt", {}).get("returncode", 0))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

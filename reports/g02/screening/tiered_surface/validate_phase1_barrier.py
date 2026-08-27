#!/usr/bin/env python3
"""Fail-closed validation for one G-02 Phase-1 worktree fragment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT_SINGLE_WRITERS = {
    "reports/g02/screening/LIVE_STATE.md",
    "reports/g02/screening/LEDGER.md",
    "reports/g02/screening/tiered_surface/tier_census.json",
    "reports/g02/screening/tiered_surface/kaggle_budget.json",
}
FORBIDDEN_OUTCOME_KEYS = {"cv57", "public", "submission"}
MEMO_ONLY_MARKERS = {
    "-0.010026991115",
    "-0.009715825391",
    "+0.000647668550",
    "+0.002606214347",
    "66/223",
    "157/223",
    "10290/1051/724",
    "10327/958/687",
    "10218/1016/796",
    "32 merges",
    "34 missing endpoints",
    "proposal 1552",
}
REQUIRED_FROZEN_PINS = {
    "n1_interface": "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "grouped_folds": "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "p0": "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
}


def run_git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=worktree,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit(
            f"git {' '.join(args)} failed: {result.stdout}{result.stderr}"
        )
    return result.stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_paths(worktree: Path, base: str) -> list[str]:
    committed = run_git(
        worktree,
        "diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        f"{base}..HEAD",
    ).splitlines()
    working = run_git(
        worktree,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    paths = set(committed)
    for line in working:
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.add(value)
    return sorted(paths)


def walk_values(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            strings.append(str(key))
            strings.extend(walk_values(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(walk_values(child))
    elif value is not None:
        strings.append(str(value))
    return strings


def scan_tokens(worktree: Path, paths: list[str], tokens: set[str]) -> set[str]:
    encoded = {token: token.encode("utf-8") for token in tokens}
    overlap = max((len(value) for value in encoded.values()), default=1) - 1
    found: set[str] = set()
    for relative in paths:
        path = worktree / relative
        if not path.is_file():
            continue
        carry = b""
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                value = carry + chunk
                for token, needle in encoded.items():
                    if token not in found and needle in value:
                        found.add(token)
                carry = value[-overlap:] if overlap else b""
        if found == tokens:
            break
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--allow-prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worktree = args.worktree.resolve()
    branch = run_git(worktree, "branch", "--show-current").strip()
    if branch != args.branch:
        raise SystemExit(f"branch mismatch: {branch}")
    if run_git(worktree, "merge-base", args.base, "HEAD").strip() != args.base:
        raise SystemExit("worktree does not descend from the frozen git base")

    paths = changed_paths(worktree, args.base)
    if not paths:
        raise SystemExit("worktree has no Phase-1 artifact")
    violations = [
        path
        for path in paths
        if path in ROOT_SINGLE_WRITERS or not path.startswith(args.allow_prefix)
    ]
    if violations:
        raise SystemExit(f"write-scope violation: {violations}")
    symlinks = [path for path in paths if (worktree / path).is_symlink()]
    if symlinks:
        raise SystemExit(f"symlinked lane artifact is not allowed: {symlinks}")
    ignored_lane_files = run_git(
        worktree,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        args.allow_prefix,
    ).splitlines()
    generated = [
        path for path in paths if "__pycache__/" in path or path.endswith(".pyc")
    ] + [
        path
        for path in ignored_lane_files
        if "__pycache__/" in path or path.endswith(".pyc")
    ]
    if generated:
        raise SystemExit(f"generated Python cache in lane artifact: {generated}")

    token_hits = scan_tokens(
        worktree,
        paths,
        set(REQUIRED_FROZEN_PINS.values()) | MEMO_ONLY_MARKERS,
    )
    missing_pins = {
        name: digest
        for name, digest in REQUIRED_FROZEN_PINS.items()
        if digest not in token_hits
    }
    if missing_pins:
        raise SystemExit(f"lane did not authenticate frozen inputs: {missing_pins}")

    fragment_path = worktree / args.allow_prefix / "state_fragment.json"
    terminal_path = worktree / args.allow_prefix / "TERMINAL_REPORT.md"
    ledger_path = worktree / args.allow_prefix / "PROPOSED_LEDGER_ROWS.md"
    for required in (fragment_path, terminal_path, ledger_path):
        if not required.is_file():
            raise SystemExit(f"missing required output: {required}")

    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    if not isinstance(fragment, dict):
        raise SystemExit("state fragment must be a JSON object")
    nodes = fragment.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise SystemExit("state fragment must contain at least one node")
    for node in nodes:
        if node.get("tier") not in {"LIVE", "PARK_A", "PARK_B", "PARK_C", "DEAD"}:
            raise SystemExit(f"invalid proposed tier: {node.get('id')}")
        if node.get("tier") == "DEAD" and (
            not node.get("human_authorized") or not node.get("named_mechanism")
        ):
            raise SystemExit(f"unauthorized DEAD proposal: {node.get('id')}")
        evidence = node.get("evidence_artifact")
        evidence_hash = node.get("evidence_sha256")
        if not evidence or not evidence_hash:
            raise SystemExit(f"uncited tier state: {node.get('id')}")
        if not str(evidence).startswith(args.allow_prefix):
            raise SystemExit(f"tier evidence escapes lane scope: {node.get('id')}")
        evidence_path = worktree / evidence
        if not evidence_path.is_file() or sha256(evidence_path) != evidence_hash:
            raise SystemExit(f"unresolvable tier evidence: {node.get('id')}")

    flat = walk_values(fragment)
    lowered = {item.lower() for item in flat}
    forbidden_scope = fragment.get("forbidden_scope")
    if not isinstance(forbidden_scope, dict):
        raise SystemExit("fragment lacks an explicit forbidden-scope attestation")
    if (
        forbidden_scope.get("cv57_used") is not False
        or forbidden_scope.get("submission_action") is not False
        or forbidden_scope.get("runpod_action") is not False
    ):
        raise SystemExit("forbidden downstream scope is not explicitly false")
    forbidden_true_fields = sorted(
        key
        for key, value in forbidden_scope.items()
        if value is True
        and any(
            token in key.lower()
            for token in (
                "cv57",
                "public",
                "submission",
                "runpod",
                "external",
                "gpu",
                "kaggle",
                "dead",
                "plan_amendment",
                "sibling",
            )
        )
    )
    if forbidden_true_fields:
        raise SystemExit(
            f"fragment reports prohibited actions: {forbidden_true_fields}"
        )
    if any(key in item for key in FORBIDDEN_OUTCOME_KEYS for item in lowered):
        if (
            forbidden_scope["cv57_used"] is not False
            or forbidden_scope["submission_action"] is not False
            or forbidden_scope["runpod_action"] is not False
        ):
            raise SystemExit("forbidden downstream outcome appears in fragment")
    if args.allow_prefix.rstrip("/").endswith("sa_swap"):
        if forbidden_scope.get("sibling_live_output_used") is not False:
            raise SystemExit("SA-SWAP did not explicitly exclude live SA-REPR input")
        population = fragment.get("n1_population_validation", {})
        candidate_count = population.get("candidate_count")
        witness_count = population.get("witness_compile_count")
        if (
            not isinstance(candidate_count, int)
            or candidate_count <= 0
            or witness_count != candidate_count
            or population.get("reject_count") != 0
            or population.get("full_baseline_final_batch_compiled") is not True
        ):
            raise SystemExit("SA-SWAP lacks exhaustive frozen-N1 population proof")
        proof_artifact = population.get("proof_artifact")
        proof_sha256 = population.get("proof_sha256")
        proof_path = worktree / str(proof_artifact)
        if (
            not proof_artifact
            or not proof_sha256
            or not proof_path.is_file()
            or sha256(proof_path) != proof_sha256
        ):
            raise SystemExit("SA-SWAP N1 population proof is not hash-resolvable")
    memo_hits = sorted(
        marker for marker in MEMO_ONLY_MARKERS if marker in token_hits
    )
    if memo_hits and fragment.get("memo_quarantine") is not True:
        raise SystemExit(f"memo-only numbers are not quarantined: {memo_hits}")
    forbidden_memo_surfaces = "\n".join(
        (
            json.dumps(fragment.get("nodes", []), sort_keys=True),
            ledger_path.read_text(encoding="utf-8"),
        )
    )
    tiering_hits = sorted(
        marker for marker in MEMO_ONLY_MARKERS if marker in forbidden_memo_surfaces
    )
    if tiering_hits:
        raise SystemExit(f"memo-only numbers reached tier/ledger surfaces: {tiering_hits}")

    family = fragment.get("family_execution", {})
    if family.get("6bba_ran"):
        gate = family.get("family_44b6_gate", {})
        if not (
            gate.get("status") == "GO"
            and gate.get("adjusted_delta", float("-inf")) >= 0.005
            and gate.get("raw_delta", float("-inf")) >= 0
            and gate.get("node_delta", float("-inf")) >= 0
        ):
            raise SystemExit("6bba ran without a qualifying 44b6 gate")

    result = {
        "schema": "celltrack-alpha.phase1-worktree-validation.v1",
        "status": "PASS",
        "worktree": str(worktree),
        "branch": branch,
        "head_commit": run_git(worktree, "rev-parse", "HEAD").strip(),
        "base": args.base,
        "allow_prefix": args.allow_prefix,
        "changed_paths": paths,
        "fragment_sha256": sha256(fragment_path),
        "terminal_report_sha256": sha256(terminal_path),
        "proposed_ledger_sha256": sha256(ledger_path),
        "memo_markers_present_and_quarantined": memo_hits,
        "memo_markers_absent_from_tier_and_ledger_surfaces": True,
        "required_frozen_pins_present": REQUIRED_FROZEN_PINS,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

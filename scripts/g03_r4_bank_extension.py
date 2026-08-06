#!/usr/bin/env python3
"""Hash-gated, target-blind R4 association-bank extension.

The development command reuses the promoted R1 implementation for the exact
fit105/6bba ``dev69`` population.  It exposes no scoring command.  The hidden
test command is a separate downstream stage and validates a local-confirmation
gate receipt before it performs any population, cache, model, or raw-data work.

Production execution requires a root-authored promotion receipt with this
score-free shape (paths are supplied separately on the command line)::

    {
      "schema": "celltrack-alpha.g03-r4-r1-promotion.v1",
      "status": "R1_FINALIZED_AND_HASHES_PROMOTED_BY_ROOT",
      "r1_finalized": true,
      "artifacts": {
        "contract_sha256": "...",
        "decision_amendment_sha256": "...",
        "implementation_sha256": "...",
        "adapter_file_sha256": "...",
        "adapter_state_tensor_sha256": "...",
        "checkpoint_model_sha256": "...",
        "checkpoint_archive_sha256": "...",
        "trackastra_source_tree_sha256": "...",
        "preflight_sha256": "...",
        "inference_seal_sha256": "...",
        "held_fit_association_bank_sha256": "...",
        "held_fit_parent_graph_sha256": "...",
        "validated_result_sha256": "...",
        "validation_receipt_sha256": "..."
      },
      "gt_access_by_r4": false,
      "score_read_by_r4": false,
      "score_values_in_receipt": false
    }

The two result hashes are bound as opaque identities.  This module never opens
the R1 result, held-fit score, ground truth, or a scorer output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from types import ModuleType
from typing import Any


POPULATION_SCHEMA = "celltrack-alpha.g03-r4-division-population.v1"
POPULATION_AMENDMENT_SCHEMA = (
    "celltrack-alpha.g03-r4-division-population-amendment.v1"
)
PROMOTION_SCHEMA = "celltrack-alpha.g03-r4-r1-promotion.v1"
LOCAL_CONFIRMATION_SCHEMA = "celltrack-alpha.g03-r4-local-confirmation-gate.v1"
R1_CONTRACT_SCHEMA = "celltrack-alpha.g03-r1-zero-shot-contract.v1"
R1_AMENDMENT_SCHEMA = "celltrack-alpha.g03-r1-decision-reference-amendment.v1"
R1_PREFLIGHT_SCHEMA = "celltrack-alpha.g03-r1-preflight.v1"
R1_INFERENCE_SEAL_SCHEMA = "celltrack-alpha.g03-r1-inference-seal.v1"
R4_PREFLIGHT_SCHEMA = "celltrack-alpha.g03-r4-bank-extension-preflight.v1"
VIDEO_SCHEMA = "celltrack-alpha.g03-r4-bank-extension-video.v1"
DEV_SEAL_SCHEMA = "celltrack-alpha.g03-r4-dev69-bank-seal.v1"
HIDDEN_SEAL_SCHEMA = "celltrack-alpha.g03-r4-hidden4-bank-seal.v1"

EXPECTED_POPULATION_FILE_SHA256 = (
    "b58fe1acd794c35c9a2f8e63e65327f74d5877cff223ed122a19b68f9b309f47"
)
EXPECTED_POPULATION_SHA256 = (
    "e25416da1d240ceabf2b1457ceb6625b361523e7a537ed2f4ac195280b425954"
)
EXPECTED_POPULATION_AMENDMENT_SHA256 = (
    "bc09d4420c5ed5d66e93b1e5fe5debe1e1ec0e4c8999dacfbc35979e7d3b981d"
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = re.compile(r"^(train|test):(44b6|6bba)_[0-9a-f]{8}$")
SCIENTIFIC_CONFIGURATION_KEYS = ("model_config", "adapter", "compiler", "runtime")


class R4BankError(RuntimeError):
    """An integrity or authorization check failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R4BankError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    require(bool(HEX64.fullmatch(expected)), f"{label}: invalid expected SHA-256")
    require(path.is_file() and not path.is_symlink(), f"{label}: missing regular file")
    observed = sha256(path)
    require(observed == expected, f"{label}: SHA-256 mismatch: {observed}")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def scientific_configuration(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact scientific fields while excluding relocatable file paths."""
    adapter = dict(contract["adapter"])
    adapter.pop("path", None)
    return {
        "model_config": contract["model_config"],
        "adapter": adapter,
        "compiler": contract["compiler"],
        "runtime": contract["runtime"],
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label}: missing regular file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R4BankError(f"{label}: invalid JSON") from error
    require(isinstance(value, dict), f"{label}: expected JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    require(not path.exists(), f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def tree_sha256(root: Path) -> str:
    require(root.is_dir() and not root.is_symlink(), "Trackastra source root is invalid")
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    require(bool(paths), "Trackastra source tree is empty")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _hash_field(value: Any, label: str) -> str:
    text = str(value)
    require(bool(HEX64.fullmatch(text)), f"{label}: missing SHA-256")
    return text


def slug(identity: str) -> str:
    require(bool(IDENTITY.fullmatch(identity)), f"malformed identity: {identity}")
    return identity.replace(":", "__", 1)


@dataclass(frozen=True)
class R1Binding:
    contract: dict[str, Any]
    contract_sha256: str
    amendment_sha256: str
    implementation_sha256: str
    adapter_file_sha256: str
    adapter_state_tensor_sha256: str
    checkpoint_model_sha256: str
    checkpoint_archive_sha256: str
    source_tree_sha256: str
    preflight_sha256: str
    inference_seal_sha256: str
    held_fit_association_bank_sha256: str
    held_fit_parent_graph_sha256: str
    validated_result_sha256: str
    validation_receipt_sha256: str
    promotion_sha256: str
    scientific_configuration_sha256: str
    binding_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "decision_amendment_sha256": self.amendment_sha256,
            "implementation_sha256": self.implementation_sha256,
            "adapter_file_sha256": self.adapter_file_sha256,
            "adapter_state_tensor_sha256": self.adapter_state_tensor_sha256,
            "checkpoint_model_sha256": self.checkpoint_model_sha256,
            "checkpoint_archive_sha256": self.checkpoint_archive_sha256,
            "trackastra_source_tree_sha256": self.source_tree_sha256,
            "preflight_sha256": self.preflight_sha256,
            "inference_seal_sha256": self.inference_seal_sha256,
            "held_fit_association_bank_sha256": (
                self.held_fit_association_bank_sha256
            ),
            "held_fit_parent_graph_sha256": self.held_fit_parent_graph_sha256,
            "validated_result_sha256": self.validated_result_sha256,
            "validation_receipt_sha256": self.validation_receipt_sha256,
            "promotion_sha256": self.promotion_sha256,
            "scientific_configuration_sha256": (
                self.scientific_configuration_sha256
            ),
            "binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True)
class Population:
    manifest_sha256: str
    amendment_sha256: str
    population_sha256: str
    development: tuple[str, ...]
    held_fit: tuple[str, ...]
    hidden: tuple[str, ...]
    development_fold_by_identity: Mapping[str, int]
    group_by_identity: Mapping[str, str]


def validate_r1_binding(
    *,
    promotion_path: Path,
    promotion_sha256: str,
    contract_path: Path,
    amendment_path: Path,
    implementation_path: Path,
    adapter_path: Path,
    checkpoint_model_path: Path,
    checkpoint_archive_path: Path,
    source_root: Path,
    preflight_path: Path,
    inference_seal_path: Path,
) -> R1Binding:
    """Validate only score-free R1 provenance and return its content binding."""
    require_hash(promotion_path, promotion_sha256, "R1 promotion receipt")
    promotion = load_json(promotion_path, "R1 promotion receipt")
    require(promotion.get("schema") == PROMOTION_SCHEMA, "R1 promotion schema drifted")
    require(
        set(promotion)
        == {
            "schema",
            "status",
            "r1_finalized",
            "artifacts",
            "gt_access_by_r4",
            "score_read_by_r4",
            "score_values_in_receipt",
        },
        "R1 promotion receipt has undeclared fields",
    )
    require(
        promotion.get("status") == "R1_FINALIZED_AND_HASHES_PROMOTED_BY_ROOT",
        "R1 has not been finalized and promoted by root",
    )
    require(promotion.get("r1_finalized") is True, "R1 finalization is not affirmative")
    require(promotion.get("gt_access_by_r4") is False, "promotion claims R4 GT access")
    require(promotion.get("score_read_by_r4") is False, "promotion claims R4 score read")
    require(
        promotion.get("score_values_in_receipt") is False,
        "promotion receipt must be score-free",
    )
    artifacts = promotion.get("artifacts")
    require(isinstance(artifacts, Mapping), "promotion artifacts are missing")
    require(
        set(artifacts)
        == {
            "contract_sha256",
            "decision_amendment_sha256",
            "implementation_sha256",
            "adapter_file_sha256",
            "adapter_state_tensor_sha256",
            "checkpoint_model_sha256",
            "checkpoint_archive_sha256",
            "trackastra_source_tree_sha256",
            "preflight_sha256",
            "inference_seal_sha256",
            "held_fit_association_bank_sha256",
            "held_fit_parent_graph_sha256",
            "validated_result_sha256",
            "validation_receipt_sha256",
        },
        "R1 promotion artifact set drifted",
    )

    contract_sha = _hash_field(artifacts.get("contract_sha256"), "R1 contract")
    amendment_sha = _hash_field(
        artifacts.get("decision_amendment_sha256"), "R1 amendment"
    )
    implementation_sha = _hash_field(
        artifacts.get("implementation_sha256"), "R1 implementation"
    )
    adapter_sha = _hash_field(
        artifacts.get("adapter_file_sha256"), "R1 adapter file"
    )
    adapter_state_sha = _hash_field(
        artifacts.get("adapter_state_tensor_sha256"), "R1 adapter state"
    )
    checkpoint_model_sha = _hash_field(
        artifacts.get("checkpoint_model_sha256"), "R1 checkpoint model"
    )
    checkpoint_archive_sha = _hash_field(
        artifacts.get("checkpoint_archive_sha256"), "R1 checkpoint archive"
    )
    source_sha = _hash_field(
        artifacts.get("trackastra_source_tree_sha256"), "Trackastra source tree"
    )
    preflight_sha = _hash_field(artifacts.get("preflight_sha256"), "R1 preflight")
    inference_seal_sha = _hash_field(
        artifacts.get("inference_seal_sha256"), "R1 inference seal"
    )
    held_bank_sha = _hash_field(
        artifacts.get("held_fit_association_bank_sha256"),
        "R1 held-fit association-bank aggregate",
    )
    held_parent_sha = _hash_field(
        artifacts.get("held_fit_parent_graph_sha256"),
        "R1 held-fit parent-graph aggregate",
    )
    result_sha = _hash_field(
        artifacts.get("validated_result_sha256"), "validated R1 result"
    )
    validation_sha = _hash_field(
        artifacts.get("validation_receipt_sha256"), "R1 validation receipt"
    )

    for path, wanted, label in (
        (contract_path, contract_sha, "R1 contract"),
        (amendment_path, amendment_sha, "R1 decision amendment"),
        (implementation_path, implementation_sha, "R1 implementation"),
        (adapter_path, adapter_sha, "R1 adapter"),
        (checkpoint_model_path, checkpoint_model_sha, "Trackastra model"),
        (checkpoint_archive_path, checkpoint_archive_sha, "Trackastra archive"),
        (preflight_path, preflight_sha, "R1 preflight"),
        (inference_seal_path, inference_seal_sha, "R1 inference seal"),
    ):
        require_hash(path, wanted, label)
    require(tree_sha256(source_root) == source_sha, "Trackastra source tree SHA-256 mismatch")

    contract = load_json(contract_path, "R1 contract")
    require(contract.get("schema") == R1_CONTRACT_SCHEMA, "R1 contract schema drifted")
    require(
        contract.get("status") == "SEALED_BEFORE_HELD_FIT_SCORE_READ",
        "R1 contract was not sealed before scoring",
    )
    require(
        contract.get("implementation_sha256") == implementation_sha,
        "R1 contract does not bind implementation",
    )
    scope = contract.get("scope", {})
    require(scope.get("training") is False, "R1 contract permits training")
    require(scope.get("gpu") is False, "R1 contract permits GPU")
    require(scope.get("kaggle") is False, "R1 contract permits Kaggle")
    require(scope.get("cv57") is False, "R1 contract permits CV57")
    require(scope.get("submission") is False, "R1 contract permits submission")
    require(
        contract.get("compiler", {}).get("divisions") is False,
        "R1 parent compiler is not DIVISIONS_OFF",
    )
    require(contract.get("runtime", {}).get("device") == "cpu", "R1 runtime is not CPU")
    adapter = contract.get("adapter", {})
    checkpoint = contract.get("inputs", {}).get("trackastra_checkpoint", {})
    require(adapter.get("file_sha256") == adapter_sha, "contract adapter hash drifted")
    require(
        adapter.get("state_tensor_sha256") == adapter_state_sha,
        "contract adapter-state hash drifted",
    )
    require(int(adapter.get("training_steps", -1)) == 0, "R1 adapter was trained")
    require(
        checkpoint.get("model_sha256") == checkpoint_model_sha,
        "contract checkpoint-model hash drifted",
    )
    require(
        checkpoint.get("archive_sha256") == checkpoint_archive_sha,
        "contract checkpoint-archive hash drifted",
    )
    require(
        contract.get("inputs", {}).get("trackastra_source_tree_sha256") == source_sha,
        "contract source-tree hash drifted",
    )

    amendment = load_json(amendment_path, "R1 decision amendment")
    require(amendment.get("schema") == R1_AMENDMENT_SCHEMA, "R1 amendment schema drifted")
    require(
        amendment.get("status") == "SEALED_BEFORE_HELD_FIT_SCORE_READ",
        "R1 amendment was not sealed before scoring",
    )
    require(
        amendment.get("amends_contract_sha256") == contract_sha,
        "R1 amendment targets another contract",
    )
    require(
        amendment.get("scientific_configuration_changed") is False,
        "R1 amendment changed the scientific configuration",
    )
    integrity = amendment.get("integrity", {})
    require(integrity.get("ground_truth_access") is False, "R1 amendment reports GT access")
    require(
        integrity.get("held_fit_score_read") is False,
        "R1 amendment reports a pre-amendment score read",
    )

    preflight = load_json(preflight_path, "R1 preflight")
    require(preflight.get("schema") == R1_PREFLIGHT_SCHEMA, "R1 preflight schema drifted")
    require(
        preflight.get("status") == "PASS_SCORE_BLIND_PREFLIGHT",
        "R1 preflight did not pass score-blind validation",
    )
    require(
        preflight.get("contract", {}).get("sha256") == contract_sha,
        "R1 preflight contract hash drifted",
    )
    require(preflight.get("implementation_sha256") == implementation_sha, "R1 preflight implementation drifted")
    require(preflight.get("adapter", {}).get("file_sha256") == adapter_sha, "R1 preflight adapter drifted")
    require(
        preflight.get("adapter", {}).get("state_tensor_sha256") == adapter_state_sha,
        "R1 preflight adapter state drifted",
    )
    require(
        preflight.get("trackastra_checkpoint", {}).get("model_sha256")
        == checkpoint_model_sha,
        "R1 preflight checkpoint model drifted",
    )
    require(
        preflight.get("trackastra_checkpoint", {}).get("archive_sha256")
        == checkpoint_archive_sha,
        "R1 preflight checkpoint archive drifted",
    )
    require(
        preflight.get("trackastra_source_tree_sha256") == source_sha,
        "R1 preflight source tree drifted",
    )
    require(preflight.get("gt_access") is False, "R1 preflight reports GT access")
    require(
        preflight.get("held_fit_score_read") is False,
        "R1 preflight reports held-fit score read",
    )

    inference_seal = load_json(inference_seal_path, "R1 inference seal")
    require(
        inference_seal.get("schema") == R1_INFERENCE_SEAL_SCHEMA,
        "R1 inference-seal schema drifted",
    )
    require(
        inference_seal.get("status") == "SEALED_ALL_36_BEFORE_SCORE_READ",
        "R1 held-fit inference was not completely sealed before scoring",
    )
    require(
        inference_seal.get("contract", {}).get("sha256") == contract_sha,
        "R1 inference-seal contract drifted",
    )
    r1_population = inference_seal.get("population", {})
    require(
        r1_population.get("family") == "44b6"
        and r1_population.get("identity_count") == 36,
        "R1 inference seal is not exact held-fit36",
    )
    require(
        tuple(r1_population.get("identities", ()))
        == tuple(contract.get("scope", {}).get("identities", ())),
        "R1 inference-seal population drifted",
    )
    require(
        inference_seal.get("aggregate", {}).get("association_bank_sha256")
        == held_bank_sha,
        "promotion held-fit bank hash does not match R1 inference seal",
    )
    require(
        inference_seal.get("aggregate", {}).get("fragment_sha256")
        == held_parent_sha,
        "promotion held-fit parent hash does not match R1 inference seal",
    )
    require(
        inference_seal.get("aggregate", {}).get("predicted_forks") == 0,
        "R1 held-fit parent graphs are not DIVISIONS_OFF",
    )
    require(inference_seal.get("gt_access") is False, "R1 inference seal reports GT access")
    require(
        inference_seal.get("held_fit_score_read") is False,
        "R1 inference seal was not pre-score",
    )

    scientific_sha = canonical_sha256(scientific_configuration(contract))
    binding_payload = {
        "contract_sha256": contract_sha,
        "decision_amendment_sha256": amendment_sha,
        "implementation_sha256": implementation_sha,
        "adapter_file_sha256": adapter_sha,
        "adapter_state_tensor_sha256": adapter_state_sha,
        "checkpoint_model_sha256": checkpoint_model_sha,
        "checkpoint_archive_sha256": checkpoint_archive_sha,
        "trackastra_source_tree_sha256": source_sha,
        "preflight_sha256": preflight_sha,
        "inference_seal_sha256": inference_seal_sha,
        "held_fit_association_bank_sha256": held_bank_sha,
        "held_fit_parent_graph_sha256": held_parent_sha,
        "validated_result_sha256": result_sha,
        "validation_receipt_sha256": validation_sha,
        "promotion_sha256": promotion_sha256,
        "scientific_configuration_sha256": scientific_sha,
    }
    return R1Binding(
        contract=contract,
        contract_sha256=contract_sha,
        amendment_sha256=amendment_sha,
        implementation_sha256=implementation_sha,
        adapter_file_sha256=adapter_sha,
        adapter_state_tensor_sha256=adapter_state_sha,
        checkpoint_model_sha256=checkpoint_model_sha,
        checkpoint_archive_sha256=checkpoint_archive_sha,
        source_tree_sha256=source_sha,
        preflight_sha256=preflight_sha,
        inference_seal_sha256=inference_seal_sha,
        held_fit_association_bank_sha256=held_bank_sha,
        held_fit_parent_graph_sha256=held_parent_sha,
        validated_result_sha256=result_sha,
        validation_receipt_sha256=validation_sha,
        promotion_sha256=promotion_sha256,
        scientific_configuration_sha256=scientific_sha,
        binding_sha256=canonical_sha256(binding_payload),
    )


def validate_population_values(
    manifest: Mapping[str, Any],
    amendment: Mapping[str, Any],
    *,
    manifest_sha256: str,
    amendment_sha256: str,
    expected_population_sha256: str,
) -> Population:
    """Validate exact role membership without reading labels or target artifacts."""
    require(manifest.get("schema") == POPULATION_SCHEMA, "population schema drifted")
    require(
        manifest.get("status") == "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population is not pre-outcome locked",
    )
    require(
        manifest.get("population_sha256") == expected_population_sha256,
        "population membership SHA-256 drifted",
    )
    roles = manifest.get("roles")
    require(isinstance(roles, Mapping), "population roles are missing")

    def identities(role: str, count: int, prefix: str) -> tuple[str, ...]:
        record = roles.get(role)
        require(isinstance(record, Mapping), f"population role missing: {role}")
        videos = record.get("videos")
        require(isinstance(videos, list), f"population role videos malformed: {role}")
        values = tuple(str(row.get("identity")) for row in videos if isinstance(row, Mapping))
        require(len(values) == count == int(record.get("video_count", -1)), f"{role}: count drifted")
        require(len(set(values)) == count, f"{role}: duplicate identity")
        require(all(value.startswith(prefix) for value in values), f"{role}: family/split drifted")
        require(all(IDENTITY.fullmatch(value) for value in values), f"{role}: malformed identity")
        return values

    development = identities("development_nested_oof", 69, "train:6bba_")
    held_fit = identities("held_fit_one_shot_confirmation", 36, "train:44b6_")
    hidden = identities("sealed_hidden_test_probe", 4, "test:")
    require(set(development).isdisjoint(held_fit), "development overlaps held-fit")
    require(set(development).isdisjoint(hidden), "development overlaps hidden test")
    require(set(held_fit).isdisjoint(hidden), "held-fit overlaps hidden test")

    folds = manifest.get("outer_folds")
    require(isinstance(folds, Mapping) and set(folds) == set(map(str, range(5))), "outer folds drifted")
    fold_by_identity: dict[str, int] = {}
    group_by_identity: dict[str, str] = {}
    fold_sizes = []
    for fold_text in map(str, range(5)):
        rows = folds[fold_text]
        require(isinstance(rows, list), f"fold {fold_text}: malformed rows")
        fold_sizes.append(len(rows))
        for row in rows:
            require(isinstance(row, Mapping), f"fold {fold_text}: malformed row")
            identity = str(row.get("identity"))
            group_id = str(row.get("group_id"))
            require(identity not in fold_by_identity, f"fold identity repeats: {identity}")
            require(group_id and group_id != "None", f"fold group missing: {identity}")
            fold_by_identity[identity] = int(fold_text)
            group_by_identity[identity] = group_id
    require(set(fold_by_identity) == set(development), "folds do not exactly cover dev69")
    require(len(set(group_by_identity.values())) == 69, "development groups leak across videos")
    require(sorted(fold_sizes) == [13, 13, 14, 14, 15], "dev69 fold sizes drifted")

    constraints = manifest.get("constraints", {})
    require(constraints.get("division_labels_used_for_split") is False, "population split used division labels")
    require(
        constraints.get("all_association_banks_use_one_R1_scientific_configuration")
        is True,
        "population permits multiple R1 configurations",
    )
    require(constraints.get("development_bank_count_required") == 69, "dev bank requirement drifted")
    require(constraints.get("held_fit_bank_count_required") == 36, "held bank requirement drifted")
    require(constraints.get("hidden_test_bank_count_required") == 4, "hidden bank requirement drifted")
    require(constraints.get("contaminated_audit20_bank_count_required") == 0, "audit20 bank access is not forbidden")

    require(amendment.get("schema") == POPULATION_AMENDMENT_SCHEMA, "population amendment schema drifted")
    require(
        amendment.get("status") == "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "population amendment is not sealed pre-outcome",
    )
    population_record = amendment.get("population_manifest", {})
    require(
        population_record.get("file_sha256") == manifest_sha256,
        "amendment does not bind population file",
    )
    require(
        population_record.get("population_sha256") == expected_population_sha256,
        "amendment does not bind population membership",
    )
    scope = amendment.get("association_bank_scope", {})
    require(scope.get("development_bank_count") == 69, "amended dev bank count drifted")
    require(scope.get("held_fit_bank_count") == 36, "amended held bank count drifted")
    require(scope.get("hidden_test_bank_count") == 4, "amended hidden bank count drifted")
    require(scope.get("contaminated_audit20_bank_count") == 0, "amendment permits audit20 banks")
    require(
        scope.get("development_and_held_fit_bank_hashes_sealed_before_development_scoring")
        is True,
        "amendment no longer seals local banks before scoring",
    )
    require(
        scope.get("hidden_test_inference_only_after_local_confirmation_gate") is True,
        "hidden-test inference is not confirmation-gated",
    )
    return Population(
        manifest_sha256=manifest_sha256,
        amendment_sha256=amendment_sha256,
        population_sha256=expected_population_sha256,
        development=development,
        held_fit=held_fit,
        hidden=hidden,
        development_fold_by_identity=fold_by_identity,
        group_by_identity=group_by_identity,
    )


def validate_population(manifest_path: Path, amendment_path: Path) -> Population:
    require_hash(manifest_path, EXPECTED_POPULATION_FILE_SHA256, "R4 population manifest")
    require_hash(
        amendment_path,
        EXPECTED_POPULATION_AMENDMENT_SHA256,
        "R4 population amendment",
    )
    return validate_population_values(
        load_json(manifest_path, "R4 population manifest"),
        load_json(amendment_path, "R4 population amendment"),
        manifest_sha256=EXPECTED_POPULATION_FILE_SHA256,
        amendment_sha256=EXPECTED_POPULATION_AMENDMENT_SHA256,
        expected_population_sha256=EXPECTED_POPULATION_SHA256,
    )


def verify_cache_population(binding: R1Binding, identities: Sequence[str]) -> dict[str, int]:
    """Authenticate target-free sparse detections for one exact role."""
    cache_record = binding.contract.get("inputs", {}).get("cache_manifest", {})
    cache_path = Path(str(cache_record.get("path", "")))
    cache_sha = _hash_field(cache_record.get("sha256"), "R1 cache manifest")
    require_hash(cache_path, cache_sha, "R1 cache manifest")
    manifest = load_json(cache_path, "R1 cache manifest")
    files = [
        row
        for row in manifest.get("files", [])
        if isinstance(row, Mapping) and str(row.get("identity")) in set(identities)
    ]
    require({str(row.get("identity")) for row in files} == set(identities), "cache misses role identities")
    root = Path(str(binding.contract.get("inputs", {}).get("repository_root", "")))
    for row in files:
        relative = str(row.get("path", ""))
        require(relative and not Path(relative).is_absolute(), "cache path escaped repository")
        path = root / relative
        require(path.stat().st_size == int(row.get("bytes", -1)), f"cache size drifted: {relative}")
        require_hash(path, _hash_field(row.get("sha256"), relative), relative)
    return {"identity_count": len(identities), "file_count": len(files)}


def create_preflight(
    *,
    output: Path,
    binding: R1Binding,
    population: Population,
) -> dict[str, Any]:
    cache = verify_cache_population(binding, population.development)
    raw = binding.contract.get("inputs", {}).get("raw_archive", {})
    raw_path = Path(str(raw.get("path", "")))
    require(raw_path.is_file() and not raw_path.is_symlink(), "R1 raw archive is missing")
    require(raw_path.stat().st_size == int(raw.get("bytes", -1)), "R1 raw archive size drifted")
    report = {
        "schema": R4_PREFLIGHT_SCHEMA,
        "status": "PASS_TARGET_BLIND_DEV69_PREFLIGHT",
        "r4_bank_extension_implementation_sha256": sha256(Path(__file__).resolve()),
        "population": {
            "manifest_sha256": population.manifest_sha256,
            "amendment_sha256": population.amendment_sha256,
            "population_sha256": population.population_sha256,
            "role": "development_nested_oof",
            "video_count": 69,
            "identity_sha256": canonical_sha256(sorted(population.development)),
        },
        "r1_binding": binding.receipt(),
        "cache": cache,
        "raw_archive": {
            "sha256": raw.get("sha256"),
            "bytes": int(raw.get("bytes", -1)),
            "verified_by_r1_preflight_sha256": binding.preflight_sha256,
            "current_size_verified": True,
            "raw_members_opened": False,
        },
        "scientific_configuration": {
            key: value
            for key, value in scientific_configuration(binding.contract).items()
        },
        "gt_access": False,
        "score_read": False,
        "test_raw_members_opened": False,
        "gpu_hours": 0.0,
    }
    atomic_json(output, report)
    return report


def validate_preflight(
    path: Path,
    expected_sha256: str,
    binding: R1Binding,
    population: Population,
) -> dict[str, Any]:
    require_hash(path, expected_sha256, "R4 dev69 preflight")
    report = load_json(path, "R4 dev69 preflight")
    require(report.get("schema") == R4_PREFLIGHT_SCHEMA, "R4 preflight schema drifted")
    require(
        report.get("status") == "PASS_TARGET_BLIND_DEV69_PREFLIGHT",
        "R4 dev69 preflight did not pass",
    )
    require(
        report.get("r4_bank_extension_implementation_sha256")
        == sha256(Path(__file__).resolve()),
        "R4 bank-extension implementation drifted after preflight",
    )
    require(report.get("r1_binding", {}).get("binding_sha256") == binding.binding_sha256, "R4 preflight R1 binding drifted")
    require(report.get("population", {}).get("population_sha256") == population.population_sha256, "R4 preflight population drifted")
    require(report.get("population", {}).get("video_count") == 69, "R4 preflight count drifted")
    require(report.get("gt_access") is False, "R4 preflight reports GT access")
    require(report.get("score_read") is False, "R4 preflight reports score read")
    require(report.get("test_raw_members_opened") is False, "R4 preflight opened test raw members")
    return report


def load_r1_module(path: Path, expected_sha256: str) -> ModuleType:
    require_hash(path, expected_sha256, "R1 implementation")
    specification = importlib.util.spec_from_file_location(
        f"g03_r1_promoted_{expected_sha256[:12]}", path
    )
    require(specification is not None and specification.loader is not None, "cannot load R1 implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for symbol in (
        "configure_runtime",
        "load_trackastra",
        "load_adapter",
        "run_video",
        "graph_from_arrays",
        "write_association_bank",
        "write_graph",
        "ZipStore",
        "zarr",
    ):
        require(hasattr(module, symbol), f"R1 implementation lacks {symbol}")
    return module


def relocated_contract(
    binding: R1Binding,
    *,
    adapter_path: Path,
    checkpoint_model_path: Path,
    checkpoint_archive_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Relocate only content-addressed paths; scientific values remain byte-equal."""
    value = json.loads(json.dumps(binding.contract))
    value["adapter"]["path"] = str(adapter_path.resolve())
    checkpoint = value["inputs"]["trackastra_checkpoint"]
    checkpoint["model_path"] = str(checkpoint_model_path.resolve())
    checkpoint["archive_path"] = str(checkpoint_archive_path.resolve())
    value["inputs"]["trackastra_source_root"] = str(source_root.resolve())
    require(
        canonical_sha256(scientific_configuration(value))
        == binding.scientific_configuration_sha256,
        "content-addressed relocation changed the R1 scientific configuration",
    )
    return value


def aggregate_artifact_sha(records: Sequence[Mapping[str, Any]], field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda value: str(value["identity"])):
        digest.update(str(row["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row[field]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_video_fields(
    *,
    identity: str,
    role: str,
    fold: int | None,
    binding: R1Binding,
    population: Population,
    authorization_receipt_sha256: str,
    authorization_receipt_kind: str,
) -> dict[str, Any]:
    return {
        "schema": VIDEO_SCHEMA,
        "identity": identity,
        "role": role,
        "outer_fold": fold,
        "population_sha256": population.population_sha256,
        "population_manifest_sha256": population.manifest_sha256,
        "r1_binding_sha256": binding.binding_sha256,
        "r1_contract_sha256": binding.contract_sha256,
        "r1_scientific_configuration_sha256": binding.scientific_configuration_sha256,
        "adapter_file_sha256": binding.adapter_file_sha256,
        "checkpoint_model_sha256": binding.checkpoint_model_sha256,
        "authorization_receipt_sha256": authorization_receipt_sha256,
        "authorization_receipt_kind": authorization_receipt_kind,
        "association_bank_target_free": True,
        "association_bank_complete": True,
        "parent_graph_mode": "DIVISIONS_OFF",
        "parent_graph_divisions": False,
        "predicted_forks": 0,
        "sealed_before_R4_gt_or_scorer_access": True,
        "gt_access": False,
        "score_read": False,
        "gpu_hours": 0.0,
    }


def completed_video(
    video_dir: Path,
    *,
    identity: str,
    role: str,
    fold: int | None,
    binding: R1Binding,
    population: Population,
    authorization_receipt_sha256: str,
    authorization_receipt_kind: str,
) -> dict[str, Any] | None:
    if not video_dir.exists():
        return None
    require(video_dir.is_dir() and not video_dir.is_symlink(), f"partial video state: {identity}")
    bank = video_dir / "association_bank.npz"
    graph = video_dir / "DIVISIONS_OFF_PARENT.csv"
    receipt_path = video_dir / "RECEIPT.json"
    require(
        all(path.is_file() and not path.is_symlink() for path in (bank, graph, receipt_path)),
        f"partial video state: {identity}",
    )
    require(
        {path.name for path in video_dir.iterdir()}
        == {bank.name, graph.name, receipt_path.name},
        f"unexpected files in completed video: {identity}",
    )
    receipt = load_json(receipt_path, f"{identity} receipt")
    for key, expected in _expected_video_fields(
        identity=identity,
        role=role,
        fold=fold,
        binding=binding,
        population=population,
        authorization_receipt_sha256=authorization_receipt_sha256,
        authorization_receipt_kind=authorization_receipt_kind,
    ).items():
        require(receipt.get(key) == expected, f"{identity}: receipt field drifted: {key}")
    require(receipt.get("association_bank_sha256") == sha256(bank), f"{identity}: bank hash drifted")
    require(receipt.get("parent_graph_sha256") == sha256(graph), f"{identity}: parent hash drifted")
    require(int(receipt.get("association_bank_bytes", -1)) == bank.stat().st_size, f"{identity}: bank size drifted")
    require(int(receipt.get("parent_graph_bytes", -1)) == graph.stat().st_size, f"{identity}: parent size drifted")
    require(int(receipt.get("node_count", -1)) >= 0, f"{identity}: invalid node count")
    require(int(receipt.get("candidate_edge_count", -1)) >= 0, f"{identity}: invalid candidate-edge count")
    require(int(receipt.get("parent_node_count", -1)) >= 0, f"{identity}: invalid parent-node count")
    require(int(receipt.get("parent_edge_count", -1)) >= 0, f"{identity}: invalid parent-edge count")
    require(
        receipt.get("runtime", {}).get("identity") == identity,
        f"{identity}: runtime identity drifted",
    )
    return receipt


def _assert_divisions_off(graph: Any) -> None:
    edges = list(graph.edges)
    sources = [int(edge.source_id) for edge in edges]
    targets = [int(edge.target_id) for edge in edges]
    require(len(sources) == len(set(sources)), "R1 parent graph contains a division")
    require(len(targets) == len(set(targets)), "R1 parent graph contains an incoming merge")


def _run_r1_video(
    r1: ModuleType,
    *,
    identity: str,
    model: Any,
    adapter: Any,
    store: Any,
    contract: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    split, dataset = identity.split(":", 1)
    if split == "train":
        return r1.run_video(identity, model, adapter, store, contract)
    require(split == "test", f"unsupported split: {identity}")
    original_open_array = r1.zarr.open_array

    def split_open_array(*args: Any, **kwargs: Any) -> Any:
        expected = f"train/{dataset}.zarr/0"
        require(kwargs.get("path") == expected, "R1 hidden path request drifted")
        forwarded = dict(kwargs)
        forwarded["path"] = f"test/{dataset}.zarr/0"
        return original_open_array(*args, **forwarded)

    r1.zarr.open_array = split_open_array
    try:
        return r1.run_video(identity, model, adapter, store, contract)
    finally:
        r1.zarr.open_array = original_open_array


def publish_video(
    *,
    videos_root: Path,
    identity: str,
    role: str,
    fold: int | None,
    ordinal: int,
    r1: ModuleType,
    model: Any,
    adapter: Any,
    store: Any,
    contract: Mapping[str, Any],
    binding: R1Binding,
    population: Population,
    authorization_receipt_sha256: str,
    authorization_receipt_kind: str,
) -> dict[str, Any]:
    final = videos_root / slug(identity)
    existing = completed_video(
        final,
        identity=identity,
        role=role,
        fold=fold,
        binding=binding,
        population=population,
        authorization_receipt_sha256=authorization_receipt_sha256,
        authorization_receipt_kind=authorization_receipt_kind,
    )
    if existing is not None:
        return existing
    stage = Path(tempfile.mkdtemp(prefix=f".{slug(identity)}.", dir=videos_root))
    try:
        nodes, edges, runtime = _run_r1_video(
            r1,
            identity=identity,
            model=model,
            adapter=adapter,
            store=store,
            contract=contract,
        )
        graph = r1.graph_from_arrays(identity, nodes, edges, contract)
        _assert_divisions_off(graph)
        bank = stage / "association_bank.npz"
        parent = stage / "DIVISIONS_OFF_PARENT.csv"
        bank_sha = r1.write_association_bank(bank, identity, nodes, edges)
        parent_sha = r1.write_graph(parent, graph, contract)
        require(bank_sha == sha256(bank), f"{identity}: R1 bank writer hash mismatch")
        require(parent_sha == sha256(parent), f"{identity}: R1 graph writer hash mismatch")
        receipt = _expected_video_fields(
            identity=identity,
            role=role,
            fold=fold,
            binding=binding,
            population=population,
            authorization_receipt_sha256=authorization_receipt_sha256,
            authorization_receipt_kind=authorization_receipt_kind,
        ) | {
            "ordinal": ordinal,
            "association_bank_sha256": bank_sha,
            "association_bank_bytes": bank.stat().st_size,
            "parent_graph_sha256": parent_sha,
            "parent_graph_bytes": parent.stat().st_size,
            "node_count": int(runtime["node_count"]),
            "candidate_edge_count": int(runtime["candidate_edge_count"]),
            "parent_node_count": len(graph.nodes),
            "parent_edge_count": len(graph.edges),
            "runtime": runtime,
        }
        atomic_json(stage / "RECEIPT.json", receipt)
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require(not final.exists(), f"refusing to overwrite completed video: {identity}")
        os.rename(stage, final)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _role_spec(
    role: str, population: Population
) -> tuple[tuple[str, ...], str, str, str]:
    if role == "development_nested_oof":
        return (
            population.development,
            DEV_SEAL_SCHEMA,
            "SEALED_EXACT_DEV69_BEFORE_R4_GT_OR_SCORER_ACCESS",
            "DEV69_INFERENCE_SEAL.json",
        )
    if role == "sealed_hidden_test_probe":
        return (
            population.hidden,
            HIDDEN_SEAL_SCHEMA,
            "SEALED_EXACT_HIDDEN4_AFTER_LOCAL_CONFIRMATION_GATE",
            "HIDDEN4_INFERENCE_SEAL.json",
        )
    raise AssertionError(role)


def validate_role_seal(
    path: Path,
    *,
    role: str,
    binding: R1Binding,
    population: Population,
    local_confirmation_sha256: str | None,
    expected_authorization_receipt_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    identities, schema, status, _name = _role_spec(role, population)
    seal = load_json(path, f"{role} seal")
    require(seal.get("schema") == schema and seal.get("status") == status, f"{role}: seal status drifted")
    require(seal.get("role") == role, f"{role}: seal role drifted")
    require(seal.get("population_sha256") == population.population_sha256, f"{role}: seal population drifted")
    require(seal.get("r1_binding") == binding.receipt(), f"{role}: seal R1 binding drifted")
    require(seal.get("population_manifest_sha256") == population.manifest_sha256, f"{role}: manifest binding drifted")
    require(seal.get("population_amendment_sha256") == population.amendment_sha256, f"{role}: amendment binding drifted")
    require(seal.get("local_confirmation_gate_sha256") == local_confirmation_sha256, f"{role}: confirmation binding drifted")
    authorization = seal.get("authorization_receipt")
    require(isinstance(authorization, Mapping), f"{role}: authorization receipt is missing")
    authorization_sha = _hash_field(
        authorization.get("sha256"), f"{role} authorization receipt"
    )
    authorization_kind = str(authorization.get("kind"))
    expected_kind = (
        "r4_dev69_preflight"
        if role == "development_nested_oof"
        else "local_confirmation_gate"
    )
    require(authorization_kind == expected_kind, f"{role}: authorization kind drifted")
    if expected_authorization_receipt_sha256 is not None:
        require(
            authorization_sha == expected_authorization_receipt_sha256,
            f"{role}: authorization receipt hash drifted",
        )
    require(seal.get("gt_access") is False and seal.get("score_read") is False, f"{role}: seal reports target access")
    require(seal.get("sealed_before_R4_gt_or_scorer_access") is True, f"{role}: seal ordering drifted")
    records = seal.get("videos")
    require(isinstance(records, list) and len(records) == len(identities), f"{role}: seal count drifted")
    require(tuple(row.get("identity") for row in records) == identities, f"{role}: seal order/membership drifted")
    require(seal.get("identity_count") == len(identities), f"{role}: identity count drifted")
    require(tuple(seal.get("identities", ())) == identities, f"{role}: identity manifest drifted")
    require(seal.get("association_banks_target_free") is True, f"{role}: banks are not target-free")
    require(seal.get("parent_graph_mode") == "DIVISIONS_OFF", f"{role}: parent mode drifted")
    videos_root = path.parent / "videos"
    verified = []
    for identity in identities:
        fold = population.development_fold_by_identity.get(identity)
        record = completed_video(
            videos_root / slug(identity),
            identity=identity,
            role=role,
            fold=fold,
            binding=binding,
            population=population,
            authorization_receipt_sha256=authorization_sha,
            authorization_receipt_kind=authorization_kind,
        )
        require(record is not None, f"{role}: sealed video is missing: {identity}")
        verified.append(record)
    require(records == verified, f"{role}: per-video seal records drifted")
    aggregate = seal.get("aggregate", {})
    require(
        aggregate.get("association_bank_sha256")
        == aggregate_artifact_sha(verified, "association_bank_sha256"),
        f"{role}: aggregate bank hash drifted",
    )
    require(
        aggregate.get("parent_graph_sha256")
        == aggregate_artifact_sha(verified, "parent_graph_sha256"),
        f"{role}: aggregate parent hash drifted",
    )
    if role == "development_nested_oof":
        local_scope = seal.get("local_105_bank_scope", {})
        require(local_scope.get("held_fit_video_count") == 36, "local scope held-fit count drifted")
        require(local_scope.get("development_video_count") == 69, "local scope dev count drifted")
        require(local_scope.get("total_video_count") == 105, "local scope total count drifted")
        require(
            local_scope.get("held_fit_association_bank_sha256")
            == binding.held_fit_association_bank_sha256,
            "local scope held-fit bank hash drifted",
        )
        require(
            local_scope.get("held_fit_parent_graph_sha256")
            == binding.held_fit_parent_graph_sha256,
            "local scope held-fit parent hash drifted",
        )
        require(
            local_scope.get("development_association_bank_sha256")
            == aggregate.get("association_bank_sha256"),
            "local scope dev bank hash drifted",
        )
        require(
            local_scope.get("development_parent_graph_sha256")
            == aggregate.get("parent_graph_sha256"),
            "local scope dev parent hash drifted",
        )
        require(
            local_scope.get("all_hashes_sealed_before_R4_development_scoring")
            is True,
            "local 105 hashes were not sealed before R4 scoring",
        )
    else:
        require(seal.get("local_105_bank_scope") is None, "hidden seal carries local scope")
    return seal


def run_role(
    *,
    role: str,
    output: Path,
    binding: R1Binding,
    population: Population,
    implementation_path: Path,
    adapter_path: Path,
    checkpoint_model_path: Path,
    checkpoint_archive_path: Path,
    source_root: Path,
    local_confirmation_sha256: str | None = None,
    authorization_receipt_sha256: str,
    authorization_receipt_kind: str,
) -> dict[str, Any]:
    identities, _schema, _status, seal_name = _role_spec(role, population)
    require(
        (role == "sealed_hidden_test_probe") == (local_confirmation_sha256 is not None),
        "hidden4 requires exactly one local-confirmation binding",
    )
    _hash_field(authorization_receipt_sha256, "role authorization receipt")
    require(
        authorization_receipt_kind
        == (
            "r4_dev69_preflight"
            if role == "development_nested_oof"
            else "local_confirmation_gate"
        ),
        "role authorization receipt kind drifted",
    )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".INFERENCE_LOCK"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise R4BankError(f"another {role} writer holds the inference lock") from error
        seal_path = output / seal_name
        existing_seal = validate_role_seal(
            seal_path,
            role=role,
            binding=binding,
            population=population,
            local_confirmation_sha256=local_confirmation_sha256,
            expected_authorization_receipt_sha256=authorization_receipt_sha256,
        )
        if existing_seal is not None:
            return existing_seal
        videos_root = output / "videos"
        videos_root.mkdir(exist_ok=True)
        expected_slugs = {slug(identity) for identity in identities}
        unexpected = {
            path.name
            for path in videos_root.iterdir()
            if not path.name.startswith(".") and path.name not in expected_slugs
        }
        require(not unexpected, f"unexpected role output directories: {sorted(unexpected)}")
        completed = []
        for identity in identities:
            record = completed_video(
                videos_root / slug(identity),
                identity=identity,
                role=role,
                fold=population.development_fold_by_identity.get(identity),
                binding=binding,
                population=population,
                authorization_receipt_sha256=authorization_receipt_sha256,
                authorization_receipt_kind=authorization_receipt_kind,
            )
            if record is not None:
                completed.append(record)
        r1 = model = adapter = None
        contract = relocated_contract(
            binding,
            adapter_path=adapter_path,
            checkpoint_model_path=checkpoint_model_path,
            checkpoint_archive_path=checkpoint_archive_path,
            source_root=source_root,
        )
        started = time.monotonic()
        deadline = float(contract["runtime"]["hard_wall_seconds"])
        raw_path = Path(str(contract["inputs"]["raw_archive"]["path"]))
        records_by_identity = {str(row["identity"]): row for row in completed}
        missing = [identity for identity in identities if identity not in records_by_identity]
        if missing:
            require_hash(
                checkpoint_model_path,
                binding.checkpoint_model_sha256,
                "Trackastra model immediately before inference",
            )
            require_hash(
                checkpoint_archive_path,
                binding.checkpoint_archive_sha256,
                "Trackastra archive immediately before inference",
            )
            require(
                tree_sha256(source_root) == binding.source_tree_sha256,
                "Trackastra source tree drifted immediately before inference",
            )
            r1 = load_r1_module(implementation_path, binding.implementation_sha256)
            r1.configure_runtime(contract)
            model = r1.load_trackastra(contract)
            adapter = r1.load_adapter(contract)
            with r1.ZipStore(raw_path, mode="r") as store:
                for ordinal, identity in enumerate(identities, start=1):
                    if identity in records_by_identity:
                        continue
                    if time.monotonic() - started >= deadline - 120.0:
                        raise TimeoutError(
                            f"{role} per-process hard wall reached; resume without overwriting"
                        )
                    record = publish_video(
                        videos_root=videos_root,
                        identity=identity,
                        role=role,
                        fold=population.development_fold_by_identity.get(identity),
                        ordinal=ordinal,
                        r1=r1,
                        model=model,
                        adapter=adapter,
                        store=store,
                        contract=contract,
                        binding=binding,
                        population=population,
                        authorization_receipt_sha256=authorization_receipt_sha256,
                        authorization_receipt_kind=authorization_receipt_kind,
                    )
                    records_by_identity[identity] = record
                    print(
                        f"R4 {role} {ordinal:02d}/{len(identities)} {identity} sealed",
                        flush=True,
                    )
        records = [records_by_identity[identity] for identity in identities]
        require(len(records) == len(identities), f"{role}: incomplete role population")
        seal = {
            "schema": _role_spec(role, population)[1],
            "status": _role_spec(role, population)[2],
            "role": role,
            "population_sha256": population.population_sha256,
            "population_manifest_sha256": population.manifest_sha256,
            "population_amendment_sha256": population.amendment_sha256,
            "identity_count": len(identities),
            "identities": list(identities),
            "outer_fold_sizes": (
                {
                    str(index): sum(
                        population.development_fold_by_identity[value] == index
                        for value in identities
                    )
                    for index in range(5)
                }
                if role == "development_nested_oof"
                else None
            ),
            "r1_binding": binding.receipt(),
            "local_confirmation_gate_sha256": local_confirmation_sha256,
            "authorization_receipt": {
                "kind": authorization_receipt_kind,
                "sha256": authorization_receipt_sha256,
            },
            "aggregate": {
                "association_bank_sha256": aggregate_artifact_sha(
                    records, "association_bank_sha256"
                ),
                "parent_graph_sha256": aggregate_artifact_sha(
                    records, "parent_graph_sha256"
                ),
                "association_bank_bytes": sum(
                    int(row["association_bank_bytes"]) for row in records
                ),
                "parent_graph_bytes": sum(
                    int(row["parent_graph_bytes"]) for row in records
                ),
                "node_count": sum(int(row["node_count"]) for row in records),
                "candidate_edge_count": sum(
                    int(row["candidate_edge_count"]) for row in records
                ),
                "parent_node_count": sum(
                    int(row["parent_node_count"]) for row in records
                ),
                "parent_edge_count": sum(
                    int(row["parent_edge_count"]) for row in records
                ),
                "predicted_forks": 0,
            },
            "local_105_bank_scope": (
                {
                    "held_fit_video_count": 36,
                    "held_fit_association_bank_sha256": (
                        binding.held_fit_association_bank_sha256
                    ),
                    "held_fit_parent_graph_sha256": (
                        binding.held_fit_parent_graph_sha256
                    ),
                    "development_video_count": 69,
                    "development_association_bank_sha256": aggregate_artifact_sha(
                        records, "association_bank_sha256"
                    ),
                    "development_parent_graph_sha256": aggregate_artifact_sha(
                        records, "parent_graph_sha256"
                    ),
                    "total_video_count": 105,
                    "all_hashes_sealed_before_R4_development_scoring": True,
                }
                if role == "development_nested_oof"
                else None
            ),
            "videos": records,
            "association_banks_target_free": True,
            "parent_graph_mode": "DIVISIONS_OFF",
            "sealed_before_R4_gt_or_scorer_access": True,
            "gt_access": False,
            "score_read": False,
            "gpu_hours": 0.0,
        }
        atomic_json(seal_path, seal)
        return seal


def validate_local_confirmation_value(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    binding: R1Binding,
    population: Population,
    development_seal_sha256: str,
) -> None:
    require(
        set(receipt)
        == {
            "schema",
            "status",
            "hidden4_inference_authorized",
            "population_sha256",
            "r1_binding_sha256",
            "development_seal_sha256",
            "selection_complete_before_confirmation",
            "parent_and_overlay_graph_hashes_sealed",
            "held_fit_target_access_count",
            "score_values_in_receipt",
            "test_raw_members_opened",
            "hidden4_started",
            "deployable_config_sha256",
            "held_fit_parent_graph_sha256",
            "held_fit_overlay_graph_sha256",
        },
        "local-confirmation receipt has undeclared fields",
    )
    require(receipt.get("schema") == LOCAL_CONFIRMATION_SCHEMA, "local-confirmation schema drifted")
    require(
        receipt.get("status") == "PASS_LOCAL_CONFIRMATION_GATE_HIDDEN4_AUTHORIZED",
        "local-confirmation gate does not authorize hidden4",
    )
    require(receipt.get("hidden4_inference_authorized") is True, "hidden4 authorization is absent")
    require(receipt.get("population_sha256") == population.population_sha256, "confirmation population drifted")
    require(receipt.get("r1_binding_sha256") == binding.binding_sha256, "confirmation R1 binding drifted")
    require(
        receipt.get("development_seal_sha256") == development_seal_sha256,
        "confirmation does not bind the dev69 seal",
    )
    require(receipt.get("selection_complete_before_confirmation") is True, "selection was not frozen before confirmation")
    require(receipt.get("parent_and_overlay_graph_hashes_sealed") is True, "confirmation graphs were not presealed")
    require(int(receipt.get("held_fit_target_access_count", -1)) == 1, "confirmation was not exactly one target access")
    require(receipt.get("score_values_in_receipt") is False, "confirmation gate receipt must omit score values")
    require(receipt.get("test_raw_members_opened") is False, "confirmation already opened test raw members")
    require(receipt.get("hidden4_started") is False, "hidden4 already started")
    for field in (
        "deployable_config_sha256",
        "held_fit_parent_graph_sha256",
        "held_fit_overlay_graph_sha256",
    ):
        _hash_field(receipt.get(field), f"local confirmation {field}")
    _hash_field(receipt_sha256, "local confirmation receipt")


def validate_local_confirmation_file(
    path: Path,
    expected_sha256: str,
    *,
    binding: R1Binding,
    population: Population,
    development_seal_sha256: str,
) -> dict[str, Any]:
    require_hash(path, expected_sha256, "local-confirmation gate receipt")
    receipt = load_json(path, "local-confirmation gate receipt")
    validate_local_confirmation_value(
        receipt,
        receipt_sha256=expected_sha256,
        binding=binding,
        population=population,
        development_seal_sha256=development_seal_sha256,
    )
    return receipt


def _add_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--promotion-sha256", required=True)
    parser.add_argument("--r1-contract", type=Path, required=True)
    parser.add_argument("--r1-amendment", type=Path, required=True)
    parser.add_argument("--r1-implementation", type=Path, required=True)
    parser.add_argument("--r1-adapter", type=Path, required=True)
    parser.add_argument("--r1-checkpoint-model", type=Path, required=True)
    parser.add_argument("--r1-checkpoint-archive", type=Path, required=True)
    parser.add_argument("--r1-source-root", type=Path, required=True)
    parser.add_argument("--r1-preflight", type=Path, required=True)
    parser.add_argument("--r1-inference-seal", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--population-amendment", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight-dev69")
    _add_binding_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    dev = subcommands.add_parser("infer-dev69")
    _add_binding_arguments(dev)
    dev.add_argument("--r4-preflight", type=Path, required=True)
    dev.add_argument("--r4-preflight-sha256", required=True)
    dev.add_argument("--output", type=Path, required=True)
    hidden = subcommands.add_parser("infer-hidden4")
    _add_binding_arguments(hidden)
    hidden.add_argument("--development-seal", type=Path, required=True)
    hidden.add_argument("--development-seal-sha256", required=True)
    hidden.add_argument("--local-confirmation", type=Path, required=True)
    hidden.add_argument("--local-confirmation-sha256", required=True)
    hidden.add_argument("--output", type=Path, required=True)
    return result


def _binding_from_args(args: argparse.Namespace) -> R1Binding:
    return validate_r1_binding(
        promotion_path=args.promotion,
        promotion_sha256=args.promotion_sha256,
        contract_path=args.r1_contract,
        amendment_path=args.r1_amendment,
        implementation_path=args.r1_implementation,
        adapter_path=args.r1_adapter,
        checkpoint_model_path=args.r1_checkpoint_model,
        checkpoint_archive_path=args.r1_checkpoint_archive,
        source_root=args.r1_source_root,
        preflight_path=args.r1_preflight,
        inference_seal_path=args.r1_inference_seal,
    )


def main() -> int:
    args = parser().parse_args()
    if args.command == "infer-hidden4":
        # This existence/hash gate is deliberately the first operation.  In
        # particular, no population/cache/model/test-raw path is touched first.
        require_hash(
            args.local_confirmation,
            args.local_confirmation_sha256,
            "local-confirmation gate receipt",
        )
    binding = _binding_from_args(args)
    population = validate_population(args.population_manifest, args.population_amendment)
    if args.command == "preflight-dev69":
        report = create_preflight(output=args.output, binding=binding, population=population)
    elif args.command == "infer-dev69":
        validate_preflight(
            args.r4_preflight,
            args.r4_preflight_sha256,
            binding,
            population,
        )
        verify_cache_population(binding, population.development)
        report = run_role(
            role="development_nested_oof",
            output=args.output,
            binding=binding,
            population=population,
            implementation_path=args.r1_implementation,
            adapter_path=args.r1_adapter,
            checkpoint_model_path=args.r1_checkpoint_model,
            checkpoint_archive_path=args.r1_checkpoint_archive,
            source_root=args.r1_source_root,
            authorization_receipt_sha256=args.r4_preflight_sha256,
            authorization_receipt_kind="r4_dev69_preflight",
        )
    elif args.command == "infer-hidden4":
        require_hash(
            args.development_seal,
            args.development_seal_sha256,
            "dev69 inference seal",
        )
        development_seal = validate_role_seal(
            args.development_seal,
            role="development_nested_oof",
            binding=binding,
            population=population,
            local_confirmation_sha256=None,
        )
        require(development_seal is not None, "dev69 inference seal is absent")
        validate_local_confirmation_file(
            args.local_confirmation,
            args.local_confirmation_sha256,
            binding=binding,
            population=population,
            development_seal_sha256=args.development_seal_sha256,
        )
        verify_cache_population(binding, population.hidden)
        report = run_role(
            role="sealed_hidden_test_probe",
            output=args.output,
            binding=binding,
            population=population,
            implementation_path=args.r1_implementation,
            adapter_path=args.r1_adapter,
            checkpoint_model_path=args.r1_checkpoint_model,
            checkpoint_archive_path=args.r1_checkpoint_archive,
            source_root=args.r1_source_root,
            local_confirmation_sha256=args.local_confirmation_sha256,
            authorization_receipt_sha256=args.local_confirmation_sha256,
            authorization_receipt_kind="local_confirmation_gate",
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R4BankError as error:
        print(f"R4_BANK_EXTENSION_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error

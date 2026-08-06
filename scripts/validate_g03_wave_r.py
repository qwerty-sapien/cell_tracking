#!/usr/bin/env python3
"""Validate G-03 Wave-R policy, evidence lanes, and final allocation report."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.g03_division_decode_manifest import build_manifest  # noqa: E402
from scripts.g03_division_population_manifest import (  # noqa: E402
    build_manifest as build_population_manifest,
)


BEST_KNOWN_SCORE = Decimal("0.881522069960")
NEIGHBORHOOD_MULTIPLIER = Decimal("0.985")
NEIGHBORHOOD_FLOOR = BEST_KNOWN_SCORE * NEIGHBORHOOD_MULTIPLIER
EXPECTED_R2_MD_SHA256 = "1b2927d1ea6de63a3d4aace4296ad7f190cff45ea21c84b540c385ae85a95f6f"
EXPECTED_R2_JSON_SHA256 = "20464ec8e7a3a5bbb2e3d1ba957f0e957d603e3f86157054fa7ec5fb017278e1"
EXPECTED_R4_CONFIG_SHA256 = "fe72a4c973915602051867feaee7a13b71ac63e37e3588e1bbb02cdf3b394adc"
EXPECTED_R4_MANIFEST_SHA256 = "656bb6fa053bef5682dbcbef9d79377424a81d0eb680e23a2a8c20614067c83a"
EXPECTED_R4_DESIGN_SHA256 = "6628d5ecb7067b12973f615c2a34122319d0bda5ff60249fd27bc1ba7f5de99a"
EXPECTED_R4_PROTOCOL_SHA256 = "3b7582ebcfb233ecc8f5d5a3999dcafeb6d1f9b1d22cfb1d79a3230c04c6d02a"
EXPECTED_R4_POPULATION_SHA256 = "e25416da1d240ceabf2b1457ceb6625b361523e7a537ed2f4ac195280b425954"
EXPECTED_R4_POPULATION_FILE_SHA256 = "b58fe1acd794c35c9a2f8e63e65327f74d5877cff223ed122a19b68f9b309f47"
EXPECTED_R4_POPULATION_AMENDMENT_SHA256 = "bc09d4420c5ed5d66e93b1e5fe5debe1e1ec0e4c8999dacfbc35979e7d3b981d"
EXPECTED_R4_HISTORICAL_DECODER_AMENDMENT_SHA256 = "34fcd826af4a03ba428b9eb8473c7a601f45e568bf41b201b2bddae6ed62183e"
EXPECTED_R4_DECODER_AMENDMENT_SHA256 = "bd5a3af7d2b8d7b69130e237e2a41eee5cabd2ff8f700d7bfaf71b19baecabcf"
EXPECTED_R4_SEMANTIC_RUNTIME_RECEIPT_SHA256 = "7b2ff492a3f8af0a37242417c2063b48e1ffae0c34b828a6565df6956f9006b5"
EXPECTED_R4_DECODER_SHA256 = "a58c49644914db618d2cad8be753d9b9237e3cc69eeec38d20deeabec17521c0"
EXPECTED_R4_DECODER_TEST_SHA256 = "e3f341d4028e0eccc480befd3de4a149eee6a257ff9d9bdfe865a80784504826"
EXPECTED_R4_BANK_EXTENSION_SHA256 = "7cc4c7d2d27d5b4ed45b11cbdd0e73d33a07099f9904fa579bdd6012659e0a9c"
EXPECTED_R4_BANK_EXTENSION_TEST_SHA256 = "0962716ad1f19a72df7d84180f1bb57ca6600a1a562e973ae733eb5cb53c6c93"
EXPECTED_R4_OOF_ORCHESTRATOR_SHA256 = "853e11a9acba9685c3de64d436519a9e8006152ea275cc7fb1c76d3e9b4b22d7"
EXPECTED_R4_OOF_ORCHESTRATOR_TEST_SHA256 = "3e5100e3f38157f5e883f4617bdc310159ea321b2374d5b5a430de22e187320f"
EXPECTED_R4_OOF_RUNNER_SHA256 = "cc37fc63729d4f8d5418951dabc4ecd2ff1ccc2b7f684011793263af73cde304"
EXPECTED_R4_OOF_RUNNER_TEST_SHA256 = "c500164826b715215f2ac2ceb26ef71872278dee2ae72ab2195ca8df7b6b0c39"
EXPECTED_R4_INPUT_GENERATOR_SHA256 = "a7372055e791c6120509fd1967676ec253a5c40cdf9d3f6155a459dc92caed71"
EXPECTED_R4_INPUT_GENERATOR_TEST_SHA256 = "4d81ee2c4cf3ed44e10af5218eaf659f75298c5aabc5d71d56b16b1c04315259"
EXPECTED_R4_DEV69_SEAL_SHA256 = "588397b13ba330192ebb8bd3fb6f226ec61d2f2ba9ee9635728765dde00ee6d8"
EXPECTED_R4_PRETARGET_INPUT_SHA256 = "b3d0610cc4ab4abc07ce630aed06dc34ed958e939703fdffdc3f7ac3411064a1"
EXPECTED_R4_PRETARGET_PLAN_SHA256 = "ecbb7467c2507bde24d863187e8db98c8d6d367668e9739cf4b764410af665a4"
EXPECTED_R4_PRETARGET_PLAN_PAYLOAD_SHA256 = "8f4a07954218b0070e73f2b126b953433c7d1ca2f715da0e7726f97941b0e7b1"
EXPECTED_R4_RUNNER_INPUT_SHA256 = "be6f7a4e349d1ea38dca9bf5cf473f8ad26764a0573fec566e715cf25766ee23"
EXPECTED_R4_HISTORICAL_PRETARGET_PLAN_V3_SHA256 = "da2bfa6136ffff5d955d3b7e49d02dafdf72168496211b10d7a1b30b54a4d96d"
EXPECTED_R4_HISTORICAL_RUNNER_INPUT_V3_SHA256 = "bc815cafb01d16313e5b65c6cfbdf9a62f118d960cf2aa1fe924f4defc3a0322"
EXPECTED_R4_HISTORICAL_DECODER_SHA256 = "d310394310116ebf257d74159a39be1d77679a9dd7aec14d97f3589e9f7da44a"
EXPECTED_R4_HISTORICAL_RUNNER_SHA256 = "2b9cb30e2c808fd2c7c8242a42d16e54258c319e490cc151e4365e04bbed7284"
EXPECTED_R4_PRETARGET_REPAIR_SHA256 = "92a4f3fca19c78a49dc27a5dc9e6fb2d6909f9185d6bf1ac067989ea0c2b4e95"
EXPECTED_R4_RUNNER_BINDING_REPAIR_SHA256 = "14a581b072ef430c61ef7f0bef528143223a2f088b7a96091cc0f338cd88dfb6"
EXPECTED_R4_FACTORIZATION_REPAIR_SHA256 = "8cbc272716d1c491aa81c93b61aecba9e8591ceeb8b5ddcef000543af6158109"
EXPECTED_R4_EXACT_BENCHMARK_SCRIPT_SHA256 = "4b3982926f733618c50076a0ab7bc7e292f3bad5768dd2e614b4eaf17859b775"
EXPECTED_R4_EXACT_BENCHMARK_RESULT_SHA256 = "b15d11a35e76edc49b9ff5f956657caf710c10a69ee87470cbd4bdc0d98884fb"
EXPECTED_R4_CONFLICT_AUDIT_SCRIPT_SHA256 = "528c9ea8a34b3006a4a66b2501aa4635a9c6a63335cf31a55f08242e5a687056"
EXPECTED_R4_CONFLICT_AUDIT_RESULT_SHA256 = "0c27d45865dd04371d641a3ccf34fb1325e3ab08f460f6a96c7a07d59e92224d"
EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_SCRIPT_SHA256 = "cc8c9462faf20d963071bad0a2420e0d81d5e126552e94ecba53a1e509e1dd65"
EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_RESULT_SHA256 = "0ba158beac5a957be166ca29ea93b16d4bc4cf838a2804144a675ee296259a3f"
EXPECTED_R4_AMENDED_ALL_GRID_AUDIT_SCRIPT_SHA256 = "f24a03b6e44f7bd05bbef9d6b38aef5223ebd3ef46379018abbf57f152377184"
EXPECTED_R4_AMENDED_ALL_GRID_AUDIT_RESULT_SHA256 = "c3b3934ae9bc9b0995e27741e7fa45aaf2300ccf9b3f664b6e89830f57d317c8"
EXPECTED_R4_AMENDED_WORST_BENCHMARK_SCRIPT_SHA256 = "00dbd37dbb9a8c617de201601b9803ec273c34a52831e448d71360f78e7718ca"
EXPECTED_R4_AMENDED_WORST_BENCHMARK_TEST_SHA256 = "794dc57122e78c86aefb6cd4570c8fcfac76af3328b0480d9a1c2cf576415abf"
EXPECTED_R4_AMENDED_WORST_BENCHMARK_RESULT_SHA256 = "fc479872f5df0d27a6cb8850b34be602b9b6bc5bd51814c96af8374a73ea3be9"
EXPECTED_R4_ACTION_COMPILER_SHA256 = "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"
EXPECTED_R4_GRAPH_MODEL_SHA256 = "d8f93181c38a319c79c8075fd730281d9aa8521ddefb6fdae8f64b5b27c373e4"
EXPECTED_R1_RESULT_SHA256 = "55300582b831454ab6ba2b538cba7271d0c99bb230c2db24a2c049477f514372"
EXPECTED_R1_VALIDATION_SHA256 = "b25665f2724d8437ed862a574f30d59afdc40de21708585f3738e759c04ee16c"
EXPECTED_R4_BANK_REPAIR_SHA256 = "a6404fcef1b5ff7b4620c15e66f82807c1d0dcdb903ded2ea7f818a6be594d53"
EXPECTED_R4_BANK_PREFLIGHT_V2_SHA256 = "960e3c619d2f85ecbc0a7050608be8212374280555353f21ffa2f389abbc4466"
EXPECTED_R1_POSTHOC_AMENDMENT_SHA256 = "f5c3084e7fbafd0506f947806baafd09cdb2c7da618fef4d93307dfa4434bd4f"
EXPECTED_R1_RECOVERY_HELPER_SHA256 = "b1ca85fce69c7e7cd20a8ea773f2b55dba3da0d7673a13cb9a1fba85a19b82c6"
EXPECTED_R1_FINALIZER_HELPER_SHA256 = "0b1afb0ffdb027412898f65bf017bb0d5c8b6090aa45a807dcaa614dd0520654"


class WaveRValidationError(RuntimeError):
    """A Wave-R policy, provenance, or result invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise WaveRValidationError(f"{path}: expected a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WaveRValidationError(message)


def decimal(value: Any, label: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as error:  # pragma: no cover - Decimal explains malformed input
        raise WaveRValidationError(f"{label}: not a decimal") from error


def expected_branch(score: Decimal, *, new_best: bool = False) -> str:
    return "A" if new_best or score >= NEIGHBORHOOD_FLOOR else "B"


def validate_policy_text(root: Path) -> dict[str, Any]:
    policy = root / "reports/g03/screening/WAVE_R_POLICY_AND_DISPATCH_20260801.md"
    live = root / "reports/g03/screening/LIVE_STATE.md"
    ledger = root / "reports/g03/screening/LEDGER.md"
    for path in (policy, live, ledger):
        require(path.is_file(), f"missing policy/state artifact: {path}")
    exact_floor = str(NEIGHBORHOOD_FLOOR)
    for path in (policy, live, ledger):
        text = path.read_text(encoding="utf-8")
        require(exact_floor in text, f"{path}: exact neighborhood floor missing")
    policy_text = " ".join(policy.read_text(encoding="utf-8").split())
    for phrase in (
        "RunPod remains forbidden",
        "Only the final-slot selection returns to the owner",
        "Branch A",
        "Branch B",
    ):
        require(phrase in policy_text, f"policy missing binding phrase: {phrase}")
    return {
        "best_known_score": str(BEST_KNOWN_SCORE),
        "neighborhood_floor": str(NEIGHBORHOOD_FLOOR),
        "policy_sha256": sha256(policy),
    }


def validate_r2(root: Path) -> dict[str, Any]:
    directory = root / "reports/g03/screening/wave_r"
    md = directory / "R2_LITERATURE_AUDIT.md"
    table = directory / "R2_PRIMARY_SOURCE_TABLE.json"
    require(sha256(md) == EXPECTED_R2_MD_SHA256, "R2 Markdown identity drift")
    require(sha256(table) == EXPECTED_R2_JSON_SHA256, "R2 source-table identity drift")
    value = load_json(table)
    reference = value.get("decision_reference", {})
    require(
        decimal(reference.get("best_known_score"), "R2 best") == BEST_KNOWN_SCORE,
        "R2 best-known reference drift",
    )
    require(
        decimal(reference.get("neighborhood_multiplier"), "R2 multiplier")
        == NEIGHBORHOOD_MULTIPLIER,
        "R2 multiplier drift",
    )
    require(
        decimal(reference.get("neighborhood_floor"), "R2 floor") == NEIGHBORHOOD_FLOOR,
        "R2 neighborhood-floor arithmetic drift",
    )
    verdict = value.get("branch_b_verdict", {})
    require(verdict.get("overall") == "FAIL", "R2 Branch-B overall verdict must be FAIL")
    require(
        all(
            verdict.get(key) == "FAIL_NOT_VALIDATED"
            for key in ("within_30_t4h_comparability", "path_to_0915", "path_to_0945")
        ),
        "R2 Branch-B subgates are not all fail-closed",
    )
    sources = value.get("sources", [])
    require(isinstance(sources, list) and len(sources) >= 7, "R2 primary-source ledger is incomplete")
    require(
        all(
            isinstance(row, Mapping)
            and str(row.get("primary_url", "")).startswith("https://")
            and row.get("limitations")
            for row in sources
        ),
        "R2 source row lacks a primary URL or transfer limitation",
    )
    return {"verdict": "BRANCH_B_FAIL", "source_count": len(sources)}


def validate_r3(root: Path) -> dict[str, Any]:
    path = root / "reports/g03/screening/WAVE_R_R3_ASSET_AUDIT.md"
    text = " ".join(path.read_text(encoding="utf-8").split())
    for required in (
        "no reusable artifact supplies G-03's required real 12-dimensional `wrfeat`",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
        "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
        "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19",
        "198aeda1d0c0522bd1aedacdbb48e2b343aca0b25cb41d68dd8cb72cbfc87b14",
        "6015a1a6440af4d4da705a416472f3f6b3dab7be764549a969bf507b4e1ddd76",
    ):
        require(required in text, f"R3 missing required identity/finding: {required}")
    return {"asset_audit_sha256": sha256(path), "real_wrfeat_reusable": False}


def validate_r4(root: Path) -> dict[str, Any]:
    design = root / "reports/g03/screening/WAVE_R_R4_DIVISION_DECODE_DESIGN.md"
    require(design.is_file(), "R4 design is missing")
    require(sha256(design) == EXPECTED_R4_DESIGN_SHA256, "R4 design identity drift")
    manifest = build_manifest()
    require(manifest["active_configuration_count"] == 18_144, "R4 active grid count drift")
    require(
        manifest["configuration_count_including_null"] == 18_145,
        "R4 total grid count drift",
    )
    require(
        manifest["configuration_sha256"] == EXPECTED_R4_CONFIG_SHA256,
        "R4 canonical configuration hash drift",
    )
    text = design.read_text(encoding="utf-8")
    for phrase in (
        "nested/grouped-OOF",
        "Sealed one-shot confirmation",
        "duplicate_survivors` must equal zero",
        "18,144",
    ):
        require(phrase in text, f"R4 design missing protocol phrase: {phrase}")
    manifest_path = (
        root
        / "reports/g03/screening/wave_r/R4_DIVISION_DECODE_CONFIG_MANIFEST.json"
    )
    require(
        sha256(manifest_path) == EXPECTED_R4_MANIFEST_SHA256,
        "retained R4 configuration-manifest identity drift",
    )
    require(load_json(manifest_path) == manifest, "retained R4 manifest content drift")
    protocol_path = (
        root / "reports/g03/screening/wave_r/R4_DIVISION_PROTOCOL_CONTRACT.json"
    )
    require(
        sha256(protocol_path) == EXPECTED_R4_PROTOCOL_SHA256,
        "R4 protocol-contract identity drift",
    )
    protocol = load_json(protocol_path)
    require(
        protocol.get("status") == "SEALED_BEFORE_ANY_G03_DIVISION_OUTCOME",
        "R4 protocol was not sealed before outcomes",
    )
    protocol_rules = protocol.get("protocol", {})
    for rule in (
        "adaptive_search_forbidden",
        "configuration_splicing_forbidden",
        "gt_for_candidate_generation_forbidden",
        "same_population_parent_non_regression_required",
        "division_terms_reported_separately",
    ):
        require(protocol_rules.get(rule) is True, f"R4 protocol rule is not binding: {rule}")
    require(
        protocol.get("configuration_manifest", {}).get("file_sha256")
        == EXPECTED_R4_MANIFEST_SHA256,
        "R4 protocol does not bind retained manifest",
    )
    for relative, wanted in protocol.get("frozen_dependencies", {}).items():
        require(sha256(root / str(relative)) == wanted, f"R4 dependency drift: {relative}")
    population_path = (
        root / "reports/g03/screening/wave_r/R4_DIVISION_POPULATION_MANIFEST.json"
    )
    require(
        sha256(population_path) == EXPECTED_R4_POPULATION_FILE_SHA256,
        "R4 population-manifest identity drift",
    )
    population = load_json(population_path)
    require(
        population == build_population_manifest(root),
        "R4 population manifest does not reproduce from frozen sources",
    )
    require(
        population.get("population_sha256") == EXPECTED_R4_POPULATION_SHA256,
        "R4 population membership hash drift",
    )
    require(
        [
            population["roles"][role]["video_count"]
            for role in (
                "development_nested_oof",
                "held_fit_one_shot_confirmation",
                "sealed_hidden_test_probe",
            )
        ]
        == [69, 36, 4],
        "R4 role counts are not the architecture-aligned 69/36 plus hidden-test 4",
    )
    amendment_path = (
        root / "reports/g03/screening/wave_r/R4_DIVISION_POPULATION_AMENDMENT.json"
    )
    require(
        sha256(amendment_path) == EXPECTED_R4_POPULATION_AMENDMENT_SHA256,
        "R4 population-amendment identity drift",
    )
    amendment = load_json(amendment_path)
    require(
        amendment.get("status")
        == "SEALED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "R4 population amendment was not sealed before outcomes",
    )
    require(
        amendment.get("population_manifest", {}).get("population_sha256")
        == EXPECTED_R4_POPULATION_SHA256,
        "R4 amendment does not bind population membership",
    )
    require(
        amendment.get("association_bank_scope", {}).get("required_total_video_count")
        == 109,
        "R4 amendment does not require complete 109-video bank coverage",
    )
    require(
        amendment.get("binding_roles", {})
        .get("excluded_contaminated_audit20", {})
        .get("target_access_for_R4_allowed")
        is False,
        "R4 still treats contaminated audit20 as confirmation",
    )
    require(
        amendment.get("binding_roles", {})
        .get("sealed_hidden_test_probe", {})
        .get("submissions_allowed")
        == 1,
        "R4 hidden-test confirmation is not exactly one probe",
    )
    require(
        amendment.get("binding_roles", {})
        .get("held_fit_one_shot_confirmation", {})
        .get("decoder_overlay_evaluations_allowed")
        == 1,
        "R4 held-fit decoder confirmation is not exactly one overlay",
    )
    require(
        amendment.get("scientific_configuration", {}).get("modal_outer_winner_rule")
        == "highest outer-winner count, then lexicographically smallest config name",
        "R4 modal-winner tie rule is underdefined",
    )
    require(
        amendment.get("association_bank_scope", {}).get(
            "hidden_test_inference_only_after_local_confirmation_gate"
        )
        is True,
        "R4 hidden-test inference is not gated after local confirmation",
    )
    references = amendment.get("held_fit_reference_set", [])
    require(
        isinstance(references, list)
        and [row.get("id") for row in references]
        == ["APNCC_44b6_CONTROL", "R1_DIVISIONS_OFF"],
        "R4 held-fit eligible-reference set is not closed",
    )
    historical_decoder_amendment_path = (
        root / "reports/g03/screening/wave_r/R4_DECODER_IMPLEMENTATION_AMENDMENT.json"
    )
    require(
        sha256(historical_decoder_amendment_path)
        == EXPECTED_R4_HISTORICAL_DECODER_AMENDMENT_SHA256,
        "historical R4 decoder amendment identity drift",
    )
    decoder_amendment_path = (
        root / "reports/g03/screening/wave_r/R4_DECODER_IMPLEMENTATION_AMENDMENT_V2.json"
    )
    require(
        sha256(decoder_amendment_path) == EXPECTED_R4_DECODER_AMENDMENT_SHA256,
        "R4 decoder-amendment V2 identity drift",
    )
    decoder_amendment = load_json(decoder_amendment_path)
    require(
        decoder_amendment.get("status")
        == "PRE_OUTCOME_RESEALED_PENDING_FINAL_INDEPENDENT_AUDIT"
        and decoder_amendment.get("target_access_at_seal") is False
        and decoder_amendment.get("division_outcome_access_at_seal") is False
        and decoder_amendment.get("metric_rows_at_seal") == 0,
        "R4 decoder V2 was not resealed target-blind",
    )
    supersession = decoder_amendment.get("supersession", {})
    preserved = decoder_amendment.get("preserved_scientific_contracts", {})
    decoder_refs = decoder_amendment.get("frozen_sources", {})
    require(
        supersession.get("supersedes_sha256")
        == EXPECTED_R4_HISTORICAL_DECODER_AMENDMENT_SHA256
        and supersession.get("predictions_and_predicted_fork_counts_may_change")
        is True
        and preserved.get("configuration_sha256") == EXPECTED_R4_CONFIG_SHA256
        and preserved.get("population_sha256") == EXPECTED_R4_POPULATION_SHA256
        and preserved.get("configuration_names_values_folds_selection_rules_and_gates_changed")
        is False
        and preserved.get("retrospective_metric_use") is False
        and preserved.get("configuration_search") is False
        and decoder_refs.get("decoder", {}).get("sha256")
        == EXPECTED_R4_DECODER_SHA256
        and decoder_refs.get("decoder_test", {}).get("sha256")
        == EXPECTED_R4_DECODER_TEST_SHA256,
        "R4 decoder V2 preservation or source reference drift",
    )
    require(
        sha256(root / "scripts/g03_division_decode.py") == EXPECTED_R4_DECODER_SHA256,
        "R4 decoder identity drift",
    )
    require(
        sha256(root / "tests/test_g03_division_decode.py")
        == EXPECTED_R4_DECODER_TEST_SHA256,
        "R4 decoder-test identity drift",
    )
    require(
        sha256(root / "scripts/g03_r4_bank_extension.py")
        == EXPECTED_R4_BANK_EXTENSION_SHA256,
        "R4 bank-extension identity drift",
    )
    require(
        sha256(root / "tests/test_g03_r4_bank_extension.py")
        == EXPECTED_R4_BANK_EXTENSION_TEST_SHA256,
        "R4 bank-extension test identity drift",
    )
    repair_path = (
        root / "reports/g03/screening/wave_r/R4_BANK_EXTENSION_REPAIR_RECEIPT.json"
    )
    require(
        sha256(repair_path) == EXPECTED_R4_BANK_REPAIR_SHA256,
        "R4 bank-extension repair receipt identity drift",
    )
    repair = load_json(repair_path)
    repair_body = repair.get("repair", {})
    failed_attempts = repair.get("failed_attempts", [])
    require(
        repair.get("status") == "REPAIRED_BEFORE_FIRST_ATOMIC_VIDEO_PUBLICATION"
        and isinstance(failed_attempts, list)
        and len(failed_attempts) == 2
        and all(row.get("published_video_count") == 0 for row in failed_attempts)
        and repair_body.get("scientific_configuration_changed") is False
        and repair_body.get("target_or_score_access") is False
        and repair_body.get("corrected_implementation_sha256")
        == EXPECTED_R4_BANK_EXTENSION_SHA256
        and repair_body.get("corrected_test_sha256")
        == EXPECTED_R4_BANK_EXTENSION_TEST_SHA256
        and repair_body.get("corrected_preflight_sha256")
        == EXPECTED_R4_BANK_PREFLIGHT_V2_SHA256,
        "R4 bank-extension repair accounting drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_artifacts/bank_extension/PREFLIGHT_DEV69_V2.json"
        )
        == EXPECTED_R4_BANK_PREFLIGHT_V2_SHA256,
        "R4 bank-extension V2 preflight identity drift",
    )
    require(
        sha256(root / "scripts/g03_division_oof_orchestrator.py")
        == EXPECTED_R4_OOF_ORCHESTRATOR_SHA256,
        "R4 OOF-orchestrator identity drift",
    )
    require(
        sha256(root / "tests/test_g03_division_oof_orchestrator.py")
        == EXPECTED_R4_OOF_ORCHESTRATOR_TEST_SHA256,
        "R4 OOF-orchestrator test identity drift",
    )
    require(
        sha256(root / "scripts/g03_division_oof_runner.py")
        == EXPECTED_R4_OOF_RUNNER_SHA256,
        "R4 OOF-runner identity drift",
    )
    require(
        sha256(root / "tests/test_g03_division_oof_runner.py")
        == EXPECTED_R4_OOF_RUNNER_TEST_SHA256,
        "R4 OOF-runner test identity drift",
    )
    require(
        sha256(root / "scripts/g03_prepare_r4_oof_inputs.py")
        == EXPECTED_R4_INPUT_GENERATOR_SHA256,
        "R4 input-generator identity drift",
    )
    require(
        sha256(root / "tests/test_g03_prepare_r4_oof_inputs.py")
        == EXPECTED_R4_INPUT_GENERATOR_TEST_SHA256,
        "R4 input-generator test identity drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_artifacts/bank_extension/dev69/DEV69_INFERENCE_SEAL.json"
        )
        == EXPECTED_R4_DEV69_SEAL_SHA256,
        "R4 dev69 seal identity drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_artifacts/oof/PRETARGET_INPUT.json"
        )
        == EXPECTED_R4_PRETARGET_INPUT_SHA256,
        "R4 pretarget-input identity drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_artifacts/oof/PRETARGET_PLAN_V4.json"
        )
        == EXPECTED_R4_PRETARGET_PLAN_SHA256,
        "R4 pretarget-plan identity drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_artifacts/oof/RUNNER_INPUT_V4.json"
        )
        == EXPECTED_R4_RUNNER_INPUT_SHA256,
        "R4 runner-input identity drift",
    )
    pretarget_plan_v4 = load_json(
        root / "reports/g03/screening/wave_r/R4_artifacts/oof/PRETARGET_PLAN_V4.json"
    )
    runner_input_v4 = load_json(
        root / "reports/g03/screening/wave_r/R4_artifacts/oof/RUNNER_INPUT_V4.json"
    )
    require(
        pretarget_plan_v4.get("payload_sha256")
        == EXPECTED_R4_PRETARGET_PLAN_PAYLOAD_SHA256
        and pretarget_plan_v4.get("payload", {})
        .get("execution_source_bindings", {})
        .get("runner", {})
        .get("sha256")
        == EXPECTED_R4_OOF_RUNNER_SHA256
        and runner_input_v4.get("decoder_source", {}).get("sha256")
        == EXPECTED_R4_DECODER_SHA256
        and runner_input_v4.get("orchestrator_source", {}).get("sha256")
        == EXPECTED_R4_OOF_ORCHESTRATOR_SHA256
        and runner_input_v4.get("pretarget_plan", {}).get("sha256")
        == EXPECTED_R4_PRETARGET_PLAN_SHA256
        and runner_input_v4.get("one_pass_dependencies", {})
        .get("src/celltrack_alpha/action_compiler.py", {})
        .get("sha256")
        == EXPECTED_R4_ACTION_COMPILER_SHA256
        and runner_input_v4.get("one_pass_dependencies", {})
        .get("src/celltrack_alpha/kaggle_calibration.py", {})
        .get("sha256")
        == EXPECTED_R4_GRAPH_MODEL_SHA256,
        "R4 V4 plan, runner, decoder, or one-pass dependency seal drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_PRETARGET_PREPUBLICATION_REPAIR_RECEIPT.json"
        )
        == EXPECTED_R4_PRETARGET_REPAIR_SHA256,
        "R4 pretarget-repair accounting drift",
    )
    require(
        sha256(
            root
            / "reports/g03/screening/wave_r/R4_RUNNER_CONFIGURATION_BINDING_REPAIR_RECEIPT.json"
        )
        == EXPECTED_R4_RUNNER_BINDING_REPAIR_SHA256,
        "R4 runner-binding repair accounting drift",
    )
    factorization_path = (
        root
        / "reports/g03/screening/wave_r/R4_EXACT_FACTORIZATION_REPAIR_RECEIPT.json"
    )
    require(
        sha256(factorization_path) == EXPECTED_R4_FACTORIZATION_REPAIR_SHA256,
        "R4 exact-factorization repair accounting drift",
    )
    factorization = load_json(factorization_path)
    boundary = factorization.get("repair_boundary", {})
    exact = factorization.get("exact_factorization", {})
    benchmark = factorization.get("verification", {}).get(
        "target_free_exact_runner_benchmark", {}
    )
    benchmark_script_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/g03_r4_exact_runner_benchmark.py"
    )
    benchmark_result_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/EXACT_RUNNER_SMALL_RESULT.json"
    )
    require(
        sha256(benchmark_script_path) == EXPECTED_R4_EXACT_BENCHMARK_SCRIPT_SHA256
        and sha256(benchmark_result_path) == EXPECTED_R4_EXACT_BENCHMARK_RESULT_SHA256,
        "R4 exact target-free benchmark evidence drift",
    )
    benchmark_result = load_json(benchmark_result_path)
    require(
        factorization.get("status")
        == "SEALED_EXACT_IN_MEMORY_FACTORIZATION_BEFORE_DEVELOPMENT_SCORING"
        and boundary.get("configuration_names_or_indices_changed") is False
        and boundary.get("late_window_retained_in_exact_generation_key") is True
        and boundary.get("decoder_axis_canonicalized") is False
        and boundary.get("persistent_configuration_action_index") is False
        and boundary.get("candidate_generator_reimplemented") is False
        and boundary.get("trusted_generate_candidates_used") is True
        and boundary.get("trusted_suppress_candidates_used_for_each_configuration")
        is True
        and boundary.get("first_member_of_every_topology_class_checked_by_full_decode_video")
        is True
        and boundary.get("target_or_score_access") is False,
        "R4 exact-factorization safety boundary drift",
    )
    require(
        exact.get("generation_groups_per_video") == 1008
        and exact.get("generation_reuses_per_video") == 17136
        and exact.get("full_configuration_rows_per_video") == 18145
        and benchmark.get("configuration_count") == 18145
        and benchmark.get("candidate_generation_requests") == 1008
        and benchmark.get("candidate_generation_reuses") == 17136
        and benchmark.get("full_graph_decodes") == 47
        and benchmark.get("action_set_reuses") == 18098
        and benchmark.get("unique_graphs") == 47
        and benchmark.get("metric_rows") == 0
        and benchmark.get("scorer_modules_imported_during_load_production_inputs")
        is True
        and benchmark.get("scorer_backend_opened") is False
        and benchmark.get("metric_functions_invoked") is False
        and benchmark.get("competition_archive_hashed_as_opaque_bytes") is True
        and benchmark.get("archive_member_access") is False
        and benchmark.get("scorer_access") is False
        and benchmark.get("benchmark_script_sha256")
        == EXPECTED_R4_EXACT_BENCHMARK_SCRIPT_SHA256
        and benchmark.get("benchmark_result_sha256")
        == EXPECTED_R4_EXACT_BENCHMARK_RESULT_SHA256
        and benchmark_result.get("runner_sha256")
        == EXPECTED_R4_HISTORICAL_RUNNER_SHA256
        and benchmark_result.get("runner_input_sha256")
        == EXPECTED_R4_HISTORICAL_RUNNER_INPUT_V3_SHA256
        and benchmark_result.get("counts", {}).get("full_graph_decodes") == 47
        and benchmark_result.get("counts", {}).get("action_set_reuses") == 18098
        and benchmark_result.get("target_access") is False
        and benchmark_result.get("gt_access") is False
        and benchmark_result.get("scorer_access") is False
        and benchmark_result.get("archive_member_access") is False,
        "R4 exact-factorization counts or target-free benchmark drift",
    )
    conflict_script_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/g03_r4_actual_conflict_audit.py"
    )
    conflict_result_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/ACTUAL_CONFLICT_AUDIT_RESULT.json"
    )
    require(
        sha256(conflict_script_path) == EXPECTED_R4_CONFLICT_AUDIT_SCRIPT_SHA256
        and sha256(conflict_result_path) == EXPECTED_R4_CONFLICT_AUDIT_RESULT_SHA256,
        "R4 actual conflict-ownership evidence drift",
    )
    conflict = load_json(conflict_result_path)
    conflict_bindings = conflict.get("bindings", {})
    conflict_audit = conflict.get("conflict_audit", {})
    minimal_amendment = conflict.get("minimal_amendment", {})
    require(
        conflict.get("status")
        == "BLOCK_PRODUCTION_REQUIRES_CONFLICT_OWNERSHIP_AMENDMENT"
        and conflict.get("identity") == "train:6bba_57b7cc1e"
        and conflict.get("configuration_name") == "t035_g100_l0_dp0_gp0_n0_call"
        and conflict_bindings.get("diagnostic_script_sha256")
        == EXPECTED_R4_CONFLICT_AUDIT_SCRIPT_SHA256
        and conflict_bindings.get("pretarget_plan_v3_sha256")
        == EXPECTED_R4_HISTORICAL_PRETARGET_PLAN_V3_SHA256
        and conflict_bindings.get("decoder_sha256")
        == EXPECTED_R4_HISTORICAL_DECODER_SHA256
        and conflict.get("existing_suppression", {}).get("accepted_events") == 11829
        and conflict_audit.get("shared_removed_edge_count") == 85
        and conflict_audit.get("maximum_removed_edge_multiplicity") == 2
        and conflict_audit.get("conflict_safe_after_deterministic_ranked_ownership")
        == 11749
        and conflict_audit.get("rejected_later_events") == 80
        and minimal_amendment.get("configuration_rows_names_and_axes_preserved")
        is True
        and minimal_amendment.get("predictions_and_suppression_counts_change") is True
        and minimal_amendment.get("explicit_owner_authorization_required_by_safety_gate")
        is True
        and all(value is False for key, value in conflict.get("access", {}).items() if key != "metric_rows")
        and conflict.get("access", {}).get("metric_rows") == 0,
        "R4 actual conflict-ownership blocker accounting drift",
    )
    all_grid_script_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/g03_r4_all_grid_conflict_audit.py"
    )
    all_grid_result_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/ALL_GRID_CONFLICT_AUDIT_RESULT.json"
    )
    require(
        sha256(all_grid_script_path)
        == EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_SCRIPT_SHA256
        and sha256(all_grid_result_path)
        == EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_RESULT_SHA256,
        "R4 all-grid conflict-ownership evidence drift",
    )
    all_grid = load_json(all_grid_result_path)
    all_grid_bindings = all_grid.get("bindings", {})
    all_grid_summary = all_grid.get("all_grid_summary", {})
    representative_worst = all_grid.get("representative_worst", {})
    all_grid_access = all_grid.get("access_provenance", {})
    all_grid_governance = all_grid.get("governance_conclusion", {})
    require(
        all_grid.get("status")
        == "BLOCK_PRODUCTION_REQUIRES_AUTHORIZED_CONFLICT_OWNERSHIP_AMENDMENT"
        and all_grid.get("identity") == "train:6bba_57b7cc1e"
        and all_grid_bindings.get("audit_script", {}).get("sha256")
        == EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_SCRIPT_SHA256
        and all_grid_bindings.get("pretarget_plan_v3", {}).get("sha256")
        == EXPECTED_R4_HISTORICAL_PRETARGET_PLAN_V3_SHA256
        and all_grid_bindings.get("decoder", {}).get("sha256")
        == EXPECTED_R4_HISTORICAL_DECODER_SHA256
        and all_grid_summary.get("processed_active_rows") == 18144
        and all_grid_summary.get("sequential_invalid_rows") == 1350
        and all_grid_summary.get("topology_affected_rows") == 1350
        and all_grid_summary.get("accepted_count_changed_rows") == 1269
        and all_grid_summary.get("same_count_changed_topology_rows") == 81
        and representative_worst.get("configuration_name")
        == "t035_g100_l0_dp0_gp0_n0_call"
        and representative_worst.get("accepted_before") == 11829
        and representative_worst.get("accepted_after_integrated") == 11775
        and representative_worst.get("naive_postfilter_count") == 11749
        and representative_worst.get("integrated_conflict_rejections") == 92
        and representative_worst.get("integrated_replacement_events") == 36
        and representative_worst.get("shared_conflict_edges") == 85
        and all_grid_access.get("target_access") is False
        and all_grid_access.get("ground_truth_access") is False
        and all_grid_access.get("scorer_backend_access") is False
        and all_grid_access.get("metric_function_access") is False
        and all_grid_access.get("target_archive_member_access") is False
        and all_grid_access.get("metric_rows") == 0
        and all_grid_governance.get("decoded_graphs_and_predicted_forks_change")
        is True
        and all_grid_governance.get("existing_implementation_amendment_must_be_superseded")
        is True,
        "R4 all-grid conflict counts, provenance, or governance boundary drift",
    )
    amended_grid_script_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/g03_r4_amended_all_grid_audit.py"
    )
    amended_grid_result_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/AMENDED_ALL_GRID_AUDIT_RESULT.json"
    )
    require(
        sha256(amended_grid_script_path)
        == EXPECTED_R4_AMENDED_ALL_GRID_AUDIT_SCRIPT_SHA256
        and sha256(amended_grid_result_path)
        == EXPECTED_R4_AMENDED_ALL_GRID_AUDIT_RESULT_SHA256,
        "R4 amended all-grid evidence identity drift",
    )
    amended_grid = load_json(amended_grid_result_path)
    amended_full = amended_grid.get("full_grid", {})
    amended_worst = amended_grid.get("production_worst", {})
    amended_oracle = amended_grid.get("incident_conflict_oracle", {})
    amended_order = amended_grid.get("suppression_order_oracle", {})
    amended_access = amended_grid.get("access_provenance", {})
    require(
        amended_grid.get("status")
        == "PASS_TARGET_FREE_AMENDED_ALL_GRID_SEMANTICS"
        and amended_grid.get("bindings", {}).get("decoder", {}).get("sha256")
        == EXPECTED_R4_DECODER_SHA256
        and amended_grid.get("bindings", {}).get("runner", {}).get("sha256")
        == EXPECTED_R4_OOF_RUNNER_SHA256
        and amended_full.get("processed_active_rows") == 18144
        and amended_full.get("sequential_invalid_rows") == 1350
        and amended_full.get("accepted_count_changed_rows") == 1269
        and amended_full.get("same_count_changed_topology_rows") == 81
        and amended_full.get("accepted_shared_conflict_edges") == 0
        and amended_full.get("indexed_conflict_queries_checked") == 1860786
        and amended_worst.get("old_node_only_accepted") == 11829
        and amended_worst.get("amended_production_accepted") == 11775
        and amended_worst.get("replacement_events_admitted") == 36
        and amended_worst.get("suppression", {}).get(
            "rejected_conflict_edge_ownership"
        )
        == 92
        and amended_worst.get(
            "naive_11749_explicitly_rejected_as_amended_semantics"
        )
        is True
        and all(
            amended_worst.get("caps", {}).get(str(cap), {}).get("unchanged_from_old")
            is True
            for cap in (1, 2, 3, 5, 8)
        )
        and amended_oracle.get(
            "all_accepted_indexed_queries_match_independent_index"
        )
        is True
        and amended_oracle.get("literal_full_edge_scan_queries_checked") == 512
        and amended_order.get("dynamic_rejected_event_reserves_nothing") is True
        and amended_order.get("required_order")
        == [
            "exact_equivalence",
            "duplicate_temporal_window",
            "node_ownership",
            "conflict_edge_ownership",
            "per_video_cap",
            "accept_and_reserve",
        ]
        and amended_access.get("target_access") is False
        and amended_access.get("ground_truth_access") is False
        and amended_access.get("scorer_backend_access") is False
        and amended_access.get("metric_function_access") is False
        and amended_access.get("competition_archive_member_access") is False
        and amended_access.get("metric_rows") == 0,
        "R4 amended all-grid semantics, oracle, or access gate drift",
    )
    amended_worst_script_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/g03_r4_amended_actual_worst_benchmark.py"
    )
    amended_worst_test_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/test_g03_r4_amended_actual_worst_benchmark.py"
    )
    amended_worst_result_path = (
        root
        / "reports/g03/screening/wave_r/R4_artifacts/oof/benchmarks/AMENDED_ACTUAL_WORST_BENCHMARK_RESULT.json"
    )
    require(
        sha256(amended_worst_script_path)
        == EXPECTED_R4_AMENDED_WORST_BENCHMARK_SCRIPT_SHA256
        and sha256(amended_worst_test_path)
        == EXPECTED_R4_AMENDED_WORST_BENCHMARK_TEST_SHA256
        and sha256(amended_worst_result_path)
        == EXPECTED_R4_AMENDED_WORST_BENCHMARK_RESULT_SHA256,
        "R4 amended actual-worst evidence identity drift",
    )
    runtime = load_json(amended_worst_result_path)
    runtime_counts = runtime.get("counts", {})
    runtime_performance = runtime.get("performance", {})
    runtime_graph = runtime.get("graph_contract", {})
    runtime_access = runtime.get("access_audit", {})
    runtime_materialization = runtime.get("materialization_receipt", {})
    require(
        runtime.get("status")
        == "PASS_TARGET_FREE_AMENDED_ACTUAL_WORST_ONE_PASS"
        and runtime.get("bindings", {}).get("decoder", {}).get("sha256")
        == EXPECTED_R4_DECODER_SHA256
        and runtime.get("bindings", {}).get("runner", {}).get("sha256")
        == EXPECTED_R4_OOF_RUNNER_SHA256
        and runtime_counts.get("raw_events") == 42235
        and runtime_counts.get("accepted_events") == 11775
        and runtime_counts.get("rejected_conflict_edge_ownership") == 92
        and runtime_counts.get("production_one_pass_invocations") == 1
        and runtime_counts.get("full_graph_constructions") == 1
        and runtime_counts.get("full_graph_hashes") == 1
        and decimal(runtime_performance.get("one_pass_seconds"), "R4 worst wall")
        < Decimal("30")
        and int(runtime_performance.get("process_peak_rss_bytes", 2**63))
        < 1024**3
        and int(runtime_performance.get("swap_growth_bytes", 2**63))
        < 256 * 1024**2
        and runtime_performance.get("external_kill_enforced") is True
        and runtime_performance.get("measurement_method", {}).get("wall_clock")
        == "time.perf_counter_without_python_allocation_tracing"
        and runtime_graph.get("direct_edge_algebra_sha256")
        == runtime_graph.get("decoded_edge_set_sha256")
        and runtime_graph.get("node_set_preserved") is True
        and runtime_graph.get("edge_ownership_coherent") is True
        and runtime_materialization.get("compiler_source_sha256")
        == EXPECTED_R4_ACTION_COMPILER_SHA256
        and runtime_materialization.get("graph_model_source_sha256")
        == EXPECTED_R4_GRAPH_MODEL_SHA256
        and runtime_materialization.get("incident_index_built_once") is True
        and all(
            runtime_access.get(field) == 0
            for field in (
                "archive_accesses",
                "backend_accesses",
                "cache_publications",
                "gt_accesses",
                "metric_rows",
                "network_accesses",
                "official_eval_imports",
                "scoreall_imports",
                "scorer_accesses",
                "tracking_cellmot_imports",
                "worker_writes",
            )
        ),
        "R4 amended actual-worst runtime, provenance, or access gate drift",
    )
    semantic_receipt_path = (
        root
        / "reports/g03/screening/wave_r/R4_SEMANTIC_RUNTIME_AMENDMENT_RECEIPT.json"
    )
    require(
        sha256(semantic_receipt_path)
        == EXPECTED_R4_SEMANTIC_RUNTIME_RECEIPT_SHA256,
        "R4 semantic/runtime amendment receipt identity drift",
    )
    semantic_receipt = load_json(semantic_receipt_path)
    require(
        semantic_receipt.get("status")
        == "RESEALED_TARGET_BLIND_PENDING_FINAL_INDEPENDENT_PRE_SCORE_AUDIT"
        and semantic_receipt.get("authorization", {}).get("amendment_v2_sha256")
        == EXPECTED_R4_DECODER_AMENDMENT_SHA256
        and semantic_receipt.get("source_seal", {}).get("runner_sha256")
        == EXPECTED_R4_OOF_RUNNER_SHA256
        and semantic_receipt.get("contract_seal", {}).get(
            "pretarget_plan_v4_sha256"
        )
        == EXPECTED_R4_PRETARGET_PLAN_SHA256
        and semantic_receipt.get("contract_seal", {}).get("runner_input_v4_sha256")
        == EXPECTED_R4_RUNNER_INPUT_SHA256
        and semantic_receipt.get("cache_provenance", {}).get("old_receipts_accepted")
        is False
        and semantic_receipt.get("cache_provenance", {}).get("old_cache_deleted")
        is False
        and semantic_receipt.get("campaign_state", {}).get("development_score_exposed")
        is False,
        "R4 semantic/runtime reseal receipt drift",
    )
    semantics = decoder_amendment.get("binding_semantics", {})
    require(
        semantics.get("ranked_acceptance_order")
        == [
            "exact_equivalence_suppression",
            "duplicate_temporal_window_suppression",
            "node_ownership_check",
            "frozen_baseline_incident_conflict_edge_ownership_check",
            "per_video_cap_check",
            "accept_and_reserve_nodes_and_conflict_edges",
        ]
        and "Rejected candidates reserve neither" in str(
            semantics.get("reservation_rule", "")
        )
        and "once per video" in str(semantics.get("incident_index", ""))
        and "exactly one full materialized graph" in str(
            semantics.get("one_pass_installation", "")
        ),
        "R4 amended decoder implementation semantics are incomplete",
    )
    return {
        "active_configurations": 18_144,
        "total_configurations": 18_145,
        "configuration_sha256": EXPECTED_R4_CONFIG_SHA256,
        "manifest_sha256": EXPECTED_R4_MANIFEST_SHA256,
        "protocol_sha256": EXPECTED_R4_PROTOCOL_SHA256,
        "population_sha256": EXPECTED_R4_POPULATION_SHA256,
        "population_counts": {"development": 69, "held_fit": 36, "hidden_test": 4},
        "population_amendment_sha256": EXPECTED_R4_POPULATION_AMENDMENT_SHA256,
        "decoder_amendment_sha256": EXPECTED_R4_DECODER_AMENDMENT_SHA256,
        "decoder_sha256": EXPECTED_R4_DECODER_SHA256,
        "bank_extension_sha256": EXPECTED_R4_BANK_EXTENSION_SHA256,
        "bank_repair_sha256": EXPECTED_R4_BANK_REPAIR_SHA256,
        "oof_orchestrator_sha256": EXPECTED_R4_OOF_ORCHESTRATOR_SHA256,
        "oof_runner_sha256": EXPECTED_R4_OOF_RUNNER_SHA256,
        "input_generator_sha256": EXPECTED_R4_INPUT_GENERATOR_SHA256,
        "dev69_seal_sha256": EXPECTED_R4_DEV69_SEAL_SHA256,
        "pretarget_plan_sha256": EXPECTED_R4_PRETARGET_PLAN_SHA256,
        "runner_input_sha256": EXPECTED_R4_RUNNER_INPUT_SHA256,
        "factorization_repair_sha256": EXPECTED_R4_FACTORIZATION_REPAIR_SHA256,
        "conflict_audit_result_sha256": EXPECTED_R4_CONFLICT_AUDIT_RESULT_SHA256,
        "all_grid_conflict_audit_result_sha256": EXPECTED_R4_ALL_GRID_CONFLICT_AUDIT_RESULT_SHA256,
        "amended_all_grid_audit_result_sha256": EXPECTED_R4_AMENDED_ALL_GRID_AUDIT_RESULT_SHA256,
        "amended_actual_worst_result_sha256": EXPECTED_R4_AMENDED_WORST_BENCHMARK_RESULT_SHA256,
        "semantic_runtime_receipt_sha256": EXPECTED_R4_SEMANTIC_RUNTIME_RECEIPT_SHA256,
    }


def _candidate_result(value: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping):
        raise WaveRValidationError("R1 result lacks candidate object")
    return candidate


def validate_r1(root: Path) -> dict[str, Any]:
    path = root / "reports/g03/screening/wave_r/R1_ZERO_SHOT_RESULT.json"
    require(path.is_file(), "R1 zero-shot result is pending")
    require(sha256(path) == EXPECTED_R1_RESULT_SHA256, "R1 final-result identity drift")
    value = load_json(path)
    reference = value.get("decision_reference", {})
    require(isinstance(reference, Mapping), "R1 decision_reference is missing")
    require(
        decimal(reference.get("best_known_score"), "R1 best") == BEST_KNOWN_SCORE,
        "R1 best-known reference drift",
    )
    require(
        decimal(reference.get("neighborhood_floor"), "R1 floor") == NEIGHBORHOOD_FLOOR,
        "R1 neighborhood-floor arithmetic drift",
    )
    candidate = _candidate_result(value)
    score = decimal(candidate.get("score"), "R1 score")
    branch = str(candidate.get("branch", "")).upper().replace("BRANCH_", "")
    require(branch == expected_branch(score, new_best=score > BEST_KNOWN_SCORE), "R1 branch mismatch")
    for field in ("adj_edge_jaccard", "edge_jaccard", "node_recall"):
        require(candidate.get(field) is not None, f"R1 missing score component: {field}")
    for field in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn"):
        require(int(candidate.get(field, -1)) >= 0, f"R1 invalid count: {field}")
    require(candidate.get("division_jaccard") is not None, "R1 missing division Jaccard")
    execution = value.get("execution", {})
    require(isinstance(execution, Mapping), "R1 execution receipt missing")
    require(decimal(execution.get("gpu_h", -1), "R1 GPU hours") == 0, "R1 used GPU")
    require(execution.get("trained") is False, "R1 must be zero-shot/no-training")
    provenance = value.get("provenance", {})
    require(isinstance(provenance, Mapping), "R1 provenance missing")
    require(provenance.get("sealed_before_score") is True, "R1 was not sealed before scoring")
    for field in (
        "config_sha256",
        "adapter_weight_sha256",
        "archive_sha256",
        "association_bank_sha256",
        "aggregate_graph_sha256",
    ):
        require(
            isinstance(provenance.get(field), str) and len(str(provenance[field])) == 64,
            f"R1 provenance missing SHA-256: {field}",
        )
    validation_path = root / "reports/g03/screening/wave_r/R1_VALIDATION_RECEIPT.json"
    require(
        sha256(validation_path) == EXPECTED_R1_VALIDATION_SHA256,
        "R1 terminal-validation identity drift",
    )
    validation = load_json(validation_path)
    require(
        validation.get("status") == "PASS_R1_MEASUREMENT_AND_R4_BANK_HANDOFF",
        "R1 terminal validation did not pass",
    )
    accounting = validation.get("protocol_accounting", {})
    require(
        accounting.get("target_access_attempt_count") == 2
        and accounting.get("scientific_configuration_count") == 1
        and accounting.get("candidate_score_exposure_count") == 1
        and accounting.get("technical_receipt_recovery_count") == 1
        and accounting.get("post_result_tuning") is False,
        "R1 serialization-incident accounting drift",
    )
    validated_candidate = validation.get("candidate", {})
    for field in (
        "score",
        "adj_edge_jaccard",
        "edge_jaccard",
        "node_recall",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "division_jaccard",
    ):
        require(
            validated_candidate.get(field) == candidate.get(field),
            f"R1 result/validation mismatch: {field}",
        )
    require(
        validation.get("artifact_chain", {}).get("final_result", {}).get("sha256")
        == EXPECTED_R1_RESULT_SHA256,
        "R1 validation does not bind final result",
    )
    posthoc_path = (
        root
        / "reports/g03/screening/wave_r/R1_POSTHOC_REPRODUCIBILITY_AMENDMENT.json"
    )
    require(
        sha256(posthoc_path) == EXPECTED_R1_POSTHOC_AMENDMENT_SHA256,
        "R1 post-hoc reproducibility amendment identity drift",
    )
    posthoc = load_json(posthoc_path)
    scope = posthoc.get("scope", {})
    require(
        posthoc.get("status")
        == "POSTHOC_BYTES_RETAINED_NOT_RETROACTIVELY_PREPINNED"
        and scope.get("verified_r1_score_changed") is False
        and scope.get("new_target_access") is False
        and scope.get("rerun_allowed") is False
        and scope.get("retroactive_prepin_claimed") is False,
        "R1 post-hoc provenance scope drift",
    )
    require(
        sha256(root / "scripts/g03_r1_recover_score_receipt.py")
        == EXPECTED_R1_RECOVERY_HELPER_SHA256
        and sha256(root / "scripts/g03_r1_finalize_exact.py")
        == EXPECTED_R1_FINALIZER_HELPER_SHA256,
        "R1 retained helper identity drift",
    )
    helpers = posthoc.get("retained_helpers", {})
    require(
        helpers.get("recovery", {}).get("sha256")
        == EXPECTED_R1_RECOVERY_HELPER_SHA256
        and helpers.get("exact_finalizer", {}).get("sha256")
        == EXPECTED_R1_FINALIZER_HELPER_SHA256,
        "R1 post-hoc amendment does not bind retained helpers",
    )
    allocation_path = (
        root / "reports/g03/screening/wave_r/WAVE_R_ALLOCATION_REPORT_20260801.json"
    )
    allocation = load_json(allocation_path)
    require(
        allocation.get("status") == "COMPLETE_BRANCH_B_NO_GPU_R4_CPU_ACTIVE",
        "Wave-R allocation is not terminal Branch B",
    )
    allocation_decision = allocation.get("allocation", {})
    require(
        allocation.get("r1_candidate", {}).get("branch") == "B"
        and allocation_decision.get("eligible_candidates") == []
        and decimal(allocation_decision.get("allocated_t4_h"), "allocated T4-h") == 0
        and decimal(allocation_decision.get("wave_r_gpu_h_consumed"), "Wave-R GPU-h") == 0,
        "Wave-R Branch-B allocation drift",
    )
    return {
        "score": str(score),
        "branch": branch,
        "beats_best": score > BEST_KNOWN_SCORE,
        "result_sha256": sha256(path),
        "validation_sha256": EXPECTED_R1_VALIDATION_SHA256,
        "posthoc_reproducibility_sha256": EXPECTED_R1_POSTHOC_AMENDMENT_SHA256,
        "allocated_t4_h": "0",
    }


def validate(root: Path, *, allow_pending_r1: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "celltrack-alpha.g03-wave-r-validation.v1",
        "policy": validate_policy_text(root),
        "r2": validate_r2(root),
        "r3": validate_r3(root),
        "r4": validate_r4(root),
    }
    try:
        report["r1"] = validate_r1(root)
        report["status"] = "PASS_COMPLETE_WAVE_R_EVIDENCE"
    except WaveRValidationError as error:
        if not allow_pending_r1 or str(error) != "R1 zero-shot result is pending":
            raise
        report["r1"] = {"status": "PENDING"}
        report["status"] = "PASS_PREREQUISITES_R1_PENDING"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-pending-r1", action="store_true")
    args = parser.parse_args()
    try:
        report = validate(args.root.resolve(), allow_pending_r1=args.allow_pending_r1)
    except (OSError, ValueError, WaveRValidationError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def policy() -> dict:
    text = (ROOT / "LMT.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- LMT_POLICY_JSON_BEGIN -->\s*```json\s*(.*?)\s*```\s*<!-- LMT_POLICY_JSON_END -->",
        text,
        flags=re.DOTALL,
    )
    assert match, "canonical LMT JSON block missing"
    return json.loads(match.group(1))


def test_paid_compute_authority_is_bounded_and_anti_sharding() -> None:
    external = policy()["external_scope"]
    assert external["paid_compute_auto_approved_if_conservative_max_usd_strictly_below"] > 0
    assert external["paid_compute_at_or_above_usd_requires_explicit_human_approval"] is True
    assert external["paid_compute_threshold_evasion_by_sharding_forbidden"] is True


def test_kaggle_scientific_and_integrity_gates_remain_binding() -> None:
    kaggle = policy()["kaggle"]
    assert kaggle["productive_candidate_required"] is True
    assert kaggle["serial_launch_lock_required"] is True
    assert kaggle["authoritative_quota_reconciliation_required"] is True
    assert kaggle["one_shot_receipt_required"] is True
    assert kaggle["threshold_evasion_by_sharding_forbidden"] is True

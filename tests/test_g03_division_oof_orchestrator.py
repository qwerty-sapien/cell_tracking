from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "g03_division_oof_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("g03_division_oof_orchestrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)


def file_ref(path: Path, content: str = "sealed\n") -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="ascii")
    return {"path": str(path), "sha256": MODULE.decode.sha256_file(path)}


def syntax_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": "a" * 64}


def execution_sources(source: dict[str, str]) -> dict[str, object]:
    runner_path = SCRIPT.with_name("g03_division_oof_runner.py").resolve()
    runner_source = {
        "path": str(runner_path),
        "sha256": MODULE.decode.sha256_file(runner_path),
    }
    decoder_source = {
        "path": str(Path(MODULE.decode.__file__).resolve()),
        "sha256": MODULE.decode.sha256_file(Path(MODULE.decode.__file__).resolve()),
    }
    orchestrator_source = {
        "path": str(SCRIPT.resolve()),
        "sha256": MODULE.decode.sha256_file(SCRIPT.resolve()),
    }
    return {
        "runner": runner_source,
        "decoder": decoder_source,
        "orchestrator": orchestrator_source,
        "scoreall": dict(source),
        "scoreall_dependencies": {
            relative: dict(source) for relative in MODULE.SCORER_DEPENDENCY_PATHS
        },
        "one_pass_dependencies": {
            relative: {
                "path": str((ROOT / relative).resolve()),
                "sha256": MODULE.decode.sha256_file(ROOT / relative),
            }
            for relative in MODULE.ONE_PASS_DEPENDENCY_PATHS
        },
    }


def role_rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {"identity": f"{prefix}{index:03d}", "group_id": f"group-{prefix}-{index:03d}"}
        for index in range(count)
    ]


def population() -> dict[str, object]:
    development = role_rows("train:6bba_dev_", 69)
    held = role_rows("train:44b6_held_", 36)
    hidden = role_rows("test:hidden_", 4)
    audit = role_rows("train:audit_", 20)
    folds = {str(index): [] for index in range(5)}
    for index, row in enumerate(development):
        folds[str(index % 5)].append(dict(row))
    return {
        "roles": {
            MODULE.DEVELOPMENT_ROLE: {"videos": development},
            MODULE.HELD_ROLE: {"videos": held},
            MODULE.HIDDEN_ROLE: {"videos": hidden},
            MODULE.AUDIT_ROLE: {"videos": audit},
        },
        "outer_folds": folds,
    }


def coverage_contract(root: Path, pop: dict[str, object], *, materialize: bool) -> dict[str, object]:
    r1_name = "R1_SCIENTIFIC"
    inference = (
        file_ref(root / "r1-inference.json")
        if materialize
        else syntax_ref(root / "missing-r1-inference.json")
    )
    records = []
    roles = pop["roles"]
    assert isinstance(roles, dict)
    for role in MODULE.FIT105_ROLES:
        role_record = roles[role]
        assert isinstance(role_record, dict)
        for row in role_record["videos"]:
            identity = row["identity"]
            slug = identity.replace(":", "_")
            refs = {}
            for kind in ("association_bank", "parent_graph", "reference_graph"):
                path = root / kind / f"{slug}.json"
                refs[kind] = file_ref(path, f"{identity}:{kind}\n") if materialize else syntax_ref(path)
            records.append(
                {
                    "identity": identity,
                    "role": role,
                    "r1_configuration_name": r1_name,
                    **refs,
                }
            )
    return {
        "sealed_before_development_scoring": True,
        "r1_scientific_configuration": {"name": r1_name, "inference_seal": inference},
        "records": records,
    }


def metrics(
    score: float = 0.8,
    *,
    division: tuple[int, int, int] = (1, 0, 1),
    edge: tuple[int, int, int] = (8, 1, 1),
    adjusted: float | None = None,
    node: float = 0.9,
    forks: int = 1,
) -> dict[str, object]:
    def jaccard(counts: tuple[int, int, int]) -> float:
        denominator = sum(counts)
        return 1.0 if denominator == 0 else counts[0] / denominator

    division_jaccard = jaccard(division)
    adjusted_value = (
        score - (0.1 * division_jaccard if sum(division) > 0 else 0.0)
        if adjusted is None
        else adjusted
    )
    return {
        "combined_graph_score": score,
        "node_recall": node,
        "predicted_forks": forks,
        "edge": {
            "tp": edge[0],
            "fp": edge[1],
            "fn": edge[2],
            "jaccard": jaccard(edge),
            "adjusted_jaccard": adjusted_value,
        },
        "division": {
            "tp": division[0],
            "fp": division[1],
            "fn": division[2],
            "jaccard": division_jaccard,
        },
    }


def small_plan() -> dict[str, object]:
    folds = {str(index): [f"dev-{index}-a", f"dev-{index}-b"] for index in range(5)}
    rows = []
    for outer in MODULE.FOLD_IDS:
        inner = [fold for fold in MODULE.FOLD_IDS if fold != outer]
        rows.append(
            {
                "outer_fold": outer,
                "outer_identities": folds[outer],
                "inner_folds": inner,
                "inner_identities": [identity for fold in inner for identity in folds[fold]],
                "inner_fold_identities": {fold: folds[fold] for fold in inner},
            }
        )
    return {
        "outer_folds": rows,
        "execution_source_bindings": {
            "runner": execution_sources({"path": "unused", "sha256": "0" * 64})[
                "runner"
            ]
        },
        "inner_tie_order": ["sealed-order"],
        "deployable_rule": (
            "highest outer-winner count, then lexicographically smallest config name"
        ),
    }


def inner_receipt(
    plan: dict[str, object],
    scorer: dict[str, str],
    rows: list[dict[str, object]],
    *,
    outer: str = "0",
    plan_sha: str = "1" * 64,
) -> dict[str, object]:
    fold = MODULE._fold_plan(plan, outer)
    parent_folds = {
        fold_id: parent_fold(list(fold["inner_fold_identities"][fold_id]))
        for fold_id in fold["inner_folds"]
    }
    return {
        "schema": MODULE.INNER_SCORE_SCHEMA,
        "status": "SEALED_INNER_OOF_SCORES",
        "pretarget_plan_sha256": plan_sha,
        "outer_fold": outer,
        "inner_folds": list(fold["inner_folds"]),
        "evaluated_identities": list(fold["inner_identities"]),
        "configuration_count": len(rows),
        "scorer_source": scorer,
        "execution_sources": execution_sources(scorer),
        "parent_folds": parent_folds,
        "rows": rows,
    }


def candidate_fold(
    identities: list[str],
    value: dict[str, object] | None = None,
    *,
    duplicates: int = 0,
) -> dict[str, object]:
    return {
        "evaluated_identities": identities,
        "graph_sha256_by_identity": {
            identity: hashlib.sha256(f"candidate:{identity}".encode()).hexdigest()
            for identity in identities
        },
        "duplicate_survivors": duplicates,
        "metrics": copy.deepcopy(metrics(0.84) if value is None else value),
    }


def parent_fold(
    identities: list[str], value: dict[str, object] | None = None
) -> dict[str, object]:
    parent_metrics = metrics(
        0.78,
        division=(0, 0, 2),
        edge=(7, 1, 2),
        node=0.89,
        forks=0,
    ) if value is None else value
    return {
        "reference_label": MODULE.SAME_POPULATION_REFERENCE_LABEL,
        "evaluated_identities": identities,
        "graph_sha256_by_identity": {
            identity: hashlib.sha256(f"parent:{identity}".encode()).hexdigest()
            for identity in identities
        },
        "gt_division_count": int(parent_metrics["division"]["tp"])
        + int(parent_metrics["division"]["fn"]),
        "metrics": copy.deepcopy(parent_metrics),
    }


def inner_row(name: str, pooled: dict[str, object], fold_ids: list[str]) -> dict[str, object]:
    fold_evaluations = {
        fold: candidate_fold(
            [f"dev-{fold}-a", f"dev-{fold}-b"], copy.deepcopy(pooled)
        )
        for fold in fold_ids
    }
    exact_pooled = MODULE._pool_fold_metrics(
        [
            {
                "evaluated_identities": fold_evaluations[fold]["evaluated_identities"],
                "candidate_metrics": fold_evaluations[fold]["metrics"],
            }
            for fold in fold_ids
        ],
        "candidate",
    )
    return {
        "configuration_name": name,
        "fold_evaluations": fold_evaluations,
        "pooled_inner_oof": exact_pooled,
    }


def passing_development_gate() -> dict[str, object]:
    folds = []
    for fold in MODULE.FOLD_IDS:
        identities = [f"dev-{fold}-a", f"dev-{fold}-b"]
        candidate = candidate_fold(identities)
        parent = parent_fold(identities)
        folds.append(
            {
                "outer_fold": fold,
                "outer_result_sha256": chr(97 + int(fold)) * 64,
                "configuration_name": "cfg-a",
                "evaluated_identities": identities,
                "gt_division_count": parent["gt_division_count"],
                "candidate_graph_sha256_by_identity": candidate["graph_sha256_by_identity"],
                "parent_graph_sha256_by_identity": parent["graph_sha256_by_identity"],
                "candidate_metrics": candidate["metrics"],
                "parent_metrics": parent["metrics"],
                "duplicate_survivors": 0,
                "deltas": MODULE._metric_deltas(candidate["metrics"], parent["metrics"]),
            }
        )
    return MODULE._development_gate_from_folds(folds)


def deployable_artifact(config: str = "cfg-a", plan_sha: str = "1" * 64) -> dict[str, object]:
    gate = passing_development_gate()
    for fold in gate["folds"]:
        fold["configuration_name"] = config
    gate = MODULE._development_gate_from_folds(gate["folds"])
    return MODULE._artifact(
        MODULE.DEPLOYABLE_SCHEMA,
        MODULE.DEPLOYABLE_STATUS,
        {
            "pretarget_plan_sha256": plan_sha,
            "outer_results": [
                {
                    "outer_fold": fold["outer_fold"],
                    "configuration_name": config,
                    "result_sha256": fold["outer_result_sha256"],
                }
                for fold in gate["folds"]
            ],
            "outer_winner_counts": {config: 5},
            "configuration_name": config,
            "selection_rule": (
                "highest outer-winner count, then lexicographically smallest config name"
            ),
            "outer_or_held_metrics_used_for_modal_tie": False,
            "metric_repooling_for_deployable_selection": False,
            "development_gate": gate,
        },
    )


def test_incomplete_coverage_fails_before_opening_missing_bank_files(tmp_path: Path) -> None:
    pop = population()
    coverage = coverage_contract(tmp_path, pop, materialize=False)
    coverage["records"] = coverage["records"][:-1]
    with pytest.raises(MODULE.DivisionOOFError, match="exact 105-bank"):
        MODULE.verify_exact_fit105_coverage(coverage, population=pop, base=tmp_path)


def test_exact_fit105_coverage_hashes_all_three_artifact_classes(tmp_path: Path) -> None:
    pop = population()
    coverage = coverage_contract(tmp_path, pop, materialize=True)
    rows, identity, r1 = MODULE.verify_exact_fit105_coverage(
        coverage, population=pop, base=tmp_path
    )
    assert len(rows) == 105
    assert sum(row["role"] == MODULE.DEVELOPMENT_ROLE for row in rows) == 69
    assert sum(row["role"] == MODULE.HELD_ROLE for row in rows) == 36
    assert len(identity) == 64
    assert r1["name"] == "R1_SCIENTIFIC"
    assert all(set(row) >= {"association_bank", "parent_graph", "reference_graph"} for row in rows)


def test_coverage_refuses_audit20_role_and_mixed_r1_configuration(tmp_path: Path) -> None:
    pop = population()
    coverage = coverage_contract(tmp_path, pop, materialize=False)
    audit = copy.deepcopy(coverage)
    audit["records"][0]["role"] = MODULE.AUDIT_ROLE
    with pytest.raises(MODULE.DivisionOOFError, match="forbidden role"):
        MODULE.verify_exact_fit105_coverage(audit, population=pop, base=tmp_path)
    mixed = copy.deepcopy(coverage)
    mixed["records"][0]["r1_configuration_name"] = "OTHER_R1"
    with pytest.raises(MODULE.DivisionOOFError, match="one R1 configuration"):
        MODULE.verify_exact_fit105_coverage(mixed, population=pop, base=tmp_path)


def test_fixed_fold_plan_uses_four_folds_and_never_leaks_outer() -> None:
    rows = MODULE.build_fixed_fold_plan(population())
    assert [row["outer_fold"] for row in rows] == list(MODULE.FOLD_IDS)
    for row in rows:
        assert len(row["inner_folds"]) == 4
        assert set(row["outer_identities"]).isdisjoint(row["inner_identities"])
        assert len(row["outer_identities"]) + len(row["inner_identities"]) == 69


def test_metric_bundle_requires_separate_exact_edge_and_division_counts() -> None:
    value = metrics()
    normalized = MODULE.validate_metric_bundle(value, "synthetic")
    assert set(normalized["edge"]) == {"tp", "fp", "fn", "jaccard", "adjusted_jaccard"}
    assert set(normalized["division"]) == {"tp", "fp", "fn", "jaccard"}
    merged = copy.deepcopy(value)
    merged.pop("division")
    with pytest.raises(MODULE.DivisionOOFError, match="keys drifted"):
        MODULE.validate_metric_bundle(merged, "synthetic")
    inconsistent = copy.deepcopy(value)
    inconsistent["division"]["jaccard"] = 0.9
    with pytest.raises(MODULE.DivisionOOFError, match="disagrees"):
        MODULE.validate_metric_bundle(inconsistent, "synthetic")
    maximum = metrics(1.1, division=(1, 0, 0), adjusted=1.0)
    assert MODULE.validate_metric_bundle(maximum, "official maximum")[
        "combined_graph_score"
    ] == 1.1
    too_large = copy.deepcopy(maximum)
    too_large["combined_graph_score"] = 1.1000001
    with pytest.raises(MODULE.DivisionOOFError, match=r"\[0, 1\.1\]"):
        MODULE.validate_metric_bundle(too_large, "too large")


def test_fold_repooling_uses_official_edge_weights_and_micro_division_counts() -> None:
    first = metrics(
        0.95,
        edge=(8, 1, 1),
        division=(1, 0, 1),
        adjusted=0.9,
    )
    second = metrics(
        0.55,
        edge=(18, 6, 6),
        division=(1, 0, 1),
        adjusted=0.5,
    )
    pooled = MODULE._pool_fold_metrics(
        [
            {"evaluated_identities": ["a"], "candidate_metrics": first},
            {"evaluated_identities": ["b"], "candidate_metrics": second},
        ],
        "candidate",
    )
    assert pooled["edge"]["adjusted_jaccard"] == pytest.approx(0.6)
    assert pooled["division"] == {"tp": 2, "fp": 0, "fn": 2, "jaccard": 0.5}
    assert pooled["combined_graph_score"] == pytest.approx(0.65)


def test_inner_winner_uses_frozen_tie_chain_and_lex_last(tmp_path: Path) -> None:
    plan = small_plan()
    fold_ids = MODULE._fold_plan(plan, "0")["inner_folds"]
    scorer = file_ref(tmp_path / "scorer.py")
    # Same combined score and division Jaccard; cfg-a has fewer division FP.
    receipt = inner_receipt(
        plan,
        scorer,
        [
            inner_row("cfg-b", metrics(0.9, division=(2, 2, 0)), fold_ids),
            inner_row("cfg-a", metrics(0.9, division=(1, 0, 1)), fold_ids),
        ],
    )
    winner, _row = MODULE.select_inner_winner(
        plan=plan,
        receipt=receipt,
        configuration_names=("cfg-a", "cfg-b"),
        receipt_base=tmp_path,
    )
    assert winner == "cfg-a"
    tied = inner_receipt(
        plan,
        scorer,
        [
            inner_row("cfg-z", metrics(0.9), fold_ids),
            inner_row("cfg-a", metrics(0.9), fold_ids),
        ],
    )
    winner, _row = MODULE.select_inner_winner(
        plan=plan,
        receipt=tied,
        configuration_names=("cfg-z", "cfg-a"),
        receipt_base=tmp_path,
    )
    assert winner == "cfg-a"


def test_inner_receipt_refuses_outer_leak_missing_config_and_plan_drift(tmp_path: Path) -> None:
    plan = small_plan()
    fold = MODULE._fold_plan(plan, "0")
    scorer = file_ref(tmp_path / "scorer.py")
    row = inner_row("cfg-a", metrics(), fold["inner_folds"])
    receipt = inner_receipt(plan, scorer, [row])
    receipt["evaluated_identities"][0] = fold["outer_identities"][0]
    with pytest.raises(MODULE.DivisionOOFError, match="four tuning folds"):
        MODULE.select_inner_winner(
            plan=plan,
            receipt=receipt,
            configuration_names=("cfg-a",),
            receipt_base=tmp_path,
        )
    complete = inner_receipt(plan, scorer, [row])
    with pytest.raises(MODULE.DivisionOOFError, match="configuration coverage"):
        MODULE.select_inner_winner(
            plan=plan,
            receipt=complete,
            configuration_names=("cfg-a", "cfg-b"),
            receipt_base=tmp_path,
        )
    complete["pretarget_plan_sha256"] = "2" * 64
    with pytest.raises(MODULE.DivisionOOFError, match="different pre-target plan"):
        MODULE.make_inner_winner_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            receipt=complete,
            receipt_sha256="3" * 64,
            configuration_names=("cfg-a",),
            receipt_base=tmp_path,
        )


def test_execution_source_helper_never_aliases_runner_to_scorer(tmp_path: Path) -> None:
    scorer = file_ref(tmp_path / "scorer.py")
    sources = execution_sources(scorer)
    assert sources["runner"] != sources["scoreall"]
    assert Path(sources["runner"]["path"]) == SCRIPT.with_name(
        "g03_division_oof_runner.py"
    ).resolve()


@pytest.mark.parametrize("substitution", ("path", "hash"))
def test_inner_receipt_rejects_runner_source_substitution(
    tmp_path: Path, substitution: str
) -> None:
    plan = small_plan()
    fold = MODULE._fold_plan(plan, "0")
    scorer = file_ref(tmp_path / "scorer.py")
    receipt = inner_receipt(
        plan, scorer, [inner_row("cfg-a", metrics(), fold["inner_folds"])]
    )
    if substitution == "path":
        receipt["execution_sources"]["runner"] = file_ref(
            tmp_path / "counterfeit-runner.py", "# substituted runner bytes\n"
        )
    else:
        receipt["execution_sources"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(MODULE.DivisionOOFError, match="runner"):
        MODULE.select_inner_winner(
            plan=plan,
            receipt=receipt,
            configuration_names=("cfg-a",),
            receipt_base=tmp_path,
        )


def test_outer_result_accepts_only_presealed_winner_on_exact_holdout(tmp_path: Path) -> None:
    plan = small_plan()
    plan_sha = "1" * 64
    winner_sha = "2" * 64
    winner = MODULE._artifact(
        MODULE.INNER_WINNER_SCHEMA,
        "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
        {
            "pretarget_plan_sha256": plan_sha,
            "inner_score_receipt_sha256": "4" * 64,
            "outer_fold": "0",
            "outer_identities": list(MODULE._fold_plan(plan, "0")["outer_identities"]),
            "configuration_name": "cfg-a",
            "pooled_inner_oof": metrics(),
            "selection_order": ["sealed-order"],
            "outer_evaluation_authorized_for_winner_only": True,
            "outer_metric_read_before_winner_seal": False,
        },
    )
    scorer = file_ref(tmp_path / "scorer.py")
    outer_ids = list(MODULE._fold_plan(plan, "0")["outer_identities"])
    candidate = candidate_fold(outer_ids)
    parent = parent_fold(outer_ids)
    receipt = {
        "schema": MODULE.OUTER_SCORE_SCHEMA,
        "status": "SEALED_SINGLE_WINNER_OUTER_HOLDOUT_SCORE",
        "pretarget_plan_sha256": plan_sha,
        "inner_winner_seal_sha256": winner_sha,
        "outer_fold": "0",
        "configuration_name": "cfg-a",
        "evaluated_identities": list(outer_ids),
        "scorer_source": scorer,
        "execution_sources": execution_sources(scorer),
        "candidate": candidate,
        "parent": parent,
        "gt_division_count": parent["gt_division_count"],
    }
    result = MODULE.make_outer_result_artifact(
        plan=plan,
        plan_sha256=plan_sha,
        winner_artifact=winner,
        winner_sha256=winner_sha,
        receipt=receipt,
        receipt_sha256="5" * 64,
        receipt_base=tmp_path,
    )
    assert result["payload"]["configuration_name"] == "cfg-a"
    assert result["payload"]["deltas"] == MODULE._metric_deltas(
        candidate["metrics"], parent["metrics"]
    )
    wrong = copy.deepcopy(receipt)
    wrong["configuration_name"] = "cfg-b"
    with pytest.raises(MODULE.DivisionOOFError, match="sealed winner"):
        MODULE.make_outer_result_artifact(
            plan=plan,
            plan_sha256=plan_sha,
            winner_artifact=winner,
            winner_sha256=winner_sha,
            receipt=wrong,
            receipt_sha256="5" * 64,
            receipt_base=tmp_path,
        )


@pytest.mark.parametrize("substitution", ("path", "hash"))
def test_outer_receipt_rejects_runner_source_substitution(
    tmp_path: Path, substitution: str
) -> None:
    plan = small_plan()
    plan_sha = "1" * 64
    winner_sha = "2" * 64
    winner = MODULE._artifact(
        MODULE.INNER_WINNER_SCHEMA,
        "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
        {
            "pretarget_plan_sha256": plan_sha,
            "inner_score_receipt_sha256": "4" * 64,
            "outer_fold": "0",
            "outer_identities": list(
                MODULE._fold_plan(plan, "0")["outer_identities"]
            ),
            "configuration_name": "cfg-a",
            "pooled_inner_oof": metrics(),
            "selection_order": ["sealed-order"],
            "outer_evaluation_authorized_for_winner_only": True,
            "outer_metric_read_before_winner_seal": False,
        },
    )
    scorer = file_ref(tmp_path / "scorer.py")
    outer_ids = list(MODULE._fold_plan(plan, "0")["outer_identities"])
    parent = parent_fold(outer_ids)
    receipt = {
        "schema": MODULE.OUTER_SCORE_SCHEMA,
        "status": "SEALED_SINGLE_WINNER_OUTER_HOLDOUT_SCORE",
        "pretarget_plan_sha256": plan_sha,
        "inner_winner_seal_sha256": winner_sha,
        "outer_fold": "0",
        "configuration_name": "cfg-a",
        "evaluated_identities": outer_ids,
        "scorer_source": scorer,
        "execution_sources": execution_sources(scorer),
        "candidate": candidate_fold(outer_ids),
        "parent": parent,
        "gt_division_count": parent["gt_division_count"],
    }
    if substitution == "path":
        receipt["execution_sources"]["runner"] = file_ref(
            tmp_path / "counterfeit-runner.py", "# substituted runner bytes\n"
        )
    else:
        receipt["execution_sources"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(MODULE.DivisionOOFError, match="runner"):
        MODULE.make_outer_result_artifact(
            plan=plan,
            plan_sha256=plan_sha,
            winner_artifact=winner,
            winner_sha256=winner_sha,
            receipt=receipt,
            receipt_sha256="5" * 64,
            receipt_base=tmp_path,
        )


def outer_artifact(
    plan: dict[str, object], fold: str, config: str, score: float = 0.5
) -> dict[str, object]:
    identities = list(MODULE._fold_plan(plan, fold)["outer_identities"])
    candidate_metrics = metrics(score)
    parent_metrics = metrics(
        max(0.0, score - 0.06),
        division=(0, 0, 2),
        edge=(7, 1, 2),
        node=0.89,
        forks=0,
    )
    candidate = candidate_fold(identities, candidate_metrics)
    parent = parent_fold(identities, parent_metrics)
    return MODULE._artifact(
        MODULE.OUTER_RESULT_SCHEMA,
        "SEALED_OUTER_HOLDOUT_RESULT",
        {
            "pretarget_plan_sha256": "1" * 64,
            "inner_winner_seal_sha256": (str(int(fold) + 1) * 64)[:64],
            "outer_score_receipt_sha256": (str(int(fold) + 5) * 64)[:64],
            "outer_fold": fold,
            "configuration_name": config,
            "evaluated_identities": identities,
            "candidate": candidate,
            "parent": parent,
            "gt_division_count": parent["gt_division_count"],
            "deltas": MODULE._metric_deltas(candidate_metrics, parent_metrics),
        },
    )


def test_final_deployable_is_modal_then_lex_and_ignores_all_outer_metrics() -> None:
    plan = small_plan()
    configs = ("cfg-z", "cfg-a", "cfg-z", "cfg-a", "cfg-q")
    scores = (0.99, 0.55, 0.98, 0.56, 1.0)
    results = [outer_artifact(plan, fold, config, score) for fold, config, score in zip(MODULE.FOLD_IDS, configs, scores)]
    deploy = MODULE.make_deployable_artifact(
        plan=plan,
        plan_sha256="1" * 64,
        outer_result_artifacts=results,
        outer_result_sha256s=[chr(97 + index) * 64 for index in range(5)],
        configuration_names=("cfg-a", "cfg-q", "cfg-z"),
    )
    assert deploy["payload"]["outer_winner_counts"] == {"cfg-a": 2, "cfg-q": 1, "cfg-z": 2}
    assert deploy["payload"]["configuration_name"] == "cfg-a"
    assert deploy["payload"]["outer_or_held_metrics_used_for_modal_tie"] is False


def test_deployable_refuses_outer_identity_drift_and_unknown_configuration() -> None:
    plan = small_plan()
    results = [outer_artifact(plan, fold, "cfg-a") for fold in MODULE.FOLD_IDS]
    changed_identity = copy.deepcopy(results)
    changed_identity[0]["payload"]["evaluated_identities"][0] = "leaked-inner-video"
    changed_identity[0]["payload_sha256"] = MODULE.decode.canonical_sha256(
        changed_identity[0]["payload"]
    )
    with pytest.raises(MODULE.DivisionOOFError, match="binding or fold uniqueness"):
        MODULE.make_deployable_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            outer_result_artifacts=changed_identity,
            outer_result_sha256s=[chr(97 + index) * 64 for index in range(5)],
            configuration_names=("cfg-a",),
        )
    unknown = copy.deepcopy(results)
    unknown[4]["payload"]["configuration_name"] = "unsealed-config"
    unknown[4]["payload_sha256"] = MODULE.decode.canonical_sha256(unknown[4]["payload"])
    with pytest.raises(MODULE.DivisionOOFError, match="binding or fold uniqueness"):
        MODULE.make_deployable_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            outer_result_artifacts=unknown,
            outer_result_sha256s=[chr(97 + index) * 64 for index in range(5)],
            configuration_names=("cfg-a",),
        )


def _refresh_artifact(artifact: dict[str, object]) -> None:
    artifact["payload_sha256"] = MODULE.decode.canonical_sha256(artifact["payload"])


def _refresh_outer_comparison(artifact: dict[str, object]) -> None:
    payload = artifact["payload"]
    payload["deltas"] = MODULE._metric_deltas(
        payload["candidate"]["metrics"], payload["parent"]["metrics"]
    )
    _refresh_artifact(artifact)


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        ("gt", "all_folds_gt_division_positive"),
        ("fold_delta", "all_fold_combined_attribution_nonnegative"),
        ("tp", "pooled_division_tp_positive"),
        ("jaccard", "pooled_division_jaccard_at_least_0_30"),
        ("duplicates", "duplicate_survivors_zero"),
        ("adjusted", "pooled_adjusted_edge_nonregression"),
        ("raw", "pooled_raw_edge_nonregression"),
        ("node", "pooled_node_nonregression"),
    ],
)
def test_development_gate_refuses_every_preregistered_failure(
    gate: str, expected: str
) -> None:
    plan = small_plan()
    results = [outer_artifact(plan, fold, "cfg-a", 0.8) for fold in MODULE.FOLD_IDS]
    targets = results if gate in {"tp", "jaccard", "adjusted", "raw", "node"} else results[:1]
    for artifact in targets:
        payload = artifact["payload"]
        candidate = payload["candidate"]["metrics"]
        parent = payload["parent"]["metrics"]
        if gate == "gt":
            for side in (candidate, parent):
                side["division"] = {"tp": 0, "fp": 0, "fn": 0, "jaccard": 1.0}
                side["combined_graph_score"] = side["edge"]["adjusted_jaccard"]
            payload["gt_division_count"] = 0
            payload["parent"]["gt_division_count"] = 0
        elif gate == "fold_delta":
            candidate["combined_graph_score"] = parent["combined_graph_score"] - 0.01
            candidate["edge"]["adjusted_jaccard"] = (
                candidate["combined_graph_score"]
                - 0.1 * candidate["division"]["jaccard"]
            )
        elif gate == "tp":
            candidate["division"] = {"tp": 0, "fp": 0, "fn": 2, "jaccard": 0.0}
            candidate["combined_graph_score"] = candidate["edge"]["adjusted_jaccard"]
        elif gate == "jaccard":
            candidate["division"] = {"tp": 1, "fp": 4, "fn": 1, "jaccard": 1 / 6}
            candidate["combined_graph_score"] = (
                candidate["edge"]["adjusted_jaccard"] + 0.1 / 6
            )
        elif gate == "duplicates":
            payload["candidate"]["duplicate_survivors"] = 1
        elif gate == "adjusted":
            candidate["edge"]["adjusted_jaccard"] = (
                parent["edge"]["adjusted_jaccard"] - 0.01
            )
            candidate["combined_graph_score"] = (
                candidate["edge"]["adjusted_jaccard"]
                + 0.1 * candidate["division"]["jaccard"]
            )
        elif gate == "raw":
            candidate["edge"].update(
                {"tp": 6, "fp": 2, "fn": 2, "jaccard": 0.6}
            )
        elif gate == "node":
            candidate["node_recall"] = 0.88
        _refresh_outer_comparison(artifact)
    with pytest.raises(MODULE.DivisionOOFError, match=expected):
        MODULE.make_deployable_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            outer_result_artifacts=results,
            outer_result_sha256s=[chr(97 + index) * 64 for index in range(5)],
            configuration_names=("cfg-a",),
        )


def test_development_gate_binds_exact_references_and_recomputes_claims() -> None:
    gate = passing_development_gate()
    assert gate["campaign_reference"]["score"] == "0.881522069960"
    assert gate["same_population_reference_label"] == MODULE.SAME_POPULATION_REFERENCE_LABEL
    changed = copy.deepcopy(gate)
    changed["campaign_reference"]["score"] = "0.9"
    with pytest.raises(MODULE.DivisionOOFError, match="campaign reference drifted"):
        MODULE.validate_development_gate(changed)
    changed = copy.deepcopy(gate)
    changed["same_population_reference_label"] = "CALLER_SELECTED_PARENT"
    with pytest.raises(MODULE.DivisionOOFError, match="same-population reference label"):
        MODULE.validate_development_gate(changed)
    deployable = deployable_artifact()
    deployable["payload"]["outer_results"][0]["result_sha256"] = "f" * 64
    _refresh_artifact(deployable)
    with pytest.raises(MODULE.DivisionOOFError, match="not bound to its five outer results"):
        MODULE._validate_deployable_development_gate(deployable["payload"])


def held_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    records = []
    inventory_records = []
    overlay_seal_rows = []
    for index in range(36):
        identity = f"train:44b6_held_{index:03d}"
        parent = file_ref(tmp_path / "parent" / f"{index}.csv", f"parent-{index}\n")
        overlay = file_ref(tmp_path / "overlay" / f"{index}.csv", f"overlay-{index}\n")
        records.append(
            {
                "identity": identity,
                "role": MODULE.HELD_ROLE,
                "parent_graph": parent,
            }
        )
        inventory_records.append(
            {"identity": identity, "parent_graph": parent, "overlay_graph": overlay}
        )
        overlay_seal_rows.append(
            {
                "identity": identity,
                "path": f"{index}.csv",
                "sha256": overlay["sha256"],
            }
        )
    plan = {
        "fit105_coverage": {
            "records": records,
            "r1_scientific_configuration": {"name": "R1_SCIENTIFIC"},
        },
        "forbidden_roles": [MODULE.AUDIT_ROLE],
        "deployable_rule": (
            "highest outer-winner count, then lexicographically smallest config name"
        ),
    }
    inventory = {
        "schema": MODULE.HELD_INVENTORY_SCHEMA,
        "status": "SEALED_BEFORE_HELD_FIT_TARGET_ACCESS",
        "pretarget_plan_sha256": "1" * 64,
        "deployable_seal_sha256": "2" * 64,
        "configuration_name": "cfg-a",
        "overlay_prediction_seal": file_ref(
            tmp_path / "overlay" / "SEAL.json",
            json.dumps({"graphs": overlay_seal_rows}, sort_keys=True) + "\n",
        ),
        "records": inventory_records,
    }
    r1 = {
        "schema": MODULE.R1_BINDING_SCHEMA,
        "status": "PASS_VALIDATED_COMPLETE",
        "scientific_configuration_name": "R1_SCIENTIFIC",
        "score": "0.875",
        "aggregate_graph_sha256": "3" * 64,
        "validated_result_report": file_ref(tmp_path / "r1-report.json"),
        "inference_seal": file_ref(tmp_path / "r1-seal.json"),
    }
    return plan, inventory, r1


def amendment() -> dict[str, object]:
    return {
        "held_fit_reference_set": [
            dict(MODULE.APNCC_REFERENCE),
            {
                "id": MODULE.R1_REFERENCE_ID,
                "score": "BIND_FROM_VALIDATED_R1_RESULT_BEFORE_R4_HELD_FIT_ACCESS",
                "aggregate_graph_sha256": "BIND_FROM_R1_INFERENCE_SEAL_BEFORE_R4_HELD_FIT_ACCESS",
            },
        ]
    }


def test_held_marker_binds_exact_held36_graphs_and_closed_reference_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, inventory, r1 = held_fixture(tmp_path)
    monkeypatch.setattr(
        MODULE.decode,
        "verify_sealed_output",
        lambda *_args, **_kwargs: {
            "status": "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING",
            "configuration_name": "cfg-a",
            "video_count": 36,
        },
    )
    marker = MODULE.make_held_marker_artifact(
        plan=plan,
        plan_sha256="1" * 64,
        deployable_artifact=deployable_artifact(),
        deployable_sha256="2" * 64,
        amendment=amendment(),
        inventory=inventory,
        r1_binding=r1,
        base=tmp_path,
    )
    payload = marker["payload"]
    assert payload["video_count"] == 36
    assert payload["closed_reference_ids"] == ["APNCC_44b6_CONTROL", "R1_DIVISIONS_OFF"]
    assert payload["closed_reference_set"][0] == MODULE.APNCC_REFERENCE
    assert payload["closed_reference_set"][1]["score"] == "0.875"
    assert payload["target_opened"] is False
    written = MODULE.write_held_fit_marker(tmp_path / "held-run", marker)
    assert Path(written["path"]).is_file()
    with pytest.raises(MODULE.DivisionOOFError, match="already consumed"):
        MODULE.write_held_fit_marker(tmp_path / "held-run", marker)


def test_held_marker_refuses_reference_drift_and_prior_score_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, inventory, r1 = held_fixture(tmp_path)
    monkeypatch.setattr(
        MODULE.decode,
        "verify_sealed_output",
        lambda *_args, **_kwargs: {
            "status": "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING",
            "configuration_name": "cfg-a",
            "video_count": 36,
        },
    )
    changed = amendment()
    changed["held_fit_reference_set"][0]["score"] = "0.9"
    with pytest.raises(MODULE.DivisionOOFError, match="APNCC"):
        MODULE.make_held_marker_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            deployable_artifact=deployable_artifact(),
            deployable_sha256="2" * 64,
            amendment=changed,
            inventory=inventory,
            r1_binding=r1,
            base=tmp_path,
        )
    run_dir = tmp_path / "tainted"
    run_dir.mkdir()
    (run_dir / "HELD_FIT_SCORE_RECEIPT.json").write_text("{}\n", encoding="ascii")
    marker = MODULE._artifact(MODULE.HELD_MARKER_SCHEMA, "MARKER_WRITTEN_BEFORE_HELD_FIT_TARGET_OPEN", {})
    with pytest.raises(MODULE.DivisionOOFError, match="already consumed"):
        MODULE.write_held_fit_marker(run_dir, marker)


def test_held_marker_refuses_overlay_not_bound_by_decoder_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, inventory, r1 = held_fixture(tmp_path)
    monkeypatch.setattr(
        MODULE.decode,
        "verify_sealed_output",
        lambda *_args, **_kwargs: {
            "status": "SEALED_COMPLETE_PREDICTIONS_BEFORE_SCORING",
            "configuration_name": "cfg-a",
            "video_count": 36,
        },
    )
    inventory["records"][0]["overlay_graph"] = file_ref(
        tmp_path / "different-overlay.csv"
    )
    with pytest.raises(MODULE.DivisionOOFError, match="decoder prediction seal"):
        MODULE.make_held_marker_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            deployable_artifact=deployable_artifact(),
            deployable_sha256="2" * 64,
            amendment=amendment(),
            inventory=inventory,
            r1_binding=r1,
            base=tmp_path,
        )


def test_held_marker_rejects_handcrafted_pass_booleans_before_inventory_access(
    tmp_path: Path,
) -> None:
    plan, inventory, r1 = held_fixture(tmp_path)
    deployable = deployable_artifact()
    gate = deployable["payload"]["development_gate"]
    fold = gate["folds"][0]
    fold["candidate_metrics"]["combined_graph_score"] = (
        fold["parent_metrics"]["combined_graph_score"] - 0.01
    )
    fold["candidate_metrics"]["edge"]["adjusted_jaccard"] = (
        fold["candidate_metrics"]["combined_graph_score"]
        - 0.1 * fold["candidate_metrics"]["division"]["jaccard"]
    )
    fold["deltas"] = MODULE._metric_deltas(
        fold["candidate_metrics"], fold["parent_metrics"]
    )
    # The forged caller leaves every advertised check set to true and re-hashes
    # the wrapper.  Held preparation must recompute, not trust those assertions.
    assert all(gate["checks"].values())
    _refresh_artifact(deployable)
    with pytest.raises(
        MODULE.DivisionOOFError,
        match="all_fold_combined_attribution_nonnegative",
    ):
        MODULE.make_held_marker_artifact(
            plan=plan,
            plan_sha256="1" * 64,
            deployable_artifact=deployable,
            deployable_sha256="2" * 64,
            amendment=amendment(),
            inventory=inventory,
            r1_binding=r1,
            base=tmp_path,
        )


def local_gate(status: str = "PASS_LOCAL_CONFIRMATION_GATE") -> dict[str, object]:
    return {
        "schema": MODULE.LOCAL_GATE_SCHEMA,
        "status": status,
        "pretarget_plan_sha256": "1" * 64,
        "deployable_seal_sha256": "2" * 64,
        "held_fit_access_marker_sha256": "3" * 64,
        "checks": {
            "same_population_score_pass": True,
            "division_jaccard_pass": True,
            "division_tp_positive": True,
            "duplicate_survivors_zero": True,
            "edge_non_regression": True,
            "node_non_regression": True,
            "all_outer_folds_nonnegative_delta": True,
        },
    }


def hidden_inventory(tmp_path: Path, pop: dict[str, object], *, materialize: bool) -> dict[str, object]:
    roles = pop["roles"]
    rows = []
    for index, record in enumerate(roles[MODULE.HIDDEN_ROLE]["videos"]):
        identity = record["identity"]
        bank_path = tmp_path / "hidden-bank" / f"{index}.json"
        graph_path = tmp_path / "hidden-graph" / f"{index}.csv"
        rows.append(
            {
                "identity": identity,
                "association_bank": file_ref(bank_path) if materialize else syntax_ref(bank_path),
                "decoded_graph": file_ref(graph_path) if materialize else syntax_ref(graph_path),
            }
        )
    zip_path = tmp_path / "submission.zip"
    return {
        "schema": MODULE.HIDDEN_INVENTORY_SCHEMA,
        "status": "SEALED_BEFORE_HIDDEN_PROBE",
        "pretarget_plan_sha256": "1" * 64,
        "deployable_seal_sha256": "2" * 64,
        "configuration_name": "cfg-a",
        "submission_zip": file_ref(zip_path) if materialize else syntax_ref(zip_path),
        "records": rows,
    }


def test_hidden4_is_inaccessible_until_every_local_gate_check_passes(tmp_path: Path) -> None:
    pop = population()
    inventory = hidden_inventory(tmp_path, pop, materialize=False)
    failed = local_gate("FAIL_LOCAL_CONFIRMATION_GATE")
    with pytest.raises(MODULE.DivisionOOFError, match="hidden4 remains locked"):
        MODULE.make_hidden_marker_artifact(
            plan={},
            plan_sha256="1" * 64,
            deployable_artifact=deployable_artifact(),
            deployable_sha256="2" * 64,
            held_marker_sha256="3" * 64,
            local_gate=failed,
            local_gate_sha256="4" * 64,
            inventory=inventory,
            population=pop,
            base=tmp_path,
        )
    one_failed = local_gate()
    one_failed["checks"]["division_tp_positive"] = False
    with pytest.raises(MODULE.DivisionOOFError, match="hidden4 remains locked"):
        MODULE.make_hidden_marker_artifact(
            plan={},
            plan_sha256="1" * 64,
            deployable_artifact=deployable_artifact(),
            deployable_sha256="2" * 64,
            held_marker_sha256="3" * 64,
            local_gate=one_failed,
            local_gate_sha256="4" * 64,
            inventory=inventory,
            population=pop,
            base=tmp_path,
        )


def test_hidden4_marker_requires_exact_hidden_population_and_sealed_zip(tmp_path: Path) -> None:
    pop = population()
    inventory = hidden_inventory(tmp_path, pop, materialize=True)
    marker = MODULE.make_hidden_marker_artifact(
        plan={},
        plan_sha256="1" * 64,
        deployable_artifact=deployable_artifact(),
        deployable_sha256="2" * 64,
        held_marker_sha256="3" * 64,
        local_gate=local_gate(),
        local_gate_sha256="4" * 64,
        inventory=inventory,
        population=pop,
        base=tmp_path,
    )
    assert marker["status"] == "MARKER_WRITTEN_AFTER_LOCAL_PASS_BEFORE_SUBMISSION"
    assert marker["payload"]["video_count"] == 4
    assert marker["payload"]["submission_made"] is False
    contaminated = copy.deepcopy(inventory)
    contaminated["records"][0]["identity"] = pop["roles"][MODULE.AUDIT_ROLE]["videos"][0]["identity"]
    with pytest.raises(MODULE.DivisionOOFError, match="audit identity"):
        MODULE.make_hidden_marker_artifact(
            plan={},
            plan_sha256="1" * 64,
            deployable_artifact=deployable_artifact(),
            deployable_sha256="2" * 64,
            held_marker_sha256="3" * 64,
            local_gate=local_gate(),
            local_gate_sha256="4" * 64,
            inventory=contaminated,
            population=pop,
            base=tmp_path,
        )


def test_atomic_create_is_canonical_and_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    sha = MODULE.atomic_create_json(target, {"b": 2, "a": 1})
    assert target.read_bytes() == b'{"a":1,"b":2}\n'
    assert sha == MODULE.decode.sha256_file(target)
    with pytest.raises(MODULE.DivisionOOFError, match="overwrite"):
        MODULE.atomic_create_json(target, {"replacement": True})
    assert json.loads(target.read_text(encoding="ascii")) == {"a": 1, "b": 2}

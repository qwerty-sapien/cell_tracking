from __future__ import annotations

from pathlib import Path

from scripts.g03_division_population_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_population_roles_are_disjoint_and_architecture_aligned() -> None:
    manifest = build_manifest(ROOT)
    roles = manifest["roles"]
    development = {
        row["identity"] for row in roles["development_nested_oof"]["videos"]
    }
    held_fit = {
        row["identity"] for row in roles["held_fit_one_shot_confirmation"]["videos"]
    }
    hidden_test = {row["identity"] for row in roles["sealed_hidden_test_probe"]["videos"]}
    contaminated_audit = {
        row["identity"] for row in roles["excluded_contaminated_audit20"]["videos"]
    }
    assert (len(development), len(held_fit), len(hidden_test), len(contaminated_audit)) == (69, 36, 4, 20)
    assert development.isdisjoint(held_fit)
    assert development.isdisjoint(hidden_test)
    assert held_fit.isdisjoint(hidden_test)
    assert hidden_test.isdisjoint(contaminated_audit)
    assert all(identity.startswith("train:6bba_") for identity in development)
    assert all(identity.startswith("train:44b6_") for identity in held_fit)


def test_five_grouped_folds_cover_development_once() -> None:
    manifest = build_manifest(ROOT)
    development = {
        row["identity"]
        for row in manifest["roles"]["development_nested_oof"]["videos"]
    }
    fold_rows = [row for rows in manifest["outer_folds"].values() for row in rows]
    assert len(fold_rows) == len(development)
    assert {row["identity"] for row in fold_rows} == development
    assert len({row["group_id"] for row in fold_rows}) == len(fold_rows)
    assert sorted(len(rows) for rows in manifest["outer_folds"].values()) == [13, 13, 14, 14, 15]


def test_split_is_target_blind_and_hash_stable() -> None:
    manifest = build_manifest(ROOT)
    assert manifest["constraints"]["division_labels_used_for_split"] is False
    assert manifest["population_sha256"] == (
        "e25416da1d240ceabf2b1457ceb6625b361523e7a537ed2f4ac195280b425954"
    )

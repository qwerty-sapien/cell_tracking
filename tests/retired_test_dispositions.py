"""Reviewed strict-XFAIL registry for retained retired-release documentation.

Every entry is an exact pytest node ID. ``tests/conftest.py`` rejects stale
entries at collection time, and ``strict=True`` turns an unexpected pass into a
failure requiring a fresh lifecycle review.
"""

from __future__ import annotations


RETIRED_XFAILS: dict[str, str] = {
    # R01 — the retired V45 A/B release builder intentionally retains its
    # package contract, but three frozen private source members no longer exist.
    "tests/test_ab45.py::V45ABPackageTests::test_both_manifests_cover_the_rewritten_private_package":
        "XFAIL-WITH-REASON R01: retired V45 A/B package; frozen v45_fast/v45_head/t_adapter source members were removed",
    "tests/test_ab45.py::V45ABPackageTests::test_compile_mode_signs_the_distinct_kernel_and_runtime_contract":
        "XFAIL-WITH-REASON R01: retired V45 A/B package; frozen v45_fast/v45_head/t_adapter source members were removed",
    "tests/test_ab45.py::V45ABPackageTests::test_private_ab_package_is_deterministic_and_under_budget":
        "XFAIL-WITH-REASON R01: retired V45 A/B package; frozen v45_fast/v45_head/t_adapter source members were removed",

    # R02/R03 — historical BH9/V45 release attestations must continue to show
    # why their immutable source pins no longer authenticate current code.
    "tests/test_bh9gb.py::SelectionTests::test_real_archive_selection_is_frozen":
        "XFAIL-WITH-REASON R02: retired BH9 selection artifact pins the pre-repair official_eval.py hash",
    "tests/test_bv45f.py::BuildV45FastTests::test_builds_bounded_package_with_exact_fit_projection":
        "XFAIL-WITH-REASON R03: retired V45-fast package omits the historical temporal_pretrain module",
    "tests/test_bhg.py::LocalV45ArtifactParityTests::test_banked_sparse_ties_match_dense_diagnostic":
        "XFAIL-WITH-REASON R12: retired V45 banked LAPJV parity has filler-dependent equal-cost ties that are outside the active sparse-solver contract",

    # R04 — terminal G03 Wave-R evidence is immutable and is not a current
    # package. Its R4 dependency hash intentionally records the old scorer.
    "tests/test_g03_wave_r.py::test_promoted_wave_r_evidence_validates_after_r1_finalization":
        "XFAIL-WITH-REASON R04: terminal G03 Wave-R evidence pins the pre-repair official_eval.py hash",

    # R05 — this exact signed RunPod package belongs to the PARK-COST crop-
    # wrfeat route. Re-signing it would be an unauthorized fourth configuration.
    "tests/test_g04_runpod_gate.py::test_package_is_exact_and_unsigned_template_fails_closed":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_owner_approval_emits_benchmarks_only_for_every_authorized_class":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_measured_a40_parity_projection_passes_and_unlock_is_local_only":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_every_authorized_class_must_have_a_terminal_benchmark":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_failed_authorized_class_is_terminal_but_not_selectable":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_changed_label_signature_fails_closed":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_stale_price_receipt_fails_closed":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_no_speedup_a100_fails_preregistered_cost_cap":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_frame_hard_bound_cannot_fit_one_frame":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",
    "tests/test_g04_runpod_gate.py::test_benchmark_spend_and_unavailable_class_fail_closed":
        "XFAIL-WITH-REASON R05: PARK-COST crop-wrfeat RunPod package predates the banked geometry repair",

    # R06/R07 — H1/BH9 were terminally closed. Their release-integrity tests are
    # retained to document the exact stale pins, not to authenticate new code.
    "tests/test_h1_feature_cache_prep.py::H1FeatureCachePrepTests::test_hash_pins_match_actual_files_kernel_and_report":
        "XFAIL-WITH-REASON R06: DEAD H1 zero-wrfeat cache package pins the pre-repair bh_graph.py hash",
    "tests/test_pkgbh9.py::Bh9PackageTests::test_archive_freezes_byte_only_selection_and_exact_runtime_sources":
        "XFAIL-WITH-REASON R07: retired BH9 archive no longer contains two historical curated source members",
    "tests/test_pkgbh9.py::Bh9PackageTests::test_archive_has_exact_curated_sources_and_no_secret_like_names":
        "XFAIL-WITH-REASON R07: retired BH9 archive no longer contains two historical curated source members",

    # R08/R09/R10 — the archived released-core temporal-adaptation route used
    # owner-local checkpoint and alpha-cache fixtures that are deliberately not
    # repository dependencies. The unit-only portions of these modules stay live.
    "tests/test_rst.py::RunnerTest::test_06_armed_main_writes_inventory":
        "XFAIL-WITH-REASON R08: retired released-core trainer requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_rst.py::RunnerTest::test_07_atomic_resume_restores_identity":
        "XFAIL-WITH-REASON R08: retired released-core trainer requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_rst.py::RunnerTest::test_08_incompatible_resume_rejects":
        "XFAIL-WITH-REASON R08: retired released-core trainer requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_rst.py::RunnerTest::test_09_dry_run_is_deterministic":
        "XFAIL-WITH-REASON R08: retired released-core trainer requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_rst.py::RunnerTest::test_24_exact_real_fixture_evidence_validates":
        "XFAIL-WITH-REASON R08: retired trainer's exact real smoke fixture was owner-local and is absent",
    "tests/test_ta.py::AdapterTest::test_01_exact_config":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_02_exact_files":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_03_strict_restore_identity":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_14_output_schema_and_specialists":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_15_accounting_and_transformer_exclusion":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_15b_released_detector_stays_unchanged_during_auxiliary_step":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_ta.py::AdapterTest::test_15c_teacher_detector_policy_is_explicit_and_backbone_stays_frozen":
        "XFAIL-WITH-REASON R09: retired released-core adapter requires absent owner-local /private/tmp/ckpt-v1 assets",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_01_only_released_temporal_parameters_are_trainable":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent released checkpoint assets",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_04_real_alpha_bundle_is_sparse_fit_source_and_digest_bound":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent kaggle_kernel/bh9o7 alpha cache",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_05_checkpoint_is_fold_artifact_and_frozen_core_bound":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent released checkpoint assets",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_06_resume_restores_only_the_student_and_exact_identity":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent released checkpoint assets",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_09_outputs_are_resumable_local_sidecars":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent released checkpoint assets",
    "tests/test_temporal_pretrain.py::TemporalPretrainTest::test_10_released_backbone_executes_sparse_temporal_backward":
        "XFAIL-WITH-REASON R10: retired temporal-pretrain integration requires absent released checkpoint assets",

    # R11 — U0-T4CANON is a frozen dossier whose external base report was later
    # amended. The test remains a precise record of the original authentication.
    "tests/test_u0_t4canon.py::U0T4CanonTests::test_positive_36_69_authenticated_coverage":
        "XFAIL-WITH-REASON R11: frozen U0-T4CANON dossier pins a superseded owner-local base-report hash",
}

# Test-suite triage and reviewed dispositions

Date: 2026-08-03  
Scope: canonical active pytest collection under `pyproject.toml`

## Result

The earlier audit's “25 failures” was the result of a `--maxfail=25` cap, not
the complete failure population. An uncapped reproduction found **43 failures**:
the original 25 below plus 18 additional failures that had never been reached.

Final disposition totals:

- `DELETE`: 0 — no failing test lacked documentary or active-contract value;
- `XFAIL-WITH-REASON`: 39 — exact-node, strict reviewed dispositions for
  terminal/retired packages or unavailable historical owner-local fixtures;
- `FIX`: 4 — active scorer semantics, scorer re-freeze, and sandbox-compatible
  process-pool coverage.

The final canonical run is **1,198 passed, 39 xfailed, 267 subtests passed,
zero failed**. XFAILs are `strict=True`: an unexpected pass makes the suite red
and forces a lifecycle review.

## Original capped 25

| # | Exact test | Disposition | One-line cause |
|---:|---|---|---|
| 1 | `tests/test_ab45.py::V45ABPackageTests::test_both_manifests_cover_the_rewritten_private_package` | XFAIL-WITH-REASON R01 | Retired V45 A/B packaging still names three historical source members that no longer exist. |
| 2 | `tests/test_ab45.py::V45ABPackageTests::test_compile_mode_signs_the_distinct_kernel_and_runtime_contract` | XFAIL-WITH-REASON R01 | The retired compile-mode package cannot be rebuilt after removal of its frozen private source members. |
| 3 | `tests/test_ab45.py::V45ABPackageTests::test_private_ab_package_is_deterministic_and_under_budget` | XFAIL-WITH-REASON R01 | Deterministic rebuilding is documentary only because the retired package's source set is incomplete. |
| 4 | `tests/test_bh9gb.py::SelectionTests::test_real_archive_selection_is_frozen` | XFAIL-WITH-REASON R02 | Terminal BH9 selection evidence pins the pre-zero-edge-repair `official_eval.py` hash. |
| 5 | `tests/test_bhg.py::LocalV45ArtifactParityTests::test_banked_sparse_ties_match_dense_diagnostic` | XFAIL-WITH-REASON R12 | Retired V45 LAPJV parity depends on filler-sensitive equal-cost ties outside the active sparse-solver contract. |
| 6 | `tests/test_bv45f.py::BuildV45FastTests::test_builds_bounded_package_with_exact_fit_projection` | XFAIL-WITH-REASON R03 | The retired V45-fast archive omits its historical `temporal_pretrain` source member. |
| 7 | `tests/test_evaluate_synth_div_sim2real.py::test_parallel_two_fragment_generation_is_byte_identical_to_serial` | FIX | The managed macOS sandbox denied only the semaphore-limit probe; the test now bypasses that probe while exercising the real process pool. |
| 8 | `tests/test_evaluate_synth_div_sim2real.py::test_parallel_failure_helper_terminates_running_workers` | FIX | The same denied informational sysconf prevented worker-start/termination coverage before any worker was created. |
| 9 | `tests/test_g02_integrity.py::G02IntegrityTests::test_division_connected_component_cases_call_local_official_code` (`extra_fork`) | FIX | The stale expectation treated a shared-child graph as one valid fork; the organizer marks both forks malformed (`0/1/2`). |
| 10 | `tests/test_g03_wave_r.py::test_promoted_wave_r_evidence_validates_after_r1_finalization` | XFAIL-WITH-REASON R04 | Terminal Wave-R R4 evidence immutably pins the pre-repair scorer source. |
| 11 | `tests/test_g04_runpod_gate.py::test_package_is_exact_and_unsigned_template_fails_closed` | XFAIL-WITH-REASON R05 | The signed RunPod package predates the banked crop-geometry repair and its lane is PARK-COST. |
| 12 | `tests/test_g04_runpod_gate.py::test_owner_approval_emits_benchmarks_only_for_every_authorized_class` | XFAIL-WITH-REASON R05 | The PARK-COST package cannot be revalidated or re-signed without a newly authorized architecture. |
| 13 | `tests/test_g04_runpod_gate.py::test_measured_a40_parity_projection_passes_and_unlock_is_local_only` | XFAIL-WITH-REASON R05 | Its A40 projection belongs to the superseded signed package, not the banked repaired geometry. |
| 14 | `tests/test_g04_runpod_gate.py::test_every_authorized_class_must_have_a_terminal_benchmark` | XFAIL-WITH-REASON R05 | The terminal benchmark contract is retained only as documentation for the PARK-COST package. |
| 15 | `tests/test_g04_runpod_gate.py::test_failed_authorized_class_is_terminal_but_not_selectable` | XFAIL-WITH-REASON R05 | The selection rule is documentary because the exact signed package was retired on cost. |
| 16 | `tests/test_g04_runpod_gate.py::test_changed_label_signature_fails_closed` | XFAIL-WITH-REASON R05 | The label-signature mutation test is bound to the superseded package hash. |
| 17 | `tests/test_g04_runpod_gate.py::test_stale_price_receipt_fails_closed` | XFAIL-WITH-REASON R05 | The price-receipt gate is bound to a RunPod package that cannot be launched under PARK-COST. |
| 18 | `tests/test_g04_runpod_gate.py::test_no_speedup_a100_fails_preregistered_cost_cap` | XFAIL-WITH-REASON R05 | The cost-cap scenario documents the retired multi-orientation package. |
| 19 | `tests/test_g04_runpod_gate.py::test_frame_hard_bound_cannot_fit_one_frame` | XFAIL-WITH-REASON R05 | The hard-bound assertion is package-specific and the package source hash is intentionally stale. |
| 20 | `tests/test_g04_runpod_gate.py::test_benchmark_spend_and_unavailable_class_fail_closed` | XFAIL-WITH-REASON R05 | Spend/class gating remains documentary for the PARK-COST signed package. |
| 21 | `tests/test_h1_feature_cache_prep.py::H1FeatureCachePrepTests::test_hash_pins_match_actual_files_kernel_and_report` | XFAIL-WITH-REASON R06 | DEAD H1 zero-wrfeat cache evidence pins the pre-repair `bh_graph.py` hash. |
| 22 | `tests/test_pkgbh9.py::Bh9PackageTests::test_archive_freezes_byte_only_selection_and_exact_runtime_sources` | XFAIL-WITH-REASON R07 | The retired BH9 archive no longer includes two historical curated sources. |
| 23 | `tests/test_pkgbh9.py::Bh9PackageTests::test_archive_has_exact_curated_sources_and_no_secret_like_names` | XFAIL-WITH-REASON R07 | Its exact-member assertion documents the superseded archive inventory. |
| 24 | `tests/test_rst.py::RunnerTest::test_06_armed_main_writes_inventory` | XFAIL-WITH-REASON R08 | The retired released-core trainer requires absent owner-local `/private/tmp/ckpt-v1` assets. |
| 25 | `tests/test_rst.py::RunnerTest::test_07_atomic_resume_restores_identity` | XFAIL-WITH-REASON R08 | Resume integration requires the same unavailable released checkpoint fixture. |

## Additional 18 exposed by the uncapped run

| # | Exact test | Disposition | One-line cause |
|---:|---|---|---|
| 26 | `tests/test_rst.py::RunnerTest::test_08_incompatible_resume_rejects` | XFAIL-WITH-REASON R08 | The retired resume rejection path cannot reach its assertion without the owner-local checkpoint. |
| 27 | `tests/test_rst.py::RunnerTest::test_09_dry_run_is_deterministic` | XFAIL-WITH-REASON R08 | Historical dry-run setup verifies the absent released files before producing its manifest. |
| 28 | `tests/test_rst.py::RunnerTest::test_24_exact_real_fixture_evidence_validates` | XFAIL-WITH-REASON R08 | The exact smoke/xtract/fit fixtures were owner-local and are no longer present. |
| 29 | `tests/test_score_bh_cv57.py::CV57ScoreTests::test_postpatch_scorer_is_attested_by_green_parity_ledger` | FIX | Active scoring still pointed at immutable parity-v1; parity-v2 now re-freezes the repaired scorer and active defaults. |
| 30 | `tests/test_ta.py::AdapterTest::test_01_exact_config` | XFAIL-WITH-REASON R09 | Retired released-core adapter config existed only under owner-local `/private/tmp/ckpt-v1`. |
| 31 | `tests/test_ta.py::AdapterTest::test_02_exact_files` | XFAIL-WITH-REASON R09 | Exact-file authentication requires the unavailable released weights and config. |
| 32 | `tests/test_ta.py::AdapterTest::test_03_strict_restore_identity` | XFAIL-WITH-REASON R09 | Strict restore integration requires the unavailable released checkpoint. |
| 33 | `tests/test_ta.py::AdapterTest::test_14_output_schema_and_specialists` | XFAIL-WITH-REASON R09 | The model-schema integration fixture cannot load the retired released detector. |
| 34 | `tests/test_ta.py::AdapterTest::test_15_accounting_and_transformer_exclusion` | XFAIL-WITH-REASON R09 | Parameter accounting is checkpoint-specific to the retired released detector. |
| 35 | `tests/test_ta.py::AdapterTest::test_15b_released_detector_stays_unchanged_during_auxiliary_step` | XFAIL-WITH-REASON R09 | Frozen-detector integration requires unavailable released weights. |
| 36 | `tests/test_ta.py::AdapterTest::test_15c_teacher_detector_policy_is_explicit_and_backbone_stays_frozen` | XFAIL-WITH-REASON R09 | Teacher-policy integration requires the same retired checkpoint fixture. |
| 37 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_01_only_released_temporal_parameters_are_trainable` | XFAIL-WITH-REASON R10 | Retired temporal-pretrain integration requires absent released checkpoint assets. |
| 38 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_04_real_alpha_bundle_is_sparse_fit_source_and_digest_bound` | XFAIL-WITH-REASON R10 | The historical `kaggle_kernel/bh9o7` alpha cache is absent. |
| 39 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_05_checkpoint_is_fold_artifact_and_frozen_core_bound` | XFAIL-WITH-REASON R10 | Fold-artifact integration depends on the unavailable released core. |
| 40 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_06_resume_restores_only_the_student_and_exact_identity` | XFAIL-WITH-REASON R10 | Student-resume integration cannot instantiate the absent released backbone. |
| 41 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_09_outputs_are_resumable_local_sidecars` | XFAIL-WITH-REASON R10 | Sidecar integration requires the retired released-core fixture. |
| 42 | `tests/test_temporal_pretrain.py::TemporalPretrainTest::test_10_released_backbone_executes_sparse_temporal_backward` | XFAIL-WITH-REASON R10 | Sparse backward integration requires the unavailable released backbone. |
| 43 | `tests/test_u0_t4canon.py::U0T4CanonTests::test_positive_36_69_authenticated_coverage` | XFAIL-WITH-REASON R11 | Frozen U0-T4CANON pins a superseded owner-local base-report hash. |

## Implementation

- `tests/retired_test_dispositions.py` is the reviewed exact-node registry.
- `tests/conftest.py` applies `strict=True` XFAIL markers and rejects stale
  registry entries during the canonical full collection.
- `reports/g02/scorer_parity_v2.json` preserves parity-v1 and re-freezes the
  corrected active scorer; active scorer defaults now point to v2.
- No historical receipt or frozen parity-v1 artifact was rewritten.

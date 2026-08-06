# Cruise workspace changelog

> User-only document. Cruise updates this script-only summary through
> deterministic commit-triggered code. Agents must never read it or use its
> contents as context.

<!-- CRUISE:SINCE-LAST-PUSH:START -->
Generated: 2026-08-01T07:09:45.962Z

## Since last push

- .agents/skills/bioservices/scripts/batch_id_converter.py: delete - Removes the script from the workspace.
- .agents/skills/bioservices/scripts/compound_cross_reference.py: delete - Removes the script from the workspace.
- .agents/skills/bioservices/scripts/pathway_analysis.py: delete - Removes the script and its distributions, distribution behavior.
- .agents/skills/bioservices/scripts/protein_analysis_workflow.py: delete - Removes the script from the workspace.
- .agents/skills/bulk-rnaseq/scripts/build_counts_matrix.py: delete - Removes the script from the workspace.
- .agents/skills/bulk-rnaseq/scripts/validate_samplesheet.py: delete - Removes the script and its Report behavior.
- bhv45-submit.py: new - Adds a new executable or source script.
- dag_prompts/2026-07-30-continuation/dinov2/runs/N2_dprc_fixture_test.py: new - Adds a new script implementing DPRCFixtureTest.
- dag_prompts/2026-08-01-global-reassign-oracle/global_reassign_oracle.py: new - Adds a new script implementing OracleError, Chain.
- dag_prompts/2026-08-01-global-reassign-oracle/run_guarded.py: new - Adds a new executable or source script.
- dag_prompts/2026-08-01-global-reassign-oracle/test_global_reassign_oracle.py: new - Adds a new script implementing ChainTests.
- kaggle_kernel/celltrack_baseline/__init__.py: delete - Removes the script from the workspace.
- kaggle_kernel/celltrack_baseline/rule_based.py: delete - Removes the script and its BaselineConfig behavior.
- kaggle_kernel/rule_based_submission.py: delete - Removes the script and its BaselineConfig behavior.
- kfast/run.py: new - Adds a new executable or source script.
- reports/g02/screening/tiered_surface/branches/sa_assignment_separation_v1/generate_assignment_features.py: new - Adds a new script implementing FeatureError.
- reports/g02/screening/tiered_surface/branches/sa_assignment_separation_v1/validate_assignment_screen.py: new - Adds a new executable or source script.
- reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py: new - Adds a new executable or source script.
- reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py: new - Adds a new script implementing GuardError.
- reports/g02/screening/tiered_surface/kaggle_prelaunch/test_kaggle_prelaunch_guard.py: new - Adds a new script implementing PrelaunchGuardTest.
- scripts/ablate.py: new - Adds a new script implementing AblateError, Cand.
- scripts/analyze_achievable_headroom.py: new - Adds a new script implementing AchievableHeadroomError.
- scripts/analyze_action_class_oracle.py: new - Adds a new script implementing oracle, ActionOracleError.
- scripts/analyze_appearance_residual.py: new - Adds a new script implementing ResidualAuditError.
- scripts/analyze_candgen_ceiling.py: new - Adds a new script implementing CandgenCeilingError.
- scripts/analyze_division_sandbox_holdout.py: new - Adds a new executable or source script.
- scripts/analyze_fit_division_geometry.py: new - Adds a new script implementing DivisionGeometryError.
- scripts/analyze_fit_node_threshold.py: new - Adds a new script implementing NodeThresholdError.
- scripts/analyze_h0_proxy_validity.py: new - Adds a new executable or source script.
- scripts/analyze_holdout_fragmentation.py: new - Adds a new script implementing FragmentationAuditError, DisjointSet.
- scripts/analyze_public_calibration_intervals.py: new - Adds a new script implementing CalibrationIntervalError.
- scripts/analyze_source_softmax_node_threshold.py: new - Adds a new script implementing ThresholdAnalysisError.
- scripts/attest_h1_cache.py: new - Adds a new script implementing H1CacheAttestationError.
- scripts/audit_c1_hoct_lite.py: new - Adds a new script implementing C1AuditError.
- scripts/audit_compact_point_context_data.py: new - Adds a new script implementing PointContextAuditError.
- scripts/audit_crop_linker_data.py: new - Adds a new script implementing CropLinkerAuditError.
- scripts/audit_crop_linker_family_thresholds.py: new - Adds a new script implementing FamilyThresholdAuditError.
- scripts/audit_extended_radius_reachability.py: new - Adds a new script implementing ReachabilityAuditError.
- scripts/audit_extended_radius_shell_appearance_support.py: new - Adds a new script implementing ShellSupportError.
- scripts/audit_extended_radius_shell_motion_gate.py: new - Adds a new script implementing MotionShellError.
- scripts/audit_general_route_cv.py: new - Adds a new executable or source script.
- scripts/audit_point_context_oof_threshold_headroom.py: new - Adds a new script implementing ThresholdHeadroomError.
- scripts/audit_selected_graph_appearance_combined.py: new - Adds a new script implementing CombinedAuditError.
- scripts/audit_selected_graph_appearance_support.py: new - Adds a new script implementing AppearanceSupportError.
- scripts/audit_selected_graph_appearance.py: new - Adds a new script implementing AppearanceAuditError.
- scripts/audit_selected_graph_gate_support.py: new - Adds a new script implementing SupportAuditError.
- scripts/audit_selected_graph_tracklet_context.py: new - Adds a new script implementing TrackletContextError.
- scripts/audit_trackastra_full_pool_feasibility.py: new - Adds a new script implementing TrackastraFullPoolError.
- scripts/audit_trackastra_pretrained_assets.py: new - Adds a new script implementing TrackastraAssetAuditError, report.
- scripts/augment_extended_radius_shell_appearance.py: new - Adds a new script implementing ShellAppearanceError.
- scripts/bh9gb.py: new - Adds a new script implementing Bh9Error, Bh9Deadline, Bh9Rejected, HardLimit.
- scripts/build_candidate_graph_training_table.py: new - Adds a new executable or source script.
- scripts/build_dispersion_ladder_observation_matrix.py: new - Adds a new script implementing DispersionMatrixError, Rung.
- scripts/build_duplicate_geff_submission.py: delete - Removes the script from the workspace.
- scripts/build_e4.py: new - Adds a new script implementing PackageError, E4LaunchError.
- scripts/build_fitcache_kaggle.py: new - Adds a new script implementing FitCacheBuildError.
- scripts/build_fixed_node_association_oracle.py: new - Adds a new script implementing AssociationOracleError.
- scripts/build_general_model_manifest.py: new - Adds a new executable or source script.
- scripts/build_kaggle_calibration_probes.py: new - Adds a new executable or source script.
- scripts/build_kaggle_kernel.py: patch - Updates _FallbackArray, _FallbackGroup.
- scripts/build_obs_diff_baseline.py: new - Adds a new executable or source script.
- scripts/build_tier_census.py: new - Adds a new script implementing to, oracle.
- scripts/build_trackastra_6bba_kaggle_manifest.py: new - Adds a new script implementing ManifestError.
- scripts/build_trackastra_apncc_base_cache.py: new - Adds a new executable or source script.
- scripts/build_trackastra_full_pool_feature_cache.py: new - Adds a new script implementing TrackastraFeatureCacheError.
- scripts/build_v45f.py: new - Adds a new script implementing PackageError, FastLaunchError, WrapperError.
- scripts/build_v45z.py: new - Adds a new script implementing V45ZeroControlError.
- scripts/calibrate_alpha_offsets.py: patch - Updates the script behavior or integration.
- scripts/check_scorer_parity.py: new - Adds a new executable or source script.
- scripts/cmpgraph.py: new - Adds a new script implementing ComparisonError, Node, Graph, SparseStats, _UnionFind.
- scripts/compare_alpha_routes_zip.py: new - Adds a new executable or source script.
- scripts/compare_submissions.py: new - Adds a new script implementing PredictionArtifact.
- scripts/diagnose_alpha_detection.py: patch - Updates the script behavior or integration.
- scripts/diagnose_candidate_edge_oracle.py: new - Adds a new executable or source script.
- scripts/diagnose_division_failure.py: new - Adds a new script implementing DivisionDiagnosticError.
- scripts/diagnose_h1_zero_wrfeat_label_join.py: new - Adds a new executable or source script.
- scripts/diagnose_localization_failures.py: new - Adds a new executable or source script.
- scripts/diagnose_phase1_distance_motion_gate.py: new - Adds a new script implementing Phase1DiagnosticError.
- scripts/diagnose_trackastra_stack_restore.py: new - Adds a new script implementing StackRestoreDiagnosticError.
- scripts/edgefit.py: new - Adds a new script implementing EdgeFitError, FitScope.
- scripts/enrich_trackastra_apncc_cache.py: new - Adds a new executable or source script.
- scripts/evaluate_appearance_direct_policy_fit.py: new - Adds a new script implementing PolicyFitError.
- scripts/evaluate_c1_hoct_lite_graph_gate.py: new - Adds a new script implementing C1GraphGateError.
- scripts/evaluate_erlang_division_prior_cv57.py: new - Adds a new script implementing ErlangExperimentError.
- scripts/evaluate_h1_zero_wrfeat_graph_gate.py: new - Adds a new script implementing H1ZeroWrfeatGateError.
- scripts/evaluate_point_context_oof_graph_gate.py: new - Adds a new script implementing PointContextOOFGraphGateError.
- scripts/evaluate_relative_geometry_swap_gate.py: new - Adds a new script implementing SwapGateEvalError.
- scripts/evaluate_selected_graph_conflict_gate.py: new - Adds a new script implementing SelectedGraphEvalError.
- scripts/evaluate_selected_graph_direct_swap.py: new - Adds a new script implementing DirectSwapError.
- scripts/evaluate_temporal_rank_oof_graph_gate.py: new - Adds a new script implementing TemporalOOFGraphGateError.
- scripts/evaluate_trackastra_linear_oof_graph_gate.py: new - Adds a new script implementing TrackastraOOFGraphGateError.
- scripts/evaluate_trackastra_zero_wrfeat_signal.py: new - Adds a new script implementing TrackastraSignalError.
- scripts/falsify_e4_e5.py: new - Adds a new script implementing FalsificationError, MatchedVideo, AppearanceEvent, MorphologyRecord.
- scripts/fit_compact_point_context_pilot.py: new - Adds a new script implementing PointContextPilotError.
- scripts/fit_crop_linker_6bba_oof.py: new - Adds a new script implementing CropOofError, FoldRoles.
- scripts/fit_crop_linker_pilot.py: new - Adds a new script implementing CropPilotError, Examples.
- scripts/fit_motion_gated_shell_recall.py: new - Adds a new script implementing MotionGatedShellError.
- scripts/fit_relative_geometry_swap_gate.py: new - Adds a new script implementing SwapGateFitError.
- scripts/fit_selected_graph_background_gate.py: new - Adds a new script implementing BackgroundFitError.
- scripts/fit_selected_graph_conflict_gate.py: new - Adds a new script implementing SelectedGraphFitError.
- scripts/fit_selected_graph_scenario_calibration_gate.py: new - Adds a new script implementing ScenarioCalibrationError.
- scripts/fit_selected_graph_temporal_background_gate.py: new - Adds a new script implementing TemporalBackgroundFitError, TemporalGate.
- scripts/fit_selected_graph_temporal_family_rank_gate.py: new - Adds a new script implementing TemporalFamilyRankError.
- scripts/fit_selected_graph_temporal_rank_gate.py: new - Adds a new script implementing TemporalRankError.
- scripts/freeze_g02_integrity.py: new - Adds a new executable or source script.
- scripts/freeze_n1_feasibility_compiler.py: new - Adds a new executable or source script.
- scripts/git_push_chunked.sh: new - Adds a new script implementing usage, cleanup, inventory_objects, push_seed_batch.
- scripts/git-safe-add.sh: new - Adds a new executable or source script.
- scripts/guard_large_files.sh: new - Adds a new script implementing usage, ignore_pattern_for_path.
- scripts/install_git_hooks.sh: new - Adds a new executable or source script.
- scripts/label_final_edges.py: new - Adds a new executable or source script.
- scripts/merge_fitcache_corpus.py: new - Adds a new script implementing FitCacheMergeError.
- scripts/mkab45.py: new - Adds a new script implementing PackageError, ABError, Encoder, Proxy.
- scripts/mkfastk.py: new - Adds a new executable or source script.
- scripts/mknb45.py: new - Adds a new script implementing NotebookPackageError.
- scripts/mknb9.py: new - Adds a new script implementing NotebookError.
- scripts/mknbab.py: new - Adds a new script implementing NotebookError.
- scripts/mkt05.py: new - Adds a new executable or source script.
- scripts/mkt05art.py: new - Adds a new executable or source script.
- scripts/mkvs45.py: new - Adds a new script implementing PackageError, ViewError.
- scripts/p4_edge_attention_cv57.py: new - Adds a new script implementing P4CVError.
- scripts/p4_edge_attention_gpu_benchmark.py: new - Adds a new script implementing BenchmarkError.
- scripts/p4_edge_attention.py: new - Adds a new script implementing P4Error.
- scripts/pack_t04.py: new - Adds a new executable or source script.
- scripts/pack_t05.py: new - Adds a new script implementing TemporalAdapterModel.
- scripts/pack_tnb.py: new - Adds a new script implementing Tiny, _Module, _Cuda.
- scripts/packfast.py: new - Adds a new executable or source script.
- scripts/phase1_e3_motion_coefficient_cv57.py: new - Adds a new script implementing E3Error.
- scripts/phase1_e6_affine_motion_residual_cv57.py: new - Adds a new script implementing E6Error.
- scripts/phase1_e6_affine_tvl1_prefix52.py: new - Adds a new script implementing E6Error.
- scripts/pkgbh.py: new - Adds a new script implementing PackageError, LaunchError.
- scripts/pkgbh9.py: new - Adds a new script implementing PackageError, LaunchError.
- scripts/prepare_crop_linker_dataset.py: new - Adds a new script implementing CropDatasetError.
- scripts/prepare_extended_radius_shell.py: new - Adds a new script implementing ShellPrepareError, and.
- scripts/prepare_h1_feature_cache.py: new - Adds a new script implementing H1PrepError.
- scripts/prepare_relative_geometry_swap_gate.py: new - Adds a new script implementing SwapGatePrepareError.
- scripts/prepare_selected_graph_background.py: new - Adds a new script implementing BackgroundPrepareError.
- scripts/prepare_selected_graph_conflict_gate.py: new - Adds a new script implementing SelectedGraphPrepareError.
- scripts/prepare_selected_graph_temporal_background_gate.py: new - Adds a new script implementing TemporalBackgroundPrepareError.
- scripts/pretrain_temporal.py: new - Adds a new script implementing ImageWindowSource.
- scripts/probe_endpoint_backward_variants.py: new - Adds a new executable or source script.
- scripts/probe_endpoint_forward_variants.py: new - Adds a new executable or source script.
- scripts/probe_offset_clone_filter_variants.py: new - Adds a new executable or source script.
- scripts/probe_offset_clone_variants.py: new - Adds a new executable or source script.
- scripts/probe_postclone_filter_variants.py: new - Adds a new executable or source script.
- scripts/probe_route_link_params.py: new - Adds a new executable or source script.
- scripts/prune_fragment_components.py: new - Adds a new script implementing FragmentPruneError.
- scripts/push_kaggle_kernel.sh: patch - Updates the script behavior or integration.
- scripts/recover_v45feval.py: new - Adds a new script implementing RecoveryError.
- scripts/replay_u0_t4canon.py: new - Adds a new executable or source script.
- scripts/rescue_fragment_divisions.py: new - Adds a new script implementing DivisionRescueError.
- scripts/run_alpha_routed_zip.py: patch - Updates the script behavior or integration.
- scripts/run_bh.py: new - Adds a new script implementing RunnerError, PreflightError, DiscoveryError, DeadlineError, ResumeHashError, OutputValidationError, RegistryRecord, VideoJob.
- scripts/run_c1_hoct_lite_dry.py: new - Adds a new script implementing C1DryError.
- scripts/run_cgl_d14_fastfail.py: new - Adds a new script implementing CGLFastFailError, VideoRows, TailRankHead.
- scripts/run_local_training_guard.py: new - Adds a new script implementing LocalTrainingGuardError.
- scripts/run_u0_t4canon_guard.py: new - Adds a new executable or source script.
- scripts/run_v45feval_isolated.py: new - Adds a new script implementing IsolatedEvalError.
- scripts/score_alpha_motion_hybrid.py: new - Adds a new executable or source script.
- scripts/score_bh_cv57.py: new - Adds a new script implementing CV57ScoreError.
- scripts/score_cv_submission.py: new - Adds a new executable or source script.
- scripts/score_phase1_topk_gate_cv57.py: new - Adds a new script implementing TopKGateError.
- scripts/score_ta_gate.py: new - Adds a new executable or source script.
- scripts/score_v6.py: new - Adds a new script implementing V6Error.
- scripts/scoreall.py: new - Adds a new script implementing ScoreAllError, FragmentSetError, Selection, FragmentManifest.
- scripts/screen_cg1_abstention.py: new - Adds a new script implementing CG1ScreenError.
- scripts/screen_cg1b_representation.py: new - Adds a new script implementing CG1bError.
- scripts/submit_kaggle_alpha.py: new - Adds a new executable or source script.
- scripts/sweep_bh_graph_weights.py: new - Adds a new script implementing SweepError, SweepGraphConfig, MemoizedImageRefiner.
- scripts/sweep_metric_asymmetry_cv57.py: new - Adds a new script implementing AsymmetrySweepError, MatchState.
- scripts/trackastra_checkpoint_loader.py: new - Adds a new script implementing TrackastraCheckpointLoaderError.
- scripts/trackastra_parity_diagnostic.py: new - Adds a new script implementing DiagnosticError.
- scripts/train_candidate_pruner.py: new - Adds a new executable or source script.
- scripts/train_coordinate_corrector.py: new - Adds a new executable or source script.
- scripts/train_edge_scorer.py: new - Adds a new executable or source script.
- scripts/train_ta.py: new - Adds a new script implementing LaunchError, SourceRecord, SparseWindowSource.
- scripts/train_template_proposal.py: new - Adds a new executable or source script.
- scripts/train_v45f.py: new - Adds a new script implementing V45FastTrain.
- scripts/train_v45h.py: new - Adds a new script implementing V45HeadTrainError.
- scripts/u0_t4canon_preflight.py: new - Adds a new executable or source script.
- scripts/update_kaggle_calibration_ledger.py: new - Adds a new executable or source script.
- scripts/v45feval.py: new - Adds a new script implementing V45FastEval, EvalScope.
- scripts/v45gdiag.py: new - Adds a new script implementing V45GraphDiagnosticError.
- scripts/v45par.py: new - Adds a new script implementing V45ParityError, RawGraph, ExpectedGraph, LazyFrameCache.
- scripts/validate_model_candidate_gate.py: new - Adds a new executable or source script.
- scripts/validate_remote_submission_csv.py: new - Adds a new executable or source script.
- scripts/verify_h1_kaggle_registration.py: new - Adds a new script implementing RegistrationError.
- scripts/verify_n1_phase0_freeze.py: new - Adds a new executable or source script.
- src/celltrack_alpha/action_compiler.py: new - Adds a new script implementing import, ActionCompileError, ActionKind, GraphAction, CompiledAction, CompiledBatch, contract.
- src/celltrack_alpha/appearance_swap.py: new - Adds a new script implementing AppearanceSwapError.
- src/celltrack_alpha/bh_edge.py: new - Adds a new script implementing SparseEdgeError, EdgeArtifactError, SparseEdgeConfig, CandidateStats, SparseCandidateCOO, ProbabilityStats, SparsePairProbabilities, A0ScorerConfig.
- src/celltrack_alpha/bh_graph.py: new - Adds a new script implementing GraphCalibrationError, SparseAssignmentError, GraphInvariantError, V45GraphConfig, GraphNode, GraphEdge, ErlangDivisionPrior, LearnedEdgeProbability.
- src/celltrack_alpha/bh_model.py: new - Adds a new script implementing BHModelError, ArtifactValidationError, DependencyUnavailableError, ModelValidationError, CacheValidationError, V45Config, ArtifactHashes, ArtifactBundle.
- src/celltrack_alpha/bh_route.py: new - Adds a new script implementing RouteModelError, FeatureValidationError, ProvenanceError, ModelSerializationError, RoutePrediction, OOFFitResult, _TreeNode, ScenarioRouter.
- src/celltrack_alpha/c1_conflict_set_attention.py: new - Adds a new script implementing C1ConflictSetError, ComponentExample, ConflictSetAttention, FittedConflictHead.
- src/celltrack_alpha/crop_linker.py: new - Adds a new script implementing CropLinkerError, CropEncoder, SiameseCropLinker, TemperatureCalibration.
- src/celltrack_alpha/duplicate.py: delete - Removes the script from the workspace.
- src/celltrack_alpha/edge_attention.py: new - Adds a new script implementing EdgeAttentionError, EdgeFrame, EdgeResidualModel.
- src/celltrack_alpha/edge_data.py: new - Adds a new script implementing EdgeDataError, EdgeExamples, ConflictPairExamples, SelectedGraphConflictExamples.
- src/celltrack_alpha/fast_data.py: new - Adds a new script implementing FastDataError, FastSource.
- src/celltrack_alpha/g02_integrity.py: new - Adds a new script implementing IntegrityError, SubmissionIntegrityError, ValidatedSubmission, _UnionFind.
- src/celltrack_alpha/kaggle_calibration.py: new - Adds a new script implementing SubmissionNode, SubmissionEdge, SubmissionGraph.
- src/celltrack_alpha/localization.py: new - Adds a new script implementing UnsafeCalibrationError, OffsetCalibration.
- src/celltrack_alpha/model_candidate_gate.py: new - Adds a new script implementing CandidateGateError.
- src/celltrack_alpha/official_eval.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/pipeline.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/point_context_swap.py: new - Adds a new script implementing PointContextSwapError, RuntimeProposalWindow.
- src/celltrack_alpha/point_context.py: new - Adds a new script implementing PointContextError, ProposalWindow, PointContextTransformer.
- src/celltrack_alpha/proposals.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/relative_geometry_gate.py: new - Adds a new script implementing is, RelativeGeometryGateError, SwapDecision, RelativeGeometryGate.
- src/celltrack_alpha/routing.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/scorer_parity.py: new - Adds a new executable or source script.
- src/celltrack_alpha/scoring.py: new - Adds a new script implementing UnsafeScoringCalibrationError, ScoringCalibration, NodePruningStats, EdgeScoringStats, CoordinateCorrectionStats.
- src/celltrack_alpha/site_calibration.py: new - Adds a new script implementing CalibrationModel, BinaryProbabilityCalibration.
- src/celltrack_alpha/solver.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/synthetic.py: patch - Updates the script behavior or integration.
- src/celltrack_alpha/t_adapter.py: new - Adds a new script implementing CheckpointError, TargetError, CoreLoad, TrainedLoad, SparseTargets, _Specialist, TemporalAdapterModel.
- src/celltrack_alpha/ta_hybrid.py: new - Adds a new script implementing HybridRerankError, ChildMotion, HybridRerankResult.
- src/celltrack_alpha/ta_inference.py: new - Adds a new script implementing GateError, TrainOutputAttestation, PretrainAttestation, GateAttestation, HybridAttestation, FrameOutput.
- src/celltrack_alpha/temporal_pretrain.py: new - Adds a new script implementing TemporalPretrainError, TemporalPretrainLoad, PseudoLink, DivisionMotif, PseudoLinkWindow, AlphaPseudoLinkArtifact, ReleasedTemporalStudent, is.
- src/celltrack_alpha/temporal_rank_swap.py: new - Adds a new script implementing TemporalRankSwapError, TemporalRankGate.
- src/celltrack_alpha/trackastra_linear.py: new - Adds a new script implementing TrackastraLinearError, LinearHead.
- src/celltrack_alpha/types.py: patch - Updates RouteConfig.
- src/celltrack_alpha/u0_t4canon.py: new - Adds a new script implementing U0T4CanonError.
- src/celltrack_alpha/v45_fast.py: new - Adds a new script implementing V45FastError, FastIdent, FastVideo, FrozenV45Fast.
- src/celltrack_alpha/v45_head.py: new - Adds a new script implementing rather, V45HeadError, V45AuxiliaryCheckpointError, V45AuxCheckpointIdentity, V45AuxReadout, FrozenV45Head, explicit.
- src/celltrack_ultrack/__init__.py: delete - Removes the script from the workspace.
- src/celltrack_ultrack/core.py: delete - Removes the script and its UltrackConfig behavior.
- tests/test_ab45.py: new - Adds a new script implementing V45ABPackageTests, TemporalUNet3D, SimpleNodeTransformer.
- tests/test_ablate.py: new - Adds a new script implementing CsvTests, PartTests, TransformTests, ScoreTests.
- tests/test_action_class_oracle.py: new - Adds a new script implementing ActionClassOracleTest.
- tests/test_action_compiler.py: new - Adds a new script implementing ActionCompilerTest.
- tests/test_alpha_solver.py: patch - Updates AlphaSolverTests.
- tests/test_analyze_achievable_headroom.py: new - Adds a new script implementing AchievableHeadroomTests.
- tests/test_analyze_appearance_residual.py: new - Adds a new executable or source script.
- tests/test_analyze_candgen_ceiling.py: new - Adds a new script implementing CandgenCeilingTests.
- tests/test_audit_extended_radius_shell_appearance_support.py: new - Adds a new script implementing ThresholdSelectionTests.
- tests/test_audit_point_context_oof_threshold_headroom.py: new - Adds a new executable or source script.
- tests/test_audit_trackastra_full_pool_feasibility.py: new - Adds a new executable or source script.
- tests/test_audit_trackastra_pretrained_assets.py: new - Adds a new executable or source script.
- tests/test_augment_extended_radius_shell_appearance.py: new - Adds a new script implementing AppearanceRowFeatureTests.
- tests/test_bh_division_holdout.py: new - Adds a new script implementing FrozenHoldoutDivisionTests.
- tests/test_bh.py: new - Adds a new script implementing SimulatedKernelCrash, DiscoveryTests, ModelGraphRouteTests, FakeModel, ImageSupportProjectionTests, WorkerFailurePropagationTests, FakeProcess, ResumeDeadlineMergeTests.
- tests/test_bh9gb.py: new - Adds a new script implementing SelectionTests, RunnerContractTests, EvidenceTests, LocalStore, ZipStore.
- tests/test_bha0.py: new - Adds a new script implementing FakeA0, A0RunnerTests, BadProbability, Factory, CacheIdentity.
- tests/test_bhe_a0.py: new - Adds a new script implementing _RecordingBlock, _RecordingPairMLP, _A0Transformer, A0CheckpointScorerTests.
- tests/test_bhe.py: new - Adds a new script implementing _ForbiddenBlock, _PairMLP, _CheckpointTransformer, CandidateGateTests, SparseScorerTests, SparseLabelTests, EdgeArtifactTests.
- tests/test_bhg.py: new - Adds a new script implementing V45ConfigTests, SweepConfig, SparseAssignmentTests, MotionTests, GapClosingTests, DivisionTests, PostprocessingTests, InvariantAndProvenanceTests.
- tests/test_bhm.py: new - Adds a new script implementing SourceSoftmaxTests, D4AndNormalizationTests, TemporalApiTests, EchoTemporal, FakeTemporal, ArtifactTests, CacheAndGraphValidationTests.
- tests/test_bhr.py: new - Adds a new script implementing FeatureSchemaTests, LabelConstructionTests, ProvenanceTests, RouterTests, SerializationTests.
- tests/test_build_trackastra_full_pool_feature_cache.py: new - Adds a new executable or source script.
- tests/test_bv45f.py: new - Adds a new script implementing BuildV45FastTests.
- tests/test_c1_conflict_set_attention.py: new - Adds a new script implementing ConflictSetAttentionTests.
- tests/test_cgl_d14_fastfail.py: new - Adds a new executable or source script.
- tests/test_cmpgraph.py: new - Adds a new script implementing CompareGraphTests.
- tests/test_compact_point_context_data.py: new - Adds a new executable or source script.
- tests/test_compare_submissions.py: new - Adds a new script implementing CompareSubmissionsTest.
- tests/test_crop_linker_6bba_oof.py: new - Adds a new executable or source script.
- tests/test_crop_linker_data.py: new - Adds a new executable or source script.
- tests/test_crop_linker_family_thresholds.py: new - Adds a new executable or source script.
- tests/test_crop_linker_pilot.py: new - Adds a new executable or source script.
- tests/test_crop_linker.py: new - Adds a new executable or source script.
- tests/test_dispersion_ladder_observation_matrix.py: new - Adds a new executable or source script.
- tests/test_e4.py: new - Adds a new script implementing E4PackageTests.
- tests/test_edge_attention.py: new - Adds a new executable or source script.
- tests/test_edged.py: new - Adds a new script implementing SparseMetricMatchTests.
- tests/test_efit.py: new - Adds a new script implementing EdgeFitScopeTests.
- tests/test_evaluate_c1_hoct_lite_graph_gate.py: new - Adds a new script implementing C1GraphGateTests.
- tests/test_evaluate_h1_zero_wrfeat_graph_gate.py: new - Adds a new script implementing H1ZeroWrfeatGraphGateTests.
- tests/test_evaluate_trackastra_linear_oof_graph_gate.py: new - Adds a new executable or source script.
- tests/test_evaluate_trackastra_zero_wrfeat_signal.py: new - Adds a new executable or source script.
- tests/test_extended_radius_reachability.py: new - Adds a new executable or source script.
- tests/test_extended_radius_shell_motion_gate.py: new - Adds a new executable or source script.
- tests/test_extended_radius_shell.py: new - Adds a new executable or source script.
- tests/test_falsify_e4_e5.py: new - Adds a new script implementing TestFalsifyE4E5.
- tests/test_fdat.py: new - Adds a new script implementing FastDataTests.
- tests/test_fit_motion_gated_shell_recall.py: new - Adds a new script implementing MotionGatedShellRecallTest.
- tests/test_fit_node_threshold.py: new - Adds a new script implementing FitNodeThresholdTests.
- tests/test_fit_selected_graph_temporal_background_gate.py: new - Adds a new script implementing ProposalConfidenceTests.
- tests/test_fit_selected_graph_temporal_family_rank_gate.py: new - Adds a new script implementing FamilyTests.
- tests/test_fit_selected_graph_temporal_rank_gate.py: new - Adds a new script implementing CappedVideoMetricsTests.
- tests/test_fitcache_kaggle.py: new - Adds a new script implementing FitCachePackageTests.
- tests/test_fixed_node_association_oracle.py: new - Adds a new script implementing FixedNodeAssociationOracleTests.
- tests/test_g02_integrity.py: new - Adds a new script implementing _TrackedImageStream, _FixedImageStream, _RecordingZipFile, G02IntegrityTests.
- tests/test_gpuinf.py: new - Adds a new script implementing FastInferenceTests, CountingTree.
- tests/test_h0_proxy_validity.py: new - Adds a new script implementing CorrelationLogicTest, DecisionLogicTest.
- tests/test_h1_feature_cache_prep.py: new - Adds a new script implementing H1FeatureCachePrepTests.
- tests/test_h1_parity_repair.py: new - Adds a new script implementing H1ParityRepairTests.
- tests/test_holdout_fragmentation.py: new - Adds a new script implementing BreakClassificationTests, FrozenArtifactTests.
- tests/test_kaggle_calibration.py: new - Adds a new script implementing KaggleCalibrationTests.
- tests/test_kfast.py: new - Adds a new script implementing KfastTests.
- tests/test_localization_provenance.py: new - Adds a new script implementing LocalizationProvenanceTests.
- tests/test_mknb45.py: new - Adds a new script implementing NotebookPackageTests.
- tests/test_mknb9.py: new - Adds a new script implementing Bh9NotebookTests.
- tests/test_mknbab.py: new - Adds a new script implementing V45ABNotebookTests.
- tests/test_mkt05.py: new - Adds a new script implementing T05UploadTests.
- tests/test_mkt05art.py: new - Adds a new script implementing T05ArtifactTests.
- tests/test_mkvs45.py: new - Adds a new script implementing ViewDiagnosticPackageTests.
- tests/test_model_candidate_gate.py: new - Adds a new script implementing CandidateGateTests.
- tests/test_official_scorer_parity.py: new - Adds a new script implementing OfficialScorerParityTests.
- tests/test_phase1_distance_motion_gate.py: new - Adds a new script implementing DistanceRankTests, MotionPredictionTests, DecisionRuleTests, FrozenReportTests.
- tests/test_phase1_e3_e6.py: new - Adds a new script implementing MotionCoefficientTests, AffineFitTests.
- tests/test_phase1_e6_prefix52.py: new - Adds a new executable or source script.
- tests/test_phase1_topk_gate.py: new - Adds a new script implementing TopKPairTests.
- tests/test_pipeline_validation.py: patch - Updates PipelineValidationTests.
- tests/test_pkgbh.py: new - Adds a new script implementing KagglePackageTests, TemporalUNet3D, SimpleNodeTransformer, FakeCuda.
- tests/test_pkgbh9.py: new - Adds a new script implementing Bh9PackageTests, TemporalUNet3D, SimpleNodeTransformer, FutureAux, FakeCuda.
- tests/test_point_context_oof_graph_gate.py: new - Adds a new executable or source script.
- tests/test_point_context_swap.py: new - Adds a new executable or source script.
- tests/test_point_context.py: new - Adds a new executable or source script.
- tests/test_prepare_selected_graph_temporal_background_gate.py: new - Adds a new script implementing ConflictKindTests.
- tests/test_pseudo_ssl.py: new - Adds a new script implementing _Backbone, _CompactModel, PseudoSSLTest.
- tests/test_public_calibration_intervals.py: new - Adds a new executable or source script.
- tests/test_relative_geometry_gate.py: new - Adds a new executable or source script.
- tests/test_reproducibility_manifest.py: new - Adds a new script implementing ReproducibilityManifestTest.
- tests/test_route_config.py: new - Adds a new script implementing RouteConfigTests.
- tests/test_routing.py: new - Adds a new script implementing RoutingTests.
- tests/test_rst.py: new - Adds a new script implementing Tiny, Rows, Tracks, Dataset, FakeSource, RunnerTest, Array.
- tests/test_score_bh_cv57.py: new - Adds a new script implementing CV57ScoreTests.
- tests/test_scoreall.py: new - Adds a new script implementing ScoreAllTests.
- tests/test_scoring_calibration.py: new - Adds a new script implementing ScoringCalibrationTests.
- tests/test_screen_cg1_abstention.py: new - Adds a new executable or source script.
- tests/test_screen_cg1b_representation.py: new - Adds a new executable or source script.
- tests/test_selected_graph_appearance.py: new - Adds a new executable or source script.
- tests/test_selected_graph_conflict_gate.py: new - Adds a new script implementing _PairMLP, _Transformer, _FirstFeatureGate.
- tests/test_selected_graph_scenario_calibration_gate.py: new - Adds a new executable or source script.
- tests/test_selected_graph_tracklet_context.py: new - Adds a new executable or source script.
- tests/test_site_calibration.py: new - Adds a new script implementing SiteCalibrationTests.
- tests/test_source_softmax_node_threshold.py: new - Adds a new script implementing SourceSoftmaxNodeThresholdTests.
- tests/test_submit_kaggle_alpha.py: new - Adds a new script implementing SubmitKaggleAlphaTests.
- tests/test_sweep_bh_graph_weights.py: new - Adds a new script implementing SweepTopologyContractTests.
- tests/test_t05.py: new - Adds a new script implementing T05PackageTests, TinyBackbone.
- tests/test_ta_hybrid.py: new - Adds a new script implementing AlphaMotionHybridTest.
- tests/test_ta_inference_gate.py: new - Adds a new script implementing TemporalAdapterInferenceGateTest.
- tests/test_ta.py: new - Adds a new script implementing AdapterTest.
- tests/test_temporal_pretrain.py: new - Adds a new script implementing _ZeroAnchor, TemporalPretrainTest, TinyModel.
- tests/test_temporal_rank_oof_graph_gate.py: new - Adds a new executable or source script.
- tests/test_temporal_rank_swap.py: new - Adds a new script implementing ConflictKindTests.
- tests/test_tnb.py: new - Adds a new script implementing T, Tiny, A, G.
- tests/test_trackastra_linear.py: new - Adds a new executable or source script.
- tests/test_trackastra_parity_diagnostic.py: new - Adds a new executable or source script.
- tests/test_u0_t4canon.py: new - Adds a new script implementing U0T4CanonTests.
- tests/test_ultrack_config.py: new - Adds a new script implementing UltrackConfigTests.
- tests/test_v45f.py: new - Adds a new script implementing _FakeV45, V45FastTests.
- tests/test_v45fe.py: new - Adds a new script implementing V45FastEvalTests, StubScorer, Candidates, Pair.
- tests/test_v45g.py: new - Adds a new script implementing V45GraphDiagnosticTests.
- tests/test_v45h.py: new - Adds a new script implementing _FakeV45, FrozenV45HeadTests.
- tests/test_v45par.py: new - Adds a new script implementing HashValidationTests, TinyZarrFixtureTests, TinyCsvFixtureTests, ComparisonTests.
- tests/test_v45z.py: new - Adds a new script implementing V45ZeroControlTests.
- tests/test_v6.py: new - Adds a new script implementing V6ScoreTest.
- tests/test_verify_h1_kaggle_registration.py: new - Adds a new script implementing H1KaggleRegistrationTests.
<!-- CRUISE:SINCE-LAST-PUSH:END -->

## Commit history

<!-- CRUISE:COMMIT-HISTORY:START -->
<!-- CRUISE:COMMIT:3f38a52403018257b2e8e66fc7a2b92d59551173 -->
### 3f38a52 — g02: freeze division deep preregistration

Committed: 2026-07-28T17:58:25+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:cd7f24d30ed17bac9940432a90213a87a1f68e78 -->
### cd7f24d — preregister authentic node topology generator

Committed: 2026-07-28T17:58:28+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:9151934e410fed17cf367afead28ee1558c7ef77 -->
### 9151934 — record node topology governance hashes

Committed: 2026-07-29T00:00:42+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:17c8d6e64d1bab326a53574caf39cc9b72ce5927 -->
### 17c8d6e — docs(g02): attest assoc-cycle governance

Committed: 2026-07-29T00:01:54+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:d23028416cded3c25364d417bd7289b7d283b920 -->
### d230284 — preregister video subsample fidelity gate

Committed: 2026-07-29T02:38:18+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:c6e995ce811a04b229e4f3c6da149fafbe7a2ad9 -->
### c6e995c — freeze common-universe fidelity amendment

Committed: 2026-07-29T02:48:19+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:f1b5dc4d6b932386d5c5f9284a6bbbcc3b0903c9 -->
### f1b5dc4 — preregister division realizability audit

Committed: 2026-07-29T02:50:24+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:8759e1f5082e8c2729e8251c6e08dfed70e2082e -->
### 8759e1f — docs: add subsample fidelity method citations

Committed: 2026-07-29T02:49:44+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:ad35a0e9fd9b60bd1ccfd007e39855c7b4e06dab -->
### ad35a0e — bank unsafe video subsample barrier

Committed: 2026-07-29T03:12:44+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:117801abb6a78508caf981c1445425ed8fc22891 -->
### 117801a — Seal authentic reverse review barrier

Committed: 2026-07-30T00:09:53+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:333251222c0e2c6b59443312da00a785cfa24c98 -->
### 3332512 — Complete reverse supplier flight attestation

Committed: 2026-07-30T00:11:23+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:ee90596cf6731d4b4cfcfb1aaeb7ce6d3fedb01a -->
### ee90596 — g02: authorize bounded DINOv2 GPU bakeoff

Committed: 2026-07-30T01:54:59+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:326d861188a83457ba4fe5046c331d56a8d4a2ae -->
### 326d861 — g02: record DINOv2 GPU launch

Committed: 2026-07-30T01:56:05+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:b10ef911c490aca0cf8ac87c4e649bbca5d00b8e -->
### b10ef91 — g02: record CELLECT mounted-layout pass

Committed: 2026-07-30T02:14:55+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:f7bed832dfb04d83b9debc6edbefd075b5e10d23 -->
### f7bed83 — g02: terminalize two-supplier GPU bakeoff

Committed: 2026-07-30T02:25:52+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT:f16e73e2e416b5d76ea0fbdf0beb6d6d7a186f45 -->
### f16e73e — docs(g03): add Wave R2 literature audit

Committed: 2026-08-01T15:06:34+08:00

- No script changes in this commit.
<!-- CRUISE:COMMIT-HISTORY:END -->

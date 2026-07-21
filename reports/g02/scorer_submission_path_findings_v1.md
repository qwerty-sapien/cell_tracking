# Scorer/submission path unification findings v1

Status: Step 2 released in substance; Step 3 package built and locally verified;
no Kaggle upload or kernel start has been performed.

## Verified

1. The parity gate's actual association-port question passed. On the identical
   persisted `run_bh` caches and the identical restored scorer,
   `run_bh.score_sparse_edge_cache` and `v45feval.sparse_probs` produced exact
   candidate probabilities, exact no-link probabilities, and exact statistics
   for all three videos. The candidate counts were 90,226, 82,240, and 74,329;
   the fitted artifact has 578 parameters and zero runtime-trainable parameters.
   The evidence is
   `reports/g02/run_bh_conflict_edge_f0_parity3_v1/edge_port_parity.json`.

2. Whole-fragment parity was the wrong criterion because the compared fragments
   did not share a detector inference record. The three `run_bh` versus reference
   edge-count deltas `(tp, fp, fn)` were `(-1, -1, +1)`, `(+2, -6, -2)`, and
   `(0, 0, 0)`. Those differences do not contradict the exact identical-cache
   association result. The superseded comparison remains recorded in
   `reports/g02/run_bh_conflict_edge_f0_parity3_v1/parity.json`.

3. The 0.8495036625176058 score is retired as a baseline. It is a one-view,
   57-video `v45feval` result from
   `reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/scores.json`.
   `v45feval` has no `submission.csv` path, so this score cannot be the local
   counterpart of a submission and must not be cited again as the submission
   baseline. The released `run_bh` detector is selected for operational parity,
   not because a detector-quality comparison favored it.

4. The terminal fold-0 scope is exactly 57 videos: the split manifest's 58
   held-out IDs minus explicitly excluded `train:6bba_f4ae811c`. The sorted
   selection SHA-256 is
   `58d090bc83702065b3bc8bc035c5d8e8911519c71a8d3a58f995e5abaf3099f1`.

5. The private scoring package is
   `kaggle_kernel/bhcv57pkg` (`onebitbrain/bhv45-cv57`). It contains five
   submitted files totaling 182,379 bytes. Its deterministic archive contains
   12 members, 611,567 uncompressed bytes, and SHA-256
   `27f0257b020a3c9123e8567ce3191a1307ad934b1c97a4b573d4fd3e5ad7023c`.
   The package manifest SHA-256 is
   `76c4541e4c9cac8160c3ff31e64be326dd3c16e7e49981105cc03ba0f883b052`.

6. The embedded run is frozen to `--mode cv --split train`, the exact 57 IDs,
   `--devices cuda:0,cuda:1`, `--pair-batch-size 1`, `--view-batch-size 1`,
   `--graph-variant legacy`, and edge artifact SHA-256
   `5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034`.
   Here `view-batch-size=1` means sequential execution of the production D4-8
   detector; it does not select one-view inference. No `--test-only-one-view`,
   CPU allowance, accelerated view gate, A0 topology, or division enablement is
   present. The artifact config and package run identity both record divisions
   as `false`.

7. The detector remains the released `bh_model` D4 detector. No compact detector
   path is selected. The packaged `v45_fast`, `v45_head`, and `t_adapter` modules
   are present only because the specified edge artifact's frozen auxiliary
   readout requires them during artifact restoration.

8. `kernel-metadata.json` pins `machine_shape` to `NvidiaTeslaT4`. The launcher
   fails closed after classifying the observed GPU state as one of
   `no_cuda_devices`, `unexpected_accelerator_family`,
   `honored_t4_wrong_device_count`, or `exact_t4x2`; only `exact_t4x2` proceeds.
   Unit tests exercise all four classifications.

9. The claim in `reports/g02/f4cmp.json` that the approximately -0.0062 gap
   establishes a view-ensemble-removal effect is invalid. The cited execution
   evidence at `reports/g02/f4ason2/report.json` records `execution.d4_views` as
   `[0]`; it is itself a one-view run and cannot identify a TTA effect.

## Inferred or pending

- Approximately 73 seconds per video and approximately 1.2 hours total are
  planning projections, not measurements from this package.
- No eight-view 57-video score exists yet. The eventual frozen-scorer result will
  become the new submission-path baseline. Any numerical difference from the
  retired 0.8495036625176058 result will confound detector, artifact, and division
  mode and must not be presented as a TTA measurement.
- Step 4 scoring, the Step 5 submit package, public-test group mapping, Kaggle
  submission, and `scorer_parity_v1.json` update remain pending.

## Reproducibility and boundary

- Work is on `codex/fragmentation-57-rollback`. The prerequisite port/evidence
  commit is `2335ab7` (`Unify run_bh edge artifact scoring`).
- The package was rebuilt byte-for-byte deterministically and all archive member
  sizes/hashes, Python compilation, compact-registry membership, embedded command,
  metadata, and manifest identities were checked locally.
- Relevant tests: 33 passed across `tests.test_pkgbh`, `tests.test_pkgbh9`,
  `tests.test_fitcache_kaggle`, and `tests.test_mkvs45`.
- No training, push, upload, kernel start, public artifact publication, or Kaggle
  submission occurred.

# Scorer/submission path unification findings v1

Status: Step 2 released in substance; the Step 3 kernel completed; Step 4 was
scored with the frozen scorer; and the Step 5 submit package is locally ready.
No submit-kernel upload or Kaggle competition submission has been performed.

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

5. Versions 1 and 2 of private kernel `onebitbrain/bhv45-cv57` were zero-detector
   startup failures. Version 1 stopped after 0.000092214 seconds because Kaggle
   did not export the launcher's sibling `bh45man.json`; the package was made
   self-contained in the monolithic entrypoint. Version 2 attested exact T4x2
   but stopped after 21.163016 seconds because `legacy` was incorrectly passed
   as the graph-variant token. The repaired package uses released graph token
   `v45-motion-lazyrefine-locked` while graph construction separately records
   topology mode `legacy` and divisions `false`. Neither failed version ran
   detector inference or produced scored fragments.

6. Version 3 completed all 57 videos. It attested two Tesla T4 devices as
   `exact_t4x2`, executed D4 views `[0,1,2,3,4,5,6,7]` in exact-sequential mode,
   and produced 57 fragments, 57 feature records, and 57 statuses. The merged
   output has 2,209,761 rows and SHA-256
   `2451a6fbaefece18b65c31d2cb24cc411ff619710823ecd5ac09588ad6813b42`.
   Runner time was 3,828.723371 seconds and launcher time was 3,851.059078
   seconds, or 67.170585 wall-seconds/video. Worker times ranged from
   109.598434 to 187.582864 seconds, with mean 130.040970 seconds.

7. The frozen-scorer result is the new submission-path baseline:
   `score=adj_edge_jaccard=0.8711885823490648`,
   `edge_jaccard=0.8761665470208184`, and
   `node_recall=0.9903079300966847` over 57 videos. With divisions disabled,
   predictions had `division_tp=0`, `division_fp=0`; the ground truth contributed
   `division_fn=42`, so `division_jaccard=0`. The score report SHA-256 is
   `97c86697bb1e3273312d8ef767c1efcf8b439f641adedb5ebbe04aa55477ef32`.
   The 57-fragment aggregate SHA-256 is
   `d0eac26d00052a00f84078372f63fa7e48c3cdd028912462b0eb634a40e023c5`.

8. Numerically the new score is `+0.021684919831459037` above the retired
   0.8495036625176058 result. This is not a TTA measurement: detector, edge
   artifact, and division mode all differ. The old result remains retired and
   must not be reused as the submission baseline.

9. The detector remains the released `bh_model` D4 detector. No compact detector
   path is selected. The packaged `v45_fast`, `v45_head`, and `t_adapter` modules
   are present only because the specified edge artifact's frozen auxiliary
   readout requires them during artifact restoration.

10. The Step 5 package at `kaggle_kernel/bh45pkg` is private kernel
    `onebitbrain/bhv45-private`. Its embedded command is `run_bh.py --mode submit
    --split test`, D4-8 exact-sequential, released graph token, legacy topology,
    divisions off, and the same conflict artifact SHA-256
    `5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034`.
    The physical monolithic entrypoint is 238,819 bytes with SHA-256
    `d4fb3cc137958dc179083fdcc31a8ba5b0ed5a5386e7b8cc4238aaa1051f9ea7`;
    the source archive is 139,102 bytes with SHA-256
    `fa5e04d6ecc348cccee8d89bad12808ceeb658371d3f487decc9655f904ac9d1`.

11. `kernel-metadata.json` pins `machine_shape` to `NvidiaTeslaT4`. The launcher
   fails closed after classifying the observed GPU state as one of
   `no_cuda_devices`, `unexpected_accelerator_family`,
   `honored_t4_wrong_device_count`, or `exact_t4x2`; only `exact_t4x2` proceeds.
   Unit tests exercise all four classifications.

12. The four public test videos map exactly to the split manifest's four
    `strata.excluded_group_ids`: `test:44b6_0113de3b` to
    `g02-b5b4c8611d14800598f7`, `test:44b6_0b24845f` to
    `g02-766deec83da7d2f1909f`, `test:6bba_05b6850b` to
    `g02-58c0feddf2d101f28b12`, and `test:6bba_05db0fb1` to
    `g02-c9c155b6b532ca532f63`. Each group contains the same-named train member,
    matching the four known train-test duplicates. The four group IDs have zero
    intersection with the conflict artifact's 117 fit source groups. Thus the
    public sample is verified unseen by the grouped CV folds and by this edge
    artifact fit. This mapping does not establish whether the separately
    released detector's pretraining saw related imagery.

13. The claim in `reports/g02/f4cmp.json` that the approximately -0.0062 gap
   establishes a view-ensemble-removal effect is invalid. The cited execution
   evidence at `reports/g02/f4ason2/report.json` records `execution.d4_views` as
   `[0]`; it is itself a one-view run and cannot identify a TTA effect.

## Inferred or pending

- The released detector's pretraining exposure to imagery related to the four
  public test groups is not established by the grouped-split and edge-artifact
  evidence above.
- Upload/start of the private submit kernel and the subsequent Kaggle competition
  submission both require explicit approval. Neither has occurred.
- `reports/g02/scorer_parity_v1.json` correctly remains
  `no_observed_kaggle_pairs`; it will be updated only after Kaggle returns the
  score paired with local baseline 0.8711885823490648.

## Reproducibility and boundary

- Work is on `codex/fragmentation-57-rollback`. The prerequisite port/evidence
  commit is `2335ab7` (`Unify run_bh edge artifact scoring`).
- The submit package was rebuilt byte-for-byte deterministically and all archive
  member sizes/hashes, Python compilation, four-video compact-registry membership,
  embedded command, metadata, and manifest identities were checked locally.
- Relevant tests: 48 passed across `tests.test_pkgbh`,
  `tests.test_score_bh_cv57`, `tests.test_scoreall`, `tests.test_pkgbh9`,
  `tests.test_fitcache_kaggle`, and `tests.test_mkvs45`.
- No training, push, submit-kernel upload, public artifact publication, or Kaggle
  competition submission occurred. The private CV57 kernel versions described
  above are the only external compute used in this work.

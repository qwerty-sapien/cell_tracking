# G-02 active flight plan

status: active

## Outcome

Execute `reports/g02/screening/DAG.md` toward a materially stronger cell-tracking
baseline above 0.88 and then competitive 0.90s performance, retaining 0.92 as the
contractual endpoint.

## Current micro-goal

Bank the frozen pretrained Trackastra signal and design its grouped-OOF
held-graph successor:

1. preserve APNCC `0.8815220699597315` as the banked candidate and exact graph
   control;
2. bank the authenticated full-pool runtime and resumable five-fold gate;
3. PARK the v1 point-context model at held OOF precision `0.942940`, before
   graph reconstruction;
4. bank the fixed-grid threshold diagnostic: no shared calibration precision
   floor preserves both cross-fold precision and useful support;
5. bank the pinned BSD-3-Clause Trackastra checkpoint/source qualification and
   its zero-wrfeat fit-only signal gate (`0.918491` ROC AUC versus `0.841227`
   for pair NCC over 497 unique proposals);
6. bank complete deployed-proposal feasibility: all 110,471 geometry-admissible
   proposals across 69 fit-train videos fit into 5,824 transition calls with
   zero missing endpoints, maximum 870 required nodes, and a safety-factored
   1.213-hour frozen-inference projection;
7. preregister and implement a five-fold frozen-head candidate whose promotion requires
   proposal precision plus exact held-fit graph gain `>= +0.005` over APNCC;
8. keep validation12, CV57, Kaggle, GPU, and RunPod closed until the applicable
   gate explicitly opens them.

The deterministic 10–12 µm shell branch is closed: exact full-pool counting
showed the train-selected 12 µm constant-velocity tolerance reduced 5,342,071
shell candidates to 3,396,642 while retaining 344/456 correct edges. Frozen
NCC plus node-disjoint/capped acceptance was unsafe on validation12 (3 correct,
4 known-negative, 192 sampled-unlabeled; precision 0.0151; utility -389). The
next useful representation must model competition and abstention over the
ordinary deployment pool rather than expand the geometric shell.

## Completed learned-model research micro-goal (2026-07-24)

Research and preregister the next learned-model options before any compute launch:

1. freeze the fit-only measured bottlenecks and the current `0.881522070` local candidate
   as the evidence base;
2. review primary literature and official implementations for models whose mechanisms
   address those bottlenecks;
3. separate candidates into an M5 lane (<=18 GiB, >=6 GiB background reserve, no swap
   growth, <=7 hours per unattended stage) and a Kaggle lane capable of reaching a
   credible training state inside 30 cumulative T4 GPU-hours;
4. estimate parameters, activation memory, throughput, and time-to-credible-state before
   implementation;
5. define the cheapest fit-only falsification and promotion criteria for each candidate;
6. reject fashionable but weakly matched, compute-infeasible, or leakage-dependent
   architectures;
7. produce a ranked, cited two-lane shortlist and machine-valid candidate manifests.

This is research-only: do not launch local, Kaggle, or RunPod training and do not inspect
the holdout.

Outcome: `reports/g02/model_landscape_20260724.{md,json}` ranks a compact
Trackastra-style point-context transformer first for the local fast-fail lane and
pretrained Trackastra head/LoRA adaptation first for a conditional T4 timing pilot.
All candidate manifests remain `DRAFT`; literature establishes mechanism relevance but
does not by itself establish the locked 3.5% competition-score improvement.

## Banked implementation micro-goal

Bank and operationalize the first confirmed local result above `0.88`:

1. retain the organizer and background-regularized selected-graph lines as PARK;
2. BANK the fit-derived, geometry-prefiltered full 3-D NCC direct-swap mechanism
   at `0.881522070` on the frozen CV57 scorer (`+0.008372476`, with
   `+115 TP / -226 FP / -115 FN`);
3. preserve its leakage boundary, exact thresholds, per-video cap, and
   observation-level comparison as the new local candidate identity;
4. preregister it as the first high-range P0B public probe because it adds more
   calibration leverage than the near-tied `bhp0-lb*` packages;
5. retain the hash-pinned `bhap-submit` package after exact 57/57 package-path
   parity at `0.881522070`, and stop at the explicit human launch/submission
   gate (7 minutes expected, at most 0.25 Kaggle GPU-hours, one screening
   submission);
6. bank the read-only residual audit: 4,407 FP+FN remain, top 10 videos carry
   52.33%, and 378 ideal wrong-to-right swaps would reach local 0.90 at fixed
   node adjustment;
7. PARK the first fit-only linear tracklet-context proxy: it raised validation
   utility by 24 but missed the 0.95 precision floor both out-of-fold and on
   validation;
8. move to fit-only extended-radius reachability/background-abstention work;
   never use CV57 residual identity or rank as a router;
9. BANK the reachability result: 719 fit-only correct edges lie outside the
   10 µm/cap-8 tensor, and 12 µm/cap-8 recovers 456 at 1.678x candidate cost;
   require explicit 10–12 µm shell-background abstention before replay;
10. PARK the exact shell geometry + incumbent-relative 3-D NCC gate: its best
    supported rule achieved only 0.1437 fit-train and 0.2258 fixed-validation
    precision after sampled unmatched background. Do not replay the shell on
    CV57; pivot to orthogonal improvements on the banked 0.881522 candidate;
11. PARK the exact full-graph temporal/background gate: every grouped-OOF
    configuration violated at least one held-out video background budget, and
    the safest diagnostic reached validation utility 111 versus 148 for the
    banked pair-NCC rule. The next association representation must recover
    recall rather than merely strengthen abstention;
12. PARK the cap-aware temporal ranker: it recovered validation utility `+5`
    but fell to 0.9004 precision, with grouped-OOF precision only 0.9196.
    Cross-video ranking/calibration—not proposal count—is now the local
    bottleneck;
13. PARK the all-family temporal model, but preregister a `6bba`-only hybrid:
    `6bba` clears OOF precision 0.9648 with +313 utility and validation
    precision 0.9835 with +28 utility; `44b6` stays on banked APNCC. Advance
    only if the hybrid beats 0.881522 on exact CV57.

## Touchpoints

- `.cruise/FLIGHT.md`
- `scripts/compare_submissions.py`
- `scripts/sweep_bh_graph_weights.py`
- `scripts/fit_relative_geometry_swap_gate.py`
- `scripts/prepare_relative_geometry_swap_gate.py`
- `scripts/evaluate_relative_geometry_swap_gate.py`
- `scripts/prepare_selected_graph_conflict_gate.py`
- `scripts/fit_selected_graph_conflict_gate.py`
- `scripts/evaluate_selected_graph_conflict_gate.py`
- `scripts/audit_selected_graph_gate_support.py`
- `scripts/audit_selected_graph_appearance.py`
- `scripts/audit_selected_graph_appearance_support.py`
- `scripts/audit_selected_graph_appearance_combined.py`
- `scripts/evaluate_selected_graph_direct_swap.py`
- `scripts/evaluate_organizer_fit_proxy.py`
- `scripts/audit_organizer_fit_monitor.py`
- `scripts/analyze_appearance_residual.py`
- `scripts/audit_selected_graph_tracklet_context.py`
- `scripts/audit_extended_radius_reachability.py`
- `scripts/prepare_extended_radius_shell.py`
- `scripts/augment_extended_radius_shell_appearance.py`
- `scripts/audit_extended_radius_shell_appearance_support.py`
- `scripts/prepare_selected_graph_temporal_background_gate.py`
- `scripts/fit_selected_graph_temporal_background_gate.py`
- `scripts/fit_selected_graph_temporal_rank_gate.py`
- `scripts/fit_selected_graph_temporal_family_rank_gate.py`
- `src/celltrack_alpha/temporal_rank_swap.py`
- `src/celltrack_alpha/point_context.py`
- `src/celltrack_alpha/point_context_swap.py`
- `src/celltrack_alpha/bh_model.py`
- `scripts/evaluate_point_context_oof_graph_gate.py`
- `scripts/falsify_e4_e5.py`
- `src/celltrack_alpha/edge_data.py`
- `src/celltrack_alpha/relative_geometry_gate.py`
- focused tests under `tests/`
- `reports/g02/screening/**`
- narrowly necessary scorer/read-only artifact inputs already named by the DAG

## Invariants

- Use only the attested scorer path and official physical matching semantics.
- Keep scorer semantic parity separate from local-to-Kaggle transfer.
- Preserve grouped folds and untouched audit membership.
- Do not modify or discard unrelated dirty-worktree changes.
- Do not collapse observation distance into an arbitrary scalar.
- No Kaggle submission or RunPod launch without explicit human approval.
- Cap initial training at 30 cumulative Kaggle T4 GPU-hours per model and archive
  candidates that cannot become viable inside it.
- Prefer literature-validated lightweight fast-fail candidates. Before implementation or
  training, require a passing candidate manifest with a primary source, mechanism-to-fit
  failure mapping, >=3.5% expected improvement, and preregistered failure/promotion gates.
- Keep at most one fit-only local M5 training lane available in parallel with eligible
  external work. Enforce <=18 GiB training memory, >=6 GiB background reserve, no holdout
  access, zero swap growth, and <=7 hours per unattended stage with the local guard.
- Do not consider RunPod before an accepted score reaches 0.89; thereafter require a
  higher-quality-GPU-dependent optimization with literature/theory-backed expected
  improvement greater than 5%, the $10 total cap, and exact-launch approval. The ordinary
  hard improvement gate is 3.5%.
- Count every future calibration probe against five submissions/day and preserve two
  final selections.
- RunPod remains launch-specific approval-gated; commits and private pushes are allowed.

## Non-goals for this micro-goal

- Kaggle submission or score retrieval
- Kaggle GPU or RunPod launch
- paid compute
- final portfolio selection
- destructive history changes
- rebuilding the dead P4 edge-attention layer or introducing a GNN

## Acceptance evidence

- focused comparison-tool tests pass;
- canonical equality is invariant to row order and presentation IDs;
- spatial matching is deterministic, time-aware, one-to-one, and in physical units;
- node, edge, and fork/division comparison remains a named vector;
- cached topology replay records the exact selected identities, topology mode,
  learned/no-link artifact identities, baseline score, and scored-partial coverage;
- the pre-registered slice uses the banked `c=0.25` fragments as its control;
- a full 57-video confirm is entered only at `>= +0.005` slice improvement;
- the banked direct-swap candidate reproduces `0.8815220699597315` and
  `34,320 / 2,459 / 1,948` on the frozen scorer;
- its package aborts on any missing/mismatched threshold, artifact, stage, or
  proposal-cap identity;
- the next public action consumes at most one screening submission before P0B is
  updated and the remaining order is redesigned;
- residual-headroom arithmetic reconciles all 57 before/after scorer rows and
  sizes local 0.90 without selecting from CV57 labels;
- all changes pass focused tests and staged-diff checks.

## Completed evidence

14. The preregistered `6bba` temporal hybrid completed full frozen CV57 at
    `0.8801140023577207` (`34,308 / 2,511 / 1,960`), which is
    `-0.0014080676020108` versus the banked APNCC direct swap. It is PARK;
    APNCC `0.8815220699597315` remains the banked baseline and public package.
15. The lower APNCC threshold was screened only on untouched
    fit-validation12. It was observation-distinct but score-neutral
    (`-0.0000003433`, identical TP/FP/FN), so it is PARK and did not enter
    CV57.
16. The compact point-context candidate passed the coordinator pre-training
    gate and full fit117 dry-data audit: 25,289 tokens, 8,382 windows, max 28
    tokens/window, 100% positive retention, max three parents/target, zero
    multi-positive targets, and zero CV57 overlap. A guarded local pilot is
    now the next action; external compute remains unauthorized.
17. Two guarded pilot launches were terminated before optimization because
    host swap grew by 1.50 MiB and 80.75 MiB. Peak child RSS stayed below
    0.73 GiB, so this is a host-pressure runtime block, not a model failure.
    Do not weaken the zero-swap rule or kill unrelated processes. The exact
    pilot was subsequently retried under stable host conditions in item 21.
18. The independently motivated calibrated 3-D crop-linker passed its fit117
    data gate: 4,008 positives, 1,885 guaranteed negatives, 19,396 explicitly
    masked uncertain pairs, disjoint 93/12/12 fit/calibration/validation
    identities, zero CV57 overlap, and verified `16×32×32` crop IO. Its model
    and temperature-calibration primitives are implemented and tested.
19. Its hash-pinned 117-video crop dataset contains 5,893 supervised pairs in
    590,520,183 bytes. The guarded 500-step CPU pilot completed with zero swap
    growth and improved validation utility by 33 at 0.9846 overall precision,
    but PARKed because 44b6 precision was 0.8889 versus the locked 0.93 floor.
    A post-hoc threshold audit could not repair 44b6. The exploratory 6bba-only
    result is promising but cannot unlock CV57; any continuation needs a new
    grouped-OOF preregistration.
20. The distinct 6bba-only crop candidate completed five-fold nested OOF on 69
    eligible videos without validation12 or CV57. It gained 471 utility over
    pair NCC and improved support, but aggregate precision was 0.9233 and two
    folds were below 0.93. PARK the crop-linker branch; do not threshold-retune
    or replay it on CV57.
21. The compact point-context pilot completed 500 guarded CPU steps with zero
    swap growth. It improved validation utility by 39 at 0.9636 overall
    precision, but PARKed because 44b6 precision was 0.8333. The 6bba slice
    reached 182/184 at 0.9891, but prior surrogate-to-graph failure means any
    new family-scoped successor must pass actual held-fit graph scoring before
    CV57.
22. The held-fit surrogate-to-graph diagnostic reproduced all five historical
    6bba temporal thresholds, generated one leakage-safe prediction for each of
    69 fit-train105 videos, and produced 69/69 observation-distinct fragments.
    Despite proposal OOF precision 0.96485, the learned graph lost 0.000612
    adjusted edge Jaccard and 0.000641 raw edge Jaccard to exact APNCC
    (+2 TP, +49 FP, -2 FN). The graph gate correctly PARKed the known CV57
    failure. BANK the gate, keep APNCC banked, and require it for every learned
    successor. Guard peak RSS was 1.304 GiB with zero swap growth.
23. The deterministic 10–12 µm motion-shell screen is PARK. The fixed 8 µm
    constant-velocity tolerance reduced sampled train candidates by 99.19% but
    retained only 14/405 positives, admitted two known harmful links and 854
    sampled-background proposals. Frozen validation retained 6/51 positives,
    one harmful link and 159 sampled-background proposals. Outputs reproduced
    byte-for-byte, 17 focused tests passed, and isolated commit `1bc8a6b`
    touched only the screen, test and final JSON/Markdown artifacts. No CV57,
    Kaggle, GPU or RunPod action occurred.
24. The 6bba point-context successor now has a hash-locked five-fold nested
    fit/calibration/held contract and a deployment-complete runtime that keeps
    every challenger and displaced-conflict identity. A real 52,323-edge video
    produced 144,677 raw proposals, 3,605 geometry-admissible tokens over 99
    transitions, and no sampled-background inference proxy. Five guarded
    launches stopped before fitting on global swap growth (4.69, 131.56, 46.07,
    99.00, and 38.26 MiB); partition-at-a-time loading and the informed v5
    retry reduced peak process RSS to 0.845 GiB, confirming host pressure
    rather than a model working-set breach. Preserve the guard records, do not
    weaken zero-swap policy, and retry the same sealed screen only when host
    paging stabilizes.
25. P0B now has a six-rung full-CV57 dispersion ladder spanning
    `0.848394516` to `0.881522070`. A memory-bounded Phase-0 comparison
    hash-validated every fragment tree and found all 15 rung pairs
    observation-distinct. The `lb16`/`c0` and `lb4`/`c2` matched pairs are also
    distinct, not vacuous, but remain lower-value same-system interactions.
    APNCC-direct is the only local >0.88 rung and remains the one recommended
    public measurement behind explicit human approval. No submission, GPU or
    RunPod action occurred.
26. P0B Stage A now consolidates exact mechanism-diverse anchors. P4
    no-attention provides one cached matched-local interaction; APNCC-global is
    a score-separated solver control; APNCC-direct is the only high-range
    anchor with a reproduced fail-closed public package. The full association
    cost, detector-threshold, and division candidates all failed before
    full-CV57 promotion. Do not spend a calibration-only slot; if approved,
    measure APNCC-direct alone and redesign after its rounded public result.
27. The interval-aware CAL audit proves that no affine line intersects all
    three existing public display bins, even without a nonnegative-slope
    constraint. Each bin needs an additional 0.000343 expansion per side
    before feasibility; leave-one-pair-out misses are 0.017000, 0.000686, and
    0.000715. Pairwise APNCC-direct projections span 0.859749–0.885996, so the
    center OLS forecast is descriptive only. P0B remains STILL-UNRESOLVED and
    APNCC-direct remains the exact next approved measurement.
28. The point-context training loader now authenticates each complete sparse
    cache but decodes only its detection registry. The largest cache dropped
    from 0.669 to 0.446 GiB peak RSS, and a complete 41-video fit partition
    completed at 0.464 GiB with zero swap growth. v6 still stopped on global
    paging; two idle `/bin/sleep` controls independently grew swap by 27.87 and
    10.06 MiB at 0.001 GiB RSS. BANK the memory repair, make no model-quality
    claim, and require a fresh 45-second zero-swap idle preflight before v7.
29. The sealed point-context screen now publishes each completed outer fold as
    one atomic, resumable directory. Its fitted model, calibration metrics,
    held-proposal metrics, contract SHA, implementation hashes, partitions,
    and artifact hash are cross-validated on restore; partial or drifted folds
    fail closed, while unpublished staging directories are non-authoritative.
    Contract SHA is `a374a9e0…6f53`; 22 focused tests pass. This changes
    execution durability only—not folds, labels, thresholds, or graph gates.
30. The preregistered fit-only per-video competition-calibration screen is
    PARK. Across 11 raw/robust/eCDF/cap-rank rules, none beat raw OOF utility;
    the best-precision normalized rule lost 195 OOF utility and 56 validation
    utility, with both family guards failing. This closes scalar distribution
    normalization without CV57 and strengthens the need for explicit
    point-context competition rather than another threshold transform.
31. Two fresh 45-second idle controls passed with exactly zero swap growth.
    The resumable compact point-context screen then completed all five
    fit/calibration/held folds. It accepted 2,068 labeled held proposals
    (1,950 correct / 118 harmful) at precision `0.942940`, below the locked
    `0.95` prerequisite; folds 2 and 4 were `0.901304` and `0.897500`.
    The candidate is PARK before graph reconstruction, validation12 and CV57
    remained closed, and APNCC `0.881522070` remains the banked baseline.
    Final guarded peak RSS was 0.639 GiB with zero swap growth; report SHA-256
    is `48868756…ec36`.
32. The preregistered point-context threshold-headroom diagnostic reproduced
    all five banked held metrics, then tested shared calibration precision
    floors 0.95–0.99. None passed: floor 0.99 reached aggregate precision
    0.954376 but fold 2 remained 0.925049, fold 3 accepted nothing, and utility
    fell to 927. A held-label oracle reached 0.960812/1,869 utility, showing
    separation exists but calibration transfer does not. Close threshold-only
    repair; no graph, validation12, or CV57 action occurred. Report SHA-256 is
    `18c16840…d683`.
33. The official Trackastra CTC checkpoint and its pinned training source
    passed a fail-closed provenance audit: BSD-3-Clause, source commit
    `fb265888…e936`, 27,456,880 parameters, and finite local CPU inference at
    the 1,024-token cap. A preregistered five-video fit-train signal probe then
    scored 497 unique challenger/incumbent proposals without fitting:
    Trackastra ROC AUC `0.918491` versus pair-NCC `0.841227`
    (`+0.077265`). BANK mechanism evidence and authorize grouped-OOF
    frozen-head design only. The released checkpoint expects twelve
    mask-derived features, so zero-filled compatibility remains a domain
    mismatch; require complete-pool feasibility and exact held-fit graph gain
    before validation12/CV57. No Kaggle, GPU or RunPod action occurred.
34. The label-free Trackastra complete-pool audit passed all 69 fit-train
    `6bba` videos. The frozen APNCC geometry gate retained 110,471 of
    2,309,298 conflicting proposals over 5,824 transitions. Every challenger
    and incumbent endpoint fits in one deterministic 1,024-token window:
    maximum required endpoints `870`, maximum raw four-frame detections
    `4,157`, and zero missing identities. Safety-factored frozen inference
    projects to `1.213` hours. Authorize grouped-OOF frozen-head
    implementation, not validation12/CV57; the densest transition's 154
    optional context tokens remain a transfer risk. Report SHA-256 is
    `2296f6b9…2cc7`.
35. The frozen Trackastra successor now has a locked nested five-fold OOF and
    exact held-fit graph contract (`89350403…da6`). It uses a fixed 49-feature
    ridge head with no hyperparameter search, disjoint calibration thresholds,
    overall/every-fold precision and support gates, and the banked APNCC graph
    as its exact control. The label-free cache builder and resumable runner
    pass 19 focused tests plus five model-gate subtests. A governed build
    authenticated and atomically published video 1 (3,605 proposals / 99
    calls), then stopped on 11.50 MiB system swap growth at only 1.06 GiB RSS;
    an earlier attempt stopped on 33.88 MiB, while a sleep-only control
    independently grew 54.87 MiB. Preserve the valid checkpoint, do not weaken
    the zero-swap rule, and resume only after a fresh clean idle preflight.
    Validation12, CV57, Kaggle, GPU and RunPod remained closed.

## Stop and realign

Stop for user input only if work requires a Kaggle submission, GPU launch, new external
authority, changed audit membership, changed scorer semantics, or a materially different
model direction.

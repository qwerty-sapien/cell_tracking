# Closeout: G-02 - Trained Hybrid Tracker to 0.92

- Execution state: Active
- Approval and Start record: migrated paths `.cruise/runs/sparse-hybrid/plan-ap2.yaml` and `goal-appr.yaml`; exact mapping in `migration.yaml`
- Activation: native Goal active on thread `019f4267-9bc0-7900-a696-a3ff416f89c2` at `2026-07-16T09:12:39Z`
- Executed baseline: commit `c3b9b703b049fc25f10e2b42bb986bb94f137a72`, dirty-state digest `6c4d0a9af20a760e7ff6dbde5dcdae0c3f0f72bc43c8b2c1336dbcaf5238ca6d`
- Current T-02 source identity: `g02_integrity.py` `1836155b...fb52`; `freeze_g02_integrity.py` `9ef08dd5...ef33`; `test_g02_integrity.py` `1358c388...2857`
- Current integrity artifacts: parity `0df49311...ae16`; registry `21f3b3f8...2158`; split `ddb313cb...8769`; provenance `6a9139bf...f14f`
- Current T-03 installed identity: adapter `b90cc88c...e243`; trainer `b59e5467...9306`; tests `07835b10...4f2c` and `cf983619...d494`
- Last updated: `2026-07-24` (authority/DAG amendment adopted)

## 2026-07-24 coordinator amendment

The coordinator explicitly adopted the revised screening/calibration DAG. Scorer semantic
parity gates local scoring; local-to-Kaggle transfer calibration gates public forecasting,
probe interpretation, and final selection only. Local model development remains active
while transfer calibration is unresolved. The five-submissions-per-competition-day and
two-final-selection caps apply to calibration probes. Ordinary commits and pushes to this
private repository are permitted. Every RunPod launch requires explicit human approval
after its job identity, estimated GPU-hours, and maximum cost are recorded. The immediate
performance milestone is to break 0.88, followed by robust performance in the competitive
0.90s; the contractual completion target remains 0.92.

## Outcome

Cruise readiness is restored and Doctor passes with runtime digest `1b97b53d...edd9`. T-02 is complete. T-03 implementation is now ready for private training: the exact released checkpoint restores strictly, the compact adapter and armed-only sparse trainer are installed through action `A-T03E2`, and one real fold-fit GEFF/Zarr window completed an optimizer step. T-03 remains active until actual fold training produces retained diagnostics and checkpoints.

## Node attempts

| Node | Attempt | Owner | Actual result/output | Focused evidence | State |
| --- | --- | --- | --- | --- | --- |
| T-02A | 01 | evaluation/data-integrity worker | Located official metric sources and defined the bounded grouped-split implementation | Legacy T-02A report recorded in migration history | Complete |
| T-02B/B2 | 01-03 | integrity implementation worker | Initial control mutation failed safely; retry produced three attested drafts | `t02b2-rpt.yaml` | Complete after retry |
| T-02C/C2 | 01-02 | integration worker | Initial stale guard stopped safely; retry projected source, CLI, and tests | `t02c2-rpt.yaml`; 8 tests | Complete after retry |
| T-02D/E | 01-02 | data-integrity worker | Established actual archive bounds of 199 train + 4 test = 203 and corrected explicit limits | `t02d-rpt.yaml`, `t02e-rpt.yaml`; 11 tests | Complete |
| T-02F/G | 01-02 | data-integrity worker | Diagnosed unresolved signatures: all 20,300 image chunks exceed 1 MiB | `t02f-rpt.yaml`, `t02g-rpt.yaml` | Complete diagnosis |
| T-02H/H2 | 01-02 | implementation/test worker | Added bounded-prefix streaming; repaired test-only fixture defects | `t02h2-rpt.yaml`; 14 tests, 0 skips | Complete |
| T-02I | 01 | data-integrity worker | Two bounded real-data freezes produced byte-identical manifests | `t02i-rpt.yaml`; A 61.67 s, B 35.51 s | Complete |
| T-02J/J2 | 01-02 | projection worker | First retry used obsolete guard alias; second was rejected because the reinstalled launcher protects worker-created run controls | `t02j-rpt.yaml`, `t02j2-rpt.yaml`; no product writes | Failed safely, preserved |
| T-02K | 01 | root integration owner | Adopted the approved `reports/g02` parent ledger and projected exact T-02I bytes under attributed action `A-T02K` | `drift-a6.yaml`, `drift-r6.yaml`; guard clean | Complete |
| T-02L | 01 | independent read-only verifier | All byte, semantic, linkage, leakage, and focused-test gates passed; worker-only host-lock check was unavailable | `t02l-rpt.yaml`; root canonical guard check passed | Complete with root adjudication |
| T-03A/B | 01-02 | checkpoint orientation workers | Established the official compact architecture and then exact public checkpoint identity/restore | `t03b-rpt.yaml`, `ckpt-src.yaml` | Complete orientation |
| T-03C | 01 | adapter/trainer worker | Produced strict-restore adapter drafts, but root review found the non-dry trainer was intentionally gated | `t03c-rpt.yaml` | Rejected before projection |
| T-03D/D2 | 01-02 | trainer repair worker | First attempt hit a temp-root boundary; retry implemented real optimizer, streaming, resume, and 28 tests | `t03d-rpt.yaml`, `t03d2-rpt.yaml` | Superseded by real-format repair |
| T-03E/E2 | 01-02 | real-format repair worker | Repaired directory-Zarr and registry group-to-stem defects; proved one real fit-only optimizer step and 40 tests | `t03e2-rpt.yaml`; 10.52 s, 5.54 GiB RSS | Ready for training launch |
| T-03V | 01 | independent read-only verifier | Product/action/manifests/real/dry evidence passed; worker scratch lock prevented 25 test setups | `t03v-rpt.yaml`; root 40/40 and guard clean | Infrastructure-limited, root adjudicated |

## Changes and effects

| Scope/resource | Change/effect | Permission | Evidence |
| --- | --- | --- | --- |
| Native Goal | Activated G-02 | Approved Start | `activation.yaml` |
| Cruise naming | Migrated active run artifacts to <=15-character names | Explicit user approval | `.cruise/repo.yaml`, `migration.yaml` |
| T-02 implementation | Added integrity module, freeze CLI, and focused tests | T-02 owned paths | Actions `A-T02C2`, `A-T02E`, `A-T02H`, `A-T02H2` |
| T-02 evidence | Added four deterministic JSON manifests under `reports/g02` | T-02 owned paths | Action `A-T02K`; exact T-02I hashes |
| Data access | Read ZIP central directory, bounded metadata, and at most three 1 MiB image prefixes/video per freeze | Approved local bounded reads | 609 image chunks; no full extraction |
| External effects | No network, Kaggle mutation, package install, paid compute, or submission | Preserved boundary | T-02 reports and clean guard |
| T-03 checkpoint | Attested public baseline bytes; strict CPU double-load/core restore; discarded dense transformer | T-03 owned evidence | `ckpt-src.yaml`, `reports/g02/ckpt-v1.json` |
| T-03 implementation | Added compact adapter, sparse trainer, restore/training tests, and training manifest | T-03 owned paths | Action `A-T03E2`; `t03-ready.yaml` |
| T-03 real smoke | Extracted one fit-only GEFF plus two image chunks and ran one CPU optimizer step | Bounded local fit read | 8 nodes, 4 edges, no collisions, no Kaggle/spend |

## Completion evidence

| AC | Expected check/result | Current evidence | Result |
| --- | --- | --- | --- |
| AC-01 | Official scorer semantics and transfer | 7 fixtures pass; official source/constants frozen; transfer evidence is tracked independently from scorer semantics | PASS for local semantics; interval-aware transfer remains unresolved under T-08/T-09 |
| AC-02 | Leakage-safe grouped three-fold and untouched audit20 | 199 train/4 test/203 total; 199 groups; exact audit20; folds hold out 58/59/58; zero fit overlap | PASS for split/provenance gate; score threshold pending models |
| AC-03 through AC-07 | Model, solver, router, runtime, spend, and Kaggle gates | Compact trainer implementation and one real-fit step pass; actual folds, graph integration, routing, scores, and Kaggle remain pending | IN_PROGRESS |

## Verification attempts

| Attempt | Gate | Status | Evidence/finding | Adjudication |
| --- | --- | --- | --- | --- |
| 01 | E-01/E-02 synthetic | PASS | 14 tests, 0 failures/errors/skips; 8 known no-edge warnings | Accepted |
| 02 | E-01 scorer semantics | PASS | Overall and edge-only parity pass; calibration ledger has zero pairs and null residual | Accepted locally |
| 03 | E-02 registry/split | PASS | 203 videos, 609 chunks, zero unresolved evidence, exact audit20 and grouped folds | Accepted |
| 04 | E-02 fit provenance | PASS | Train/calibration/router/threshold sources equal each fold fit set with no audit or held-out overlap | Accepted |
| 05 | Determinism | PASS | Four A/B files byte-identical; installed files match both | Accepted |
| 06 | Guard integrity | PASS | `A-T02K` accepted and root canonical `foreign-diff-check` reports clean | Worker EPERM was environment-only |
| 07 | E-03 checkpoint/model | PASS | 136 tensors, 2,077,996 elements, exact restore, 29,660 trainable at depth 0, no product transformer | Accepted for training |
| 08 | E-02/E-03 real fit smoke | PASS | One fold-0 fit window, 8 nodes/4 edges, one finite optimizer step, 10.52 s, 5.54 GiB RSS | Accepted as smoke only |
| 09 | E-03/E-06 focused trainer | PASS | Root installed-path suite 40/40; fold-0 dry output byte-identical; folds contain 12 fit-only train videos | Accepted; no fold training yet |
| 10 | Independent verifier | INFRA_BLOCKED | Product/evidence passed; worker could not reset test scratch path, so 25 cases failed in setup | Root 40/40 and fresh guard adjudicate product readiness |

## Deviations and reopened decisions

No product requirement, score threshold, or audit membership changed. The archive has 203 videos rather than the initial 199-total assumption; the corrected explicit bound is 199 train, 4 test, 203 total. Oversized chunks are sampled through a single capped prefix read rather than treated as unresolved. A one-byte `.keep` was intentionally adopted only to establish Cruise's directory-metadata checkpoint, then deleted inside `A-T02K`.

## Remaining risk and next work

T-02 proves local metric semantics and leakage-safe data ownership. T-03 now proves checkpoint provenance, compact sparse training mechanics, and one real-format fit step, not model quality. Actual three-fold training/checkpoints, sparse graph solving, router calibration, grouped CV >=0.92, runtime/cost limits, scored submission, and interval-aware local-to-Kaggle transfer calibration remain pending. Transfer uncertainty does not block leakage-safe local model development. The next action is the private T-03 training entrypoint and pre-launch ledger.

## Current determination

- T-02 mandatory evidence: PASS
- T-02 mandatory gates: PASS
- Goal state: Active
- T-03 implementation checkpoint: PASS; actual fold training NOT_RUN
- Next ready node: T-03 private training launch

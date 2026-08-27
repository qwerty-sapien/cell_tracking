# U0-N3 — owner lock or reject dossier

Date: 2026-07-27  
Verdict: **HUMAN-DECISION**  
Status: analytically complete; execution halted

## Decision requested

Choose exactly one:

1. **`LOCK U0-T4CANON`**
2. **`REJECT`**
3. **`REQUEST REVISION`**

No choice is inferred from the original six-branch trigger. This node is an explicit
human lock boundary.

## What has been proven

- A genuinely separate T4-native contract exists with exactly 12 machine-checkable
  invariants. Its SHA-256 is
  `08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1`.
- The old `H1-ZWRFEAT-CACHE-ATTEST` verdict remains
  `DEAD_STRUCTURAL_M5_T4_DRIFT`; no old evidence or tolerance was amended.
- The fail-closed interface passed 12/12 tests, covering 105 T4 base payloads and
  sidecars plus 105 T4 feature payloads and sidecars.
- Changed row, ID, fold, schema, payload hash, scorer hash, forbidden path, and M5
  decision input all fail closed.
- Two fresh-process 105-video replays are byte-identical at SHA-256
  `1d1a8ac2273acb22aa68f097cd3645fce4929f71faf7fba9f029aaab932b4758`.
- Fitted arrays, calibration decisions, selected IDs, authenticated input-array
  hashes, graph-mutation payloads, and manifests match exactly.
- With the challenger disabled, the canonical mutation payload equals APNCC on
  105/105 videos.
- Preflight and both final replays had 0 MiB swap growth. Peak replay RSS was
  0.332 GiB. No scorer, CV57, GPU, Kaggle, RunPod, or external action occurred.

Primary mechanism context remains the frozen Trackastra representation of Gallusser
and Weigert, ECCV 2024
([paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)).
U0 makes no novelty or model-performance claim; it is beta/governance.

## Material negative evidence

All five 44b6 folds deterministically return `NO_ADMISSIBLE_THRESHOLD` under the
unchanged minimum-100 / precision-0.95 calibration rule. All five 6bba folds have
admissible thresholds.

Consequences:

- U0 establishes reproducible T4-domain governance, not 44b6 model signal.
- `C0-T4TA-LINEAR` is a fail-closed 44b6 no-op under this calibration contract.
- A U0 lock can open independently preregistered C1/C2 cache-only work; it cannot be
  cited as evidence that either alpha works.
- C2 still separately lacks its frozen C0 edge/no-link assignment-cost contract, so
  a U0 lock alone does not satisfy every C2 dependency.

No threshold was lowered to erase this negative result.

## Downstream gates if locked

Every association branch remains:

1. fit-only manifest/leakage validation;
2. 44b6 held-fit graph first;
3. kill immediately at adjusted graph delta `<=0`;
4. require the branch's preregistered 44b6 floor before 6bba
   (`>=+0.005` standalone, or `>=+0.003` only for a branch explicitly frozen as a
   stackable wrapper);
5. require raw edge delta and node-recall delta `>=0`;
6. run 6bba only after the complete 44b6 gate;
7. use CV57 only after both family gates and never for tuning.

## Authority boundary

`LOCK U0-T4CANON` would:

- lock contract SHA-256 `08cd90e5…016c1`;
- accept T4 as the canonical feature domain for later local, cache-only, branch-
  governed experiments;
- return control to the global orchestrator so dependency-satisfied C1/C2 nodes may
  resume under their own gates.

It would **not**:

- resurrect or reinterpret old H1 N1;
- declare the M5↔T4 differences benign;
- weaken a tolerance, threshold, scorer, split, or graph gate;
- authorize a GPU, Kaggle mutation, submission, RunPod, paid compute, push, or other
  external action;
- authorize CV57 before both family gates;
- authorize D1/R0 before a named association arm passes both families;
- claim a score or leaderboard improvement.

## Artifact audit

| Artifact | SHA-256 |
|---|---|
| `U0_N0_contract_preregistration.md` | `b5bb9747f77fb4aaed74cd8c7d41dc6541b5621f589492233fa61791062e5592` |
| `u0_t4canon_contract_draft.json` | `08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1` |
| `U0_N1_interface_verification.md` | `58f7071d9a442676e138aa2dbb2bfae745f8e7dd9cddd77951b602b6b869d298` |
| `u0_t4canon_implementation_manifest.json` | `bdb683e75ddfb8d298e97a7758506bec9cec914d9c1cab6fa933af8b0968c0d1` |
| `U0_N2_exact_replay.md` | `5201f73c6e958f5e3516315f2f39f83b374ed0855ca35d6c28000599b4001fd8` |
| `u0_t4canon_exact_replay_attestation.json` | `b5916b077f8ed66cd6a726e7404946354ceab423ce038877c239411fe2e8fcc1` |

## Non-GO routes

- **REJECT**: leave all U0-dependent graph gates closed.
- **REQUEST REVISION**: specify the invariant or authority defect; no dependent
  execution starts.
- Any request to reinterpret old N1, weaken exactness, or broaden authority is
  **ESCALATE**, not an in-node edit.

## Proposed ledger row

`U0-T4CANON-LOCK-GATE | owner T4-canonical contract decision | HUMAN-GATE |
12 invariants; 12/12 tests; two exact 105-video replays; 44b6 no admissible threshold
in 5/5 folds | NOT SCORED | 0 submissions | 0 GPU-h | 0 RunPod | HUMAN-DECISION |
LOCK / REJECT / REQUEST REVISION; old N1 unchanged DEAD | U0_N3_owner_dossier.md`

## Cascade

Halt. Only a later explicit owner message containing `LOCK U0-T4CANON`, `REJECT`, or
`REQUEST REVISION` resolves this node.

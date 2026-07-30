# SA-DIVISION-DEEP terminal report

Date: 2026-07-28
Branch: `codex/sa-division-deep-wave2`
Verdict: **PARK_DEPENDENCY**
Tier proposal: none; retain `DIV-F1` at `PARK_A`

## Outcome

The frozen full-CV57 division-off identity contains **42** GT division FNs.
The deterministic component audit found:

| Exact class | Count | Denominator |
|---|---:|---:|
| strict scorer roles present, but no component spans parent plus both daughter lineages (`PRESENT_FRAGMENTED`) | **38** | **38/42 = 90.4762%** |
| strict roles present and at least one component already spans all three roles, but no eligible local parent fork | **1** | **1/42 = 2.3810%** |
| at least one strict scorer role missing | **3** | **3/42 = 7.1429%** |
| all strict roles present | **39** | **39/42 = 92.8571%** |

Thus the answer to “how many of the 42 are present-but-fragmented?” is
**38/42**, not 39/42. The 39th role-covered event is already component-spanning
and fails at the local-fork condition. Role presence is not scorer equivalence:
a credited TP additionally needs an eligible local binary fork and global
one-to-one pairing.

Primary evidence:
`oracle_audit_v2.json` @
`6d1a2754bbe597d53c29863290cba51225d3138b002b83ac4d4ad9c5d306c119`.
Its frozen source is
`reports/g02/division_failure_holdout57_postpatch_v1.json` @
`10a8e480cbd0134f0fec18c1d302b388c4dd3ecbaf02f6eb3298f79eb40d2b44`.

The historical prefix-52 result, **35/37** disconnected with **2/37** role
missing, is a different prediction identity and denominator. It is consistent
with the mechanism but was not pooled or substituted. Evidence:
`reports/g02/division_bug_investigation_holdout_prefix52_v1.md` @
`f90b3906cbaf9f5009d60831c8846477e6342872a93869091b5ca1f1513167dc`.

## Frozen D1 contract

Preregistration was committed before the Wave-2 component audit at
`3f38a52403018257b2e8e66fc7a2b92d59551173`. It fixes:

- direct and one-frame-late realizations as timing variants of one
  parent-lineage/unordered-daughter-pair equivalence class;
- the active persistent-fork arm as requiring both branches through the
  grandchild evidence window;
- exact pre-emission suppression by equivalence key and occupied-node
  ownership;
- atomic frozen-N1 `division_fork` ownership of both added edges and every
  incident conflict;
- persistence edges as evidence only, never invented;
- zero duplicate survivors.

On the 39 strict-role-covered events, **39/39** deterministic diagnostic direct
fork actions compiled individually through frozen N1
`2a72950c…f865` / implementation `457f6607…24f5`. The active persistence rule
selected **36**, suppressed **3** for missing two-branch persistence, suppressed
0 for duplicate equivalence and 0 for node conflict, and left **0 duplicate
survivors**. These are held-label oracle actions and structural feasibility
receipts, not model predictions or graph outcomes.

## Dependency and kill line

Fitting remains locked: the qualifying both-family association-continuity-arm
count is **0**. Resume only after one single banked arm passes adjusted
edge-Jaccard delta `>=+0.005`, raw-edge delta `>=0`, node-recall delta `>=0` on
both 44b6 and 6bba, plus a positive label-free connected-fork-opportunity
delta. Do not splice arms or families.

After that dependency passes, evaluate unchanged/D0/D1/persistence-ablation
44b6 first. Kill the D1 instantiation at division Jaccard `<=0.15`, any
negative adjusted/raw/node delta, nonpositive combined delta, or any duplicate
survivor. GO requires division Jaccard `>=0.30`, nonnegative graph vectors,
positive combined delta, and D1 attribution beyond D0 on both families.

No DEAD verdict is proposed.

## Validation and resources

- latest successful idle preflight: **45.062 s**, zero swap growth,
  `preflight_retry4.json` @
  `348ec627be5981887cd588747d5623aaf63e97876dbe5b75e48695d1acfe623b`;
- successful audit: **14.905 s**, peak process-tree RSS **0.212 GiB**, zero
  swap growth, `audit_guard_retry4.json` @
  `45b8541e89f8acd1a5c0fd03568be6747d9e2b1535cca4e60ffbc2438eadb1f6`;
- focused tests: **7/7 PASS**, zero swap growth,
  `test_guard_retry1.json` @
  `4f32d5e4b01dff7d23d74624d390f8f3daf13db5564ed022343966c30e6c04ba`;
- input pins and 26 individual division-bearing-fragment hashes:
  `INPUT_HASH_RECEIPTS.json`.

Two audit attempts failed closed before an outcome because the first Python
environment lacked SciPy and the initial implementation had not converted
official scorer-internal node IDs back to submission IDs. A third authenticated
audit succeeded after that conversion. All failed attempt guards recorded zero
swap growth; their receipts are retained. `oracle_audit.json` is the successful
pre-suppression audit and is superseded by `oracle_audit_v2.json`.

CPU/cache only. No fitting, held-family graph score, scorer rerun, GPU, Kaggle,
budget debit, submission, RunPod, CELLECT touch, external mutation, root
single-writer edit, DEAD action, or quarantined C2/Branch-B number occurred.

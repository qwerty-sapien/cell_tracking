# Division plan-amendment dossier

Status: **HUMAN DECISION REQUIRED — NO AMENDMENT MADE — FITTING LOCKED**

This dossier prepares one decision: whether to unlock a fit-only division arm from the
association-continuity prerequisite. It is not the amendment, it grants no execution
authority, and it does not change any frozen gate.

## Decision summary

The old prerequisite has become a poor sequencing rule, but the deployment problem has
not disappeared. The banked local score is `0.881522`; the planning range corresponding
to public `0.910` is `0.9255–0.9422`, leaving `+0.043978–+0.060678`
(reported as `+0.044–+0.061`). An edge-only current-pool projection caps near `0.896`,
still `+0.0295–+0.0462` short of that target range. Division is therefore one of the few
levers with enough arithmetic headroom:

| Zero-FP division TP | Division Jaccard | Combined-score addition |
|---:|---:|---:|
| 12 | `0.285714285714` | `+0.028571428571` (`+0.0286`) |
| 20 | `0.476190476190` | `+0.047619047619` (`+0.0476`) |
| 25 | `0.595238095238` | `+0.059523809524` (`+0.0595`) |

The evidence supports human review of an unlock; it does not establish that a
non-oracle selector can reach those TP counts.

## Why the association-continuity prerequisite is stale

`B1_division_tp.md` measured a late postprocessor on an already fragmented graph:
production added no edge on 13 valid fixtures and restored `0/51` removed fork edges.
That was valid evidence that constants or a wider orphan gate could not repair missing
connectivity after the fact. It predates two material facts:

1. the campaign now has **38 exact, compiler-valid repair bundles** rather than only a
   generic postprocessor; and
2. six subsequent association mechanisms have produced approximately zero or negative
   usable graph lift, with the formal qualifying both-family arm count still `0`.

Keeping division fitting behind an association win therefore makes the only measured
large residual lever depend on a lane that has repeatedly failed. The proposed decision
is about sequencing only: allow a strictly isolated division selector to be falsified
directly. It is not permission to ignore edge FP, duplicate, or held-fold controls.

## Oracle-only ceiling through frozen N1

The hash-pinned diagnostic execution authenticated all 38 Wave-3 actions, compiled every
action through frozen N1, applied the bundles to the frozen CV57 graphs, and ran the
frozen scorer. This dossier freshly reverified its result, all 38 compiler receipts, the
54-check validation, and its zero-swap resource receipt. This is explicitly
**GT-DEPENDENT** and is a ceiling, not achievable gain:

- division TP/FP/FN: `38/0/4`;
- division Jaccard: `0.904761904762`;
- adjusted/raw/node deltas: `+0.001591651042` /
  `+0.001607524868` /
  `0.000000000000`;
- weighted division term: `0.1 × 0.904761904762 =
  +0.090476190476`;
- exact combined delta: `+0.092067841519`.

Identity check:
`+0.001591651042 + 0.1 ×
0.904761904762 = +0.092067841519`.

This quantifies the ceiling requested for the decision. The separate deployment audit
still says only `1/38` is reachable from the current pool/features at perfect selection
(`+0.002406490`); `37/38` require a stronger representation. Oracle headroom is not
selector evidence.

## False-positive risk is binding

For 25 true division forks, FP erodes the division-only score contribution as follows:

| TP | FP | Jaccard `TP/(42+FP)` | Combined addition |
|---:|---:|---:|---:|
| 25 | 0 | `0.595238095238` | `+0.059523809524` |
| 25 | 5 | `0.531914893617` | `+0.053191489362` |
| 25 | 10 | `0.480769230769` | `+0.048076923077` |

Thus 5 FP reduce `+0.0595` to `+0.0532`; 10 FP reduce it to `+0.0481`. This table is
edge-neutral. Any wrong fork can also damage adjusted/raw edge Jaccard, so it is an
optimistic bound.

## The arm that would run only if a human unlocks it

The existing preregistered `D1-LATE-PERSISTENT-ONE-TO-ONE-FORK` comparison remains
unchanged:

1. `UNCHANGED_GRAPH`;
2. `D0_PARENTAL_SOFTMAX_GREEDY_FORK`;
3. `D1_LATE_PERSISTENT_ONE_TO_ONE_FORK`;
4. `D1_PERSISTENCE_ABLATED`.

Direct and one-frame-late realizations share one scorer-equivalence class. Keep at most
one candidate per equivalence key, then enforce occupied-node ownership, require both
branches to persist, emit the two-edge fork atomically through frozen N1, and require
zero duplicate survivors.

The execution order would be `44b6 → 6bba → CV57`, with **44b6 first**. Kill the
instantiation immediately on 44b6 if division Jaccard is `<=0.15`, any adjusted/raw/node
delta is negative, combined delta is nonpositive, or any duplicate survives. GO remains
division Jaccard `>=0.30`, nonnegative graph vectors, positive combined delta, and D1
attribution beyond D0 on both families. No threshold rescue, family splicing, or
post-outcome change is permitted.

This is a description of the already preregistered arm, not authorization to run it.

## Per-event repair bundles (38/42)

Counts: 9 `44b6`, 29 `6bba`; 37
`SINGLE_DAUGHTER_REPARENT_DIRECT`, 1
`DOUBLE_DAUGHTER_REPARENT_DIRECT`; 36 have a deterministic late-fork equivalent and 2
are direct-only because two-branch persistence is absent. There are no cross-bundle
occupied-node overlaps. For the 36 events with two timing realizations, the duplicate
risk is within-event: direct and late are mutually exclusive representations of the
same scorer event and must be suppressed by equivalence key before node ownership.

| # | Event | Family | Direct required edges | Motif | Preferred late equivalence | Duplicate-suppression risk |
|---:|---|---|---|---|---|---|
| 1 | `train:44b6_267148e4|gt-divider:3|event:0` | 44b6 | `645→841, 645→850` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 645→841; fork 841→1047, 841→1059 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 2 | `train:44b6_341df25f|gt-divider:12|event:0` | 44b6 | `454→555, 454→556` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 454→556; fork 556→653, 556→654 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 3 | `train:44b6_74d0c52e|gt-divider:58|event:0` | 44b6 | `6244→6351, 6244→6358` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 6244→6358; fork 6358→6458, 6358→6466 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 4 | `train:44b6_996155de|gt-divider:59|event:0` | 44b6 | `1762→1860, 1762→1861` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 1762→1861; fork 1861→1958, 1861→1959 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 5 | `train:44b6_d2f34f90|gt-divider:102|event:0` | 44b6 | `9760→9947, 9760→9959` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 9760→9959; fork 9959→10149, 9959→10164 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 6 | `train:44b6_d2f34f90|gt-divider:136|event:1` | 44b6 | `12088→12289, 12088→12302` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 12088→12302; fork 12302→12519, 12302→12532 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 7 | `train:44b6_d5e7d891|gt-divider:86|event:0` | 44b6 | `4351→4718, 4351→4768` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 4351→4718; fork 4718→5103, 4718→5168 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 8 | `train:44b6_d5e7d891|gt-divider:456|event:1` | 44b6 | `20706→21242, 20706→21274` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | no; two distinct grandchild-stage persistence nodes are absent | DIRECT_ONLY: no late alternative; equivalence key remains unique and no cross-bundle node overlap exists |
| 9 | `train:44b6_deabac95|gt-divider:14|event:0` | 44b6 | `6844→7065, 6844→7079` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 6844→7079; fork 7079→7280, 7079→7290 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 10 | `train:6bba_07e24132|gt-divider:120|event:0` | 6bba | `14329→14684, 14329→14694` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | no; two distinct grandchild-stage persistence nodes are absent | DIRECT_ONLY: no late alternative; equivalence key remains unique and no cross-bundle node overlap exists |
| 11 | `train:6bba_07e24132|gt-divider:231|event:1` | 6bba | `24848→25187, 24848→25200` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 24848→25200; fork 25200→25500, 25200→25544 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 12 | `train:6bba_2312ac41|gt-divider:157|event:0` | 6bba | `2466→2662, 2466→2680` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 2466→2680; fork 2680→2862, 2680→2890 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 13 | `train:6bba_2312ac41|gt-divider:284|event:1` | 6bba | `4206→4383, 4206→4387` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 4206→4383; fork 4383→4562, 4383→4567 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 14 | `train:6bba_2819ca14|gt-divider:596|event:0` | 6bba | `3667→3713, 3667→3717` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 3667→3713; fork 3713→3767, 3713→3774 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 15 | `train:6bba_3abfe10a|gt-divider:662|event:0` | 6bba | `49742→50476, 49742→50612` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 49742→50476; fork 50476→51267, 50476→51373 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 16 | `train:6bba_4ffd3da3|gt-divider:245|event:0` | 6bba | `11423→11654, 11423→11664` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 11423→11664; fork 11664→11898, 11664→11915 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 17 | `train:6bba_4ffd3da3|gt-divider:649|event:2` | 6bba | `22508→22695, 22508→22715` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 22508→22715; fork 22715→22911, 22715→22932 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 18 | `train:6bba_61ecbe65|gt-divider:207|event:0` | 6bba | `2545→2603, 2545→2604` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 2545→2603; fork 2603→2657, 2603→2661 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 19 | `train:6bba_6321a359|gt-divider:79|event:0` | 6bba | `408→457, 408→458` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 408→458; fork 458→499, 458→505 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 20 | `train:6bba_6ca87370|gt-divider:526|event:0` | 6bba | `4077→4163, 4077→4168` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 4077→4168; fork 4168→4255, 4168→4261 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 21 | `train:6bba_705ec2c9|gt-divider:262|event:0` | 6bba | `3687→3770, 3687→3772` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 3687→3772; fork 3772→3862, 3772→3863 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 22 | `train:6bba_786893ac|gt-divider:35|event:0` | 6bba | `1179→1969, 1179→2003` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 1179→1969; fork 1969→2740, 1969→2871 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 23 | `train:6bba_786893ac|gt-divider:405|event:1` | 6bba | `13610→14398, 13610→14405` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 13610→14398; fork 14398→15204, 14398→15205 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 24 | `train:6bba_786893ac|gt-divider:1154|event:2` | 6bba | `47697→48382, 47697→48384` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 47697→48382; fork 48382→49074, 48382→74730 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 25 | `train:6bba_7af54fde|gt-divider:575|event:0` | 6bba | `7682→7764, 7682→7779` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 7682→7779; fork 7779→7860, 7779→7876 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 26 | `train:6bba_7af54fde|gt-divider:610|event:1` | 6bba | `7960→8055, 7960→8056` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 7960→8056; fork 8056→8155, 8056→8156 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 27 | `train:6bba_7f87b3d8|gt-divider:201|event:0` | 6bba | `4560→4655, 4560→4664` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 4560→4664; fork 4664→4752, 4664→4759 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 28 | `train:6bba_7f87b3d8|gt-divider:369|event:1` | 6bba | `5569→5643, 5569→5648` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 5569→5648; fork 5648→5730, 5648→5738 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 29 | `train:6bba_87289e13|gt-divider:576|event:0` | 6bba | `13328→13390, 13328→13391` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 13328→13390; fork 13390→13460, 13390→13461 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 30 | `train:6bba_8b7818bf|gt-divider:19|event:0` | 6bba | `193→259, 193→260` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 193→259; fork 259→321, 259→322 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 31 | `train:6bba_8b7818bf|gt-divider:217|event:1` | 6bba | `2288→2363, 2288→2367` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 2288→2363; fork 2363→2433, 2363→2442 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 32 | `train:6bba_b204cac7|gt-divider:611|event:0` | 6bba | `4417→4477, 4417→4480` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 4417→4477; fork 4477→4528, 4477→4534 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 33 | `train:6bba_b204cac7|gt-divider:780|event:1` | 6bba | `5531→5579, 5531→5580` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 5531→5580; fork 5580→5623, 5580→5624 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 34 | `train:6bba_bb9f20c3|gt-divider:565|event:0` | 6bba | `8895→9170, 8895→9188` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 8895→9188; fork 9188→9436, 9188→9461 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 35 | `train:6bba_bb9f20c3|gt-divider:895|event:1` | 6bba | `12911→13164, 12911→13172` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 12911→13164; fork 13164→13414, 13164→13431 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 36 | `train:6bba_bb9f20c3|gt-divider:1660|event:2` | 6bba | `21904→22102, 21904→22112` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 21904→22112; fork 22112→22299, 22112→22303 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 37 | `train:6bba_ebdf3b34|gt-divider:184|event:0` | 6bba | `8481→9089, 8481→9099` | `DOUBLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 8481→9089; fork 9089→9740, 9089→9746 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |
| 38 | `train:6bba_ebdf3b34|gt-divider:745|event:1` | 6bba | `55535→56074, 55535→56126` | `SINGLE_DAUGHTER_REPARENT_DIRECT` | yes; linear 55535→56074; fork 56074→56587, 56074→56668 | MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence key before occupied-node ownership; no cross-bundle node overlap |

Machine-readable bundle details, including net adds/deletes and equivalence keys, are
in `DIVISION_REPAIR_BUNDLES.json`.

## Decision options and barrier

- **Approve a plan amendment:** unlock only the preregistered fit-only D1 arm above,
  preserving its 44b6-first kill line, held-fold discipline, FP accounting, and all
  downstream human gates.
- **Decline or defer:** keep `DIV-F1` dependency-parked and division output OFF.

Until the owner decides, **FITTING REMAINS LOCKED**. No training, fit, graph execution,
6bba, CV57, submission, Kaggle, RunPod, tier move, DEAD promotion, or plan amendment was
performed by this dossier.

## Evidence identities

- frozen plan: `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`;
- B1 result: `c9f2b4d1b8e33f5dae456de9e0ac1649b08f2cda8591a67b9d00d4a355dffbb1`;
- 38-bundle target spec: `0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221`;
- oracle measurement receipt: `0b3c52995a3e18fc705a09bfa5f7f95de6878412acee97901bd5d720199d73cc`;
- per-event dossier: `198aeda1d0c0522bd1aedacdbb48e2b343aca0b25cb41d68dd8cb72cbfc87b14`;
- frozen N1 implementation: `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5`.

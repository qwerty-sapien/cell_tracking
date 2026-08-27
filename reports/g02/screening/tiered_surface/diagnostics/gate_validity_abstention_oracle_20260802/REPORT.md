# Gate validity, abstention oracle, and near-zero kill review

Status: **COMPLETE — OWNER BARRIER / NO GATE CHANGE** (2026-08-02).

This diagnostic used existing fit-only caches and banked scorer results. It opened no
CV57, 6bba, GPU, Kaggle, RunPod, or submission path. Definitions were frozen in
`PREREGISTRATION.md` before the oracle replay.

## D1 — 44b6 gate validity

The exact comparison is the banked APNCC-DIRECT held-44b6 graph against the exact
pre-APNCC legacy graph with `motion_error_weight=0.25`, on the same 36 identities.
All 72 fragment hashes were re-authenticated before the vector was read.

| | adjusted | raw | node recall |
|---|---:|---:|---:|
| pre-APNCC `c=0.25` control | 0.859584981641 | 0.856184798808 | 0.993925630490 |
| APNCC-DIRECT | 0.866534526036 | 0.862596057467 | 0.992618000284 |
| **delta** | **+0.006949544395** | **+0.006411258660** | **−0.001307630206** |

**Verdict: FAIL** under the locked 44b6-first gate `adjusted/raw/node >=
+0.005/0/0`. APNCC clears the adjusted and raw clauses, but its node-recall delta is
negative. Yet APNCC-DIRECT is the campaign's only confirmed full-CV57 win
(`+0.008372`). Therefore the three-component 44b6-first rule is not a valid universal
filter for CV57 improvement. This is a methodological finding only: the gate is not
changed, no failed mechanism is reopened, and further action requires the owner.

## D2 — abstention versus evidence

An action is oracle-beneficial only when its individual exact frozen-score delta
`S(G_action) - S(G_control)` is strictly positive. For R0 both endpoints are free and
the node set is fixed, so an official-matching GT edge (label `1`, or `-2` for a
division truth edge) strictly improves exact edge score; labels `0/-1` add a false
edge and strictly harm it. R0 contained no `-2` actions.

| lane / frozen set | candidates | beneficial | neutral | harmful / false | threshold or sealed set | diagnosis |
|---|---:|---:|---:|---:|---:|---|
| A7 occupied-swap universe | 45,961 | **134** | 45,132 | 695 | accepted 0 | **vacuous abstention**; cannot evidence-park R9 |
| reverse literal-consensus pass set | 268 | **0** | 268 | 0 | node-disjoint 53, all neutral | **evidence-negative for this authentic consensus contract** |
| R0 topology-0 shell universe | 38,081 | **5** | — | 38,076 unmatched FP | sealed 4, beneficial 0 | **selector missed nonempty support**; cannot evidence-park AE-P3/R0 |

For context, reverse began from the same 45,961 occupied-swap universe containing
134 beneficial actions; authentic forward/reverse agreement selected a 268-action
subset containing none. That supports an evidence-park for the literal consensus
mechanism, rather than diagnosing only its cap. A7 and R0 do the opposite: their own
candidate universes contain beneficial support, while their threshold rules found
none. The prior terminal join's reclassification of R9 and AE-P3/R0 to evidence is
therefore retracted. The cited lane-4 snapshot remains authoritative at
evidence/dependency/control/policy/owner-override `22/4/6/1/2`. No tier moves follow.

## D3 — ranked near-zero 44b6 kill review

Rank is owner-request value after D1, not numerical delta order. Cost is a lower-bound
linear projection from the banked held run and excludes any missing CV57 candidate
materialization plus the scientific cost of an irreversible read. No CV57 object was
opened.

| rank | valid historical kill | edits | adjusted / raw / node delta | minimum CV57 work | request? |
|---:|---|---:|---|---|---|
| 1 | SA-SWAP-W1 exact regret | 3,308 | −0.000310288 / −0.000296592 / 0 | ~2.02 CPU-h; no CV57 candidate bank | **No under current scope.** Its deployed edits are closest to APNCC (local, node-disjoint, recall-preserving), but the selector rescored all 45,961 actions and is excluded by the governing full-population-reranking rule. D1 does not override that rule. |
| 2 | BLOCKPEAK-RESTORE-V1 / NODETOPO | 50 | −0.000003436 / 0 / 0 | ~0.063 CPU-h after a CV57 action bank; none exists | **No.** Its frozen generated population has 0 beneficial actions and oracle ceiling 0. |
| 3 | PEFT adapter | 129 | +0.000133827 / +0.000133160 / 0 | >=0.056 T4-h lower bound; no sealed CV57 model/action bank | **No.** Requires new GPU application, rescored the full delete pool, and retained only 3/1,118 beneficial actions. |
| 4 | DELETE-O0 | 14,170 | −0.000127119 / −0.000131807 / 0 | ~0.108 CPU-h; no CV57 candidate bank | **No.** Broad deletion is not recall-preserving and contradicts the banked locality ordering. |
| 5 | DIV-A geometry (historical) | 1 | −0.000141523 / −0.000140538 / 0 | not applicable | **No; superseded.** Section 6.5 replaced this underpowered family-only kill with the already-complete full-38 test, which failed. |

The inventory excludes A7/reverse/R0 zero-edit outputs (D2 abstention diagnostics),
invalid empty DIV-D consensus arithmetic, and older validation12/residual-slice/6bba
screens that were not 44b6-first kills.

Thus D1 makes SA-SWAP the most informative member of the review, but **none of the
near-zero kills is worth a CV57 request under the current design scope**.

## Evidence

- Machine result: `RESULT.json`.
- D3 machine inventory: `D3_NEAR_ZERO_KILL_REVIEW.json`.
- APNCC held graph: `../../../ga_h1_zero_wrfeat_graph/44b6/report.json`.
- pre-APNCC control and NODETOPO graph: `../../branches/sa_nodetopo/graph_gate_44b6/report.json`.
- occupied-swap oracle population: `../../branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json`.
- reverse selection: `../../branches/reverse_endpoint_consensus_audit_v1/SELECTION.json`.
- R0 sealed result: `../../branches/r0_shell_nolink_own_merits_v1/run_v2/RESULT.json`.

**Stop condition reached:** report to owner; do not alter the gate or open any CV57
read autonomously.

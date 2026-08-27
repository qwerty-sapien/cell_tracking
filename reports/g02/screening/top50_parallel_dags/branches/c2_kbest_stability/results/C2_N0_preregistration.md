# C2-N0 preregistration — assignment-level stability

Date: 2026-07-27  
Run: `20260727_top50_six_01`  
Verdict: **PARK — U0/C0 dependency not locked**

## Objective and result

The falsifiable objective is to determine whether one label-blind uncertainty rule
over *feasible component assignments* improves the frozen 44b6 held-fit adjusted
edge Jaccard over APNCC by at least `+0.003`, without raw edge-Jaccard or node-recall
regression and with positive attribution over the same-cost rank-1/MAP control.

C2-N0 does **not** pass. The required owner-locked `U0-T4CANON` contract does not
exist: U0 may analytically preregister, implement, and rehearse, but its written DAG
terminates at `U0-N3` for a separate owner decision. There is also no independently
locked C0 assignment-cost contract defining the complete edge/no-link cost matrix,
dummy/no-link costs, feasibility constraints, and rank-1/MAP output. The old
H1-ZWRFEAT graph contract is locked but is not C0: it specifies thresholded,
node-disjoint direct swaps and remains closed after structural M5↔T4 drift. Treating
that old contract as the missing C0 assignment source would both change the named
mechanism and bypass U0.

Therefore this node is PARKed before implementation, labels, held-fit graphs, 6bba,
or CV57. The remaining sections freeze the proposed transfer so a future owner lock
cannot be used to tune it.

## Hypothesis card

1. **Named mechanism and class.** `C2-KBEST-STABILITY`; alpha wrapper. The beta
   controls are exact APNCC and the rank-1/MAP assignment produced from the identical
   future C0 cost matrix.
2. **Literature basis.** Murty introduced ordered enumeration of assignments
   ([Operations Research 1968](https://doi.org/10.1287/opre.16.3.682)); Lawler
   supplied a general partitioning procedure for K-best solutions
   ([Management Science 1972](https://doi.org/10.1287/mnsc.18.7.401)). Both are
   peer-reviewed primary sources. Trackastra supplies the frozen pretrained
   cell-association representation
   ([Gallusser and Weigert, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)).
   These papers support ranked feasible assignments and pretrained association,
   respectively; they do not claim the G-02 graph gain.
3. **Crowding defense.** A strong pretrained tracker is beta. The differentiating
   transfer is to measure uncertainty over globally feasible one-to-one assignments,
   then make a whole-component decision only when near-optimal assignments and exact
   constrained optima agree. This cross-disciplinary operations-research transfer is
   less discoverable than another pairwise-score threshold.
4. **Causal hypothesis.** APNCC and pairwise Trackastra scores can select an edge
   that becomes fragile when several sources compete for the same destination. If a
   proposed component edit is present across nearly every low-cost feasible
   assignment and is also favored by the exact include/exclude constrained optimum,
   it should remove competition-sensitive false positives without reopening graph
   costs. The effect is expected to be most useful on 44b6, where from-scratch
   pairwise heads failed to generalize.
5. **Null and negative control.** The null is no adjusted graph gain over the
   stronger of APNCC and same-cost C0, within an exact-rerun envelope of zero. The
   mechanism-ablated control is C0 rank-1/MAP: identical costs and feasibility but
   no K-best stability filter. APNCC is the campaign control. An identity-output C2
   is PARK, not evidence.
6. **Test unit and split.** Components are built within observation-distinct videos.
   Fit-side grouped folds are `sha256(video identity)[0:8] modulo 5`; edge rows from
   one video cannot cross folds. Held 44b6 labels cannot affect costs, K, membership
   cutoffs, margins, fallback, or arm choice.
7. **Primary endpoint.** Exact frozen-scorer vector: adjusted edge Jaccard delta,
   raw edge-Jaccard delta, node-recall delta, and TP/FP/FN, first on 44b6. Proposal
   support/AUC/precision are diagnostic only.
8. **Expected sign, minimum effect, and kill line.** Expected adjusted delta is
   positive. `<=0` on 44b6 is DEAD; `(0,+0.003)` is PARK; `>=+0.003` with raw/node
   deltas `>=0`, distinct output, and positive C2-over-C0 attribution may test 6bba.
   Only `>=+0.005` in both families opens CV57.
9. **Budget and stopping rule.** Exactly one `K=8` arm; no candidate grammar, cost
   coefficient, K, membership, margin, fallback, threshold, or graph-cost sweep.
   CPU/cache-only under the future U0 limit, maximum 4 GiB RSS, three wall-hours per
   family, zero swap growth. No GPU, Kaggle, RunPod, submission, push, or external
   effect.
10. **Outcome classification.** Grounded and locally falsifiable but dependency
    locked: **research bet / PARK**, not a scored plan and not a guess.

## Frozen transfer rule

The future active contract may fill only identities and the already-defined C0 cost
encoding; it may not change this rule:

- Enumerate the eight lowest-cost **distinct feasible** assignments of each C0
  component, ordered by `(total_cost, canonical_assignment_tuple)`.
- Components with fewer than eight feasible assignments fall back to APNCC unchanged.
- For edge `e`, let `C_include(e)` and `C_exclude(e)` be exact constrained optimal
  component costs under the unchanged C0 matrix. Define
  `tau = 100 * eps(float64) * max(1, abs(C_best))`.
- An introduced edge must occur in at least `7/8` assignments and satisfy
  `C_exclude(e) - C_include(e) > tau`.
- An APNCC edge proposed for removal must occur in at most `1/8` assignments and
  satisfy `C_include(e) - C_exclude(e) > tau`.
- Apply the edit only if **every** introduced and removed edge passes. Otherwise the
  whole component falls back to APNCC. Infeasible constrained solves also fall back.
- No score normalization, coefficient adjustment, learned threshold, adaptive K,
  family-specific rule, or held-result selection is allowed.

This is a membership/min-marginal conjunction, not an edge threshold and not a graph
cost sweep.

## Available evidence hashes (non-activating)

| Artifact | SHA-256 | Interpretation |
|---|---|---|
| `h1_cache_attestation.json` | `e7c0ca58fe5f207e3ec909a5977909cd067bcbafa34666c839e46670e0f1ce3b` | Authenticates 105 T4 cache payloads; old N1 remains DEAD |
| T4 feature report | `d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf` | 105-video cache report |
| fit105 manifest | `ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28` | Fit identities |
| old H1 graph contract | `d35e2d3dbbc93de99061d0f34c7650ac85867734f44108b28ad35af665847f48` | Historical direct-swap design; explicitly not U0/C0 |
| APNCC control contract | `cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7` | Future control identity |

Missing activation identities:

- owner decision `LOCK U0-T4CANON`;
- final U0 contract and replay hash;
- frozen C0 edge/no-link cost-source contract and rank-1/MAP output hash;
- C0 component builder and feasibility-constraint hashes.

## Forbidden-access audit

- Held-fit graph labels opened: **no**
- 44b6 graph score executed: **no**
- 6bba or CV57 accessed: **no**
- C2 implementation started: **no**
- Hyperparameter or graph-cost sweep: **no**
- GPU/external usage: **zero**

## Proposed ledger row

`C2-KBEST-PREREG-20260727 | assignment-level K-best stability | LITERATURE / ALPHA |
K=8; 7/8 introduce; 1/8 remove; exact constrained min-marginal; whole-component
APNCC fallback; no sweep | NOT SCORED | U0 owner lock and frozen C0 assignment-cost
contract absent | no prediction emitted | 0 submissions | 0 GPU-h | 0 RunPod |
PARK_DEPENDENCY | primary Murty/Lawler transfer frozen before outcomes; N1/N2/6bba/
CV57 prohibited | C2_N0_preregistration.md`

## Verdict and cascade

**PARK.** The exact C2 rule is preregistered, but N0's mandatory U0/C0 identity gate
is false. Do not advance to `N1_implement_verify.md`. A future continuation requires
the owner to record `LOCK U0-T4CANON` and a complete frozen C0 assignment-cost
contract; it must reuse this rule without alteration.

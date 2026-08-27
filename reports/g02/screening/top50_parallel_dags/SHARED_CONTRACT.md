# Shared contract — six-branch top-50 campaign

Date: 2026-07-27  
Status: foundation only; no experiment or external-compute authority

> Load `../LIVE_STATE.md`, `../campaign_narrowing_and_hypothesis_dag_v1.md`,
> `../top50_seeking_alpha_research_20260727.md`,
> `.agents/skills/seeking-alpha/SKILL.md`, and
> `.agents/skills/dag-prompts/SKILL.md` before dispatching any branch.

## Objective and null

For a candidate graph \(G\), optimize only the frozen official local score:

`S(G) = weighted_adjusted_edge_jaccard(G) + 0.1 * division_jaccard(G)`.

The public objective is `>=0.910`; the current planning target is local
`0.9255–0.9422`, explicitly soft because public transfer is sparsely calibrated.
The branch-level null is no improvement outside the exact-rerun envelope on the
frozen graph scorer. Proposal precision, AUC, utility, training loss, LNK, TRA, and
paper benchmark scores are diagnostics, never promotion gates.

## Scientific-method record required before a run

Every branch report must pin this hypothesis card before observing its gate result:

1. **Named mechanism and class:** beta control, alpha, or governance enabler.
2. **Literature basis:** primary source, exact claimed mechanism, and whether the
   source is peer-reviewed or a preprint.
3. **Crowding defense:** why a thousand competent teams would not converge on the
   exact alpha; a strong pretrained model alone is beta.
4. **Causal hypothesis:** why the intervention should move `S(G)` under G-02
   conditions and specifically address 44b6.
5. **Null and negative control:** what no effect looks like, including exact rerun
   jitter and a mechanism-ablated control.
6. **Test unit and split:** observation-distinct videos/groups; no edge-row leakage.
7. **Primary endpoint:** exact frozen-scorer vector and units.
8. **Expected sign, minimum effect, and kill line:** written before execution.
9. **Budget and stopping rule:** bounded variants, wall/RAM estimate, and permission
   boundary.
10. **Outcome classification:** plan, research bet, or guess under `seeking-alpha`.

## Common falsification order

`manifest + literature → leakage/provenance → fixed grouped fit folds → 44b6 held-fit
graph → 6bba held-fit graph → full CV57 → HUMAN-GATE → public probe`

- Any association arm with 44b6 adjusted graph delta `<=0` stops before 6bba.
- Default standalone survival is adjusted graph delta `>=+0.005` in each family,
  with raw edge score and node recall nonnegative.
- A deliberately stackable wrapper may use a preregistered `>=+0.003` floor only
  where its branch DAG says so before execution.
- CV57 is confirmatory and may not choose architecture, features, thresholds,
  branch order, or ensemble membership.
- Division arms additionally report TP/FP/FN/Jaccard and combined-score delta.
- Exact reruns and mechanism-ablated controls must share manifests and seeds with
  the active arm.

## Bounded high-throughput screening

High-throughput screening is allowed only as a **design-space falsifier**, not as an
unbounded leaderboard search:

- Pin the candidate grammar, maximum number of configurations, seeds, folds, and
  selection rule before results. Default maximum is 16 configurations per branch
  unless its DAG is stricter.
- Screen only on fit-side grouped folds. Rank by the minimum fold/family graph
  delta, with raw/node constraints; never by the best single fold or proposal proxy.
- Carry at most one fixed active configuration plus one beta/ablation control into
  the held-fit 44b6 graph gate.
- Do not revisit the same held-fit/CV57 slice to tune after a non-GO verdict. A new
  attempt requires a new named mechanism and preregistration.
- Report the full configuration matrix, including negative results; do not retain
  only the winner.

## Dependency and permission locks

- `U0-T4CANON` is a new contract. It preserves old H1-ZWRFEAT N1 as DEAD and may not
  redefine the observed M5↔T4 mismatch as benign.
- `C1` and `C2` may prepare code and manifests concurrently, but graph execution
  waits for a human-locked T4-canonical contract.
- `D1` and `R0` may complete literature, feasibility, and implementation foundations
  concurrently; objective-gate execution waits for an association arm to pass both
  families.
- `F0` may complete literature/runtime feasibility concurrently; GPU extraction
  waits for both an evidence-based unlock and explicit human approval.
- Kaggle GPU, Kaggle submission, RunPod, paid compute, pushes, and other external or
  irreversible effects always halt at a **HUMAN-GATE**, even after an analytical GO.
- No branch may edit `LEDGER.md`. It returns a proposed row; the root coordinator
  reconciles one authoritative row. Every actual result or gate change must update
  `../LIVE_STATE.md`.

## Evidence spine

- Trackastra (ECCV 2024):
  <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>
- Higher-Order Cell Tracking Transformer (2026 preprint):
  <https://arxiv.org/abs/2607.11754>
- Dense microscopy embeddings for linking (MIDL 2026):
  <https://openreview.net/pdf/2d2323e7e5b7ecc9da47e2e748e46f75876e5d04.pdf>
- Cell tracking with error prediction (Nature Methods 2025):
  <https://www.nature.com/articles/s41592-025-02845-6>
- nnPU risk (NeurIPS 2017):
  <https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html>
- Conformal risk control (ICLR 2024):
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html>
- Ranked assignments (Murty, Operations Research 1968):
  <https://doi.org/10.1287/opre.16.3.682>


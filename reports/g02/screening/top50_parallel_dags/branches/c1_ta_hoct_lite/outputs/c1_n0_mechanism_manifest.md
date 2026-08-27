# C1 N0 — mechanism and manifest result

Date: 2026-07-27  
Verdict: **GO**  
Classification: **research bet** (grounded and locally falsifiable; objective gate not
yet run)

## Hypothesis card

- **Mechanism / class:** `C1-TA-HOCT-LITE`, alpha candidate. One compact
  edge-to-edge set-attention block acts only within deterministic conflict
  components made from immutable challenger/displaced-edge endpoint incidence. A
  learned no-link token competes with every proposal. The Trackastra representation
  and `C0-T4TA-LINEAR` are beta controls.
- **Primary literature:** Trackastra is peer-reviewed at ECCV 2024. It predicts
  pairwise associations from full spatiotemporal detection context and introduces a
  quiet parental softmax whose constant-denominator alternative permits no parent:
  <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>.
  HOCT is a 2026 **preprint**, not peer-reviewed. Its stated mechanism is that
  candidate edges sharing nodes have near-random label agreement, making
  node-topology GNN aggregation unhelpful; its proposed remedy is edge-centric
  attention under a geometric prior:
  <https://arxiv.org/abs/2607.11754>.
- **Crowding defense:** using frozen Trackastra or a linear head is obvious beta.
  The differentiating claim is the competition-specific distillation of HOCT's
  edge-interaction mechanism onto authenticated APNCC swap-conflict sets, without
  full HOCT, image-backbone training, or graph-topology propagation.
- **Causal hypothesis:** APNCC and C0 score each swap independently, although their
  graph effects compete through shared endpoints. Set attention can represent
  conditional preference/abstention among those swaps and should therefore reduce
  retained conflict false positives after the frozen solve, especially on 44b6.
- **Null / negative control:** exact identical reconstructions define
  `epsilon_null`; APNCC, C0, and `C1-ABL-NOATTN` share rows/folds. No effect means
  adjusted graph movement within the exact-rerun envelope. A 44b6 adjusted delta
  `<=0` is DEAD.
- **Test unit / split:** video identity, not edge row. Stable five-fold assignment is
  `sha256(identity)[0:8] mod 5`; calibration is `(held+1) mod 5`.
- **Primary endpoint:** exact frozen adjusted edge-Jaccard delta. Promotion later
  requires `>=+0.005` per family, raw edge-Jaccard and node recall nonnegative, and
  44b6 C1-minus-C0 `>=max(0.001, epsilon_null)`.
- **Budget / stop:** one fixed 20,097-parameter CPU head, one seed, no architecture
  search, graph-cost sweep, GPU, external action, or proposal-metric promotion.

## Identity and feasibility audit

The machine-readable manifest
[`c1_n0_mechanism_manifest.json`](c1_n0_mechanism_manifest.json) authenticates every
feature payload and its corresponding fit-label payload.

| check | result |
|---|---:|
| authenticated fit videos | 105: 36 `44b6`, 69 `6bba` |
| frozen proposals / conflict rows | 195,354 / 280,095 |
| deterministic conflict components | 160,490 |
| largest component | 17 proposals |
| stable fold populations | 19 / 21 / 24 / 18 / 23 |
| exactly joined labeled proposals | 1,217 |
| joined positives / negatives | 1,186 / 31 |
| trainable parameters | 20,097 |
| dry prediction repeats | 2, byte-identical |
| graph scorer / validation12 / CV57 | not called / not accessed / not accessed |

Five videos have zero exact overlap with the sparse prepared challenger-label table.
This is not missing inference coverage: all five retain authenticated full-pool
features and conflict keys. They must be inference-only within their held fold; they
may never be silently labeled negative. N2 must prove the C0/C1 loader handles this
fail-closed and predicts every row. The sparse 31-negative training support is a
material risk and is why C1 remains a research bet rather than a plan.

## Disqualifier audit

PASS: frozen pretrained features; set attention rather than scalar calibration;
explicit no-link; no GNN or topology messages; no communication across components; no
division decoder; no threshold/cost search; no family-specific held tuning; no
proposal-only promotion; fit-only identities. The old H1 N1 verdict remains DEAD and
is neither relabeled nor weakened.

## Verdict rationale

**GO.** All inference rows, component keys, folds, feature payloads, and fit-label
payloads have reproducible identity. C0 is mathematically replayable on the same rows:
fit/calibrate only on exact joined labels, treat zero-label videos as inference-only,
and score all proposals. The mechanism is materially different from the DEAD
scalar/from-scratch classes. N1 may freeze the single design; no graph gate is opened.

## Proposed LEDGER row

`C1-TA-HOCT-LITE-N0 | LITERATURE/MANIFEST | PASS: 105 videos, 195354 proposals,
160490 components | f1227f3d…770e | 0 external usage | GO | grounded conflict-set
alpha; sparse negatives explicitly carried as risk`

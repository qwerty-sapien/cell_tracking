# P4 edge-centric attention — exact CV57 result

Status: **DEAD** under the revised gate.

The prototype used the frozen fitted pairwise edge logits, cap-8 candidates,
and the exact 105/12 fit-only split. Both arms received the same 504 strict,
label-safe confusable source decisions. The attention arm differed only by one
edge-self-attention layer. Attention was allowed between edges sharing a source
or target, or whose physical midpoints were within 15 um. No GNN was built.

| Arm | Adjusted edge Jaccard | Edge TP / FP / FN | Delta vs 0.873149594 |
|---|---:|---:|---:|
| Banked backward motion, c=0.25 | 0.873149594 | 34,205 / 2,685 / 2,063 | — |
| Identical model, no attention | 0.874290046 | 34,227 / 2,658 / 2,041 | +0.001140452 |
| One-layer edge attention | 0.874006533 | 34,225 / 2,670 / 2,043 | +0.000856939 |

The causal attention delta is **-0.000283513** versus the identical no-layer
model. The primary delta is below +0.002, and the attention layer itself is
harmful. Therefore P4 is dead as specified; do not proceed to the 6–9 day build.

The exact patched scorer used all 57 held-out videos, divisions remained off,
and no Kaggle action occurred. Actual GPU seconds per frame pair are not yet
reported: the local machine has no usable CUDA/MPS device, and paid external GPU
execution requires explicit approval. CPU timing is not substituted for that
measurement.

Machine-readable records:

- `p4_edge_attention_prototype_v1/prepare.json`
- `p4_edge_attention_prototype_v1/fit.json`
- `p4_edge_attention_cv57_v1/report.json`

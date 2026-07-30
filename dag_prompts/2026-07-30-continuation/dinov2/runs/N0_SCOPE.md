# N0 — scope sieve: DPRC-O0

Date: 2026-07-30 (Asia/Singapore)
Verdict: **GO**

## Objective and non-negotiables

The only downstream success measure remains the frozen 44b6 graph-level delta,
first for delete and occupied-swap; no proposal-only result can promote this
candidate. The R5 local, CPU, and GPU routes are terminal and are excluded,
including their checkpoint, package, request payloads, routes, and quarantined
partial rows. No model, data, asset, or compute was opened at N0.

## Selected mechanism — `DPRC-O0`

**DINOv2-register patch correspondence (DPRC).** A new, independently pinned
DINOv2 ViT-S/14-with-registers supplier would produce its full spatial patch grid
for a *freshly materialized* cell-centred crop. A public segmentation mask is
rasterized to that grid; the candidate descriptor is the foreground-only,
mutual-local-nearest-neighbour patch-cycle coverage and similarity, with no
fine-tuning. The descriptor is passed only to the already-required frozen,
fit-only evaluation protocol if a later human authorizes it.

The causal claim is deliberately narrow: a wrong incumbent link that is globally
similar but locally inconsistent receives lower cycle coverage and can be ranked
for **delete**; in an occupied conflict, a locally consistent alternative can
receive a higher value than the incumbent and supply **swap** discrimination. It
does not change graph costs, detection, a threshold, or the scorer.

The fixture observable is `C(i,j)`, the foreground-grid mean whose matched
terms are cosine similarities only when a source patch's best target match is
within one 4-by-4 grid cell and maps back; failed cycles contribute zero. A
same-mean spatial permutation must tie under global pooling yet have lower
`C`; uniform tokens must produce zero
incumbent-versus-alternative margin rather than a spurious preference.

## Exact distinction from terminal R5

R5 was `F0-DINOV2-VITS14-XY-V1`: the terminal 88,283,115-byte, non-register
checkpoint and a 768-D CLS-plus-central-2x2 endpoint embedding, transported via
its terminal 77-request package. DPRC instead requires a not-yet-obtained,
separately hash-pinned **register** checkpoint, the full foreground spatial grid,
and fresh manifest/input materialization. It must not read R5 feature rows,
checkpoint `b938bf1b…0cd9`, source archive, request files, input bundle, or
quarantined shards. It is therefore a new named mechanism, not an execution
repair, packaging change, or reinterpretation of an R5 artifact.

## Two novelty generators, three candidates

| Generator | Candidate | Sieve result |
|---|---|---|
| Cross-disciplinary transfer: point correspondence | **DPRC-O0** — foreground patch-cycle coverage from localized DINO features | **Selected.** It changes both the supplier checkpoint and the information used; it has the objective-linked `C(i,j)` / conflict-margin observable. |
| Literature-edge mining: long-video test-time adaptation | DINO-Tracker-style per-video Delta-DINO refinement | Rejected: it requires per-video training and an unbounded adaptation route; it is not the frozen, cheap-to-kill mechanism required here. |
| Constraint inversion: use every R5 patch token instead of its pooled endpoint vector | All-token matching with the existing R5 ViT-S/14 checkpoint/package | Rejected: despite a different reducer, it reuses terminal R5 information and is prohibited by the no-retry rule. |

Crowding assessment: “try DINO globally” is beta and terminal. The defensible
alpha node is instead the combination of a *new* register supplier, foreground
masking, and a falsifiable local correspondence contract. It is still only a
candidate: no biological effect has been claimed.

## Proposed ledger row — not written

`DPRC-O0 | DINOv2-register foreground patch-cycle correspondence | N0 GO;
literature/falsifier pending | new named supplier; R5 assets/rows excluded |`

**Cascade:** GO to N1.

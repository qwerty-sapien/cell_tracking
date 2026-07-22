# CELLECT Excerpt from the Alternative Tracking Research Ledger

Preserved from `docs/alternative_tracking_approaches_research.md` when CELLECT was
paused on 2026-07-23. The broader multi-method ledger remains active; this
CELLECT-specific recommendation is archived and must not be acted on unless the
CELLECT direction is explicitly resumed.

## Original method-matrix entry

CELLECT uses contrastive embedding learning for large-scale efficient 3D cell
tracking, sparse annotations, and centre/feature/division maps. It was assessed as
scientifically high-fit but medium-to-low budget fit, and as a possible long-term
target if pretrained weights transferred. Training from scratch was not
recommended without Kaggle-free feasibility evidence.

Sources:

- <https://www.nature.com/articles/s41592-025-02886-x>
- <https://github.com/zzz333za/CELLECT>

## Original source note

CELLECT was considered relevant because it combines sparse annotations,
contrastive embeddings, centre prediction, division prediction, and lightweight
MLPs for intraframe duplicate removal and interframe link/division classification.
It reported strong Cell Tracking Challenge performance on 3D *C. elegans* data and
real-time-scale 3D tracking on a GTX 3090. Reproduction or adaptation under
Kaggle-free compute was identified as nontrivial, though its architecture was
initially considered a useful blueprint.

That recommendation is now outranked by the evidence and resume conditions in
`../PAUSE_RATIONALE.md`.

# MAIN N1 result — duration-aware phase hazard literature audit

**Verdict: GO.** The literature grounds the mechanism; this is not authorization to
fit, score, or use external compute.

## Candidate

`AE-P7-DURATION-HAZARD`: use a biologically constrained distribution of mitotic-phase
durations as a local incompatibility signal for a proposed association. It differs from
the frozen static phase-template contract by requiring a duration-aware, uncertainty
calibrated transition likelihood—not a fixed phase label or a scalar threshold.

## Primary support and transfer argument

1. Held et al. developed an automated live-cell method that classifies seven mitotic
   phases over time and measures phase length, demonstrating that phase duration is a
   usable temporal observable rather than only an endpoint label. [PubMed: 19797680](https://pubmed.ncbi.nlm.nih.gov/19797680/)
2. McDole et al. reconstruct whole-embryo lineages with a tracking graph that incorporates
   biological knowledge about track length and cell cycle, and report recovery from noisy
   detection/segmentation through spatiotemporal context. [Nature Biotechnology / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/)

The structural correspondence is narrow: both works use temporal biological constraints
to distinguish otherwise plausible associations. Neither is evidence that the proposed
hazard improves this frozen score; that is why the next gate must be a disconfirming
local test. The crowding rationale is that this uses calibrated transition-duration
compatibility at conflicted associations, not the ordinary fixed radius/threshold or a
generic foundation-model feature.

## Counterevidence / limits

- The 2009 study is a different assay and depends on image features sufficient for phase
  categorization; this project has not established such features.
- McDole et al. validates a global lineage method, not this action compiler or 44b6.
- Therefore no effect-size or Kaggle request can be inferred from these sources.

## Proposed LEDGER row (not written)

`AE-P7-DURATION-HAZARD | literature-grounded research hypothesis | 2 primary sources | GO to cheap-falsifier design`

# NODETOPO N1 result — local neighborhood conservation audit

**Verdict: GO.** Literature supports evaluating a local spatial/topological observable;
it does not establish a score gain or authorize a data run.

## Candidate

`NODETOPO-NEIGHBOR-CONSERVATION`: for an already-existing association candidate, retain
only a compact local neighborhood signature across adjacent frames. It neither adds
nodes nor regenerates raw candidates; its intended purpose is to discriminate a local
identity switch from a topology-preserving continuation.

## Primary support

1. Makrogiannis et al. explicitly use an adjacency graph to extract features from
   time-lapse images and then apply robust graph-based tracking with multiple cell
   features, supporting local spatial relations as an association signal. [PubMed: 36859457](https://pubmed.ncbi.nlm.nih.gov/36859457/)
2. Kumar et al.'s Lineage Mapper uses overlap and biologically derived local image
   properties to resolve tracking, mitosis, and touching-cell errors, including cell
   geometry at ambiguous events. [PMC: 5113068](https://pmc.ncbi.nlm.nih.gov/articles/PMC5113068/)

The mechanism is a restrained transfer: both sources show that local spatial/morphology
relationships can constrain association under contact/mitosis ambiguity. Neither source
claims that neighbor conservation alone solves the terminal 44b6 restoration cases, so
the effect remains unmeasured.

## Rejects and limits

- A generic global graph repair is not accepted: that aliases terminal NODETOPO work.
- Neighbor signatures can be unstable at division or segmentation failure; the null
  fixture must include both, rather than treating smoothness as proof.
- No current source demonstrates this repository's exact candidate-topology contract.

## Proposed LEDGER row (not written)

`NODETOPO-NEIGHBOR-CONSERVATION | literature-grounded research hypothesis | 2 primary sources | GO to synthetic falsifier`

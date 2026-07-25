# Selected-graph conflict gate v1

Status: **PARK before score; support gate failed**

This CPU-only bet repaired the previous gate's supervision mismatch by defining
every comparison from the actual frozen `c=0.25` legacy graph selection.

## Fit-only evidence

The exact 105/12 fit partition produced 8,954 labeled comparisons:

- 5,537 rescue rows from reachable one-to-one GT continuations blocked at an
  occupied selected source or target;
- 3,417 controls where the selected graph edge was GT-correct and the challenger
  was a fully matched non-edge;
- 143 division truths excluded;
- 535 reachable misses without a candidate-backed selected conflict excluded.

The fitted proposal-level threshold was 0.82.  On the 12 fit-validation videos it
accepted 172/312 proposals: 165 correct rescues, 7 harmful swaps, 95.93% precision,
and 66.53% rescue recall.

## Deployment-support failure

Before a scored CV57 replay, the same frozen model was applied to the complete
unlabeled candidate population of all 12 fit-validation videos.  A sparse
deployment budget was declared as:

`max(16, ceil(0.005 * selected_raw_edges))` proposals per video.

No threshold simultaneously retained at least 10 labeled proposals at >=95%
precision and kept every validation video within its sparse budget.

| threshold | labeled accepted | precision | worst budget ratio |
|---:|---:|---:|---:|
| 0.92 | 109 | 98.17% | 17.09x |
| 0.965 | 48 | 95.83% | 8.63x |
| 0.975 | 24 | 100% | 4.50x |
| 0.98 | 15 | 100% | 2.54x |

At 0.995 the support budget passes only because zero labeled proposals survive.
Therefore thresholding cannot repair the support mismatch.

One frozen CV57 video had begun replay before this full support audit was added.
It emitted 3,051 substitutions and the run was terminated without producing a
score.  That partial run is audit-exposed and inadmissible; it was not used to
select a new threshold.

## Decision

**PARK.** No CV57 score, full-confirm run, GPU action, package, or submission is
authorized.

The finding is stronger than a weak score: a classifier trained only on
GT-observable conflicts is overconfident over the much larger unlabeled
deployment population.  A successor must train an explicit low-weight
abstention prior on deterministically sampled unlabeled selected-graph
challengers and pass this support gate before touching CV57.

Artifacts:

- `reports/g02/selected_graph_conflict_prepare_v1/prepare.json`
- `reports/g02/selected_graph_conflict_gate_v1/report.json`
- `reports/g02/selected_graph_conflict_support_v1.json`

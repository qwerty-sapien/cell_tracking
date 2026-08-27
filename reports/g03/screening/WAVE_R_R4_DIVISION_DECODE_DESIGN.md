# G-03 Wave R / R4 — post-hoc division decode sweep design

Status: **design only; no target, score, or sealed-confirmation access**.  This
document fixes a finite search and a nested grouped evaluation procedure before any
G-03 division result is read.  It does not reuse the failed G-02 A/B/C realizers as
predictions.  It reuses their compiler, scorer, equivalence grammar, and measured
false-positive risk as inputs.

## Objective and reference points

The decision quantity is the exact frozen combined graph score after applying a
division decoder to one already-fixed association output.  Every wave report must
show all of:

- final combined score versus the campaign decision reference
  `best_known_score = 0.881522069960`;
- policy floor `0.881522069960 * 0.985 = 0.868299238910600`;
- attribution-only delta versus the immutable no-division parent graph, identified
  by its graph SHA and absolute score; and
- adjusted edge Jaccard, raw edge Jaccard, node recall, and division TP/FP/FN/Jaccard
  separately.  The division term is never folded into an unlabeled “edge gain.”

The first two reference values are cross-population campaign policy references if the
evaluated slice is held-fit 44b6 rather than CV57; that limitation must be printed, not
silently treated as a calibrated transfer.  Only scores from the same frozen population
may be used for causal attribution to the decoder.

The banked G-02 sensitivity is a risk prior, not a G-03 performance claim: for `TP=25`,
the division contribution is `+0.059523810`, `+0.053191489`, and `+0.048076923` at
`FP=0`, `5`, and `10`.  False forks are therefore the primary tie-break and stopping
risk.

## Frozen candidate grammar

Starting from the fixed association tensor/bank, enumerate every source with at least
two forward-time candidate descendants.  A candidate event may be:

1. `DIRECT`: one parent receives two children in the next frame; or
2. `LATE(k)`: the parent continues linearly for `k` frames, then the continuation
   receives two children, for `1 <= k <= late_window`.

Direct and late events that represent the same predicted parent lineage and unordered
daughter lineages share one label-free equivalence key.  Grandchild evidence may
validate a proposal but is never invented as an emitted edge.  Every accepted event is
compiled as one atomic N1 fork action, including all incident-conflict deletions.

Forbidden inputs to candidate generation and ranking are GT IDs, official matching,
repair-bundle membership, division labels, scorer rows, prior arm outcomes, CV57, and
the sealed confirmation set.  The 38 G-02 bundles may be opened only by the evaluator
after predictions for the relevant historical diagnostic have been sealed; they are
never a feature or a way to construct a G-03 candidate.

## Complete finite hyperparameter lattice

All configurations use a single family-shared setting; family-specific splicing is
forbidden.  The exact lattice is:

| axis | values | interpretation |
|---|---|---|
| `fork_threshold` | `0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95` | minimum second-child association probability |
| `max_primary_secondary_gap` | `0.05, 0.10, 0.20, 1.00` | maximum score gap between the two child edges; `1.00` disables this filter |
| `late_window` | `0, 1, 2` frames | direct only through at most two-frame-late equivalence |
| `daughter_persistence` | `0, 1, 2, 3` frames | minimum survival of each distinct daughter branch |
| `grandchild_persistence` | `0, 1, 2` frames | minimum post-daughter continuation depth on both branches |
| `duplicate_temporal_window` | `0, 1, 2` frames | suppress lower-ranked equivalent forks on the same predicted parent lineage within this radius |
| `per_video_cap` | `1, 2, 3, 5, 8, UNLIMITED` | maximum surviving events after suppression |

This is `7 * 4 * 3 * 4 * 3 * 3 * 6 = 18,144` active configurations plus one
`DIVISIONS_OFF` null.  There is no adaptive expansion, threshold interpolation, random
search, Bayesian optimization, or manual neighborhood search after an outcome.  A
configuration manifest lists all 18,145 names and its SHA is sealed before evaluation.

The following operations are invariants rather than tunable axes:

- candidates are ordered by descending second-child score, ascending primary-secondary
  gap, `DIRECT < LATE`, parent ID, child IDs, persistence-node IDs, then config name;
- exact equivalence-key suppression occurs before node ownership;
- parent, child, continuation, and persistence-evidence nodes have single-event
  ownership; rejected collisions record winner and reason;
- both emitted daughters must be distinct, forward-time, present in the base node set,
  and yield indegree `<=1`, outdegree `<=2`, no merge, and an acyclic graph after N1;
- `duplicate_survivors` must equal zero; and
- score computations deduplicate byte-identical graph hashes so the large lattice does
  not create repeated scorer reads.

## Leakage-proof nested/grouped-OOF protocol

### Population split

1. Keep `reports/g02/grouped_3fold_audit20_v1.json` and its 20-video audit partition
   immutable and untouched.  It is the one-shot confirmation population.
2. On the remaining fit-only population, construct five deterministic outer folds by
   embryo/group ID, stratified by family and GT division-count bin.  No embryo/group may
   cross folds.  Seal all identities and the split SHA before reading any candidate
   outcome.
3. Within each outer training partition, use the other four outer folds as inner grouped
   folds.  Every inner score is produced on videos not used to choose or fit any
   calibration for that inner prediction.

### Selection procedure

For each outer fold independently:

1. materialize candidate banks without labels;
2. run all 18,145 fixed configurations on each inner held fold;
3. select exactly one configuration by the pooled inner-OOF **absolute combined graph
   score**, then higher division Jaccard, lower division FP, higher adjusted edge
   Jaccard, higher raw edge Jaccard, higher node recall, fewer predicted forks, and
   lexicographic config name;
4. freeze that winner before evaluating the untouched outer fold once; and
5. append the outer predictions to one pooled nested-OOF graph bank.

No outer result may change an earlier/later outer winner.  The nested-OOF result estimates
the complete selection procedure, not the best retrospectively pooled hyperparameter.
The final deployable setting is fixed without audit access as the modal outer winner;
ties use the same preregistered metric ordering computed only from inner-OOF rows.

### Sealed one-shot confirmation

After all five outer folds, candidate predictions, config winners, hashes, scorer source
hashes, and the deployable setting are immutable, an access guard permits exactly one
score of the 20-video audit set.  The audit runner refuses if any prior audit-output file,
access marker, or scorer receipt exists, writes an access marker before opening targets,
and emits hashes even on failure.  There is no second configuration, repaired threshold,
or retry after a scientifically valid score.  Infrastructure failure before target access
may be repaired only by a new hash-pinned transport amendment that leaves all scientific
bytes unchanged.

## Gates and stopping rules

Nested-OOF promotion to sealed confirmation requires all of the following:

- exact final score is strictly greater than the explicit campaign reference
  `0.881522069960` on the declared comparison population, or the report labels the
  cross-population policy comparison as non-calibrated and uses the same-population
  frozen best as the binding score reference;
- division Jaccard `>=0.30`, duplicate survivors `=0`, and positive division TP;
- adjusted edge, raw edge, and node terms do not regress versus the explicitly named
  parent graph on the same population;
- every outer fold has at least one GT division and no fold has a negative combined
  attribution delta versus its named parent graph; and
- all split, target-access, compiler, scorer, graph validity, and hash guards pass.

The sealed audit confirms only if it satisfies the same structural/non-regression rules
and its final score is strictly greater than its frozen same-population best reference.
Otherwise the decoder is PARK-C for this association checkpoint.  A failure does not
make the general division mechanism DEAD and cannot be repaired by further audit search.

## Execution schedule and resource accounting

R4 can begin candidate-bank extraction as soon as R1 (or a later fixed training
checkpoint) publishes an immutable association-output SHA.  Candidate extraction,
graph compilation, unique-hash collapse, and nested scoring are CPU-only and may run in
parallel with the single GPU training lane.  The sweep records CPU wall time, peak RSS,
configuration count, unique graph count, and scorer invocations.  It consumes `0 T4-h`
and cannot acquire the Kaggle GPU serial lock.

Historical diagnostics carried forward with explicit scope:

- G-02 full-38 oracle ceiling: division `TP/FP/FN=38/0/4`, Jaccard
  `0.904761905`, adjusted/raw/node deltas
  `+0.001591651/+0.001607525/0`, combined `+0.092067842` **[GT-dependent ceiling]**;
- banked `TP=25` false-fork curve above **[risk surface, not achievability]**; and
- G-02 target-free realizers A/B/C recovered `0/38` and are PARK-C at those exact
  instantiations.  G-03 does not relabel those negatives as positive evidence.

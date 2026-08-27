# VELOCITY-AWARE-CANDIDATE-WINDOW-v1

## Scope

This is a new named *per-detection candidate recentering* diagnostic on top of
the completed whole-frame global-jump (G1) corrected APNCC control. It is not
a radius widening, a re-screen of `R12SHELL` / `MGSHELL` / `TEMPBG`, or a
cost/motion retune. It leaves the radius at 10 um, the cap at 8, the APNCC
edge scorer, and graph machinery untouched during the ceiling audit.

The named GA-R12REACH cohort is the sealed fit-only117 shell bank at
`../../../g02/extended_radius_shell_prepare_v1/examples/`, with its exact cache
at `../../../g02/fitcache_117/bh9out/cache/`. Its canonical `456` original-shell
positives and G1 correction arithmetic are defined by
`scripts/g04_score_pathology_repairs.py:shell_crosscheck()`: the residual is
the `366` records whose post-G1 distance remains over 10 um after the `90`
rescues. The G1 score report records adjusted/raw/node
`0.886764420260 / 0.891461012028 / 0.990377056057`. Its CV57 BHC-pregraph
reachability count (`25,553`) is retained only as a separately labelled
transfer/background ceiling—not as an endpoint-identical GA topology.

## Candidate-coordinate convention

Let `a[i,t]` be the original voxel Z/Y/X detection coordinate and `c[t]` the
G1 cumulative raw-voxel whole-frame shift. Candidate geometry is evaluated in
physical microns as

`q[i,t] = (a[i,t] + c[t]) * (1.625, 0.40625, 0.40625)`.

`a` remains the evaluator/output coordinate. No output coordinate is changed,
no image or feature sample is translated, and neither `c` nor the optional
storage-only common offset is applied twice.

For a source detection `i` at frame `t` with an immediately preceding,
one-frame G1-selected parent `h` at `t-1`, form

`v[i,t] = q[i,t] - q[h,t-1]`

and recenter only its candidate query at

`qhat[i,t+1] = q[i,t] + 0.30 * v[i,t]`.

The `0.30` is the published lag-1 directional-persistence prior; the published
median step `1.86 um/frame` is recorded as a physical-sanity diagnostic, not
silently substituted for a direction. A source without such a parent falls
back exactly to the G1 centre, `qhat=q[i,t]`.

Candidates are the deterministic nearest eight target detections `j` satisfying
`||q[j,t+1] - qhat[i,t+1]|| <= 10 um`, ordered by `(distance-to-qhat,
target-node-id, target-row-index)`. Local density is separately counted in its
same-frame physical `r=10 um` ball, excluding the point itself.

## Pre-scorer ceiling gate

The audit first measures the entire G1 control candidate pool and the proposed
window pool, and labels GT only to establish the reachability ceiling. Let `N`
be the number of held-CV57 true continuations absent from the G1 candidate set
but present in the proposed set. The no-new-FP raw edge ceiling is

`N / (34389 + 2308 + 1879)`.

The exact scorer may run only if all three conditions hold:

1. `N >= 20` (raw no-new-FP ceiling at least `0.0005`),
2. the proposed pool's all-nonpositive candidate burden is at most 5% above
   G1, and
3. at least 20 of the 366 residual `(10,12] um` shell GT continuations are
   retained after the fixed cap-eight rule.

This is deliberately only an eligibility ceiling: it neither forecasts a
score gain nor authorizes a scorer/graph change. Failure means `NO_SCORE`.

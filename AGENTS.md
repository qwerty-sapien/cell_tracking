# Agent instructions

## Current state entry point

Read `reports/CURRENT.md` first when it exists. It is the only routine
cross-campaign entry point and must stay below 4 KB.

If it is absent, identify the active campaign from the exact task and the
newest relevant campaign directory; read only that campaign's compact state,
current frontier, protected gates, key numbers, and exact artifact pointers.
Do not scan historical campaigns or a whole ledger merely to manufacture a
root summary. Create `reports/CURRENT.md` only when campaign coordination is
itself in scope.

`reports/g02/screening/LIVE_STATE.md` is archived and larger than 100 KB. Read
it only when the task names G-02 or needs an exact historical G-02 decision,
and then by heading or bounded line range. Do not re-screen a G-02 DEAD
mechanism without a new named mechanism. For G-02 scientific work, the
human-locked plan remains
`reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`; open only
the relevant heading.

`reports/g03/screening/LIVE_STATE.md` is the compact G-03 state in the inspected
snapshot. Read it only for a G-03 task. The root coordinator alone updates the
applicable compact state and campaign ledger after a decision, launch, or
terminal result; detailed evidence belongs in node reports.

## Default exclusions

Do not inspect these by default:

```text
assets/
codex_history/
old DAG runs
archived reports
predictions
reverse caches
feature caches
remote outputs
logs
*.npz
*.npy
*.pt
*.pth
*.ckpt
*.json files above 64 KB
images and videos
```

Use file metadata, hashes, schemas, row counts, selected keys, and bounded
samples through helper scripts. Never paste binary or large generated content
into model context.

## Source-first navigation

1. Read the exact task and the smallest relevant current-state entry.
2. Inspect `git status --short`.
3. Search the relevant symbol under `src/`, `scripts/`, or the named campaign
   directory.
4. Open one bounded source window and the nearest test.
5. Check the scorer, data split, or state contract only when the change can
   affect it.
6. Stop discovery once the owner, invariant, edit surface, and focused test are
   known.

Do not scan all scripts, tests, DAG prompts, or reports. Expand context only
when a caller, invariant, or contradictory test remains unresolved.

## Scientific and competition invariants

Use at least L5 for scorer or metric semantics; CV, holdout, final-population,
or leakage boundaries; hypothesis, preregistration, stopping, or statistical
claims; biological grouping or independence; model-selection decisions;
Kaggle packaging, launch, submission, monitoring, or quota; and external
compute or paid resources.

Keep descendants of an independent group in one split. Never tune on a
protected holdout. Distinguish exploratory evidence from confirmatory evidence.
Preserve immutable package, scorer, and launch hashes.

`LMT.md` is the sole current authority for autonomous scope and resource
limits; older numeric limits in plans, prompts, skills, and receipts are
historical only. For an external or compute action, read only its precedence,
machine-policy, and relevant action heading. Do not copy LMT numbers into
another active policy.

Standing Kaggle authorization applies only when the productive-candidate,
locked-plan, integrity, serial-lock, quota, exact-version, and monitoring gates
pass. Quota expiry alone is not productivity. RunPod requires explicit human
approval unless a newer repository policy says otherwise; do not shard work to
evade an approval boundary.

## Continuous research execution

The root coordinator continuously advances the active research frontier. After
receiving a terminal result or identifying a causally justified successor, start
the next authorized step without waiting for another user prompt. Pause only at
a compulsory human gate defined by current policy, a protected-outcome boundary,
an unsafe or destructive action, or a genuinely missing scientific choice that
would change the identity. The paid-compute threshold and anti-sharding rule live
only in `LMT.md`; do not copy their numeric values into plans or receipts.

Maintain several non-overlapping frontier queues so one model family cannot
monopolize progress: target evidence, non-M/dormant mechanisms, submission
readiness, and portfolio control. Parallelism is bounded by available agents and
resource locks, not by the number of historical nodes. When a slot frees, take
the highest-information eligible item from the queues.

Every terminal result must classify the strongest supported cause as transport,
runtime/environment, data-contract, scientific/mechanistic, or unresolved. A
failed provider run is never scientific evidence. Preserve invalidations and
successes in a compact dynamic hypothesis graph, state which assumption changed,
and choose the cheapest test that distinguishes the leading remaining causes.
Use fresh immutable identities for repaired executions and never retry an
existing remote version. Prefer fail-fast boundary tests and source/runtime
closure before material runs.

The elapsed time since the last competition submission raises the priority of
the fastest defensible submission path, but never relaxes leakage, score,
holdout, legality, package, or submission gates. A completed kernel is not by
itself a submission candidate.

For result retrieval, prefer the authenticated Kaggle CLI. If browser Computer
Use is necessary, make one bounded direct download attempt. If the result ZIP
does not materialize, stop and request the user-provided file; do not spend
tokens on retry cascades or download subagent loops.

## Model routing

- L1: deterministic extraction, headings, counts, formatting.
- L2-L3: local utilities and clearly bounded tests.
- L4: ordinary cross-module pipeline work without scientific-semantic changes.
- L5+: scientific semantics, scorer, holdout, competition, external compute,
  or fragile pipeline state.

## Verification

Start with the exact test:

```sh
python -m pytest -q path/to/test_file.py::test_name
```

Then run the affected test module or package. Run the full configured pytest
suite only at a stable final diff or when shared scorer, split, packaging, or
campaign-control code changed.

Never launch remote compute as a verification substitute, and never claim a
scientific or competition gate passed from unit-test evidence alone.

# Division-ckpt N0 — does an admissible frozen external checkpoint exist?

> Load `DAG.md`, `../../2026-07-30-continuation/division/runs/N0_SCOPE.md`,
> `N1_LITERATURE.md`, `N2_FALSIFIER.md`, `N3_THREE_STAGE_WEEKLY_PLAN.md`,
> `reports/g02/screening/LIVE_STATE.md`, and `B1_division_tp.md`.
> CPU only, 0 GPU-h. Authenticated retrieval goes through **computer-use cmux**, never the
> agent-shell token. Hash-only fetches into an ignored temporary directory.

## Objective / gate

Is there a **publicly released, frozen** division-relevant temporal representation
checkpoint that is source-pinned, weight-pinned, license-pinned, provably not trained on
any competition movie, and runnable inference-only inside a single T4 at 16 GiB?

As of 2026-07-30 the answer is recorded as no. This node either changes that with exact
identities or terminates the pilot cheaply.

## Tasks

1. Enumerate the candidate released checkpoints for time-arrow / temporal-asymmetry /
   mitosis-symmetry representations, starting from the `tarrow` source and its published
   artifacts, and including any independently released weights from the cited primary
   literature. Record every candidate examined, including the rejected ones.
2. For each candidate, authenticate: repository revision, weight URL, exact byte size,
   SHA-256, license file and its hash, and the constructor/config needed to load it. A
   candidate without all of these is not a candidate.
3. **Provenance exclusion.** Establish from the source's own documentation what data the
   weight was trained on, and show it excludes every competition movie. An unstated or
   unverifiable training corpus is a fail, not an assumption.
4. **Rights.** Record the license terms for research use and for derived embeddings, plus
   any retention or redistribution constraint. Do not download campaign-adjacent data.
5. **T4 inference feasibility.** From the source's published configuration, state
   parameter count, dtype, and the memory required for inference only. The upstream default
   assumes `>=24 GiB` for *training*; show the inference path fits 16 GiB without changing
   the architecture. A batch-size reduction for inference is permitted and must be recorded;
   any architecture or backbone substitution is a different mechanism and fails this node.
6. Confirm no artifact, fragment, request, row, or launch lineage from any terminal route
   is reused — specifically not `TAP-SYM-DIV`'s escalated design, not R5, not CELLECT.
7. Do not download campaign data, create a Kaggle asset, load a model onto a GPU, or run
   inference.

## Output

`runs/N0_CHECKPOINT_PROVENANCE.md` — the candidate table with exact identities, the
provenance-exclusion argument, the rights record, the inference-memory derivation, and a
proposed (not written) LEDGER row.

## Verdict (preregistered)

- **GO**: `>=1` candidate has all of — pinned source revision, pinned weight bytes +
  SHA-256, pinned license permitting research use, documented training corpus excluding
  every competition movie, and a source-published inference configuration fitting 16 GiB.
- **PARK**: a checkpoint exists but rights, corpus documentation, or 16 GiB inference
  feasibility cannot be established. Do not improvise a substitute; report and halt. The
  fallback is the separate `~56 T4-h / 4-week` training budget decision, which is an owner
  budget change, not a repair of this node.
- **DEAD**: no released checkpoint exists for this mechanism class, or every candidate is
  trained on competition data. The frozen-external-representation route is dead; training
  from scratch remains a separate named option.
- **ESCALATE**: an admissible checkpoint exists but only under terms requiring a plan or
  governance change.

## Cascade

- On **GO** → proceed directly to `N1_frozen_inference_falsifier.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.

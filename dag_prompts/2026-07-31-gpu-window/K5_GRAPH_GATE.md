# K5 — 44b6 held-fit graph gate (zero GPU)

**Precondition: `K4` returned PASS on every preregistered clause.** If `K4` failed, this
node does not open — do not "look anyway".

## Scope

`K4` already computes the frozen-scorer graph deltas as its selection metric, so this node
is **verification and adjudication**, not a new measurement. Its job is to make the result
survive an adversarial read.

1. **Independent replay.** A separate evaluator, not the fitting code, reproduces the
   adjusted-edge, raw-edge, and node-recall deltas from the emitted edit sets. The campaign
   has done this before (NODETOPO: a separate evaluator independently reproduced exact
   scores, deltas, and verdict). Match to `1e-12`.
2. **Provenance closure.** Bind the archive SHA-256, the preregistered central directory,
   the evaluator, the wrapper, and the imported scorer. Prove the evidence chain is
   self-contained **at a clean checkout** — a prior independent review caught exactly this
   gap, where a root evidence chain referenced an unmerged artifact.
3. **Duplicate and legality audit.** Zero duplicate survivors; every edit compiled through
   the frozen N1 compiler; complete incident-conflict ownership; no merge violations.
4. **Leakage re-check from the artifacts**, independently of the training-time receipt.
5. **Baseline identity.** Confirm the unchanged edge TP/FP/FN reference `10,341/1,064/673`
   and that the comparison baseline is the sealed one, not a convenient nearby run. The
   APNCC `1,096`-row scare was a **reference-selection error** (`BENIGN_FP`) — pick the
   right baseline and prove it.

## What a PASS does and does not unlock

**Unlocks:** one supplier-class-specific 44b6 graph contract for this exact PEFT realization.

**Does not unlock:** `6bba`, CV57, a tier move, a DEAD promotion, a plan amendment, a
submission, or a second PEFT variant. Each needs its own gate. Trackastra's earlier AUC
`0.918491` still led to a failed graph — a good number upstream has repeatedly not survived
this gate, which is why the gate is the gate.

## Value context — read before proposing anything downstream

Even a PASS at `+0.005` is small against what a submission would need:

| Quantity | Value |
|---|---|
| Banked APNCC local → public | `0.881522` → `0.881` |
| Local floor for submission-worthiness | `>= 0.89` |
| Local needed for an accepted public `0.89` | `~0.895–0.900` |
| Edge-only current-pool projection cap | `~0.896` |
| User-supplied top-50 bar (public) | `0.910` → local `~0.9255–0.9422` |

So a PASS is a **scientific** unlock, not a leaderboard one. Say so in the report rather
than letting a positive delta read as a submission signal.

## Done when

The independent replay matches, provenance closure is bound and self-contained at clean
checkout, duplicates are `0`, and the adjudication is written with an explicit statement of
what remains closed. Then, and only then, `K6` may be **proposed** to the owner.

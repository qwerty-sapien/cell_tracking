# Pasteable Codex triggers — 2026-07-30 recall + division-checkpoint DAGs

Three blocks. Both lanes are independent and safe to dispatch together; neither can spend
compute without a separate owner approval at its HUMAN-GATE. Use block 3 only if you want
one lane at a time.

---

## 1. Both lanes (recommended)

```text
Read CLAUDE.md and reports/g02/screening/LIVE_STATE.md first; their no-retry and
terminal rules override everything below.

Dispatch two independent DAGs under dag_prompts/2026-07-30-recall-and-division/.
Read that README.md for the inherited standing rules, then each lane's DAG.md before
its first node.

Lane A — recall supply. Root: recall/N0_pool_recall_ceiling.md
Lane B — division checkpoint pilot. Root: division_ckpt/N0_checkpoint_provenance.md

Cascade rule: a node's GO auto-advances to the next node in the same run with no human
turn. PARK, DEAD, ESCALATE, and HUMAN-DECISION halt that lane immediately and are
reported as-is; do not soften, retry, or reframe a stop verdict into a GO.

Every node's GO threshold is preregistered in its own file. Do not adjust a threshold
after seeing a result. If a threshold turns out to be unmeasurable as written, emit
ESCALATE and say so; do not substitute a proxy.

Hard stops that apply even on GO:
- recall/N3_human_kaggle_gate.md and division_ckpt/N2_human_kaggle_pilot.md and
  division_ckpt/N3_efficacy_gate.md are HUMAN-GATEs. Prepare the protocol, then halt.
- No Kaggle asset, dataset upload, kernel push, kernel run, GPU debit, submission, or
  push to any remote without explicit owner approval in the task. RunPod is forbidden.
- division_ckpt/N3 additionally requires two written owner amendments that do not exist
  yet (open division fitting; the B1 association-continuity prerequisite). Absent
  either, its verdict is HUMAN-DECISION and nothing runs.
- Do not re-screen any DEAD mechanism. Do not reopen any terminal/no-retry route.
- Full frozen 44b6 is mandatory; no video subsampling (0/12 usable cells).
- Local runs keep the 45-second idle preflight and the zero-swap rule.
- Only the root coordinator writes reports/g02/screening/LEDGER.md. Nodes propose rows.
- Update reports/g02/screening/LIVE_STATE.md after any real state change, per its
  update protocol: refresh Current frontier / Key numbers and append a dated Log line.

Write each node's artifact to the lane's runs/ directory using the filename named in
that node. Report at the end: per-lane verdict chain, the exact numbers each gate was
decided on, and what the owner must decide next.
```

---

## 2. Context the lanes assume (paste as a follow-up if Codex asks why)

```text
Why these two lanes exist.

Banked best is APNCC local 0.881522 / public 0.881. Public 0.910 maps to a planning
range of local 0.9255-0.9422, so roughly +0.044 to +0.061 local is needed. The in-pool
re-ranking oracle projects only ~0.896, and on the frozen action population the
free-add and node-add oracles are exactly 0. Selection over the existing candidate pool
therefore cannot cover the gap.

Lane A tests the alternative: candidate SUPPLY. The plan's H2 node ("a learned no-link
model makes the recall shell reachable") was conditioned on H1's frozen-Trackastra
features; H1 is DEAD at the 44b6 graph gate, so H2 was orphaned and never replaced.
44b6 held edge counts are TP/FP/FN 10,341/1,064/673, and H2 sized 456 correct edges in
the 10-12 um shell against a 5.34M background. N0 measures, for free, how many missing
true edges are absent from the pool entirely and what their zero-FP oracle ceiling is.
That number licenses or kills the whole lane before any mechanism is named.

Lane B is the owner-approved one-week pilot on a FROZEN EXTERNAL division checkpoint.
It exists because the five-fold fit-only TAP training plan needs at least 40 T4-h,
against a 10-16 h weekly envelope and 25.47 GPU-h launch capacity. A frozen external
weight also removes the transductive leakage that made one shared encoder illegal.
Per the three-stage plan this is a NEW named mechanism, not a TAP execution repair, so
it restarts at N0. No such checkpoint is currently authenticated, so N0 is a real gate.

The owner's continuation rule for Lane B is +10% on the division number. The frozen
scorer adds 0.1 * division_jaccard, so the division term is 0.1/1.1 = 9.09% of the
attainable total and +10% of it is +0.91% on the previous best score. CV57 has 42
ground-truth divisions and divisions are currently OFF. The binding form is therefore
>= 4 zero-FP division TP (division_jaccard >= 0.08022; local >= 0.889544). It is a
continuation rule, not the target: it is far short of public 0.910.
```

---

## 3. Single-lane variants

```text
Read CLAUDE.md and reports/g02/screening/LIVE_STATE.md first, then
dag_prompts/2026-07-30-recall-and-division/README.md and recall/DAG.md.
Dispatch recall/N0_pool_recall_ceiling.md and auto-cascade on GO only.
PARK/DEAD/ESCALATE/HUMAN-DECISION halt. recall/N3 is a HUMAN-GATE: prepare the
protocol, do not create an asset, launch a kernel, debit budget, submit, or push.
Preregistered thresholds are in the node files and may not be adjusted after a result.
```

```text
Read CLAUDE.md and reports/g02/screening/LIVE_STATE.md first, then
dag_prompts/2026-07-30-recall-and-division/README.md and division_ckpt/DAG.md.
Dispatch division_ckpt/N0_checkpoint_provenance.md and auto-cascade on GO only.
N0 and N1 are 0 GPU-h. N2 and N3 are HUMAN-GATEs and halt even on GO. N3 also
requires two written owner amendments that do not exist yet. Authenticated Kaggle CLI
work, if it is ever approved, goes through computer-use cmux and never the agent-shell
token. Preregistered thresholds are in the node files and may not be adjusted after a
result.
```

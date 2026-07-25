---
name: dag-prompts
description: >-
  Use this whenever the task is to CREATE a set of DAG prompts — a folder of
  dispatchable prompt files wired by a DAG.md that CASCADE: a GO verdict on one node
  automatically triggers the next, and so on, without a human turn between them.
  Trigger it whenever the user says "create the DAG prompts", "make a DAG of
  subtasks/prompts", "wire these as a cascading DAG", "decompose this into a prompt
  chain", or asks for a multi-step investigation/build where each node has a
  preregistered verdict and GO should auto-advance. Keep it lightweight; pair it with
  seeking-alpha (the DAG's root node is usually seeking-alpha's "make the objective
  computable" gate).
---

# DAG prompts

## What this produces

A **folder** of small, self-contained prompt files plus a **`DAG.md`** that wires them.
Each file is one dispatchable node. The defining property is **auto-cascade**: when a
node returns its GO verdict, the orchestrator proceeds *directly* to the next node in the
same run — the human is not in the loop for every gate. The human re-enters only at
**permission boundaries** and at stop verdicts. This turns a multi-step investigation
into one launch that runs until it hits a real decision.

Keep it lightweight: 3–6 nodes is typical, each under a page. Do not over-engineer.

## Anatomy

```
<dag-name>/
├── DAG.md                 (wiring, node index, verdict vocabulary, cascade policy)
├── <N0>_<root>.md         (root gate — usually the deterministic "is this even possible" test)
├── <N1>_<...>.md
└── <N2>_<...>.md
```

## The node contract

Every node file MUST contain, in this order, so it can be dispatched alone and can
cascade:

1. **Load line** — the skill(s) and upstream artifacts to read first (e.g.
   `seeking-alpha/SKILL.md`, `DAG.md`, the previous node's output).
2. **Objective / gate** — the single falsifiable question this node answers, stated as
   something computable (this is why the root is usually seeking-alpha Phase 1).
3. **Tasks** — the concrete work.
4. **Output** — the artifact(s) to write and the `LEDGER.md` row.
5. **Verdict** — one token from the controlled vocabulary below, with its preregistered
   threshold written *before* running.
6. **Cascade** — the explicit routing block (below).

## Verdict vocabulary

| Verdict | Meaning | Routing |
|---|---|---|
| **GO** | Gate passed at the preregistered threshold | Cascade to the next node |
| **PARK** | Weak/negative; this line stops but is revisitable | Halt; report |
| **DEAD** | Compelling negative; do not revisit without a new mechanism | Halt; report |
| **ESCALATE** | A locked decision or the goal itself must change | Halt; hand to the owner |
| **HUMAN-DECISION** | A judgment only the human should make | Halt; ask |

Preregister the numeric threshold that separates GO from the others *before* running, so
the cascade is honest and not retrofitted to whatever happened.

## The cascade rule (the defining behavior)

End every node with a Cascade block of this shape:

```
## Cascade
- On **GO** → proceed directly to `<next-node>.md` in the same run; do not wait for a
  human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
```

The orchestrator auto-advances through GO gates. A GO recorded for the root node means
the second node runs immediately, and so on down the chain.

## Hard-stop override (never cascade past a permission boundary)

Auto-cascade **must not** cross a permission boundary. Any node whose next action is a
Kaggle submission, a GPU/RunPod/paid-compute launch, a push, or any irreversible or
outward-facing effect is a **HUMAN-GATE**: it halts for explicit human approval *even on
GO*. Mark such nodes `HUMAN-GATE` in `DAG.md` and give them a `HUMAN-DECISION`-style stop
regardless of their analytical verdict. The cascade automates *analysis*, never *spend*
or *external effects*.

## DAG.md contents

- **Why this DAG exists** — one paragraph, ideally the seeking-alpha Phase-1 objective.
- **Node graph** — a small mermaid graph with the GO/stop edges labeled.
- **Node index** — a table: id, file, purpose, the exact gate to advance.
- **Cascade policy** — restate the GO-auto-advances / HUMAN-GATE-halts rule so the
  orchestrator sees it up front.
- **Standing rules** — inherited gates that every node respects.

## Minimal node template

```markdown
# <ID> — <title>

> Load `<skill>`, `DAG.md`, `<upstream artifact>`. [HUMAN-GATE if it spends/pushes.]

## Objective / gate
<the one computable, falsifiable question; the preregistered GO threshold>

## Tasks
1. ...

## Output
`<artifact>` + a LEDGER row.

## Verdict & cascade
- **GO** (meets <threshold>) → proceed directly to `<next-node>.md`.
- PARK / DEAD / ESCALATE → halt and report.
```

## Anti-patterns

- **Cascading past a HUMAN-GATE.** Analysis auto-advances; spend and external effects
  never do.
- **Auto-advancing on anything but GO.** PARK/DEAD/ESCALATE always halt.
- **Un-preregistered verdicts.** If the GO threshold is decided after seeing the result,
  the cascade is just motivated reasoning with extra steps.
- **A node that can't emit a crisp verdict.** If it can't pass/fail against a stated
  number, it isn't a DAG node yet — sharpen its objective first.
- **Over-building.** More nodes ≠ more rigor. Keep it lightweight.

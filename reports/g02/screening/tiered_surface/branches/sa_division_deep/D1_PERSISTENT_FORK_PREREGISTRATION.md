# D1-PERSISTENT-FORK deep preregistration

Frozen: 2026-07-28, before the Wave-2 component audit was executed.

Status: **ORACLE / PREREGISTRATION ONLY**. This document opens no fitting,
held-family graph, CV57, GPU, Kaggle, submission, RunPod, or DEAD gate.

## Frozen inputs

All paths below resolve under the frozen coordinator root
`/Users/agent/Desktop/cell_tracking` and are read-only.

| Input | SHA-256 |
|---|---|
| `reports/g02/screening/tiered_surface/ALIGNMENT.md` | `eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37` |
| `reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md` | `066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49` |
| `reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json` | `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865` |
| `src/celltrack_alpha/action_compiler.py` | `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5` |
| `reports/g02/grouped_3fold_audit20_v1.json` | `ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769` |
| `reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json` | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06` |
| `reports/g02/division_failure_holdout57_postpatch_v1.json` | `10a8e480cbd0134f0fec18c1d302b388c4dd3ecbaf02f6eb3298f79eb40d2b44` |
| `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/d1_persistent_fork/N0_preregistration.json` | `34b2aa020099e14ad7a73820d97e9baefadec386e0e2d9acbd7ab5c1cb07620f` |
| `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/d1_persistent_fork/N1_arm_manifest.json` | `327af0d29dbde19172a61bf396b951d150040bb357c3fbba90e79249035e4776` |
| `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/d1_persistent_fork/N1_dry_validation.json` | `f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9` |

The full-CV57 division-failure artifact and its referenced existing
`kaggle_kernel/bhcv57out_v3/fragments/` are diagnostic scorer inputs already
opened and banked by the campaign. This Wave-2 audit does not score, refit, or
alter them.

## Event grammar and equivalence classes

An event has one parent-side matched lineage and two unordered daughter
lineages. Scorer timing is represented by three mutually exclusive labels:

1. `DIRECT`: the emitted binary fork is on the matched divider/parent-side
   node.
2. `LATE`: the parent-side lineage continues linearly for one frame and the
   binary fork occurs on the matched daughter-stage continuation.
3. `PERSISTENT_FORK`: either direct or late, with both child branches carrying
   distinct evidence through the grandchild stage.

`PERSISTENT_FORK` is an evidence refinement, not a third timing hypothesis.
Direct and late realizations that cover the same parent-side lineage and the
same unordered daughter-lineage pair belong to one scorer-equivalence class.
The class key is:

`(video_identity, parent_lineage_anchor, sorted(branch_a_anchor, branch_b_anchor))`.

Oracle labels may use the frozen official match solely to enumerate the class;
a future fitted decoder must construct a label-free predicted-graph analogue
from its fit-only upstream graph. Oracle identity is never an inference
feature.

## Exact duplicate suppression

Candidates are sorted by:

`descending frozen score, timing rank DIRECT<LATE, parent id, sorted child ids,
sorted grandchild ids, equivalence key`.

Before graph emission:

- keep at most one candidate per equivalence key;
- keep at most one candidate using any parent, child, or persistence-evidence
  node already owned by a higher-ranked candidate;
- reject missing two-branch persistence in the active D1 arm;
- record every rejected candidate with winner key and reason;
- require `duplicate_survivors == 0`.

The persistence ablation removes only the two-grandchild requirement. It does
not change ordering, equivalence, ownership, or suppression.

## N1 action ownership

Each accepted event compiles as one atomic N1 `fork` action. It owns both added
parent-to-child edges and every pre-existing incident edge that N1 must remove
to satisfy indegree `<=1`, outdegree `<=2`, forward time, endpoint existence,
and no-merge constraints. No other event may own or modify an endpoint in that
atomic action. Persistence edges are evidence and are never invented by D1.
An incomplete incident-conflict set, nonbinary fork, duplicate endpoint, or
invalid time relation fails closed.

## Wave-2 diagnostic audit

The already-banked full-CV57 report states 42 GT division FNs. The Wave-2
question is narrower: among those 42, how many have every strict scorer role
present but split such that no prediction component contains the parent-side
role plus both distinct daughter-lineage roles?

For each of the 42 events, the audit will:

1. authenticate the frozen report and every referenced prediction fragment;
2. parse only existing node/edge rows;
3. compute undirected prediction components deterministically;
4. preserve the frozen official-match role sets from the report;
5. classify:
   - `PRESENT_FRAGMENTED`: all roles present, no component spans the parent
     and both daughter lineages, and required role nodes occupy at least two
     components;
   - `PRESENT_CONNECTED_NO_LOCAL_FORK`: all roles present and some component
     spans all roles, but the banked scorer report found no eligible local
     parent fork;
   - `ROLE_MISSING`: at least one strict role absent.

The primary count is `PRESENT_FRAGMENTED / 42`. Secondary denominators are
role-covered events and the historical prefix-52 audit. They must not be
silently equated. “Role present” and “scorer TP” are not equivalent: a TP also
requires the local fork and global one-to-one pairing.

## Dependency and kill line

Fitting remains locked until one single banked association-continuity arm
passes **both families** at adjusted edge-Jaccard delta `>= +0.005`, raw-edge
delta `>= 0`, node-recall delta `>= 0`, and a positive label-free connected
fork-opportunity delta. Families or arms may not be spliced.

After that dependency passes, D1 is evaluated `44b6 -> 6bba -> CV57` against
unchanged, D0, and the persistence ablation. Kill the D1 instantiation
immediately on 44b6 if any adjusted/raw/node delta is negative, division
Jaccard is `<= 0.15`, combined score delta is not positive, or any duplicate
survives. GO requires division Jaccard `>= 0.30`, nonnegative graph vectors,
positive combined delta, and D1 attribution beyond D0 on both families.

This preregistration cannot write a DEAD verdict. A killed instantiation is at
most a cited PARK-C proposal; mechanism-level DEAD remains human/root-only.


# SA-DELETE Phase-1 DAG

Status: **N1 PREREGISTERED; HELD GRAPH OUTCOMES CLOSED**

Objective: test whether a conservative, feasibility-exact true-edge
retention/no-link beta improves the frozen graph objective over the exact P0
`c=0.25` baseline.

```text
D0 authenticate five frozen inputs + transitive P0 payloads
 |
 v
D1 freeze implementation, fixtures, features, capacity, folds, threshold, null
 |
 v
D2 fresh 45 s idle preflight -> fixture tests under zero-swap guard
 |
 v
D3 fresh 45 s idle preflight -> fit-only nested models -> 44b6 graph gate
 |\
 | +-- KILL: Δadj < +0.005 or Δraw < 0 or Δnode < 0 or invalid action
 |       -> PARK-C this instantiation; 6bba/CV57 closed
 |
 +-- GO: Δadj >= +0.005 and Δraw,Δnode >= 0, invalid count 0
         |
         v
       D4 identical frozen 6bba graph gate
         |\
         | +-- KILL -> PARK-C this instantiation; CV57 closed
         |
         +-- GO -> report only; CV57 remains root/human-gated
```

Every video emits exactly one N1 action: a bundled `delete_edges` action when
the fixed admission rule is met, otherwise an explicit `noop`. Each action is
compiled alone and as the complete one-action final batch; hashes must agree
and the invalid count must be zero.

This is intentionally the crowded beta/capacity anchor for later association
alphas. It makes no novelty claim. Its scientific value is the cheap,
objective-level falsification of whether pretrained detection evidence plus
conservative exact retention realizes any of P0's deletion headroom.

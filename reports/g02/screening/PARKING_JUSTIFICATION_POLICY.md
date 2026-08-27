# Parking justification policy

Status: **BINDING FOR NEW OR UPDATED PARK DECISIONS**

A `PARK` is a reversible evidence decision, not shorthand for failure and not a
scientific negative. Every new or materially updated parked node must include a
human-readable justification and, when the node already has machine receipts, a
machine-readable companion with the following fields:

1. **Exact realization** — candidate/mechanism ID, frozen observation, estimator,
   action, assets, and execution boundary.
2. **Work actually performed** — bounded acquisition, static checks, imports,
   forwards, identities, resources, and external mutations, including zero counts.
3. **First causal blocker** — the narrowest stable failure class and the exact
   receipt/log/hash that proves it.
4. **What is invalidated** — only conclusions logically established by that
   evidence.
5. **What is not invalidated** — especially scientific quality when execution
   stopped before prediction, action, media, or score.
6. **Substitution analysis** — why an available model, checkpoint, dataset,
   runtime, or interface is or is not an equivalent realization. Silent
   substitution is forbidden.
7. **Unblock predicate** — concrete bytes, provenance, runtime, trust, execution,
   media, or score evidence required to resume.
8. **Next fail-fast test** — the cheapest outcome-independent test that can kill or
   advance the exact realization, with an explicit stop condition.
9. **Resource/external accounting** — compute, cost, quota, instances, uploads,
   pushes, and submissions consumed.
10. **Decision scope** — `PARK-DEPENDENCY`, `PARK-INFRASTRUCTURE`,
    `PARK-GOVERNANCE`, `PARK-SCIENTIFIC-REVISE`, or another precise subtype;
    `DEAD` requires direct scientific falsification of the frozen realization.

Minimum machine-readable keys are:

```json
{
  "mechanism_id": "...",
  "verdict": "PARK-...",
  "realization": {},
  "work_performed": [],
  "first_causal_blocker": {},
  "invalidated": [],
  "not_invalidated": [],
  "substitutions": [],
  "unblock_predicate": [],
  "next_fail_fast_test": {},
  "resource_accounting": {},
  "evidence": []
}
```

`LIVE_STATE.md` should contain only the current concise justification and link to
the dossier. Historical receipts are not rewritten; a new dossier may supersede
their wording while preserving their bytes.

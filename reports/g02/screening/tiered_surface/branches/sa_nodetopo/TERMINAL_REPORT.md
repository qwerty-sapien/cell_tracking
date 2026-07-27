# SA-NODETOPO terminal report

Verdict: **PARK_NO_COMPLETE_AUTHENTIC_CANDIDATE_UNIVERSE**

## Outcome

The N4 prerequisite was executed as a deterministic provenance audit, not stopped at
planning. All frozen pins matched. The immutable preregistration was committed at
`1abc8c7` before a score or held-family outcome opened.

The audit authenticated a label-independent legacy proposal procedure, but not an
honest full proposal universe:

- legacy persisted tables: 15,240 rows / 14,280 canonical identities / zero identity
  coordinate conflicts;
- partial fit105 coverage: 4/36 44b6 and 3/69 6bba datasets, one three-frame clip per
  dataset;
- embedded GT-derived fields are present in those training artifacts and were excluded
  from selection;
- extracted raw evidence: 12 training Zarrs, 10 overlapping fit105, not full coverage;
- exact full-universe unlabeled background: absent;
- candidate-to-baseline incident-topology/conflict join: absent.

P0 itself reports the node-restore background as undefined with a null count. Its GT
oracle was not used as a proposal source. No coordinates, masks, candidates, edges, or
background actions were fabricated.

## N1 validity and controls

The frozen N1 implementation compiled one complete synthetic `restore_topology`
transaction and rejected its isolated coordinate-only counterpart. This validates the
interface semantics only. Zero real candidate actions were emitted because the real
identity/topology join is absent.

The no-op, isolated-node-add, topology-restore score controls and false-node stress
curve were not run: constructing them without the complete background bank would
violate the preregistration. Consequently there is no 44b6 graph vector, 6bba remained
closed, and CV57 remained closed.

## Verification

- Focused audit tests: **7/7 PASS**.
- Fresh idle preflight: **45.451 s**, swap growth **0 MiB**.
- Guarded test+audit run: **PASS**, swap growth **0 MiB**.
- First guarded attempt: stopped before tests because system Python lacked SciPy;
  recorded separately with swap growth **0 MiB**. The repository environment was then
  used after a fresh preflight.
- No audit/test/CV57 data, held graph score, external service, Kaggle GPU, RunPod,
  submission, sibling output, DEAD rescreen, or root single-writer mutation occurred.
- Memo C2 values and Branch-B values remained quarantined and unused.

## State

No tier move is proposed. `NODETOPO-O0` retains its P0-banked PARK-A headroom while
blocked on an authentic candidate universe. `NODERESTORE-O0` remains PARK-B. No
`GPU_REQUEST.json` is emitted because there is no complete authenticated candidate path
or credible end-to-end projection.

Resume condition: root freezes, by hash, a full fit105 label-independent candidate bank,
an exact deployment background, and a candidate-to-baseline join that enumerates every
added incident edge and displaced conflict under N1.

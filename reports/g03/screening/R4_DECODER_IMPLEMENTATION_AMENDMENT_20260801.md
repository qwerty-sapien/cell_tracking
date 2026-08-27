# G-03 Wave R / R4 — decoder implementation amendment

Date: 2026-08-01  
Status: **pre-outcome sealed; no target or division outcome accessed**

This amendment removes implementation freedom discovered while turning the already
sealed R4 grammar into executable code. It does not alter the `18,144` active
configurations, the `DIVISIONS_OFF` null, the effective dev69/held-fit36/hidden4
population, the nested/grouped-OOF selection rule, or any gate.

The machine-readable authority is
`wave_r/R4_DECODER_IMPLEMENTATION_AMENDMENT.json`. It binds decoder SHA-256
`d310394310116ebf257d74159a39be1d77679a9dd7aec14d97f3589e9f7da44a`
and test SHA-256
`4af489481a4afd3c5187b67e9d6cfc3797c96e1048c930237ddc979ba693a7a5`.

## Frozen resolutions

1. The association parent graph owns topology. A fork source and both emitted
   daughters must already be parent nodes. Bank-only detections can support persistence
   but can never be emitted; the decoded node set must remain unchanged.
2. Daughter persistence is measured on deterministic one-frame steps already selected
   in the parent graph. Grandchild persistence follows deterministic highest-probability
   forward bank steps after those daughter paths, with ascending target ID as the tie
   break.
3. Each accepted event is compiled atomically with frozen `compile_action`, then events
   are installed sequentially. The final graph must be valid and forward versus reverse
   installation must produce the same graph hash. Frozen `compile_batch` is not the
   installer because it rejects the action-local delete-and-readd of a retained primary
   edge before resolving that valid atomic action.
4. Exact predicted-lineage equivalence runs before ownership, temporal suppression, and
   caps. DIRECT sorts before LATE, so an otherwise-equivalent LATE event is suppressed.
   The literal frozen grammar can therefore make `late_window` output-inert. The lattice
   remains unchanged and repeated graph hashes are deduplicated; no post-outcome grammar
   repair is allowed.
5. `DIVISIONS_OFF` is an exact parent-graph no-op. Hash drift, node-set change, invalid
   graph, order-dependent result, or a surviving duplicate fails closed.

The decoder and all contract checks passed the promoted combined Wave-R test surface
before any development score. The known inert-axis result is reported as a feasibility
finding, not hidden as nominal search diversity. A future mechanism that gives LATE a
distinct label-free meaning would require a new named, preregistered node.

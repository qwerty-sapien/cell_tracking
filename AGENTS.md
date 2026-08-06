# Agent instructions

**G-02 live state.** Before acting, READ `reports/g02/screening/LIVE_STATE.md` — the single
shared source of truth for the cell-tracking campaign, kept current by both Claude and Codex.
After any state change (a node result, a gate flip, a DEAD promotion, a decision), UPDATE it:
refresh Current frontier / Key numbers and append a dated Log line (newest first). Keep it
short — detail belongs in the linked reports.

- Human-locked plan: `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md` (2026-07-25).
- Only the root coordinator writes `reports/g02/screening/LEDGER.md`.
- Do not re-screen any DEAD mechanism (LIVE_STATE "DEAD" list / DAG §6.1) without a new named mechanism.
- **Standing owner authorization (2026-07-31):** Kaggle kernel launch,
  competition submission, and monitoring are pre-authorized when an
  observation-distinct candidate or approved experimental node makes the
  action productive. Do not ask again for Kaggle access in that case. Quota
  expiry alone is not productivity; preserve the locked-plan/preregistration
  gates, shared serial lock, authoritative quota reconciliation, prelaunch
  integrity checks, exact version accounting, terminal monitoring, and the
  shared 30 h/week GPU ledger. For Kaggle GPU runs, an authenticated projection
  below 8 h is auto-approved and has no per-run wall-clock kill; a projection
  of 8 h or more requires an explicit human approval after a literature-cited
  pitch. Never shard or restructure a run merely to evade that gate. RunPod
  still requires explicit human approval.

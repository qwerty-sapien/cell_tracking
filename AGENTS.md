# Agent instructions

**G-02 live state.** Before acting, READ `reports/g02/screening/LIVE_STATE.md` — the single
shared source of truth for the cell-tracking campaign, kept current by both Claude and Codex.
After any state change (a node result, a gate flip, a DEAD promotion, a decision), UPDATE it:
refresh Current frontier / Key numbers and append a dated Log line (newest first). Keep it
short — detail belongs in the linked reports.

- Human-locked plan: `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md` (2026-07-25).
- Only the root coordinator writes `reports/g02/screening/LEDGER.md`.
- Do not re-screen any DEAD mechanism (LIVE_STATE "DEAD" list / DAG §6.1) without a new named mechanism.
- No Kaggle GPU/submission or RunPod launch without explicit human approval.

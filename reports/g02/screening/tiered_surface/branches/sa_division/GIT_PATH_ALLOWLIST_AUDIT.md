# SA-DIVISION git/path allowlist audit

Date: 2026-07-28  
Frozen git base: `2c74eed32c6f080faff045ae22847b98bbde8db0`  
Branch: `codex/sa-division`  
Mandatory worktree: `/private/tmp/g02_phase1_20260728/sa_division`  
Only allowed prefix:
`reports/g02/screening/tiered_surface/branches/sa_division/`

The final barrier validator must confirm:

- branch equals `codex/sa-division`;
- merge-base equals the frozen base;
- every changed path begins with the allowed prefix;
- no root single-writer file changed;
- `TERMINAL_REPORT.md`, `PROPOSED_LEDGER_ROWS.md`, and
  `state_fragment.json` exist;
- the fragment's evidence artifact resolves and matches SHA-256;
- forbidden-scope flags are explicit false;
- `6bba_ran` is false;
- memo-only markers, if present in documentation, are explicitly quarantined.

No shared root file or sibling subtree was modified.


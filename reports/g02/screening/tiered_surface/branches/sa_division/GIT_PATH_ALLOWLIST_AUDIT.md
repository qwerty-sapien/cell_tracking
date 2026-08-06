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

## Follow-on oracle refinement

The same frozen base, branch, worktree, and allowed prefix govern the
division-oracle refinement. All new source, preregistration, repair history,
preflight/guard records, diagnostic result, fragment update, terminal report,
and proposed ledger row remain inside the exact SA-DIVISION prefix.

The final validator must additionally resolve
`state_fragment.json`'s evidence to `oracle_refinement_results.json` at
SHA-256
`aded494fd19aca50a9123dfdbcccfd37448709330f8449e76984359833c44c22`,
confirm `tier_move_proposed=false`, and confirm both family-execution flags
remain false. P0 diagnostic values may appear only with
`diagnostics_promote=false`.

Final guarded validator result: **PASS** against frozen base
`2c74eed32c6f080faff045ae22847b98bbde8db0`. It resolved fragment evidence,
found every changed path under the exact allowed prefix, found all frozen pins,
confirmed the quarantined `66/223` marker was absent from tier and ledger
surfaces, and rejected no path. The barrier guard ran in 0.645 seconds at
0.0220 GiB peak process-tree RSS with zero swap growth.

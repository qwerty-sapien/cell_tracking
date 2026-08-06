# Independent review — division realizer wave

**Verdict: FAIL CLOSED on full-wave integrity; hold the 44b6 barrier.**

The approved plan amendment and CELLECT PARK-C decision are authenticated. The
corrected inventory is 21 held `44b6` fragments and 36 forbidden `6bba`
fragments. Frozen N1/scorer hashes, CPU resource guards, zero-duplicate
suppression, and all external hard stops match.

Arm outcomes:

| Arm | Integrity adjudication | TP / FP / FN | Division Jaccard | Adjusted / raw / node delta | Combined delta |
|---|---|---:|---:|---:|---:|
| A geometry/motion | **VALID FAIL** | 0 / 1 / 11 | 0 | −0.000141523 / −0.000140538 / 0 | −0.000141523 |
| B persistence | **INVALID — information firewall**; metrics diagnostic only | 0 / 49 / 11 | 0 | −0.009455519 / −0.009480291 / 0 | −0.009455519 |
| C appearance | **NOT EVALUATED — fail closed**; no adequate frozen intensity cache | — | — | — | — |
| D 3-of-3 consensus | **INVALID REQUIRED INPUTS**; mechanical zero only | 0 / 0 / 11 | 0 | 0 / 0 / 0 | 0 |

Arm B's operator ran this broad discovery command before preregistration commit
`506e438e`:

```text
rg -n "late_fork|late-fork|D1|persistence|compile|action" reports/g02/screening/tiered_surface/branches/sa_division_value reports/g02/screening/tiered_surface/branches/sa_division_cont -g '*.py' -g '*.md' -g '*.json' | head -n 240
```

It exposed structural lines from forbidden target-bundle/dossier paths. The
predictor itself has no target argument, the prediction SHA was committed before
scoring, GT fields are absent, and the diagnostic metrics were not changed.
Those facts do not repair the pre-prereg operator contamination, so B is
non-promotable and no rerun is authorized.

D's zero output follows the frozen rule because C has no vote, but B is also an
invalid required input. Its byte hashes and zero-score arithmetic are authentic;
it is not a valid scientific screen of the A–C consensus. Root's postrun
adjudication agrees and assigns `FAIL_CLOSED_INVALID_COMPONENT_INPUTS`
(`881c26cf…f630`).

All arms have zero duplicate survivors. The fixed 25-TP sensitivity is preserved:
0/5/10 FP gives `+0.059523810/+0.053191489/+0.048076923`.

The four oracle FNs are correctly split into three missing-role cases and one
already-connected, locally forkless event with missing branch persistence.
`D2_ROLE_RECOVERY_DIAGNOSTIC` and
`D3_CONNECTED_NONPERSISTENT_LOCAL_FORK_DIAGNOSTIC` are proposals only.

Verification included byte-reading all pinned contract, target, N1, scorer,
prediction, result, consensus-input, and fragment receipts; 21/21 fragment hashes
matched for A, B, and D; JSON parsing passed; Arm C's focused tests passed 3/3;
duplicate and score-formula checks passed. No scientific run was repeated.

Result: A is the only valid scientific arm outcome and it fails. B is
quarantined, C is unevaluated, and D is invalid as a combined screen. Do not open
`6bba`, CV57, submission, GPU/Kaggle, RunPod, a full-pool cache, or DEAD action.

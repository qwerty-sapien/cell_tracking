# N3 — Local CV57 → public transfer calibration

> Load `.agents/skills/seeking-alpha/SKILL.md`, `DAG.md`,
> `reports/g02/screening/p0_parity.md`, `runs/N0_SUBMISSION_FORENSICS.md`.
> **Blocked on N0 = GO.** Without a working scoring channel there is nothing to calibrate
> against.

## Objective / gate

`p0_parity.md` records: P0A **PASS** (our local scorer is byte-attested against the official
checkout), P0B coarse high-range **PASS**, P0B **fine global transfer: STILL-UNRESOLVED**.
The recorded failure is concrete — two near-baseline configurations *reverse order*: `c=2` is
lower locally by 0.00099965 and higher publicly by one displayed bin.

That unresolved fine ranking is why we cannot currently tell whether a +0.005 local gain is
real. It also blocks S1 final-submission selection (2 slots only).

Known pairs:

| Local CV57 | Public | Δ |
|---:|---:|---:|
| 0.8483945156 | 0.860 | +0.0116 |
| 0.8721499475 | 0.878 | +0.0059 |
| 0.8731495935 | 0.877 | +0.0039 |
| 0.8815220700 | 0.881 | −0.0000 |

The delta is **shrinking monotonically as local score rises** — extrapolating it to
0.914358 is exactly the kind of unvalidated inference this node exists to kill or confirm.

**Preregistered GO threshold:** a transfer relation that correctly orders two configurations
differing by **<0.005** locally, validated on a pair not used to fit it. Fitting a line to
four points and declaring victory is PARK, not GO.

## Tasks

1. Re-derive the transfer relation over all local/public pairs, including any newly scored
   submission unlocked by N0.
2. Quantify the leaderboard **display quantization** (public scores show 3 decimals) and
   determine the minimum local delta that is even *observable* publicly. This may prove the
   fine-ranking question unanswerable at current precision — a legitimate PARK.
3. State the public interval predicted for local 0.914358, with its uncertainty.
4. Decide whether local CV57 can be used to choose between candidates, or only to reject
   large regressions.

## Output

`runs/N3_TRANSFER_CALIBRATION.md` + a LEDGER row. Update `p0_parity.md`'s P0B verdict.

## Verdict & cascade

- **GO** (correct ordering on a held-out sub-0.005 pair) → proceed to
  `N4_KERNEL_PACKAGE_SUBMIT.md` (which then halts as a HUMAN-GATE).
- **PARK** (quantization makes fine ranking unobservable) → halt; report, and record that
  local CV57 is a *regression filter only*, not a selector.

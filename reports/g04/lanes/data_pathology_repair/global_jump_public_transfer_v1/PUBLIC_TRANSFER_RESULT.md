# APNCC+G1 public transfer — 2026-08-04

## Lifecycle status

Fresh slug `onebitbrain/g1jump-submit` exact Version 1 completed and the
downloaded output passed the hidden-compatible terminal predicate. The
candidate is SHA-256 `366a12d64e155d2722fccb579d76fb0c040a889f47f857ae089430f613c9a8c8`,
13,448,137 bytes, 263,260 data rows, 134,409 nodes, and 128,851 edges.

Exactly one code-output competition submission was accepted as ref `55221497`
at `2026-08-03T21:00:28Z`. The one-shot registry is consumed and retry is
prohibited. Kaggle terminal state: **COMPLETE**; displayed public score:
**`0.881`**, observed at `2026-08-04T11:57:56Z`.

The bracketed launch started at authoritative quota `5.10/24.90 h`
used/remaining and terminal reconciliation measured `5.27/24.73 h`, for a
`0.17 h` debit. The `6.0 h` ceiling is released and there are no G1
reservations.

## Local-to-public anchors

| Anchor | Candidate | Exact CV57 adjusted | Displayed public | Submission ref |
| ---: | --- | ---: | ---: | ---: |
| 1 | APNCC | `0.881522069960` | `0.881` | `54942864` |
| 2 | APNCC + G1 global-jump correction | `0.886764420260` | `0.881` | `55221497` |

The second anchor is terminal. G1 gained `+0.005242350300` locally but `+0.000`
at displayed public precision. This is a strong non-transfer observation: if
the same three-decimal display rule applies to both submissions, any hidden
public improvement is below the display-bin width and therefore at least about
80% smaller than the local gain. It does not identify whether the cause is
hidden-population shift, correction-opportunity prevalence, or another
distribution-dependent interaction; exact local and displayed public values
remain separate.

## Integrity receipts

- prelaunch: `KERNEL_PRELAUNCH_RECEIPT.json` (`895a33fd…d7c`);
- terminal kernel status: `cmux_receipts/kernel_status_2026-08-03T205705283350Z.json`
  (`ed809498…827`);
- exact Version-1 output: `cmux_receipts/kernel_output_2026-08-03T205845039316Z.json`
  (`17c8e1e7…602`);
- submission predicate: `CODE_SUBMISSION_PREDICATE.json` (`1d20ce86…1ed`);
- one-shot submission: `submission_receipts/competition_code_submit_2026-08-03T210022833522Z.json`
  (`9ef384f9…6a2`);
- terminal monitoring receipt: `cmux_receipts/competition_submissions_2026-08-04T115753645064Z.json`
  (`2f5e33e3…1f70`).

The task heartbeat was stopped after this terminal record. It never submitted
or read V32 state.

No V32 status read, shard mutation, or cadence change occurred.

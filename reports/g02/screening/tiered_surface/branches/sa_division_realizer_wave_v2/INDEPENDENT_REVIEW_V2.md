# Independent review v2 — full-38 division wave

Status: **PASS for the binding A/B/C conclusions; HOLD at the barrier.**
Clean D is **invalid/nonbinding**, not a passing or valid abstention arm.

I independently re-counted each sealed prediction against all frozen direct
repair and declared late-fork variants in the 38-event target. The recount
matches the binding terminal:

| Arm | Role | TP / FP / FN | Division Jaccard | Adjusted / raw / node delta | Exact combined delta | Verdict |
|---|---|---:|---:|---:|---:|---|
| A geometry/motion | binding | 0 / 23 / 38 | 0 | −0.000022335 / −0.000022463 / 0 | −0.000022335 | FAIL-PARK-C |
| B persistence | binding | 0 / 48 / 38 | 0 | +0.000029258 / +0.000028813 / 0 | +0.000029258 | FAIL-PARK-C |
| C appearance | binding | 0 / 22 / 38 | 0 | −0.000022335 / −0.000022463 / 0 | −0.000022335 | FAIL-PARK-C |
| D exact 3-of-3 | nonbinding diagnostic | 0 / 0 / 38 | 0 | 0 / 0 / 0 | 0 | INVALID-NONBINDING |

B’s tiny positive edge delta does not satisfy the unchanged gate because it
recovers no target event and has division Jaccard `0 < 0.30`. A and C also
regress edge score. No binding arm passes.

## Integrity and information firewall

- The target bytes hash to
  `0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221`
  and contain exactly `9` 44b6 plus `29` 6bba events. The same 57 fragment
  identities and hashes appear in A, clean B, C, and the terminal scorer
  receipts.
- A and C use the hash-pinned target-free predictor and were sealed before
  evaluation. Clean B’s own preregistration, predictor, actions, run receipt,
  structural replay, and firewall chain all match their declared hashes. Its
  firewall records zero target/label, scorer/evaluator, outcome, and other-arm
  reads. The 48 clean actions adapt exactly to the 48 evaluated predictions.
- All binding arms enforce one candidate per video and have zero duplicate
  survivors. The banked 25-TP sensitivity recomputes to
  `+0.059523810 / +0.053191489 / +0.048076923` at `0 / 5 / 10` FP.
- C’s cache claim is accurate: 57 JSON files, 385 graph-local candidates,
  eight candidates per video at most, and 476,307 bytes total. The cache
  declares no voxels or labels and is not a full-pool cache.

Clean D’s content is target-free and its exact A∩B∩C intersection is empty.
However, its preregistration SHA
`36b606df…d50617` is not the original D SHA pinned by either pre-outcome
implementation freeze. The final provenance adjudication therefore correctly
marks D invalid/nonbinding; no retroactive freeze was created.

The older root-level `WAVE_TERMINAL_RESULT.json` is a superseded,
diagnostic-only record from the target-exposed coordinator B/D route. It must
not be consumed as binding. The binding machine record is
`clean_terminal/WAVE_TERMINAL_RESULT.json`
(`ca7307c0…b6067`), and the report is
`WAVE_TERMINAL_REPORT.md` (`b6da6073…5b18d`).

## PEFT preregistration

The final PEFT artifact is internally consistent and is a preregistration
only—not launch authority:

- the split path is `reports/g02/grouped_3fold_audit20_v1.json`; its declared
  and actual SHA are both `ddb313cb…f8769`, and declared and actual fold counts
  are both three;
- selection is nested OOF on 44b6 held-fit frozen-scorer graph delta, with
  6bba closed;
- the backbone stays frozen, full fine-tuning is forbidden, and optimization
  must abort if the authenticated trainable/base ratio is `>=1%`;
- projection is `4.8` T4-h, the hard kill is `4 h 55 min`, current
  authoritative balance is `24.98 h`, and projected remainder is `20.18 h`;
  a fresh authoritative quota query and the existing human GPU gate remain
  mandatory.

PEFT preregistration SHA: `0d96b336…e71ad8`. Budget reconciliation SHA:
`6e7cbf74…d8f5fb`. No PEFT launch occurred.

## APNCC transport distinction

The two claims are separate and both are now supported:

1. `REMOTE_SUBMISSION_SCHEMA_VALIDATION.json` proves structural CSV validity:
   263,282 rows, 134,409 nodes, 128,873 edges, four datasets, zero validation
   errors, SHA `6944ed58…8bea3`.
2. `PUBLIC_ARCHIVE_PARITY_ADJUDICATION.json` proves byte identity to the
   correct authenticated public-0.881 T4 archive
   (`results-bhap.zip`, SHA `031aa5f6…7219`). I independently confirmed the
   submission and all four fragment member hashes, plus identical normalized
   non-timing `run.json` content (SHA `3e01fade…cc6`).

The older parity failure compared against the wrong pre-appearance baseline.
It remains an immutable historical receipt but is superseded for the transport
conclusion. Transport validity does not mean a scored submission occurred and
does not authorize a graph, CV57, tier, or scientific conclusion.

## Barrier

`pass_arms=[]`, `downstream_unlock=false`. No submission, PEFT launch, RunPod
action, full-pool cache, DEAD change, or other unauthorized unlock was found.
The campaign remains held at the full-38 human barrier.

Review limitations: the ignored reports tree has no committed Git chronology,
so sealing order was assessed through embedded hashes, receipts, fail-closed
adjudications, and filesystem ordering. The 87.4 GB raw archive’s banked SHA
was not independently recomputed; its exact size and frozen evaluator pin were
checked.

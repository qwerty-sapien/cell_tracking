# 6bba-only temporal hybrid confirmation

Decision: **PARK**.

The preregistered hybrid kept all 21 `44b6` videos on the banked APNCC
direct-swap policy and replaced proposals only for the 36 `6bba` videos with
the hash-pinned temporal family ranker.

| candidate | CV57 | TP | FP | FN |
|---|---:|---:|---:|---:|
| banked APNCC direct swap | **0.8815220700** | 34,320 | 2,459 | 1,948 |
| `44b6` APNCC + `6bba` temporal | 0.8801140024 | 34,308 | 2,511 | 1,960 |
| hybrid minus banked | **-0.0014080676** | -12 | +52 | +12 |

The temporal gate had passed independent fit-only OOF and validation support
within `6bba`, but its full frozen CV57 realization did not beat the
preregistered reference. This is evidence that proposal-level precision and
utility were insufficient proxies for graph-level direct-swap quality, not a
reason to weaken the gate after seeing CV57.

The raw generated report retained a legacy comparison against the older
`bhcv57out_v3` fragment baseline (`0.8711885823`). Its candidate score and
confusion counts are valid, but that `CONFIRM` decision is not the
preregistered decision. `reconciled_decision_v1.json` supersedes it and the
evaluator now selects the correct banked reference explicitly.

No Kaggle, GPU, RunPod, or paid action occurred. The banked public candidate
remains APNCC direct swap at `0.8815220699597315`.

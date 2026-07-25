# Extended-radius shell preparation

Decision: **BANK the diagnostic preparation; require an abstaining support gate
before any graph replay.**

The exact fit-only replay enumerated the newly admitted `10 < d <= 12 µm`,
cap-8 shell around the frozen backward-motion `c=0.25` graph. It reconciled the
earlier reachability audit exactly:

| partition | videos | shell pairs | correct one-to-one links | known negatives | unmatched/unknown |
|---|---:|---:|---:|---:|---:|
| fit-train | 105 | 4,959,618 | 405 | 2,832 | 4,956,369 |
| fit-validation | 12 | 382,453 | 51 | 99 | 382,301 |
| total | 117 | 5,342,071 | 456 | 2,931 | 5,338,670 |

All 456 correct links were absent from the selected graph; none was an
already-selected duplicate. The preparation retains every correct and known
negative pair and, per video, at most 512 trajectory-hard plus 512
hash-uniform unmatched-background pairs. Each example records physical
geometry, rank/density, exact selected-graph topology, conflict count, local
trajectory residuals, no-link probability, and incumbent geometry.

The class burden is the governing result: useful links are only 0.0085% of the
full shell. Radius extension is therefore not an admissible standalone change.
Any continuation must first reject deployment background on fit-only data and
then pass the fixed 12-video validation partition.

Artifacts:

- `scripts/prepare_extended_radius_shell.py`
- `reports/g02/extended_radius_shell_prepare_v1/prepare.json`
- `tests/test_extended_radius_shell.py`

Execution was CPU-only (1,286.0 seconds). It emitted no prediction, used no
held-out/audit/test labels, and consumed no Kaggle submission or GPU budget.

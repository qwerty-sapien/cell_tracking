# Extended-radius shell appearance-support screen

Decision: **PARK. Do not replay the r10–12 µm shell on CV57.**

Geometry alone could not isolate a positive held-out shell prefix without
sampled unmatched background. The stronger follow-up therefore added the same
exact appearance mechanism that transferred in the banked `0.881522` model:
zero-mean, unit-norm NCC over centroid-aligned `16×32×32` raw patches, including
the exact frozen selected-graph incumbent margins.

The augmentation covered all 117 fit-only videos and preserved per-video
identity, graph edge count, topology, labels, pairs, and checksums. It took
1,872.5 CPU-seconds and did not emit predictions.

## Frozen selection contract

The rule grid was selected on the 105 fit-train videos only:

- score = `pair_ncc + w × pair_minus_maximum_conflict_ncc`;
- `w ∈ {0, 0.25, 0.5, 1, 2, 4}`;
- topology policy ∈ `{all, free, conflict}`;
- at least 25 correct training links;
- sampled precision at least 0.95;
- zero sampled unmatched-background acceptances.

The fixed 12-video validation partition was used once after selection and
could not change the rule.

## Result

Appearance ranks links better than geometry (NCC-margin AUC 0.7907 train and
0.8287 validation), but it does not supply deployment-grade abstention:

| split | correct | known negative | sampled unknown | precision | utility |
|---|---:|---:|---:|---:|---:|
| fit-train, best supported rule | 25 | 8 | 141 | 0.1437 | -273 |
| fixed fit-validation | 7 | 1 | 23 | 0.2258 | -41 |

The selected diagnostic rule used `w=0.5`, all topology classes, threshold
`1.2961942`. No rule in the preregistered grid was train-admissible. The
validation result independently confirms that the failure is not a scalar
threshold accident.

The 456 reachable links remain real headroom, but evaluating or adding the
full 5.34M shell would be both expensive and unsafe. Revisit only with a
materially stronger learned no-link/background representation—not another
NCC/geometry threshold—and require a new fit-only support gate before CV57.

Artifacts:

- `scripts/augment_extended_radius_shell_appearance.py`
- `scripts/audit_extended_radius_shell_appearance_support.py`
- `reports/g02/extended_radius_shell_appearance_v1/report.json`
- `reports/g02/extended_radius_shell_appearance_support_v1.json`
- focused tests under `tests/`

No Kaggle submission, GPU launch, RunPod action, or public calibration claim
was made.

# G-03 Phase-2 h000 terminal report

Date: 2026-08-03  
Status: **COMPLETE / EXACT ZERO-FEATURE PARITY / BRANCH B**

## Result

The immutable zero-hour checkpoint is
`h000/h000.pt`, SHA-256
`603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95`.
Its final `Linear(48,12)` weight and bias contain zero nonzero elements, and
the adapter emits twelve bitwise float32 zeros on every finite crop. The
checkpoint records zero optimizer steps and zero scientific GPU-hours.

The canonical graph seal re-decoded all 36 frozen CUDA-produced zero-feature
association banks under the unchanged R1 compiler. All `36/36` fragments are
byte-identical to the established zero-feature fragments; both aggregate
fragment SHA-256 values are
`5dfe083f13a4ff6f671918fa5ff6b1cee84c24b37ba699663fe9f1115b3d9c0a`.
The frozen held-fit36 scorer then returned:

| metric | corrected h000 |
|---|---:|
| adjusted graph score | `0.7630876414338513` |
| raw edge Jaccard | `0.7581383935874366` |
| node recall | `0.9747454252716499` |
| edge TP / FP / FN | `9269 / 1212 / 1745` |
| division TP / FP / FN / Jaccard | `0 / 0 / 9 / 0` |

The parity error against the same-population zero-feature control is exactly
`0.0` (tolerance `1e-12`).

## Decoding-path diagnosis

A fresh local CPU Trackastra replay was deliberately stopped before scoring
after one of its first two videos differed by one selected edge. The adapter
output was still bitwise zero, candidate nodes/edges and distances were exact,
and the maximum association-probability drift was only `2.0862e-5`. For edge
`1865 -> 2099`, the frozen CUDA bank contained `0.3499999046` while local CPU
inference produced `0.3500036001`, straddling the fixed `0.35` threshold.

This is a Trackastra CPU/CUDA floating-point forward difference before a
discontinuous decoder—not crop, adapter, population, compiler, or scorer
drift. The score-blind runtime-equivalence amendment therefore used the
already-frozen CUDA association bank; it did not run a GPU or change a graph,
threshold, scorer, or metric.

## Controlled interpretation and allocation

The cited native `0.829194058080` is a CV57 result and is not a held-fit36
baseline. On the matched held-fit36 population, historical random-adapter R1
was `0.7637338804169099`, only `+0.0006462389830586` above zero-init h000.
Consequently, the claimed `~0.066` random-initialization penalty is not
supported by a controlled comparison. Zero initialization remains the correct
interface-preserving design, but it does not repair the low held-fit36
zero-feature score.

H000 is `-0.10521159747674869` below the allocation floor and
`-0.11843442852614872` below the best known score, so it is Branch B. The
effect-size dossier fails the required `<=30 T4-h` comparability and calibrated
paths to `0.915` and `0.945`; allocation is therefore `0/30 T4-h`.

## Immutable evidence

- `h000/PUBLICATION_RECEIPT.json`: checkpoint identity and zero-hour state.
- `h000/PREFLIGHT.json`: exact model, archive, scorer, and historical R1 pins.
- `H000_DECODE_PATH_DIAGNOSIS.md`: score-blind CPU/CUDA diagnosis.
- `H000_RUNTIME_EQUIVALENCE_AMENDMENT.json`: preregistered canonical runtime.
- `h000/canonical/CANONICAL_INFERENCE_SEAL.json`: 36/36 graph parity before score.
- `h000/CANONICAL_SCORE.json`: corrected graph score and allocation decision.
- `h000/SCORE_SERIALIZATION_RECOVERY.json`: serialization-only recovery receipt.

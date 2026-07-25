# 6bba-only crop-linker nested OOF confirmation

Decision: **PARK**. No CV57 replay.

The 6bba-only mechanism was assigned a distinct candidate identity because it
was discovered after the all-family validation12 result. Confirmation therefore
excluded validation12 completely. The eligible pool was the 69 6bba videos from
the old fit93 and calibration12 source partitions, containing 4,196 supervised
crop pairs.

The fixed five-fold design assigned videos by
`uint64_be(sha256(identity)[0:8]) mod 5`. For outer fold `k`, fold `k+1` alone
selected temperature and threshold, the remaining three folds trained the
model, and fold `k` was scored once. Pair NCC used the identical calibration
fold, threshold rule, and per-video cap.

| outer fold | crop correct / harmful | crop precision | control utility | crop utility | gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 520 / 7 | 0.9867 | 328 | 506 | +178 |
| 1 | 515 / 31 | 0.9432 | 305 | 453 | +148 |
| 2 | 570 / 67 | 0.8948 | 331 | 436 | +105 |
| 3 | 230 / 1 | 0.9957 | 217 | 228 | +11 |
| 4 | 452 / 84 | 0.8433 | 255 | 284 | +29 |
| **OOF** | **2,287 / 190** | **0.9233** | **1,436** | **1,907** | **+471** |

The candidate passed the utility, support, calibration, and nonnegative-fold
gates:

- accepted support increased from 1,910 to 2,477;
- all five fold utility gains were nonnegative;
- temperature scaling improved both NLL and Brier in all five folds;
- every candidate and control calibration fold had an admissible threshold.

It failed the governing precision gates. Aggregate precision was 0.9233 versus
the required 0.95, and folds 2 and 4 were below the required 0.93. The
identically nested pair-NCC control was also weak at 0.9173. This confirms that
the crop representation adds useful recall and net utility but does not make
the high-confidence operating threshold transferable across 6bba videos.

The guarded CPU run completed in 588.56 seconds, peaked at 2.158 GiB RSS,
preserved the 70% free-memory reserve, and caused zero swap growth. It accessed
neither validation12 examples nor CV57. The preregistered failure action is to
PARK the 6bba crop-linker branch; threshold retuning and CV57 confirmation are
not admissible for this identity.

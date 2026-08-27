# Compact point-context dry-data audit

Data decision: **PASS**. Exact pilot decision: **PARK**.

The literature/mechanism candidate passed the locked pre-training manifest
gate. The dry audit then traversed all 117 fit-only videos (105 train, 12
validation), loaded only the existing prepared proposal/context tables and
fit117 caches, and confirmed zero overlap with CV57.

| quantity | result |
|---|---:|
| proposal tokens | 25,289 |
| adjacent-transition context windows | 8,382 |
| maximum tokens in a window | 28 |
| 512-token-cap violations | 0 |
| positive proposals retained by the cap | 4,007 / 4,007 |
| parental groups | 25,067 |
| maximum candidate parents per target | 3 |
| targets with multiple positive parents | 0 |
| no-parent groups | 21,060 |

The two-layer/four-head fp32 attention-score tensor is only 8 MiB even at the
declared 512-token ceiling; the observed 28-token maximum is much smaller.
This validates the data shape, explicit no-parent supervision, and memory
premise. It does not validate model quality.

The guarded local fit-only fast-fail subsequently completed all 500 optimizer
steps. Validation payloads and caches were kept physically unopened until
training and train scoring finished.

| metric | pair-NCC control | point-context |
|---|---:|---:|
| validation correct / harmful | 185 / 14 | 212 / 8 |
| validation precision | 0.9296 | **0.9636** |
| conservative utility | 157 | **196** |
| utility gain | — | **+39** |

The family breakdown determined the decision:

| family | point-context correct / harmful | precision | utility gain |
|---|---:|---:|---:|
| 44b6 | 30 / 6 | **0.8333** | +5 |
| 6bba | 182 / 2 | **0.9891** | +34 |

The exact model PARKs because 44b6 missed the preregistered 0.93 pilot
precision floor. The run completed in 52.91 seconds (57.81 seconds including
the guard), peaked at 1.218 GiB RSS, held the 71% free-memory reserve, and
caused zero swap growth.

This is useful mechanism evidence but not permission for surrogate-only OOF or
CV57. Earlier 6bba temporal work showed that high proposal precision can still
regress after graph application. Any new family-scoped point-context identity
must therefore be evaluated with nested held-fit predictions **and actual
fit-only graph reconstruction/scoring** before CV57 is considered.

CV57, Kaggle, GPU, and RunPod remained closed.

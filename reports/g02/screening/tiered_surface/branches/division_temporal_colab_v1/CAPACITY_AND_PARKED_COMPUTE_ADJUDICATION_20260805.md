# Capacity and parked-compute adjudication

Decision: the Colab path is prepared, but it is **not yet proven necessary** for the new division detector. The authenticated Kaggle snapshot leaves `24.73 - 7.67 = 17.06 T4-h` unreserved after V32. The exact longer-context architecture, number of optimizer steps, and validation schedule are intentionally pending V32's residual-error taxonomy, so a numerical total-training projection would currently be invented rather than measured.

## Current hours and split

| Quantity | Hours | Meaning |
|---|---:|---|
| Kaggle remaining | 24.73 | authenticated snapshot supplied by the owner |
| V32 reservation | 7.67 | immutable active lane; untouched |
| Kaggle unreserved after V32 | 17.06 | capacity available to a scientifically unlocked node |
| Division lane committed on Kaggle | 0.00 | no objective or timing receipt yet |
| Division lane committed on Colab | 0.00 in every GPU bucket | no Colab session launched |
| Division lane committed on RunPod | 0.00 | no per-run approval and no launch |
| Measured total-training projection | **not established** | frozen architecture/schedule and timing pilot do not yet exist |

The capacity decision is preregistered as follows. After the taxonomy freezes the model, time the identical training workload for 30–60 minutes on an authenticated T4. Project

`training seconds = observed seconds/optimizer-step × frozen optimizer-steps × 1.35 + measured validation/checkpoint seconds`.

Freeze that receipt before the full run. If it is `<=17.06 T4-h`, allocate **Kaggle 100%, Colab 0%**. If it exceeds `17.06 T4-h`, allocate Kaggle up to the available T4 capacity and benchmark the identical workload on the actually assigned Colab or RunPod device before allocating a remainder. L4, A100, and T4 wall-hours remain separate; they are not “T4-equivalent” without a same-workload throughput measurement.

This makes the honest answer to “is Colab actually needed?” **not presently**. It is resilience/capacity insurance, not a justified compute allocation yet.

## Colab operating boundary

The notebook is exactly three manually executed cells: verified setup/resume, training, and final checkpoint/log publication. Free-tier sessions target 4–6 hours with immutable state publication every 600 seconds; a session interruption loses at most the uncommitted part of that cadence if the frozen entrypoint obeys its optimizer-step callback contract. There is no browser driver, activity injection, session cycling, or account multiplexing. Google's FAQ says resource availability and maximum lifetime vary; free notebooks can run for at most roughly 12 hours depending on availability/usage, while Pro+ supports up to 24 hours of continuous execution when sufficient compute units are available. It also distinguishes idle termination from a cell that is executing code. Sources: [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) and [Google Colab Terms](https://research.google.com/colaboratory/tos_v5.html).

Each session writes `SESSION_START` immediately with the assigned GPU model and full hardware fingerprint. `SESSION_END` records wall-hours and a cumulative subtotal only within that exact GPU-model bucket. If a disconnect prevents the end row, the next manual start reconciles it as a labelled conservative upper bound: last attributable checkpoint plus 600 seconds, or the six-hour session ceiling if none exists. The ledger schema rejects Kaggle hours and rejects any cross-device `total_colab_hours` field.

## RunPod planning envelope

The owner's approximate SGD 20 envelope is recorded for planning, not as a launch approval. Using a transparent planning assumption of SGD 1.29/USD gives about USD 15.50. RunPod currently lists A100 PCIe 80 GB at USD 1.39/h and A100 SXM 80 GB at USD 1.49/h, corresponding to at most about 11.15 or 10.41 instance-hours before storage. Sources: [RunPod GPU pricing](https://www.runpod.io/pricing) and the [MAS exchange-rate collection](https://data.gov.sg/collections/1353/view).

The locked campaign record still carries a USD 10 total RunPod cap. “Do consider SGD 20” is treated as a planning envelope, not a silent cap amendment; the currently governed USD 10 ceiling would buy about 7.19 PCIe or 6.71 SXM hours before storage. Spending above it needs an explicit locked-plan amendment as well as explicit approval for the particular run.

One 4–6 hour A100 shard is economically plausible even under that existing cap: USD 5.56–8.34 on PCIe or USD 5.96–8.94 on SXM before storage. It becomes useful only if the frozen same-workload calibration shows a real Kaggle-capacity gap; A100 hours cannot be substituted numerically for T4 hours from the price table alone.

No RunPod job is launched. The repository still requires explicit approval for each RunPod run, plus an identical-workload timing calibration. The SGD 20 indication does not silently waive the locked scientific prerequisites.

## Parked-compute audit

| Node | Adjudication | Why |
|---|---|---|
| Longer-context parent-event detector | Infrastructure ready; scientific objective blocked | New observation-distinct architecture. Taxonomy and timing receipt are still required. |
| DPRC-O0 DINOv2 registers patch-cycle | Preparation may resume; launch remains blocked | Its distinct `<=4 h` preflight fits current Kaggle headroom, so Colab is not needed. The fresh manifest, exact crop/channel contract, and `<=5 s` local preflight are incomplete. |
| G04 crop-wrfeat | `PARK-COST` unchanged | Measured routes are 195.59 and 659.856 T4-h. Fitting the cheaper/expensive route into the A100 budget would require at least 17.54×/59.16× speedup before storage, with no such same-workload measurement. The locked reopening condition is a named architecture removing tri-orientation per-detection redundancy, not merely more compute. |
| Full organizer TemporalUNet from scratch | `DEAD/ARCHIVE` unchanged | The locked DAG records ~478 T4-h and a masked-FP failure. Compute availability does not reopen a DEAD mechanism. |
| DINOv2 R5 identical rerun | `FAIL_PARK_C` unchanged | This was a scientific separation failure, not a capacity failure. |

Thus “re-alive” applies only to scientifically observation-distinct preparation. It does not authorize identical reruns or turn a larger compute pool into a new mechanism.

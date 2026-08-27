# Calibrated 3-D crop-linker data gate

Data decision: **PASS**. Exact pilot decision: **PARK**.

The candidate passed the locked literature/resource manifest and the full
fit117 label/crop audit. Detection-to-GT assignment mirrors the official
time-aware 7 µm physical matching. Labels are deliberately partial:

| label | count | training use |
|---|---:|---|
| mapped pair is a GT edge | 4,008 | positive |
| both nodes map, pair is not a GT edge | 1,885 | guaranteed negative |
| either node is unmatched | 19,396 | **masked** |

The existing deployment-background sample is not a negative set: 19,396 of
19,398 unique background pairs are uncertain, one is positive, and one is a
guaranteed negative. Treating those unmatched rows as negatives would
manufacture supervision from sparse labels.

The stable, family-stratified split is:

| role | videos | 44b6 pos / neg / masked | 6bba pos / neg / masked |
|---|---:|---:|---:|
| model fit | 93 | 655 / 459 / 7,644 | 2,654 / 1,053 / 9,566 |
| temperature calibration | 12 | 111 / 160 / 1,024 | 340 / 149 / 1,162 |
| final fit validation | 12 | 45 / 29 / 0 | 203 / 35 / 0 |

There is zero CV57 overlap. Calibration identities are selected from train105
by stable SHA-256 ordering before fitting; validation12 is not used to fit the
model or temperature.

Sixty-four real source/target crop pairs passed deterministic extraction,
finite/non-flat, and `16×32×32` shape checks at 16.30 pairs/s. Raw fp32 crop
input is 8 MiB per batch of 64. The preregistered model is a shared
8/16/32-channel 3-D encoder, directional embedding comparison, the frozen
48-feature geometry/motion context, and a binary link/no-link head. A positive
scalar temperature is fit only on calibration12 and must improve both NLL and
Brier score.

The pilot contract is hash-pinned in
`reports/g02/crop_linker_pilot_contract_v1.json`. The materialized dataset
contains 5,893 supervised pairs (4,008 positive and 1,885 guaranteed negative)
in 117 hash-pinned per-video files. It occupies 590,520,183 bytes and was built
without loading all raw volumes simultaneously.

## Guarded pilot result

The 30,057-parameter model completed its preregistered 500 CPU steps in 120.70
seconds. The outer guard completed in 126.52 seconds, peaked at 1.665 GiB RSS,
kept the 74% free-memory reserve unchanged, and caused **zero swap growth**.
Validation examples were not opened until after all 500 fit steps and
temperature calibration.

Temperature scaling improved both registered metrics:

| metric | raw | calibrated |
|---|---:|---:|
| NLL | 0.475560 | 0.459841 |
| Brier | 0.148018 | 0.146770 |
| ECE | 0.048085 | 0.035588 |

At the calibration-selected global threshold, validation utility improved from
91 for pair NCC to 124 for the crop model, a **+33** gain. Overall precision was
128/130 = **0.9846**. The gain was positive in both families:

| family | crop correct / harmful | precision | utility gain vs control |
|---|---:|---:|---:|
| 44b6 | 16 / 2 | 0.8889 | +6 |
| 6bba | 112 / 0 | 1.0000 | +27 |

The exact model PARKs because the preregistered per-family pilot precision floor
was 0.93 and 44b6 missed it. No grouped OOF or CV57 replay is permitted for this
model identity.

## Post-hoc mechanism audit

A calibration-only family-threshold audit is explicitly marked
`EXPLORATORY-NONCONFIRMATORY`. No 44b6 threshold could accept at least 20
calibration links at 0.95 precision, so a simple family threshold does not
repair the exact model. The same audit found a strong 6bba-only signal
(validation 180/182, precision 0.9890, utility 176 versus control 87), but that
result was discovered after validation unsealing. It may motivate a separately
preregistered 6bba mechanism with grouped OOF; it cannot promote this model or
unlock CV57.

CV57, Kaggle, GPU, and RunPod remained closed throughout.

## Separate 6bba-only confirmation

Because the family mechanism was selected after validation12 unsealing, it was
given a new identity and tested using nested grouped OOF on the 69 eligible
6bba videos from the old fit and calibration pools. Validation12 remained
closed. The candidate improved conservative OOF utility by **+471** over the
identically nested pair-NCC control and had positive utility gain in every
fold, but precision was only **0.9233** overall. Two outer folds fell below
0.93 (0.8948 and 0.8433). The 6bba-only branch therefore PARKs without CV57.

This stronger confirmation replaces the earlier post-hoc optimism: crop
appearance adds association recall and net utility, but the operating threshold
still fails to transfer across videos.

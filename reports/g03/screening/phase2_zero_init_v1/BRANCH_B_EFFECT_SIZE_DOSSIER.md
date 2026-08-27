# G-03 Phase-2 Branch-B effect-size dossier

Date: 2026-08-03  
Candidate: `G03-TA-CROPADAPTER-E2E-V1`  
Decision: whether a maximum allocation of 30 cumulative T4-hours is justified  
Verdict: **FAIL — 0/30 T4-hours funded**

## Owner-policy answer

The audited primary literature does not make the required numerical case.
Appearance features can improve cell linking, but the closest interface-level
study reports only `+0.0022` and `+0.0039` LNK over a retrained Trackastra
control on two 2-D HeLa datasets. It reports a single A100 GPU and eight seeded
training runs, but not the number of training GPU-hours or an elapsed-time
curve. It therefore cannot establish that this different, randomly initialized
3-D crop adapter reaches the G-03 reference within 30 T4-hours.

The campaign runway from the named best reference, and the larger gains this
candidate itself would need from corrected h000, are:

| Requirement | From campaign best | From corrected h000 |
|---|---:|---:|
| Comparability (`0.881522069960`) | already at reference | `+0.118434428526` |
| Path to `0.915` | `+0.033477930040` | `+0.151912358566` |
| Path to `0.945` | `+0.063477930040` | `+0.181912358566` |

No paper supplies a calibration from CTC LNK/TRA/AOGM to the G-03 adjusted
graph score. Ratios between those unlike metrics are not evidence of
transfer. As a magnitude warning only, the two best MIDL gains over retrained
Trackastra are just `1.86%` and `3.29%` of this candidate's numeric gap to
comparability, `1.45%` and `2.57%` of its gap to `0.915`, and `1.21%` and
`2.14%` of its gap to `0.945`.

The corrected same-protocol h000 result and allocation reclassification come
from the immutable score receipt after parity was sealed. This result also
invalidates the proposed `~0.066` initialization-loss arithmetic: `0.829194`
was measured on CV57, not held-fit36. On the common held-fit36 population,
zero-init h000 is `0.000646238983` below historical random-adapter R1
(`0.763733880417`), rather than `~0.066` above it.

| Corrected h000 field | Value |
|---|---:|
| held-fit36 adjusted graph score | `0.763087641434` |
| delta to `0.881522069960` | `-0.118434428526` |
| gain required to `0.915` | `+0.151912358566` |
| gain required to `0.945` | `+0.181912358566` |
| allocation branch | `B` |

## Closest intervention: dense embeddings inside Trackastra

Dalinghaus, Archit, and Pape integrate SAM, microSAM, or MAESTER cell
embeddings into Trackastra and report LNK on `Fluo-N2DL-HeLa` and
`DIC-C2DH-HeLa`. Both are 2-D Cell Tracking Challenge datasets. Sequence 02 is
used for training and sequence 01 for evaluation. Table 1 says the uncertainty
term is the **variance** from eight training runs with different seeds; it is
not a standard deviation or confidence interval.

### Source-native effect sizes

| Dataset | Intervention | LNK | Delta vs retrained Trackastra |
|---|---|---:|---:|
| Fluo-N2DL-HeLa | retrained Trackastra | `0.9922 +/- 3.19e-07 variance` | control |
| Fluo-N2DL-HeLa | SAM mask mean | `0.9923 +/- 2.4e-07` | `+0.0001` |
| Fluo-N2DL-HeLa | SAM centroid | `0.9924 +/- 4.6e-07` | `+0.0002` |
| Fluo-N2DL-HeLa | microSAM mask mean | `0.9930 +/- 2.1e-07` | `+0.0008` |
| Fluo-N2DL-HeLa | microSAM centroid | `0.9930 +/- 2.0e-07` | `+0.0008` |
| Fluo-N2DL-HeLa | MAESTER mask mean | `0.9935 +/- 4.2e-07` | `+0.0013` |
| Fluo-N2DL-HeLa | MAESTER centroid | `0.9944 +/- 4.2e-07` | **`+0.0022`** |
| DIC-C2DH-HeLa | retrained Trackastra | `0.9859 +/- 1.78e-05 variance` | control |
| DIC-C2DH-HeLa | SAM mask mean | `0.9890 +/- 5.9e-06` | `+0.0031` |
| DIC-C2DH-HeLa | SAM centroid | `0.9884 +/- 1.5e-05` | `+0.0025` |
| DIC-C2DH-HeLa | microSAM mask mean | `0.9898 +/- 1.8e-06` | **`+0.0039`** |
| DIC-C2DH-HeLa | microSAM centroid | `0.9869 +/- 2.6e-06` | `+0.0010` |
| DIC-C2DH-HeLa | MAESTER mask mean | `0.9857 +/- 4.1e-06` | `-0.0002` |
| DIC-C2DH-HeLa | MAESTER centroid | `0.9805 +/- 1.7e-05` | `-0.0054` |

Against the paper's un-retrained public checkpoint, the best gains are
`0.9944 - 0.9904 = +0.0040` and `0.9898 - 0.9815 = +0.0083`. Those are not
the controlled fine-tuning effects; the retrained comparisons above are the
relevant ones.

### Training-budget audit

| Disclosed by the primary source | Not disclosed |
|---|---|
| one NVIDIA A100 for training | elapsed time per run |
| eight independent seeded training runs | GPU-hours per run or total GPU-hours |
| full checkpoint persisted after every epoch | the paper's epoch count in the audited report |
| CPU inference over saved checkpoints | A100-to-T4 normalization |
| Slurm reproduction files in the official repository | any compute-to-LNK trajectory |

This corrects the 2026-08-01 preliminary R2 statement that the hardware was
missing: the supplement does disclose a single A100. The allocation failure
remains because duration, GPU-hours, and a compute-to-quality curve are absent.
Moreover, the evaluated embeddings are pretrained 2-D models; they are not a
randomly initialized 22,684-parameter 3-D crop encoder trained through the
G-03 association objective.

Primary source: [MIDL 2026 OpenReview record and paper](https://openreview.net/forum?id=4OLrHi9ROr),
[official reproduction repository](https://github.com/MIDL26-Short-Tracking/trackastra_y).

## Original Trackastra feature ablation

The Trackastra paper supplies a clean same-decoder mechanism result on its
2-D Bacteria benchmark (six test videos; three runs per model):

| Features / decoder | TRA | AOGM | FP edges | FN edges | FP divisions | FN divisions |
|---|---:|---:|---:|---:|---:|---:|
| positions only + ILP | `0.995` | `136` | `39` | `39` | `20` | `29` |
| shallow features + ILP | `0.999` | `23` | `7` | `8` | `2` | `3` |
| effect | **`+0.004`** | **`-113` (`-83.1%`)** | `-32` (`-82.1%`) | `-31` (`-79.5%`) | `-18` (`-90.0%`) | `-26` (`-89.7%`) |

This is real evidence that intensity, area, and inertia features can reduce
linking and division errors. It is still 2-D, uses engineered shallow features,
and does not isolate a pretrained zero-feature-to-learned-crop transition.
The paper states that a prototypical 2-D model can train on one GPU such as an
RTX 4090, but reports no elapsed hours, epoch/step count, 3-D adapter run, or
compute-to-quality curve.

Primary source: [Trackastra, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf).

## Crop-derived representation evidence with a different interface

Attrackt reports means over ten runs on two 2-D datasets. Its autoencoder
representations are used as graph costs or frozen targets for an unsupervised
association objective; they are not injected as Trackastra `wrfeat` channels.

| Dataset | Method | TRA | Delta TRA vs position | AOGM | Delta AOGM | Division F1 | Delta Division F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Bacteria | position only | `0.972` | control | `712.667` | control | `0.585` | control |
| Bacteria | autoencoder embeddings | `0.988` | `+0.016` | `285.375` | `-427.292` (`-60.0%`) | `0.724` | `+0.139` |
| Bacteria | unsupervised associations | `0.993` | `+0.021` | `175.017` | `-537.650` (`-75.4%`) | `0.831` | `+0.246` |
| HeLa | position only | `0.999` | control | `352.500` | control | `0.781` | control |
| HeLa | autoencoder embeddings | `0.998` | `-0.001` | `642.750` | `+290.250` (worse) | `0.800` | `+0.019` |
| HeLa | unsupervised associations | `0.999` | `0.000` | `231.350` | `-121.150` (`-34.4%`) | `0.859` | `+0.078` |

The unsupervised pretraining schedule is 600 epochs and the interactive
fine-tuning configuration uses LoRA rank 32. Hardware, elapsed duration,
GPU-hours, and a T4 normalization are not reported. The positive Bacteria
effect and the mixed HeLa effect reinforce that transfer is dataset-dependent.

Primary source: [Attrackt, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/BIC/papers/Lalit_An_Investigation_of_Unsupervised_Cell_Tracking_and_Interactive_Fine-Tuning_ICCVW_2025_paper.pdf),
[supplement](https://openaccess.thecvf.com/content/ICCV2025W/BIC/supplemental/Lalit_An_Investigation_of_ICCVW_2025_supplemental.pdf).

## Why related 3-D embryo endpoints do not prove this path

Published Fluo-N3DH-CE endpoints show that high-quality 3-D embryo lineage
tracking is possible: the official general Trackastra submission has mean LNK
`0.9713595` across its two videos, and other specialized methods report high
TRA/LNK. These are endpoints under CTC data, detection, and metrics—not deltas
from this G-03 h000 state, not 30-hour learning curves, and not calibrated G-03
scores. They cannot be used as numerical bridges to `0.915` or `0.945`.

## Gate-by-gate adjudication

| Owner requirement | Evidence needed | Audited result | Gate |
|---|---|---|---|
| comparability with `0.881522069960` within <=30 T4-h | target-domain or calibrated transfer effect plus T4 compute curve | no source joins the architecture, 3-D domain, G-03 metric, and budget | **FAIL** |
| realistic path to `0.915` | credible target-calibrated evidence for candidate gain `+0.151912358566` (including `+0.033477930040` beyond the current best) | closest interface study is 2-D LNK, best controlled gains `+0.0022`/`+0.0039`, no metric calibration | **FAIL** |
| realistic path to `0.945` | further demonstrated candidate runway supporting `+0.181912358566` (`+0.063477930040` beyond the current best) | no calibrated runway; some feature variants are neutral or negative | **FAIL** |

The evidence supports a research hypothesis, not the required allocation
claim. Per owner policy, h005 and h030 are therefore **not funded**. The fixed
h005 contract remains useful if a future named mechanism or new primary/local
evidence reopens the gate, but no scientific GPU hour may be launched now.

## Audit boundary

- Primary papers, supplements, official repositories, and official benchmark
  outputs are used; secondary directional summaries are excluded.
- Missing training time is recorded as unknown, not inferred from GPU model.
- Source-native metrics are never relabeled as G-03 score effects.
- The historical R2 report remains immutable. This dossier supersedes its
  hardware-availability statement and applies the owner policy to corrected
  Phase-2 h000.

# G-03 Wave R2 — primary-source literature audit

**Date:** 2026-08-01  
**Lane:** R2 (read-only literature audit)  
**Decision governed:** whether the proposed Trackastra + real-`wrfeat` / learned 3-D crop-feature route satisfies the owner’s Branch-B evidence test for a 30 T4-hour allocation if R1 scores below the neighborhood floor.  
**Primary-source ledger:** [`R2_PRIMARY_SOURCE_TABLE.json`](R2_PRIMARY_SOURCE_TABLE.json)

## Executive verdict

**Branch-B dossier verdict: FAIL — the published evidence does not satisfy either required proposition.**

1. No primary source found reports that Trackastra fine-tuning with a new 3-D crop encoder reaches target-comparable quality within **<=30 T4-hours**. The Trackastra paper says prototypical 2-D models can be trained on one RTX 4090, but gives no elapsed time, epoch count, step count, or GPU-hours. The proposed G-03 crop adapter is not the image-feature mechanism evaluated in that paper.
2. No primary source establishes a realistic numerical path from the frozen G-03 reference `0.881522069960` to `0.915` and then `0.945` on the G-03 graph scorer. The best direct published feature-augmentation evidence is 2-D and improves CTC LNK by only `+0.0022` to `+0.0039` over a retrained Trackastra control, depending on dataset. LNK is not the G-03 score and cannot be converted into it.

This is a **conditional allocation result**, because R1 owns the actual G-03 zero-shot measurement:

- If R1 is **>= `0.868299238910600`** (or is a new best), the candidate enters Branch A. R2 does not disqualify Branch-A eligibility, but it rates the claim of *outsized* gain from the crop adapter as a **research bet**, not a literature-established expectation.
- If R1 is **< `0.868299238910600`**, the candidate enters Branch B and this dossier does **not** authorize the 30 T4-hours.

Published Trackastra results on the CTC 3-D C. elegans embryo benchmark are genuinely strong: the official generalizable submission has LNK `0.991203` and `0.951516` on the two videos (mean `0.9713595`). That is evidence of endpoint capability on a related embryo dataset. It is **not** evidence for headroom from G-03 fine-tuning, for the 30-hour cost claim, or for a conversion to the competition’s custom score.

## Frozen decision arithmetic

All target deltas below use the explicitly mandated reference point, not an unstated control:

| Quantity | Value |
|---|---:|
| `best_known_score` reference | `0.881522069960` |
| `neighborhood_floor = best_known_score * 0.985` | `0.868299238910600` |
| Absolute gain required for `0.915` | `+0.033477930040` |
| Absolute gain required for `0.945` | `+0.063477930040` |

The target score is G-03’s frozen graph-level objective, reported locally as adjusted edge Jaccard plus `0.1 * division Jaccard`. CTC TRA and CTC LNK use normalized AOGM-family edit costs. Numerically similar values are not “equivalent”; no primary source supplies a calibration between them. Consequently, a published TRA/LNK of `0.945`, `0.975`, or `0.979` cannot be substituted for a G-03 score of `0.945`.

Local reference: G-02’s terminal campaign report records `0.881522069960` as the best native score and Trackastra at `0.829194058080`, with division `TP/FP/FN/Jaccard = 0/0/42/0.0`. Those are local observations, not literature results, and R1’s real-`wrfeat` zero-shot measurement supersedes them for Branch classification.

## What published 3-D+t embryo trackers actually achieve

The closest public benchmark is **Fluo-N3DH-CE**, confocal fluorescence time-lapse imaging of a developing *C. elegans* embryo. The table reports source-native metrics, not fabricated G-03 equivalents.

| Method / source | Architecture and supervision | 3-D embryo result | Compute reporting | Honest transfer statement |
|---|---|---|---|---|
| **Trackastra-general**, official Cell Linking Benchmark | Association transformer; one generalizable model trained across 2-D and 3-D CTC data, using object positions and shallow mask/intensity features | Fluo-N3DH-CE LNK: video 01 `0.991203`, video 02 `0.951516`, mean `0.9713595` | Paper: one RTX 4090 is sufficient for prototypical 2-D training; no duration or GPU-hours | Strongest direct evidence that pretrained Trackastra can associate cells in one 3-D embryo domain. It gives neither a target-domain fine-tuning delta nor evidence about the new crop adapter. |
| **SAM2 linking / Chen et al.** | Zero-shot pretrained SAM2 patch propagation and memory-feature similarity over supplied masks; no labeled tracking training | Fluo-N3DH-CE blind-test LNK `0.982`; BIO `0.862` | RTX 4090; roughly one hour for a generic 200-frame, about-100-cell/frame sequence | Directly relevant 3-D embryo linking result and an observation-distinct future candidate. It is not Trackastra, does not use G-03's scorer, and supplies no conversion to `0.915`/`0.945`; it cannot rescue the present candidate's Branch-B dossier. |
| **Linajea / JAN-US** | 4-D U-Net candidate/movement network, 3-D ResNet-18 cell-state classifier, structured/ILP graph optimization | Fluo-N3DH-CE DET `0.981`, TRA `0.979` | 400k tracking iterations plus 60k cell-state iterations are reported; hardware and elapsed GPU-hours are not | Demonstrates high-quality 3-D embryo lineage tracking with dense spatiotemporal CNNs and global optimization. It is not a Trackastra fine-tune and includes a detection pipeline. |
| **ELEPHANT** | 3-D U-Net detection plus 3-D U-Net optical flow, iterative human-in-the-loop correction | Fluo-N3DH-CE SEG `0.631`, DET `0.979`, TRA `0.975` | GTX 1080 Ti stated; generic training up to 200 epochs and 10-epoch fine-tuning stated; total benchmark GPU-hours not reported | Shows that 3-D spatial appearance plus flow can support high TRA. Its detector/flow stack and interactive workflow are materially different from fixed-detection G-03. |
| **KTH-SE / Baxter**, as reevaluated by ELEPHANT | Established CTC embryo pipeline | Fluo-N3DH-CE SEG `0.662`, DET `0.959`, TRA `0.945` | No budget claim usable for G-03 in the audited source | A literal `0.945` exists on TRA, but it is not a `0.945` G-03 score. |

These papers establish source-native 3-D+t embryo TRA/LNK values through `0.982` (with other reported endpoints in the `0.945–0.979` range). They do **not** establish a metric equivalence, a fixed-detection G-03 result, or a <=30 T4-hour route.

## Direct evidence about Trackastra features and fine-tuning

### 1. Original Trackastra paper: shallow features matter, but the experiment is 2-D

Trackastra represents each detection with Fourier-encoded position plus shallow object features: mean intensity, area, and inertia tensor. Its default association transformer uses six encoder and six decoder layers, embedding dimension 256, four attention heads, temporal windows of six frames, at most 2,048 tokens per window, and batch size eight. Parental softmax enforces at most one parent while allowing multiple children; the loss upweights divisions.

On the 2-D Bacteria benchmark with ground-truth detections:

| Input / decoder | TRA | AOGM | FP edges | FN edges | FP divisions | FN divisions |
|---|---:|---:|---:|---:|---:|---:|
| Positions only + ILP | `0.995` | `136` | `39` | `39` | `20` | `29` |
| Shallow features + greedy | `0.999` | `36` | `10` | `12` | `5` | `3` |
| Shallow features + ILP | `0.999` | `23` | `7` | `8` | `2` | `3` |

This is strong mechanism evidence that object attributes can reduce graph errors. It does not isolate **fine-tuning from a pretrained zero-feature checkpoint**, does not use a learned 3-D crop encoder, and does not establish a target-domain custom-score delta.

The same paper also supplies negative evidence relevant to G-03:

- On DeepCell with predicted Caliban masks, Trackastra ILP has TRA `0.991` but division F1 `0.79`, below Caliban’s `0.92`. A near-ceiling aggregate association metric can coexist with materially weaker division recovery.
- The general model outperforms the specialized model out of domain on 2-D HeLa (AOGM `96` versus `190` with ILP). Domain fine-tuning is therefore not monotonically beneficial.
- Increasing beyond six encoder/decoder layers produced no notable improvement in the reported ablation. More transformer depth is not documented as an easy headroom source.

### 2. Dense learned embeddings: positive, small, 2-D-only deltas

The MIDL 2026 primary study *Dense Embeddings from Self-Supervision and Foundation Models Improve Cell Linking Performance* injects per-cell MAESTER, SAM, or microSAM embeddings into Trackastra. It keeps Trackastra’s training procedure and projects the embeddings into the object-feature space. Eight independent training runs are reported.

| Dataset | Retrained Trackastra LNK | Pretrained Trackastra LNK | Best feature model LNK | Gain vs retrained | Gain vs pretrained |
|---|---:|---:|---:|---:|---:|
| Fluo-N2DL-HeLa | `0.9922` | `0.9904` | MAESTER centroid `0.9944` | `+0.0022` | `+0.0040` |
| DIC-C2DH-HeLa | `0.9859` | `0.9815` | microSAM mask mean `0.9898` | `+0.0039` | `+0.0083` |

This is the closest published intervention to richer `wrfeat`. It is encouraging for mechanism, but not sufficient for Branch B:

- both datasets are 2-D HeLa, not 3-D embryo;
- the feature extractors are pretrained 2-D foundation/self-supervised models, not G-03’s randomly initialized <=1M-parameter 3-D crop CNN;
- one feature family (MAESTER on DIC-C2DH-HeLa) fails to improve over the retrained control, showing dataset dependence;
- no hardware, elapsed training time, or GPU-hours are reported; and
- its observed deltas are on LNK and are much smaller than G-03’s required absolute score gains of `+0.033477930040` and `+0.063477930040`. This magnitude comparison is descriptive only, because the metrics differ.

### 3. Attrackt: appearance-derived objectives and edge costs can help divisions, still only in 2-D

The ICCV Workshops 2025 Attrackt study evaluates crop-based autoencoder and re-identification embeddings as distances/costs for global-solver baselines. Its Trackastra-based Attrackt model does **not** inject those embeddings as `wrfeat` inputs: autoencoder embeddings instead provide frozen prediction targets for its unsupervised objective, with optional interactive LoRA tuning. Unsupervised pretraining runs 600 epochs, and LoRA uses rank 32 with a stopping rule tied to annotated nodes; hardware and elapsed time are not reported.

On 2-D Bacteria, TRA rises from `0.972` using positions to `0.988` with autoencoder features and `0.993` with unsupervised association training. On 2-D HeLa, where TRA is saturated at `0.998–0.999`, division F1 rises from `0.781` for positions to `0.859` with unsupervised associations and `0.906` with supervised associations. This supports the broader *direction* that appearance-derived supervision/costs can help division semantics. It is not interface-equivalent evidence for populating Trackastra `wrfeat`, and it still does not validate the G-03 architecture, 3-D transfer, target score, or <=30-hour limit.

### 4. Zero-shot SAM2 is a distinct 3-D embryo candidate, not Trackastra headroom

Chen et al. report zero-shot pretrained SAM2 linking on 13 blind CTC test datasets. On Fluo-N3DH-CE they report LNK `0.982` and BIO `0.862`; the method needs no labeled tracking training. The stated hardware is an RTX 4090, with roughly one hour for a generic 200-frame sequence containing about 100 cells per frame. The method uses patch propagation or SAM2 memory-feature similarity and detects mitosis from competing candidate similarities.

This closes an omission in the method survey and supports a new observation-distinct future candidate. It does **not** change the present allocation verdict: SAM2 linking is not Trackastra fine-tuning, LNK/BIO do not convert to the G-03 score, and no local zero-shot G-03 measurement yet exists. It must receive its own score and Branch classification before any GPU allocation.

### 5. Official Trackastra image-feature assets do not close the 3-D gap

The official pretrained registry lists the general `ctc` model for dimensions `[2, 3]`. Its SAM2-feature variant, `general_2d_w_SAM2_features`, is explicitly dimension `[2]` and described as beneficial particularly for bacteria. The related experimental Trackastra-and-Ultra repository demonstrates SAM2.1/CoTracker3 features on a 2-D example. Neither is published evidence for a 3-D crop adapter or embryo fine-tuning.

## Compute audit against the <=30 T4-hour requirement

| Evidence item | What is reported | What is missing for the policy |
|---|---|---|
| Trackastra ECCV 2024 | A prototypical 2-D model can train on one RTX 4090; architecture and batch size are specified | elapsed time, epochs/steps, data volume, GPU-hours, 3-D crop-encoder cost, and target-domain endpoint |
| Dense-embedding MIDL 2026 | eight training runs and embedding/projection design | hardware, duration, GPU-hours, any 3-D run |
| Attrackt ICCVW 2025 | 600 pretraining epochs; LoRA configuration and stopping rule | hardware, elapsed time, GPU-hours, 3-D data |
| Zero-shot SAM2 linking 2025 | RTX 4090; roughly one hour for a generic 200-frame/about-100-cell sequence; no labeled tracking training | G-03 runtime, G-03 score, and any metric conversion; distinct architecture rather than Trackastra headroom |
| ELEPHANT | GTX 1080 Ti; up to 200 epochs; 10-epoch local fine-tuning | full benchmark training duration/GPU-hours and T4-normalized cost |
| Linajea | 400k tracking and 60k state-classifier iterations | hardware, elapsed time, T4-normalized cost |

**Finding:** it may be operationally possible to run a bounded G-03 pilot inside 30 T4-hours, but the literature does not validate that it will reach comparability, `0.915`, or `0.945` in that budget. “Runnable within 30 hours” and “plausibly reaches the required endpoint within 30 hours” are different claims; only the first is weakly supported by the model’s modest architecture.

The local architecture dossier’s older `140–188` T4-hour campaign envelope is an engineering estimate, not a primary literature result. It is nevertheless direct local evidence that the full originally proposed campaign was not designed for a 30-hour ceiling.

## Branch-B gate adjudication

### Gate (i): <=30 T4-hours plausibly reaches comparability with the best

**FAIL.** No audited source connects:

`Trackastra checkpoint + random 3-D crop CNN + real wrfeat + <=30 T4-hours -> G-03 score >= 0.881522069960`.

The CTC result proves that a general Trackastra model can be excellent on Fluo-N3DH-CE, but it is an endpoint under another data/metric protocol, not a compute-to-quality curve. The richer-feature papers provide small 2-D deltas and omit GPU-hour accounting.

### Gate (ii-a): realistic published path to `0.915`

**FAIL.** The required G-03 gain is `+0.033477930040` over the named reference. No source reports a calibrated G-03 delta, a 3-D embryo crop-feature delta, or an association/division decomposition that can be transported to this objective.

### Gate (ii-b): realistic published path to `0.945` with further runway

**FAIL.** The required G-03 gain is `+0.063477930040`. Published embryo TRA/LNK endpoints show that excellent lineage graphs are feasible in related domains, but there is no metric conversion and no evidence that the selected adaptation supplies this gain. The best directly analogous feature study is already near the ceiling of LNK and exhibits only low-thousandth-scale improvements; it does not document continued runway.

## Evidence taxonomy and recommended policy use

| Claim | Classification | Basis |
|---|---|---|
| Run the official pretrained Trackastra checkpoint zero-shot and measure the frozen graph score | **Plan** | Existing code/checkpoint plus direct target-domain measurement; this is R1 |
| Appearance/object features can reduce association and division errors | **Established mechanism** | Original Trackastra ablation plus two direct 2-D feature studies |
| A small learned 3-D crop adapter will transfer that benefit to G-03 | **Research bet** | Biologically coherent and architecture-compatible, but no audited 3-D result |
| <=30 T4-hours will reach `0.881522069960` from a below-floor R1 score | **Unsupported numeric claim** | No compute-to-quality curve or target-domain result |
| The same route reaches `0.915`, then `0.945` | **Unsupported numeric claim** | No custom-score calibration or observed headroom of the required magnitude |

Accordingly:

- **Below floor:** do not allocate the 30 T4-hours to this candidate under Branch B on the present dossier.
- **At/above floor:** Branch A may permit a bounded pilot, but it should be justified by R1’s target-domain score and preregistered graph-level gates, not by treating CTC metrics as equivalents. Report all deltas against `0.881522069960` (or a subsequently updated best-known reference named before evaluation).
- Keep CPU post-hoc division decoding as an orthogonal lane. Literature shows division F1 can lag even when TRA is near ceiling; the G-03 score must therefore continue to report division TP/FP/FN/Jaccard separately from the edge term.

## Primary sources

1. Weigert et al., *Trackastra: Transformer-based cell tracking for live-cell microscopy*, ECCV 2024: [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf), [official repository](https://github.com/weigertlab/trackastra), [pretrained registry](https://github.com/weigertlab/trackastra/blob/main/trackastra/model/pretrained.json).
2. Cell Tracking Challenge: [EPFL-CH / Trackastra method page](https://celltrackingchallenge.net/participants/EPFL-CH/), [evaluation methodology](https://celltrackingchallenge.net/evaluation-methodology/), [official Cell Linking Benchmark workbook](https://public.celltrackingchallenge.net/documents/CellLinkingBenchmark.xlsx). Workbook SHA-256 as retrieved 2026-08-01: `35bc3f74e952d7b392888bab3d40324dcfd88e883e636adb43da424deb715f6f`.
3. Sugawara et al., *ELEPHANT: tracking cell lineages across space and time using deep learning*, eLife 2022: [article](https://elifesciences.org/articles/69380), [official code](https://github.com/elephant-track/elephant-server).
4. Pape et al., *Tracking by weakly-supervised learning and graph optimization for whole-embryo C. elegans lineages*, MICCAI 2022: [primary paper](https://openreview.net/pdf/ea74c7dceca4c8d964835335c0210451ebd90cea.pdf), [conference record](https://conferences.miccai.org/2022/papers/517-Paper0500.html).
5. Rother et al., *Dense Embeddings from Self-Supervision and Foundation Models Improve Cell Linking Performance*, MIDL 2026: [OpenReview record and paper](https://openreview.net/forum?id=4OLrHi9ROr).
6. Lalit et al., *An Investigation of Unsupervised Cell Tracking and Interactive Fine-Tuning*, ICCV Workshops 2025: [paper](https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Lalit_An_Investigation_of_Unsupervised_Cell_Tracking_and_Interactive_Fine-Tuning_ICCVW_2025_paper.html), [supplement](https://openaccess.thecvf.com/content/ICCV2025W/BIC/supplemental/Lalit_An_Investigation_of_ICCVW_2025_supplemental.pdf).
7. Achard et al., experimental Trackastra dense-feature integration: [Trackastra-et-Ultra primary repository](https://github.com/C-Achard/Trackastra-et-Ultra).
8. Chen et al., *Segment Anything for Cell Tracking*, MedAGI/MICCAI Workshops 2025: [arXiv primary paper](https://arxiv.org/abs/2509.09943), [proceedings record](https://doi.org/10.1007/978-3-032-07845-2_2), [official code](https://github.com/zhuchen96/sam4celltracking).

## Audit limitations

- “Primary source only” excludes leaderboard summaries, blog posts, and secondary reviews.
- Public embryo literature is dominated by CTC TRA/LNK/DET/SEG. No paper was found that evaluates the frozen G-03 score or supplies a valid conversion.
- Absence of reported GPU-hours is recorded as **unknown**, not estimated from hardware names.
- The official CTC workbook is mutable; its downloaded SHA-256 is pinned above for reproducibility.
- R1’s actual score was not available to R2 at the time of this report, so the policy classification is explicitly conditional.

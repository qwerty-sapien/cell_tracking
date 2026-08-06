# G-03 Phase 0 architecture dossier — end-to-end association training

Date: 2026-07-31  
Status: **architecture selected; no G-03 GPU work may start before the Phase 1 cross-session continuation proof**

## Decision

Select **G03-TA-CROPADAPTER-E2E-V1**: a pretrained Trackastra association transformer, fine-tuned end-to-end with a small trainable 3-D intensity-crop adapter. The adapter emits the twelve `wrfeat` values expected by the pinned Trackastra interface; the adapter, transformer, association head, optimizer, and scheduler all train together. It uses deterministic raw-intensity crops centered on existing competition detections and does not modify the detection front end.

This is a new named mechanism, not a re-screen of G-02's dead compact crop linker: the latter was a 30,057-parameter Siamese pair ranker trained from scratch, whereas this candidate conditions an already pretrained contextual association transformer on image information and retains parental association normalization.

The leading hypothesis is **partly confirmed and corrected**:

- Confirmed: every G-02 Trackastra screen used the `48 + 12` point interface with the twelve `wrfeat` channels filled with zero. It never tested a trainable, real crop-derived appearance signal at those channels.
- Rejected as worded: original Trackastra does not have a pretrained raw-crop encoder. Its published interface turns images and instance masks into shallow object features (including mean intensity); the paper specifically avoids dense-image processing. Calling real 3-D crops Trackastra's “designed appearance input” would therefore be false.
- Selected correction: retain the pretrained association transformer and add a separately identified crop adapter. The new, randomly initialized component and its cost are explicit rather than represented as pretrained Trackastra.

## Primary-source basis

Trackastra's ECCV paper defines direct association prediction over complete spatiotemporal detection windows, including divisions, with a parental softmax that allows one-to-many but not many-to-one associations. It documents shallow object features rather than a dense crop encoder. The authors' current code documents image/mask input and CTC-format training, and exposes a newer model using learned SAM2 features. This supports the association backbone and the value of richer features, but not the claim that the new 3-D crop encoder is pretrained.

- Gallusser & Weigert, *Trackastra: Transformer-based cell tracking for live-cell microscopy*, ECCV 2024: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>.
- Authors' implementation and training/interface documentation: <https://github.com/weigertlab/trackastra>.
- Dalinghaus, Archit & Pape, *Dense Embeddings from Self-Supervision and Foundation Models Improve Cell Linking Performance*, MIDL 2026 Short Paper: <https://openreview.net/forum?id=4OLrHi9ROr&noteId=jzL2NIVKk6>.

The organizer TemporalUNet3D is a primary-source alternative, not the chosen architecture. Its local training loop jointly learns full-volume detection and association; that is misaligned with G-03's association-only target and the existing fixed detections.

## Candidate comparison and T4-h costing

All ranges are campaign-grade envelopes, not quality forecasts. They include checkpoint handoffs and graph reads where stated. A 12-hour Kaggle session is a session limit, not a license to spend the entire envelope.

| candidate | science / interface | cost basis | G-03 envelope | decision |
|---|---|---|---:|---|
| **G03-TA-CROPADAPTER-E2E-V1** | pretrained contextual Trackastra + trainable <=1M-parameter 3-D crop adapter projected into formerly zero `wrfeat`; full association fine-tune | G-02 full Trackastra cache took 2.0348 T4-h and one warm-checkpoint estimate was 4.05–5.96 T4-h. Crop adapter is conservatively charged a 1.5x feature/backprop factor until measured. | 16–20 proof + 90–120 gated training + 28–42 confirmation/transfer + 6 package = **140–188 T4-h** | **SELECT.** Under 200 T4-h; the only candidate that tests image information absent from the G-02 interface while retaining pretrained association context. |
| organizer TemporalUNet3D + node transformer | jointly changes full-volume detection and association; no pretrained association transformer | historical rate: 478/300 = 1.593 T4-h/epoch. A 3-epoch useful but nonconverged checkpoint costs about 4.8 T4-h; 7 epochs about 11.2 T4-h before handoffs. | **6–15 T4-h** for one 3–7-epoch comparator; about 120 T4-h for 75 epochs before evaluation reserve | **REJECT as primary.** Existing organizer checkpoint precision was 3.96%, and the model entangles detection with association. |
| compact 3-D association transformer from scratch | crop/geometry transformer with no pretrained association representation | G-02's 30,057-parameter crop linker is a lower-bound implementation anchor: 500 CPU steps in 120.696 s, but failed both 44b6 and 6bba precision gates. | **20–80 T4-h** to reach a fair graph-gated trajectory; a credible seed/architecture sweep exceeds the scientific budget. | **REJECT.** A different parameterization of G-02's dead from-scratch ranker-without-pretraining mechanism. |

The selected envelope leaves at least 67 T4-h inside the approximately 255 T4-h campaign budget and leaves the charter's 25 T4-h reserve unallocated.

## Architecture boundary

1. For each existing detection, extract an immutable normalized `(3, 16, 32, 32)` `t,z,y,x` intensity crop with fixed boundary padding.
2. A small 3-D CNN with global pooling projects that crop to the twelve `wrfeat` values. It is explicitly non-pretrained and all its state belongs to the checkpoint.
3. The pinned Trackastra transformer preserves positional/time features, the association matrix and parental-softmax loss, and is fully fine-tuned.
4. The existing association-to-graph compiler and byte-exact APNCC transport are unchanged. No threshold search, proposal metric, or loss/AUC may select a checkpoint.

Phase 1 must authenticate raw-volume/crop handling and measure T4 VRAM, examples/sec, checkpoint size, and session handoff. A missing raw source, a 12-channel contract drift, or a measured worst-case campaign projection above 200 T4-h invalidates this selection before scientific training begins.

## Required next preregistration

This dossier is not a training preregistration. The Phase 1 contract must fix:

- the 105-video fit scope; 44b6 held-fit groups are evaluation-only and audit groups are untouched;
- crop normalization/shape/padding, adapter architecture and parameter census;
- fixed cumulative training intervals at 30, 60 and 90 T4-h; every graph read uses a checkpoint fixed before its held-fit result is read;
- owner-amended stop lines (2026-08-01): best 44b6 adjusted graph delta `>=+0.010` at 30 T4-h and `>=+0.035` at 90 T4-h. The former 60-hour `+0.002` threshold is superseded and no replacement 60-hour gate is inferred; it requires an explicit later amendment; and
- the Phase 1 hard stop: a tiny model must show new-kernel, bit-identical continuation with atomic handoff and SHA-verified full state before this model receives GPU time.

The carried measured surface—DELETE AUC/retention `0.85/0.020394` and occupied-SWAP `0.90/0.067910`—remains diagnostic only. It is not a proposal metric or a substitute for the held-fit graph objective.

## Cross-session checkpoint/resume protocol

The Phase 1 proof uses the same protocol that a long training trajectory will use. A handoff is valid only if all of the following are true.

1. **Immutable run identity.** `run_manifest.json` records the code tree hash, exact dependency lock, CUDA/PyTorch/cudnn settings, model/config hash, 105-video data manifest, crop contract, seed, deterministic-algorithm flags, and planned batch schedule. The schedule is generated before training; crop augmentation is stateless and keyed by `(run_seed, global_step, sample_id)`. The proof uses `num_workers=0`; later workers require a pinned worker-state protocol before promotion.
2. **Complete state.** At a batch boundary, write a `state.pt` with model, optimizer, scheduler, AMP scaler, global step/epoch, gradient-accumulation state, Python/NumPy/Torch CPU/Torch CUDA RNG state, sampler cursor, and the previous checkpoint's hash. A separate canonical tensor-hash manifest avoids confusing PyTorch container bytes with state equality.
3. **Atomic handoff.** Write into an otherwise empty `handoff/<run_id>/<step>.partial/`, fsync each file, SHA-256 every member, write `COMPLETE.json` last, then atomically rename to `<step>/`. Only this committed directory is packaged as the next kernel's immutable Kaggle input; partial directories, mutable aliases, and latest-file pointers are invalid.
4. **Fail-closed resume.** The new kernel verifies the full member tree, state tensor hashes, previous-hash chain, manifest/data/code/config identities, and exact accelerator family before deserializing. Any mismatch exits before CUDA training. If `<run_id>/<step>/RESULT.json` already exists, restart returns that receipt without training: handoff publication and restart are idempotent.
5. **Bit-identical proof.** Run A trains the tiny model for one hour, checkpoints and is deliberately killed. Run B starts in a new kernel from only the committed handoff. Compare against an uninterrupted reference at the same step: canonical model/optimizer/scaler/RNG tensor hashes, next-batch identifiers, loss sequence, and fixed prediction bytes must all match exactly. Then invoke the fixed checkpoint-to-44b6 graph emitter once, recording its output hash; it is an infrastructure receipt, never a checkpoint-selection read.

The proof fails if a kernel cannot consume an immutable prior handoff, any identity is unpinned, restart trains an already completed segment, or any continuation comparison differs. Ten failed handoffs is not an allowance: a single unexplainable Phase 1 failure halts the campaign until the protocol itself is repaired and re-preregistered.

## Evidence anchors

| artifact | SHA-256 | fact used |
|---|---|---|
| `reports/g02/trackastra_h1_feature_cache/report.json` | `d52a4a42…d820c0cf` | G-02's exact zero-filled twelve-channel interface |
| `reports/g02/screening/tiered_surface/branches/training_feasibility_v1/DECISION_DOSSIER.md` | `147f0993…5ba6b248` | measured Trackastra anchors and organizer rate |
| `reports/g02/model_landscape_20260724.md` | `0a70c168…eb747e1db` | 478 T4-h convergence estimate and crop-linker failure |
| `external/Tracking_Support/repo/scripts/train_unet_transformer.py` | `c4f63177…d35dc9ea` | organizer joint detector/association loop |
| `external/Tracking_Support/repo/src/biohub_tracking/models/temporal_unet.py` | `d809c35d…345666cac` | organizer temporal 3-D U-Net |

## Explicit non-actions

No GPU kernel was launched, resumed, or submitted for this dossier. No held-fit graph was read, checkpoint selected, or submission slot spent. The legacy G-02 native Trackastra run is not a G-03 experiment and may not be resumed under G-03; remote status/quota reconciliation is mandatory before Phase 1.

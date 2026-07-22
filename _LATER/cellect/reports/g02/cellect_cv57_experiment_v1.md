# CELLECT fit117 → CV57 experiment contract

Status: **adaptation and fit-only index validated; E4 passed and training is
unlocked, but the full GPU run has not been authorized or executed**.

## Frozen scope

- Production floor: exact 57-video division-off baseline, adjusted edge Jaccard
  `0.8711885823490648`, TP/FP/FN `34174/2736/2094`.
- Training corpus: the existing fit-only 117-video cache manifest. The new
  whole-video internal split is 94 training videos and 23 validation videos.
- Untouched evaluation: the exact 57-video fold-0 cohort. It is excluded from
  training and internal validation.
- Detection: unchanged released detector and D4-8 cache. CELLECT is an edge
  representation/cost experiment, not a new detector.
- Divisions: learned positives may include both daughters, as in CELLECT, but
  division edges remain disabled in output until a patched-official-metric TP
  is demonstrated.

## Upstream and license

CELLECT is pinned as `external/CELLECT` at commit
`3586070926f7f1fd5d8df37456861d22bdc63236`. The published 2,541,972-parameter
3D U-Net checkpoint is loaded strictly and its 64-channel embedding map is used
without replacing the existing detector. The upstream GPL-2.0 text and a
project notice are in `third_party_licenses/`.

## Adaptation

The first experiment deliberately keeps the parts of CELLECT that match this
dataset's supervision:

1. Exact upstream two-frame 3D U-Net and published initialization.
2. A second forward pass on `(t+1,t+2)`, matching upstream temporal training.
3. L2-normalized 64-channel embeddings sampled at raw released-detector
   `FrameDet` centers during both fit and evaluation. Fit positives are GT links
   transferred onto detector IDs by the patched official node matcher; GT
   coordinates never enter the optimizer.
4. CELLECT-style squared-distance hard-positive/hard-negative triplet loss with
   margin `0.3`; division daughters are multiple positives.
5. Guaranteed same-frame negatives, with physical top-10 candidates used for
   hard mining.
6. `16×64×64` `(z,y,x)` training crops, AMP mandatory on CUDA.

The segmentation, center, size, and division heads are present because they are
part of the exact checkpoint, but their predictions are not used in the scored
path. This keeps the solved detector and the `0.8711886` baseline as the floor.

## Gates

Training refuses to start unless the versioned E4/E5 report is complete, E4 is
`PASS`, and its raw FrameDet FN cohort is exactly `1232` in-gate, `317`
above-gate, and `545` excluded (`464` officially unmatched plus `81` raw
endpoint absences). That gate passed: 3D NCC improved the 14,759 all-edge
2–4-candidate cases from `0.93604` to `0.94993` (delta `+0.01389`, paired
video-cluster 95% CI `+0.00581` to `+0.02439`). On the 907 confusable in-gate
FNs, it improved `0.28997` to `0.55568`.

After training, the checkpoint qualifies for one untouched CV57 run only when
its best 23-video internal-validation top-1 link accuracy exceeds physical
distance on the identical top-k candidate rows. A CV57 build must then satisfy
all of the following before it can replace the baseline:

- official patched score, not an approximate metric;
- exact 57-video selection and `34174/2736/2094` control reproduction;
- divisions off;
- no detector forward calls (reuse the released cache);
- wall-clock projection within 3.6 GPU-seconds per frame pair on 2×T4;
- score strictly above `0.8711885823490648`.

No Kaggle dataset, kernel, or submission action is authorized by this contract.

## Validated artifacts

- `reports/g02/cellect_cv57_preflight_v2.json`: exact upstream commit, source
  hash, checkpoint hash, GPL hash, fit117/CV57 boundary, and model smoke forward.
- `reports/g02/cellect_fit117_index_v1.json`: 78,621 nodes, 75,916 adjacent GT
  edges, 75,287 three-frame-trainable edges, 83 division sources, and 10,955
  frame-pair samples across fit117.
- `reports/g02/cellect_fit117_raw_supervision_v1.json`: 73,956 raw-detector
  positive edges (`97.4182%` of 75,916 adjacent GT edges), including 73,365
  three-frame-trainable positives, 71 retained division sources, 608,963,711
  guaranteed same-frame negative pairs, and 17,722,946 temporal negative pairs.
  The frozen split remains 94 training / 23 validation videos.
- `src/celltrack_alpha/cellect_adapter.py`: exact-model loader, preprocessing,
  center sampling, top-k gate, triplet loss, and rank metric.
- `scripts/train_cellect_fit117.py`: E4-gated AMP tuning and fit-validation
  qualification harness.

The two one-step CPU training artifacts and the ten-crop untuned diagnostic are
superseded development smokes: they were created before supervision was moved
from GT centers onto raw `FrameDet` centers and therefore are not performance
evidence. They remain only as provenance for the fail-closed confusable-row
requirement. The raw-supervision audit is the current training boundary; no raw-
coordinate tuned or untuned performance result has been produced locally.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/cellect_cv57_preflight.py --smoke
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/build_cellect_fit117_index.py
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/train_cellect_fit117.py --supervision-audit-only
```

The full run is now scientifically unlocked by E4, but remains operationally
blocked because the local runtime has no CUDA device and no Kaggle action is
authorized:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_cellect_fit117.py --device cuda
```

# G-03 Wave R / R3 — reusable G-02 asset audit

Date: 2026-08-01  
Status: **complete / read-only audit**

## Bottom line

G-02 provides a strong graph, scoring, transport, and provenance substrate, but no
reusable artifact supplies G-03's required real 12-dimensional `wrfeat`.  Every banked
Trackastra inference filled those twelve channels with zero.  R1 must extract
deterministic 3-D crops from authenticated raw bytes, run the hash-pinned untrained crop
adapter, and perform fresh Trackastra inference.

The exact allocation floor is `0.881522069960 * 0.985 = 0.868299238910600`, not the
owner's loose `~0.8678` approximation.  The banked 36-video APNCC held-fit control
`0.866534526036` is `0.001764712875` below that floor, but it is a reference rather than
R1's G-03 classification score.

## Reusable assets and exact plug points

| asset | identity | G-03 plug point | validity / caveat |
|---|---|---|---|
| N1 feasibility compiler | `src/celltrack_alpha/action_compiler.py`, SHA `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5`; interface `../../g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json`, SHA `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865` | Compile post-hoc fork overlays; enforce forward time, no merge, outdegree <=2, edge ownership, duplicate/conflict handling, and atomic batches. | Current bytes match the freeze. It proves feasibility only; the historical fit117 scope is not G-03 data authority. |
| Frozen official scorer | `src/celltrack_alpha/official_eval.py`, SHA `22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6`; official `external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py`, SHA `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444`; `division_metrics.py`, SHA `0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9` | Score frozen fragments with edge TP/FP/FN, raw and adjusted edge Jaccard, node recall, and separate division TP/FP/FN/Jaccard. Combined score is adjusted edge Jaccard plus `0.1 * division Jaccard`. | Valid frozen sources. Candidate outputs must be sealed before GT read. Current `../../g02/scorer_parity_v1.json` SHA `d8cfe66b…a3cb6` differs from older cited artifact identities, so use its internally pinned source hashes, not its outer-file hash as historical authority. |
| APNCC 44b6 reference | `../../g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json`, SHA `530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d`; `control_fragments/` aggregate SHA `48b99c5d…` | Immediate same-population association control and CPU division base. | Exact 36-video score/adjusted `0.866534526036`, raw `0.862596057467`, node recall `0.992618000284`, edge `TP/FP/FN=10327/958/687`, division `0/0/9`, Jaccard `0`. |
| H1 zero-`wrfeat` Trackastra graph | Same report's `candidate_fragments/`, aggregate SHA `a476e9fdeebc5cd59e17d0bcbc702e8d59db2394abed892a40aad7e3c2ee938a` | Emitter/parity control for R1. | Score `0.855386449455`; edge `10267/1043/747`; division `0/0/9`. Not the requested real-`wrfeat` result. |
| H1 feature cache | `../../g02/trackastra_h1_feature_cache/report.json`, SHA `d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf` | Reuse detection identities, coordinates, candidate scope, cached APNCC context, and interface checks. | 105 fit videos, 280,095 rows, 195,354 proposals. Trackastra input was 48 base plus twelve zero channels; cache output has 49 values including margin. No crops or real 12-D `wrfeat`. |
| Native association bank | `../../g02/screening/tiered_surface/branches/sa_native_base_wave_v1/run_v1/kaggle_output/trackastra_native_fit105_bank_v1/`, report SHA `005d2c38298b09dee4c655c37fd166e4e188aed727a956c906a6c59aa4a34ae0` | CPU-cheap decoder input: 105 per-video NPZ/JSON banks with 12,523,336 candidate edges and 2,732,827 nodes. | Exact zero-`wrfeat` output. Its selected graph scored `0.8110244`; emitter semantics differ from the H1 `0.8553864` graph and the values cannot be mixed. |
| Pretrained Trackastra asset | `../../g02/trackastra_pretrained_asset_audit_v1.json`, current outer SHA `2f809076fa570388d114f2f2bfe0a9be7dd5186cda41d67b3992127f10fa0132`; source commit `fb265888…`; checkpoint member SHA `24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f` | Freeze source/config/checkpoint for R1 and scientific training. | 27,456,880 parameters, `feat_dim=12`, window 4, max tokens 1024. |
| Fit cache | `../../g02/fitcache_117/manifest.json`, SHA `20ef8056bea8afb60f76cf99357c7b62215b2db6c408d78a1c3ba21c41431a33` | Stable detections, IDs, PairContext, and APNCC fragments. | No raw crop voxels or real `wrfeat`; reauthenticate the intended fit105/44b6 subset. |
| Raw archive | `../../../data/kaggle/raw/biohub-cell-tracking-during-development.zip`; SHA `538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19`; 87,393,127,165 bytes | Authoritative deterministic G-03 crop source. | Reauthenticate member scope; never load test members; crop extraction is GT-blind. |
| Byte-exact transport | `../../../results-bhap.zip`, SHA `031aa5f64fd6eaf918213fc072a098657e2a6bc3d3e5a2d2ae0642d4037b7219`; `../../g02/screening/apncc_transport_smoke_v2/PUBLIC_ARCHIVE_PARITY_ADJUDICATION.json`, SHA `d0cf4c80ece8cb442e232531b80e908fc34ecd06603eadc670819294766e3e15`; `scripts/cmux_ops.py`, SHA `8a0d79aa917d5c649f9dead6f79bbda61002ff1c602d3d0e808bdb7df29c6bad` | Reuse concatenation, schema validation, archive build, remote byte-parity receipts, serial/quota guards, and submission transport after replacing predictions. | PASS only for the old APNCC bytes and four public test videos; transport method is reusable, scientific verdict and prediction bytes are not. |
| 38 repair bundles | `../../g02/screening/tiered_surface/branches/sa_division_value/DIVISION_REPAIR_BUNDLES.json`, SHA `198aeda1d0c0522bd1aedacdbb48e2b343aca0b25cb41d68dd8cb72cbfc87b14` | Reuse event grammar, equivalence ordering, conflict semantics, and risk priors for R4. | GT-derived CV57 oracle asset: 9 44b6, 29 6bba. Never a training/tuning feature. Ceiling division `38/0/4`, Jaccard `0.9047619`, edge delta `+0.001591651`, combined `+0.092067842`. |
| FP sensitivity | `../../g02/screening/tiered_surface/branches/sa_division_value/FP_SENSITIVITY.json`, SHA `6015a1a6440af4d4da705a416472f3f6b3dab7be764549a969bf507b4e1ddd76` | Preregister false-fork-aware ranking and caps. | At TP=25, gains are `+0.059523810/+0.053191489/+0.048076923` for FP `0/5/10`; risk surface, not achievability. |
| Selector requirement surface | `../../g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json`, SHA `6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b`; barrier SHA `d07ef7ef2c9be381e22c18c76ed683844bf018ed194e2c393dae15895d70bd54`; terminal SHA `226e0ffa90b851fc7cddfd57f89f018720ccd600d3cd23d77f63f0bb891ba49e`; amendment SHA `3526b54b29c6372488f15ef715ace4f76e4744efc1a7031215abc28cef64951d` | Calibration prior for abstention, retention, and selector-quality scale. | Synthetic 44b6 action-selector surface against CG0, not a division or APNCC guarantee. DELETE `0.85/0.020394`; occupied swap `0.90/0.067910`; robust points `0.90/0.044946`, `0.95/0.147388`. Re-measure for division. |

## Stale or unsafe inputs

- `../../g02/trackastra_full_pool_feature_cache_v1` contains only one witness; v2 is
  empty.  Neither is a G-03 feature bank.
- `../../g02/trackastra_apncc_base_cache_v2` is partial 6bba material and irrelevant to
  the 44b6 R1 objective.
- DINOv2 v3 assets are XY endpoint/action features, not 3-D crop embeddings or
  Trackastra `wrfeat`.
- The 57-video division appearance cache contains CV57 patch summaries, not voxels or
  reusable embeddings; it is scope-unsafe and structurally incompatible.
- Phase-1 B-v2's `0/0/0` receipt is structural plumbing only.
- G-02 CV57 and its `0.881522069960` score remain campaign reference/provenance and may
  not tune R1 or R4.

## Minimum new R1 freeze

Before a score read, freeze: the exact 36-video identity/member map; crop shape,
physical scale, centering, padding, normalization, channel order, dtype and batching;
adapter architecture, output ordering/range, initialization seed and weight hash;
Trackastra source/config/checkpoint, window, token cap, candidate radius, solver,
thresholds and division mode; graph-emitter identity; and a GT-blind inference receipt.
The scorer opens only after all 36 fragment hashes are sealed.

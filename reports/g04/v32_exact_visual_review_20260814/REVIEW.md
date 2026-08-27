# Frozen V32 visual review

Date: 2026-08-14  
Scope: diagnostic review of the recovered exact frozen V32 CV57 output  
Verdict: `DIAGNOSTIC_ONLY / NOT_A_LOCAL_PREFLIGHT`

The full browser catalogue is `index.html`. Its 57 MP4s cover all 21 `44b6`
and 36 `6bba` CV57 identities and bind recovered V32 prediction fragments with
set SHA-256
`d89ff328826811d1bbde302424dd2bf3758abf82cbdbf7f7a3459a982329350b`.
The catalogue manifest SHA-256 is
`0e49e807fae69a9146e5a9b59adf1f4ff650847132e1e00e782d9bb0c7727715`.

## Representative inspection

Four 100-frame, 15-second MP4s were inspected across both families and both
ends of the exact-score distribution.

- `train:44b6_87bba6c4` (`0.600446914229`): the crowded, expanding tissue
  front contains visibly short and repeatedly interrupted trajectories, with
  distributed association and fragmentation markers rather than one isolated
  bad transition. The catalogue reports edge TP/FP/FN `118/47/31` and division
  TP/FP/FN `0/5/0`. This is evidence for changing the information or hypothesis
  universe in crowded regions, not for another global scalar motion weight.
- `train:6bba_786893ac` (`0.725948664618`): high-density motion remains broadly
  coherent, but errors are spread through the field and all three annotated
  divisions are missed; edge TP/FP/FN is `1209/271/186`, division TP/FP/FN is
  `0/6/3`. A division mechanism must first prove parent/daughter candidate
  coverage; geometry-only relinking has weak visual support here.
- `train:44b6_aaf8b0ea` (`1.004398693641`): ordinary links are visually stable
  and the catalogue reports edge TP/FP/FN `202/1/4`, yet its one annotated
  division is still missed (`0/0/1`). Good ordinary association therefore does
  not establish division competence.
- `train:6bba_76db78c1` (`1.005746652935`): sparse, well-separated cells show
  coherent short-displacement tracks and no annotated edge errors
  (`604/0/0`). This is a useful negative control for any router: an asymmetric
  mechanism should leave these easy regions untouched.

The overlays correctly warn that ground truth is sparse. Gray-ringed nuclei
are unannotated/unknown, not false positives; no detector-precision conclusion
is drawn from them. The strongest diagnostic contrast is crowded versus easy
regions and division versus ordinary association, not raw node density.

## Boundary

These videos were rendered locally from a recovered historical Kaggle output;
they are not evidence that V32 or any changed candidate executed locally.
Accordingly this review must not be used as `LOCAL_VISUAL_PREFLIGHT.json` and
cannot authorize a kernel. Every future model kernel still needs a fresh local
run of its exact candidate, a persisted prediction artifact, candidate-bound
MP4 sidecars, explicit human observations, and a PASS receipt accepted by the
central prelaunch guard.

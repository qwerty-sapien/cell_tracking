# Current Next Steps

Budget policy: use local compute and Kaggle free compute first. Paid GPU work is allowed only for a scoped deep-learning pilot whose measured extrapolated training cost is under `$25`; stop if the pilot implies the full experiment will exceed that cap.

1. Build a ZIP-only local rule-based baseline for smoke testing against the downloaded Kaggle archive.
2. Build a Kaggle-compatible script for free notebook execution against `/kaggle/input/competitions/biohub-cell-tracking-during-development`.
3. Start with multi-scale 3D DoG detection and conservative Hungarian `t -> t+1` linking.
4. Keep division prediction disabled for the first submission.
5. Tune detection count and link radius against sparse train CV before spending any paid GPU money.
6. Fourth approach: evaluate pretrained/small deep models only after the metric/coordinate harness is trusted. Preferred order is official sparse-supervised U-Net+transformer baseline, Trackastra/CTC pretrained association over candidate masks, then Cell-TRACTR-inspired ideas. Do not train a full transformer tracker from scratch under the `$25` cap.

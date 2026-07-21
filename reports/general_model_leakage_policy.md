# G-01 General Model Leakage Policy

Created: 2026-07-12T00:19:47+08:00

## Boundary

G-01 evaluates only the non-duplicate/general tracking route. The four train IDs that are byte-identical to public/test Zarrs are excluded from claimed general-model score evidence:

- `44b6_0113de3b`
- `44b6_0b24845f`
- `6bba_05b6850b`
- `6bba_05db0fb1`

Historical duplicate-route reports may remain in the repository, but `scripts/build_duplicate_geff_submission.py` outputs and any train-GEFF copy route are not valid evidence for AC-01.

## Split

The locked split manifest is `reports/general_model_split_manifest.json`. It assigns eligible non-duplicate train datasets to five video-level folds by embryo prefix. Development clips may be used for debugging, but the final score gate must use official metrics over the locked validation folds or record a blocked stop.

## Offset And Calibration Rules

- Hard-coded prefix offsets such as the historical `44b6 -> (0, -16, +8)` correction are diagnostic only.
- Any offset/localizer calibration used for a fold must be derived only from that fold's `calibration_train` datasets.
- The calibration manifest must record source datasets, parameters, command, config hash, and output hash.
- Held-out validation labels, test labels, duplicate public IDs, and leaderboard feedback cannot be used to set route parameters.

## Side Effects

Allowed now: local source/script/report/prediction/derived writes under the G-01 plan.

Not allowed without later approval: Kaggle push/submission, paid GPU, Runpod, secret reads, or unbounded full-ZIP exhaustive runs.

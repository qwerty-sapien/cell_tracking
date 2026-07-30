# Division N2 — non-fitting evidence contract

**Verdict: ESCALATE.** The contract below distinguishes new raw-image representation evidence from a current-pool selector, but a leakage-safe execution cannot fit the immutable plan, current Kaggle capacity, or published memory envelope. No N3 human request is produced.

## Preregistered evidence-only contract: `TAP-SYM-DIV-FALSIFIER-v1`

### Fixed scope and input bans

- Future scope only: the existing frozen five outer folds and their pre-existing 44b6/6bba group identities. The 38 direct continuity bundles remain oracle diagnostic targets and are never inputs.
- Legal inputs: raw intensity XY patches at fixed physical scale around one parent at `t-1` and unordered daughters at `t`, plus `t+1` context. Detection coordinates and voxel scale locate crops only.
- Forbidden: GT/official-scorer labels, 38-bundle identities, current edge/action scores or ranks, association-continuity outputs, fit labels, candidate outcomes, campaign-chosen thresholds, and every held-side raw frame while training that fold's encoder.
- Required output: append-only `representation_evidence` containing raw-crop hashes, encoder hash, forward-minus-reversed time-arrow response `A`, reflected daughter coherence `S`, parent-to-daughter-union coherence `U`, and daughter-order checksum. It contains no action ID, score rank, selected edge, graph edit, or result label.

### Observable, null, criterion, and kill line

For a fixed encoder, define `Q = A × S × U`. This is evidence, not an operational selection threshold. The analytic fixture defines a parent blob splitting into two equal reflected daughters. Its non-split null has identical locations, coordinate/current-pool fields, and crop-hash slots but pixels of one translated blob; a second null reverses time.

Before any campaign movie is read, a source-pinned encoder must satisfy all of: `Q(split) > Q(non-split)`; `A(forward) > A(reverse)`; swapping daughter order and reflecting both daughters changes `Q` by at most `1e-6` in float64; and the record schema rejects every forbidden current-pool/GT field.

**Kill line:** any failed condition, encoder/model-hash mismatch, held-fold raw-frame inclusion, or attempt to materialize an action/fit/score terminates the mechanism as `DEAD_CONTRACT` before a graph experiment. Passing would show only that a new raw-pixel channel is mechanically distinguishable; it would not be a performance GO.

### Read-only static verification

No candidate code, representation asset, crop, fit, or score was created or run. A source-only inspection of `/Users/agent/Desktop/cell_tracking/tests/test_action_compiler.py` confirms that the extant compiler represents a binary `DIVISION_FORK` with two edges and rejects omitted incident-conflict ownership, out-degree >2, missing endpoints, and invalid batch ownership. This is compatible with keeping the proposed evidence record outside the compiler until a later review. No candidate-specific raw-image schema or synthetic TAP fixture exists; passing a generic fork fixture off as representation validation would be invalid.

## Why it is discriminating

Two records can have identical node coordinates and all current-pool/action-table columns while split and non-split raw patches yield different `A`, `S`, and `U`. Conversely, current-pool columns are forbidden to the encoder and record. The contract tests a new observation channel, not a threshold or refit, and never uses the 38/42 oracle diagnosis to choose a threshold.

## Feasibility result

The released TAP implementation documents raw 2D+time input, a default configuration requiring an NVIDIA GPU with at least 24 GB memory, and roughly **8 GPU-hours for one model** ([repository README](https://github.com/weigertlab/tarrow)). Fold isolation requires one encoder per frozen outer fold: one shared encoder trained on all raw videos would expose each held fold to its own raw distribution and bypass held-fit discipline. The published-rate lower-bound is therefore **5 × 8 = 40 GPU-hours**, before inference and integrity work.

`LIVE_STATE.md` sets a 30 GPU-hour weekly cap and records only **25.8111 GPU-hours** of owner-authenticated launch capacity. Forty hours exceeds the weekly cap by 10 hours and the current capacity by 14.1889 hours. The source's >=24 GB default is also not an authenticated fit for the campaign's existing Kaggle environment. Lowering batch size, changing backbone, training one transductive encoder, or obtaining a different pretrained checkpoint would each be a new unapproved mechanism/assumption rather than an execution repair.

The non-fitting contract is discriminating, but there is no bounded eligible compute protocol under the locked constraints. This is an `ESCALATE`, not permission to spend remaining quota.

## Proposed ledger row

`TAP-SYM-DIV | N2 ESCALATE | discriminating non-fitting split-vs-nonsplit contract; five-fold leakage-safe TAP estimate 40 GPU-h and >=24 GB default exceed 30 h weekly and 25.8111 h capacity | no asset/fit/score/action | no tier move`.

This row is proposed only; `LEDGER.md` was not edited.

**Cascade:** halt on `ESCALATE`. N3 remains closed.

# N2 — revised DPRC fixture preregistration

Date: 2026-07-30 (Asia/Singapore)
Status: **preregistered before the recorded N2 result**

## Scope

This is an entirely synthetic, standard-library unit fixture. It may use only
4-by-4 arrays created in the test file; no model, package, campaign data, video,
feature, R5 input/checkpoint/row, or external process is in scope.

## Fixed observable

For every foreground source patch, select its highest-cosine target candidate in
the fixed one-cell local neighbourhood. A contribution is its cosine only if the
target maps back to that exact source under the same rule; otherwise it is zero.
C(i,j) is the sum of these contributions divided by *all* foreground source
patches. This denominator makes cycle coverage inseparable from similarity.

The occupied-swap observable is the fixed signed difference
C(i, alternative) - C(i, incumbent). Empty foreground support is an error, not
a neutral score. Ties use the lower patch index deterministically.

## Null, expected direction, and kill line

| Fixture | Null | Expected outcome | Kill line |
|---|---|---|---|
| Identity vs reversed 4-by-4 basis grid | Global pooled means are identical, so global cosine is 1.0 for both pairs. | Identity C=1.0; reversed-grid C<=0.25; global cosine remains 1.0 for both. | Any mismatch or a reversed score above 0.25. |
| Uniform-token conflict | Local information supplies no identity distinction. | Incumbent and alternative tie exactly; margin is 0.0. | Any nonzero margin. |
| Empty target foreground | No valid correspondence exists. | Raise ValueError; emit no score. | Any numeric score. |
| Repeat | The algorithm is deterministic. | Two calls are exactly equal. | Any difference. |

Any failed assertion, dependence on a third-party package, changed one-cell
neighbourhood, changed denominator, or fallback to global pooling is **DEAD for
this exact DPRC-O0 contract**. A pass validates only the described unit contract,
not microscopy transfer or a graph-level effect.

## Revision disclosure

An exploratory first implementation averaged only successful cycles and failed the
spatial-permutation assertion (0.8 > 0.25), proving that it could conceal poor
coverage. It is discarded as non-preregistered implementation exploration. The
contract above changes the denominator to all source foreground patches, is now
fixed, and is the only version evaluated for N2.

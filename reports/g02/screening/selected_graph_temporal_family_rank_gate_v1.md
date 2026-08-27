# Family-tagged temporal rank gate

Decision: **PARK the all-family model; preregister a `6bba`-only hybrid
confirmation.**

The same linear/MLP temporal rankers were trained and thresholded separately
for the runtime-known `44b6` and `6bba` dataset prefixes. Five-fold OOF
assignment remained video-grouped within each family.

The best linear configuration produced:

| family | OOF correct / harmful | precision | utility | pair-NCC control utility |
|---|---:|---:|---:|---:|
| `44b6` | 479 / 52 | 0.9021 | 375 | 344 |
| `6bba` | 1,894 / 69 | **0.9648** | **1,756** | 1,443 |

The all-family model is inadmissible because `44b6` fails the 0.95 precision
floor. The fixed validation result nevertheless supports a narrower,
preregistered hybrid:

| family | temporal utility / precision | control utility / precision |
|---|---:|---:|
| `44b6` | 22 / 0.9286 | 13 / 0.8947 |
| `6bba` | **173 / 0.9835** | 145 / 0.9429 |

Therefore the next confirmation changes only `6bba`, where both OOF and
validation independently pass. `44b6` remains byte-for-byte on the banked
appearance-NCC direct-swap policy. The hybrid must reproduce all `44b6`
proposals exactly, use the hash-pinned `6bba` gate, and beat the banked full
CV57 score `0.8815220699597315`; otherwise it is PARK.

No public, GPU, or paid action is authorized by this preregistration.

## Confirmation result

The hybrid completed all 57 videos on CPU at `0.8801140023577207`, with
`34,308 / 2,511 / 1,960` TP/FP/FN. Relative to the preregistered banked APNCC
reference it is `-0.0014080676020108`, with 12 fewer TP, 52 more FP, and 12
more FN. The hybrid is therefore **PARK** and the banked APNCC candidate
remains unchanged.

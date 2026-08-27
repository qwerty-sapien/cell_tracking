# H000 decode-path diagnosis and score-blind runtime amendment

Date: 2026-08-03  
Status: **DIAGNOSED / CANONICAL SAME-RUNTIME RE-DECODE AUTHORIZED / SCORE STILL CLOSED**

## Finding

The zero-initialization fix works. On real crops, the Phase-2 adapter hidden
activations are nonzero and all twelve output channels are bitwise float32
zero. The first two local h000 association banks also have exactly the same
node IDs, frames, coordinates, candidate edge IDs, and distances as the frozen
native bank.

The initial byte-parity gate nevertheless caught one graph difference:

| held-fit36 identity | native runtime | local runtime | maximum probability difference | selected-edge difference |
|---|---|---|---:|---:|
| `train:44b6_cf8fed6b` | CUDA `cuda:0` | CPU, PyTorch 2.13.0 | `2.086162567138672e-05` | one extra CPU edge |
| `train:44b6_12dfb391` | CUDA `cuda:0` | CPU, PyTorch 2.13.0 | `1.8417835235595703e-05` | zero; fragment byte-identical |

For the sole changed edge `(1865, 2099)`, the frozen CUDA probability is
`0.34999990463256836` and the local CPU probability is
`0.35000360012054443`. They straddle the fixed decoder threshold `0.35`.
Thus a roughly `3.7e-06` floating-point change becomes a discrete graph change.
This is a Trackastra CPU/CUDA forward-path difference before decoding; it is
not an adapter-init, crop, population, candidate construction, compiler, or
scorer discrepancy.

The local replay was interrupted immediately after the discrepancy was
observed. No held-fit score, ground truth, CV57 result, or scientific GPU hour
was opened or consumed.

## Resolution sealed before scoring

At h000, `tanh(Linear(hidden))` has exactly zero weight and bias, so the adapter
supplies exactly the same float32 zero tensor as the established native
zero-`wrfeat` interface for every finite crop. The canonical h000 graph is
therefore re-decoded from the frozen native CUDA association banks under the
unchanged compiler. This keeps the Trackastra numerical runtime common between
the baseline and h000 while still emitting a new immutable set of h000 graph
fragments.

The canonical seal must satisfy all of the following before the official
scorer can open:

1. the h000 checkpoint and Phase-2 contract hashes remain unchanged;
2. all 36 frozen CUDA banks and their CUDA receipts match their aggregate hash;
3. all 36 graphs are newly re-decoded with the exact R1 compiler; and
4. every new graph fragment is byte-identical to the frozen zero-feature
   reference.

The machine-readable amendment is
[`H000_RUNTIME_EQUIVALENCE_AMENDMENT.json`](H000_RUNTIME_EQUIVALENCE_AMENDMENT.json).
H005 and H030 must use one frozen accelerator/toolchain path; cross-device
forward replays are portability diagnostics, not score-comparable checkpoints.

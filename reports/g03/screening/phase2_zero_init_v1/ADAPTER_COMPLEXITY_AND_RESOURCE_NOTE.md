# Phase-2 adapter complexity and resource note

Date: 2026-08-03  
Scope: h000 CPU validation only; this is not a T4 launch calibration

For batch size `N`, the fixed float32 crop input contains
`N * 3 * 16 * 32 * 32` elements (`196,608 N` bytes). The fixed-shape forward
has linear batch complexity. After `AvgPool3d(4,8,8)`, its principal activation
sizes are `192 N`, `1,024 N`, and `3,072 N` float32 elements for the pooled
input, first convolution, and second convolution; hidden and output tensors are
`48 N` and `12 N` elements. The two convolutions perform approximately
`82,944 N` and `1,327,104 N` multiply-accumulates, and the projection performs
`576 N`. The adapter has 22,684 parameters and the immutable h000 checkpoint is
101,075 bytes.

A single-thread CPU inference-only probe on the current local runtime, after
three warmups and across 20 repetitions, measured:

| Batch | Mean forward | Mean per crop |
|---:|---:|---:|
| 1 | `0.1844 ms` | `0.1844 ms` |
| 2 | `0.2888 ms` | `0.1444 ms` |
| 4 | `0.5187 ms` | `0.1297 ms` |

These measurements establish bounded local validation cost only. They do not
substitute for the frozen T4 throughput, training-memory, and quota calibration
required before a scientific launch. Because the Branch-B dossier failed,
h005 was not funded; that T4 calibration gate remains closed, actual h005
debit is exactly zero, and no accelerator-memory claim is made.

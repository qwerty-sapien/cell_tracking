# G-04 Lane A — permissive division frontier

The exact APNCC CV57 baseline reproduced at `0.8815220699597315`. The bounded 20-policy sweep found **no division TP**. Its nondominated measured frontier was:

| Policy | Applied forks | TP | FP | FN | Division J | Combined S | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 0 | 0 | 0 | 42 | 0 | 0.8815220699597315 | 0 |
| unfiltered geometry cap 1 | 54 | 0 | 2 | 42 | 0 | 0.8815741350411858 | +0.0000520650814543 |
| unfiltered geometry cap 3 | 155 | 0 | 8 | 42 | 0 | 0.8816069249327657 | +0.0000848549730342 |
| unfiltered geometry cap 5 | 239 | 0 | 11 | 42 | 0 | 0.8816354767369976 | +0.0001134067772661 |

`unfiltered_geometry_cap5` is promoted only because the owner explicitly authorized any local improvement above the exact baseline. The local gain is edge-only: it does **not** close the zero-division-credit gap, and it is below the established fine-transfer resolution. The public submission is therefore a useful observation-distinct transfer probe, not evidence that division decoding succeeded.

Exact analytic artifacts quantify the potential upside separately. For example, at fixed adjusted-edge score, 15 TP / 15 FP gives `J=15/57` and `S=0.9078378594334157105`.

Promoted test CSV SHA-256: `da73c7540dc0f1052b1f6f0d1399a9a9867f24de9c8463180fa6eef2bb484509`.

# APNCC direct-swap policy screen

Status: **PARK**.

The banked APNCC policy raises its fit-train appearance threshold from
`-0.14937132596969604` to `-0.04768145084381103` to satisfy the sampled
background-support budget before runtime truncation. The deployed direct-swap
path then independently applies a strict `max(16, 0.5% of selected edges)`
per-video cap.

This screen tests one bounded hypothesis: the cap may already provide enough
abstention that retaining the original fit-train threshold recovers useful
proposals without sacrificing graph-level precision.

The exact frozen scorer will compare the two policies only on the untouched
fit-validation12 partition:

| policy | appearance threshold | cap |
|---|---:|---:|
| banked control | -0.0476814508 | max(16, 0.5%) |
| lower-threshold candidate | -0.1493713260 | max(16, 0.5%) |

The candidate reaches CV57 only if it gains at least `+0.005` adjusted edge
Jaccard on fit-validation12, does not regress raw edge Jaccard, and loses no
more than `0.0001` node recall. No threshold or cap is selected from
fit-validation12 or CV57.

No Kaggle, GPU, RunPod, or paid action is authorized.

## Result

The exact fit-validation12 scorer returned:

| policy | adjusted edge Jaccard | edge Jaccard | TP / FP / FN |
|---|---:|---:|---:|
| banked threshold | **0.8702098920** | 0.8823006226 | 5,952 / 400 / 394 |
| lower threshold | 0.8702095487 | 0.8823006226 | 5,952 / 400 / 394 |

The lower threshold added nine proposals across two videos. It was
observation-distinct (node Jaccard `0.9999403`, edge Jaccard `0.9998929`) but
corrected no scorer events and changed adjusted score by only
`-0.0000003433`. It therefore fails the `+0.005` promotion gate and is PARK.
No CV57 replay was performed.

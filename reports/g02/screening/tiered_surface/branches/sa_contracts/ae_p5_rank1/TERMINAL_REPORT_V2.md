# AE-P5 C0/C2 structural falsifier — complete N1 ownership

Verdict: `NO_RANK1_ACTION_EMITTED`. This is a fit-only CPU structural result,
not a graph score or tier move.

The earlier V1 result was not used for this decision because it grouped only
source and destination slots. The frozen execution contract reconstructs every
occupied swap's full N1 boundary from the authenticated APNCC control fragments:
source, destination, incumbent edge(s), challenger edge, and explicit no-op
endpoint.

| quantity | result |
|---|---:|
| frozen actions | 45,961 |
| complete-ownership components | 40,894 |
| components with fewer than 8 possible assignments | 40,853 |
| potentially decisive N1 batch compiles | 361 |
| batch rejections | 0 |
| components applying C2 rank-1 | **0** |
| selected actions | **0** |

The 41 components that could reach `K=8` all failed the unchanged fixed
membership conjunction (14 introduce failures and 27 remove failures). Thus
the fixed named C0/C2 realizer emits the empty set and is byte-identical to the
APNCC control by construction. No held labels, fitting, official score, graph
run, 6bba, CV57, GPU, Kaggle action, RunPod, or budget debit occurred.

This does not demote the broader AE-P5 family. It establishes only that a new,
separately named non-inert C2 realizer is required before any future graph gate.

Artifacts: execution contract `2dcea4ff10fc912db7da09e141d6a364a7e4e886b4053a2809188acf5d7d8e0d`;
result `073ca72f5e6ad8c2eaa8fb0bcd747799838ed3eaeecf325cb8edb94d9aa425ed`.

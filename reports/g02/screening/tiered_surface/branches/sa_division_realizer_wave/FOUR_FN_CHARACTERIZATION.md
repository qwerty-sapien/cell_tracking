# Four unrepaired division events

The frozen oracle repaired 38 of 42 events. The remaining four are outside
the 38-event fragmented-graph target and split into two structurally distinct
classes.

| Class | Events | Why the 38-bundle realizer cannot repair them |
|---|---:|---|
| Detection/localization role missing | 3 | At least one strict frozen-scorer parent/daughter role has no prediction node. An edge/fork action cannot connect a node that does not exist. |
| Connected, no local fork, missing branch persistence | 1 | All roles exist and one component spans them, but there is no eligible local fork. The direct N1 action compiled, then D1 correctly suppressed it because only one branch survived to the required persistence horizon. |

The role-missing events are `train:44b6_aaf8b0ea` divider 93,
`train:6bba_4ffd3da3` divider 312, and `train:6bba_7f87b3d8` divider 817.
The connected/nonpersistent event is `train:44b6_c8e2a523` divider 155.

Two separate future diagnostics are proposed, not authorized or executed:
`D2_ROLE_RECOVERY_DIAGNOSTIC` for upstream detection/localization recovery,
and `D3_CONNECTED_NONPERSISTENT_LOCAL_FORK_DIAGNOSTIC` for the already
connected but locally forkless case. No fitting, gate change, or plan
amendment follows from this characterization.

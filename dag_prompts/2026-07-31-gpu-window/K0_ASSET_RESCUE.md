# K0 — asset rescue: the Trackastra checkpoint lives in `/private/tmp`

**Run this first. It does not depend on `H0`, costs zero GPU, and is the highest-value
five minutes available in this window.**

## The finding

The only authenticated Trackastra checkpoint in the campaign — the frozen `27,456,880`-parameter
backbone the entire §6.2 PEFT node depends on — currently exists **only** at:

```
/private/tmp/trackastra-owner-roundtrip.20260729/resolved/model.pt
/private/tmp/trackastra-owner-roundtrip.20260729/resolved/ctc.zip
```

| Object | Bytes | SHA-256 |
|---|---:|---|
| `model.pt` | 109,906,058 | `24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f` |
| `ctc.zip` | 101,957,234 | `2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e` |

`trackastra-6bba-assets-v1/` in the repo root is **empty**. `/private/tmp` does not survive
a reboot on macOS. Re-acquiring these requires another owner-authenticated cmux SDK
retrieval — the exact step that already cost this campaign a full round trip.

## Task

1. Copy both files to a durable, git-ignored location under the repo — suggested
   `assets/trackastra-authentic-v1/`. Verify the SHA-256 of each copy **after** the copy
   and record both.
2. Confirm the destination is covered by `.gitignore` (these are 200 MB of weights; they
   must not enter git). If it is not, add the pattern in the same change.
3. Write `assets/trackastra-authentic-v1/PROVENANCE.json` recording: source path, both
   hashes, both byte counts, `base_parameter_count: 27456880`, the retrieval lineage
   (owner-cmux SDK retrieval, 2026-07-29), and the fact that
   `sa_peft_adapter_v1/PRELAUNCH_AUDIT_20260731.json` already authenticated this exact
   `model.pt` hash.
4. Re-run the hash check from the new location and paste the real output into the receipt.
   Do not assert success without the command output.

## Do not

- Do not upload these to Kaggle in this node. Packaging is `K2`'s job and needs the
  dataset-ownership registry entry first (`KFG-17`).
- Do not modify, re-serialize, convert, or "clean" the checkpoint. The pinned hash is the
  identity; any re-save breaks it. `CELLECT` lost a run to exactly this
  (direct `.pth` transport proved byte-changing).
- Do not delete the `/private/tmp` originals until the copy's hashes verify.

## Done when

`assets/trackastra-authentic-v1/` holds both files at their exact pinned hashes, a
`PROVENANCE.json`, and the verification command output is quoted in the node report.

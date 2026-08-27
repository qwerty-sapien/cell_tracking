# OCC-SLICED-W1 Repair-1 launch receipt

Date: 2026-08-09 SGT  
Dataset: `onebitbrain/g02-occ-sliced-w1-input-v1`, private Version 1  
Kernel: `onebitbrain/g02-occ-sliced-w1-repair1-v1`, private Version 1  
Disposition: **DATASET CREATED ONCE / KERNEL SENT ONCE / DO NOT POLL**

> **Owner limit amendment 2026-08-10:** this receipt and its submitted package bytes
> remain immutable. The package's 1-hour stop is historical and superseded for future
> versions by repository-root `LMT.md`. A future run requires a new slug/package and
> fresh certification; this receipt cannot authorize or describe those bytes.

## Recursive verdict

The terminal path is
`N0=Y -> N1=Y -> N2=N(NUMERICAL) -> N2B=Y -> N3=Y -> K1=Y -> R0=N(KFG-09) -> K1R=Y`.
The first kernel version was scientifically unmeasured because a mutable shared
dataset exposed the wrong carrier before heavy work.  K1R changes only input
transport; the sliced-W1 parameters, population, feature functions, output
contract, and downstream gates are unchanged.

## Dedicated input creation

The fresh-slug search returned `No datasets found`.  One CMUX
`datasets create` call then uploaded all 43 declared files and returned code
zero with no semantic failure.  Kaggle reported that the private dataset was
being created.  One post-create listing returned exactly
`G02 OCC Sliced W1 Input v1`, size 9,673,468 remote-listed bytes and zero
downloads.  The local logical payload remains 9,881,374 bytes; the list value
is Kaggle's transported size, not a replacement for the content manifest.

- input manifest SHA-256:
  `806c7fd4198792b651b2c76536b26f0ae53195fa6ad5cc20252c636b465d3a94`;
- fresh-slug receipt SHA-256:
  `e4872fbfa1eb1fcbe40c790c1c566187d5cd64d85ed3f57b1fa33ca551970d6c`;
- dataset-create receipt SHA-256:
  `fa887b4e9723ad718c628e09ee2a012c9e986fdd52fb42a8efa3208fa42ac101`;
- post-create listing receipt SHA-256:
  `3f20c8a15efb9c0626cd2cb47bd55f1f4642976d88b487fc7d76a7394ed14ab5`.

The scientific kernel's first executable Step 0 will independently authenticate
the mounted manifest and every declared path/byte/hash before competition data
or heavy work.

## Corrected kernel send

Under the shared serial lock, the fresh-kernel guard proved the repair slug was
absent and expected Version 1.  One CMUX push started at
`2026-08-09T15:50:25.272456Z` and returned code zero, no semantic failure, and
`Kernel version 1 successfully pushed.`  There was no retry.

- code SHA-256:
  `3b7990897dcf829435cc4731377b90dd65ce3e57522c9f47513b6f040193b10b`;
- metadata SHA-256:
  `aa2e9acec1c6cc34d7964b75da2b36657336dd719c48cace929564492ff3421d`;
- eligible inline-guard receipt SHA-256:
  `6e5140720f6e9a184e1184b0dcd9b2c31d862dac1aab529a8f8b7d3ae90deecb`;
- fresh-kernel status receipt SHA-256:
  `49caa52c3246356e6895bd6a7e9eaed69ad35a7ba8463de13f0491622661da15`;
- push receipt SHA-256:
  `433841b725a4488a92fadbdcd0da30e7c42786ee70f34dafeba8778d991fc742`;
- pre/post quota receipt SHA-256:
  `84dccd60a329b9554b8c171f0b5afd4a8eaec6293019c28a3cfdde26e873716a` /
  `16e43a496c08f5a0426bfb43851921b0c99b8764ce0e9c75027c50839527a518`.

Authoritative quota was unchanged at `6.88 h` used / `23.12 h` remaining of
`30 h`.  The kernel is CPU-only, so scientific GPU allocation remains
`0.0/15.0 h`.  The CMUX wrapper retains `0.01 h` administrative reservations
for each of the failed original and active repair slugs (`0.02 h` total) until
later separately authorized terminal reconciliation.

Per owner instruction, the corrected kernel was not polled.  No output was
downloaded or scored, and no competition submission was made.

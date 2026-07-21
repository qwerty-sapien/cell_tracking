#!/usr/bin/env python3
"""Build private Kaggle jobs that materialize fold-0 fit caches only.

The jobs reuse the hash-checked V45/A0 runtime bundle, but deliberately do not
score or submit anything.  They cover the fold-0 fit partition, omit already
complete local bundles, and balance the remaining videos by exact compressed
image bytes across a small number of private T4x2 jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bh9gb, pkgbh9  # noqa: E402


ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
EXISTING = ROOT / "kaggle_kernel/bh9o7/bh9out"
OUTPUT_ROOT = ROOT / "kaggle_kernel/fitcache"

FOLD = 0
DEFAULT_SHARDS = 2
FIT_DEADLINE_SECONDS = 8_100
CHILD_DEADLINE_MINUTES = 120.0
ENTRYPOINT = "fitc.py"
METADATA = "kernel-metadata.json"
BUILD_REPORT = "build.json"


class FitCacheBuildError(RuntimeError):
    """The fit-cache package is incomplete or would cross a data boundary."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FitCacheBuildError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, Mapping):
        raise FitCacheBuildError(f"JSON input is not an object: {path}")
    return value


def fold_fit_ids(split_path: Path = SPLIT, *, fold: int = FOLD) -> tuple[str, ...]:
    split = _load(split_path)
    folds = split.get("folds")
    row = folds.get(str(fold)) if isinstance(folds, Mapping) else None
    values = row.get("fit_video_ids") if isinstance(row, Mapping) else None
    audit = split.get("audit")
    audit_ids = set(audit.get("video_ids", ())) if isinstance(audit, Mapping) else set()
    if (
        not isinstance(values, list)
        or len(values) < 2
        or any(not isinstance(item, str) or not item.startswith("train:") for item in values)
        or len(set(values)) != len(values)
        or set(values) & audit_ids
    ):
        raise FitCacheBuildError("fold fit selection is malformed or touches audit data")
    return tuple(sorted(values))


def _slug(identity: str) -> str:
    return identity.replace(":", "__", 1)


def complete_bundle(root: Path, identity: str) -> bool:
    slug = _slug(identity)
    required = (
        root / "cache" / f"{slug}.bhc",
        root / "cache" / f"{slug}.json",
        root / "cache" / f"{slug}.intent.json",
        root / "features" / f"{slug}.json",
        root / "fragments" / f"{slug}.csv",
        root / "status" / f"{slug}.json",
        root / "transactions" / f"{slug}.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def balanced_shards(
    identities: Sequence[str], sizes: Mapping[str, int], shard_count: int
) -> tuple[tuple[str, ...], ...]:
    if shard_count < 1 or not identities or len(set(identities)) != len(identities):
        raise FitCacheBuildError("shard selection must be nonempty, unique, and positively sharded")
    if not set(identities).issubset(sizes) or any(
        type(sizes[identity]) is not int or sizes[identity] <= 0 for identity in identities
    ):
        raise FitCacheBuildError("compressed-byte inventory does not cover the selection")
    rows: list[list[str]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    for identity in sorted(identities, key=lambda item: (-sizes[item], item)):
        index = min(range(shard_count), key=lambda item: (totals[item], item))
        rows[index].append(identity)
        totals[index] += sizes[identity]
    if any(not row for row in rows):
        raise FitCacheBuildError("requested more shards than materialization sources")
    return tuple(tuple(sorted(row)) for row in rows)


def selection_plan(
    *,
    archive: Path = ARCHIVE,
    registry: Path = REGISTRY,
    split: Path = SPLIT,
    existing_root: Path = EXISTING,
    shard_count: int = DEFAULT_SHARDS,
) -> dict[str, Any]:
    fit_ids = fold_fit_ids(split)
    registry_ids, _registry = bh9gb.load_train_registry(registry)
    if not set(fit_ids).issubset(registry_ids):
        raise FitCacheBuildError("fold fit sources are not a subset of the train registry")
    inventory = bh9gb.image_inventory(archive, registry_ids)
    sizes = inventory["compressed_bytes"]
    existing = tuple(identity for identity in fit_ids if complete_bundle(existing_root, identity))
    missing = tuple(identity for identity in fit_ids if identity not in set(existing))
    shards = balanced_shards(missing, sizes, shard_count)
    shard_rows = [
        {
            "index": index,
            "video_ids": list(values),
            "video_count": len(values),
            "compressed_bytes": sum(sizes[identity] for identity in values),
            "selection_sha256": sha256_bytes(canonical_bytes(list(values))),
        }
        for index, values in enumerate(shards)
    ]
    core = {
        "schema": "celltrack-alpha.fit-cache-plan.v1",
        "fold": FOLD,
        "fit_video_ids": list(fit_ids),
        "fit_video_count": len(fit_ids),
        "fit_compressed_bytes": sum(sizes[identity] for identity in fit_ids),
        "train_compressed_bytes": int(inventory["train_image_bytes"]),
        "already_complete_video_ids": list(existing),
        "already_complete_video_count": len(existing),
        "missing_video_ids": list(missing),
        "missing_video_count": len(missing),
        "shard_count": len(shards),
        "shards": shard_rows,
        "data_boundary": {
            "train_fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "competition_submission_performed": False,
        },
        "source": {
            "archive_central_sha256": inventory["central_sha256"],
            "registry_sha256": bh9gb.sha256_file(registry),
            "split_sha256": bh9gb.sha256_file(split),
        },
    }
    return {**core, "plan_sha256": sha256_bytes(canonical_bytes(core))}


def _replacement_main(video_ids: Sequence[str], selection_sha256: str) -> str:
    encoded_ids = json.dumps(list(video_ids), ensure_ascii=True, separators=(",", ":"))
    return f'''FIT_IDS=json.loads({json.dumps(encoded_ids)})
FIT_SELECTION_SHA256={selection_sha256!r}
FIT_DEADLINE={FIT_DEADLINE_SECONDS}
FIT_CHILD_MINUTES={CHILD_DEADLINE_MINUTES!r}

def main():
 started=time.monotonic(); end=started+FIT_DEADLINE; stage="embedded_materialization"
 def alarm(_s,_f): raise LaunchError("fit-cache notebook deadline reached")
 signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,FIT_DEADLINE)
 try:
  if OK.exists() or FAIL.exists(): raise LaunchError("one-trial fit-cache job refuses existing evidence")
  if not FIT_IDS or len(FIT_IDS)!=len(set(FIT_IDS)) or any(not x.startswith("train:") for x in FIT_IDS): raise LaunchError("fit-cache selection is invalid")
  materialize(); stage="attachment_attestation"; support,wheels,plan=verify_att()
  stage="dependency_attestation"; deps=repair(wheels,plan,end)
  stage="source_attestation"; extract(); paths=child_paths(support); env=child_env(paths)
  stage="required_import_attestation"; imports=probe_required(paths,env,end)
  stage="gpu_attestation"; names=gpu()
  stage="competition_discovery"; competition=one_mount(CM,"competition")
  registry=safe_file(ROOT/"bh9reg.json",ROOT,"registry")
  known={{x.get("video_id") for x in load(registry).get("records",[])}}
  if not set(FIT_IDS).issubset(known): raise LaunchError("fit-cache selection escaped the compact train registry")
  script=safe_file(ROOT/"scripts/run_bh.py",ROOT,"runner child script")
  safe_dir(competition,competition,"competition mount"); safe_dir(support,support,"support mount"); safe_dir(OUT.parent,WORK,"child output parent")
  cmd=[str(python_exe()),"-P",str(script),"--input",str(competition),"--registry",str(registry),"--support-root",str(support),"--out-dir",str(OUT),"--output-file","fit.csv","--mode","cv","--split","train","--datasets",",".join(FIT_IDS),"--devices","cuda:0,cuda:1","--pair-batch-size","1","--view-batch-size","1","--deadline-min",str(FIT_CHILD_MINUTES),"--graph-variant","a0"]
  stage="a0_fit_cache_materialization"
  run=subprocess.run(cmd,check=False,cwd=str(ROOT),env=env,timeout=remaining(end,"fit-cache completion"))
  if run.returncode!=0: raise LaunchError("fit-cache runner exited %s"%run.returncode)
  report=load(OUT/"run.json")
  if report.get("ok") is not True or report.get("mode")!="cv" or report.get("graph_variant")!="a0" or report.get("requested_identities")!=sorted(FIT_IDS): raise LaunchError("fit-cache run report is incomplete")
  required={{"cache":(".bhc",".json",".intent.json"),"features":(".json",),"fragments":(".csv",),"status":(".json",),"transactions":(".json",)}}
  counts={{}}
  for folder,suffixes in required.items():
   base=safe_dir(OUT/folder,OUT,"fit-cache "+folder)
   for identity in FIT_IDS:
    slug=identity.replace(":","__",1)
    for suffix in suffixes:
     safe_file(base/(slug+suffix),base,"fit-cache artifact")
   counts[folder]=len([x for x in base.iterdir() if x.is_file()])
  evidence={{"schema":"celltrack-alpha.fit-cache-run.v1","status":"completed","purpose":"fold0-fit-supervised-scaling","fold":0,"selection_sha256":FIT_SELECTION_SHA256,"selected_video_ids":FIT_IDS,"selected_video_count":len(FIT_IDS),"elapsed_seconds":time.monotonic()-started,"gpus":names,"gpu_attestation":GPU_DIAG,"run_sha256":sha(OUT/"run.json"),"artifact_file_counts":counts,"data_boundary":{{"train_fit_only":True,"held_out_used":False,"audit_used":False,"test_used":False,"competition_submission_performed":False}},"dependencies":deps,"required_imports":imports}}
  with OK.open("x",encoding="ascii") as h: h.write(canon(evidence).decode("ascii")+"\\n")
  print(json.dumps({{"evidence":str(OK),"output":str(OUT),"selected_video_count":len(FIT_IDS),"selection_sha256":FIT_SELECTION_SHA256,"gpus":names,"gpu_attestation":GPU_DIAG}},sort_keys=True))
  return 0
 except BaseException as exc:
  fail(exc,stage,started); traceback.print_exc(file=sys.stderr); return 2
 finally: signal.setitimer(signal.ITIMER_REAL,0)
if __name__=="__main__": raise SystemExit(main())
'''


def transform_launcher(source: str, video_ids: Sequence[str], selection_sha256: str) -> str:
    start = source.rfind("\ndef main():\n")
    marker = '\nif __name__=="__main__": raise SystemExit(main())\n'
    finish = source.rfind(marker)
    if start < 0 or finish <= start or source.find("\ndef main():\n") != start:
        raise FitCacheBuildError("base launcher main block is not uniquely replaceable")
    transformed = source[: start + 1] + _replacement_main(video_ids, selection_sha256)
    compile(transformed, ENTRYPOINT, "exec")
    if len(transformed.encode("ascii")) >= pkgbh9.SOURCE_LIMIT:
        raise FitCacheBuildError("fit-cache launcher exceeds Kaggle source limit")
    return transformed


def _metadata(slug: str) -> bytes:
    value = {
        "id": f"onebitbrain/{slug}",
        "title": slug,
        "code_file": ENTRYPOINT,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": pkgbh9.KAGGLE_ACCELERATOR_ID,
        "competition_sources": [pkgbh9.COMPETITION_SLUG],
        "dataset_sources": [pkgbh9.SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def build_shard(output: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FitCacheBuildError(f"refusing to overwrite fit-cache package: {output}")
    ids = row.get("video_ids")
    selection_sha256 = row.get("selection_sha256")
    if not isinstance(ids, list) or not isinstance(selection_sha256, str):
        raise FitCacheBuildError("fit-cache shard row is malformed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fcbase-", dir=output.parent) as temporary:
        base = Path(temporary) / "base"
        pkgbh9.build_package(output_dir=base)
        source = (base / pkgbh9.ENTRYPOINT_NAME).read_text(encoding="ascii")
    slug = f"bh-fc0{int(row['index'])}"
    launcher = transform_launcher(source, ids, selection_sha256).encode("ascii")
    metadata = _metadata(slug)
    build_core = {
        "schema": "celltrack-alpha.fit-cache-package.v1",
        "kernel_ref": f"onebitbrain/{slug}",
        "entrypoint": {"path": ENTRYPOINT, "bytes": len(launcher), "sha256": sha256_bytes(launcher)},
        "selection": dict(row),
        "runtime": {
            "accelerator": "T4x2",
            "internet": False,
            "private": True,
            "deadline_seconds": FIT_DEADLINE_SECONDS,
            "child_deadline_minutes": CHILD_DEADLINE_MINUTES,
            "graph_variant": "a0",
            "competition_submission_performed": False,
        },
    }
    build_report = {**build_core, "identity_sha256": sha256_bytes(canonical_bytes(build_core))}
    output.mkdir(parents=True, exist_ok=False)
    (output / ENTRYPOINT).write_bytes(launcher)
    (output / METADATA).write_bytes(metadata)
    (output / BUILD_REPORT).write_bytes(canonical_bytes(build_report) + b"\n")
    return build_report


def build_all(
    output_root: Path = OUTPUT_ROOT,
    *,
    archive: Path = ARCHIVE,
    registry: Path = REGISTRY,
    split: Path = SPLIT,
    existing_root: Path = EXISTING,
    shard_count: int = DEFAULT_SHARDS,
) -> dict[str, Any]:
    plan = selection_plan(
        archive=archive,
        registry=registry,
        split=split,
        existing_root=existing_root,
        shard_count=shard_count,
    )
    packages = [
        build_shard(output_root / f"fc0{row['index']}", row)
        for row in plan["shards"]
    ]
    report = {"plan": plan, "packages": packages}
    plan_path = output_root / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(canonical_bytes(report) + b"\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--existing-root", type=Path, default=EXISTING)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    args = parser.parse_args(argv)
    report = build_all(
        args.output_root,
        archive=args.archive,
        registry=args.registry,
        split=args.split,
        existing_root=args.existing_root,
        shard_count=args.shards,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run frozen-held-out V45 D4 subsets and write local-scoring CSVs only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
MANIFEST = HERE / "vs45man.json"
CONFIG = HERE / "vs45cfg.json"
OUTPUT = WORK / "vs45out"


class ViewError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key %r" % key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject)


def _verify_package():
    payload = _load(MANIFEST)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise ViewError("view diagnostic package identity mismatch")
    observed = MANIFEST.stat().st_size
    for item in identity.get("files", []):
        path = HERE / str(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise ViewError("view diagnostic member identity mismatch: %s" % path.name)
        observed += path.stat().st_size
    if observed != identity.get("package_bytes") or observed >= identity.get("budget_bytes", 0):
        raise ViewError("view diagnostic package source budget mismatch")
    return payload


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) + b"\n")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _job(bh_run, competition, item):
    record = bh_run.RegistryRecord(
        item["split"], item["dataset"], item["video_id"], item["exact_identity_sha256"], tuple(item["image_shape"])
    )
    image = competition / "train" / (record.dataset + ".zarr")
    if not image.is_dir():
        raise ViewError("held-out train image is unavailable: %s" % image)
    return bh_run.VideoJob(record, "directory", str(competition), str(image))


def _predict_one(bh_model, bh_graph, bh_run, model, bundle, scorer, job, views):
    handle = bh_run.open_video(job)
    try:
        q_low, q_high = bh_run.read_quantile_attrs(handle.attrs)
        frames = bh_run.read_all_frames(handle.array, 100, None)
        identity = bh_model.CacheIdentity.from_artifact(bundle, d4_views=views)
        started = time.perf_counter()
        inference = bh_model.infer_video(
            model, frames, q_low=q_low, q_high=q_high, start_frame_index=0,
            node_id_start=0, batch_size=1, view_batch_size=1, d4_views=views,
        )
        if inference.d4_views != tuple(views):
            raise ViewError("inference D4 provenance differs from requested views")
        cache = SimpleNamespace(detections=inference.detections, contexts=inference.contexts)
        probabilities, audit = bh_run.score_a0_cache(cache, scorer, identity)
        nodes = bh_run.detections_to_graph_nodes(inference.detections, bh_graph.GraphNode)
        refiner = bh_run.LazyImageRefiner(handle.array, bh_graph.intensity_centroid_refine, deadline_epoch=None)
        graph = bh_graph.calibrate_v45_graph(
            nodes, learned_edge_probabilities=probabilities, image_refine_callback=refiner,
        )
        graph = bh_run.project_graph_result_to_image_support(
            job.record.dataset, graph, job.record.image_shape,
            scale_zyx=bh_graph.V45_GRAPH_CONFIG.scale_zyx,
            max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
        )
        rows = bh_run.graph_rows(job.record.dataset, graph, job.record.image_shape)
        return rows, {
            "seconds": time.perf_counter() - started,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "d4_views": list(inference.d4_views),
            "config_sha256": identity.config_sha256,
            "a0": audit,
            "lazy_refinement": refiner.report(),
        }
    finally:
        handle.close()


def main():
    started = time.monotonic()
    diag = {"schema": "bhv45-d4-view-run.v1", "ok": False, "submission_generation": False, "network": "forbidden"}
    try:
        package = _verify_package()
        config = _load(CONFIG)
        if package["identity"]["diagnostic"].get("config_sha256") != _sha_file(CONFIG):
            raise ViewError("view configuration identity differs from signed package")
        base = bh45._verify_package()
        diag["package_identity_sha256"] = package["identity_sha256"]
        diag["base_package_identity_sha256"] = base["identity_sha256"]
        diag["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diag)
        diag["dependencies_final"] = bh45._install_missing(diag["dependencies_before"], wheels, diag)
        bh45.SOURCE_ROOT = WORK / "vs45src"
        bh45._extract_source(base)
        competition = bh45._competition_root()
        if not (competition / "train").is_dir():
            raise ViewError("competition input does not expose train images")
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "src"))
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "scripts"))
        sys.path.insert(0, str(support / "repo" / "src"))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(bh45.SOURCE_ROOT / "src"), str(support / "repo" / "src")])
        import run_bh as bh_run
        from celltrack_alpha import bh_edge, bh_graph, bh_model
        import torch
        if not torch.cuda.is_available():
            raise ViewError("view diagnostic requires CUDA")
        model, bundle = bh_model.load_v45_model(support, device="cuda:0")
        full_identity = bh_model.CacheIdentity.from_artifact(bundle)
        scorer = bh_run.build_a0_scorer(bh_run.A0_GRAPH_VARIANT, model, bundle, full_identity, edge_module=bh_edge)
        targets = [_job(bh_run, competition, item) for item in config["targets"]]
        view_sets = [(str(item["name"]), tuple(int(value) for value in item["views"])) for item in config["view_sets"]]
        diag.update({"checkpoint_sha256": bundle.hashes.weight_sha256, "targets": [job.identity for job in targets], "runs": {}})
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, views in view_sets:
            if time.monotonic() - started >= float(config["deadline_seconds"]):
                raise ViewError("global diagnostic deadline reached before %s" % name)
            rows = []
            details = {}
            run_started = time.perf_counter()
            for job in targets:
                part, report = _predict_one(bh_model, bh_graph, bh_run, model, bundle, scorer, job, views)
                rows.extend(part)
                details[job.identity] = report
            for index, row in enumerate(rows):
                row["id"] = index
            bh_run.validate_submission_rows(rows, [job.record for job in targets])
            output = OUTPUT / (name + ".csv")
            _write_csv(output, rows, bh_run.CSV_FIELDS)
            diag["runs"][name] = {
                "views": list(views), "seconds": time.perf_counter() - run_started,
                "csv": output.name, "csv_sha256": _sha_file(output), "rows": len(rows), "videos": details,
            }
            _write_json(OUTPUT / "vs45run.json", diag)
        diag["elapsed_seconds"] = time.monotonic() - started
        diag["ok"] = True
        _write_json(OUTPUT / "vs45run.json", diag)
        return 0
    except Exception as exc:
        diag["error"] = "%s: %s" % (type(exc).__name__, exc)
        diag["elapsed_seconds"] = time.monotonic() - started
        _write_json(OUTPUT / "vs45run.json", diag)
        print(diag["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

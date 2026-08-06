#!/usr/bin/env python3
"""CPU-only metric runner for the sealed G-03 R4 nested-OOF protocol.

Only two development commands exist:

``inner``
    Decode every fixed configuration on exactly four development folds, score
    identity/graph-hash-unique graphs once, and emit ``INNER_SCORE_SCHEMA``.

``outer``
    Require an already-written inner-winner seal, decode exactly that winner on
    its untouched development fold, and emit ``OUTER_SCORE_SCHEMA``.

There is intentionally no held-fit, audit, hidden, submission, or GPU command.
The sole resumable state is a target-blind, content-addressed decoded-graph
cache.  Metric rows live only in an ephemeral SQLite spool and are deleted on
success or failure.  Final receipt bundles are atomically renamed into place.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


RUNNER_INPUT_SCHEMA = "celltrack-alpha.g03-r4-oof-runner-input.v1"
RUNNER_DIAGNOSTICS_SCHEMA = "celltrack-alpha.g03-r4-oof-runner-diagnostics.v1"
RUNNER_OUTPUT_SEAL_SCHEMA = "celltrack-alpha.g03-r4-oof-runner-output-seal.v1"
GRAPH_CACHE_SCHEMA = "celltrack-alpha.g03-r4-oof-graph-cache-entry.v1"
PRETARGET_PLAN_SCHEMA = "celltrack-alpha.g03-r4-oof-pretarget-plan.v1"
DEV_SEAL_SCHEMA = "celltrack-alpha.g03-r4-dev69-bank-seal.v1"
FOLD_IDS = ("0", "1", "2", "3", "4")
DEVELOPMENT_ROLE = "development_nested_oof"
HELD_ROLE = "held_fit_one_shot_confirmation"
AUDIT_ROLE = "excluded_contaminated_audit20"
HIDDEN_ROLE = "sealed_hidden_test_probe"
DEVELOPMENT_COUNT = 69
HELD_COUNT = 36
LOCAL105_COUNT = 105
CONFIGURATION_COUNT = 18_145
PARENT_CONFIG_INDEX = -1
PARENT_CONFIG_NAME = "__EXACT_SAME_VIDEO_DIVISIONS_OFF_PARENT__"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_RE = re.compile(r"^train:6bba_[0-9a-f]{8}$")
SCOREALL_DEPENDENCIES = (
    "src/celltrack_alpha/official_eval.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py",
)
ONE_PASS_DEPENDENCIES = (
    "src/celltrack_alpha/action_compiler.py",
    "src/celltrack_alpha/kaggle_calibration.py",
)
METRIC_ROW_FIELDS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
    "node_recall",
    "total_node_ratio",
    "edge_jaccard",
    "adj_edge_jaccard",
)


class OOFRunnerError(RuntimeError):
    """A production runner invariant failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OOFRunnerError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        paths = nonfinite_paths(value)
        diagnostic = ",".join(paths) if paths else "unknown-non-JSON-value"
        raise OOFRunnerError(
            "NONFINITE_JSON_DIAGNOSTIC: "
            f"paths={diagnostic}; aggregate metrics were not altered; no receipt written"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pretarget_configuration_names_sha256(
    decoder: ModuleType, names: Sequence[str]
) -> str:
    """Reproduce the decoder/orchestrator digest sealed in the pretarget plan."""

    return str(decoder.canonical_sha256(list(names)))


def nonfinite_paths(value: Any, prefix: str = "$") -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        result.append(prefix)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.extend(nonfinite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(nonfinite_paths(child, f"{prefix}[{index}]"))
    return tuple(result)


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{label} keys drifted; missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )


def _sha(value: Any, label: str) -> str:
    result = str(value)
    require(bool(SHA256_RE.fullmatch(result)), f"{label} is not a lowercase SHA-256")
    return result


def _resolve(base: Path, raw: Any) -> Path:
    candidate = Path(str(raw))
    return candidate if candidate.is_absolute() else base / candidate


def _ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    require(isinstance(value, Mapping), f"{label} must be a path/SHA reference")
    _strict(value, {"path", "sha256"}, label)
    return _resolve(base, value["path"]).resolve(), _sha(value["sha256"], label)


def verify_ref(value: Any, base: Path, label: str) -> tuple[Path, str]:
    path, expected = _ref(value, base, label)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    observed = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {observed}")
    return path, expected


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OOFRunnerError(f"{label} is not strict ASCII JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_verified_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    observed = sha256_file(path)
    require(observed == _sha(expected, label), f"{label} SHA-256 mismatch: {observed}")
    return load_json(path, label)


def absolute_ref(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(sha256, "reference hash")}


def atomic_file(path: Path, payload: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise OOFRunnerError(f"refusing to overwrite: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def atomic_output_bundle(
    output: Path,
    *,
    receipt_name: str,
    receipt: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, str]:
    """Publish receipt, diagnostics, and their seal with one directory rename."""

    output = output.resolve()
    require(not output.exists() and not output.is_symlink(), f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before creating a staging directory.  A NaN therefore leaves no
    # output at all and the caller receives the explicit diagnostic exception.
    receipt_bytes = canonical_bytes(receipt)
    diagnostics_bytes = canonical_bytes(diagnostics)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        receipt_path = stage / receipt_name
        diagnostics_path = stage / "RUNNER_DIAGNOSTICS.json"
        with receipt_path.open("xb") as handle:
            handle.write(receipt_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with diagnostics_path.open("xb") as handle:
            handle.write(diagnostics_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        seal = {
            "schema": RUNNER_OUTPUT_SEAL_SCHEMA,
            "status": "SEALED_COMPLETE_CPU_OOF_RUNNER_OUTPUT",
            "receipt": {
                "path": receipt_name,
                "sha256": sha256_file(receipt_path),
                "bytes": receipt_path.stat().st_size,
            },
            "diagnostics": {
                "path": diagnostics_path.name,
                "sha256": sha256_file(diagnostics_path),
                "bytes": diagnostics_path.stat().st_size,
            },
        }
        seal_path = stage / "SEAL.json"
        with seal_path.open("xb") as handle:
            handle.write(canonical_bytes(seal))
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.rename(stage, output)
        except OSError as exc:
            if output.exists() or output.is_symlink():
                raise OOFRunnerError(f"refusing to overwrite: {output}") from exc
            raise
        parent_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "output": str(output),
        "receipt_path": str(output / receipt_name),
        "receipt_sha256": seal["receipt"]["sha256"],
        "diagnostics_path": str(output / "RUNNER_DIAGNOSTICS.json"),
        "diagnostics_sha256": seal["diagnostics"]["sha256"],
        "seal_sha256": sha256_file(output / "SEAL.json"),
    }


def import_verified_module(path: Path, expected: str, name: str) -> ModuleType:
    observed = sha256_file(path)
    require(observed == _sha(expected, name), f"{name} SHA-256 mismatch: {observed}")
    specification = importlib.util.spec_from_file_location(
        f"g03_r4_{name.replace(' ', '_')}_{expected[:12]}", path
    )
    require(
        specification is not None and specification.loader is not None,
        f"cannot load {name}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(specification.name, None)
        raise
    return module


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True)
class VideoInput:
    identity: str
    fold: str
    association_bank_path: Path
    association_bank_sha256: str
    parent_graph_path: Path
    parent_graph_sha256: str
    bank: Any
    parent_graph: Any


@dataclass(frozen=True)
class FixedConfig:
    name: str
    value: Any


@dataclass(frozen=True)
class CoreContext:
    plan: Mapping[str, Any]
    plan_sha256: str
    folds: Mapping[str, tuple[str, ...]]
    videos: Mapping[str, VideoInput]
    configs: tuple[FixedConfig, ...]
    decoder: ModuleType
    orchestrator: ModuleType
    api: Mapping[str, Any]
    scorer_source: Mapping[str, str]
    execution_sources: Mapping[str, Any]
    source_hashes: Mapping[str, Any]
    input_manifest_sha256: str
    development_seal_sha256: str
    graph_cache: "GraphCache"


class Backend(Protocol):
    def __enter__(self) -> "Backend": ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    def score(self, identity: str, graph: Any, source: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def summarise(self, rows: list[dict[str, Any]]) -> Mapping[str, Any]: ...


class GraphCache:
    """Atomic, content-addressed cache of target-blind decoded graph bytes."""

    def __init__(
        self,
        root: Path,
        *,
        api: Mapping[str, Any],
        decoder_source_sha256: str,
        one_pass_dependency_sha256s: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.api = api
        self.decoder_source_sha256 = _sha(
            decoder_source_sha256, "decoder source"
        )
        self.one_pass_dependency_sha256s = {
            str(relative): _sha(digest, f"one-pass dependency {relative}")
            for relative, digest in sorted(
                (one_pass_dependency_sha256s or {}).items()
            )
        }
        self._validated: set[tuple[str, str]] = set()

    @staticmethod
    def _slug(identity: str) -> str:
        return identity.replace(":", "__", 1)

    def _entry(self, identity: str, graph_sha256: str) -> Path:
        _sha(graph_sha256, "semantic graph")
        return self.root / self._slug(identity) / graph_sha256

    def _expected_receipt(
        self,
        *,
        video: VideoInput,
        graph_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema": GRAPH_CACHE_SCHEMA,
            "status": "SEALED_TARGET_BLIND_CONTENT_ADDRESSED_GRAPH",
            "identity": video.identity,
            "semantic_graph_sha256": graph_sha256,
            "decoder_source_sha256": self.decoder_source_sha256,
            "one_pass_dependency_sha256s": self.one_pass_dependency_sha256s,
            "association_bank_sha256": video.association_bank_sha256,
            "parent_graph_sha256": video.parent_graph_sha256,
            "target_access": False,
            "gt_access": False,
            "scorer_access": False,
        }

    def validate(self, video: VideoInput, graph_sha256: str) -> Path:
        key = (video.identity, graph_sha256)
        entry = self._entry(*key)
        graph_path = entry / "graph.csv"
        receipt_path = entry / "CACHE_ENTRY.json"
        require(entry.is_dir() and not entry.is_symlink(), "partial graph-cache entry")
        require(
            {path.name for path in entry.iterdir()} == {"graph.csv", "CACHE_ENTRY.json"},
            "graph-cache entry contains unexpected files",
        )
        receipt = load_json(receipt_path, "graph-cache receipt")
        expected = self._expected_receipt(video=video, graph_sha256=graph_sha256)
        for field, value in expected.items():
            require(receipt.get(field) == value, f"graph-cache binding drifted: {field}")
        graph_file_sha = _sha(receipt.get("graph_file_sha256"), "cached graph file")
        require(
            graph_path.stat().st_size == int(receipt.get("graph_file_bytes", -1)),
            "cached graph byte count drifted",
        )
        require(sha256_file(graph_path) == graph_file_sha, "cached graph file hash drifted")
        graphs = self.api["read_submission"](graph_path)
        require(len(graphs) == 1, "cached graph file does not contain one graph")
        graph = next(iter(graphs.values()))
        require(graph.dataset == video.identity.split(":", 1)[1], "cached graph identity drifted")
        require(
            self.api["graph_sha256"](graph) == graph_sha256,
            "cached semantic graph hash drifted",
        )
        self._validated.add(key)
        return graph_path

    def publish(
        self,
        *,
        video: VideoInput,
        graph: Any,
        graph_sha256: str,
    ) -> tuple[Path, bool]:
        key = (video.identity, graph_sha256)
        entry = self._entry(*key)
        if key in self._validated:
            return entry / "graph.csv", True
        if entry.exists() or entry.is_symlink():
            return self.validate(video, graph_sha256), True
        require(
            self.api["graph_sha256"](graph) == graph_sha256,
            "decoder semantic graph hash drifted before cache publication",
        )
        parent = entry.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{graph_sha256}.staging-", dir=parent))
        try:
            graph_path = stage / "graph.csv"
            self.api["write_submission"]({graph.dataset: graph}, graph_path)
            with graph_path.open("rb") as handle:
                os.fsync(handle.fileno())
            receipt = {
                **self._expected_receipt(video=video, graph_sha256=graph_sha256),
                "graph_file_sha256": sha256_file(graph_path),
                "graph_file_bytes": graph_path.stat().st_size,
            }
            atomic_file(stage / "CACHE_ENTRY.json", canonical_bytes(receipt))
            descriptor = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.rename(stage, entry)
                reused = False
            except OSError:
                if not entry.exists() and not entry.is_symlink():
                    raise
                shutil.rmtree(stage)
                reused = True
            path = self.validate(video, graph_sha256)
            return path, reused
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise

    def load_graph(self, video: VideoInput, graph_sha256: str) -> Any:
        path = self.validate(video, graph_sha256)
        graphs = self.api["read_submission"](path)
        return next(iter(graphs.values()))


class MetricSpool:
    """Ephemeral graph/config and score rows; never used for resumption."""

    def __init__(self, root: Path) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix=".g03-r4-oof-spool-", dir=root))
        self.path = self.directory / "spool.sqlite3"
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=MEMORY")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.executescript(
            """
            CREATE TABLE config_graph (
                config_index INTEGER NOT NULL,
                config_name TEXT NOT NULL,
                identity TEXT NOT NULL,
                fold_id TEXT NOT NULL,
                graph_sha256 TEXT NOT NULL,
                predicted_forks INTEGER NOT NULL,
                duplicate_survivors INTEGER NOT NULL,
                PRIMARY KEY (config_index, identity)
            );
            CREATE INDEX config_graph_unique_graph
                ON config_graph(identity, graph_sha256);
            CREATE TABLE metric (
                identity TEXT NOT NULL,
                graph_sha256 TEXT NOT NULL,
                edge_tp TEXT NOT NULL,
                edge_fp TEXT NOT NULL,
                edge_fn TEXT NOT NULL,
                division_tp TEXT NOT NULL,
                division_fp TEXT NOT NULL,
                division_fn TEXT NOT NULL,
                num_pred_nodes TEXT NOT NULL,
                node_recall TEXT NOT NULL,
                total_node_ratio TEXT NOT NULL,
                edge_jaccard TEXT NOT NULL,
                adj_edge_jaccard TEXT NOT NULL,
                PRIMARY KEY (identity, graph_sha256)
            );
            """
        )

    def close(self) -> None:
        self.connection.close()
        shutil.rmtree(self.directory)

    def add_graph(
        self,
        *,
        config_index: int,
        config_name: str,
        identity: str,
        fold: str,
        graph_sha256: str,
        predicted_forks: int,
        duplicate_survivors: int,
    ) -> None:
        self.connection.execute(
            "INSERT INTO config_graph VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                config_index,
                config_name,
                identity,
                fold,
                graph_sha256,
                predicted_forks,
                duplicate_survivors,
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def unique_graphs(self) -> tuple[tuple[str, str], ...]:
        rows = self.connection.execute(
            "SELECT DISTINCT identity, graph_sha256 FROM config_graph "
            "ORDER BY identity, graph_sha256"
        )
        return tuple((str(identity), str(graph_sha)) for identity, graph_sha in rows)

    @staticmethod
    def _encode(value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OOFRunnerError("scorer metric row contains a nonnumeric field")
        if isinstance(value, int):
            return str(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value.hex()

    @staticmethod
    def _decode(value: str, *, count: bool) -> int | float:
        if count and re.fullmatch(r"-?[0-9]+", value):
            return int(value)
        if value in {"NaN", "Infinity", "-Infinity"}:
            return float(value)
        try:
            return float.fromhex(value)
        except ValueError:
            return float(value)

    def add_metric(self, identity: str, graph_sha256: str, row: Mapping[str, Any]) -> None:
        missing = set(METRIC_ROW_FIELDS) - set(row)
        require(not missing, f"scorer row lacks fields: {sorted(missing)}")
        values = [self._encode(row[field]) for field in METRIC_ROW_FIELDS]
        self.connection.execute(
            f"INSERT INTO metric VALUES ({','.join('?' for _ in range(13))})",
            (identity, graph_sha256, *values),
        )

    def rows_for_config(
        self, config_index: int
    ) -> tuple[tuple[str, str, str, int, int, dict[str, Any]], ...]:
        columns = ",".join(f"m.{field}" for field in METRIC_ROW_FIELDS)
        query = (
            "SELECT g.identity,g.fold_id,g.graph_sha256,g.predicted_forks,"
            "g.duplicate_survivors," + columns + " "
            "FROM config_graph g JOIN metric m "
            "ON g.identity=m.identity AND g.graph_sha256=m.graph_sha256 "
            "WHERE g.config_index=? ORDER BY g.identity"
        )
        result = []
        count_fields = set(METRIC_ROW_FIELDS[:7])
        for raw in self.connection.execute(query, (config_index,)):
            identity, fold, graph_sha256, predicted_forks, duplicate_survivors, *metric_values = raw
            row = {
                field: self._decode(str(value), count=field in count_fields)
                for field, value in zip(METRIC_ROW_FIELDS, metric_values, strict=True)
            }
            result.append(
                (
                    str(identity),
                    str(fold),
                    str(graph_sha256),
                    int(predicted_forks),
                    int(duplicate_survivors),
                    row,
                )
            )
        return tuple(result)


@dataclass
class RuntimeCounts:
    decoded_graph_requests: int = 0
    candidate_generation_requests: int = 0
    candidate_generation_reuses: int = 0
    full_graph_decodes: int = 0
    one_pass_graph_materializations: int = 0
    action_set_reuses: int = 0
    inert_action_set_requests: int = 0
    parent_graph_requests: int = 0
    graph_cache_hits: int = 0
    graph_cache_misses: int = 0
    scorer_invocations: int = 0


ActionSetKey = tuple[
    tuple[
        str,
        tuple[tuple[int, int], ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
    ],
    ...,
]


def _candidate_generation_key(config: Any) -> tuple[Any, ...]:
    """Return every configuration field read by ``generate_candidates``."""

    if not bool(getattr(config, "divisions", True)):
        return (False,)
    fields = (
        "fork_threshold",
        "max_primary_secondary_gap",
        "late_window",
        "daughter_persistence",
        "grandchild_persistence",
    )
    try:
        return (True, *(getattr(config, field) for field in fields))
    except AttributeError:
        # Synthetic decoder protocols may use an opaque value.  This preserves
        # correctness by disabling generation reuse for that value.
        return (True, type(config).__qualname__, repr(config))


def _event_action_set_key(
    events: Sequence[Any], *, decoder: ModuleType, incident_index: Any
) -> ActionSetKey:
    """Encode topology mutation semantics, deliberately excluding action IDs."""

    rows = []
    incident_conflicts = getattr(decoder, "_incident_conflicts", None)
    require(callable(incident_conflicts), "decoder lacks exact conflict-set API")
    for event in events:
        emitted_parent = int(event.emitted_parent)
        children = tuple(int(value) for value in event.children)
        require(len(children) == 2, "accepted division event does not have two children")
        conflicts = tuple(
            tuple(int(value) for value in pair)
            for pair in incident_conflicts(incident_index, emitted_parent, children)
        )
        rows.append(
            (
                "division_fork",
                tuple((emitted_parent, child) for child in children),
                (emitted_parent,),
                children,
                conflicts,
            )
        )
    result = tuple(sorted(rows))
    require(len(result) == len(set(result)), "accepted action semantics repeat")
    return result


def _actions_from_action_set(
    key: ActionSetKey, *, api: Mapping[str, Any]
) -> tuple[Any, ...]:
    graph_action = api.get("GraphAction")
    action_kind = api.get("ActionKind")
    require(graph_action is not None and action_kind is not None, "one-pass action API missing")
    set_sha = canonical_sha256(key)
    actions = []
    for index, row in enumerate(key):
        kind, add_edges, replace_sources, replace_targets, conflicts = row
        require(kind == "division_fork", "one-pass action-set kind drifted")
        actions.append(
            graph_action(
                action_id=f"r4.plan.{set_sha[:40]}.{index:08d}",
                kind=action_kind.DIVISION_FORK,
                add_edges=add_edges,
                replace_outgoing_from=replace_sources,
                replace_incoming_to=replace_targets,
                expected_conflict_edges=conflicts,
            )
        )
    return tuple(actions)


def _record_action_set_key(record: Mapping[str, Any]) -> ActionSetKey:
    accepted = record.get("accepted")
    require(isinstance(accepted, list), "full decoder record lacks accepted actions")
    rows = []
    for index, raw in enumerate(accepted):
        require(isinstance(raw, Mapping), f"accepted action {index} is malformed")
        action = raw.get("action")
        require(isinstance(action, Mapping), f"accepted action {index} lacks compiler semantics")
        rows.append(
            (
                str(action.get("kind")),
                tuple(tuple(int(value) for value in pair) for pair in action.get("add_edges", ())),
                tuple(int(value) for value in action.get("replace_outgoing_from", ())),
                tuple(int(value) for value in action.get("replace_incoming_to", ())),
                tuple(
                    tuple(int(value) for value in pair)
                    for pair in action.get("expected_conflict_edges", ())
                ),
            )
        )
    result = tuple(sorted(rows))
    require(len(result) == len(set(result)), "full decoder action semantics repeat")
    return result


def _planned_action_set(
    *,
    decoder: ModuleType,
    bank: Any,
    parent_graph: Any,
    config: Any,
    cached_generation: tuple[tuple[Any, ...], tuple[Any, ...]] | None,
    incident_index: Any,
    counts: RuntimeCounts,
) -> tuple[ActionSetKey, int, int, tuple[tuple[Any, ...], tuple[Any, ...]] | None]:
    generation_key = _candidate_generation_key(config)
    if generation_key == (False,):
        accepted: tuple[Any, ...] = ()
        suppression: Mapping[str, Any] = {
            "accepted_events": 0,
            "duplicate_survivors": 0,
        }
    else:
        generate = getattr(decoder, "generate_candidates", None)
        suppress = getattr(decoder, "suppress_candidates", None)
        require(callable(generate) and callable(suppress), "decoder lacks exact factorization API")
        if cached_generation is not None and cached_generation[0] == generation_key:
            events = cached_generation[1]
            rank = getattr(decoder, "_candidate_rank", None)
            if callable(rank):
                events = tuple(
                    sorted(events, key=lambda event: rank(event, str(getattr(config, "name", ""))))
                )
            counts.candidate_generation_reuses += 1
        else:
            generated, _generation = generate(bank, parent_graph, config)
            events = tuple(generated)
            cached_generation = (generation_key, events)
            counts.candidate_generation_requests += 1
        accepted_raw, _rejections, suppression = suppress(
            events, config, incident_index=incident_index
        )
        accepted = tuple(accepted_raw)
    predicted_forks = int(suppression.get("accepted_events", -1))
    duplicate_survivors = int(suppression.get("duplicate_survivors", -1))
    require(predicted_forks == len(accepted), "planned accepted-event count drifted")
    require(predicted_forks >= 0, "decoder predicted-fork count drifted")
    require(duplicate_survivors == 0, "decoder emitted a duplicate fork survivor")
    key = _event_action_set_key(
        accepted, decoder=decoder, incident_index=incident_index
    )
    if not key:
        counts.inert_action_set_requests += 1
    return key, predicted_forks, duplicate_survivors, cached_generation


def _decode_into_spool(
    *,
    context: CoreContext,
    configs: Sequence[FixedConfig],
    identities: Sequence[str],
    spool: MetricSpool,
    counts: RuntimeCounts,
    progress: Callable[[str], None] | None = None,
) -> None:
    generation_cache: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    action_cache: dict[str, dict[ActionSetKey, tuple[str, int, int]]] = {
        identity: {} for identity in identities
    }
    adjacency = getattr(context.decoder, "_adjacency", None)
    build_incident_index = getattr(context.decoder, "_build_incident_index", None)
    materialize_one_pass = getattr(
        context.decoder, "_materialize_division_actions_one_pass", None
    )
    require(
        callable(adjacency) and callable(build_incident_index),
        "decoder lacks exact frozen-baseline incident-index API",
    )
    incident_indexes = {
        identity: build_incident_index(
            adjacency(context.videos[identity].parent_graph)[2]
        )
        for identity in identities
    }
    parent_graph_sha256s = {
        identity: _sha(
            context.api["graph_sha256"](context.videos[identity].parent_graph),
            f"parent graph {identity}",
        )
        for identity in identities
    }
    production_one_pass = (
        callable(materialize_one_pass)
        and context.api.get("GraphAction") is not None
        and context.api.get("ActionKind") is not None
    )
    for config_index, config in enumerate(configs):
        for identity in identities:
            video = context.videos[identity]
            action_key, predicted_forks, duplicate_survivors, cached_generation = (
                _planned_action_set(
                    decoder=context.decoder,
                    bank=video.bank,
                    parent_graph=video.parent_graph,
                    config=config.value,
                    cached_generation=generation_cache.get(identity),
                    incident_index=incident_indexes[identity],
                    counts=counts,
                )
            )
            if cached_generation is not None:
                generation_cache[identity] = cached_generation
            cached_action = action_cache[identity].get(action_key)
            if cached_action is None:
                if production_one_pass:
                    actions = _actions_from_action_set(action_key, api=context.api)
                    decoded, materialization = materialize_one_pass(
                        baseline=video.parent_graph,
                        actions=actions,
                        api=context.api,
                        baseline_graph_sha256=parent_graph_sha256s[identity],
                    )
                    graph_sha = _sha(
                        materialization.get("after_graph_sha256"),
                        "one-pass decoded graph",
                    )
                    require(
                        materialization.get("compiler_path")
                        == "frozen_prepare_action_one_pass"
                        and materialization.get("action_count") == predicted_forks
                        and materialization.get("incident_index_built_once") is True
                        and materialization.get("full_graph_constructions") == 1
                        and materialization.get("full_graph_hashes") == 1,
                        "one-pass materialization receipt drifted",
                    )
                    counts.one_pass_graph_materializations += 1
                else:
                    decoded, record = context.decoder.decode_video(
                        video.bank, video.parent_graph, config.value, context.api
                    )
                    graph_sha = _sha(
                        record.get("decoded_graph_sha256"), "decoded graph"
                    )
                    observed_key = _record_action_set_key(record)
                    observed_forks = int(
                        record.get("suppression", {}).get("accepted_events", -1)
                    )
                    observed_duplicates = int(
                        record.get("suppression", {}).get("duplicate_survivors", -1)
                    )
                    require(
                        (observed_key, observed_forks, observed_duplicates)
                        == (action_key, predicted_forks, duplicate_survivors),
                        "full decoder differs from its exact action-set plan",
                    )
                cache_path, reused = context.graph_cache.publish(
                    video=video,
                    graph=decoded,
                    graph_sha256=graph_sha,
                )
                require(cache_path.is_file(), "content-addressed graph was not published")
                if reused:
                    counts.graph_cache_hits += 1
                else:
                    counts.graph_cache_misses += 1
                action_cache[identity][action_key] = (
                    graph_sha,
                    predicted_forks,
                    duplicate_survivors,
                )
                counts.full_graph_decodes += 1
            else:
                graph_sha, cached_forks, cached_duplicates = cached_action
                require(
                    (cached_forks, cached_duplicates)
                    == (predicted_forks, duplicate_survivors),
                    "action-set equivalent configurations disagree on suppression",
                )
                counts.action_set_reuses += 1
            counts.decoded_graph_requests += 1
            spool.add_graph(
                config_index=config_index,
                config_name=config.name,
                identity=identity,
                fold=video.fold,
                graph_sha256=graph_sha,
                predicted_forks=predicted_forks,
                duplicate_survivors=duplicate_survivors,
            )
        spool.commit()
        if progress is not None and (
            config_index == 0
            or config_index + 1 == len(configs)
            or (config_index + 1) % 100 == 0
        ):
            progress(
                f"planned {config_index + 1}/{len(configs)} fixed configs; "
                f"full_graph_decodes={counts.full_graph_decodes}; "
                f"action_set_reuses={counts.action_set_reuses}"
            )
    require(
        counts.full_graph_decodes + counts.action_set_reuses
        == counts.decoded_graph_requests,
        "exact action-set factorization lost a configuration/video request",
    )


def _register_parent_graphs(
    *,
    context: CoreContext,
    identities: Sequence[str],
    spool: MetricSpool,
    counts: RuntimeCounts,
) -> None:
    for identity in identities:
        video = context.videos[identity]
        graph_sha = _sha(context.api["graph_sha256"](video.parent_graph), "parent graph")
        cache_path, reused = context.graph_cache.publish(
            video=video,
            graph=video.parent_graph,
            graph_sha256=graph_sha,
        )
        require(cache_path.is_file(), "content-addressed parent graph was not published")
        counts.parent_graph_requests += 1
        if reused:
            counts.graph_cache_hits += 1
        else:
            counts.graph_cache_misses += 1
        spool.add_graph(
            config_index=PARENT_CONFIG_INDEX,
            config_name=PARENT_CONFIG_NAME,
            identity=identity,
            fold=video.fold,
            graph_sha256=graph_sha,
            predicted_forks=0,
            duplicate_survivors=0,
        )
    spool.commit()


def _score_unique_graphs(
    *,
    context: CoreContext,
    spool: MetricSpool,
    backend_factory: Callable[[], Backend],
    counts: RuntimeCounts,
) -> None:
    # Backend construction and __enter__ are the first operations allowed to open
    # the competition archive.  All source, plan, seal, config, winner, decode, and
    # cache validation has already completed by this point.
    with backend_factory() as backend:
        for identity, graph_sha in spool.unique_graphs():
            video = context.videos[identity]
            graph = context.graph_cache.load_graph(video, graph_sha)
            row = backend.score(
                identity,
                graph,
                source={
                    "mode": "content_addressed_graph_cache",
                    "semantic_graph_sha256": graph_sha,
                },
            )
            spool.add_metric(identity, graph_sha, row)
            counts.scorer_invocations += 1
        spool.commit()


def metric_bundle(
    *,
    rows: Sequence[Mapping[str, Any]],
    predicted_forks: int,
    summarise: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
    orchestrator: ModuleType,
    label: str,
) -> dict[str, Any]:
    require(bool(rows), f"{label} contains no metric rows")
    row_nonfinite = nonfinite_paths(rows)
    if row_nonfinite:
        raise OOFRunnerError(
            "NONFINITE_SCORER_METRIC: "
            f"paths={','.join(row_nonfinite)}; aggregate metrics were not altered; no receipt written"
        )
    summary = dict(summarise([dict(row) for row in rows]))
    edge = {
        field: sum(int(row[field]) for row in rows)
        for field in ("edge_tp", "edge_fp", "edge_fn")
    }
    division = {
        field: sum(int(row[field]) for row in rows)
        for field in ("division_tp", "division_fp", "division_fn")
    }
    value = {
        "combined_graph_score": summary.get("score"),
        "node_recall": summary.get("node_recall"),
        "predicted_forks": int(predicted_forks),
        "edge": {
            "tp": edge["edge_tp"],
            "fp": edge["edge_fp"],
            "fn": edge["edge_fn"],
            "jaccard": summary.get("edge_jaccard"),
            "adjusted_jaccard": summary.get("adj_edge_jaccard"),
        },
        "division": {
            "tp": division["division_tp"],
            "fp": division["division_fp"],
            "fn": division["division_fn"],
            "jaccard": summary.get("division_jaccard"),
        },
    }
    paths = nonfinite_paths(value)
    if paths:
        raise OOFRunnerError(
            "NONFINITE_AGGREGATE_METRIC: "
            f"paths={','.join(paths)}; aggregate metrics were not altered; no receipt written"
        )
    try:
        return orchestrator.validate_metric_bundle(value, label)
    except Exception as exc:
        raise OOFRunnerError(f"{label} violates the orchestrator metric schema: {exc}") from exc


def _summaries_for_config(
    *,
    spool: MetricSpool,
    config_index: int,
    inner_folds: Sequence[str],
    fold_identities: Mapping[str, Sequence[str]],
    summarise: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
    orchestrator: ModuleType,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = spool.rows_for_config(config_index)
    fold_evaluations: dict[str, dict[str, Any]] = {}
    all_rows = []
    all_forks = 0
    for fold in inner_folds:
        expected = list(fold_identities[fold])
        selected = [record for record in records if record[1] == fold]
        require(
            len(selected) == len(expected)
            and {record[0] for record in selected} == set(expected),
            f"metric rows do not exactly cover fold {fold}",
        )
        by_identity = {record[0]: record for record in selected}
        subset = [by_identity[identity][5] for identity in expected]
        forks = sum(by_identity[identity][3] for identity in expected)
        duplicates = sum(by_identity[identity][4] for identity in expected)
        metrics = metric_bundle(
            rows=subset,
            predicted_forks=forks,
            summarise=summarise,
            orchestrator=orchestrator,
            label=f"inner fold {fold}",
        )
        fold_evaluations[fold] = {
            "evaluated_identities": expected,
            "graph_sha256_by_identity": {
                identity: by_identity[identity][2] for identity in expected
            },
            "duplicate_survivors": duplicates,
            "metrics": metrics,
        }
        all_rows.extend(subset)
        all_forks += forks
    pooled = metric_bundle(
        rows=all_rows,
        predicted_forks=all_forks,
        summarise=summarise,
        orchestrator=orchestrator,
        label="pooled inner OOF",
    )
    return pooled, fold_evaluations


def _parent_folds(
    *,
    spool: MetricSpool,
    folds: Sequence[str],
    fold_identities: Mapping[str, Sequence[str]],
    summarise: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
    orchestrator: ModuleType,
) -> dict[str, dict[str, Any]]:
    records = spool.rows_for_config(PARENT_CONFIG_INDEX)
    result: dict[str, dict[str, Any]] = {}
    for fold in folds:
        expected = list(fold_identities[fold])
        selected = [record for record in records if record[1] == fold]
        require(
            len(selected) == len(expected)
            and {record[0] for record in selected} == set(expected),
            f"parent metric rows do not exactly cover fold {fold}",
        )
        by_identity = {record[0]: record for record in selected}
        rows = [by_identity[identity][5] for identity in expected]
        metrics = metric_bundle(
            rows=rows,
            predicted_forks=0,
            summarise=summarise,
            orchestrator=orchestrator,
            label=f"same-video parent fold {fold}",
        )
        gt_count = sum(int(row["division_tp"]) + int(row["division_fn"]) for row in rows)
        require(
            gt_count == metrics["division"]["tp"] + metrics["division"]["fn"],
            f"parent fold {fold} GT division count drifted",
        )
        result[fold] = {
            "reference_label": orchestrator.SAME_POPULATION_REFERENCE_LABEL,
            "evaluated_identities": expected,
            "graph_sha256_by_identity": {
                identity: by_identity[identity][2] for identity in expected
            },
            "gt_division_count": gt_count,
            "metrics": metrics,
        }
    return result


def _diagnostics(
    *,
    mode: str,
    context: CoreContext,
    counts: RuntimeCounts,
    unique_graphs: int,
    wall_started: float,
    cpu_started: float,
    outer_fold: str,
    configuration_count: int,
) -> dict[str, Any]:
    return {
        "schema": RUNNER_DIAGNOSTICS_SCHEMA,
        "status": "COMPLETE_CPU_ONLY",
        "mode": mode,
        "outer_fold": outer_fold,
        "configuration_count": configuration_count,
        "input_manifest_sha256": context.input_manifest_sha256,
        "pretarget_plan_sha256": context.plan_sha256,
        "development_seal_sha256": context.development_seal_sha256,
        "source_hashes": dict(context.source_hashes),
        "counts": {
            "decoded_graph_requests": counts.decoded_graph_requests,
            "candidate_generation_requests": counts.candidate_generation_requests,
            "candidate_generation_reuses": counts.candidate_generation_reuses,
            "full_graph_decodes": counts.full_graph_decodes,
            "one_pass_graph_materializations": counts.one_pass_graph_materializations,
            "action_set_reuses": counts.action_set_reuses,
            "avoided_full_graph_decodes": counts.action_set_reuses,
            "inert_action_set_requests": counts.inert_action_set_requests,
            "parent_graph_requests": counts.parent_graph_requests,
            "unique_identity_graphs": unique_graphs,
            "graph_cache_hits": counts.graph_cache_hits,
            "graph_cache_misses": counts.graph_cache_misses,
            "scorer_invocations": counts.scorer_invocations,
        },
        "resources": {
            "device": "cpu",
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "resumability": {
            "persistent_content_addressed_graph_cache": True,
            "persistent_metric_or_outcome_cache": False,
            "ephemeral_metric_spool_deleted": True,
        },
        "access_scope": {
            "development_only": True,
            "held_fit_access": False,
            "audit20_access": False,
            "hidden4_access": False,
            "gpu_hours": 0.0,
        },
    }


def run_inner_core(
    *,
    context: CoreContext,
    outer_fold: str,
    backend_factory: Callable[[], Backend],
    output: Path,
    work_root: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    require(outer_fold in FOLD_IDS, "inner mode outer fold is invalid")
    inner_folds = tuple(fold for fold in FOLD_IDS if fold != outer_fold)
    identities = tuple(identity for fold in inner_folds for identity in context.folds[fold])
    require(
        set(identities).isdisjoint(context.folds[outer_fold]),
        "outer holdout leaked into inner scoring",
    )
    require(
        len(context.configs) == int(context.plan.get("configuration_count", -1)),
        "inner mode does not contain every fixed configuration",
    )
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    counts = RuntimeCounts()
    spool = MetricSpool(work_root)
    try:
        _decode_into_spool(
            context=context,
            configs=context.configs,
            identities=identities,
            spool=spool,
            counts=counts,
            progress=progress,
        )
        _register_parent_graphs(
            context=context,
            identities=identities,
            spool=spool,
            counts=counts,
        )
        unique_graphs = len(spool.unique_graphs())
        _score_unique_graphs(
            context=context,
            spool=spool,
            backend_factory=backend_factory,
            counts=counts,
        )
        require(
            counts.scorer_invocations == unique_graphs,
            "graph-hash dedup did not match scorer invocation count",
        )
        rows = []
        # Backend.summarise does not require an open archive.  Use a fresh object
        # without entering it only when the factory explicitly provides this API.
        summary_backend = backend_factory()
        parent_folds = _parent_folds(
            spool=spool,
            folds=inner_folds,
            fold_identities=context.folds,
            summarise=summary_backend.summarise,
            orchestrator=context.orchestrator,
        )
        for config_index, config in enumerate(context.configs):
            pooled, folds = _summaries_for_config(
                spool=spool,
                config_index=config_index,
                inner_folds=inner_folds,
                fold_identities=context.folds,
                summarise=summary_backend.summarise,
                orchestrator=context.orchestrator,
            )
            rows.append(
                {
                    "configuration_name": config.name,
                    "fold_evaluations": folds,
                    "pooled_inner_oof": pooled,
                }
            )
        receipt = {
            "schema": context.orchestrator.INNER_SCORE_SCHEMA,
            "status": "SEALED_INNER_OOF_SCORES",
            "pretarget_plan_sha256": context.plan_sha256,
            "outer_fold": outer_fold,
            "inner_folds": list(inner_folds),
            "evaluated_identities": list(identities),
            "configuration_count": len(context.configs),
            "scorer_source": dict(context.scorer_source),
            "execution_sources": dict(context.execution_sources),
            "parent_folds": parent_folds,
            "rows": rows,
        }
        # Reuse the promoted selector as the final schema oracle.  Its returned
        # winner is deliberately discarded; only a later winner-seal command may
        # authorize an outer evaluation.
        context.orchestrator.select_inner_winner(
            plan=context.plan,
            receipt=receipt,
            configuration_names=tuple(config.name for config in context.configs),
            receipt_base=Path.cwd(),
        )
        diagnostics = _diagnostics(
            mode="inner",
            context=context,
            counts=counts,
            unique_graphs=unique_graphs,
            wall_started=wall_started,
            cpu_started=cpu_started,
            outer_fold=outer_fold,
            configuration_count=len(context.configs),
        )
        return atomic_output_bundle(
            output,
            receipt_name="INNER_SCORE_RECEIPT.json",
            receipt=receipt,
            diagnostics=diagnostics,
        )
    finally:
        spool.close()


def validate_winner(
    *,
    orchestrator: ModuleType,
    plan: Mapping[str, Any],
    plan_sha256: str,
    artifact: Mapping[str, Any],
    winner_sha256: str,
    outer_fold: str,
    configuration_names: set[str],
) -> Mapping[str, Any]:
    try:
        payload = orchestrator._validate_artifact(
            artifact,
            orchestrator.INNER_WINNER_SCHEMA,
            "SEALED_BEFORE_OUTER_HOLDOUT_EVALUATION",
            "inner winner seal",
        )
    except Exception as exc:
        raise OOFRunnerError(f"inner winner seal is invalid: {exc}") from exc
    required = {
        "pretarget_plan_sha256",
        "inner_score_receipt_sha256",
        "outer_fold",
        "outer_identities",
        "configuration_name",
        "pooled_inner_oof",
        "selection_order",
        "outer_evaluation_authorized_for_winner_only",
        "outer_metric_read_before_winner_seal",
    }
    _strict(payload, required, "inner winner payload")
    fold_plan = orchestrator._fold_plan(plan, outer_fold)
    require(payload.get("pretarget_plan_sha256") == plan_sha256, "winner plan binding drifted")
    require(str(payload.get("outer_fold")) == outer_fold, "winner outer fold drifted")
    require(
        list(payload.get("outer_identities", ())) == list(fold_plan["outer_identities"]),
        "winner outer identity scope drifted",
    )
    require(
        str(payload.get("configuration_name")) in configuration_names,
        "winner configuration is outside the sealed lattice",
    )
    require(
        payload.get("outer_evaluation_authorized_for_winner_only") is True
        and payload.get("outer_metric_read_before_winner_seal") is False,
        "winner does not authorize winner-only outer evaluation",
    )
    require(bool(SHA256_RE.fullmatch(winner_sha256)), "winner file SHA-256 is invalid")
    return payload


def run_outer_core(
    *,
    context: CoreContext,
    outer_fold: str,
    winner_artifact: Mapping[str, Any],
    winner_sha256: str,
    backend_factory: Callable[[], Backend],
    output: Path,
    work_root: Path,
) -> dict[str, str]:
    require(outer_fold in FOLD_IDS, "outer mode fold is invalid")
    by_name = {config.name: config for config in context.configs}
    winner = validate_winner(
        orchestrator=context.orchestrator,
        plan=context.plan,
        plan_sha256=context.plan_sha256,
        artifact=winner_artifact,
        winner_sha256=winner_sha256,
        outer_fold=outer_fold,
        configuration_names=set(by_name),
    )
    config = by_name[str(winner["configuration_name"])]
    identities = tuple(context.folds[outer_fold])
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    counts = RuntimeCounts()
    spool = MetricSpool(work_root)
    try:
        _decode_into_spool(
            context=context,
            configs=(config,),
            identities=identities,
            spool=spool,
            counts=counts,
        )
        _register_parent_graphs(
            context=context,
            identities=identities,
            spool=spool,
            counts=counts,
        )
        unique_graphs = len(spool.unique_graphs())
        _score_unique_graphs(
            context=context,
            spool=spool,
            backend_factory=backend_factory,
            counts=counts,
        )
        require(counts.scorer_invocations == unique_graphs, "outer scorer dedup drifted")
        summary_backend = backend_factory()
        _pooled, candidate_folds = _summaries_for_config(
            spool=spool,
            config_index=0,
            inner_folds=(outer_fold,),
            fold_identities=context.folds,
            summarise=summary_backend.summarise,
            orchestrator=context.orchestrator,
        )
        parent_folds = _parent_folds(
            spool=spool,
            folds=(outer_fold,),
            fold_identities=context.folds,
            summarise=summary_backend.summarise,
            orchestrator=context.orchestrator,
        )
        candidate = candidate_folds[outer_fold]
        parent = parent_folds[outer_fold]
        try:
            validated_parent = context.orchestrator._validate_parent_fold(
                parent,
                context.folds[outer_fold],
                "outer receipt parent",
            )
            validated_candidate = context.orchestrator._validate_candidate_fold(
                candidate,
                context.folds[outer_fold],
                gt_division_count=validated_parent["gt_division_count"],
                label="outer receipt candidate",
            )
        except Exception as exc:
            raise OOFRunnerError(
                f"outer receipt violates same-video comparison schema: {exc}"
            ) from exc
        receipt = {
            "schema": context.orchestrator.OUTER_SCORE_SCHEMA,
            "status": "SEALED_SINGLE_WINNER_OUTER_HOLDOUT_SCORE",
            "pretarget_plan_sha256": context.plan_sha256,
            "inner_winner_seal_sha256": winner_sha256,
            "outer_fold": outer_fold,
            "configuration_name": config.name,
            "evaluated_identities": list(identities),
            "scorer_source": dict(context.scorer_source),
            "execution_sources": dict(context.execution_sources),
            "candidate": validated_candidate,
            "parent": validated_parent,
            "gt_division_count": validated_parent["gt_division_count"],
        }
        diagnostics = _diagnostics(
            mode="outer",
            context=context,
            counts=counts,
            unique_graphs=unique_graphs,
            wall_started=wall_started,
            cpu_started=cpu_started,
            outer_fold=outer_fold,
            configuration_count=1,
        )
        return atomic_output_bundle(
            output,
            receipt_name="OUTER_SCORE_RECEIPT.json",
            receipt=receipt,
            diagnostics=diagnostics,
        )
    finally:
        spool.close()


def _config_value(decoder: ModuleType, row: Mapping[str, Any]) -> Any:
    expected = {
        "name",
        "divisions",
        "fork_threshold",
        "max_primary_secondary_gap",
        "late_window",
        "daughter_persistence",
        "grandchild_persistence",
        "duplicate_temporal_window",
        "per_video_cap",
    }
    _strict(row, expected, f"configuration {row.get('name')}")
    value = decoder.DivisionDecodeConfig(
        name=str(row["name"]),
        divisions=bool(row["divisions"]),
        fork_threshold=None if row["fork_threshold"] is None else float(row["fork_threshold"]),
        max_primary_secondary_gap=(
            None
            if row["max_primary_secondary_gap"] is None
            else float(row["max_primary_secondary_gap"])
        ),
        late_window=int(row["late_window"]),
        daughter_persistence=int(row["daughter_persistence"]),
        grandchild_persistence=int(row["grandchild_persistence"]),
        duplicate_temporal_window=int(row["duplicate_temporal_window"]),
        per_video_cap=None if row["per_video_cap"] is None else int(row["per_video_cap"]),
    )
    if value.divisions:
        require(
            value.fork_threshold is not None
            and value.max_primary_secondary_gap is not None
            and 0.0 <= value.fork_threshold <= 1.0
            and value.max_primary_secondary_gap >= 0.0
            and min(
                value.late_window,
                value.daughter_persistence,
                value.grandchild_persistence,
                value.duplicate_temporal_window,
            )
            >= 0
            and (value.per_video_cap is None or value.per_video_cap > 0),
            f"active configuration is malformed: {value.name}",
        )
    else:
        require(
            value.name == "DIVISIONS_OFF" and value.per_video_cap == 0,
            "null configuration drifted",
        )
    return value


def _source_path(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    require(path.is_relative_to(repo_root.resolve()), f"source path escaped repository: {relative}")
    return path


@dataclass(frozen=True)
class ProductionInputs:
    context: CoreContext
    scoreall: ModuleType
    archive_path: Path
    archive_sha256: str


def load_production_inputs(
    *,
    repo_root: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    graph_cache: Path,
) -> ProductionInputs:
    repo_root = repo_root.resolve()
    manifest = load_verified_json(input_manifest, input_manifest_sha256, "runner input")
    _strict(
        manifest,
        {
            "schema",
            "status",
            "pretarget_plan",
            "local105_inventory",
            "development_seal",
            "decoder_source",
            "orchestrator_source",
            "scoreall_source",
            "one_pass_dependencies",
            "scorer_dependencies",
            "competition_archive",
        },
        "runner input",
    )
    require(
        manifest.get("schema") == RUNNER_INPUT_SCHEMA
        and manifest.get("status") == "SEALED_BEFORE_R4_DEVELOPMENT_SCORING",
        "runner input is not pre-development-score sealed",
    )
    base = input_manifest.parent
    plan_path, plan_sha = verify_ref(manifest["pretarget_plan"], base, "pretarget plan")
    decoder_path, decoder_sha = verify_ref(manifest["decoder_source"], base, "decoder source")
    orchestrator_path, orchestrator_sha = verify_ref(
        manifest["orchestrator_source"], base, "orchestrator source"
    )
    scoreall_path, scoreall_sha = verify_ref(manifest["scoreall_source"], base, "scoreall source")
    expected_sources = {
        "decoder_source": repo_root / "scripts/g03_division_decode.py",
        "orchestrator_source": repo_root / "scripts/g03_division_oof_orchestrator.py",
        "scoreall_source": repo_root / "scripts/scoreall.py",
    }
    for label, (observed, expected) in {
        "decoder_source": (decoder_path, expected_sources["decoder_source"]),
        "orchestrator_source": (orchestrator_path, expected_sources["orchestrator_source"]),
        "scoreall_source": (scoreall_path, expected_sources["scoreall_source"]),
    }.items():
        require(observed == expected.resolve(), f"{label} path is not the repository source")

    dependency_refs = manifest.get("scorer_dependencies")
    require(isinstance(dependency_refs, Mapping), "scorer dependency map is missing")
    require(set(dependency_refs) == set(SCOREALL_DEPENDENCIES), "scorer dependency set drifted")
    dependency_hashes = {}
    for relative in SCOREALL_DEPENDENCIES:
        path, digest = verify_ref(dependency_refs[relative], base, f"scorer dependency {relative}")
        require(path == _source_path(repo_root, relative), f"scorer dependency path drifted: {relative}")
        dependency_hashes[relative] = digest

    one_pass_refs = manifest.get("one_pass_dependencies")
    require(isinstance(one_pass_refs, Mapping), "one-pass dependency map is missing")
    require(
        set(one_pass_refs) == set(ONE_PASS_DEPENDENCIES),
        "one-pass dependency set drifted",
    )
    one_pass_hashes = {}
    for relative in ONE_PASS_DEPENDENCIES:
        path, digest = verify_ref(
            one_pass_refs[relative], base, f"one-pass dependency {relative}"
        )
        require(
            path == _source_path(repo_root, relative),
            f"one-pass dependency path drifted: {relative}",
        )
        one_pass_hashes[relative] = digest

    # Dependency bytes are verified before importing scoreall, whose imports execute
    # official_eval and the two frozen external metric modules.
    decoder = import_verified_module(decoder_path, decoder_sha, "division decoder")
    require(
        getattr(decoder, "FROZEN_ACTION_COMPILER_SHA256", None)
        == one_pass_hashes["src/celltrack_alpha/action_compiler.py"]
        and getattr(decoder, "FROZEN_GRAPH_MODEL_SHA256", None)
        == one_pass_hashes["src/celltrack_alpha/kaggle_calibration.py"]
        and callable(getattr(decoder, "_build_incident_index", None))
        and callable(getattr(decoder, "_materialize_division_actions_one_pass", None)),
        "decoder one-pass source contract drifted",
    )
    if str(repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(repo_root / "scripts"))
    orchestrator = import_verified_module(
        orchestrator_path, orchestrator_sha, "OOF orchestrator"
    )
    scoreall = import_verified_module(scoreall_path, scoreall_sha, "scoreall")
    require(
        tuple(scoreall.SCORER_SOURCES) == SCOREALL_DEPENDENCIES,
        "scoreall frozen dependency list drifted",
    )
    plan_artifact = load_json(plan_path, "pretarget plan")
    try:
        plan = orchestrator._validate_artifact(
            plan_artifact,
            PRETARGET_PLAN_SCHEMA,
            "SEALED_PRE_TARGET_NO_SCORE_OR_GT_READ",
            "pretarget plan",
        )
    except Exception as exc:
        raise OOFRunnerError(f"pretarget plan is invalid: {exc}") from exc
    local105 = manifest.get("local105_inventory")
    require(isinstance(local105, Mapping), "local105 inventory binding is missing")
    _strict(
        local105,
        {"pretarget_plan_sha256", "coverage_sha256", "video_count"},
        "local105 inventory binding",
    )
    coverage = plan.get("fit105_coverage")
    require(isinstance(coverage, Mapping), "pretarget plan lacks local105 coverage")
    require(
        local105.get("pretarget_plan_sha256") == plan_sha
        and local105.get("coverage_sha256") == coverage.get("coverage_sha256")
        and local105.get("video_count") == LOCAL105_COUNT
        and coverage.get("video_count") == LOCAL105_COUNT
        and coverage.get("development_count") == DEVELOPMENT_COUNT
        and coverage.get("held_fit_count") == HELD_COUNT,
        "local105 inventory binding drifted",
    )
    records = coverage.get("records")
    require(isinstance(records, list) and len(records) == LOCAL105_COUNT, "local105 records drifted")
    development_rows = [row for row in records if isinstance(row, Mapping) and row.get("role") == DEVELOPMENT_ROLE]
    held_rows = [row for row in records if isinstance(row, Mapping) and row.get("role") == HELD_ROLE]
    require(
        len(development_rows) == DEVELOPMENT_COUNT and len(held_rows) == HELD_COUNT,
        "local105 roles are not exact dev69/held36",
    )
    require(
        all(str(row.get("identity", "")).startswith("train:44b6_") for row in held_rows),
        "held36 family drifted",
    )
    require(
        plan.get("forbidden_roles") == [AUDIT_ROLE]
        and all(row.get("role") not in {AUDIT_ROLE, HIDDEN_ROLE} for row in records),
        "development input contains audit20 or hidden4",
    )

    seal_path, seal_sha = verify_ref(manifest["development_seal"], base, "development seal")
    seal = load_json(seal_path, "development seal")
    require(
        seal.get("schema") == DEV_SEAL_SCHEMA
        and seal.get("status") == "SEALED_EXACT_DEV69_BEFORE_R4_GT_OR_SCORER_ACCESS"
        and seal.get("role") == DEVELOPMENT_ROLE
        and seal.get("identity_count") == DEVELOPMENT_COUNT
        and seal.get("association_banks_target_free") is True
        and seal.get("parent_graph_mode") == "DIVISIONS_OFF"
        and seal.get("sealed_before_R4_gt_or_scorer_access") is True
        and seal.get("gt_access") is False
        and seal.get("score_read") is False,
        "development seal target-blind contract drifted",
    )
    local_scope = seal.get("local_105_bank_scope")
    require(
        isinstance(local_scope, Mapping)
        and local_scope.get("development_video_count") == DEVELOPMENT_COUNT
        and local_scope.get("held_fit_video_count") == HELD_COUNT
        and local_scope.get("total_video_count") == LOCAL105_COUNT
        and local_scope.get("all_hashes_sealed_before_R4_development_scoring") is True,
        "development seal does not bind exact local105 pre-score coverage",
    )
    seal_rows = seal.get("videos")
    require(isinstance(seal_rows, list) and len(seal_rows) == DEVELOPMENT_COUNT, "dev69 seal rows drifted")
    sealed_by_identity = {
        str(row.get("identity")): row for row in seal_rows if isinstance(row, Mapping)
    }
    require(len(sealed_by_identity) == DEVELOPMENT_COUNT, "dev69 seal identities repeat")

    fold_rows = plan.get("outer_folds")
    require(isinstance(fold_rows, list) and len(fold_rows) == 5, "pretarget folds drifted")
    folds: dict[str, tuple[str, ...]] = {}
    for row in fold_rows:
        require(isinstance(row, Mapping), "pretarget fold row is malformed")
        fold = str(row.get("outer_fold"))
        require(fold in FOLD_IDS and fold not in folds, "pretarget fold identity drifted")
        identities = tuple(str(value) for value in row.get("outer_identities", ()))
        require(identities and len(identities) == len(set(identities)), "pretarget fold repeats identities")
        folds[fold] = identities
    development_identities = set().union(*(set(values) for values in folds.values()))
    require(len(development_identities) == DEVELOPMENT_COUNT, "five folds do not exactly cover dev69")
    require(development_identities == set(sealed_by_identity), "dev69 seal differs from folds")
    plan_development = {str(row.get("identity")): row for row in development_rows}
    require(set(plan_development) == development_identities, "local105 dev inventory differs from folds")

    api = decoder._compiler_api(repo_root)
    videos: dict[str, VideoInput] = {}
    fold_by_identity = {identity: fold for fold, values in folds.items() for identity in values}
    videos_root = seal_path.parent / "videos"
    for identity in sorted(development_identities):
        require(bool(IDENTITY_RE.fullmatch(identity)), f"non-dev69 identity in development: {identity}")
        sealed = sealed_by_identity[identity]
        plan_row = plan_development[identity]
        video_dir = videos_root / identity.replace(":", "__", 1)
        bank_path = (video_dir / "association_bank.npz").resolve()
        parent_path = (video_dir / "DIVISIONS_OFF_PARENT.csv").resolve()
        plan_bank_path, plan_bank_sha = _ref(plan_row.get("association_bank"), Path.cwd(), f"plan bank {identity}")
        plan_parent_path, plan_parent_sha = _ref(plan_row.get("parent_graph"), Path.cwd(), f"plan parent {identity}")
        require(bank_path == plan_bank_path and parent_path == plan_parent_path, f"{identity}: dev seal/path inventory drifted")
        require(
            plan_bank_sha == sealed.get("association_bank_sha256")
            and plan_parent_sha == sealed.get("parent_graph_sha256"),
            f"{identity}: dev seal/hash inventory drifted",
        )
        bank = decoder.load_association_bank(
            bank_path,
            expected_sha256=plan_bank_sha,
            expected_identity=identity,
        )
        parent_graph = decoder._load_base_graph(
            parent_path,
            expected_sha256=plan_parent_sha,
            identity=identity,
            bank=bank,
            api=api,
        )
        videos[identity] = VideoInput(
            identity=identity,
            fold=fold_by_identity[identity],
            association_bank_path=bank_path,
            association_bank_sha256=plan_bank_sha,
            parent_graph_path=parent_path,
            parent_graph_sha256=plan_parent_sha,
            bank=bank,
            parent_graph=parent_graph,
        )

    contracts = plan.get("contracts")
    require(isinstance(contracts, Mapping), "pretarget plan contracts are missing")
    config_path, config_sha = verify_ref(
        contracts.get("configuration_manifest"), Path.cwd(), "configuration manifest"
    )
    protocol_path, _protocol_sha = verify_ref(
        contracts.get("protocol"), Path.cwd(), "R4 protocol"
    )
    config_manifest = load_json(config_path, "configuration manifest")
    protocol = load_json(protocol_path, "R4 protocol")
    try:
        names = orchestrator._configuration_names(config_manifest, protocol)
    except Exception as exc:
        raise OOFRunnerError(f"configuration manifest is invalid: {exc}") from exc
    config_rows = config_manifest.get("configurations")
    require(isinstance(config_rows, list) and len(config_rows) == CONFIGURATION_COUNT, "configuration rows drifted")
    configs = tuple(
        FixedConfig(name=str(row["name"]), value=_config_value(decoder, row))
        for row in config_rows
        if isinstance(row, Mapping)
    )
    require(
        len(configs) == CONFIGURATION_COUNT
        and tuple(config.name for config in configs) == tuple(names)
        and plan.get("configuration_names_sha256")
        == pretarget_configuration_names_sha256(decoder, names),
        "fixed configuration lattice binding drifted",
    )
    archive_path, archive_sha = verify_ref(
        manifest["competition_archive"], base, "competition archive"
    )
    source_hashes = {
        "runner": sha256_file(Path(__file__).resolve()),
        "decoder": decoder_sha,
        "orchestrator": orchestrator_sha,
        "scoreall": scoreall_sha,
        "scoreall_dependencies": dependency_hashes,
        "one_pass_dependencies": one_pass_hashes,
        "configuration_manifest": config_sha,
    }
    execution_sources = {
        "runner": absolute_ref(Path(__file__).resolve(), source_hashes["runner"]),
        "decoder": absolute_ref(decoder_path, decoder_sha),
        "orchestrator": absolute_ref(orchestrator_path, orchestrator_sha),
        "scoreall": absolute_ref(scoreall_path, scoreall_sha),
        "scoreall_dependencies": {
            relative: absolute_ref(_source_path(repo_root, relative), dependency_hashes[relative])
            for relative in SCOREALL_DEPENDENCIES
        },
        "one_pass_dependencies": {
            relative: absolute_ref(
                _source_path(repo_root, relative), one_pass_hashes[relative]
            )
            for relative in ONE_PASS_DEPENDENCIES
        },
    }
    cache = GraphCache(
        graph_cache,
        api=api,
        decoder_source_sha256=decoder_sha,
        one_pass_dependency_sha256s=one_pass_hashes,
    )
    context = CoreContext(
        plan=plan,
        plan_sha256=plan_sha,
        folds=folds,
        videos=videos,
        configs=configs,
        decoder=decoder,
        orchestrator=orchestrator,
        api=api,
        scorer_source=absolute_ref(scoreall_path, scoreall_sha),
        execution_sources=execution_sources,
        source_hashes=source_hashes,
        input_manifest_sha256=input_manifest_sha256,
        development_seal_sha256=seal_sha,
        graph_cache=cache,
    )
    return ProductionInputs(
        context=context,
        scoreall=scoreall,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )


class ScoreAllBackend:
    def __init__(self, module: ModuleType, archive_path: Path, archive_sha256: str) -> None:
        self.module = module
        self.archive_path = archive_path
        self.archive_sha256 = archive_sha256
        self.store: Any = None

    def __enter__(self) -> "ScoreAllBackend":
        # Hash was verified before decoding.  Recheck immediately before target open.
        require(
            sha256_file(self.archive_path) == self.archive_sha256,
            "competition archive drifted before target access",
        )
        self.store = self.module.ZipStore(self.archive_path, mode="r")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    def score(self, identity: str, graph: Any, source: Mapping[str, Any]) -> Mapping[str, Any]:
        require(self.store is not None, "score backend is not open")
        return self.module.score_graph(self.store, identity, graph, source=source)

    def summarise(self, rows: list[dict[str, Any]]) -> Mapping[str, Any]:
        return self.module.official_summary(rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    for name in ("inner", "outer"):
        command = subcommands.add_parser(name)
        command.add_argument("--repo-root", required=True, type=Path)
        command.add_argument("--input-manifest", required=True, type=Path)
        command.add_argument("--input-manifest-sha256", required=True)
        command.add_argument("--outer-fold", required=True, choices=FOLD_IDS)
        command.add_argument("--graph-cache", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    outer = subcommands.choices["outer"]
    outer.add_argument("--winner-seal", required=True, type=Path)
    outer.add_argument("--winner-seal-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inputs = load_production_inputs(
        repo_root=args.repo_root,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        graph_cache=args.graph_cache,
    )
    factory = lambda: ScoreAllBackend(
        inputs.scoreall, inputs.archive_path, inputs.archive_sha256
    )
    work_root = args.output.resolve().parent
    work_root.mkdir(parents=True, exist_ok=True)
    if args.command == "inner":
        report = run_inner_core(
            context=inputs.context,
            outer_fold=args.outer_fold,
            backend_factory=factory,
            output=args.output,
            work_root=work_root,
            progress=lambda message: print(message, flush=True),
        )
    elif args.command == "outer":
        winner = load_verified_json(
            args.winner_seal, args.winner_seal_sha256, "inner winner seal"
        )
        report = run_outer_core(
            context=inputs.context,
            outer_fold=args.outer_fold,
            winner_artifact=winner,
            winner_sha256=args.winner_seal_sha256,
            backend_factory=factory,
            output=args.output,
            work_root=work_root,
        )
    else:  # pragma: no cover - argparse owns the finite command surface.
        raise OOFRunnerError(f"unsupported command: {args.command}")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

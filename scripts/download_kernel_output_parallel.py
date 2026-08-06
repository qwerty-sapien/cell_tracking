#!/usr/bin/env python3
"""Download a public Kaggle kernel output safely with bounded concurrency.

The Kaggle CLI downloads kernel output files serially.  Large outputs made of
thousands of modest files are dominated by request latency, so this helper
lists the same signed output URLs through the authenticated Kaggle API and
streams them concurrently.  Existing files are reused only when their byte
size matches the payload manifest; new files are installed atomically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)
import requests


class DownloadError(RuntimeError):
    """The listed output or downloaded bytes violated the frozen contract."""


def expected_sizes(manifest_path: Path) -> dict[str, int | None]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"sequences", "static"}:
        raise DownloadError("unexpected payload manifest schema")
    result: dict[str, int | None] = {}
    for section in ("sequences", "static"):
        rows = manifest[section]
        if not isinstance(rows, list):
            raise DownloadError(f"manifest {section} is not a list")
        for row in rows:
            relative = f"biohub_synthetic/{row['file']}"
            size = int(row["bytes"])
            if Path(relative).is_absolute() or ".." in Path(relative).parts or size <= 0:
                raise DownloadError(f"unsafe manifest row: {row}")
            if relative in result:
                raise DownloadError(f"duplicate manifest path: {relative}")
            result[relative] = size
    return result


def list_output_urls(kernel: str, *, page_size: int) -> dict[str, str]:
    if "/" not in kernel:
        raise DownloadError("kernel must be owner/slug")
    owner, slug = kernel.split("/", 1)
    result: dict[str, str] = {}
    token: str | None = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = page_size
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = str(item.file_name)
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise DownloadError(f"unsafe output path: {name}")
                if name in result:
                    raise DownloadError(f"duplicate output path: {name}")
                result[name] = str(item.url)
            token = response.next_page_token
            if not token:
                break
    return result


def _download_one(
    name: str,
    url: str,
    destination: Path,
    expected_size: int | None,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    target = destination / name
    if target.is_file() and expected_size is not None and target.stat().st_size == expected_size:
        return {"name": name, "bytes": expected_size, "status": "reused"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30.0, timeout_seconds),
        ) as response:
            response.raise_for_status()
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        observed_size = temporary.stat().st_size
        if expected_size is not None and observed_size != expected_size:
            raise DownloadError(
                f"byte-size mismatch for {name}: {observed_size} != {expected_size}"
            )
        os.replace(temporary, target)
        return {"name": name, "bytes": observed_size, "status": "downloaded"}
    finally:
        temporary.unlink(missing_ok=True)


def download(
    *,
    kernel: str,
    destination: Path,
    manifest_path: Path,
    workers: int,
    page_size: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if workers < 1 or workers > 32:
        raise DownloadError("workers must be in [1,32]")
    sizes = expected_sizes(manifest_path)
    urls = list_output_urls(kernel, page_size=page_size)
    required = set(sizes) | {
        "biohub_synthetic/manifest.json",
        "biohub_synthetic/metadata.json",
        "biohub_synthetic/progress.log",
    }
    if set(urls) != required:
        raise DownloadError(
            "kernel output inventory mismatch: "
            f"missing={sorted(required - set(urls))[:8]}, "
            f"extra={sorted(set(urls) - required)[:8]}, "
            f"listed={len(urls)}, expected={len(required)}"
        )
    # The two small control files precede this download and are authenticated
    # separately.  Their local sizes are still protected against truncation by
    # the HTTP response when a redownload is needed.
    sizes["biohub_synthetic/manifest.json"] = manifest_path.stat().st_size
    metadata_path = destination / "biohub_synthetic/metadata.json"
    if not metadata_path.is_file():
        raise DownloadError("metadata.json must be acquired before parallel download")
    sizes["biohub_synthetic/metadata.json"] = metadata_path.stat().st_size
    sizes["biohub_synthetic/progress.log"] = None

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                name,
                url,
                destination,
                sizes[name],
                timeout_seconds=timeout_seconds,
            ): name
            for name, url in sorted(urls.items())
        }
        for count, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise DownloadError(f"failed output file {name}: {exc}") from exc
            if count % 100 == 0 or count == len(futures):
                elapsed = time.monotonic() - started
                total_bytes = sum(int(row["bytes"]) for row in rows)
                print(
                    json.dumps(
                        {
                            "complete_files": count,
                            "total_files": len(futures),
                            "complete_gib": total_bytes / (1024**3),
                            "elapsed_seconds": elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    elapsed = time.monotonic() - started
    return {
        "schema": "celltrack-alpha.kaggle-kernel-output-parallel.v1",
        "kernel": kernel,
        "destination": str(destination.resolve()),
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "downloaded_files": sum(row["status"] == "downloaded" for row in rows),
        "reused_files": sum(row["status"] == "reused" for row in rows),
        "elapsed_seconds": elapsed,
        "workers": workers,
        "page_size": page_size,
        "status": "COMPLETE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = download(
        kernel=args.kernel,
        destination=args.destination,
        manifest_path=args.manifest,
        workers=args.workers,
        page_size=args.page_size,
        timeout_seconds=args.timeout_seconds,
    )
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_name(f".{args.receipt.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.receipt)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

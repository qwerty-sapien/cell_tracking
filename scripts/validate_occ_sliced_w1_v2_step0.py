#!/usr/bin/env python3
"""Execute the exact OCC sliced-W1 Step 0 against repair carriers."""

from __future__ import annotations

from pathlib import Path
import runpy
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CODE = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/occ_sliced_w1_v2"
    / "kaggle_kernel/kaggle_occ_sliced_w1_v2.py"
)
INPUT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/occ_sliced_w1_v1/input_dataset"
)
REMOTE_DATASET_ROOT = (
    "/kaggle/input/datasets/onebitbrain/g02-occ-sliced-w1-input-v1"
)
REMOTE_WORK_ROOT = "/kaggle/working/occ_sliced_w1_v2"


def execute_step0(dataset: Path, work_root: Path) -> dict[str, object]:
    source = CODE.read_text(encoding="utf-8")
    source = source.replace(REMOTE_DATASET_ROOT, str(dataset), 1)
    source = source.replace(REMOTE_WORK_ROOT, str(work_root), 1)
    script = work_root.parent / f"{work_root.name}.py"
    script.write_text(source, encoding="utf-8")
    return runpy.run_path(str(script), run_name="occ_sliced_w1_step0_test")


def expect_failure(dataset: Path, work_root: Path, message: str) -> None:
    try:
        execute_step0(dataset, work_root)
    except Exception as error:
        if message not in str(error):
            raise AssertionError(f"wrong Step-0 failure: {error}") from error
    else:
        raise AssertionError("Step 0 unexpectedly passed")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="occ-step0-") as temporary:
        temp = Path(temporary)

        namespace = execute_step0(INPUT, temp / "work-pass")
        step0 = namespace["STEP0"]
        assert step0["status"] == "PASS"
        assert step0["dataset_version"] == 1
        assert step0["request_count"] == 45_961
        assert step0["request_file_count"] == 36
        assert step0["wheel_count"] == 3

        legacy = temp / "legacy-mutable-carrier"
        legacy.mkdir()
        (legacy / "dinov2_cpu_bundle_manifest.json").write_text(
            "{}\n", encoding="ascii"
        )
        expect_failure(legacy, temp / "work-legacy", "dedicated input manifest mismatch")

        tampered = temp / "tampered-dedicated-carrier"
        shutil.copytree(INPUT, tampered)
        request = next(tampered.glob("feature_request__occupied_swap__*.npz"))
        with request.open("ab") as handle:
            handle.write(b"tamper")
        expect_failure(tampered, temp / "work-tamper", "byte-count drift")

    print("3 exact Step-0 repair-path checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

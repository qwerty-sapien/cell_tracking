# Acquisition and Analysis Status

As of 2026-07-08, all analysis in this workspace has respected the ZIP-only constraint. No Kaggle or CTC archive was extracted.

## Browser and Kaggle Status

- Chrome control was requested, but the Chrome connector reported no available browser session. I could not change Chrome's download directory from this environment.
- Workaround in place: `data/kaggle/raw/biohub-cell-tracking-during-development.zip` is a symlink to the in-progress Kaggle download at `/Users/agent/Desktop/biohub-cell-tracking-during-development.zip`.
- Current Kaggle archive status: `reports/kaggle_zip_inspection.md` reports `valid_zip=false`, file size `61.7 GB`, and `not a complete/valid ZIP: File is not a zip file`.
- Next action once the CLI download finishes: rerun `python3 scripts/inspect_zip_archives.py data/kaggle/raw/biohub-cell-tracking-during-development.zip --json-output reports/kaggle_zip_inspection.json --markdown-output reports/kaggle_zip_inspection.md`.

## CTC Archives Downloaded

The staged CTC auxiliary set is present under `data/ctc/raw/`:

| Archive | Size | Role |
|---|---:|---|
| `Fluo-N3DH-CHO.zip` | 104 MB | Small smoke test for nuclei detection and ZIP streaming |
| `Fluo-C3DL-MDA231.zip` | 184 MB | Small real 3D microscopy smoke test |
| `Fluo-N3DH-CE.zip` | 3.2 GB | Embryo lineage/division auxiliary data |
| `Fluo-N3DL-DRO.zip` | 5.8 GB | Closest downloaded public embryo analog by modality/scale |
| `Fluo-N3DH-SIM+.zip` | 3.2 GB | Simulated nuclei with dense division-heavy tracks |

`Fluo-N3DL-TRIC.zip` was not downloaded in this pass because it is much larger and lower priority for a first model pass.

## CTC Analysis Highlights

- `Fluo-N3DL-DRO`: valid ZIP, two raw sequences of 50 timepoints each, 16-bit volumes with 125 z-frames per sampled TIFF. Tracking annotations contain 189 and 203 tracks and no recorded division parents.
- `Fluo-N3DH-SIM+`: valid ZIP, raw sequence counts 150 and 80, 16-bit volumes with 59 z-frames per sampled TIFF. Tracking annotations contain 81/117 tracks and 37/40 division parents.
- `Fluo-N3DH-CE`: valid ZIP, raw sequence counts 250 and 250. Tracking annotations contain 720/724 tracks and 358/361 division parents.
- `Fluo-N3DH-CHO`: valid ZIP, raw sequence counts 92 and 92. Tracking annotations contain 27/24 tracks and 4/3 division parents.
- `Fluo-C3DL-MDA231`: valid ZIP, raw sequence counts 12 and 12. Tracking annotations contain 33/61 tracks and 0/1 division parents.

## Generated Workspace Artifacts

- `scripts/download_ctc_archives.sh`: resumable CTC downloader that keeps archives zipped.
- `scripts/download_kaggle_zip_to_workspace.sh`: Kaggle CLI helper that downloads only the ZIP.
- `scripts/inspect_zip_archives.py`: ZIP metadata, Kaggle-like zarr/geff/csv, and CTC track inspection without extraction.
- `scripts/analyze_ctc_zip_streaming.py`: streams TIFF members from CTC ZIPs into memory for lightweight image/track diagnostics and preview PNGs.
- `reports/ctc_*_zip_inspection.md`: ZIP central-directory and CTC annotation summaries.
- `reports/ctc_*_streaming_analysis.md`: streamed TIFF probes and sampled intensity statistics.
- `reports/plots/*.png`: derived preview images only, not extracted archive contents.

## Immediate Modeling Plan

1. Wait for the Kaggle ZIP to complete, then run the safe metadata probe.
2. Build the Kaggle manifest from ZIP metadata first: dataset names, zarr metadata, `.geff` entries, sample submission schema, and file sizes.
3. Implement baseline CV only after axes and physical voxel scale are verified against image-node overlays.
4. Start with multi-scale 3D DoG detection plus Hungarian `t -> t+1` linking, optimizing adjusted edge Jaccard before enabling division logic.
5. Use CTC data for smoke tests, pretraining, and stress tests, but report Kaggle CV separately because domain shift is material.

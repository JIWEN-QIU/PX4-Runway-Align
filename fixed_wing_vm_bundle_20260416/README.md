# Runway Segmentation Baseline

This repository contains the current PyTorch baseline for fixed-wing runway-region segmentation, with leave-one-dataset-out evaluation and an upgraded `all_data` training path.

## What is implemented

- `check_data.py`: validates prepared data, masks, and metadata splits.
- `dataset.py`: loads images and masks, supports explicit LOO splits, and builds the `all_data` monitor split.
- `model.py`: lightweight U-Net style binary segmentation model with configurable normalization.
- `utils.py`: loss, metrics, overlay utilities, GIF export, and geometry extraction hooks.
- `train.py`: fold training plus upgraded `all_data` training with monitor-set checkpoint selection.
- `predict.py`: split inference, overlay export, representative panel export, and sequence GIF export.
- `analyze_geometry.py`: geometry visualization for predicted masks, including centerline and principal direction overlays.
- `config.json`: current default experiment settings for the upgraded pipeline.
- `scripts/train_all_folds.ps1`: helper loop for training all five folds.
- `scripts/train_all_data.ps1`: helper entry for a full-data training run.
- `legacy/`: archived pre-promotion baseline files kept as local backup.

## Expected data layout

The current code assumes the prepared data has already been extracted to:

- `data/prepared/dataset_1 ... dataset_5`
- `data/metadata/all_samples.csv`
- `data/metadata/splits/loo_dataset_k_{train,test}.csv`

## Minimal usage

Check the prepared data:

```powershell
python check_data.py --data-root data/prepared --metadata-root data/metadata
```

Train one leave-one-dataset-out fold:

```powershell
python train.py --config config.json --test-dataset dataset_1
```

Train one full-data model with monitor-based checkpoint selection:

```powershell
python train.py --config config.json --mode all_data
```

Run prediction and visualization for the best checkpoint:

```powershell
python predict.py --checkpoint outputs/runway_unet_baseline/fold_dataset_1/<run_name>/checkpoints/best.pt
```

Run visualization on a standalone video and write an annotated `.mp4`:

```powershell
python predict_video.py --checkpoint outputs/runway_unet_baseline/all_data/<run_name>/checkpoints/best.pt --video path\to\input.mp4
```

By default, standalone video outputs now go under `outputs/test_videos/<video_stem>/` and keep full-frame processing with `--frame-stride 1`.

Run the five local test videos in one pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/predict_test_videos.ps1
```

For an `all_data` checkpoint, prediction defaults to the global sample list:

```powershell
python predict.py --checkpoint outputs/runway_unet_baseline/all_data/<run_name>/checkpoints/best.pt
```

Inspect geometry extracted from predicted masks:

```powershell
python analyze_geometry.py --predictions-dir outputs/runway_unet_baseline/fold_dataset_1/<run_name>/predictions
```

Audit geometry extraction quality across the latest five-fold outputs:

```powershell
python audit_geometry.py --experiment-root outputs/runway_unet_baseline
```

Smoke test on a smaller subset:

```powershell
python train.py --config config.json --test-dataset dataset_1 --epochs 1 --max-train-samples 8 --max-val-samples 4
```

For `--mode all_data`, `--max-train-samples` limits the training subset and `--max-val-samples` limits the monitor subset for a short smoke test.

## Outputs

Training writes under:

- `outputs/runway_unet_baseline/fold_<dataset>/<run_name>/checkpoints`
- `outputs/runway_unet_baseline/fold_<dataset>/<run_name>/logs`
- `outputs/runway_unet_baseline/all_data/<run_name>/checkpoints`
- `outputs/runway_unet_baseline/all_data/<run_name>/logs`

Prediction writes under:

- `.../predictions/masks`
- `.../predictions/overlays`
- `.../predictions/comparisons`
- `.../predictions/gifs`
- `.../predictions/geometry`
- `.../predictions/geometry_analysis`
- `outputs/runway_unet_baseline/geometry_audit`

## Notes

- The split logic is explicit and only uses the provided leave-one-dataset-out CSV files.
- `--mode all_data` now builds a per-dataset monitor split under `data/metadata/splits/all_data_{train,monitor}.csv` and selects `best.pt` on that monitor subset.
- The canonical file names now point to the upgraded pipeline; older entrypoints are archived under `legacy/`.
- The task remains binary runway segmentation only.
- The geometry utilities are intentionally lightweight and only serve as follow-up interfaces for later centerline and direction extraction work.
- `audit_geometry.py` scans the latest run under each fold directory and exports per-frame CSV plus a fold-level JSON summary for rapid regression checks after geometry-rule changes.

# UC Food Waste Binary Segmentation

Project for binary segmentation of plastic, glass, metal, and other contaminants in food waste. The project compares the collaborator's RGB U-Net baseline with RGB+Thermal and full 19-channel inputs.

## Folder layout

Keep this project separate from PIE. The code and local RGB/Mask data live under one UC-FoodWaste folder; the large multimodal NPZ files stay in shared Google Drive.

```text
repos/
├─ PIE-master/
└─ UC-FoodWaste/
   ├─ configs/
   ├─ metadata/data_split.csv
   ├─ notebooks/00_data_check.ipynb
   ├─ references/baseline_reference.ipynb
   ├─ scripts/check_data.py
   ├─ src/
   └─ data/
      ├─ archives/
      ├─ data_processed/images/       # 542 RGB JPGs
      ├─ data_processed/masks/        # 542 Binary Mask PNGs
      └─ splits/data_split.csv
```

Large data is excluded from Git. The multimodal NPZ directory is configured separately with `UC_FOODWASTE_NPZ_DIR`.

## Check local files in VS Code

Open `UC-FoodWaste` in VS Code. In its PowerShell terminal, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python scripts/check_data.py
```

By default, the checker uses the project's `data/` folder and `metadata/data_split.csv`. If the NPZ files are available locally, set their path before running the checker:

```powershell
$env:UC_FOODWASTE_NPZ_DIR = 'path to data_processed_multimodal\images'
```

The verified split is Train 374, Val 84, Test 84. RGB, Mask, and split filenames match. Original RGB and Masks are 1536x1024; resize both to 768x512 for training, using nearest-neighbor interpolation for Masks.

## Colab

Clone the repository and run the checked-in scripts; the training implementation does not need to be copied into notebook cells.

```bash
!git clone https://github.com/jeehyung0922-lab/UC-FoodWaste.git
%cd UC-FoodWaste
!pip install -r requirements.txt
```

After mounting Drive, point the code at the mounted folders. `UC_FOODWASTE_DATA_ROOT` must contain `data_processed/masks`, while `UC_FOODWASTE_NPZ_DIR` is the folder containing the 542 NPZ files.

```python
import os
os.environ["UC_FOODWASTE_DATA_ROOT"] = "/content/drive/MyDrive/path/to/data"
os.environ["UC_FOODWASTE_NPZ_DIR"] = "/content/drive/MyDrive/path/to/data_processed_multimodal/images"
```

Then audit and train from the repository files:

```bash
!python scripts/check_data.py
!python scripts/train.py --mode rgb_thermal --output-dir /content/drive/MyDrive/uc-foodwaste-runs
```

This repository is private. Authenticate with a GitHub account that has access before cloning.

## Reusable training pipeline

The pipeline reads filenames and splits from `metadata/data_split.csv`; it does not require train/val/test copies of the images. Every item loads `NPZ["image"]`, selects channels, converts HWC to CHW, loads the matching PNG mask, and applies the same horizontal flip to both image and mask.

Input modes are:

| mode | NPZ channels | batch image shape |
|---|---|---|
| `rgb` | `[2, 1, 0]` | `(B, 3, 512, 768)` |
| `rgb_thermal` | `[2, 1, 0, 18]` | `(B, 4, 512, 768)` |
| `all19` | `[2, 1, 0] + [3..18]` | `(B, 19, 512, 768)` |

Examples:

```bash
# Calculate and cache exact train-only mean/std without training.
python scripts/train.py --mode all19 --stats-only

# Train each comparable experiment.
python scripts/train.py --mode rgb
python scripts/train.py --mode rgb_thermal
python scripts/train.py --mode all19

# Useful when Colab Drive I/O benefits from fewer workers.
python scripts/train.py --mode all19 --batch-size 2 --num-workers 0
```

Defaults follow `references/baseline_reference.ipynb`: 512x768 input, U-Net with a ResNet-18 encoder, BCE + soft Dice loss, Adam at `1e-4`, 50 epochs, early-stopping patience 8, mixed precision on CUDA, threshold 0.5, and best-checkpoint selection by validation IoU. The reference notebook is not modified.

For each mode, outputs are written under `runs/<mode>/` (or the chosen `--output-dir`):

- `channel_stats.json`: normalization computed only from Train (374 images)
- `run_config.json`: channels and training hyperparameters used for the run
- `best.pt`: model state and reproducibility metadata
- `history.csv`: epoch-level train/validation metrics
- `test_results.csv`: per-image TP/FP/FN and area measurements
- `test_metrics.json`: aggregate Test metrics

Run `python scripts/train.py --help` for every path and hyperparameter option.

## Experiment rules

- Use the same `metadata/data_split.csv` for every input configuration.
- RGB channels: `[2, 1, 0]`; RGB+Thermal: `[2, 1, 0, 18]`; full input: `[2, 1, 0] + list(range(3, 19))`.
- Calculate per-channel normalization statistics from Train only.
- Apply the same geometric transforms to each image and its Mask. Use nearest-neighbor interpolation for Mask resizing.
- Keep the collaborator's baseline notebook unchanged in `references/` and reuse its evaluation setup for fair comparison.

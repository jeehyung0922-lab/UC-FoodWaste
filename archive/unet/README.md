# U-Net 연구 아카이브

이 디렉터리는 이전 U-Net 기준선과 RGB·멀티모달 비교 절차를 보존한다. 현재 연구의 실행 절차는 [루트 README](../../README.md)에 있다. 협업자 [참조 노트북](references/baseline_reference.ipynb)의 배포본은 이전 셀 출력·위젯 상태를 제거하고 개인 Drive 경로를 예제 경로로 바꿨다. 정리 전 파일은 로컬 `baseline_reference.local.ipynb`와 기존 Git 이력에 보존했다. 프로젝트에서 작성한 실행 스크립트와 Colab 노트북은 이동에 맞춰 경로를 조정했다.

공통 split은 저장소 루트의 `metadata/data_split.csv`, GT 마스크는 `data/data_processed/masks/`를 사용한다. 세 U-Net 모드 모두 별도의 멀티모달 NPZ 폴더가 필요하다. 공통 데이터는 기존 위치에 보존한다. 기본 U-Net 출력은 `archive/unet/runs/<mode>/`이고 `--output-dir`로 바꿀 수 있다. 정리 시점에 별도 U-Net 학습 결과 폴더는 확인되지 않았다.

저장소 루트에서 별도 가상환경을 설정하고 도움말을 확인하는 명령이다.

```powershell
python -m venv archive/unet/.venv
$py = '.\archive\unet\.venv\Scripts\python.exe'
& $py -m pip install -r archive/unet/requirements.txt
& $py archive/unet/scripts/train.py --help
& $py archive/unet/scripts/run_colab.py --help
```

## 기존 실행 문서

아래는 이전 루트 README에 있던 U-Net 절차다. 기존 학습 스크립트는 학습 종료 후 Test 평가도 실행한다. 학습과 Test 사용이 모두 명시적으로 승인된 경우에만 해당 명령을 실행한다.

Project for binary segmentation of plastic, glass, metal, and other contaminants in food waste. The project compares the collaborator's RGB U-Net baseline with RGB+Thermal and full 19-channel inputs.

## Check local files in VS Code

Open `UC-FoodWaste` in VS Code. In its PowerShell terminal, install dependencies and run:

```powershell
python -m pip install -r archive/unet/requirements.txt
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
!pip install -r archive/unet/requirements.txt
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
!python archive/unet/scripts/train.py --mode rgb_thermal --output-dir /content/drive/MyDrive/uc-foodwaste-runs
```

This repository is public. Access to the historical Drive dataset may require separate authorization.

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
python archive/unet/scripts/train.py --mode all19 --stats-only

# Train each comparable experiment.
python archive/unet/scripts/train.py --mode rgb
python archive/unet/scripts/train.py --mode rgb_thermal
python archive/unet/scripts/train.py --mode all19

# Useful when Colab Drive I/O benefits from fewer workers.
python archive/unet/scripts/train.py --mode all19 --batch-size 2 --num-workers 0
```

Defaults follow `archive/unet/references/baseline_reference.ipynb`: 512x768 input, U-Net with a ResNet-18 encoder, BCE + soft Dice loss, Adam at `1e-4`, 50 epochs, early-stopping patience 8, mixed precision on CUDA, threshold 0.5, and best-checkpoint selection by validation IoU. The reference notebook is not modified.

For each mode, outputs are written under `archive/unet/runs/<mode>/` (or the chosen `--output-dir`):

- `channel_stats.json`: normalization computed only from Train (374 images)
- `run_config.json`: channels and training hyperparameters used for the run
- `best.pt`: model state and reproducibility metadata
- `history.csv`: epoch-level train/validation metrics
- `test_results.csv`: per-image TP/FP/FN and area measurements
- `test_metrics.json`: aggregate Test metrics

Run `python archive/unet/scripts/train.py --help` for every path and hyperparameter option.

## Experiment rules

- Use the same `metadata/data_split.csv` for every input configuration.
- RGB channels: `[2, 1, 0]`; RGB+Thermal: `[2, 1, 0, 18]`; full input: `[2, 1, 0] + list(range(3, 19))`.
- Calculate per-channel normalization statistics from Train only.
- Apply the same geometric transforms to each image and its Mask. Use nearest-neighbor interpolation for Mask resizing.
- Keep the collaborator's baseline notebook unchanged in `archive/unet/references/` and reuse its evaluation setup for fair comparison.

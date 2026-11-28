# UC-FoodWaste · SegFormer-B0 연구 저장소

현재 연구는 팀원 SegFormer-B0 RGB 체크포인트의 Validation 성능 재현부터 시작한다. 팀원 원본을 `SegFormer/`에 보존했고, 최초 서버 실행과 재현 검증은 로컬 구성 검수 이후 진행한다. MSI와 Thermal 비교는 RGB 재현 이후 별도 설계한다.

- [SegFormer 독립 설치·실행 및 서버 검증](SegFormer/README.md)
- [팀원 원본·Split·체크포인트 식별 기록](SegFormer/OVERVIEW.md)
- [SF-RGB-R01 재현 계획](docs/experiments/SF-RGB-R01.md)
- [Common Evaluator v0.1](common_evaluator/README.md)

## 기존 YOLO 연구 문서

- [연구 개요](docs/RESEARCH_OVERVIEW.md) · [전체 파이프라인](docs/PIPELINE.md) · [데이터셋](docs/DATASET.md)
- [YOLO26s-sem 모델](docs/MODEL.md) · [평가 프로토콜](docs/EVALUATION.md)
- [환경과 파일 전송](docs/ENVIRONMENT.md) · [문제 해결](docs/TROUBLESHOOTING.md)
- [공통 실험 템플릿](docs/experiments/EXPERIMENT_TEMPLATE.md) · [Exp01 결과](docs/experiments/Exp01_RGB_Baseline.md) · [실험 인덱스](EXPERIMENT_LOG.md)
- [구조도와 시각화 재생성](docs/figures/README.md)

기존 YOLO26s-sem Exp01은 RGB, Background/Contaminant 2클래스, H512×W768과 고정 split을 사용했다. Train 374장으로 학습하고 Validation 84장으로 체크포인트와 threshold를 정하는 파이프라인을 보존한다. 최종 Test는 사용자 승인 후 실행한다.

## 저장소 구성

| 경로 | 역할 |
|---|---|
| `SegFormer/` | 활성 연구 프로젝트 및 보존된 팀원 코드 |
| `common_evaluator/` | 모델 독립 Validation 평가기 v0.1 |
| `configs/yolo/`, `requirements-yolo.txt` | 보존된 YOLO 실험 설정 및 Python 의존성 |
| `scripts/yolo/`, `src/yolo_sem/` | 데이터 준비, 학습, 확률맵 생성, threshold 선정과 평가 |
| `scripts/visualization/`, `docs/figures/` | CPU 시각화, 비교 도구와 연구 그림 |
| `metadata/data_split.csv`, `data/` | 고정 split, 공통 데이터, 준비 데이터 및 가중치 |
| `scripts/check_data.py`, `scripts/verify_transfer.py` | 공통 데이터 및 전송 SHA-256 검증 |
| `notebooks/00_data_check.ipynb`, `notebooks/yolo_rgb_colab.ipynb` | 공통 데이터 점검과 YOLO Colab |
| `runs/Exp01_RGB_Baseline/` | 체크포인트, 확률맵, 평가, 시각화 및 Error Analysis 정본 |
| [`archive/unet/`](archive/unet/README.md) | 이전 U-Net 연구 이력 |

대용량 데이터, 가중치, 실행 결과는 기존 Git 제외 정책을 따른다. 새 실험은 [평가 프로토콜](docs/EVALUATION.md)과 [공통 실험 양식](docs/experiments/EXPERIMENT_TEMPLATE.md)에 따라 기록한다.

## 기존 Exp01 · YOLO26s-sem RGB Baseline

### 1. Clone과 Python 환경

Windows PowerShell에서 실행한다. Exp01 GPU 서버의 Python은 3.13.15, PyTorch는 2.9.1+cu128, torchvision은 0.24.1+cu128이었다. RTX 5090에는 CUDA 12.8을 지원하는 NVIDIA 드라이버가 필요하다. 기록된 패키지 버전은 [Exp01 환경 기록](docs/experiments/exp01_results/environment.json)에 있다.

```powershell
git clone https://github.com/jeehyung0922-lab/UC-FoodWaste.git
Set-Location UC-FoodWaste
py -3.13 -m venv .venv-yolo-cuda
$py = '.\.venv-yolo-cuda\Scripts\python.exe'
& $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-yolo-exp01.txt
```

CPU 전용 PC에서는 마지막 4개 줄 대신 다음을 사용한다. CPU 환경은 준비, 평가 함수 테스트, dummy forward와 시각화를 지원한다.

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements-yolo.txt
```

### 2. 데이터 준비와 CPU 검증

Exp01과 같은 입력으로 재현하려면 보존된 UC-FoodWaste RGB JPG와 binary PNG 원본을 사용한다. `data_processed.zip`의 SHA-256과 분할·가중치 식별자는 [재현성 Manifest](metadata/exp01_inputs.json)에 있다. 팀원과 동일한 아카이브를 확보한 뒤 아래의 안내용 경로를 바꾼다. JPG·PNG가 이미 풀려 있다면 `$dataRoot/data_processed/{images,masks}`로 두고 import 단계만 생략한다. 공개 원본 배열만 가진 경우 입력 계약과 정확한 JPG 재현의 차이를 [데이터 문서](docs/DATASET.md#원본-가져오기)에서 확인한다.

```powershell
$dataRoot = Join-Path (Get-Location) 'data'
$prepared = Join-Path $dataRoot 'yolo_exp01'
$runRoot = Join-Path (Get-Location) 'runs'
$archive = 'PATH_TO_DATA_PROCESSED_ZIP'
Get-FileHash -Algorithm SHA256 -LiteralPath $archive
& $py scripts/yolo/exp01.py import-source --data-root $dataRoot --source-archive $archive
& $py scripts/yolo/exp01.py audit --data-root $dataRoot --run-root $runRoot
& $py scripts/yolo/exp01.py prepare --data-root $dataRoot --prepared $prepared --run-root $runRoot
& $py scripts/yolo/exp01.py weights --data-root $dataRoot
& $py scripts/yolo/test_cpu.py --prepared $prepared --weights (Join-Path $dataRoot 'yolo_weights/yolo26s-sem.pt')
```

데이터 준비는 고정 split의 Train 374·Validation 84만 기본적으로 읽고, Test 84는 CSV 메타데이터에서 확인한다. YAML은 자기 위치에 상대적으로 해석된다. 기존 준비 폴더가 있으면 재사용하거나 새 `--prepared`를 지정한다. `weights`는 Ultralytics 공식 릴리스 v8.4.0에서 정확한 가중치를 내려받아 SHA-256을 확인한다. 로컬 파일이 있으면 해시를 검증해 재사용한다. 기존 run이 있는 경우 새 `$runRoot`를 사용한다.

### 3. GPU Smoke와 정식 Exp01

아래는 팀원의 실행 절차다. GPU Smoke를 검토하고 정식 실험 승인을 받은 뒤 Train을 실행한다. GPU 서버에서 검증된 Exp01 설정은 **batch 16, epochs 20, seed 42, H512×W768**이다. 이식성 검증에서는 GPU 재학습을 수행하지 않았다.

```powershell
& $py scripts/yolo/exp01.py environment
$options = @('--data-root', $dataRoot, '--prepared', $prepared, '--run-root', $runRoot)
& $py scripts/yolo/exp01.py smoke --batch 2 @options
# Smoke 검토 및 승인 후
& $py scripts/yolo/exp01.py train --batch 16 --workers 0 @options
& $py scripts/yolo/exp01.py predict --split val @options
& $py scripts/yolo/exp01.py threshold @options
& $py scripts/yolo/exp01.py evaluate --split val @options
$run = Join-Path $runRoot 'Exp01_RGB_Baseline'
& $py scripts/visualization/render.py run $run --data-root $dataRoot --profile-checkpoint
& $py scripts/yolo/analyze_validation.py $run --data-root $dataRoot
& $py scripts/yolo/reproduce_dice_nan.py --run $run --out (Join-Path $run 'error_analysis/dice_reproduction.json')
```

Torch CUDA가 없으면 학습·추론 단계가 중단된다. 중단된 학습은 optimizer 상태가 있는 `weights/last.pt`로 재개한다. 전처리·시각화의 출력 폴더가 이미 존재하면 새 경로를 지정한다. Colab에서는 [YOLO 노트북](notebooks/yolo_rgb_colab.ipynb)에서 Drive 데이터 루트와 결과 경로를 지정한다.

평가 기준은 팀 공통 평가기 적용 전까지 잠정이다. [평가 규칙](docs/EVALUATION.md) · [Exp01 결과와 실행 설정](docs/experiments/Exp01_RGB_Baseline.md) · [소규모 원자료](docs/experiments/exp01_results/README.md). Validation Dice FP16 overflow는 CPU에서 재현했다. 수정에 대한 GPU 검증은 아직 수행하지 않았다. 최종 Test는 별도 승인 절차를 따른다.

### GPU 서버에서 코드 동기화

서버 작업 트리가 깨끗하면 저장소 루트에서 `git pull --ff-only origin main`을 사용한다. 기존 서버가 복사한 폴더이고 로컬 변경이 남아 있다면, 새 경로에 clone하고 데이터·준비·출력 경로를 명시하는 편이 안전하다. 기존 서버 CUDA 환경은 해당 환경의 Python 경로를 사용해 재사용할 수 있다. 머신과 경로에 맞게 준비한 YAML, 기존 체크포인트·확률맵과 결과를 보존한다.

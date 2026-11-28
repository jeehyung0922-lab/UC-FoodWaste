# UC-FoodWaste 실험 기록

## SegFormer-B0 연구

| 실험 ID | 상태 | 범위 |
|---|---|---|
| [SF-RGB-R01](docs/experiments/SF-RGB-R01.md) | 로컬 구성 완료, 재현 미실행 | 팀원 RGB 체크포인트의 Validation 추론 재현 |
| `SF-RGB-T01` | 미승인 | 필요 시 기존 RGB 설정의 재학습 |
| `SF-SENSOR-*` | 미승인 | 센서 가용성과 RGB 재현 검수 이후 비교 설계 |

원본 ZIP, 체크포인트와 Split 식별 정보는 [SegFormer 원본 기록](SegFormer/OVERVIEW.md)에 있다. 기존 팀 제출은 CPU FP32이며, 공통 평가기 v0.1에서 229/242개 영역을 탐지했다. 재현에서는 고정 Threshold 0.137과 재선정 Threshold 결과를 구분해 기록한다.

## 기존 YOLO 연구 · Exp01 RGB Baseline

## 실험 비교 인덱스

아래 비교는 동일한 잠정 Validation 프로토콜, 단일 seed 42, 20 epochs 기준이다. 영역 기반 값은 팀 평가기 정합성 확인 후 업데이트한다. 링크된 개별 문서에 상세 근거를 보존한다.

| 실험 | 상태 | 비교 요인 | Best epoch | Region Recall | Small Region Recall | Pixel Recall | IoU | FPR | 평균 추론 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [Exp01_RGB_Baseline](docs/experiments/Exp01_RGB_Baseline.md) | Val 완료 | RGB YOLO26s-sem 기준선 | 18 | 91.25% | 53.66% | 92.22% | 84.40% | 1.7644% | 4.94 ms |

Exp01의 **재현용 파이프라인 기준 커밋**은 `053b2c439290bfd67a23d43d205d1474e374008c`이며 `main`과 `origin/main`에서 확인했다. 당시 GPU 학습에 사용된 working tree의 Git SHA는 미확보다. 향후 기준 태그를 사용한다면 `exp01-reproduction-v1`처럼 재현 코드임을 표시한다.

| 실험 | 초기화 | 상태 |
|---|---|---|
| [Exp02_A_ADE20K](docs/experiments/Exp02_Weight_Init.md) | ADE20K 150클래스 사전학습 가중치 | 20 epochs 및 잠정 Validation 완료 |
| [Exp02_B_Random](docs/experiments/Exp02_Weight_Init.md) | YOLO26s-sem YAML 랜덤 초기화 | 20 epochs 및 잠정 Validation 완료 |

Exp02-A/B GPU 실행은 clean `812e2d02a30f71071a2d5089294970d95e4810f1`이다. Seed 42, 20 epochs, 잠정 Validation에서 ADE20K/Random은 각각 Region Recall 92.50%/80.42%, Small Region Recall 58.54%/31.71%, Pixel Recall 93.59%/79.11%, Contaminant IoU 84.82%/72.05%, FPR 1.9689%/1.8676%였다. 두 모델에 같은 1.7644% FPR 상한을 적용한 사후 비교에서 ADE20K와 Cityscapes의 차이는 한 영역이었다(220/240 대 219/240). 팀 공통 평가와 반복 실행 전에는 최종 초기화 방식을 확정하지 않는다. [비교·검증 및 FP32 진단](docs/experiments/exp02_results/README.md)

Small Region은 평가 격자에서 5,000 px 미만이다. 새 실험은 완료·실패·중단을 포함해 [공통 템플릿](docs/experiments/EXPERIMENT_TEMPLATE.md) 문서와 함께 추가한다. 평가 프로토콜과 추론 측정 구간이 다르면 별도 묶음에 기록한다.

| 항목 | 설정 또는 상태 |
|---|---|
| 실험 | Exp01_RGB_Baseline |
| 입력 | RGB, H512 × W768, Background=0, Contaminant=1 |
| 모델 | Ultralytics 8.4.52, YOLO26s-sem Cityscapes pretrained |
| 학습 | 20 epochs, seed 42, AMP 요청, batch CLI 기본 2, optimizer/LR/loss 라이브러리 기본 |
| 체크포인트 | Ultralytics Validation mIoU |
| 분할 | 고정 CSV. Train 374 / Val 84 / Test 84 |
| Threshold | Val Pixel FPR ≤ 0.02 아래 Region Recall 최대화. 잠정 규칙 |
| 실행 상태 | GPU 서버 정식 학습 20 epochs 및 Validation 완료. 로컬 다운로드 산출물 검증 완료 |

## 2026-11-26 검증

- Windows 11, Python 3.12.10, PyTorch 2.15.0+cpu, Ultralytics 8.4.52. CUDA 미가용.
- 542 RGB JPEG 및 542 PNG가 고정 split과 일치한다. 두 split CSV의 SHA-256이 같으며 교차 split에서 동일한 decoded RGB는 발견되지 않았다. 로컬 ZIP과 파일 1,084개가 바이트 단위로 일치한다. 팀원 환경의 파일이나 Drive NPZ와는 직접 대조 전이다.
- RGB uint8 범위는 0–255, 원본 마스크 값은 0/255, 변환은 0/1이다. Train/Val/Test 무이물질 이미지는 90/20/19장이다. H512×W768에서 잠정 8연결 영역은 1,033/240/260개다. 4연결에서 Val 영역은 243개이고 SegFormer의 제공 수치는 242개이므로 영역 평가를 직접 대조할 수 없다.
- 공식 체크포인트를 다운받아 CPU에서 19클래스 semantic 모델로 읽고, 동일 구조의 2클래스 모델로 전이 가중치를 로딩했다. train/val 배치 RGB [2,3,512,768], mask [2,512,768], dummy logits [1,2,64,96], 확률맵 [1,512,768]. 학습 affine의 경계에는 공식 ignore ID 255가 나타났다. 변환한 원본 GT와 Val에는 0과 1만 있다.
- 가중치 SHA-256 `bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435`.
- 팀원과 RGB, 고정 분할, 이진 foreground, H512×W768, SegFormer 20 epoch 및 FPR 한도를 맞췄다. YOLO는 Cityscapes 전이 학습, RGB /255, 공식 semantic loss 및 validation mIoU를 사용한다. 팀원의 실제 전처리·연결성·sPRO/AUsPRO·동률 처리와 Test 결과는 미확보다. SegFormer threshold 0.137은 참고 수치다.
- GPU 검증 항목은 AMP 실제 활성화, CUDA forward/backward, parameter update, 메모리, 저장 및 재개, Val/Test 추론이다. 정식 학습 전 smoke 결과를 검토한다.

## GPU 정식 학습 및 Validation 결과

산출물 경로는 `runs/Exp01_RGB_Baseline/`이다. NVIDIA GeForce RTX 5090에서 Python 3.13.15, PyTorch 2.9.1+cu128, Ultralytics 8.4.52로 실행했다. 20 epochs, 실제 batch 16, seed 42, AMP 활성화, 모델 입력 H512 × W768을 확인했다. Optimizer는 auto가 선택한 AdamW이고, 적용된 초기 LR은 0.001667, scheduler는 LambdaLR, 가중치 그룹 decay는 0.0005였다. 기록된 경과 시간은 368.865초, PyTorch 최대 할당 GPU 메모리는 4,199,519,744바이트(3.91 GiB)다. 모델은 6,498,756 parameters이며, batch 1 FP32 CUDA forward·보간·softmax 평균 시간은 4.94 ms였다.

### Epoch 기록

`history.csv`와 `results.csv`가 바이트 단위로 일치하며 epoch 1–20 기록이 연속이다. 아래 mIoU는 Ultralytics의 클래스 평균 지표다. 별도 threshold 적용 결과의 IoU는 Contaminant 픽셀 기준이다.

| Epoch | Train CE | Train Dice loss | Train Aux loss | Val CE | Val mIoU | Val pixel accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.46489 | 0.38855 | 0.16276 | 0.27909 | 75.56% | 91.27% |
| 2 | 0.19912 | 0.16090 | 0.07791 | 0.21294 | 84.10% | 95.00% |
| 5 | 0.12935 | 0.10075 | 0.05231 | 0.48433 | 63.72% | 83.12% |
| 10 | 0.10547 | 0.08389 | 0.04270 | 0.08824 | 88.97% | 96.80% |
| 14 | 0.08814 | 0.06948 | 0.03517 | **0.07742** | 90.40% | 97.23% |
| 18 | 0.07797 | 0.05839 | 0.03083 | 0.08050 | **90.602%** | **97.273%** |
| 20 | 0.07020 | 0.05594 | 0.02844 | 0.08016 | 90.482% | 97.235% |

세 Train loss는 전반적으로 감소했다. Val mIoU는 초반에 크게 변동했고 epoch 13–20에서 89.79–90.60% 범위에 머물렀다. Val CE 최저는 epoch 14, 선택 기준인 Val mIoU 최고는 epoch 18이었다. `val/dice_loss`는 20개 epoch 모두 NaN이며 원인은 이 산출물에서 확인되지 않았다. 저장 확률맵으로 계산한 Validation Dice score는 유한하다.

### Checkpoint와 Validation 산출물 검증

- `best.pt`와 `last.pt` 각각을 `weights/` 복사본과 SHA-256으로 대조했다. 두 체크포인트는 ZIP CRC 검사와 CPU 로딩을 통과했고 모두 Background/Contaminant 2클래스 semantic 모델이다. `best.pt` 내 mIoU 0.90602는 epoch 18 기록과 일치한다. 두 파일은 완료 후 optimizer가 제거된 추론용 체크포인트다.
- `best.pt` SHA-256은 `0ff4eed9a78467ecc665478ae7d37fac2b75dd2de003fcf793642d8b9e7afda9`이다. 이 해시, split CSV 해시, 준비 데이터 요약 해시가 확률맵 manifest 및 threshold의 평가 식별정보와 일치한다. 기록된 초기 가중치 해시도 로컬 공식 파일과 일치한다.
- `probability_maps/val/`의 NPY 84장, `predictions/val/`의 PNG 84장, `visualizations/val/`의 PNG 84장, `val_results.csv`의 84개 ID가 고정 Validation split과 일치했다. 확률맵은 모두 H512×W768 float32이고 값은 유한하며 [0,1] 안에 있다. 저장 확률맵에 threshold를 적용한 마스크 84장이 저장 예측 PNG와 픽셀 단위로 일치했다.
- 로컬 준비 GT와 저장 확률맵으로 1,002개 후보를 재계산했다. 제한을 만족한 후보는 585개였다. FPR ≤2%에서 최대 219/240 영역을 검출한 후보 중 FPR이 가장 낮은 **0.501**을 재현했다. 이미지별 TP/FP/FN/TN과 영역 카운트, 전체 threshold sweep 및 잠정 AUsPRO가 저장 결과와 일치했다.

| Validation 지표 | 결과 |
|---|---:|
| Region Recall (잠정) | 91.25% (219/240) |
| Pixel Recall | 92.22% |
| Contaminant IoU | 84.40% |
| Pixel FPR | 1.7644% |
| Pixel Precision | 90.87% |
| Dice score | 91.54% |
| sPRO (잠정) | 82.43% |
| AUsPRO (잠정) | 0.705476 |

픽셀 집계는 TP 4,873,647, FP 489,548, FN 411,120, TN 27,255,829다. B 그룹 오탐은 2/20장(10%), 평균 예측 이물질 면적 비율은 0.2359%다. FPR 2% 제한은 전체 Validation 기준이다. A 그룹 FPR은 2.6646%였고, C·D 그룹은 각각 4장·10장으로 표본이 작다.

### 팀 공통 평가 재현

팀 평가기가 이미지 ID에 대응하는 H512×W768 이물질 확률과 GT를 받으면, 저장된 NPY 84장으로 재학습이나 재추론 없이 Validation을 재평가할 수 있다. GT의 원본/리사이즈 규칙, 연결성, 영역 제외 조건, sPRO/AUsPRO 산식, threshold 탐색·동률 처리를 팀 평가 코드로 확인해야 한다. 팀 포맷이 다르면 별도 어댑터나 재추론이 필요할 수 있다. 기존 산출물은 그대로 보관하고 공통 평가 결과는 별도 위치에 저장한다. 이번 확인에서 Test 산출물은 발견되지 않았고 추가 학습과 Test 평가는 실행하지 않았다.

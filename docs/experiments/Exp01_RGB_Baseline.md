# Exp01_RGB_Baseline

## Experiment ID 및 상태
Exp01_RGB_Baseline. 정식 20 epochs, Validation 추론·threshold 보정·평가 및 사후 오류 분석 완료. Test 미실행.

## 실험 목적
향후 개선 및 Ablation에 사용할 RGB 이진 semantic segmentation 기준선을 확보한다.

## Research Question
YOLO26s-sem RGB 모델이 고정 Validation에서 FPR 2% 이하를 유지하며 검출할 수 있는 이물질 영역의 비율은 얼마인가?

## Hypothesis
RGB 사전학습 모델이 동일한 입력 격자와 FPR 제한에서 비교 가능한 기준선을 제공한다. 초기 성능 하한은 따로 사전 등록되지 않았다.

## 이전 실험과의 관계
첫 YOLO 기준선이다. SegFormer/U-Net은 팀 비교 대상이다. 팀 공통 평가 기준을 확보한 뒤 비교한다.

## 변경 변수 및 통제 변수
모델은 YOLO26s-sem으로 고정했다. 입력은 RGB이며 split 및 H512×W768 격자를 유지했다. 단일 seed 기준선이므로 모델/옵션 각각의 인과 효과를 판단할 수 없다.

## 성공 판단 기준
20 epochs와 재현 가능한 체크포인트·확률맵·평가 산출물을 확보하고, Validation에서 전체 FPR ≤2%인 threshold 후보 중 Region Recall을 최대화한다.

## 모델 및 학습 설정
Cityscapes 19클래스에서 Background/Contaminant 2클래스로 전이했다. epochs 20, seed 42, batch 16, workers 0, AMP 활성화, rect=True. optimizer auto가 AdamW를 선택했고 유효 초기 LR 0.001667, LambdaLR, weight decay 0.0005였다. 설정의 lr0 요청값 0.01과 적용된 초기 LR을 구별한다. [설정 YAML](../../configs/yolo/exp01_rgb.yaml) · [모델 및 loss](../MODEL.md)

## 데이터 및 평가 설정
Train 374 / Val 84 / Test 84 고정 split. Validation A/B/C/D 이미지 50/20/4/10. 현재 연결성 8, 최소 10% overlap, 평가 H512×W768, FPR 제한 2%는 [잠정 평가 규칙](../EVALUATION.md)을 따른다.

## 실행 환경
Windows Server 2022, RTX 5090, Python 3.13.15, PyTorch 2.9.1+cu128, Ultralytics 8.4.52. [환경 문서](../ENVIRONMENT.md)

## 실행 명령어
기록된 설정에 해당하는 CLI 재현 명령이다. 셸 원문 전체는 산출물에 없다.

```powershell
python scripts/yolo/exp01.py train --batch 16 --workers 0 --weights data/yolo_weights/yolo26s-sem.pt
python scripts/yolo/exp01.py predict --split val
python scripts/yolo/exp01.py threshold
python scripts/yolo/exp01.py evaluate --split val
```

기존 경로에 재실행할 때는 새 run root를 지정한다. 정식 학습은 별도 승인 후 실행한다.

## 학습 결과
best epoch **18**, Val mIoU **90.602%**, Val pixel accuracy **97.273%**. epoch 20 Val mIoU 90.482%. Train CE 0.46489→0.07020, Train Dice loss 0.38855→0.05594, Train aux 0.16276→0.02844. Val CE 최저는 epoch 14의 0.07742. 총 368.865 s, train peak allocated 3.91 GiB. 추론 평균 4.94 ms, median 5.044 ms, peak allocated 약 112.43 MiB. 파라미터 6,498,756, 같은 입력에서 THOP 추정 16.739 GFLOPs.

## Validation 성능
threshold **0.501**, 후보 1,002개 중 FPR 조건 만족 585개. 전체 픽셀 TP/FP/FN/TN은 4,873,647 / 489,548 / 411,120 / 27,255,829.

| 지표 | 결과 |
|---|---:|
| Region Recall (잠정) | 91.25% (219/240) |
| Small Region Recall (<5,000 px, 잠정) | 53.66% (22/41) |
| Pixel Recall | 92.22% |
| Contaminant IoU | 84.40% |
| Pixel FPR | 1.7644% |
| Pixel Precision | 90.87% |
| Dice score | 91.54% |
| sPRO / AUsPRO (잠정) | 82.43% / 0.705476 |

## 시각화
[Learning Curve](../figures/exp01_learning_curves.png) · [Threshold Analysis](../figures/exp01_threshold_analysis.png) · [Error Analysis](../figures/exp01_error_analysis.png) · [그룹 지표](../figures/exp01_group_metrics.png) · [전체 그림 안내](../figures/README.md)

## Error Analysis
14장, 21개 영역을 놓쳤다. 5,000 px 미만의 미탐은 19/41, 그 이상의 미탐은 2/199다. Group A FPR 2.6646%, A FP 중 80.64%가 GT 16 px 이내, 96.95%가 GT에 겹치는 예측 성분에 연결된다. Group B는 2/20장 오탐이며 `image000300` 18,544 px, `image000304` 6 px다. 전자는 Clean 주석 검토 대상이다. 로컬 보고서 `runs/Exp01_RGB_Baseline/error_analysis/REPORT.md`에 상세 사례가 있다.

## 기술적 문제
Val Dice loss는 20개 epoch 모두 NaN이었다. Ultralytics Val FP16 Dice 누적 overflow를 CPU에서 재현했다. Train 손실, mIoU 및 가중치 텐서는 유한했다. GPU에서 수정 검증은 아직 하지 않았다. [원인과 해결 후보](../TROUBLESHOOTING.md)

## 결과 해석
전체 Validation FPR 기준을 충족했고 비교용 가중치와 확률맵을 확보했다. 영역 성능은 객체 면적 및 그룹 구성에 따라 달랐다. A의 오탐 거리는 경계 확장 가설을 뒷받침하며 원본 주석 정확성이나 인과관계까지 판단할 수 없다.

## 한계
단일 seed와 통제된 데이터다. 팀원 평가 영역 수와 규칙이 확인되지 않았다. C/D 표본이 작다. 실시간 선별 성능은 end-to-end로 측정하지 않았다. 체크포인트 평가와 분할 데이터는 Validation 범위에 한정된다.

## 결론
연구용 RGB 기준선의 산출물과 FPR 제한 충족은 확인됐다. 영역 기반 팀 간 비교와 최종 Test 결론은 보류한다.

## 후속 실험 방향
팀 평가기를 확보해 저장 확률맵을 재평가하고, 추가 학습 전에 Val Dice FP32 계산을 GPU에서 검증한다. 작은 영역, A 경계, B 고신뢰 사례를 승인된 설계에서 검토한다.

## 관련 코드 및 산출물 경로
[실행 진입점](../../scripts/yolo/exp01.py) · [모델](../../src/yolo_sem/model.py) · [평가](../../src/yolo_sem/evaluation.py) · [공개용 원자료와 설정](exp01_results/README.md). 로컬 `runs/Exp01_RGB_Baseline/`의 `args.yaml`, `run_config.json`, `environment.json`, `history.csv`, `best.pt`, `probability_maps/val/`, `threshold_search.csv`, `val_metrics.json`, `val_results.csv`, `figures/`, `error_analysis/`을 사용했다.

## 재현성 검증 정보
best checkpoint SHA-256 `0ff4eed9a78467ecc665478ae7d37fac2b75dd2de003fcf793642d8b9e7afda9`. split SHA-256(해당 Windows 파일) `66518ca0b69f4a7196a624c9d999ab70d2489cbc415765e97e34cbb3ac63c3a6`. 복사본과 압축 무결성, 2클래스 모델 CPU 로딩, 84장 shape·ID·유한 범위, threshold와 선택 행의 수치, 이미지별 혼동행렬 및 영역 집계를 확인했다. [실험 로그](../../EXPERIMENT_LOG.md)

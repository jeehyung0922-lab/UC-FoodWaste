# 연구 그림과 생성 방법

## 전체 연구 파이프라인

```mermaid
flowchart LR
  Q["연구 질문 · 가설 · 실험 승인"] --> F["고정 분할 · 데이터 검증"]
  F --> TR["Train · 가중치 갱신"]
  F --> VA["Validation · 매 epoch mIoU"]
  TR --> VA --> C["Best checkpoint"]
  C --> I["Validation 추론"]
  I --> O["Threshold 보정"] --> E["Validation 평가 · 오류 분석"]
  E --> A["새 실험 또는 Ablation"] --> Q
  E --> L["최종 모델 · threshold · 평가기 동결"]
  L --> P["명시 승인 후 최종 Test"]
```

## 전처리와 분할

```mermaid
flowchart LR
  J["RGB JPEG · H1024×W1536"] --> R["RGB bilinear · H512×W768"]
  M["원본 GT PNG · 0/255"] --> B["Class ID 0/1 · nearest · H512×W768"]
  R --> P["이미지–GT 1:1"]
  B --> P
  P --> S["고정 Split CSV"]
  S --> T["Train · 374 · paired augment"]
  S --> V["Validation · 84 · 고정 격자"]
  S --> X["Test · 84 · 최종 승인 전 보류"]
```

![고정 분할 그룹 수와 Validation GT 영역 크기](dataset/dataset_distribution.png)

Test 막대는 CSV에 기록된 표본 수다. 크기 분포는 Validation GT만 이용했다. 실제 RGB·마스크와 예측 예시는 로컬 `runs/Exp01_RGB_Baseline/figures/examples/`의 6패널 PNG로 생성된다. RGB는 원본 JPG에서 읽어 평가 격자 또는 확대 ROI로 표시한다. GitHub용 문서에는 집계 그림과 구조도만 저장한다.

## YOLO26s-sem

```mermaid
flowchart LR
  R["RGB 3×512×768"] --> B["Backbone 0–2 · P2 · 128×128×192"]
  B --> P3["L4 P3 · 256×64×96"]
  P3 --> P4["L6 P4 · 256×32×48"]
  P4 --> P5["L10 P5 · 512×16×24 · SPPF + C2PSA"]
  P5 --> U4["Upsample ×2 · L12 Concat · L13 C3k2"]
  P4 --> U4
  U4 --> U3["Upsample ×2 · L15 Concat · L16 C3k2"]
  P3 --> U3
  U3 --> M["L17 main · 2×64×96"]
  U4 --> A["L17 auxiliary · Train only · 2×32×48"]
  M --> F["Bilinear logits · 2×512×768"]
  F --> S["Softmax class 1 · 1×512×768"]
  S --> Y["p ≥ frozen threshold"]
```

입력 크기는 H512×W768이다. [전체 레이어 C×H×W](../MODEL.md#실제-레이어와-입력-크기) · [CPU shape trace CSV](exp01_feature_maps.csv)

## Threshold Optimization

```mermaid
flowchart LR
  P["Validation 확률맵 + GT"] --> G["1,002개 threshold 후보"]
  G --> R["Region Recall · Pixel FPR · IoU · PRO"]
  R --> F["전체 FPR ≤ 2%"]
  F --> M["Region Recall 최대"]
  M --> T["동점 · FPR 최소, threshold 최대"]
  T --> X["Frozen t = 0.501"]
```

[곡선 PNG](exp01_threshold_analysis.png) · [학습 이력 PNG](exp01_learning_curves.png) · [오류 분석 PNG](exp01_error_analysis.png) · [그룹별 성능 PNG](exp01_group_metrics.png) · [효율 PNG](exp01_model_efficiency.png)

## 재생성 명령

아래 명령은 PowerShell의 저장소 루트에서 실행한다. 출력에는 존재하지 않는 새 경로를 사용한다. `--profile-checkpoint`는 신뢰할 수 있는 Ultralytics 체크포인트를 CPU에서 로딩하고 THOP로 연산량을 추정한다. 옵션을 생략하면 GFLOPs는 미측정으로 남는다.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/visualization/render.py run runs/Exp01_RGB_Baseline --profile-checkpoint --out runs/exp01_figures_new
& $py scripts/visualization/render.py dataset --regions runs/exp01_figures_new/region_profile.csv --out runs/dataset_figures_new
& $py scripts/visualization/render.py compare runs/exp01_figures_new/comparison_input.json --out runs/comparison_new
```

`run`에는 `history.csv`, `val_metrics.json`, `val_results.csv`, `val_groups.csv`, `threshold.json`, `threshold_search.csv`, `environment.json`, `run_config.json`, `train_resources.json`과 `probability_maps/val/`이 필요하다. 공통 입력 데이터 루트와 split 경로는 `--data-root`와 `--split-csv`로 지정한다. 생성물은 SVG·PNG, 정량 CSV, 모델 구조 CSV, 비교용 JSON이다. 검증기는 현재 아티팩트 계약과 region 수식을 사용하는 run의 이미지별 혼동행렬과 영역 집계를 대조한다.

`compare`에는 실제 run의 `comparison_input.json`을 여러 개 전달할 수 있다. split 해시, Validation GT 해시, 해상도, 영역 규칙, threshold 정책, FPR 제한, 평가기 ID가 다른 묶음은 거부한다. 추론 시간의 장비·dtype/범위·입력 shape·런타임이 다르면 trade-off 그림을 비활성화한다. 다른 THOP 추정 방식은 CSV에서 비교 불가로 표시한다. Exp01 단독 비교 결과는 로컬 `runs/exp01_comparison_figures/`에 있다.

Ablation은 새 run 그림 생성 시 `--ablation-metadata path.json`에 `{"baseline":"실제 기준 ID","change":"승인된 변경 변수","controls":["고정 조건"]}` 형식의 메타데이터를 전달한다. 비교 결과는 해당 입력이 있을 때 `ablation_study.csv`를 만든다. 기준 run이 같은 입력 목록에 있으면 차이를 %p로 기록한다. 팀 평가기 도입 시 실제 평가식에 맞춰 입력 어댑터·검증을 갱신하고 고유 `--protocol-id`를 부여한다. 공통 평가기의 값과 잠정 Exp01 값은 서로 다른 비교 묶음을 유지한다.

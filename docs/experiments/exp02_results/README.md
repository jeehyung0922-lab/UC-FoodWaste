# Exp01–Exp02 가중치 초기화 비교

**고정 Validation 84장, 단일 seed 42, 20 epochs, 잠정 평가.** Region Recall은 GT 8연결 성분에서 면적의 10% 이상을 예측한 비율이다. 각 모델의 threshold는 동일한 전체 FPR ≤2% 규칙으로 Validation에서 선택했다. 최종 모델 선정과 Test 평가는 수행하지 않았다.

## 산출물 검증

| 확인 항목 | 결과 |
|---|---|
| best.pt · last.pt | A/B 각각 기록 SHA-256과 실제 파일 및 `weights/` 사본 일치. CRC와 CPU 로딩 통과, 부동소수점 NaN·Inf 없음. 체크포인트 내 mIoU가 history의 best 및 최종 epoch와 일치 |
| epoch 체크포인트 | A/B 각각 `epoch0.pt`~`epoch19.pt` 20개 CRC 검사 통과 |
| 확률맵 | 각 84장, 동일 Val ID, H512×W768 float32. 값은 모두 유한하며 [0,1] 범위 |
| 입력 및 평가 식별자 | 세 run의 고정 split과 준비 데이터 요약 해시가 로컬 및 확률맵 manifest와 일치. best 체크포인트 식별자, threshold와 지표 JSON 연결 확인 |
| 평가 재현 | 모델별 threshold 후보 1,002개에서 픽셀·영역 카운트를 재계산하고 저장 threshold, sPRO 곡선, AUsPRO와 일치 확인. 예측 마스크 252장이 재구성한 마스크와 픽셀 단위로 일치 |
| 학습 통제 | 실제 설정의 seed, epoch, batch, AMP, optimizer, LR, scheduler, 손실, 모델 파라미터 수, 입력 배치 크기와 주요 augmentation 항목 일치. A/B accumulation 4, 유효 batch 64. save period와 실행 경로는 별도 설정 |
| Git과 초기 상태 | A/B 모두 `812e2d02a30f71071a2d5089294970d95e4810f1`, clean. 초기 상태 SHA가 기존 CPU 검증과 각각 일치. A 360개 전이·4개 제외, B 0개 및 source filename·SHA null |
| 기존 Exp01 | 이전 해시 기록의 정본 파일 369개 일치 |

**기록 오류.** A의 `initial_checkpoint.json`에는 `source_classes: 2`가 저장됐다. 공식 파일의 SHA-256 일치와 CPU 로딩에서 **150클래스**를 재확인했고, 전이 제외 4개 텐서의 원본 클래스 축도 150이다. 모델 생성 시 YAML의 `nc`가 2로 바뀐 뒤 참조 값을 기록하는 경로와 일관된다. 전이 수와 초기 상태 해시는 CPU 기록과 일치한다. 원본 JSON은 그대로 보존했다. [해시·범위·초기화 검증](verification.json)

## 학습과 잠정 Validation

| 지표 | Cityscapes | ADE20K | Random |
|---|---:|---:|---:|
| Best epoch | 18 | 19 | 19 |
| Best Val mIoU | 90.602% | 90.996% | 83.258% |
| Epoch 20 mIoU | 90.482% | 90.855% | 83.091% |
| mIoU 85% 최초 도달 | 9 | 11 | 미도달 |
| Epoch 20 Train CE / Dice / Aux | 0.07020 / 0.05594 / 0.02844 | 0.07050 / 0.05413 / 0.02767 | 0.21225 / 0.16911 / 0.08398 |
| Threshold | 0.501 | 0.378 | 0.508 |
| Region Recall | 91.25% (219/240) | 92.50% (222/240) | 80.42% (193/240) |
| Small Region Recall (<5,000 px) | 53.66% (22/41) | 58.54% (24/41) | 31.71% (13/41) |
| Pixel Recall | 92.22% | 93.59% | 79.11% |
| Precision | 90.87% | 90.05% | 88.97% |
| Contaminant IoU | 84.40% | 84.82% | 72.05% |
| Dice score | 91.54% | 91.79% | 83.75% |
| Pixel FPR | 1.7644% | 1.9689% | 1.8676% |
| sPRO | 82.43% | 84.92% | 63.83% |
| AUsPRO | 0.7055 | 0.7270 | 0.4984 |
| 학습 시간 | 368.865 s | 368.876 s | 369.619 s |
| 학습 peak GPU allocated | 3.911 GiB | 3.722 GiB | 3.911 GiB |
| 추론 평균 / 중앙값 | 4.940 / 5.044 ms | 4.664 / 4.764 ms | 4.516 / 4.658 ms |
| 추론 peak GPU allocated | 112.43 MiB | 112.43 MiB | 112.43 MiB |

Cityscapes는 mIoU 85%에 가장 먼저 도달했다. ADE20K는 19 epoch에서 최고 mIoU를 기록했다. Random은 처음 세 epoch에서 42.0%에 머물렀고 마지막까지 85%에 도달하지 못했다. 이는 같은 학습 예산에서 두 사전학습 모델이 Random을 앞섰다는 단일 run의 관측이다. 속도는 같은 RTX 5090에서 batch 1 FP32 forward·보간·softmax를 측정한 값이다. 모델 크기가 같고 반복 계측이 없으므로 작은 시간·메모리 차이를 초기화의 인과 효과로 해석하지 않는다. [학습 원자료](learning_history.csv) · [집계 JSON](summary.json)

ADE20K는 Cityscapes보다 영역 Recall **+1.25%p**, IoU **+0.42%p**, FPR **+0.2045%p**, Precision **−0.82%p**였다. 탐색 곡선을 Cityscapes의 1.7644% FPR 상한에서 다시 읽으면 ADE20K는 후보 0.453에서 FPR 1.7376%, 220/240개를 검출한다. 이 수치는 저장 곡선의 사후 민감도 분석이며 동결 threshold 및 결과는 바꾸지 않았다. 비슷한 오탐 예산에서 우위가 한 영역으로 줄어드므로 현재 결과만으로 최종 가중치를 선정할 수 없다.

### Validation Dice Loss

세 모델 모두 epoch 로그의 `val/dice_loss` **20/20 NaN**이다. Exp01 best의 별도 GPU 진단에서는 FP16 Dice loss가 6개 배치에서 유한했으나 평균 **1.0000**으로 포화됐다. 같은 FP16 logits의 loss만 FP32로 계산하면 **0.133494**, 전체 FP32는 **0.133498**이다. 고정 logits에서 loss dtype만 바꾸면 mIoU는 **90.60197%**로 같았다. 전체 FP32 forward mIoU는 **90.60143%**였고 913픽셀이 달랐다. GPU 진단 재실행에서 나타난 포화와 epoch 로그의 NaN은 별개의 관측이다. A/B의 별도 FP32 진단은 없으며, 학습 경로에 FP32 변경을 적용하지 않았다. 위 표의 Dice score는 확률맵에서 산출한 평가 지표다. [GPU 진단 JSON](fp32_validation_diagnostic.json)

## 대응 영역 오류

| GT 면적 | GT 수 | Cityscapes | ADE20K | Random |
|---|---:|---:|---:|---:|
| <1,000 px | 13 | 5 | 3 | 3 |
| 1,000–<5,000 px | 28 | 17 | 21 | 10 |
| 5,000–<10,000 px | 38 | 37 | 37 | 28 |
| 10,000–<25,000 px | 108 | 108 | 108 | 101 |
| ≥25,000 px | 53 | 52 | 53 | 51 |

- **공통 미탐 13개**(A 9개, D 4개) 중 12개가 5,000 px 미만이다. 세 모델이 모두 검출한 영역은 191개다.
- **ADE20K의 검출 변화**는 Cityscapes 대비 새로 검출한 7개와 놓친 4개다. 5,000 px 미만의 순증은 2개이지만, 1,000 px 미만은 5→3개로 감소했다. 개선은 1,000–<5,000 px 구간에 집중됐다.
- **미탐 GT 내 최대 확률 중앙값**은 Cityscapes 0.0300, ADE20K 0.0593, Random 0.1314다. 각각 21/18/47개 미탐으로 구성과 임계값이 달라, 이 분포만으로 확률 보정을 비교할 수 없다.
- **대표 사례**는 `image000117` 영역 1(26,885 px, ADE20K만 검출), `image000359` 영역 5(1,225 px, Cityscapes만 검출), `image000006` 영역 1(7,982 px, 공통 미탐)이다. 대응 영역 목록과 RGB·GT·확률맵·오류 오버레이는 로컬 `runs/exp02_review/`에 보존했다. `paired_regions.csv`, `all_missed.csv`, `ade_gains.csv`, `ade_losses.csv`, `examples/`에서 확인할 수 있다.

## 그룹별 오탐과 경계

| 그룹 | 표본 | Region Recall City / ADE / Random | FPR City / ADE / Random |
|---|---:|---:|---:|
| A | 50 | 93.06 / 93.98 / 80.56% | 2.6646 / 2.9610 / 2.5394% |
| B | 20 | GT 영역 없음 | 0.2359 / 0.3051 / 0.2510% |
| C | 4 | 100 / 100 / 100% | 0.2484 / 0.2752 / 0.3660% |
| D | 10 | 70 / 75 / 75% | 1.9391 / 2.1196 / 3.3912% |

- A의 GT 16 px 이내 FP는 Cityscapes **328,535/407,422(80.64%)**, ADE20K **358,762/452,735(79.24%)**, Random **280,858/388,277(72.33%)**다. ADE20K에서 GT 근처 FP와 먼 FP가 모두 늘었다. 거리값은 GT 픽셀까지의 거리이며 경계 확대와 라벨 오류를 직접 구분하지 않는다.
- B의 오탐 이미지는 **2/20, 2/20, 4/20장**이다. `image000304`는 Cityscapes 6 px에서 ADE20K 5,775 px, Random 3,984 px로 늘었다. `image000300`의 Clean 주석 의심 사례는 유지한다.
- C는 4/4 검출에도 Random IoU가 74.27%로 ADE20K 89.20%보다 낮다. D는 ADE20K와 Random이 모두 15/20이지만 Random FPR은 3.3912%다. 영역 10% 기준의 검출 여부와 마스크 품질을 함께 해석해야 한다. [전체 그룹 지표](group_metrics.csv) · [면적 원자료](size_metrics.csv)

## 시각화와 후속 연구

기존 서버 learning/threshold/error/group/efficiency 그림 파일과 세 모델 비교표를 확인했다. 새로 필요한 [학습 이력 비교](learning_comparison.png)와 [대응 영역 비교](paired_region_comparison.png)만 문서에 추가했다. 실제 RGB를 넣은 그림은 로컬에만 보존했다.

H1은 잠정 지표 일부가 ADE20K에 유리하지만 FPR·Precision의 비용과 사후 상한 비교를 고려하면 결론을 보류한다. H2는 고정 20 epochs에서 지지되는 관측이 있다. H3는 크기 구간별 차이가 보였고, 특히 1,000 px 미만의 패턴은 전체 small recall과 달랐다. 팀 공통 평가기를 확보한 뒤 동일 확률맵으로 재평가하고, 이후 승인된 설계에서 seed 반복 및 이미지 단위 paired 불확실성, 공통 미탐 주석과 B그룹 경계 사례를 확인한다. Exp03 구현과 실행은 범위 밖이다.

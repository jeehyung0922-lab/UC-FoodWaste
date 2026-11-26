# 평가 프로토콜

현재 영역·threshold 기준은 팀 공통 평가 코드 확보 전까지 **잠정적**이다. Exp01은 H512×W768에서 이미지별 확률에 `p ≥ threshold`를 적용한다. 2채널 raw logits를 bilinear 보간한 후 softmax의 클래스 1을 저장한다. TP는 GT/예측 모두 이물질, FP는 GT 배경/예측 이물질, FN은 GT 이물질/예측 배경, TN은 모두 배경이다. 그룹별 및 전체 지표는 픽셀 카운트를 합산한 뒤 계산한다.

| 지표 | 정의 |
|---|---|
| Pixel Recall | TP / (TP + FN) |
| Pixel Precision | TP / (TP + FP) |
| Contaminant IoU | TP / (TP + FP + FN) |
| Dice score | 2TP / (2TP + FP + FN) |
| Pixel FPR | FP / (FP + TN) |
| Region Recall | GT 8연결 성분 중 예측이 면적의 10% 이상을 덮는 비율 |
| sPRO (잠정) | 각 GT 연결 성분의 overlap fraction을 영역별 동일 가중 평균 |
| AUsPRO (잠정) | FPR 0–0.02에서 sPRO 곡선을 사다리꼴 적분하고 0.02로 나눈 값 |

정의역이 비면 미정의로 남긴다. 현재 GT 성분은 최소 크기 제외 없이 산출한다. 시각화 비교에서 small region은 **5,000 px 미만**의 별도 기술 통계다. GT 마스크가 빈 B 이미지에서는 영역 recall이 정의되지 않고, 그룹 합산 IoU는 FP가 있으면 0이다. B에는 오탐 이미지 비율과 평균 예측 이물질 면적도 기록한다. 픽셀별 / 배치별 손실과 후처리 Dice score는 각각 다른 정보다.

Validation threshold는 0~1을 0.001 간격으로 탐색하고 전체 음성 후보를 추가한다. 전체 FPR ≤2% 중 Region Recall 최대, 동률이면 FPR 최소, 다시 동률이면 threshold가 큰 값을 선택한다. 체크포인트 선정은 Ultralytics Validation **클래스 평균 mIoU**를 사용한다. 위 표의 IoU는 클래스 1만 집계한다. 동결된 모델/threshold 및 평가 규칙이 확정된 후 명시 승인을 얻어 Test를 평가한다.

추론 시간은 RTX 5090에서 batch 1 FP32, 1회 warm-up 후 동기화된 forward·보간·softmax다. 디스크 로딩, 전처리, host transfer, 후속 제어는 범위 밖이다. 동등한 장비, dtype, 배치, 입력 해상도 및 측정 구간에서 속도를 비교한다.

[평가 구현](../src/yolo_sem/evaluation.py) · [지표 구현](../src/yolo_sem/metrics.py) · [Threshold 구조도와 예시](figures/README.md) · [Exp01 수치](experiments/Exp01_RGB_Baseline.md)

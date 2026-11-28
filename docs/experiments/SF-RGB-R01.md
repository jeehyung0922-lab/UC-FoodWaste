# SF-RGB-R01 · 기존 SegFormer-B0 RGB 추론 재현

**상태** Phase 1 로컬 구성 완료. 서버 환경 검증 및 모델 실행은 미실행이다. 기존 팀원 모델의 Validation 재현을 다루며 RGB 재학습은 `SF-RGB-T01`, 센서 비교는 `SF-SENSOR-*`로 분리한다.

## 목적, 연구 질문, 가설

기존 RGB 체크포인트와 실제 제출 입력을 사용했을 때 RTX 5090 FP32 추론에서 팀원 CPU FP32 Validation 결과를 얼마나 재현하는지 확인한다. 체크포인트·입력·전처리·평가 프로토콜이 같으면 유사한 결과를 기대한다. 기기와 라이브러리에 따른 수치 차이는 가설이며, 관측 전 차이의 원인으로 확정하지 않는다.

## 변경·통제 변수

- 변경 후보는 실행 기기와 런타임이다. 후보 환경은 Python 3.13.15 x64, Torch 2.9.1+cu128, Transformers 4.57.1이다.
- 동일 checkpoint SHA-256, 팀원 실제 Validation packed NPZ 84장, Split, RGB `[2,1,0]`/divisor 1/ImageNet 정규화 및 FP32 eval mode를 사용한다. 기존 YOLO용 JPG·PNG는 동일성이 확인될 때까지 대체하지 않는다.
- Common Evaluator v0.1, GT·Polygon·Split 해시, Contaminant 채널 1, H512×W768, 글로벌 FPR ≤2%의 기존 재선정 규칙을 유지한다. 합성 입력과 단일 실이미지 진단에서도 모델 설정과 Split은 그대로 둔다.
- 팀원이 제출 확률맵을 생성한 실제 배치 크기와 환경을 확보해 기록한다. 기존 학습 seed 42와 20 epochs는 체크포인트 출처 정보다.

## 기준과 성공 판단

원본 기준 요약과 해시는 [SegFormer 원본 기록](../../SegFormer/OVERVIEW.md)에 있다. 팀원 비교 기준은 Threshold 0.137, 229/242 Region, TP 4,989,415, FP 554,599, FN 295,352, TN 27,190,778이다.

| 판정 | 사전 기준 |
|---|---|
| 선행 조건 | 체크포인트·입력·Split·평가 자산·프로토콜 동일성, 84개 ID와 확률맵 계약 확인 |
| 완전 재현 | 기존 Threshold 0.137에서 혼동행렬과 탐지 Region ID가 모두 일치 |
| 근접 재현 후보 | 같은 비교 조건에서 탐지 수 차이 ≤1개, Pixel Recall·Precision·Dice·IoU·Pixel FPR·Region Recall·sPRO 차이 각각 ≤0.5%p, AUsPRO 차이 ≤0.005 |
| 근접 재현 승인 | 차이 원인 및 탐지 ID 변경을 기록하고 사용자 검수 후 확정 |

기존 Threshold 0.137의 혼동행렬과 집계는 sweep의 해당 행에서 읽는다. 기본 평가 CSV의 Region별 탐지 여부는 재선정 Threshold 기준이므로, 재선정값이 0.137과 다를 때 고정 Threshold의 Region ID 검증은 별도의 승인된 진단이 필요하다. 그 진단은 원본 평가 프로토콜의 탐지 규칙을 재사용한다. AUsPRO는 곡선 전체 지표다. 재선정 결과에서는 FPR ≤2%를 별도로 확인하고 고정 0.137의 FPR 초과 여부도 보고한다.

## 다음 실행 순서와 기록

1. [독립 실행 안내](../../SegFormer/README.md)에 따라 환경 생성 및 패키지 검사, 실제 CUDA 연산을 확인한다.
2. 해시 일치 체크포인트를 엄격 로딩한다. 합성 입력의 실제 Shape·dtype·유한값을 확인한 다음 Validation 1장으로 입력 정렬을 확인한다.
3. 확인을 마친 뒤 84장을 새 `runs/segformer/SF-RGB-R01/` 또는 승인된 외부 결과 경로로 내보낸다. 체크포인트를 갱신하지 않는다.
4. Common Evaluator에서 평가하고 고정·재선정 Threshold를 나란히 비교한다. 차이가 생기면 원본 RGB 배열, CPU/GPU FP32, 라이브러리와 batch 차이를 확인한다.
5. 실행 당시 Git SHA·working tree 상태, 환경, 입력·GT·가중치·확률맵 해시, 명령, 결과·오류 사례를 문서화한다. 기존 기준 결과는 보존한다.

Team Validation packed NPZ와 입력 해시, 제출 CPU 추론 환경·batch, 재현 대조용 결과 접근 권한이 남아 있다. Test 이미지·Annotation·GT·확률맵은 이 계획의 실행 범위에서 제외한다.

# 연구 파이프라인

[전체 흐름과 구조도](figures/README.md) · [학습 실행 절차](../README.md) · [평가 기준](EVALUATION.md)

| 단계 | 입력과 처리 | 출력 |
|---|---|---|
| 설계 | 목적, 가설, 변수, 성공 기준, 승인 | 실험 ID, 설정 파일, 실험 문서 |
| 준비 | 고정 split, RGB/마스크 대조, H512×W768 정렬 | Train/Val/Test ID, class-ID PNG, 준비 요약 |
| Fine-tuning | Train 이미지와 GT, 사전학습 전이, paired augment | epoch 기록, 체크포인트 |
| 모델 선정 | 매 epoch Validation mIoU | best checkpoint |
| Validation 추론 | best checkpoint, Val RGB | 이미지별 float32 확률맵과 manifest |
| Threshold 보정 | Val 확률맵과 GT, 전체 FPR 제한 | 탐색 곡선·동결 threshold |
| Validation 평가 | 동결 threshold의 예측과 GT | 전체·이미지·그룹 지표 |
| 오류 분석 | Val 예측, GT, RGB | 크기·그룹·공간별 오류, 사례 시각화 |
| 개선·Ablation | 가설별 1개 변경을 우선, 비교 통제 기록 | 새 run, 공통 지표 및 계산 비용 비교 |
| 최종 Test | 최종 모델과 평가 규칙 확정 후 사용자 명시 승인 | 동결 threshold 기반 최종 보고 |

Train만 가중치를 갱신한다. Validation은 체크포인트 선택, threshold 보정과 연구 단계 비교에 사용한다. Test는 최종 모델·조건·평가기가 확정된 뒤에만 읽고 실행하며 Threshold를 재조정하지 않는다. 분할 그림의 Test 개수는 고정 CSV의 메타데이터다.

`import-source`, `audit`, `prepare`의 기본 스코프는 Train/Validation이다. Test 파일을 읽는 준비 작업에는 명시적인 `--prepare-test`와 승인이 필요하다. 374/84/84 표본 수는 고정 CSV에서 검증한다.

실험 기록은 [공통 템플릿](experiments/EXPERIMENT_TEMPLATE.md), 실행 로그는 [실험 인덱스](../EXPERIMENT_LOG.md)에 연결한다. 팀 공통 평가를 확정하면 기존 Val 확률맵을 활용해 별도 경로에 다시 평가하고, 잠정 결과에 덮어쓰지 않는다.

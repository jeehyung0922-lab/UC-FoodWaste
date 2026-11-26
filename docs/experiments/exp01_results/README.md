# Exp01 공개용 집계 및 재현성 기록

기존 Exp01 Validation에서 선별한 집계 결과다. [실험 문서](../Exp01_RGB_Baseline.md)에서 평가 범위와 수치의 한계를 확인할 수 있다.

| 범위 | 기록 |
|---|---|
| 학습 | [epoch별 이력](history.csv), [실제 설정](run_config.json), [환경](environment.json), [학습 자원](train_resources.json) |
| 평가 | [Validation 집계](val_metrics.json), [threshold 탐색](threshold_search.csv), [그룹별 집계](val_groups.csv), [비교 데이터](comparison_input.json) |
| 체크포인트 | [초기 가중치 해시](initial_checkpoint.json), [best/last 선택 및 해시](checkpoint_metadata.json), [CPU 모델 프로파일](model_profile.json) |
| 오류 분석 | [크기별 집계](size_analysis.csv), [그룹별 집계](group_comparison.csv), [Dice NaN CPU 재현](dice_reproduction.json) |

실제 학습 설정에서 실행 머신의 모델·데이터·출력 경로를 제거했다. 당시 실행 Git commit 및 working tree 상태는 확보하지 못했다. 이후 run은 `source_revision.json`에 기록한다. 저장소에는 집계 그래프와 표만 보관하며 원본 데이터, 확률맵, 예측 이미지, 체크포인트는 별도로 관리한다.

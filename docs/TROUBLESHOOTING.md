# 문제 해결 기록

| 문제 | 원인과 확인 | 해결 또는 후속 확인 |
|---|---|---|
| Remote SSH 전송 중 EMFILE | Drag & Drop이 가상환경의 많은 파일을 처리하던 중 핸들 초과를 보고했다. 전송 Manifest 3,293개 일치·읽기 오류 0을 확인했다. | `.venv*`는 서버에서 재생성한다. `resend_paths.txt`가 비어 있으면 검증 범위에서 재전송할 파일이 없다. 가상환경은 Manifest 범위 밖이다. |
| Dataset 절대 경로 | 예전 준비본의 `dataset.yaml`에는 생성 머신의 절대 경로가 들어 있었다. 새 YAML은 `path`를 생략하고 자신의 디렉터리 기준으로 Train/Val을 찾는다. | 오래된 준비본을 다른 머신으로 가져왔다면 기존 결과와 준비본을 보존하고 새 `--prepared`로 준비한다. 후속 단계에서 동일 경로를 전달한다. |
| 기존 Run 및 그림 경로 충돌 | 학습 또는 시각화의 생성 경로가 이미 존재한다. | 새 `--run-root` 또는 `--out`을 사용한다. 목표 run에 속한 중단 체크포인트의 optimizer 상태가 있을 때만 `train --resume`을 사용한다. 완료된 Exp01 가중치는 추론용이다. |
| Validation Dice Loss NaN | Ultralytics 8.4.52에서 Train은 CUDA autocast를 사용하고, AMP 학습 중 Val은 EMA를 FP16으로 직접 변환한다. 512×768에서 Dice의 교집합·분모 합이 FP16 범위를 넘어 inf/inf가 발생하는 현상을 CPU에서 재현했다. 세 run의 Val Dice loss에 각각 20개 NaN이 있고 저장 체크포인트 텐서는 유한했다. | Exp01 best의 별도 GPU 진단에서 FP16 Dice 1.0000 포화, 동일 logits의 FP32 Dice loss 0.133494를 확인했다. FP32 loss 진단은 완료됐으며 학습 경로 수정 적용은 보류했다. [진단 근거](experiments/exp02_results/README.md#validation-dice-loss) |

[전송·설치](ENVIRONMENT.md) · [Exp01 기술 기록](experiments/Exp01_RGB_Baseline.md#기술적-문제) · [Ultralytics loss](https://github.com/ultralytics/ultralytics/blob/v8.4.52/ultralytics/utils/loss.py) · [Validator](https://github.com/ultralytics/ultralytics/blob/v8.4.52/ultralytics/engine/validator.py)

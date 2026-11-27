# Exp02 · 초기 가중치 Ablation

## Experiment ID 및 상태

`Exp02_A_ADE20K`와 `Exp02_B_Random`의 GPU 학습 20 epochs, Validation, 잠정 평가 및 오류 분석이 완료됐다. 둘 다 clean `812e2d02a30f71071a2d5089294970d95e4810f1`에서 실행했다. Exp01 재현용 기준은 `053b2c439290bfd67a23d43d205d1474e374008c`이며 당시 GPU 학습 실행본 Git SHA는 미확보다. [검증 및 비교 보고서](exp02_results/README.md)

## 실험 목적

같은 20 epoch 학습 예산에서 초기화의 영향을 비교한다. 기존 Exp01의 Cityscapes 모델과 Validation 결과를 재사용한다.

## Research Question

Cityscapes, ADE20K, 랜덤 초기화는 이물질 분할과 작은 영역 탐지에 어떤 차이를 만드는가?

## Hypothesis

ADE20K의 다양한 장면이 전이에 도움이 될 수 있다. 제한된 Train 데이터에서는 사전학습이 랜덤보다 나을 수 있으며, 작은 영역의 Recall에서 차이가 나타날 수 있다. 세 가설은 아직 미검증이다.

## 이전 실험과의 관계

[Exp01](Exp01_RGB_Baseline.md)은 완료된 비교 기준이다. Cityscapes 초기화 학습을 다시 실행하지 않는다.

## 변경 변수 및 통제 변수

초기 상태만 변경한다. A는 ADE20K 가중치에서 전이할 수 있는 항목을 이식한다. B는 설치 패키지의 모델 YAML과 PyTorch/Ultralytics 기본 초기화를 사용한다. 데이터, 모델, 손실, 평가와 학습 예산은 Exp01 조건을 유지한다.

## 성공 판단 기준

각 arm에서 독립적인 GPU Smoke를 통과하고 20 epoch, best/last 체크포인트, 연속 확률맵 84장과 원자료를 보존한다. 팀 공통 평가기의 동일 조건에서 Region Recall, 작은 영역 Recall, IoU와 FPR을 비교한다. 현재 잠정 점수로 초기화 방식의 우열을 확정하지 않는다.

## 모델 및 학습 설정

| 항목 | Exp01 실제 | Exp02-A/B |
|---|---|---|
| 모델 | YOLO26s-sem, RGB, 2클래스, H512×W768 | 동일 구조 및 6,498,756 parameters를 CPU 확인 |
| 초기화 | Cityscapes 19-class | ADE20K 150-class / YAML 랜덤 |
| Epoch · batch · workers | 20 · 16 · 0 | 동일, 정식 CLI에서 배치·workers 검사 |
| Seed · AMP · freeze | 42 · True · None | 동일, 시작 전 실제 AMP 활성화 확인 |
| Optimizer 요청 | auto, lr0=0.01, momentum=0.937 | 동일 |
| 실제 선택 | AdamW, lr=0.001667, β=(0.9,0.999) | 동일 설정을 시작 전 검사 |
| Decay | weight 0.0005, BN/bias 0 | 동일 |
| Accumulation · effective batch | nbs 64, warmup 1→4, post-warmup 4 · 64 | 동일 |
| Scheduler · warmup | LambdaLR 선형, lrf=0.01, warmup 3 epochs, warmup momentum=0.8, auto 적용 뒤 warmup bias LR=0 | 동일 |
| Augmentation · Loss | rect, aligned affine, HSV, horizontal flip, mosaic/mixup/cutmix off. CE + Dice + Aux CE | 동일 |
| Checkpoint | Validation mIoU | 동일, 정식 실행은 기본 매 epoch 파일 보존. 디스크에 맞게 `--save-period 5` 또는 `-1` |
| Ultralytics | 8.4.52 | 동일 버전 강제 |

[ADE20K 공식 모델](https://docs.ultralytics.com/tasks/semantic) · [공식 가중치](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem-ade20k.pt) · [Ultralytics 8.4.52 optimizer](https://github.com/ultralytics/ultralytics/blob/v8.4.52/ultralytics/engine/trainer.py)

## 데이터 및 평가 설정

고정 CSV 374/84/84에서 Train과 Validation만 사용한다. Validation 연속 확률맵은 H512×W768 float32, [0,1]이며 이미지 ID와 manifest를 보존한다. 잠정 8연결·GT 10% overlap·FPR 2% 제한을 유지한다. 비교할 때 Manifest의 평가 GT 해시와 평가 프로토콜 ID를 검증한다.

## 실행 환경

로컬 CPU에서 Ultralytics 8.4.52를 사용했다. GPU 검증에는 Exp01과 같은 PyTorch 2.9.1+cu128 환경을 사용한다. AMP 호환성 확인용 Ultralytics 보조 모델이 다운로드될 수 있으며 Random 학습 모델에는 전이되지 않는다.

## 실행 명령어

아래는 **서버에서 GPU Smoke가 승인된 뒤** 사용하는 PowerShell 절차다. 저장소 루트에서 실행한다. 별도 클론을 쓴다면 `$base`는 기존 데이터·환경·Exp01이 있는 프로젝트 루트로 바꾼다. `$prepared`의 오래된 절대경로 YAML이 현재 머신 경로를 가리키는지 확인한다.

```powershell
$base = (Resolve-Path .).Path
$py = Join-Path $base '.venv-yolo-cuda/Scripts/python.exe'
$data = Join-Path $base 'data'
$prepared = Join-Path $data 'yolo_exp01'
$root = Join-Path (Get-Location) 'runs/exp02'
$cli = 'scripts/yolo/exp01.py'
$common = @('--data-root', $data, '--prepared', $prepared, '--run-root', $root)
$ade = @('--config', 'configs/yolo/exp02_a_ade20k.yaml')
$rand = @('--config', 'configs/yolo/exp02_b_random.yaml')

& $py $cli environment
& $py $cli weights @ade @common
& $py $cli diagnose-val-loss --checkpoint (Join-Path $base 'runs/Exp01_RGB_Baseline/best.pt') --out (Join-Path $root 'exp01_fp32_diagnostic.json') --batch 16 @common

& $py $cli smoke --batch 2 @ade @common
& $py $cli smoke --batch 2 @rand @common

# Smoke 및 진단 검토 후 정식 학습을 별도 승인
& $py $cli train --batch 16 --workers 0 @ade @common
& $py $cli predict --split val @ade @common
& $py $cli threshold @ade @common
& $py $cli evaluate --split val @ade @common

& $py $cli train --batch 16 --workers 0 @rand @common
& $py $cli predict --split val @rand @common
& $py $cli threshold @rand @common
& $py $cli evaluate --split val @rand @common

foreach ($id in 'Exp02_A_ADE20K','Exp02_B_Random') {
    $run = Join-Path $root $id
    & $py scripts/visualization/render.py run $run --data-root $data --profile-checkpoint
    & $py scripts/yolo/analyze_validation.py $run --data-root $data
}
& $py scripts/visualization/render.py compare `
    docs/experiments/exp01_results/comparison_input.json `
    (Join-Path $root 'Exp02_A_ADE20K/figures/comparison_input.json') `
    (Join-Path $root 'Exp02_B_Random/figures/comparison_input.json') `
    --out (Join-Path $root 'provisional_comparison')
```

Smoke는 각 arm의 `_smoke` 폴더에 쓰며 정식 학습은 각 초기화에서 다시 시작한다. 실패한 run은 보존하고 새 run root를 쓴다. 공식 ADE20K 파일을 로컬에서 제공하려면 구성에 맞는 이름의 경로를 `--weights`로 지정한다. B에는 `--weights`를 지정할 수 없다.

Random Smoke가 초기 메타데이터 기록에서 중단됐다면 기존 폴더를 보존하고 `--resume` 없이 새 root에서 B만 실행한다. `$py`, `$cli`, `$rand`, `$data`, `$prepared`, `$root`는 위와 같다.

```powershell
$retryRoot = Join-Path $root ('retry-b-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $py $cli smoke --batch 2 @rand --data-root $data --prepared $prepared --run-root $retryRoot
```

B의 `initial_checkpoint.json`에서 `filename`과 `sha256`은 null이며, `initial_state_sha256`과 `transferred_items: 0`을 확인한다. A/B 모두 `source_revision.json`에 시작 시 커밋과 working tree 상태를 남긴다.

## 학습 결과

두 arm 모두 20 epochs를 마쳤다. Best는 둘 다 epoch 19이며 Val mIoU는 ADE20K 90.996%, Random 83.258%였다. Cityscapes는 epoch 9, ADE20K는 11에 mIoU 85%에 처음 도달했고 Random은 도달하지 못했다. [Epoch별 비교](exp02_results/learning_history.csv)

## Validation 성능

잠정 Validation에서 ADE20K/Random의 Region Recall은 92.50%/80.42%, Small Region Recall은 58.54%/31.71%, Contaminant IoU는 84.82%/72.05%였다. 전체 FPR은 1.9689%/1.8676%이고 threshold는 0.378/0.508이다. [전체 지표 및 비교](exp02_results/README.md#학습과-잠정-validation)

## 시각화

서버가 생성한 개별 그림과 비교표를 검증해 재사용한다. [세 모델 학습 곡선](exp02_results/learning_comparison.png)과 [대응 영역·크기 비교](exp02_results/paired_region_comparison.png)를 추가했다.

## Error Analysis

5,000 px 미만 검출 수는 Cityscapes/ADE20K/Random 각각 22/24/13개(전체 41개)였다. 공통 미탐 13개 중 12개가 이 범위에 속한다. Cityscapes 대비 ADE20K에서 새로 검출한 영역 7개와 놓친 영역 4개를 확인했다. [대응 영역과 그룹별 FP](exp02_results/README.md#대응-영역-오류)

## 기술적 문제

세 run의 Validation Dice loss는 20/20 epoch NaN이었다. Exp01 체크포인트의 GPU 진단에서 FP16 Dice는 1.0000, 동일 logits의 FP32 loss는 0.133494였다. 동일 logits의 mIoU는 유지됐고 전체 FP32 forward는 913픽셀이 달랐다. 실제 학습 loss 및 체크포인트 선정 경로는 보존했다. A의 전이 메타데이터에는 `source_classes: 2`가 적혔으나 원본 가중치 로드 결과와 전이 제외층 원본 shape에서 150클래스를 확인했다. [진단 및 메타데이터 차이](exp02_results/README.md)

## 결과 해석

같은 20 epoch 예산에서 두 사전학습 모델의 영역 Recall과 IoU가 Random보다 높았다. ADE20K와 Cityscapes는 차이가 작고 오탐 예산도 달랐다. ADE20K 개선은 1,000–<5,000 px 구간에 집중됐으며 1,000 px 미만에서는 Cityscapes가 더 많이 검출했다.

## 한계

단일 seed, 고정 20 epoch 예산, 잠정 영역 규칙이다. 이 비교는 각 초기화의 최대 달성 성능을 추정하지 않는다. Test 사용과 팀 공통 평가 적용은 별도 단계다.

## 결론

정식 학습과 잠정 Validation, 확률맵 재계산을 완료했다. 단일 seed 및 팀 공통 평가 전 결과이므로 최적 초기화 방식 확정은 보류한다.

## 후속 실험 방향

팀 공통 평가기로 세 run의 Validation 확률맵을 재평가한다. 승인된 후속 설계에서 FPR을 맞춘 비교·seed 반복, 공통 미탐·경계 및 Clean 주석 사례 검토를 고려한다.

## 관련 코드 및 산출물 경로

[Exp02-A 설정](../../configs/yolo/exp02_a_ade20k.yaml) · [Exp02-B 설정](../../configs/yolo/exp02_b_random.yaml) · [초기화 구현](../../src/yolo_sem/initialization.py) · [Validation 진단](../../src/yolo_sem/validation_diagnostic.py). GPU run에는 초기화 리포트, `run_config.json`, `history.csv`, 체크포인트 및 해시, 자원 메타데이터, Validation 확률맵과 평가 기록을 저장한다.

## 재현성 검증 정보

ADE20K SHA-256 `feb2bd47eed6f721a4588213c670ccdad69c0ba29b4b5f97892df222b18c6c3a`. A는 364개 중 360개 state_dict 항목을 전이했고 main·aux class projection 각각의 weight/bias 4개가 shape mismatch였다. 새 출력층 값은 동일 seed의 B와 일치했다. B는 0개 전이, 동일 seed에서 초기 state SHA가 일치하고 사전학습 경로와는 달랐다. 세 초기화의 구조 서명은 Exp01 `best.pt`와 일치했다. CPU logits [1,2,64,96]과 native CE+Dice+Aux backward를 확인했다.

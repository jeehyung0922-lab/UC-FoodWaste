# YOLO26s-sem 모델

Exp01은 Ultralytics 8.4.52의 `SemanticSegmentationModel`을 사용한다. Cityscapes 19클래스로 사전학습한 `yolo26s-sem.pt`를 읽고 데이터 정의의 nc=2, channels=3으로 모델을 재구성한다. 형태가 대응하는 가중치를 전이한다. 출력 클래스 차원 때문에 형태가 다른 분류기 텐서는 새 초기화 상태로 남는다. `freeze=null`인 상태에서 fine-tuning했다. [모델 로딩 코드](../src/yolo_sem/model.py) · [공식 모델](https://docs.ultralytics.com/tasks/semantic) · [핀 고정 구현](https://github.com/ultralytics/ultralytics/tree/v8.4.52/ultralytics)

## 실제 레이어와 입력 크기

`best.pt`에서 CPU eval forward hook으로 측정했다. 표는 Batch=1의 C×H×W이며, 직접 연결은 [전체 trace CSV](figures/exp01_feature_maps.csv)와 [구조도](figures/README.md#yolo26s-sem)에서 확인할 수 있다.

| Layer | 모듈 | 출력 C×H×W |
|---:|---|---|
| 입력 | RGB | 3×512×768 |
| 0 | Conv | 32×256×384 |
| 1 | Conv | 64×128×192 |
| 2 | C3k2 | 128×128×192 |
| 3 | Conv | 128×64×96 |
| 4 | C3k2 | 256×64×96 |
| 5 | Conv | 256×32×48 |
| 6 | C3k2 | 256×32×48 |
| 7 | Conv | 512×16×24 |
| 8 | C3k2 | 512×16×24 |
| 9 | SPPF | 512×16×24 |
| 10 | C2PSA | 512×16×24 |
| 11 | Nearest Upsample ×2 | 512×32×48 |
| 12 | Concat 11, 6 | 768×32×48 |
| 13 | C3k2 | 256×32×48 |
| 14 | Nearest Upsample ×2 | 256×64×96 |
| 15 | Concat 14, 4 | 512×64×96 |
| 16 | C3k2 | 128×64×96 |
| 17 | SemanticSegment (16, 13) | 2×64×96 (eval main) |
| 후처리 | Bilinear logits → softmax | 1×512×768 (이물질 확률) |

Backbone은 0–10, top-down neck은 11–16이다. Train에서는 P3 main과 P4 auxiliary logits를 사용하며 eval에서는 P3 main을 사용한다. 입력 해상도의 1/8 logits를 추론 후 bilinear로 원래 평가 격자까지 올린다. [head 소스](https://github.com/ultralytics/ultralytics/blob/v8.4.52/ultralytics/nn/modules/head.py)

## Loss

유효 픽셀 집합 V는 GT가 255가 아닌 픽셀이다. logits는 마스크 격자로 bilinear 보간된다. $p_{ic}$는 main softmax, $y_{ic}$는 one-hot GT다.

- CE = $-\frac{1}{|V|}\sum_{i\in V}\log p_{i,y_i}$, Exp01에서 클래스 가중치 없이 계산.
- Dice loss = $\frac{1}{2}\sum_{c=0}^1\left(1-\frac{2\sum_{i\in V}p_{ic}y_{ic}+1}{\sum_{i\in V}p_{ic}+\sum_{i\in V}y_{ic}+1}\right)$.
- Train scalar = 배치 크기 × (main CE + Dice + 0.4 × aux CE). eval loss 경로는 aux 출력이 없어서 aux 항은 0이다. [loss 구현](https://github.com/ultralytics/ultralytics/blob/v8.4.52/ultralytics/utils/loss.py)

Val Dice NaN은 FP16 누적 overflow를 CPU에서 재현했다. GPU 수정 검증은 미수행이다. [문제 해결](TROUBLESHOOTING.md)

## 모델 비용

파라미터는 **6,498,756개**다. THOP 2.2.5의 정확한 직사각형 입력, unfused FP32 eval forward 측정에서 **8.369 GMACs / 16.739 GFLOPs**(2 FLOPs/MAC)로 추정됐다. Ultralytics `get_flops` 스케일링 추정치는 16.626 GFLOPs였다. 표준 THOP 훅에 포함되는 연산에 한정된 모델 연산량 추정치로, 전처리·logits 보간·softmax 비용과 GPU 계측값은 포함하지 않는다. 같은 프로파일 방법으로 비교한다.

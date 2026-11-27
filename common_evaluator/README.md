# UC-FoodWaste Common Evaluator v0.1

모델 프레임워크와 분리된 **Validation 전용 잠정 평가기**다. CLI와 Tkinter 앱은 같은 엔진을 사용한다. PyTorch, Ultralytics, GPU가 필요 없다.

## Windows 팀원 사용법

각 팀원은 권한을 확인한 원본에서 아래 절차로 Validation 자산 팩을 만든다. GUI 실행 파일 옆에 `validation_assets/`를 둔다. 내부에는 `asset_manifest.json`, `region_manifest.json`, `validation_pixels.npz`가 있어야 한다. 실행 파일에는 GT가 포함되지 않는다.

1. `UC-FoodWaste-Evaluator.exe`를 실행한다.
2. **폴더 선택**에서 확률맵 폴더를 선택한다. 폴더명이 기본 모델명이고 수정할 수 있다.
3. **평가 시작**을 누른다. `input_metadata.json`이 없으면 실제 클래스 매핑, 확률 변환, 공간 정렬 방식을 입력한다. GUI가 이 선언을 결과 폴더의 `input_metadata.json`에 저장한다. 기존 메타데이터가 있으면 그대로 검증하고 입력 창을 생략한다.
4. **결과 폴더 열기**로 사용자 홈의 `UC-FoodWaste Evaluations/<timestamp>/`에서 CSV·JSON·SVG를 확인한다.

팀원 PC에는 Python·pip·가상환경이 필요 없다. GUI에 프로토콜 수정 옵션이 없다.

## 확률맵 준비

84장의 고정 Validation ID를 `image000000.npy` 또는 숫자 ID 파일명으로 사용한다. 같은 ID의 두 표기를 함께 넣으면 오류다. 배열은 **H512×W768, float32, [0,1]**의 Contaminant 확률이다. 누락·중복·Val 외 ID·shape·dtype·NaN·Inf·범위 오류는 보정 없이 거부한다. 다음 메타데이터에서 `probability_method`를 실제 방법으로 바꿔 저장한다.

```json
{
  "positive_class": "Contaminant",
  "grid": "ucfw-val-512x768-aligned",
  "quantity": "probability",
  "thresholded": false,
  "probability_method": "Bilinear logits to the aligned Validation grid, then contaminant sigmoid/softmax"
}
```

선택 필드 `inference_ms`와 `parameters`는 모델 제공자가 확인한 외부 정보다. 클래스 의미·공간 정렬·threshold 미적용 여부는 배열만으로 확인할 수 없어 제출자가 선언한다. Letterbox, crop, flip을 사용했다면 기하 변환을 역으로 적용하여 기존 평가 격자에 정렬한다. 출력 해상도를 맞출 때의 logits 또는 probability 보간 정책을 명시한다.

| 모델 | 확률맵 생성 예 |
|---|---|
| U-Net 단일 이물질 logit | 정렬한 logit `z`에 프레임워크의 안정적인 sigmoid 적용 |
| SegFormer 2개 logit | 두 logit을 bilinear 보간한 후 클래스 축 softmax, 실제 Contaminant index 선택 |
| YOLO26s-sem | 2채널 raw logits를 보간 후 softmax 클래스 1을 저장. 기존 `probability_maps/val` 재사용 가능 |

저장 예는 `np.save(output / f"{image_id}.npy", np.asarray(p, dtype=np.float32), allow_pickle=False)`다. `p`는 batch·class 축을 제거한 2D 확률값이다.

## GT와 Polygon 팩 준비

권한을 가진 제공처에서 확보한 `data_processed/masks`의 원본 Binary PNG, `metadata/data_split.csv`, 그리고 원본 주석에서 제공자가 분리한 Validation CSV를 준비한다. PNG 파일명은 split의 `filename`에 대응한다. Polygon만으로 승인된 Binary GT를 복원할 수 없으므로 공식 Annotation과 함께 같은 마스크 원본을 확보해야 한다. 원본 제공·전처리 차이는 [데이터 문서](../docs/DATASET.md#원본-가져오기)에 기록되어 있다. Test 주석 접근이 제한된 환경에서는 Validation으로 분리된 CSV만 받는다.

공식 `Processed_Annotations.csv`의 필드는 `Image ID, Image Name, Width, Height, Polygons`다. `Polygons`는 `Label` 및 평탄화된 `Points=[x0,y0,...]`가 있는 Python literal 리스트다. 원본 좌표계는 W1536×H1024다. 데이터 제공처에서 다운로드를 안내하지만 저장소에서 GT·주석을 재배포할 수 있는 명시적 라이선스를 확인하지 못했다. **팩을 배포하기 전에 데이터 제공자 또는 권한이 있는 팀 관리자의 허용 범위를 확인한다.** 이번 빌드와 Git에는 GT·주석을 포함하지 않는다. [공식 데이터 저장소](https://github.com/CollaborativeRoboticsLab/UC-Food-Waste)

```powershell
.\ucfw-evaluate.exe build-assets --split PATH_TO_FIXED_SPLIT_CSV --masks PATH_TO_VAL_MASKS --polygons PATH_TO_PROCESSED_ANNOTATIONS --out .\validation_assets
```

공식 CSV에는 다른 분할의 행도 있다. 생성기는 고정 Val ID에 해당하는 행의 vertex와 라벨만 해석하며 Validation 마스크만 읽는다. GT는 0/1 또는 0/255의 단일 채널 PNG다. 1024×1536이면 nearest로 축소한다. Polygon마다 Pillow 13.0.0에서 원본 격자에 채운 뒤 nearest로 H512×W768에 맞춘다. 원본 순서로 ID(예: `image000000:p001`)를 고정한다. 접하거나 겹친 Polygon은 별도 영역으로 유지한다. 픽셀 지표는 승인된 binary GT, 영역 지표는 Polygon raster 기준이다. 합집합 차이는 `polygon_gt_audit.csv`에 남긴다. 원본 Annotation 누락, 면적 0, GT와 겹치지 않는 Polygon, GT/Polygon 공백 불일치를 거부하며 Connected Component로 대체하지 않는다.

## 잠정 평가 규칙

평가기는 고정 split SHA-256을 확인하고, 로딩 시 자산 팩의 두 파일을 기록된 SHA-256과 대조한다. 이어서 Region Manifest와 GT·Polygon 픽셀 내용이 각각 v0.1 기준 해시와 일치하는지 확인한다. 픽셀 내용 해시는 NPZ 압축 형식에 의존하지 않는다. 프로토콜은 프로그램의 잠긴 설정과 대조하고 파일 SHA-256을 결과에 기록한다. 불일치 시 원본 PNG·주석·split과 Pillow 버전을 확인한다.

- 픽셀 지표는 전체 또는 그룹별 TP·FP·FN·TN 합산으로 구한다. Recall=TP/(TP+FN), Precision=TP/(TP+FP), IoU=TP/(TP+FP+FN), Dice=2TP/(2TP+FP+FN), Accuracy=(TP+TN)/전체, FPR=FP/(FP+TN).
- overlap은 Polygon raster 안의 검출 픽셀 수 ÷ Polygon 면적이다. Region Recall은 overlap ≥10% 영역의 비율이다. 계산은 정수 조건 `hits*10 >= area`를 쓴다.
- sPRO는 Polygon별 `min(overlap/0.1, 1)`의 평균이다. AUsPRO는 동일 FPR에서 최대 sPRO를 유지하고 FPR 0~0.02를 사다리꼴 적분해 0.02로 정규화한다. 0.02에서 선형 보간하며 양 경계가 관측 범위 밖이면 미정의다. SVG는 원래 sweep을 그린다.
- 예측은 `probability >= threshold`다. float32 기준으로 0~1을 0.001 간격으로 탐색하고 1보다 큰 다음 float32를 추가한다. 전체 FPR ≤2%에서 Region Recall, sPRO, 높은 threshold 순으로 선택한다. sPRO 1e-12 이내 차이는 동률이다.
- 분모 0은 JSON null·CSV 빈 셀이다. Clean B 오탐은 하나라도 양성을 예측한 이미지 수/20이다. Size 구간은 [0,1000), [1000,5000), [5000,10000), [10000,25000), [25000,∞) px다.
- 팀 합의가 필요한 항목은 rasterization 정수화·경계, GT 합집합 불일치 처리, Polygon별 가중치, sPRO 포화, 중복 FPR·적분, threshold 동률 기준이다. 평가 결과에 맞춰 변경하지 않는다.

## CLI와 개발자 빌드

방향과 클래스 의미가 선언된 입력으로 검증부터 그래프까지 한 명령에서 실행한다. 출력 폴더는 새 경로로 지정한다.

```powershell
.\ucfw-evaluate.exe evaluate --model MyModel --maps PATH_TO_MAPS --assets .\validation_assets --out .\results\MyModel
.\ucfw-evaluate.exe compare --out .\results\comparison .\results\ModelA .\results\ModelB
```

요약·threshold·평가 manifest JSON, threshold sweep, 이미지·영역·크기·그룹·GT 감사 CSV, FPR–Recall/sPRO 및 크기별 Recall SVG가 생성된다. `evaluation_manifest.json`에는 확률맵, GT·Region 팩·split, 설정 및 평가기 빌드 해시와 제출 메타데이터가 저장된다. 비교 CLI는 다른 팩·프로토콜·평가기 빌드를 섞으면 중단한다.

소스 개발자는 이 폴더만 복사해 사용할 수 있다. Windows 빌드는 Python 3.12 x64 및 Tcl/Tk 환경을 사용한다. 평가기 폴더에서 실행한다.

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r build-requirements.txt
& $py -m unittest discover -v
& $py main.py --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -Python $py
```

ExecutionPolicy 옵션은 빌드 프로세스에만 적용한다. `dist/build_info.json`에 Python·Tk·라이브러리 버전과 소스 해시를 기록하고 `dist/SHA256SUMS.txt`에 두 EXE의 SHA-256을 저장한다. PowerShell의 `Get-FileHash -Algorithm SHA256`으로 대조한다. 잘못된 입력이면 오류 메시지의 파일명·shape·dtype를 확인하고, 해시 불일치 시 Validation 자산을 권한 있는 원본에서 다시 생성한다. 이미 결과 폴더가 있다면 새 경로를 사용한다.

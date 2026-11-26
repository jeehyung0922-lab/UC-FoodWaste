# 데이터셋

UC Food Waste는 통제 환경에서 수집된 542장의 코등록 RGB·MSI·Thermal 데이터와 마스크·폴리곤 주석을 제공한다. 원본 H1024×W1536 배열은 BGR 3 + MSI 15 + Thermal 1로 배열된다. 공개 마스크의 범주는 soft plastic, hard plastic, glass, metal, organic background다. 프로젝트의 JPEG 및 이진 PNG는 로컬에서 보관하며 NPZ 원본은 별도 위치에 있다. [데이터 저장소](https://github.com/CollaborativeRoboticsLab/UC-Food-Waste) · [데이터 논문](https://doi.org/10.1002/adrr.70180)

| 폴더 | 내용 |
|---|---|
| `data/data_processed/images/` | 원본 크기의 RGB JPG 542장 |
| `data/data_processed/masks/` | 원본 크기의 이진 PNG 542장 |
| `data/archives/` | 로컬 원본 데이터 아카이브 |
| `metadata/data_split.csv` | 정식 split 및 그룹, 블록·이물질 메타데이터 |
| `data/splits/data_split.csv` | 로컬 split 사본 |
| `data/yolo_exp01/` | 각 split에 대응하는 해상도 조정 RGB 및 class-ID PNG |

NPZ 경로에서는 `image[..., [2,1,0]]`으로 RGB를 선택한다. Exp01은 기존 JPEG를 RGB로 디코딩해 쓴다. PNG 원본의 0/255를 클래스 ID 0/1로 변환하고 nearest-neighbor로 축소한다. RGB는 bilinear로 H512×W768에 맞춘 뒤 모델 입력에서 /255로 스케일링한다. 학습 시 RGB/GT에 같은 공간 변환을 적용하고 affine 경계는 ignore 255다.

## 원본 가져오기

정확한 Exp01 입력 복원은 [Manifest](../metadata/exp01_inputs.json)의 `data_processed.zip` 또는 같은 파일을 풀어 놓은 RGB JPG·binary PNG 사본을 사용한다. `import-source --source-archive`는 고정 분할 ID에 맞는 항목만 새 데이터 루트로 추출한다. 기본 범위는 Train/Validation이다. `prepare`는 H512×W768의 훈련·검증 데이터와 위치 독립적인 YAML을 만든다.

공개 원본 배열만 있는 경우 `import-source --source-images PATH --source-masks PATH`를 사용한다. 각 폴더에서 split의 파일명과 같은 `.npy` 또는 `.npz`를 한 개씩 찾아야 한다. 이미지는 H1024×W1536×19, NPZ 키는 `image`다. BGR은 uint8 또는 [0,1] 실수여야 한다. 마스크는 HW 이진 또는 HWC/CHW 5개 클래스 평면이며 값은 0/1/255다. NPZ 마스크 키는 기본 `mask`이고 `--mask-key`로 바꿀 수 있다. 처음 4개 전경 평면을 합쳐 binary PNG를 생성한다.

배열에서 생성한 RGB PNG는 무손실이며 기존 Exp01 JPEG를 다시 인코딩하지 않는다. 이 경로의 JPEG 픽셀·GT 동등성은 공개 원본 파일로 검증되지 않았다. 동일 기준선 수치를 주장하려면 Manifest로 식별한 Exp01 입력 사본을 사용한다. 공개 프로젝트의 데이터 재배포 허용 범위는 확인하지 못해 실제 샘플 이미지를 Git에 포함하지 않는다.

그룹 정의는 공개 데이터의 물질 분류와 별도로 로컬 CSV의 `group_name`을 따른다.

| Split | A · Mixed | B · Clean | C · Few_contaminants | D · Vinyl | 합계 |
|---|---:|---:|---:|---:|---:|
| Train | 224 | 90 | 30 | 30 | 374 |
| Validation | 50 | 20 | 4 | 10 | 84 |
| Test | 49 | 19 | 6 | 10 | 84 |

준비 스크립트는 고정 CSV 해시(줄바꿈 정규화), 그룹 수, 파일명 대응, JPEG·PNG 디코딩, 크기, 마스크 값, 분할 간 decoded RGB 정확 중복을 검사한다. 로컬 ZIP의 1,084개와 바이트 동일성도 검증했다. 미세하게 변형된 근접 중복, 팀원 파일 및 Drive NPZ와의 픽셀 단위 동일성은 확인되지 않았다. [데이터 코드](../src/yolo_sem/data.py) · [검증 명령](ENVIRONMENT.md)

Validation의 Clean `image000300`의 GT는 전부 배경이고, RGB의 투명한 병처럼 보이는 곳에 고신뢰 예측이 집중된다. 팀 주석 검토가 필요한 사례이며 소재와 오류 여부는 미확정이다. C·D의 Validation 표본 수가 작다. 이진 변환으로 소재별 구분은 데이터에서 사용할 수 없다. [Exp01 사례](experiments/Exp01_RGB_Baseline.md#error-analysis)

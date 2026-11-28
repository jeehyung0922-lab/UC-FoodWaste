# Common Evaluator v0.1 검증 기록

이 기록은 고정 Validation 84장에 대한 **잠정 평가**다. 기존 모델은 단일 seed 42, 20 epochs에서 생성된 Exp01·Exp02-A/B다. 학습·추론을 재실행하지 않았으며 기존 결과는 그대로 두었다.

## 데이터와 정책

- 공식 `Processed_Annotations.csv` SHA-256 `827732b73f7b4c6b7c454aae48d9eab055a44ad6ae72a193bab61aaad88ecedf`.
- 독립 Polygon 총 **242개**, 그룹 A/B/C/D는 **219/0/4/19**. 클래스별 glass 100, soft plastic 64, metal 46, hard plastic 32. 이전 GT Connected Component는 240개였다. 연결 성분은 접하는 주석을 합치거나 한 Polygon을 여러 성분으로 나눌 수 있다. Polygon 방식은 원본 라벨과 순번을 보존한다.
- GT mask와 raster union은 52/84장 일치했다. 나머지 32장의 합산 차이는 GT에만 20,750 px, Polygon에만 76 px였다. 전체 합집합 IoU는 99.606%, 비어 있지 않은 이미지 중 최저는 97.790%였다. 픽셀 혼동행렬은 승인 GT, Region 지표는 원본 Polygon을 사용하며 결합 차이는 기록했다. 이 차이를 주석 오류나 재현 전처리 차이 중 하나로 단정하지 않는다.
- 초기 원본 CSV 형식 확인에는 전 분할의 행이 들어 있는 파일의 문자열 필드 빈도 확인이 포함됐다. 개별 Polygon 좌표 해석과 GT 마스크·확률맵 로딩, 평가와 GUI 검증은 고정 Validation으로 제한했다. Test 이미지·마스크나 Test 확률맵을 열거나 평가하지 않았다.
- 공개 다운로드 안내는 확인했으나 데이터·파생 GT 팩의 재배포 허가를 확인하지 못했다. EXE에는 소스 및 프로토콜만 내장하며 권한 있는 관리자가 준비하는 자산 팩은 별도다.

## 세 모델 재평가

| 모델 | Threshold | Region Recall | Pixel Recall | IoU | FPR | sPRO | AUsPRO | Small Recall (<5k px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cityscapes | 0.417 | 91.32% (221/242) | 93.08% | 84.24% | 1.9978% | 91.40% | 0.870402 | 23/42 |
| ADE20K | 0.370 | 92.56% (224/242) | 93.67% | 84.79% | 1.9967% | 93.20% | 0.886691 | 25/42 |
| Random | 0.474 | 80.58% (195/242) | 79.97% | 72.38% | 1.9996% | 81.79% | 0.750084 | 14/42 |

기존 Connected Component 결과는 각각 219/240, 222/240, 193/240이고 threshold는 0.501, 0.378, 0.508이었다. 영역 단위가 바뀌고, 기존의 평균 overlap 대신 v0.1 sPRO가 10%에서 포화되며, threshold 동률도 기존의 낮은 FPR 우선 대신 sPRO 우선이다. 따라서 차이는 전체 프로토콜 변경의 결과다. 지금의 sPRO/AUsPRO를 이전 수치와 직접 비교하지 않는다. 이 관측으로 초기화 방식의 우월성이나 최종 모델을 확정하지 않는다.

## 검증 방법과 로컬 결과

### 팀 배포 전 최종 검증

- Validation Manifest의 영역 ID 242개가 모두 고유하며 각 이미지의 원본 순서에 따라 연속적인 `pNNN` 번호를 갖는다. 세 모델의 전체 Threshold 후보 3,006행에서 `sPRO ≥ Region Recall`을 확인했다. 검출된 영역은 두 지표 모두 1을 기여하며, 미탐 영역은 sPRO에 0 이상의 값을 기여한다. AUsPRO는 중복 FPR의 최대 sPRO, 상한 관측값과 선형 보간, 적분 범위 부족을 단위 테스트로 확인했다.
- 최종 소스의 단위 테스트 14개가 통과했다. 한글·공백이 있는 확률맵·메타데이터·GT·결과 경로에서 CLI 세 모델과 Tkinter GUI를 실행했다. 이전 v0.1 대비 각 평가의 Threshold·이미지·영역·sweep 파일 네 개가 SHA-256 기준으로 일치했고, GUI가 생성한 제출 선언의 SHA-256도 평가 Manifest와 일치했다.
- Windows 11 x64, Python 3.12.10, Tcl/Tk 8.6.15, PyInstaller 6.19.0으로 빌드했다. 최종 CLI EXE는 이 PC에서 Smart App Control에 의해 시작 단계에서 차단됐다. Win32 오류는 `4551`, Code Integrity 이벤트는 `3077`, 상태는 `0xC0E90002`였다. CLI 차단과 Python 미설치 별도 Windows PC에서의 실행 검증은 미해결로 유지한다. EXE 해시는 빌드가 생성한 `SHA256SUMS.txt`에 기록한다.
- 이전 기록의 Exp01·Exp02 산출물 및 고정 Split 1,173개를 SHA-256으로 다시 대조했다. 이번 최종 검증에서는 기존 Validation 팩만 사용했으며 Test Annotation·이미지·GT·확률맵에 접근하지 않았다. 원본에서 팩을 새로 만드는 검증은 Validation 전용 주석 CSV를 확보한 환경에서 수행한다.

- 단위 테스트 12개는 정답이 알려진 예측, 경계·겹침 Polygon, 영면적, 분모 0, 고정 FPR 동률, 사다리꼴 보간, 입력 오류, GT·split·팩 검증을 포함한다. Threshold 최적화의 벡터 방식은 1,002개 후보의 직접 픽셀 비교와 일치했다.
- `runs/common_evaluation/`에 Validation 자산팩과 3개 평가 결과, 비교표를 저장한다. 기존 Exp01·Exp02 디렉터리와 비교 기록은 보존했다. Windows PyInstaller 결과의 빌드 버전과 소스 SHA는 배포 폴더의 `build_info.json`에 기록한다.
- 검수 우선순위는 Polygon rasterization과 GT 경계 차이 처리, sPRO 포화 수식 및 AUsPRO 보간 규칙에 대한 팀 합의, 권한을 확인한 Validation 팩의 배포다.

### 최종 GUI EXE 수동 실행 확인

2026-11-27 사용자가 `team_release_v01/UC-FoodWaste-Evaluator.exe`에서 Exp02-A ADE20K Validation 평가를 완료했다. EXE SHA-256은 `68280084a117d2d09fbfc6ed1fac11c6a29d5d8f8f2a17f5f19539915ae5dfd3`이며, GUI 결과의 빌드 메타데이터는 해당 배포 폴더의 `build_info.json`과 일치했다.

| Threshold | Region Recall | IoU | Pixel Recall | Pixel Precision | Pixel FPR | sPRO | AUsPRO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.37 | 92.5620% | 84.7854% | 93.6734% | 89.9354% | 1.9967% | 93.1951% | 88.6691% |

- 사용자 홈의 `UC-FoodWaste Evaluations/20261127-172302-560552/`와 기존 `runs/common_evaluation/validated_v01/ADE20K/`를 대조했다. `selected_threshold.json`과 이미지·영역·그룹·크기별 지표, GT 감사, threshold sweep CSV 6개가 SHA-256 기준으로 일치했다.
- `evaluation_summary.json`은 표시 모델명만 달랐다. SVG 4개도 모델명을 통일하면 바이트 단위로 일치했다. 평가 Manifest의 확률맵 84장, GT 팩 및 프로토콜 해시는 기준 결과와 같았다. 차이는 빌드 정보와 제출 메타데이터였으며, GUI가 생성한 `input_metadata.json`의 SHA-256은 해당 Manifest와 일치했다.

### 팀원 전달 파일과 사용 순서

1. `UC-FoodWaste-Evaluator.exe`, `SHA256SUMS.txt`, `build_info.json`, [한글 사용·확률맵 생성 가이드](README.md)를 전달한다. `ucfw-evaluate.exe`는 자산 생성 및 CLI용이며 팀 PC의 실행 정책이 허용하는 환경에서 사용한다.
2. 각 팀원은 권한 있는 원본으로 생성하고 기준 해시와 대조한 `validation_assets/`를 EXE 옆에 둔다. 내부 파일은 `asset_manifest.json`, `region_manifest.json`, `validation_pixels.npz`다. GT 팩 재배포 권한 확인 전에는 배포 묶음과 GitHub에 포함하지 않는다.
3. GUI에서 **폴더 선택 → 평가 시작**을 누른다. 고정 Validation 확률맵 84장을 사용하고, 제출 정보 파일이 없으면 실제 클래스 매핑·확률 변환·공간 정렬 방식을 입력한다. **결과 폴더 열기**로 CSV, JSON, SVG를 확인한다.

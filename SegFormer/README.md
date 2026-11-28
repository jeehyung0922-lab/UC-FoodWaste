# SegFormer-B0 · 독립 설치 및 실행

이 폴더의 [팀원 원본 안내](README_KO.md), `segformer.py`, `dataset.py`, `requirements.txt`, Split, RGB 체크포인트와 설정을 사용한다. [원본 SHA-256과 기준 결과](OVERVIEW.md)를 먼저 확인한다. Phase 1은 로컬 배치와 정적 검증 단계이며 아래 명령은 다음 단계 승인 후 실행한다.

## 1. RTX 5090 환경

Windows PowerShell에서 이 폴더를 연다. 기준 후보는 Python 3.13.15 x64, PyTorch `2.9.1+cu128`, Transformers `4.57.1`이다. 서버에서 Git으로 승인된 코드를 동기화한 뒤 진행한다. 기존 NVIDIA 드라이버와 YOLO 환경을 유지한다.

```powershell
py -3.13 -c "import platform; print(platform.python_version(), platform.machine())"
py -3.13 -m venv .venv-segformer-cuda
$py = (Resolve-Path .\.venv-segformer-cuda\Scripts\python.exe).Path
& $py -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r .\requirements.txt
& $py -m pip check
& $py .\segformer.py --help
& $py -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability()); a=torch.ones((32,32),device='cuda'); b=a@a; torch.cuda.synchronize(); assert b[0,0].item()==32"
& $py -m pip freeze
```

Python 출력이 기준 버전과 일치하는지 먼저 확인한다. GPU 인식과 실제 CUDA Tensor 연산이 통과하면 모델 체크포인트 로딩, 합성 입력 FP32 forward, 실제 Validation 1장 순으로 검증한다. 단일 이미지 검증은 원본 모델과 전처리를 재사용하는 외부 진단으로 수행할 수 있다. 원본 `export`는 고정 Validation 전체를 요구하므로 이 단계에서 Split을 축소하지 않는다. 검증을 통과한 라이브러리 버전과 OS·GPU·드라이버·Git SHA를 실행 기록에 남긴다.

## 2. 기존 RGB 체크포인트 추론

팀원이 실제 사용한 Validation packed NPZ 84장의 승인된 위치와 입력 해시를 확보한다. 파일은 `image` 배열을 포함하며 팀 입력에 맞게 RGB 채널 `[2,1,0]`, divisor 1을 사용한다. 기존 YOLO의 JPG나 PNG는 팀원 입력과 동일성이 확인될 때까지 대체 입력으로 사용하지 않는다. 체크포인트의 세 실행 필수 파일 `config.json`, `input_spec.json`, `model.safetensors`를 같은 로컬 폴더에 두고 해시를 확인한다.

환경 및 Smoke 검수가 끝난 뒤 실제 위치를 지정한다. 실행 결과에는 새 경로를 사용한다.

```powershell
$npz = 'D:\approved\validation_npz'
$out = 'D:\research_runs\segformer\SF-RGB-R01\probability_maps'
Get-FileHash -Algorithm SHA256 -LiteralPath .\checkpoints\rgb_baseline\model.safetensors
& $py .\segformer.py check --part val --modalities rgb --npz-dir $npz
& $py .\segformer.py export --checkpoint .\checkpoints\rgb_baseline --npz-dir $npz --split .\data_split.csv --device cuda --out $out
```

기존 RGB 추론은 로컬 safetensors를 읽으며 Encoder 다운로드가 필요 없다. 결과는 H512×W768 float32 확률맵 84장과 `input_metadata.json`이다. Common Evaluator v0.1은 **별도 환경 또는 GUI 배포본**에서 실행한다. 평가기 소스는 Pillow 13.0.0을 사용하며 모델의 원본 의존성은 Pillow <13이다. 평가기와 승인된 `validation_assets/` 경로를 외부에서 지정한다.

```powershell
$evalPy = 'D:\evaluator_env\Scripts\python.exe'
$evalMain = 'D:\common_evaluator\main.py'
$assets = 'D:\approved\validation_assets'
& $evalPy $evalMain evaluate --model 'SegFormer-B0 RGB' --maps $out --assets $assets --out 'D:\research_runs\segformer\SF-RGB-R01\evaluation'
```

결과의 재선정 Threshold와 기존 Threshold 0.137의 sweep 행을 각각 기록한다. 기존 팀 제출의 확률맵은 CPU FP32로 생성됐으므로 GPU 결과에 차이가 있을 때는 CPU 대조를 별도 경로에서 수행한다.

## 3. 독립 폴더와 제출 범위

모델 실행의 필수 파일은 `segformer.py`, `dataset.py`, `requirements.txt`, `data_split.csv`, 체크포인트 폴더의 세 실행 필수 파일이다. `README.md`, `README_KO.md`, `OVERVIEW.md`, 원본 `run_config.json`과 `reference/`는 설치 안내와 출처를 보존한다. 승인된 범위 내에서 전체 원본 트리의 상대 경로를 유지해 제출할 수 있다. 데이터는 외부 경로로 연결하고 가상환경, 확률맵, 결과 디렉터리는 패키지와 분리한다.

신규 학습에 한해 `nvidia/mit-b0`을 사용한다. 오프라인 신규 학습에 필요한 Encoder는 권한을 확인해 별도로 준비하고 `--encoder <로컬 폴더> --offline`으로 지정한다. Common Evaluator는 버전과 소스 해시를 고정해 독립 연결하며, 동봉은 배포 권한을 확인한 뒤 결정한다. 팀원 코드·원본 Encoder·fine-tuned 체크포인트·데이터·GT·평가기의 공유 권한을 각각 확인한다.

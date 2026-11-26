# 실행 환경과 전송 검증

| 항목 | 로컬 CPU 검증 | Exp01 GPU 서버 기록 |
|---|---|---|
| OS | Windows 11 | Windows Server 2022 |
| Python | 3.12.10 | 3.13.15 |
| PyTorch | 2.15.0+cpu | 2.9.1+cu128 |
| Ultralytics | 8.4.52 | 8.4.52 |
| CUDA/GPU | 미가용 | CUDA 빌드 12.8, RTX 5090, compute capability 12.0 |

기록 근거는 Exp01의 `environment.json`과 로컬의 [실험 로그](../EXPERIMENT_LOG.md)다. GPU는 NVIDIA 드라이버와 CUDA wheel이 필요하다. [PyTorch 설치 목록](https://pytorch.org/get-started/previous-versions/)

```powershell
python -m venv .venv-yolo-cuda
$py = '.\.venv-yolo-cuda\Scripts\python.exe'
& $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-yolo-exp01.txt
& $py scripts/yolo/exp01.py environment
```

자산 단계와 학습·평가 실행은 [README](../README.md#exp01--yolo26s-sem-rgb-baseline)를 따른다. `audit`·`prepare`는 기본적으로 Train과 Validation만 읽는다. CPU 검증은 별도 환경에서 PyTorch CPU wheel과 `requirements-yolo.txt`를 설치해 실행한다. 시각화는 `run`, `dataset`, `compare` 서브커맨드로 CPU에서 수행한다. [명령 예시](figures/README.md)

VS Code Remote SSH로 프로젝트를 옮길 때 저장소 코드·설정·필요 데이터 및 모델 가중치를 전송하고 가상환경은 서버에서 만든다. Manifest 비교는 환경 폴더를 건너뛴다. 같은 검증 스크립트를 양쪽에서 실행한다.

```powershell
python scripts/verify_transfer.py manifest --root . --out runs/transfer_check/local-manifest.json
# 서버 터미널의 프로젝트 루트에서 실행
python scripts/verify_transfer.py manifest --root . --out runs/transfer_check/server-manifest.json
# 서버 Manifest를 로컬에 내려받은 뒤 실행
python scripts/verify_transfer.py compare --local runs/transfer_check/local-manifest.json --server runs/transfer_check/server-manifest.json --out-dir runs/transfer_check/comparison
```

기존 Manifest와 보고서는 보존하며 재실행 시 새 이름을 사용한다. 프로젝트·파일 크기·SHA-256 검증은 전송 시점에 3,293파일 일치를 확인했다. 가상환경의 정합성은 별도 설치 후 `environment`와 CUDA 점검으로 확인한다. 로컬 환경에는 서버 경로, 계정 이름 등 공개 문서에 불필요한 운영정보를 복사하지 않는다.

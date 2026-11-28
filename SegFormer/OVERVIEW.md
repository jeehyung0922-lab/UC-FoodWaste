# SegFormer-B0 원본 및 기준 결과

팀 제공 ZIP의 10개 파일을 바이트 그대로 배치했다. ZIP의 최상위 `SegFormer_B0_Team/`을 이 폴더에 대응시켰으며 그 아래 경로는 원본과 같다. `README.md`와 이 문서는 로컬 연구 안내로 추가했다.

## 원본 SHA-256

| 대상 | SHA-256 |
|---|---|
| `SegFormer_B0_Team.zip` | `9aece0d05a3d5bebb181e800efbeb187af2741bbcf549657436d6a6580a9a0fd` |
| `README_KO.md` | `a650b42c05c6a2ef2f82684077a9487fe7419083e16078af8520d30b278ab674` |
| `requirements.txt` | `e149857479fb70f4901b6a26874ed095d684d9b2aa7b0d120df050091b39d200` |
| `segformer.py` | `b11c3dae56704138f6438c21a646d641d9659479d2e44d1613788141d158aac7` |
| `dataset.py` | `4b87292e6f3db8551f763d0e37ad3fe4fa72098e65ca0eb4cbdaf8af9e9ceeed` |
| `data_split.csv` | `66518ca0b69f4a7196a624c9d999ab70d2489cbc415765e97e34cbb3ac63c3a6` |
| `checkpoints/rgb_baseline/config.json` | `f26c5d04d6ab41b338c8775cd3cca4255c62fc7d3c818e5bf8490ce63555c795` |
| `checkpoints/rgb_baseline/input_spec.json` | `b4aa07af4a7dde6253a473f731ceafc73affcccfebf5371e8bddf3071628ff58` |
| `checkpoints/rgb_baseline/model.safetensors` | `537d897c0e7f3874adc9a81eda9b22ad2a8a4d2b67bac19ec0253d3a813dccd3` |
| `checkpoints/rgb_baseline/run_config.json` | `c43ed04f9430980a168806e9ff6b8f5670dc4d695ae7ff74041bdfb37a093433` |
| `reference/rgb_common_evaluation.json` | `09cad8653a79930f0152b94279d86611bb979abedf99e011b3c9a5171bda8514` |

Split은 저장소의 `metadata/data_split.csv`와 바이트 단위로 같다. Train/Validation/Test는 374/84/84장, Block은 39/10/10개이며 ID·파일명 중복 및 분할 간 Block 중복이 없다. LF 정규화 Split SHA-256은 `859dcb499be4519087815814d1abd1cdcb00377c9b551a056883047f6a0059ba`이며 정렬 Validation 파일명 목록 해시는 `631f382dd4fcc80d2365cab8366a55875f68eeff8e8de273dca43529761368b1`이다. Git 속성은 원본 파일의 줄바꿈 변환을 끈다.

## RGB 기준과 공통 평가

원본 [기준 JSON](reference/rgb_common_evaluation.json)은 팀원 제출 결과 ZIP의 요약과 일치했다. 제출 확률맵은 CPU FP32 추론에서 생성됐다. 역으로 추론 환경을 완전히 복구할 수 있는 패키지 목록, 원본 확률맵, 추론 batch는 아직 확보되지 않았다. 학습 환경 기록은 Tesla T4, PyTorch `2.14.1+cu130`, Transformers `4.57.1`이다.

| Threshold | Region Recall | Pixel Recall | Precision | Dice | IoU | FPR | sPRO | AUsPRO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.137 | 94.6281% (229/242) | 94.4113% | 89.9964% | 92.1510% | 85.4445% | 1.9989% | 95.0675% | 0.894787 |

| 평가 식별자 | SHA-256 |
|---|---|
| Common Evaluator v0.1 프로토콜 | `7fdf690057ef3c4dba7c31d4746aaf114dc2e2c7a0a8a83977972b0b890ce84f` |
| Polygon Region Manifest | `08a9c70733fa4f7f520bcd92cbf9efa88f244d4d81ab3bb0253f770f155f6291` |
| 검증본 GT·Polygon 픽셀 NPZ | `138a7743246251d953d8d00396521498e92d6839dcfbbab382566e0fd9dad608` |

체크포인트 해시는 `input_spec.json`과 제출 평가 Manifest에 일치했다. Phase 1에서는 이 기준을 기록했으며, 추론에 의한 재현은 다음 단계에서 검증한다.

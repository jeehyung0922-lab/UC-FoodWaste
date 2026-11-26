"""Locate Colab data and start the RGB+Thermal trial."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARCHIVE_ROOT.parents[1]


def locate(names, configured):
    if configured:
        candidates = [Path(configured)]
    else:
        result = subprocess.run(
            ["find", "-L", "/content", "-type", "f", "(",
             "-name", "image000010.npz", "-o", "-name", "data_processed.zip", ")"],
            capture_output=True, text=True,
        )
        candidates = [Path(line) for line in result.stdout.splitlines()]
    folders = {p.parent if p.suffix == ".npz" else p for p in candidates}
    npz = next((p for p in sorted(folders)
                if all((p / f"{name}.npz").is_file() for name in names)), None)
    return npz, [p for p in candidates if p.name == "data_processed.zip"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    split = Path(os.getenv("UC_FOODWASTE_SPLIT_CSV", PROJECT_ROOT / "metadata/data_split.csv"))
    with split.open(encoding="utf-8-sig") as stream:
        names = sorted(row["filename"] for row in csv.DictReader(stream))
    configured = os.environ.get("UC_FOODWASTE_NPZ_DIR")
    print("Colab 데이터 위치 확인 중...", flush=True)
    npz, archives = locate(names, configured)
    if npz is None:
        raise SystemExit(
            "현재 Colab에서 NPZ 542개가 있는 폴더에 접근할 수 없습니다. "
            "NPZ가 다른 계정의 '공유 문서함'에 있으면 Google Drive에서 "
            "해당 폴더의 바로가기를 내 드라이브에 추가한 뒤 다시 실행하세요."
        )
    masks = Path(os.getenv("UC_FOODWASTE_DATA_ROOT", PROJECT_ROOT / "data")) / "data_processed/masks"
    if not all((masks / f"{n}.png").is_file() for n in names):
        if configured:
            _, archives = locate([], None)
        for path in sorted(archives):
            with ZipFile(path) as archive:
                entries = {
                    Path(name).stem: name for name in archive.namelist()
                    if Path(name).parent.name == "masks" and Path(name).suffix.lower() == ".png"
                }
                if not entries.keys() >= set(names):
                    continue
                masks.mkdir(parents=True, exist_ok=True)
                for name in names:
                    target = masks / f"{name}.png"
                    if not target.is_file():
                        target.write_bytes(archive.read(entries[name]))
                break
        else:
            raise SystemExit("마스크 542개를 포함한 data_processed.zip을 현재 Colab에서 찾지 못했습니다.")
    print(f"NPZ 폴더: {npz}", flush=True)
    subprocess.run([
        sys.executable, "-u", str(ARCHIVE_ROOT / "scripts/train.py"), "--mode", "rgb_thermal",
        "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
        "--npz-dir", str(npz), "--mask-dir", str(masks), "--split-csv", str(split),
    ], check=True)


if __name__ == "__main__":
    main()

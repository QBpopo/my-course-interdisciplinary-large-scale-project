#!/usr/bin/env python
"""下载运行所需的模型文件到 ``models/``。

只需要一个模型：MediaPipe 的 Face Landmarker（3.6 MB），它负责输出 478 个面部关键点。
睁 / 闭的判定是几何公式（EAR），不需要第二个模型。

用法:

    python scripts/download_models.py
    python scripts/download_models.py --force     # 已存在也重新下载
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# 允许不安装包、直接 `python scripts/download_models.py` 也能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eye_state.config import MODELS_DIR  # noqa: E402

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/" "face_landmarker/float16/1/face_landmarker.task"


def _report(prefix: str, path: Path) -> None:
    print(f"[{prefix}] {path.name} ({path.stat().st_size / 1024:.1f} KB)")


def download_face_landmarker(force: bool = False) -> Path:
    """下载 MediaPipe 人脸关键点模型；已存在则跳过。"""
    target = MODELS_DIR / "face_landmarker.task"
    if target.exists() and not force:
        _report("skip", target)
        return target

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {target.name} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, target)
    _report("done", target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    download_face_landmarker(force=args.force)
    print(f"\n模型目录: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

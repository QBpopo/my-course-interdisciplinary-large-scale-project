"""pytest 全局配置与共享 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# 测试图片的来源目录，按优先级排列
IMAGE_DIRS = (PROJECT_ROOT / "tmp/tests", PROJECT_ROOT / "sample_images")

MISSING_IMAGES_HINT = "没有测试图片；先跑 " "python scripts/validate_on_dataset.py --per-label 5"


def _find(prefix: str) -> list[Path]:
    for directory in IMAGE_DIRS:
        if directory.is_dir():
            found = sorted(directory.glob(f"{prefix}_*.jpg"))
            if found:
                return found
    return []


@pytest.fixture(scope="session")
def landmarker_model() -> Path:
    """MediaPipe 关键点模型路径，没有就跳过。"""
    if not MODEL_PATH.exists():
        pytest.skip(f"缺少模型文件 {MODEL_PATH}，先跑 python scripts/download_models.py")
    return MODEL_PATH


@pytest.fixture(scope="session")
def open_eye_image() -> Path:
    images = _find("open")
    if not images:
        pytest.skip(MISSING_IMAGES_HINT)
    return images[0]


@pytest.fixture(scope="session")
def closed_eye_image() -> Path:
    images = _find("closed")
    if not images:
        pytest.skip(MISSING_IMAGES_HINT)
    return images[0]

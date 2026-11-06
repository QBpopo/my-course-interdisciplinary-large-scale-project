"""端到端冒烟测试 —— 需要模型文件和测试图片，缺任一则自动跳过。

准备测试图片::

    python scripts/validate_on_dataset.py --per-label 5
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_state import EyeStatePipeline
from eye_state.pipeline import METHODS


@pytest.fixture(scope="module")
def pipeline(landmarker_model: Path):
    with EyeStatePipeline(landmarker_model, method="ear", video_mode=False) as pipe:
        yield pipe


@pytest.mark.smoke
class TestPipelineSmoke:
    def test_open_image_is_detected_as_open(self, pipeline, open_eye_image):
        image = cv2.imread(str(open_eye_image))
        pipeline.smoother.clear()
        canvas, result = pipeline.process(image)

        assert result.face_found, "没检测到人脸"
        assert result.eyes, "没定位到眼睛"
        assert canvas.shape == image.shape, "输出画布尺寸应保持不变"
        assert all(eye.is_open for eye in result.eyes.values()), result.summary()
        assert result.alert is False

    def test_closed_image_is_detected_as_closed(self, pipeline, closed_eye_image):
        image = cv2.imread(str(closed_eye_image))
        pipeline.smoother.clear()
        _, result = pipeline.process(image)

        assert result.face_found
        assert result.both_closed, result.summary()

    def test_blank_image_reports_no_face(self, pipeline):
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        canvas, result = pipeline.process(blank)
        assert result.face_found is False
        assert result.eyes == {}
        assert result.summary() == "no face"
        assert canvas.shape == blank.shape

    def test_repeated_frames_are_stable(self, pipeline, open_eye_image):
        """加了滑动窗口后，连续多帧的状态不该反复横跳。"""
        image = cv2.imread(str(open_eye_image))
        pipeline.smoother.clear()
        states = []
        for _ in range(10):
            _, result = pipeline.process(image)
            states.append(result.eyes["L"].is_open)
        assert all(states), f"状态抖动: {states}"


@pytest.mark.smoke
class TestMethods:
    def test_all_methods_are_wired(self):
        assert set(METHODS) == {"ear", "ratio"}

    def test_ratio_method_runs(self, landmarker_model, open_eye_image):
        with EyeStatePipeline(landmarker_model, method="ratio", video_mode=False) as pipe:
            image = cv2.imread(str(open_eye_image))
            _, result = pipe.process(image)
            assert result.eyes, "ratio 判据没有产出结果"
            assert all(eye.is_open for eye in result.eyes.values()), result.summary()

    def test_unknown_method_is_rejected(self, landmarker_model):
        with pytest.raises(ValueError):
            EyeStatePipeline(landmarker_model, method="nope")

    def test_calibration_threshold_is_applied(self, landmarker_model, open_eye_image):
        """校准开启时，阈值应当由观测到的度量峰值推导出来。"""
        with EyeStatePipeline(landmarker_model, method="ear", video_mode=False, calibrate_seconds=0.0) as pipe:
            assert pipe.calibrated is True
            assert pipe.smoother.thresh == pytest.approx(0.24)

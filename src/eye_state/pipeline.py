"""整条流水线：MediaPipe 定位 -> 度量 -> 时序平滑 -> 阈值判定。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import (
    DEFAULT_ALERT_SECONDS,
    DEFAULT_BOX_PAD,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_THRESHOLDS,
)
from .geometry import METRICS, Box, eye_boxes, to_points
from .render import draw_alert, draw_eye, draw_hud, is_window_visible
from .smoothing import DrowsinessGuard, EyeSmoother

METHODS = ("ear", "ratio")
ROI_WINDOW = "eye ROI"


@dataclass(frozen=True)
class EyeResult:
    """单只眼睛在本帧的结果。"""

    side: str
    box: Box
    metric: float  # 平滑后的度量值（判据用的是它）
    raw: float  # 本帧原始度量值
    is_open: bool


@dataclass
class FrameResult:
    """一帧的整体结果。"""

    face_found: bool = False
    eyes: dict[str, EyeResult] = field(default_factory=dict)
    alert: bool = False

    @property
    def both_closed(self) -> bool:
        """是否所有可见的眼睛都判为闭合。"""
        return bool(self.eyes) and all(not e.is_open for e in self.eyes.values())

    def summary(self) -> str:
        """给终端打印用的单行摘要。"""
        if not self.eyes:
            return "no face"
        return " ".join(f"{side}:{'OPEN' if e.is_open else 'CLOSED'}({e.metric:.2f})" for side, e in sorted(self.eyes.items()))


class EyeStatePipeline:
    """把定位、判据、平滑、告警串起来。

    :param landmarker_path: MediaPipe Face Landmarker 的 ``.task`` 模型
    :param method: ``ear`` / ``ratio``，见 :data:`METHODS`
    :param video_mode: True 用 ``detect_for_video``（需要递增时间戳），
        False 用 ``detect``（每帧独立，适合图片）
    """

    def __init__(
        self,
        landmarker_path: Path,
        *,
        method: str = "ear",
        video_mode: bool = False,
        pad: float = DEFAULT_BOX_PAD,
        window: int = DEFAULT_SMOOTH_WINDOW,
        thresh: float | None = None,
        alert_seconds: float = DEFAULT_ALERT_SECONDS,
        calibrate_seconds: float = 0.0,
        show_roi: bool = False,
        show_hud: bool = True,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method 必须是 {METHODS} 之一，收到 {method!r}")
        self.method = method
        self.video_mode = video_mode
        self.pad = pad
        self.show_roi = show_roi
        self.show_hud = show_hud
        self._roi_shown = False  # 用户可能中途点叉关掉 ROI 窗口

        self.smoother = EyeSmoother(
            window=window,
            thresh=DEFAULT_THRESHOLDS[method] if thresh is None else thresh,
        )
        self.guard = DrowsinessGuard(alert_seconds)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=(mp_vision.RunningMode.VIDEO if video_mode else mp_vision.RunningMode.IMAGE),
            num_faces=1,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # 个人化校准：开算后 calibrate_seconds 秒内取度量峰值作为「完全睁眼」基线，
        # 用 峰值的 60% 当阈值。不同人 / 不同摄像头的 EAR 基线差异不小。
        self.calibrate_seconds = calibrate_seconds
        self._calib_start = time.perf_counter()
        self._calib_peak = 0.0
        self.calibrated = calibrate_seconds <= 0

        self._last_frame_at = time.perf_counter()
        self._fps = 0.0

    # -- 生命周期 ---------------------------------------------------------- #
    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> EyeStatePipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 主处理函数 -------------------------------------------------------- #
    def process(self, bgr: np.ndarray, ts_ms: int | None = None) -> tuple[np.ndarray, FrameResult]:
        """处理一帧，返回 ``(带标注的图, FrameResult)``。

        ``ts_ms`` 只在 ``video_mode=True`` 时需要，且必须单调递增。
        """
        height, width = bgr.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        if self.video_mode:
            detection = self.landmarker.detect_for_video(mp_image, int(ts_ms or 0))
        else:
            detection = self.landmarker.detect(mp_image)

        canvas = bgr.copy()
        result = FrameResult(face_found=bool(detection.face_landmarks))

        if not detection.face_landmarks:
            self.smoother.clear()
            self.guard.update(False, time.perf_counter())
            self._draw_hud(canvas, result)
            return canvas, result

        points = to_points(detection.face_landmarks[0], width, height)
        boxes = eye_boxes(points, width, height, self.pad)

        raw = METRICS[self.method](points)
        sides = [s for s in sorted(boxes) if s in raw]
        self._update_calibration([raw[s] for s in sides])

        roi_previews = []
        for side in sides:
            metric, is_open = self.smoother.update(side, raw[side])
            eye = EyeResult(side, boxes[side], metric, float(raw[side]), is_open)
            result.eyes[side] = eye
            draw_eye(canvas, eye.box, eye.is_open, eye.metric)
            if self.show_roi:
                x0, y0, x1, y1 = eye.box
                roi_previews.append(cv2.resize(bgr[y0:y1, x0:x1], (112, 112)))

        result.alert = self.guard.update(result.both_closed, time.perf_counter())
        self._draw_hud(canvas, result)

        if roi_previews:
            self._show_roi_previews(roi_previews)

        return canvas, result

    def _show_roi_previews(self, previews: list[np.ndarray]) -> None:
        """显示眼部裁剪小图。用户点叉关掉后就不再重复弹出。"""
        if self._roi_shown and not is_window_visible(ROI_WINDOW):
            self.show_roi = False
            return
        cv2.imshow(ROI_WINDOW, np.hstack(previews))
        self._roi_shown = True

    # -- 内部实现 ---------------------------------------------------------- #
    def _update_calibration(self, values: list[float]) -> None:
        if self.calibrated or not values:
            return
        self._calib_peak = max(self._calib_peak, max(values))
        if time.perf_counter() - self._calib_start >= self.calibrate_seconds:
            if self._calib_peak > 0.0:
                self.smoother.set_threshold(self._calib_peak * 0.6)
            self.calibrated = True

    def _draw_hud(self, canvas: np.ndarray, result: FrameResult) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_at
        if dt > 1e-6:
            self._fps = 1.0 / dt if self._fps == 0.0 else 0.9 * self._fps + 0.1 / dt
        self._last_frame_at = now

        if not self.calibrated:
            left = max(0.0, self.calibrate_seconds - (now - self._calib_start))
            third = f"calibrating {left:.1f}s (请睁大眼睛)"
        else:
            third = f"thresh {self.smoother.thresh:.3f}"

        if self.show_hud:
            draw_hud(
                canvas,
                [
                    f"FPS {self._fps:5.1f}",
                    f"method {self.method}",
                    third,
                    result.summary() if result.face_found else "no face",
                ],
            )

        if result.alert:
            draw_alert(canvas)

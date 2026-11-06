"""睁眼 / 闭眼检测：MediaPipe 关键点定位 + EAR 几何判据。

最简用法::

    from eye_state import EyeStatePipeline

    with EyeStatePipeline("models/face_landmarker.task") as pipeline:
        canvas, result = pipeline.process(bgr_image)
        print(result.summary())
"""

from .config import DEFAULT_LANDMARKER_PATH, DEFAULT_THRESHOLDS
from .geometry import EAR_PTS, EYE_CONTOUR, ear, eye_boxes, height_ratio, to_points
from .pipeline import METHODS, EyeResult, EyeStatePipeline, FrameResult
from .smoothing import DrowsinessGuard, EyeSmoother

__version__ = "0.1.0"

__all__ = [
    "EAR_PTS",
    "EYE_CONTOUR",
    "METHODS",
    "DEFAULT_LANDMARKER_PATH",
    "DEFAULT_THRESHOLDS",
    "DrowsinessGuard",
    "EyeResult",
    "EyeSmoother",
    "EyeStatePipeline",
    "FrameResult",
    "__version__",
    "ear",
    "eye_boxes",
    "height_ratio",
    "to_points",
]

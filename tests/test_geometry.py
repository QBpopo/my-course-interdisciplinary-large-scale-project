"""眼部几何计算测试 —— 纯数学，不需要模型文件。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from eye_state.geometry import (
    EAR_PTS,
    EYE_CONTOUR,
    ear,
    eye_boxes,
    height_ratio,
    to_points,
)

SIZE = 1000


class FakeLandmark(NamedTuple):
    """模拟 MediaPipe 的归一化关键点。"""

    x: float
    y: float


def build_landmarks(overrides: dict[int, tuple[float, float]], count: int = 478):
    """构造 count 个关键点，用 overrides 覆盖指定下标（归一化坐标）。"""
    base = [FakeLandmark(0.5, 0.5) for _ in range(count)]
    for idx, (x, y) in overrides.items():
        base[idx] = FakeLandmark(x, y)
    return base


def eye_overrides(pts: dict[str, list[int]], coords: list[tuple[float, float]]):
    return {idx: coord for idx, coord in zip(pts, coords)}


# 一个「完全睁开」的右眼：上睑抬起 20px、下睑下沉 20px、内外眼角相距 60px
OPEN_RIGHT_EYE = {
    33: (0.400, 0.500),  # 外眼角
    159: (0.420, 0.480),  # 上睑 1
    158: (0.440, 0.480),  # 上睑 2
    133: (0.460, 0.500),  # 内眼角
    153: (0.440, 0.520),  # 下睑 2
    145: (0.420, 0.520),  # 下睑 1
}

# 一个「几乎闭合」的右眼：上下睑只差 2px
CLOSED_RIGHT_EYE = {
    33: (0.400, 0.500),
    159: (0.420, 0.499),
    158: (0.440, 0.499),
    133: (0.460, 0.500),
    153: (0.440, 0.501),
    145: (0.420, 0.501),
}


class TestToPoints:
    def test_converts_normalized_to_pixels(self):
        landmarks = build_landmarks({0: (0.25, 0.75), 1: (0.5, 0.5)}, count=2)
        points = to_points(landmarks, SIZE, SIZE)
        assert points.shape == (2, 2)
        assert points.dtype == np.float32
        np.testing.assert_allclose(points[0], [250.0, 750.0])
        np.testing.assert_allclose(points[1], [500.0, 500.0])


class TestEar:
    def test_open_eye_matches_hand_computed_value(self):
        # |p2-p6| = 40, |p3-p5| = 40, |p1-p4| = 60
        # EAR = (40 + 40) / (2 * 60) = 0.6667
        points = to_points(build_landmarks(OPEN_RIGHT_EYE), SIZE, SIZE)
        result = ear(points)
        assert result["R"] == pytest.approx(80.0 / 120.0)

    def test_closed_eye_is_near_zero(self):
        # EAR = (2 + 2) / (2 * 60) = 0.0333
        points = to_points(build_landmarks(CLOSED_RIGHT_EYE), SIZE, SIZE)
        assert ear(points)["R"] == pytest.approx(4.0 / 120.0)

    def test_open_is_much_larger_than_closed(self):
        open_ear = ear(to_points(build_landmarks(OPEN_RIGHT_EYE), SIZE, SIZE))["R"]
        closed_ear = ear(to_points(build_landmarks(CLOSED_RIGHT_EYE), SIZE, SIZE))["R"]
        assert open_ear > closed_ear * 10

    def test_scale_invariance(self):
        """归一化后距离远近不影响结果。"""
        small = ear(to_points(build_landmarks(OPEN_RIGHT_EYE), 200, 200))["R"]
        large = ear(to_points(build_landmarks(OPEN_RIGHT_EYE), 4000, 4000))["R"]
        assert small == pytest.approx(large)

    def test_degenerate_eye_is_skipped(self):
        """内外眼角重合时分母为 0，该只眼睛应被跳过而不是抛异常。"""
        degenerate = {33: (0.45, 0.5), 133: (0.45, 0.5), 159: (0.45, 0.49), 158: (0.45, 0.49), 153: (0.45, 0.51), 145: (0.45, 0.51)}
        assert "R" not in ear(to_points(build_landmarks(degenerate), SIZE, SIZE))

    def test_returns_both_sides_when_both_present(self):
        overrides = {**OPEN_RIGHT_EYE}
        for idx, coord in zip(EAR_PTS["L"], [(0.54, 0.5), (0.56, 0.48), (0.58, 0.48), (0.60, 0.5), (0.58, 0.52), (0.56, 0.52)]):
            overrides[idx] = coord
        assert set(ear(to_points(build_landmarks(overrides), SIZE, SIZE))) == {"L", "R"}


class TestEyeBoxes:
    def test_padding_expands_around_contour(self):
        overrides = {idx: (0.45, 0.425) for idx in EYE_CONTOUR["R"]}
        overrides.update({33: (0.40, 0.425), 133: (0.50, 0.425), 159: (0.45, 0.40), 153: (0.45, 0.45)})
        points = to_points(build_landmarks(overrides), SIZE, SIZE)
        assert eye_boxes(points, SIZE, SIZE, pad=0.35)["R"] == (382, 391, 517, 458)

    def test_zero_padding_hugs_contour(self):
        overrides = {idx: (0.45, 0.425) for idx in EYE_CONTOUR["R"]}
        overrides.update({33: (0.40, 0.425), 133: (0.50, 0.425), 159: (0.45, 0.40), 153: (0.45, 0.45)})
        points = to_points(build_landmarks(overrides), SIZE, SIZE)
        # 轮廓本身占 (400,400)-(500,450)，pad=0 时框正好贴合
        assert eye_boxes(points, SIZE, SIZE, pad=0.0)["R"] == (400, 400, 500, 450)

    def test_clamped_to_image_bounds(self):
        # 眼睛贴在左上角，pad 后应当被裁到画面内
        overrides = {idx: (0.02, 0.02) for idx in EYE_CONTOUR["R"]}
        overrides.update({33: (0.00, 0.02), 133: (0.04, 0.02), 159: (0.02, 0.00), 153: (0.02, 0.04)})
        points = to_points(build_landmarks(overrides), SIZE, SIZE)
        x0, y0, x1, y1 = eye_boxes(points, SIZE, SIZE, pad=0.5)["R"]
        assert (x0, y0) == (0, 0)
        assert 0 <= x1 <= SIZE - 1 and 0 <= y1 <= SIZE - 1

    def test_tiny_eye_is_discarded(self):
        overrides = {idx: (0.500, 0.500) for idx in EYE_CONTOUR["R"]}
        points = to_points(build_landmarks(overrides), SIZE, SIZE)
        assert eye_boxes(points, SIZE, SIZE, pad=0.35) == {}


class TestHeightRatio:
    def test_open_greater_than_closed(self):
        open_ratio = height_ratio(to_points(build_landmarks(OPEN_RIGHT_EYE), SIZE, SIZE))["R"]
        closed_ratio = height_ratio(to_points(build_landmarks(CLOSED_RIGHT_EYE), SIZE, SIZE))["R"]
        assert open_ratio > closed_ratio


class TestLandmarkIndexTables:
    def test_index_tables_are_consistent(self):
        assert set(EAR_PTS) == set(EYE_CONTOUR) == {"L", "R"}
        assert all(len(idx) == 6 for idx in EAR_PTS.values())

    def test_all_indices_within_face_mesh_range(self):
        for table in (EAR_PTS, EYE_CONTOUR):
            for indices in table.values():
                assert all(0 <= i < 478 for i in indices)

    def test_ear_indices_are_part_of_contour(self):
        """EAR 用的点应当来自眼睑轮廓，避免笔误。"""
        for side in ("L", "R"):
            assert set(EAR_PTS[side]) <= set(EYE_CONTOUR[side])

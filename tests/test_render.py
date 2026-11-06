"""绘制与窗口工具的测试。

绘制部分是纯函数，不需要模型文件也不需要 GUI；
窗口可见性的测试需要能创建窗口，无 GUI 环境会自动跳过。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from eye_state import render
from eye_state.config import COLOR_CLOSED, COLOR_OPEN

CANVAS_H, CANVAS_W = 240, 320


@pytest.fixture
def canvas() -> np.ndarray:
    return np.full((CANVAS_H, CANVAS_W, 3), 200, dtype=np.uint8)


class TestPanel:
    def test_darkens_target_region(self, canvas):
        before = canvas[50:80, 40:140].mean()
        render._panel(canvas, 40, 50, 140, 80)
        assert canvas[50:80, 40:140].mean() < before

    def test_leaves_outside_untouched(self, canvas):
        original = canvas.copy()
        render._panel(canvas, 40, 50, 140, 80)
        np.testing.assert_array_equal(canvas[:50], original[:50])
        np.testing.assert_array_equal(canvas[:, :40], original[:, :40])

    def test_out_of_bounds_is_clamped_not_crashing(self, canvas):
        render._panel(canvas, -50, -50, CANVAS_W + 100, CANVAS_H + 100)
        render._panel(canvas, 500, 500, 600, 600)  # 完全在画布外
        render._panel(canvas, 100, 100, 100, 100)  # 零面积


class TestDrawHud:
    def test_paints_a_panel_from_top_left(self, canvas):
        render.draw_hud(canvas, ["FPS 30.0", "method ear"])
        assert canvas[5:40, 5:60].mean() < 150

    def test_longer_text_gets_wider_panel(self):
        narrow = np.full((CANVAS_H, CANVAS_W, 3), 200, np.uint8)
        wide = narrow.copy()
        render.draw_hud(narrow, ["hi"])
        render.draw_hud(wide, ["a much longer status line here"])
        width_narrow = int((narrow[0].mean(axis=1) < 150).sum())
        width_wide = int((wide[0].mean(axis=1) < 150).sum())
        assert width_wide > width_narrow

    def test_empty_lines_does_nothing(self, canvas):
        original = canvas.copy()
        render.draw_hud(canvas, [])
        np.testing.assert_array_equal(canvas, original)


class TestDrawEye:
    BOX = (60, 80, 200, 120)

    def test_open_and_closed_use_different_colors(self):
        box_pixels = {
            True: self._border_pixels(True),
            False: self._border_pixels(False),
        }
        # 绿色通道应明显强于红色（Open），或反之（Closed）
        assert box_pixels[True][1] > box_pixels[True][2]
        assert box_pixels[False][2] > box_pixels[False][1]

    def _border_pixels(self, is_open: bool) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 200, np.uint8)
        render.draw_eye(canvas, self.BOX, is_open, 0.40 if is_open else 0.05)
        x0, y0, x1, y1 = self.BOX
        # 取上边框内侧一条，避开标签底板
        return canvas[y0 : y0 + 2, x0 + 5 : x1 - 5].reshape(-1, 3).mean(axis=0)

    def test_label_goes_inside_box_when_near_top_edge(self):
        """眼睛框贴顶时，标签不能画到画面外。"""
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 200, np.uint8)
        render.draw_eye(canvas, (60, 0, 200, 30), True, 0.40)
        # 画面外不会出错；第一行像素应当有内容而不是纯背景
        assert canvas[0].mean() < 200

    def test_draws_within_canvas(self, canvas):
        original = canvas.copy()
        render.draw_eye(canvas, self.BOX, True, 0.40)
        assert canvas.shape == original.shape


class TestDrawAlert:
    def test_paints_center_region(self, canvas):
        render.draw_alert(canvas)
        middle = canvas[CANVAS_H // 2 - 20 : CANVAS_H // 2 + 20, CANVAS_W // 2 - 60 : CANVAS_W // 2 + 60]
        assert middle.mean() < 150

    def test_center_of_mass_is_near_horizontal_center(self):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 220, np.uint8)
        render.draw_alert(canvas)
        dark_cols = np.where(canvas[:, :, 0].min(axis=0) < 120)[0]
        assert dark_cols.size > 0
        center = (dark_cols.min() + dark_cols.max()) / 2
        assert abs(center - CANVAS_W / 2) < CANVAS_W * 0.1


class TestWindowVisible:
    """模拟用户点窗口叉按钮：关掉后 ``is_window_visible`` 必须返回 False。"""

    def test_missing_window_is_not_visible(self):
        assert render.is_window_visible("这个窗口名字根本不存在_12345") is False

    def test_imshow_without_named_window_is_visible(self):
        """CLI 里没有先 namedWindow，直接 imshow —— 这个顺序也必须判为可见。"""
        name = "eye_state_test_imshow_only"
        try:
            cv2.imshow(name, np.zeros((40, 40, 3), np.uint8))
            cv2.waitKey(1)
        except cv2.error as exc:  # pragma: no cover - 无 GUI 环境
            pytest.skip(f"当前环境无法创建窗口: {exc}")
        try:
            assert render.is_window_visible(name) is True
        finally:
            cv2.destroyAllWindows()

    def test_closed_window_is_not_visible(self):
        name = "eye_state_test_window"
        try:
            cv2.namedWindow(name)
            cv2.imshow(name, np.zeros((40, 40, 3), np.uint8))
            cv2.waitKey(1)
        except cv2.error as exc:  # pragma: no cover - 无 GUI 环境
            pytest.skip(f"当前环境无法创建窗口: {exc}")

        try:
            assert render.is_window_visible(name) is True
            cv2.destroyWindow(name)  # 等价于用户点叉
            cv2.waitKey(1)
            assert render.is_window_visible(name) is False
        finally:
            cv2.destroyAllWindows()

"""绘制：眼部框、HUD、疲劳告警。都是纯函数，直接改传入的画布。

排版策略：小字用**半透明底板 + 纯色文字**，不用描边。

原因是 OpenCV 的 Hershey 字体在小的 scale 下笔画只有约 1px，
若按常规做法配 ``thickness + 2`` 的描边，描边会比字形本身还宽，
白字被黑边吞掉，屏幕上就是一团糊。加底板则与背景完全解耦，
无论过曝、暗光还是杂乱场景都清晰。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import COLOR_ALERT, COLOR_CLOSED, COLOR_HUD, COLOR_OPEN
from .geometry import Box

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# HUD 排版
_HUD_SCALE = 0.55
_HUD_PAD_X = 8
_HUD_FIRST_BASELINE = 14  # 首行基线，等于「上内边距 + 字高」
_HUD_LINE_H = 22
_HUD_PANEL_EXTRA = 6  # 底板比文字块多出的高度，避免末行贴底
_PANEL_ALPHA = 0.55

# 眼睛标签
_LABEL_SCALE = 0.5
_LABEL_PAD = 4
_LABEL_GAP = 6  # 标签与眼眶的间距

# 告警
_ALERT_SCALE = 1.1
_ALERT_ALPHA = 0.7
_ALERT_PAD = 12


def _panel(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = _PANEL_ALPHA) -> None:
    """在画布上叠一块半透明黑色底板，自动裁到画面内。"""
    h, w = canvas.shape[:2]
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    roi[:] = cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1.0 - alpha, 0.0)


def _text_width(text: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(text, _FONT, scale, thickness)[0][0]


def is_window_visible(name: str) -> bool:
    """查询 OpenCV 窗口是否还开着（用户没点叉关掉）。

    窗口被销毁后再查会抛 ``cv2.error``，统一当作「已关闭」。
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def draw_hud(canvas: np.ndarray, lines: list[str], color: tuple[int, int, int] = COLOR_HUD) -> None:
    """左上角状态栏：一块底板盖住整个文本块，再逐行写字。"""
    if not lines:
        return

    width = max(_text_width(line, _HUD_SCALE) for line in lines)
    _panel(canvas, 0, 0, width + _HUD_PAD_X * 2, _HUD_LINE_H * len(lines) + _HUD_PANEL_EXTRA)

    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (_HUD_PAD_X, _HUD_FIRST_BASELINE + i * _HUD_LINE_H), _FONT, _HUD_SCALE, color, 1, cv2.LINE_AA)


def draw_eye(canvas: np.ndarray, box: Box, is_open: bool, metric: float) -> None:
    """画出单只眼睛的框和状态标签。标签默认在框上方，贴顶时自动移到框内。"""
    x0, y0, x1, y1 = box
    color = COLOR_OPEN if is_open else COLOR_CLOSED
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

    state = "OPEN" if is_open else "CLOSED"
    text = f"{state} {metric:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, _LABEL_SCALE, 1)

    baseline_y = y0 - _LABEL_GAP  # 默认放框上方
    if baseline_y - text_h < 0:  # 顶到画面外就塞进框里
        baseline_y = y0 + text_h + _LABEL_PAD

    _panel(canvas, x0, baseline_y - text_h - _LABEL_PAD, x0 + text_w + _LABEL_PAD * 2, baseline_y + baseline + _LABEL_PAD // 2)
    cv2.putText(canvas, text, (x0 + _LABEL_PAD, baseline_y), _FONT, _LABEL_SCALE, color, 1, cv2.LINE_AA)


def draw_alert(canvas: np.ndarray, text: str = "! DROWSINESS !") -> None:
    """画面正中打告警。按文字实际宽度居中，而不是固定比例。"""
    h, w = canvas.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, _ALERT_SCALE, 2)

    x = max(0, (w - text_w) // 2)
    y = min(h - baseline - _ALERT_PAD, h // 2 + text_h // 2)

    _panel(canvas, x - _ALERT_PAD, y - text_h - _ALERT_PAD, x + text_w + _ALERT_PAD, y + baseline + _ALERT_PAD, alpha=_ALERT_ALPHA)
    cv2.putText(canvas, text, (x, y), _FONT, _ALERT_SCALE, COLOR_ALERT, 2, cv2.LINE_AA)

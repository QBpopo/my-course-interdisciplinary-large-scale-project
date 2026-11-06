"""时序平滑与疲劳告警测试 —— 纯逻辑，不需要模型文件。"""

from __future__ import annotations

import pytest

from eye_state.smoothing import DrowsinessGuard, EyeSmoother


class TestEyeSmoother:
    def test_rejects_invalid_window(self):
        with pytest.raises(ValueError):
            EyeSmoother(window=0)

    def test_window_average(self):
        smoother = EyeSmoother(window=3, thresh=0.24)
        smoother.update("L", 0.30)
        smoother.update("L", 0.30)
        avg, is_open = smoother.update("L", 0.00)
        assert avg == pytest.approx(0.20)
        assert is_open is False

    def test_window_slides_and_drops_old_values(self):
        smoother = EyeSmoother(window=2, thresh=0.24)
        smoother.update("L", 0.00)
        smoother.update("L", 0.00)
        assert smoother.update("L", 0.40)[0] == pytest.approx(0.20)
        # 再送一个 0.40，窗口里只剩两个 0.40
        assert smoother.update("L", 0.40)[0] == pytest.approx(0.40)

    def test_sides_are_independent(self):
        smoother = EyeSmoother(window=2, thresh=0.24)
        smoother.update("L", 0.50)
        assert smoother.update("R", 0.10)[0] == pytest.approx(0.10)
        assert smoother.update("L", 0.50)[0] == pytest.approx(0.50)

    def test_threshold_boundary_is_inclusive(self):
        smoother = EyeSmoother(window=1, thresh=0.24)
        assert smoother.update("L", 0.24)[1] is True
        assert smoother.update("L", 0.2399)[1] is False

    def test_clear_resets_history(self):
        """图片批量模式必须靠这个避免跨图污染。"""
        smoother = EyeSmoother(window=4, thresh=0.24)
        for _ in range(4):
            smoother.update("L", 0.00)
        smoother.clear()
        assert smoother.update("L", 0.90)[0] == pytest.approx(0.90)

    def test_set_threshold_used_by_calibration(self):
        smoother = EyeSmoother(window=1, thresh=0.24)
        assert smoother.update("L", 0.30)[1] is True
        smoother.set_threshold(0.40)
        assert smoother.update("L", 0.30)[1] is False


class TestDrowsinessGuard:
    def test_no_alert_before_threshold(self):
        guard = DrowsinessGuard(seconds=1.5)
        assert guard.update(True, now=100.0) is False
        assert guard.update(True, now=101.0) is False

    def test_alert_after_threshold(self):
        guard = DrowsinessGuard(seconds=1.5)
        guard.update(True, now=100.0)
        assert guard.update(True, now=101.5) is True

    def test_opening_eyes_resets_timer(self):
        guard = DrowsinessGuard(seconds=1.5)
        guard.update(True, now=100.0)
        assert guard.update(False, now=101.0) is False
        # 重新闭眼后计时从头开始
        assert guard.update(True, now=102.0) is False
        assert guard.update(True, now=103.4) is False
        assert guard.update(True, now=103.5) is True

    def test_elapsed_tracks_continuous_closure(self):
        guard = DrowsinessGuard(seconds=1.5)
        assert guard.elapsed == 0.0
        guard.update(True, now=100.0)
        guard.update(True, now=100.8)
        assert guard.elapsed == pytest.approx(0.8, abs=1e-6)
        guard.update(False, now=100.9)
        assert guard.elapsed == 0.0

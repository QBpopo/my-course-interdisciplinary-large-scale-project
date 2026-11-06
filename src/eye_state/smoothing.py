"""时序平滑与疲劳告警。

单帧结果会抖 —— 眨眼只有 100~400ms，逐帧看会得到 ``睁 闭 睁 闭 睁`` 的乱跳，
所以需要一层滑动窗口把状态稳定下来。
"""

from __future__ import annotations

from collections import deque


class EyeSmoother:
    """每只眼睛维护一个滑动窗口，用窗口均值抑制单帧抖动。

    ``thresh`` 可以在运行中被改写（见 :meth:`set_threshold`），
    用于个人化校准。
    """

    def __init__(self, window: int = 9, thresh: float = 0.24) -> None:
        if window < 1:
            raise ValueError("window 必须 >= 1")
        self.window = window
        self.thresh = thresh
        self._buf: dict[str, deque[float]] = {}

    def set_threshold(self, thresh: float) -> None:
        self.thresh = thresh

    def clear(self) -> None:
        """清空窗口。

        处理互相独立的图片时**必须**在每张之间调用，
        否则前一张的结果会污染后一张。
        """
        self._buf.clear()

    def update(self, side: str, value: float) -> tuple[float, bool]:
        """送入本帧度量值，返回 ``(窗口均值, 是否判为睁眼)``。"""
        buf = self._buf.setdefault(side, deque(maxlen=self.window))
        buf.append(float(value))
        avg = sum(buf) / len(buf)
        return avg, avg >= self.thresh


class DrowsinessGuard:
    """双眼持续闭合超过给定秒数就告警。"""

    def __init__(self, seconds: float = 1.5) -> None:
        self.seconds = seconds
        self._since: float | None = None
        self._now: float = 0.0

    @property
    def elapsed(self) -> float:
        """当前已持续闭眼的秒数，未闭眼时为 0。"""
        return 0.0 if self._since is None else self._now - self._since

    def update(self, both_closed: bool, now: float) -> bool:
        """送入本帧状态和当前时间戳（秒），返回是否应当告警。"""
        self._now = now
        if not both_closed:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self.seconds

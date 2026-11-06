"""把检测结果回传给 ESP32：UDP 单包 ``ALARM:1`` / ``ALARM:0`` 控制蜂鸣器和灯带。

为什么不用 HTTP：ESP32 的 WebServer 是单线程的，``/stream`` 会一直占着 ``loop()``，
这时再发 HTTP 请求根本没人处理。UDP 由固件里独立的 FreeRTOS 任务接收，推流不受影响。
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Callable

from .config import DEFAULT_ALARM_REPEAT_SECONDS, DEFAULT_UDP_PORT

ALARM_ON = b"ALARM:1"
ALARM_OFF = b"ALARM:0"


class AlarmSender:
    """状态变化才发 UDP；报警期间按固定间隔重发，防止丢包导致蜂鸣器一直响。"""

    def __init__(
        self,
        ip: str,
        port: int = DEFAULT_UDP_PORT,
        *,
        repeat_seconds: float = DEFAULT_ALARM_REPEAT_SECONDS,
        sock: socket.socket | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ip = ip
        self.port = port
        self.repeat_seconds = repeat_seconds
        self._sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._clock = clock
        self.on = False
        self._last_sent = 0.0

    def set(self, on: bool, *, force: bool = False) -> bool:
        """同步报警状态，返回本次是否真的发了包。

        发送失败时不更新 ``on``，下一帧会自动重试。
        """
        now = self._clock()
        due = force or on != self.on or (on and now - self._last_sent >= self.repeat_seconds)
        if not due:
            return False

        message = ALARM_ON if on else ALARM_OFF
        try:
            self._sock.sendto(message, (self.ip, self.port))
        except OSError as exc:
            print(f"[warn] UDP 发送失败: {exc}", file=sys.stderr)
            return False

        self.on = on
        self._last_sent = now
        return True

    def close(self) -> None:
        """退出前确保蜂鸣器关掉，否则它会被留在响的状态。"""
        self.set(False, force=True)
        self._sock.close()

    def __enter__(self) -> AlarmSender:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

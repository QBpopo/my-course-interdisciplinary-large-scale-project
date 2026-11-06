"""用电脑声卡（喇叭 / 音响）播放报警音，作为 ESP32 蜂鸣器的替代或补充。

提示音是运行时用标准库合成的一段 WAV，不额外依赖音频素材文件，也不引入新依赖。
Windows 走 ``winsound.PlaySound``（异步 + 循环播放，读完立即返回，不阻塞视频循环），
提示音会一直响到报警结束才停；其他平台退化成终端响铃。

注意：``winsound`` 不支持「从内存异步播放」（会抛 RuntimeError），
所以首次播放前把 WAV 落到临时文件，进程退出时删掉。
"""

from __future__ import annotations

import array
import io
import math
import os
import sys
import tempfile
import time
import wave
from collections.abc import Callable

from .config import DEFAULT_ALARM_REPEAT_SECONDS

try:
    import winsound
except ImportError:  # 非 Windows
    winsound = None

TONE_HZ = 880
TONE_SECONDS = 0.3
SAMPLE_RATE = 16000
FADE_SECONDS = 0.02  # 首尾淡入淡出，避免爆音


def tone_wav(freq: float = TONE_HZ, duration: float = TONE_SECONDS) -> bytes:
    """合成一段单声道 16 bit 的正弦提示音，返回完整的 WAV 字节。"""
    count = int(SAMPLE_RATE * duration)
    fade = max(1, int(SAMPLE_RATE * FADE_SECONDS))
    samples = array.array("h")
    for i in range(count):
        envelope = min(1.0, i / fade, (count - i) / fade)
        samples.append(int(32767 * 0.6 * envelope * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return buf.getvalue()


class ToneBackend:
    """真正发声的那一层；单独抽出来是为了测试时不碰声卡。"""

    def __init__(self, wav: bytes | None = None) -> None:
        self._wav = wav or tone_wav()
        self._path: str | None = None

    def _file_path(self) -> str:
        if self._path is None:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="eye_state_alarm_")
            with os.fdopen(fd, "wb") as handle:
                handle.write(self._wav)
            self._path = path
        return self._path

    def play(self) -> None:
        if winsound is None:
            sys.stdout.write("\a")
            sys.stdout.flush()
            return
        winsound.PlaySound(self._file_path(), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

    def stop(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)

    def close(self) -> None:
        if self._path is None:
            return
        try:
            os.unlink(self._path)
        except OSError:
            pass
        self._path = None


class PcAlarm:
    """和 :class:`~eye_state.remote_alarm.AlarmSender` 同样的接口，换成电脑发声。"""

    def __init__(
        self,
        *,
        repeat_seconds: float = DEFAULT_ALARM_REPEAT_SECONDS,
        backend: ToneBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repeat_seconds = repeat_seconds
        self._backend = backend or ToneBackend()
        self._clock = clock
        self.on = False
        self._last_played = 0.0

    def set(self, on: bool, *, force: bool = False) -> bool:
        """同步报警状态，返回本次是否真的播放 / 停止了一次。"""
        now = self._clock()
        due = force or on != self.on or (on and now - self._last_played >= self.repeat_seconds)
        if not due:
            return False

        if on:
            self._backend.play()
        else:
            self._backend.stop()
        self.on = on
        self._last_played = now
        return True

    def close(self) -> None:
        self.set(False, force=True)
        closer = getattr(self._backend, "close", None)  # 清理临时 WAV
        if closer is not None:
            closer()

    def __enter__(self) -> PcAlarm:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

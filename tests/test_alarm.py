"""报警通道测试：UDP 只在状态变化 / 到期时发包，电脑喇叭同理。"""

from __future__ import annotations

import io
import wave

from eye_state.pc_alarm import PcAlarm, tone_wav
from eye_state.remote_alarm import ALARM_OFF, ALARM_ON, AlarmSender


class _FakeSock:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False
        self.fail = False

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        if self.fail:
            raise OSError("network is down")
        self.sent.append((data, addr))

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _make() -> tuple[AlarmSender, _FakeSock, _Clock]:
    sock = _FakeSock()
    clock = _Clock()
    return AlarmSender("192.168.1.50", 4210, sock=sock, clock=clock), sock, clock


class TestStateTransitions:
    def test_idle_does_not_send(self):
        sender, sock, _ = _make()
        assert sender.set(False) is False
        assert sock.sent == []

    def test_turning_alarm_on_sends_once(self):
        sender, sock, _ = _make()
        assert sender.set(True) is True
        assert sock.sent == [(ALARM_ON, ("192.168.1.50", 4210))]
        assert sender.on is True

    def test_repeated_on_within_interval_is_quiet(self):
        sender, sock, clock = _make()
        sender.set(True)
        clock.now += 0.4
        assert sender.set(True) is False
        assert len(sock.sent) == 1

    def test_on_is_resent_after_interval(self):
        """报警期间每秒重发，防止丢包后蜂鸣器停在错误状态。"""
        sender, sock, clock = _make()
        sender.set(True)
        clock.now += 1.0
        assert sender.set(True) is True
        assert len(sock.sent) == 2
        assert sock.sent[-1][0] == ALARM_ON

    def test_turning_alarm_off_is_immediate(self):
        sender, sock, clock = _make()
        sender.set(True)
        clock.now += 0.1
        assert sender.set(False) is True
        assert sock.sent[-1][0] == ALARM_OFF
        assert sender.on is False

    def test_off_stays_quiet_afterwards(self):
        sender, sock, clock = _make()
        sender.set(True)
        sender.set(False)
        for _ in range(3):
            clock.now += 2.0
            assert sender.set(False) is False
        assert len(sock.sent) == 2


class TestFailureAndClose:
    def test_send_failure_keeps_state_for_retry(self):
        sender, sock, _ = _make()
        sock.fail = True
        assert sender.set(True) is False
        assert sender.on is False
        sock.fail = False
        assert sender.set(True) is True
        assert sock.sent[-1][0] == ALARM_ON

    def test_force_resends_same_state(self):
        sender, sock, _ = _make()
        sender.set(True)
        assert sender.set(True, force=True) is True
        assert len(sock.sent) == 2

    def test_close_sends_off_and_releases_socket(self):
        sender, sock, _ = _make()
        sender.set(True, force=True)  # 初始 off -> on，会发 ALARM:1
        sender.close()
        assert sock.sent[-1][0] == ALARM_OFF
        assert sock.closed is True


class _FakeBackend:
    """不碰声卡，只记录播放 / 停止次数。"""

    def __init__(self) -> None:
        self.plays = 0
        self.stops = 0

    def play(self) -> None:
        self.plays += 1

    def stop(self) -> None:
        self.stops += 1


class TestPcAlarm:
    def test_idle_does_not_play(self):
        backend = _FakeBackend()
        sender = PcAlarm(backend=backend, clock=_Clock())
        assert sender.set(False) is False
        assert backend.plays == 0

    def test_turning_on_plays_once(self):
        backend = _FakeBackend()
        sender = PcAlarm(backend=backend, clock=_Clock())
        assert sender.set(True) is True
        assert backend.plays == 1

    def test_replays_on_interval(self):
        backend = _FakeBackend()
        clock = _Clock()
        sender = PcAlarm(backend=backend, clock=clock)
        sender.set(True)
        clock.now += 0.4
        assert sender.set(True) is False
        clock.now += 0.8
        assert sender.set(True) is True
        assert backend.plays == 2

    def test_turning_off_stops_immediately(self):
        backend = _FakeBackend()
        clock = _Clock()
        sender = PcAlarm(backend=backend, clock=clock)
        sender.set(True)
        clock.now += 0.1
        assert sender.set(False) is True
        assert backend.stops == 1

    def test_close_stops_playback(self):
        backend = _FakeBackend()
        sender = PcAlarm(backend=backend, clock=_Clock())
        sender.set(True)
        sender.close()
        assert backend.stops == 1


class TestToneWav:
    def test_generated_tone_is_playable_wav(self):
        data = tone_wav()
        with wave.open(io.BytesIO(data), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            frames = handle.getnframes()
        assert frames > 0
        assert len(data) > frames * 2 - 64  # 头部开销很小

"""MJPEG 切帧测试：用合成的字节流，不联网。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from eye_state.stream import MJPEGReader

BOUNDARY = "123456"


def _jpeg(color: tuple[int, int, int], size: int = 16) -> bytes:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = color
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _part(body: bytes, *, with_length: bool = True) -> bytes:
    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n".encode()
    if with_length:
        head += f"Content-Length: {len(body)}\r\n".encode()
    return head + b"\r\n" + body


def _multipart(frames: list[bytes]) -> bytes:
    return b"".join(_part(f) for f in frames)


class _FakeResponse:
    """按固定块大小吐出字节，模拟 TCP 分片到达。"""

    def __init__(self, payload: bytes, chunk: int = 64) -> None:
        self.payload = payload
        self.chunk = chunk
        self.pos = 0
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        if self.pos >= len(self.payload):
            return b""
        step = self.chunk if n is None or n < 0 else min(n, self.chunk)
        start, self.pos = self.pos, self.pos + step
        return self.payload[start:self.pos]

    def close(self) -> None:
        self.closed = True


def _reader(payload: bytes, chunk: int = 64) -> tuple[MJPEGReader, _FakeResponse, list[str]]:
    response = _FakeResponse(payload, chunk=chunk)
    requested: list[str] = []

    def opener(url: str, timeout: float | None = None) -> _FakeResponse:
        requested.append(url)
        return response

    reader = MJPEGReader("http://example.test/stream", opener=opener)
    return reader, response, requested


class TestReadFrame:
    def test_returns_decoded_frames_in_order(self):
        reader, _, _ = _reader(_multipart([_jpeg((0, 0, 255)), _jpeg((0, 255, 0))]))
        first = reader.read_frame()
        second = reader.read_frame()
        assert first.shape == (16, 16, 3)
        assert first[8, 8].tolist() == pytest.approx([0, 0, 255], abs=8)
        assert second[8, 8].tolist() == pytest.approx([0, 255, 0], abs=8)

    def test_handles_two_frames_arriving_in_one_read(self):
        """整段一次性到达时也必须按 Content-Length 切开，而不是当成一帧。"""
        reader, _, _ = _reader(_multipart([_jpeg((255, 0, 0)), _jpeg((0, 0, 255))]), chunk=1 << 20)
        assert len(reader.read_frame()) == 16
        assert len(reader.read_frame()) == 16

    def test_header_without_content_length_is_skipped(self):
        payload = _part(b"not a frame", with_length=False) + _part(_jpeg((0, 255, 0)))
        reader, _, _ = _reader(payload)
        assert reader.read_frame()[8, 8].tolist() == pytest.approx([0, 255, 0], abs=8)

    def test_broken_jpeg_is_skipped(self):
        payload = _part(b"xxxx") + _part(_jpeg((0, 0, 255)))
        reader, _, _ = _reader(payload)
        assert reader.read_frame()[8, 8].tolist() == pytest.approx([0, 0, 255], abs=8)

    def test_stream_end_raises_connection_error(self):
        reader, _, _ = _reader(_multipart([_jpeg((0, 0, 255))]))
        reader.read_frame()
        with pytest.raises(ConnectionError):
            reader.read_frame()


class TestLifecycle:
    def test_url_is_passed_to_opener(self):
        _, _, requested = _reader(b"")
        assert requested == ["http://example.test/stream"]

    def test_close_releases_response(self):
        reader, response, _ = _reader(b"")
        reader.close()
        assert response.closed is True

    def test_connect_resets_buffer_and_stream(self):
        """断流重连后应该从新响应的第一帧开始，不带旧缓冲。"""
        reader, response, _ = _reader(_part(_jpeg((0, 0, 255))) + b"\r\n\r\n")
        reader.read_frame()
        reader.connect()
        assert reader._buf == b""
        assert response.closed is True

    def test_closed_reader_raises_connection_error(self):
        """重连失败时 _resp 为 None，不能抛 AttributeError 把主循环打断。"""
        reader, _, _ = _reader(_multipart([_jpeg((0, 0, 255))]))
        reader.close()
        with pytest.raises(ConnectionError):
            reader.read_frame()

    def test_context_manager_closes(self):
        _, response, _ = _reader(b"")
        with MJPEGReader("http://example.test/stream", opener=lambda url, timeout=None: response):
            pass
        assert response.closed is True

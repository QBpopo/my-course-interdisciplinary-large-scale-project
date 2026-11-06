"""读取 ESP32-CAM 的 MJPEG over HTTP 流。

``cv2.VideoCapture(url)`` 也能读 MJPEG，但它依赖 FFmpeg 后端，
面对 ESP32 这种「无总长度、连接一直不关」的流，在 Windows 上经常卡住或解出坏帧。
这里按 ``Content-Length`` 手动切帧，只依赖标准库负责传输、OpenCV 负责解码，行为可预期。

流地址形如 ``http://192.168.1.100/stream``，一个响应体里连续放着多个 JPEG：

    --123456\r\n
    Content-Type: image/jpeg\r\n
    Content-Length: 5123\r\n
    \r\n
    <5123 字节的 JPEG>
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable

import cv2
import numpy as np

DEFAULT_TIMEOUT = 10.0  # 单次 socket 读超时（秒）
READ_SIZE = 8192
_HEADER_SEP = b"\r\n\r\n"


class MJPEGReader:
    """从 MJPEG 流里逐帧取出已解码的 BGR 图像。

    :param url: 流地址
    :param timeout: 单次读超时；ESP32 卡住时会抛 ``socket.timeout``
    :param opener: 注入用，默认 :func:`urllib.request.urlopen`；测试可传假对象
    """

    def __init__(
        self,
        url: str,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        opener: Callable[..., object] | None = None,
    ) -> None:
        self.url = url
        self._timeout = timeout
        self._opener = opener or urllib.request.urlopen
        self._resp = None
        self._buf = b""
        self.connect()

    def connect(self) -> None:
        """（重新）建立连接。断流后调用即可复用同一个对象。"""
        self.close()
        self._resp = self._opener(self.url, timeout=self._timeout)
        self._buf = b""

    def _fill(self) -> None:
        chunk = self._resp.read(READ_SIZE)
        if not chunk:
            raise ConnectionError("MJPEG 流已结束")
        self._buf += chunk

    def read_frame(self) -> np.ndarray:
        """阻塞读取下一帧，返回 BGR 图。

        帧头里没有 ``Content-Length``（或多帧挤在一次读里）都能正确处理；
        解码失败（坏帧）会跳过继续读下一帧。
        """
        while True:
            if self._resp is None:
                # connect() 失败后 _resp 会是 None，让调用方按「断流」统一处理重连
                raise ConnectionError("尚未连接")
            header_end = self._buf.find(_HEADER_SEP)
            if header_end < 0:
                self._fill()
                continue

            # 帧头里还含有 "--boundary" 那一行，我们只关心 Content-Length
            header = self._buf[:header_end]
            self._buf = self._buf[header_end + len(_HEADER_SEP):]

            length = None
            for line in header.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1].strip())
                    break
            if length is None:
                continue

            while len(self._buf) < length:
                self._fill()

            jpg = self._buf[:length]
            self._buf = self._buf[length:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None

    def __enter__(self) -> MJPEGReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

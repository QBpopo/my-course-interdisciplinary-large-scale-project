"""CLI 参数校验与错误提示的测试。

这些提示是用户卡住时唯一的线索，而且过去曾以原始 traceback 的形式抛出来，
所以值得逐个锁住。
"""

from __future__ import annotations

import pytest

from eye_state import cli


@pytest.fixture(scope="module")
def real_landmarker() -> str:
    """校验要求 landmarker 存在，否则会被它先拦下。"""
    path = cli.DEFAULT_LANDMARKER_PATH
    if not path.exists():
        pytest.skip(f"需要模型文件 {path}，先跑 python scripts/download_models.py")
    return str(path)


def validate(**overrides) -> str | None:
    """用默认参数跑一遍校验，按需覆盖字段。"""
    args = cli.build_parser().parse_args([])
    args.thresh = cli.DEFAULT_THRESHOLDS[args.method]
    for key, value in overrides.items():
        setattr(args, key, value)
    return cli._validate(args)


class TestDefaultArgs:
    def test_defaults_are_ear_camera(self):
        args = cli.build_parser().parse_args([])
        assert args.source == "camera"
        assert args.method == "ear"
        assert args.window == 9

    def test_default_run_passes_validation(self, real_landmarker):
        assert validate(landmarker=real_landmarker) is None


class TestLandmarkerCheck:
    def test_missing_landmarker_points_to_download_script(self):
        message = validate(landmarker="no_such_model.task")
        assert message is not None
        assert "download_models.py" in message


class TestSourcePathCheck:
    @pytest.mark.parametrize("source", ["image", "dir", "video"])
    def test_source_requires_path(self, real_landmarker, source):
        message = validate(landmarker=real_landmarker, source=source, path=None)
        assert message is not None
        assert "-p" in message

    def test_camera_does_not_require_path(self, real_landmarker):
        assert validate(landmarker=real_landmarker, source="camera", path=None) is None


class TestStreamSource:
    def test_stream_requires_url(self, real_landmarker):
        message = validate(landmarker=real_landmarker, source="stream", path=None)
        assert message is not None
        assert "-p" in message

    def test_stream_rejects_non_http_url(self, real_landmarker):
        message = validate(landmarker=real_landmarker, source="stream", path="192.168.1.100")
        assert message is not None
        assert "http" in message

    def test_stream_accepts_esp32_url(self, real_landmarker):
        assert validate(landmarker=real_landmarker, source="stream", path="http://192.168.1.100/stream", esp32_ip="192.168.1.100") is None

    def test_pc_alarm_works_without_esp32_ip(self, real_landmarker):
        assert validate(landmarker=real_landmarker, source="stream", path="http://192.168.1.100/stream", alarm="pc", esp32_ip=None) is None

    def test_both_alarm_requires_esp32_ip(self, real_landmarker):
        message = validate(landmarker=real_landmarker, source="stream", path="http://192.168.1.100/stream", alarm="both", esp32_ip=None)
        assert message is not None
        assert "--esp32-ip" in message

    def test_udp_defaults_match_firmware(self):
        args = cli.build_parser().parse_args([])
        assert args.udp_port == 4210
        assert args.esp32_ip is None
        assert args.alarm == "esp32"

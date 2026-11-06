"""路径与常量配置。

模型文件默认放在项目根目录的 ``models/`` 下，可用环境变量覆盖：

- ``EYE_STATE_HOME``        项目根目录
- ``EYE_STATE_MODELS_DIR``  模型目录
"""

from __future__ import annotations

import os
from pathlib import Path

# src/eye_state/config.py -> 上溯三级 = 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("EYE_STATE_HOME", _PROJECT_ROOT))
MODELS_DIR = Path(os.environ.get("EYE_STATE_MODELS_DIR", PROJECT_ROOT / "models"))

DEFAULT_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"

IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# 绘制颜色（BGR）
COLOR_OPEN = (80, 220, 80)
COLOR_CLOSED = (70, 70, 240)
COLOR_HUD = (240, 240, 240)
COLOR_ALERT = (60, 60, 255)

# 各判据的推荐阈值，在真实标注样本上扫描得到（见 README 5.3）
DEFAULT_THRESHOLDS = {"ear": 0.24, "ratio": 0.22}

# 窗口平滑默认帧数：眨眼约 100~400ms，窗口太短会抖，太长会迟钝
DEFAULT_SMOOTH_WINDOW = 9
# 双眼持续闭合超过该秒数判定为疲劳
DEFAULT_ALERT_SECONDS = 1.5
# 眼部裁剪框外扩比例：不外扩的话闭眼时睫毛会被切在边界上
DEFAULT_BOX_PAD = 0.35

# ESP32 报警（蜂鸣器 + 灯带）的 UDP 端口，必须与固件里的 ALARM_UDP_PORT 一致
DEFAULT_UDP_PORT = 4210
# 报警期间每多少秒重发一次 UDP，防止丢包后 ESP32 状态与检测结果不一致
DEFAULT_ALARM_REPEAT_SECONDS = 1.0
# stream 模式下按空格键手动触发报警音的时长（秒）
MANUAL_ALARM_SECONDS = 2.0

# ESP32-CAM 推流地址的常见写法，仅用于提示信息
STREAM_URL_HINT = "http://192.168.x.x/stream"

"""命令行入口：摄像头 / 视频 / 单图 / 目录 / ESP32 推流 五种输入源。"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from .config import (
    DEFAULT_ALERT_SECONDS,
    DEFAULT_BOX_PAD,
    DEFAULT_LANDMARKER_PATH,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_THRESHOLDS,
    DEFAULT_UDP_PORT,
    IMG_EXTS,
    MANUAL_ALARM_SECONDS,
    STREAM_URL_HINT,
)
from .pc_alarm import PcAlarm
from .pipeline import METHODS, EyeStatePipeline
from .remote_alarm import AlarmSender
from .render import draw_hud, is_window_visible
from .stream import MJPEGReader

WINDOW_NAME = "eye state"
QUIT_KEYS = (ord("q"), 27)  # q / ESC
MANUAL_ALARM_KEY = ord(" ")  # 空格：手动触发报警音
STREAM_READ_TIMEOUT = 3.0  # 取流线程的单次 socket 超时：断线后能尽快重连，不让主线程干等


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eye-state",
        description="睁眼 / 闭眼检测：MediaPipe 关键点定位 + EAR 几何判据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--source", default="camera", choices=["camera", "video", "image", "dir", "stream"], help="输入源类型")
    parser.add_argument("-p", "--path", default=None, help="图片/目录/视频路径；stream 模式填 ESP32 流地址；摄像头时可填设备号，默认 0")
    parser.add_argument("--method", default="ear", choices=list(METHODS), help="ear=眼睑宽高比(推荐) ratio=眼眶高宽比")
    parser.add_argument("--thresh", type=float, default=None, help="判为睁眼的阈值，默认按 method 取 " f"{DEFAULT_THRESHOLDS}")
    parser.add_argument("--window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="时序平滑窗口帧数")
    parser.add_argument("--pad", type=float, default=DEFAULT_BOX_PAD, help="眼部裁剪框外扩比例")
    parser.add_argument("--alert-after", type=float, default=DEFAULT_ALERT_SECONDS, help="连续闭眼多少秒告警")
    parser.add_argument("--calibrate", type=float, default=0.0, metavar="SECONDS", help="开算后 N 秒内做个人化校准（取度量峰值*0.6 作阈值），0=关闭")

    model_group = parser.add_argument_group("模型路径")
    model_group.add_argument("--landmarker", default=str(DEFAULT_LANDMARKER_PATH), help="MediaPipe Face Landmarker 模型")

    output_group = parser.add_argument_group("输出")
    output_group.add_argument("--out", default=None, help="image/dir 模式的输出目录")
    output_group.add_argument("--save", default=None, help="video 模式的输出视频路径")
    output_group.add_argument("--no-display", action="store_true", help="不弹窗口，只处理")
    output_group.add_argument("--show-roi", action="store_true", help="额外弹窗显示眼部裁剪")
    output_group.add_argument("--no-hud", action="store_true", help="不画左上角的 FPS / 阈值 / 状态栏")
    output_group.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧后退出，0=不限（摄像头下方便自动化）")

    alarm_group = parser.add_argument_group("远程报警（仅 stream 模式）")
    alarm_group.add_argument(
        "--alarm",
        default="esp32",
        choices=["esp32", "pc", "both", "none"],
        help="报警输出：esp32=ESP32 蜂鸣器 + 灯带 pc=电脑喇叭 both=两者都响 none=不驱动",
    )
    alarm_group.add_argument("--esp32-ip", default=None, help="ESP32 的 IP；--alarm 为 esp32 / both 时必填")
    alarm_group.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="与固件 ALARM_UDP_PORT 保持一致")
    return parser


def build_pipeline(args: argparse.Namespace, *, video_mode: bool) -> EyeStatePipeline:
    return EyeStatePipeline(
        landmarker_path=Path(args.landmarker),
        method=args.method,
        video_mode=video_mode,
        pad=args.pad,
        window=args.window,
        thresh=args.thresh,
        alert_seconds=args.alert_after,
        calibrate_seconds=args.calibrate,
        show_roi=args.show_roi,
        show_hud=not args.no_hud,
    )


def _open_capture(source: int | str) -> cv2.VideoCapture:
    """打开摄像头或视频文件；Windows 上优先用 DirectShow。"""
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(source)


def _run_stream(args: argparse.Namespace, cap: cv2.VideoCapture, frame_ms: Callable[[int], int], writer: cv2.VideoWriter | None = None) -> int:
    """摄像头 / 视频共用的主循环。"""
    pipeline = build_pipeline(args, video_mode=True)
    processed = 0
    shown_once = False  # 首帧窗口可能还没映射完，先不查可见性
    try:
        while args.max_frames <= 0 or processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                if processed == 0:
                    print("[error] 读帧失败", file=sys.stderr)
                    return 1
                break
            canvas, _ = pipeline.process(frame, frame_ms(processed))
            processed += 1
            if writer is not None:
                writer.write(canvas)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                # 点窗口的叉也得能退出：waitKey 不会因为关窗而返回，
                # 而且窗口销毁后 imshow 会把它重新创建，所以必须主动查可见性
                if key in QUIT_KEYS:
                    break
                if shown_once and not is_window_visible(WINDOW_NAME):
                    break
                shown_once = True
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        cv2.destroyAllWindows()
    print(f"处理了 {processed} 帧")
    return 0


def run_camera(args: argparse.Namespace) -> int:
    index = int(args.path) if args.path else 0
    cap = _open_capture(index)
    if not cap.isOpened():
        print(f"[error] 打不开摄像头 index={index}", file=sys.stderr)
        return 1
    if not args.no_display:
        print("摄像头已启动。按 q / ESC，或直接点窗口的叉按钮退出。")
    try:
        return _run_stream(args, cap, lambda n: int(time.perf_counter() * 1000) % (1 << 31))
    finally:
        cap.release()


def run_video(args: argparse.Namespace) -> int:
    cap = _open_capture(str(args.path))
    if not cap.isOpened():
        print(f"[error] 打不开视频 {args.path}", file=sys.stderr)
        return 1

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            str(args.save),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    try:
        return _run_stream(args, cap, lambda n: int(n * 1000 / source_fps), writer)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _next_timestamp(last: int) -> int:
    """给 ``detect_for_video`` 用的毫秒时间戳，必须严格递增。

    ESP32 只有 5 FPS 左右，墙上时钟的毫秒精度够用；万一同一毫秒取到两次，就手动 +1。
    """
    now_ms = int(time.perf_counter() * 1000) % (1 << 31)
    return now_ms if now_ms > last else last + 1


def _build_alarms(args: argparse.Namespace) -> list[AlarmSender | PcAlarm]:
    """按 ``--alarm`` 组装报警通道。"""
    alarms: list[AlarmSender | PcAlarm] = []
    if args.alarm in ("esp32", "both"):
        alarms.append(AlarmSender(args.esp32_ip, args.udp_port))
    if args.alarm in ("pc", "both"):
        alarms.append(PcAlarm())
    return alarms


def _describe_alarm(alarm: AlarmSender | PcAlarm) -> str:
    if isinstance(alarm, AlarmSender):
        return f"ESP32 蜂鸣器 + 灯带 {alarm.ip}:{alarm.port}"
    return "电脑喇叭"


def _host_of(url: str) -> str:
    """给报错提示用，从流地址里抠出主机名。"""
    return urlsplit(url).hostname or url


def _placeholder_canvas(url: str) -> np.ndarray:
    """还没读到帧时显示的占位画面 —— 窗口必须存在，空格键才能随时可用。"""
    canvas = np.zeros((480, 640, 3), np.uint8)
    draw_hud(canvas, ["waiting for stream...", url, "space = test alarm"])
    return canvas


def _stream_worker(url: str, shared: dict[str, object], stop: threading.Event) -> None:
    """后台线程：只管取流和重连，把最新一帧丢进 ``shared``。

    主线程因此不会被阻塞读（正常 200ms 一帧、断线时最长 3 秒）卡住，
    空格键和窗口操作始终能及时响应。
    """
    reader: MJPEGReader | None = None
    while not stop.is_set():
        if reader is None:
            try:
                reader = MJPEGReader(url, timeout=STREAM_READ_TIMEOUT)
            except OSError as exc:
                shared["notice"] = ("connect_failed", exc)
                stop.wait(1.0)
                continue
            shared["connected"] = True
            shared["notice"] = ("connected", None)

        try:
            frame = reader.read_frame()
        except (ConnectionError, OSError) as exc:
            reader.close()
            reader = None
            shared["connected"] = False
            shared["notice"] = ("disconnected", exc)
            stop.wait(1.0)  # 重连前缓一下，别把 CPU 打满
            continue

        shared["frame"] = frame
        shared["seq"] = int(shared["seq"]) + 1  # type: ignore[arg-type]

    if reader is not None:
        reader.close()


def run_stream(args: argparse.Namespace) -> int:
    """读 ESP32 的 MJPEG 流做检测，并按 ``--alarm`` 驱动蜂鸣器 / 电脑喇叭。

    取流放在后台线程：流没连上、中途断开、ESP32 没上电时都不退出，
    窗口照常响应空格与退出键，后台每秒重连一次。
    """
    try:
        alarms = _build_alarms(args)
    except OSError as exc:
        print(f"[error] 创建报警通道失败: {exc}", file=sys.stderr)
        return 2
    pipeline = build_pipeline(args, video_mode=True)

    canvas = _placeholder_canvas(args.path)
    result = None  # 掉线后置空，避免沿用上一帧的告警状态
    shared: dict[str, object] = {"frame": None, "seq": 0, "connected": False, "notice": None}
    stop = threading.Event()
    worker = threading.Thread(target=_stream_worker, args=(args.path, shared, stop), daemon=True)

    processed = 0
    last_seq = 0
    last_ts = 0
    shown_once = False  # 首帧窗口可能还没映射完，先不查可见性
    manual_until = 0.0  # 空格键手动触发的截止时刻
    connect_warned = False

    print(f"正在读取 {args.path}")
    for alarm in alarms:
        print(f"报警输出 -> {_describe_alarm(alarm)}")
    if not args.no_display:
        print(f"按空格键手动触发 {MANUAL_ALARM_SECONDS:g} 秒报警，q / ESC 退出")
    worker.start()

    try:
        while args.max_frames <= 0 or processed < args.max_frames:
            got_frame = False

            notice = shared["notice"]
            if notice is not None:
                shared["notice"] = None
                kind, exc = notice  # type: ignore[misc]
                if kind == "connected":
                    print(f"已连接 {args.path}")
                    connect_warned = False
                elif not connect_warned:
                    print(f"[warn] {kind} {args.path}: {exc}", file=sys.stderr)
                    print("       排查：ESP32 是否已连上同一个热点（串口看 static IP 行）", file=sys.stderr)
                    print(f"             ping {_host_of(args.path)} 是否通、浏览器能否打开该地址", file=sys.stderr)
                    print("       继续每秒重试；按空格仍可测试报警音", file=sys.stderr)
                    connect_warned = True
                else:
                    print(f"[warn] {kind}，继续重连", file=sys.stderr)

            if not shared["connected"]:
                result = None

            frame = shared["frame"]
            if frame is not None and int(shared["seq"]) != last_seq:
                last_seq = int(shared["seq"])
                shared["frame"] = None  # 只处理最新一帧，旧的直接丢
                got_frame = True
                last_ts = _next_timestamp(last_ts)
                canvas, result = pipeline.process(frame, last_ts)
                processed += 1

            alarm_on = bool(result is not None and result.alert) or time.perf_counter() < manual_until
            for alarm in alarms:
                alarm.set(alarm_on)

            if args.no_display:
                if not got_frame:
                    time.sleep(0.01)
                continue

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == MANUAL_ALARM_KEY:
                manual_until = time.perf_counter() + MANUAL_ALARM_SECONDS
                print(f"手动触发报警 {MANUAL_ALARM_SECONDS:g} 秒")
            if key in QUIT_KEYS:
                break
            if shown_once and not is_window_visible(WINDOW_NAME):
                break
            shown_once = True

            if not got_frame:
                time.sleep(0.02)  # 没帧时别空转
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        pipeline.close()
        for alarm in alarms:
            alarm.close()  # 退出时确保蜂鸣器关掉
        cv2.destroyAllWindows()

    print(f"处理了 {processed} 帧")
    return 0


def _collect_images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in IMG_EXTS)
    return [target]


def run_images(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if not target.exists():
        print(f"[error] 路径不存在: {target}", file=sys.stderr)
        return 1

    paths = _collect_images(target)
    if not paths:
        print(f"[error] {target} 下没有图片", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args, video_mode=False)
    detected = 0
    processed = 0
    try:
        for path in paths:
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            image = cv2.imread(str(path))
            if image is None:
                print(f"[skip] 读不了 {path.name}")
                continue

            # 每张图互相独立，必须清掉滑动窗口，否则会被上一张的结果污染
            pipeline.smoother.clear()
            canvas, result = pipeline.process(image)
            processed += 1
            detected += int(bool(result.eyes))
            print(f"{path.name:40s} {result.summary()}")

            cv2.imwrite(str(out_dir / f"{path.stem}_out{path.suffix}"), canvas)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(0) & 0xFF
                if key in QUIT_KEYS or not is_window_visible(WINDOW_NAME):
                    break
    finally:
        pipeline.close()
        cv2.destroyAllWindows()

    print(f"\n共 {processed} 张，检测到眼睛 {detected} 张；标注图已存到 {out_dir}")
    return 0


RUNNERS = {
    "camera": run_camera,
    "video": run_video,
    "image": run_images,
    "dir": run_images,
    "stream": run_stream,
}


def _validate(args: argparse.Namespace) -> str | None:
    """返回错误信息；None 表示通过。

    可预期的用户错误尽量在这里全部拦下，给干净的 [error] 提示 ——
    否则会在深处抛原始 traceback，对使用者很不友好。
    """
    if not Path(args.landmarker).exists():
        return f"找不到 {args.landmarker}\n      先跑：python scripts/download_models.py"

    if args.source in ("image", "dir", "video") and not args.path:
        return f"{args.source} 模式必须用 -p 指定路径"

    if args.source == "stream":
        if not args.path:
            return f"stream 模式必须用 -p 指定 ESP32 的流地址，例如 -p {STREAM_URL_HINT}"
        if not args.path.lower().startswith(("http://", "https://")):
            return f"stream 模式只接受 http/https 地址，收到 {args.path!r}"
        if args.alarm in ("esp32", "both") and not args.esp32_ip:
            return "--alarm 用了 ESP32 蜂鸣器，必须同时给 --esp32-ip；只想用电脑喇叭就加 --alarm pc"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    error = _validate(args)
    if error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    if args.thresh is None:
        args.thresh = DEFAULT_THRESHOLDS[args.method]

    if args.out is None:
        if args.path and args.source == "dir":
            parent = Path(args.path)
            args.out = str(parent.parent / f"{parent.name}_out")
        else:
            args.out = "eyes_out"

    return RUNNERS[args.source](args)

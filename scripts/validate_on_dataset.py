#!/usr/bin/env python
"""在 ``MichalMlodawski/closed-open-eyes`` 数据集上验证整条流水线。

会从 HuggingFace 拉带标注的真实样本（图片缓存到 ``tmp/tests/``），
跑完后打印准确率、混淆情况和阈值扫描表。

依赖：``pip install pillow huggingface-hub``

用法::

    python scripts/validate_on_dataset.py --per-label 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from eye_state import EyeStatePipeline  # noqa: E402
from eye_state.config import DEFAULT_LANDMARKER_PATH, PROJECT_ROOT  # noqa: E402

DATASET = "MichalMlodawski/closed-open-eyes"
ROWS_API = "https://datasets-server.huggingface.co/rows" f"?dataset={DATASET}&config=default&split=train&offset={{off}}&length={{n}}"

CACHE = PROJECT_ROOT / "tmp/tests"

# 数据集的标签是分块排列的（不是全局有序），下面是已知含有对应标签的区间起点
SCAN_OFFSETS = {
    "closed": (0, 40000, 50000, 60000),
    "open": (16000, 24000, 32000, 104000, 112000, 120000),
}

THRESH_SWEEP = (0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)


def fetch_rows(offset: int, length: int, retries: int = 7) -> list[dict]:
    """带指数退避，绕开 datasets-server 的 429 限流。"""
    url = ROWS_API.format(off=offset, n=length)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())["rows"]
        except Exception:  # noqa: BLE001 - 网络问题统一重试
            if attempt == retries - 1:
                raise
            time.sleep(min(2.0**attempt, 30.0))
    return []


def fetch_file(url: str, dest: Path, retries: int = 5) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(min(2.0**attempt, 30.0))


def collect(per_label: int) -> list[tuple[str, Path]]:
    """按标签各收集 per_label 张，图片缓存到 tmp/tests/。"""
    CACHE.mkdir(exist_ok=True)
    buckets: dict[str, list[Path]] = {"closed": [], "open": []}

    for label, offsets in SCAN_OFFSETS.items():
        for offset in offsets:
            if len(buckets[label]) >= per_label:
                break
            need = min(100, per_label - len(buckets[label]))
            for row in fetch_rows(offset, need):
                if len(buckets[label]) >= per_label:
                    break
                cell = row["row"]
                got = "closed" if "closed" in str(cell["Label"]).lower() else "open"
                if got != label:
                    continue
                info = cell["Image_data"]
                dest = CACHE / f"{label}_{cell['Image_id']}_{info['filename'][:12]}.jpg"
                fetch_file(info["file"]["src"], dest)
                buckets[label].append(dest)
            print(f"  已收集 closed={len(buckets['closed'])} open={len(buckets['open'])}")

    return [(label, path) for label, paths in buckets.items() for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-label", type=int, default=80, help="每个类别取多少张")
    parser.add_argument("--method", default="ear", choices=["ear", "ratio"])
    parser.add_argument("--thresh", type=float, default=0.24)
    parser.add_argument("--pad", type=float, default=0.35)
    parser.add_argument("--no-cache", action="store_true", help="忽略本地缓存重新收集")
    args = parser.parse_args()

    if args.no_cache and CACHE.exists():
        for path in CACHE.glob("*.jpg"):
            path.unlink()

    print(f"从 {DATASET} 收集样本 ...")
    samples = collect(args.per_label)
    if not samples:
        print("[error] 没收集到样本", file=sys.stderr)
        return 1
    closed_n = sum(label == "closed" for label, _ in samples)
    print(f"共 {len(samples)} 张（closed={closed_n}, open={len(samples) - closed_n}）\n")

    pipeline = EyeStatePipeline(
        landmarker_path=DEFAULT_LANDMARKER_PATH,
        method=args.method,
        video_mode=False,
        pad=args.pad,
        thresh=args.thresh,
    )

    stats = {label: {"n": 0, "ok": 0, "no_face": 0} for label in ("closed", "open")}
    eye_stats = {label: {"n": 0, "ok": 0} for label in ("closed", "open")}
    eye_records: list[tuple[bool, float]] = []
    misses: list[tuple[str, str, str]] = []

    try:
        for label, path in samples:
            image = cv2.imread(str(path))
            if image is None:
                continue

            # 每张图互相独立，必须清掉滑动窗口，否则会被上一张的结果污染
            pipeline.smoother.clear()
            _, result = pipeline.process(image)
            stats[label]["n"] += 1

            if not result.eyes:
                stats[label]["no_face"] += 1
                misses.append((label, path.name, "no_face"))
                continue

            is_open_truth = label == "open"
            preds = [eye.is_open for eye in result.eyes.values()]
            eye_records.extend((is_open_truth, eye.metric) for eye in result.eyes.values())

            for pred in preds:
                eye_stats[label]["n"] += 1
                eye_stats[label]["ok"] += int(pred == is_open_truth)

            if (sum(preds) > len(preds) / 2) == is_open_truth:
                stats[label]["ok"] += 1
            else:
                detail = " ".join(f"{'O' if eye.is_open else 'C'}{eye.metric:.2f}" for eye in result.eyes.values())
                misses.append((label, path.name, detail))
    finally:
        pipeline.close()

    _print_summary(stats, eye_stats, misses)
    if eye_records:
        _print_sweep(eye_records, args.thresh)

    return 0 if not misses else 0


def _print_summary(stats: dict, eye_stats: dict, misses: list) -> None:
    print("=" * 68)
    print(f"{'标签':<8}{'样本':>6}{'判对':>6}{'准确率':>10}{'未检出人脸':>12}")
    print("-" * 68)
    total_n = total_ok = total_no_face = 0
    for label in ("closed", "open"):
        n, ok, no_face = stats[label]["n"], stats[label]["ok"], stats[label]["no_face"]
        total_n, total_ok, total_no_face = total_n + n, total_ok + ok, total_no_face + no_face
        acc = ok / n * 100 if n else 0.0
        print(f"{label:<8}{n:>6}{ok:>6}{acc:>9.1f}%{no_face:>12}")
    print("-" * 68)
    overall = total_ok / total_n * 100 if total_n else 0.0
    print(f"{'合计':<8}{total_n:>6}{total_ok:>6}{overall:>9.1f}%{total_no_face:>12}")

    print("\n单眼级别:")
    for label in ("closed", "open"):
        n, ok = eye_stats[label]["n"], eye_stats[label]["ok"]
        print(f"  {label:<7} {ok}/{n} = {ok / n * 100 if n else 0:.1f}%")

    detected = total_n - total_no_face
    print(f"\n人脸检出率: {detected / total_n * 100 if total_n else 0:.1f}%  ({detected}/{total_n})")

    if misses:
        print(f"\n判错的 {len(misses)} 个例子（最多列 15 个）:")
        for label, name, detail in misses[:15]:
            print(f"  真值={label:<7} {name:<40} {detail}")


def _print_sweep(eye_records: list, current: float) -> None:
    truth = np.array([t for t, _ in eye_records])
    values = np.array([v for _, v in eye_records])

    print("\n阈值扫描（单眼级）:")
    print(f"  {'阈值':>8}{'准确率':>10}{'睁眼召回':>10}{'闭眼召回':>10}")
    for thresh in THRESH_SWEEP:
        hit = values >= thresh
        acc = (hit == truth).mean() * 100
        recall_open = hit[truth].mean() * 100
        recall_closed = (~hit[~truth]).mean() * 100
        mark = "  <- 当前" if abs(thresh - current) < 1e-9 else ""
        print(f"  {thresh:>8.2f}{acc:>9.1f}%{recall_open:>9.1f}%{recall_closed:>9.1f}%{mark}")


if __name__ == "__main__":
    raise SystemExit(main())

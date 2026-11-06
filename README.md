# eye-state-detector

实时**睁眼 / 闭眼**检测：MediaPipe Face Landmarker 定位 478 个面部关键点，再用 **EAR 几何公式**
判断睁闭；双眼持续闭合超过 1.5 秒即报警 —— 可以驱动 ESP32-S3-CAM 的蜂鸣器与 92 颗同心圆灯带，
也可以只用电脑喇叭。全程离线，纯 CPU 就够，不需要 GPU。

| 项目 | 说明 |
| --- | --- |
| 判据 | EAR 眼睑宽高比（默认 0.24）/ 眼眶高宽比 ratio（默认 0.22），纯几何 |
| 模型 | MediaPipe Face Landmarker，3.6 MB，478 点，`num_faces=1` |
| 输入源 | 摄像头 / 视频 / 单图 / 图片目录 / ESP32 MJPEG 推流（5 种） |
| 报警 | ESP32 UDP `ALARM:1`（蜂鸣器 + 灯带）/ 电脑喇叭 / 两者同时 / 不驱动 |
| 准确率 | **98.1%**（160 张真实标注样本） |
| 人脸检出率 | **100%**（160/160） |
| 核心依赖 | 3 个（mediapipe / numpy / opencv-python） |
| 硬件 | ESP32-S3-CAM + OV5640 + 92 颗 WS2812 + 蜂鸣器 + 声音传感器 |

> 设计取向：**一个模型 + 一个公式**。让 MediaPipe 负责"眼睛在哪"，让 EAR 负责"睁还是闭"，
> 而不是再叠一个分类网络。理由见 [2.2](#22-为什么用-ear-而不是分类器)。

## 目录

- [一、快速开始](#一快速开始)
- [二、它是怎么工作的](#二它是怎么工作的)
- [三、硬件与接线](#三硬件与接线)
- [四、命令行](#四命令行)
- [五、测试与验证](#五测试与验证)
- [六、踩过的坑](#六踩过的坑)
- [七、项目结构](#七项目结构)
- [八、依赖与环境](#八依赖与环境)
- [附录：维护](#附录维护)

## 一、快速开始

### 1.1 安装

**用 uv（推荐）**

```bash
uv sync --extra dev          # 跑主程序 + 测试
uv sync --all-extras         # 连数据集验证的依赖一起装
```

两个 extra 都定义在 `pyproject.toml` 的 `[project.optional-dependencies]` 里：
`dev`（pytest，**是 extra 名，不是 `dev/` 目录**）、`validate`（huggingface-hub + pillow，仅拉数据集验证时需要）。
**只跑主程序不需要任何 extra** —— 核心依赖就三个。

**用 pip**

```bash
pip install -e ".[dev]"
```

本项目**不用** `requirements.txt` —— 依赖只在 `pyproject.toml` 声明一处，避免两边漂移。
若目标环境只能 `pip install -r`，从锁文件导出即可：

```bash
uv export --format requirements.txt --no-hashes -o requirements.txt
```

解释器要求 `>=3.10`；`.python-version` 固定为 **3.14**，MediaPipe 已支持 3.13 / 3.14，不需要降级。

### 1.2 下载模型

```bash
python scripts/download_models.py
```

只下 MediaPipe 的人脸关键点模型（3.6 MB）到 `models/face_landmarker.task`。
仓库里不含 `models/` 目录，程序启动前必须先下模型，否则会提示：

```text
[error] 找不到 .../models/face_landmarker.task
      先跑：python scripts/download_models.py
```

模型路径可用 `--landmarker` 指定，或用环境变量改目录：

```bash
set EYE_STATE_MODELS_DIR=D:\some\where\models   # Windows
```

### 1.3 运行

```bash
uv run eye-state                     # 电脑摄像头实时，q / ESC / 关窗口都可退出
uv run eye-state -s image -p face.jpg
uv run eye-state -s video -p clip.mp4
uv run eye-state -s dir   -p ./my_eyes       # 结果存到 ./my_eyes_out

# 完整形态：ESP32 推流 + 蜂鸣器 + 灯带
uv run eye-state -s stream -p http://192.168.137.120/stream --esp32-ip 192.168.137.120
```

用 pip 安装的话，把 `uv run eye-state` 换成 `python -m eye_state`。
没接硬件也能先跑 `camera` / `image` 模式验证判据 —— 检测逻辑与硬件完全解耦。

各输入源的差别：

| 源 | 行为 |
| --- | --- |
| `camera` | 打开本机摄像头（Windows 优先 DirectShow；`-p` 可填设备号，默认 0） |
| `video` | 逐帧处理视频；加 `--save out.mp4` 会把标注结果写成 mp4v |
| `image` | 处理单张图，标注图写到 `--out` 指定的目录（默认 `eyes_out`） |
| `dir` | 处理目录下所有 `jpg/jpeg/png/bmp/webp`，默认输出到同级 `<目录名>_out` |
| `stream` | 读 ESP32 的 MJPEG 流，并可用 `--alarm` 驱动蜂鸣器 / 灯带 / 电脑喇叭 |

`image` / `dir` 模式里每张图都会先清空平滑窗口，图片之间互不影响。

### 1.4 不接硬件先验证

```bash
uv run eye-state -s image -p face.jpg --show-roi
```

`--show-roi` 会额外弹窗显示裁出来的眼部小图，能直接看出裁剪框位置对不对。
判据、阈值、窗口这些参数都不依赖 ESP32，电脑摄像头就能完整跑通。

<br />

## 二、它是怎么工作的

### 2.1 数据流

```text
五种输入源（camera / video / image / dir / stream）都汇入同一条流水线
stream 模式由 stream.py 按 Content-Length 切 MJPEG 帧，后台线程取流
        ↓
【MediaPipe Face Landmarker】478 个面部关键点（num_faces=1）  ← 唯一的机器学习模型
        ↓
geometry.py：取眼睑 6 个点算 EAR                             ← 纯几何公式，无模型
        ↓
smoothing.py：滑动窗口均值平滑（默认 9 帧）                   ← 抗眨眼抖动
        ↓
pipeline.py：阈值判定（EAR ≥ 0.24 判睁眼）
        ↓
双眼闭合超过 1.5 秒 → 报警
        ├─ ESP32：UDP "ALARM:1" → 蜂鸣器 + 灯带（暖白渐变为红黄警示流光）
        └─ 电脑：pc_alarm.py 合成 WAV，用声卡循环播放
ESP32 本地另有一条独立链路：声音传感器 D0 检测到吹气 → 冷白灯环逐环绽放
```

| 环节 | 用什么 | 说明 |
| --- | --- | --- |
| 取帧 | OpenCV / `MJPEGReader` | 摄像头、文件、目录、HTTP 推流 |
| 人脸 / 眼睛定位 | MediaPipe Face Landmarker | Apache-2.0，478 关键点，`num_faces=1` |
| 睁 / 闭判据 | EAR 几何公式 | 阈值 0.24（`ratio` 判据为 0.22），零模型 |
| 时序稳定 | `EyeSmoother` 滑动窗口均值 | 默认 9 帧，`deque(maxlen=window)` |
| 疲劳判定 | `DrowsinessGuard` | 双眼闭合持续 > 1.5 秒 |
| 报警输出 | `remote_alarm.py` / `pc_alarm.py` | UDP 通知 ESP32，或电脑声卡提示音 |

### 2.2 为什么用 EAR 而不是分类器

这个任务本来就不需要第二个网络：

- **零额外体积** —— 不用跑第二个推理
- **可解释** —— 数值就是"眼睛张开高度 / 宽度"，出问题能直接定位
- **可分性极好** —— 实测闭眼均值 0.13、睁眼均值 0.38，中间留有余量
- **ML 一点没少** —— MediaPipe 那 478 个点依然靠神经网络

整个项目**只有一个模型**：MediaPipe 的 Face Landmarker（3.6 MB）。

### 2.3 EAR 怎么算

用眼睑 **6 个关键点**，按 *外眼角 → 上睑 → 内眼角 → 下睑* 的顺序：

$$\mathrm{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\,\|p_1 - p_4\|}$$

分子是上下眼睑的垂直距离，分母是眼睛的横向宽度。用宽度归一化后，
人离摄像头远近、脸大脸小都不影响结果。

实际索引（`geometry.py` 的 `EAR_PTS`，MediaPipe 478 点体系）：

```python
EAR_PTS = {
    "R": [33, 159, 158, 133, 153, 145],
    "L": [362, 386, 387, 263, 374, 380],
}
```

眼部 ROI 另用 `EYE_CONTOUR` 的 16 点轮廓求外接框，再按 `--pad`（默认 0.35）外扩；
`ratio` 判据就是"外接框高 / 宽"，同一套流程只换度量函数（`METRICS = {"ear": ear, "ratio": height_ratio}`）。

实测分布（80 只闭眼 + 80 只睁眼，样本取自 5.2 的验证集）：

| | 均值 | p5 | p50 | p95 | 最大 |
| --- | --- | --- | --- | --- | --- |
| 闭眼 | 0.13 | 0.02 | 0.17 | 0.24 | 0.39 |
| 睁眼 | 0.38 | 0.31 | 0.38 | 0.48 | 0.57 |

两点值得注意，否则容易误判成 bug：

- **睁眼的中位数只有 0.38**，不是"应该接近 1"。这是几何决定的：人眼的睑裂
  （上下睑间距）约 10 mm、眼宽约 30 mm，天生是 1:3 的扁平形状，
  即使刻意瞪到最大也只到 0.4\~0.55。阈值 0.24 相对睁眼中位数有约 **1.6 倍余量**。
- **有风险的是尾巴而不是峰值**：闭眼 p95 已经到 0.24，正好压在阈值上；
  睁眼 p5 是 0.31，距阈值也只有 0.07。这正是阈值扫描里 0.22\~0.30 是平台、
  而 0.18 以下闭眼召回会崩的原因。

**点集不是照抄 dlib 的经典组合。** 在闭眼 / 睁眼各 50 张真值样本上对比过多组候选点集
（3 组 EAR、眼眶宽高比、上下沿平均 / 最大 / 中段间距），
只有这组在 MediaPipe 的 478 点体系下表现最好，准确率 96.9% → 98.1%。

### 2.4 平滑与阈值

单帧判定不可靠：眨眼只有 100\~400 ms，逐帧输出会得到 `睁 闭 睁 闭 睁` 的乱跳。
`EyeSmoother` 对每只眼维护一个长度 `--window`（默认 9）的队列，取均值再与阈值比较；
`DrowsinessGuard` 只在"双眼都闭"持续 `--alert-after`（默认 1.5 秒）后才置 `alert=True`，
睁眼即复位。

## 三、硬件与接线

### 3.1 元件清单

与 [`doc.csv`](doc.csv) 里的采购清单一一对应，全部接在 **同一块 ESP32-S3-CAM** 上，
不需要第二个主控板。

| 元件 | 规格 | 数量 | 在本项目里的角色 |
| --- | --- | --- | --- |
| 开发板 | **ESP32-S3-CAM**（板载摄像头 FPC 座 + PSRAM） | 1 | 唯一主控：推流 + 灯带 + 蜂鸣器 + 采麦克风 |
| 摄像头 | **OV5640**（板载，FPC 排线） | 1 | VGA 640×480、JPEG、5 FPS |
| 麦克风 | **37-in-1 套件 Sound Sensor**（LM393 + 驻极体 + 蓝色电位器） | 1 | 声控触发，走 `D0` 数字脚 |
| 蜂鸣器 | **37-in-1 套件 Buzzer 模块** | 1 | 报警鸣响，走 `S`/`IO` 脚 |
| LED（32 枚） | WS2812 5050 | 1 环 | 最外环，`DIN` 接 GPIO38 |
| LED（24 枚） | WS2812 5050 | 1 环 | 第 4 环，串在外环 `DOUT` 后面 |
| LED（16 枚） | WS2812 5050 | 1 环 | 第 3 环 |
| LED（12 枚） | WS2812 5050 | 1 环 | 第 2 环 |
| LED（8 枚） | WS2812 5050 | 1 环 | 最内环，`DOUT` 悬空，共 **32+24+16+12+8 = 92 颗** |

> 5 环按 **外 → 内** 串联成一条灯带，固件把它们当成一颗 92 灯的整条灯带驱动。
> 环数、每环颗数写在固件顶部的 `RING_COUNTS[]`（按数据进入顺序填，默认 `{32,24,16,12,8}`），
> 换灯环只改这个数组 + `TOTAL_LEDS`（必须等于总和，否则整条亮红）。
>
> **每个环都要单独接 5V 和 GND**。WS2812 要靠自身供电才能整形转发数据，
> 只给第一环供电时，后面的环即使 `DOUT→DIN` 接对了也不会转发，
> 表现就是「只有第一圈亮，其余全黑」。

### 3.2 接线总表

先看摄像头占了哪些脚 —— 它一个吃掉 15 个 GPIO，剩下的才轮到外设：

| 摄像头信号 | GPIO | | 摄像头信号 | GPIO |
| --- | --- | --- | --- | --- |
| XCLK | 15 | | D0 | 11 |
| SIOD (SDA) | 4 | | D1 | 9 |
| SIOC (SCL) | 5 | | D2 | 8 |
| VSYNC | 6 | | D3 | 10 |
| HREF | 7 | | D4 | 12 |
| PCLK | 13 | | D5 | 18 |
| PWDN / RESET | 未接 | | D6 / D7 | 17 / 16 |

再扣掉八角 PSRAM（35\~37）、USB（19/20）、串口（43/44）和启动脚（0/3/45/46），
**实际能自由用的就 GPIO14 / 38 / 39 / 40 / 41 这五个**。本项目用前三个：

| 器件 | 引脚 | 接到 ESP32-S3-CAM | 说明 |
| --- | --- | --- | --- |
| 灯带（外环 32 颗） | `DIN` | **GPIO38** | 中间串 **330\~470Ω**，建议加 3.3V→5V 电平转换 |
| 灯带（5 环全部） | `5V` | **外部 5V/3A+ 电源** | 5 个环并联，不要只接第一个环 |
| 灯带（5 环全部） | `GND` | 外部电源 GND **+** ESP32 GND | 每环都要接，且必须与 ESP32 共地 |
| 蜂鸣器模块 | `S`（或 `IO`） | **GPIO14** | 模块自带三极管驱动，3.3V 信号够用 |
| 蜂鸣器模块 | `VCC` | **外部 5V**（与灯带并联） | 和灯带共用电源；接板子 5V 也行，它电流很小 |
| 蜂鸣器模块 | `GND` | `GND` | |
| 声音传感器 | `D0` | **GPIO39** | 数字输出，灵敏度用模块上的电位器调 |
| 声音传感器 | `VCC` | **板子 `3V3`** | 接 5V 会让 D0 输出 5V，超出 GPIO 耐压 |
| 声音传感器 | `GND` | `GND` | |
| 声音传感器 | `A0` | **不接** | 本方案只用 D0；可用脚里没有空闲 ADC1 |

### 3.3 接线图

```text
数据链（单向，从外环进）：

GPIO38 ─[330~470Ω]─► 32环 DIN
                      32环 DOUT ─► 24环 DIN
                                    24环 DOUT ─► 16环 DIN
                                                  16环 DOUT ─► 12环 DIN
                                                                12环 DOUT ─► 8环 DIN
                                                                              8环 DOUT 悬空

信号与地：

ESP32-S3-CAM
  ├─ GPIO14 ────────────► 蜂鸣器模块 S
  ├─ GPIO39 ◄──────────── 声音传感器 D0
  ├─ 3V3 ───────────────► 声音传感器 VCC     ← 一定不要接 5V
  └─ GND ───────────────┬► 声音传感器 GND
                        ├► 蜂鸣器模块 GND
                        ├► 32 / 24 / 16 / 12 / 8 环 GND
                        └► 外部 5V 电源 GND  ← 必须共地

外部 5V/3A+ ──┬─► 32 / 24 / 16 / 12 / 8 环 5V（每环单独接，就近并 1000µF 电容）
              ├─► 蜂鸣器模块 VCC
              └─► 板子 5V（可选，给板子供电）
```

一共 10 根线：

| # | 从 | 到 |
| --- | --- | --- |
| 1 | GPIO38（经 330\~470Ω） | 外环 `DIN` |
| 2\~5 | 每环 `DOUT` | 下一环 `DIN`（32→24→16→12→8，末端悬空） |
| 6 | GPIO14 | 蜂鸣器 `S` |
| 7 | 3V3 | 声音传感器 `VCC` |
| 8 | GPIO39 | 声音传感器 `D0` |
| 9 | GND（ESP32） | 声音 `GND` + 蜂鸣器 `GND` + 5 环 `GND` + 外部电源 `GND` |
| 10 | 外部 5V | 5 环 `5V`（并联）+ 蜂鸣器 `VCC` |

接线顺序建议：**先共地 → 再 5V → 最后接数据线**，避免带电插拔时打坏第一颗灯。

#### 只用一路电源

如果不想摆两个电源，用 **一个 5V/3A+ 的 USB 适配器**，在进入板子之前分流：

```text
5V/3A+ 适配器（USB-A 口）
        │
   USB-A 公头转接线端子
        ├─► Micro USB 线 ──► ESP32（只给板子，约 0.4A）
        └─► 5V/GND 端子 ──┬─► 5 个灯环 5V（就近并 1000µF）
                          ├─► 5 个灯环 GND ──► ESP32 GND（共地）
                          └─► 蜂鸣器 VCC/GND（可选）
```

**不要把灯带直接挂到板子的 5V 脚**，原因：

- 板子 5V 来自 USB 的 VBUS，中间串了二极管/保险丝，会掉 0.3\~0.5V；
- 92 颗灯报警时约 1A，白色更接近 1.3A，超过 USB 口和板内走线的承受能力；
- 大电流突变会抬高地电位，表现为摄像头绿条纹、花屏、ESP32 重启。

临时验证通路时，可以把亮度降到 `MAX_BRIGHTNESS 40` 再从板子 5V 取电，正式使用必须换外部电源。

#### 用万用表确认每个环都有电

上电后直流档逐个量每环的 `5V` 与 `GND` 焊盘，都应是 4.8\~5.2V。
量到 0V 的那个环就是没接电源，它后面的灯也会全部不亮。

### 3.4 灯带：最容易出问题的部分

- **串联方向**：`外环 DIN` ← GPIO38；`外环 DOUT` → `第 4 环 DIN` → … → `最内环`，
  最内环的 `DOUT` 悬空。接反（把 DOUT 当输入）的表现是整条不亮或只有第一颗亮。
  固件的 `RING_COUNTS[]` 按这个数据方向填（`{32,24,16,12,8}`），
  若改成从内环进，把它改回 `{8,12,16,24,32}` 即可，动画顺序会自动跟着对。
- **数据电平**：ESP32 是 3.3V 逻辑，WS2812 按 5V 供电时要求数据高电平 ≥ 0.7×VDD ≈ 3.5V。
  短线、少灯通常能"能用"，但线一长或灯一多就会出现第一颗灯乱色、后面全乱。
  正式用请加 **BSS138 / 74AHCT125 电平转换**。
- **串联电阻**：数据线串 **330\~470Ω**，抑制反射、保护第一颗灯的输入级。
- **颜色、亮度、节奏**：改固件顶部的灯效参数区 —— `MAX_BRIGHTNESS`（默认 60）、
  待机呼吸的 `IDLE_*`（每环点亮 8/5/3/2/1 颗、按黄金角排成螺旋并整体缓慢旋转，
  `IDLE_SPIN_MS` 控制旋转速度，`IDLE_MIN/MAX_LEVEL` 控制亮度）、
  疲劳流光的 `DROWSY_* / FLOW_*`、吹气绽放的 `BLOW_COLOR` 与
  `RING_GAP_MS`（默认 250 ms，相邻两环的间隔）。5 环全部亮完约 1.25 秒。
  三种颜色都是三通道全开，亮度不要照抄橙红时的 120。
- **电源**：92 颗灯全白、满亮度时理论峰值约 92×60mA ≈ **5.5A**。本项目是白色 + 亮度 60，
  估算约 **1.3A**（以电流表实测为准）。仍建议 **5V/3A 以上独立电源**，
  别从板子的 5V 脚或电脑 USB 取电。
  改回橙红（`CRGB(255,40,0)`）时电流约 1/3，亮度可以调到 120 左右。
- **只亮第一圈（外环 32 颗）**：数据在第一个环之后就停了。优先查两件事 ——
  后面 4 环有没有各自接 5V/GND；第一跳 `32 环 DOUT → 24 环 DIN` 是否开路或接反。
  灯珠本身大概率没坏：坏一颗只会让它之后的灯全黑。
- **整条不亮**：GPIO38 接到了第一环的 `DOUT` 而不是 `DIN`，或 5V/GND 没接。
- **电容**：灯带电源两端并 **1000µF 电解电容**，能显著减少闪烁和首灯偏色。
- **只给一组焊盘的灯环不能串联**：那种型号没有 `DOUT`，每环必须独占一个 GPIO
  （可用脚不够 5 环，所以请确认手上的环是带 `DIN`/`DOUT` 的可串联版本）。

### 3.5 蜂鸣器：先分清有源 / 无源

37-in-1 套件里两种都有，丝印不一定写。**通电后把 `S` 脚直接碰一下 `3V3`**：

| 现象 | 类型 | 固件设置 |
| --- | --- | --- |
| 立刻"嘀"一声 | **有源** | `BUZZER_TONE_HZ = 0`（默认，拉高即响） |
| 毫无反应 | **无源** | `BUZZER_TONE_HZ = 2730`（用 `tone()` 给方波） |

- 优先用**有源**：`tone()` 在 ESP32 上要走 LEDC，而摄像头的 XCLK 已经占了
  `LEDC_CHANNEL_0`，两者偶尔会打架（表现为摄像头出图异常或蜂鸣器不响）。
- 套件模块自带三极管，`S` 脚直接接 GPIO14 就行；如果手上是**裸蜂鸣器**（两根线），
  5V 的必须用 NPN 三极管（S8050 + 1kΩ 基极电阻）从 5V 供电，不能直连 GPIO。

### 3.6 声音传感器：只用 D0，不用 A0

模块上有 `A0`（模拟）和 `D0`（数字，LM393 比较器输出）两个输出：

- 本项目接 **`D0` → GPIO39**，代码只做 `digitalRead`，**灵敏度完全由板上蓝色电位器决定**，
  不需要在固件里调阈值。
- `A0` 悬空不接。可用脚里没有空闲 ADC1，接了也没法读。
- LM393 是开漏输出、模块自带上拉：**安静时高电平，听到声音时拉低**，
  对应固件默认的 `MIC_ACTIVE_LEVEL = LOW`。如果行为相反，改成 `HIGH`。
- 供电接 **3V3** 而不是 5V：模块的 `D0` 高电平等于 VCC，接 5V 会顶到 GPIO。
  实在要接 5V，就在 `D0` 与 GPIO39 之间加 10k/20k 分压。
- 调灵敏度：把 `MIC_DEBUG` 改成 `1`，串口（115200）会打印 `[mic] DO = HIGH/LOW`，
  一边吹气一边拧电位器，直到安静时稳定 HIGH、吹气时变 LOW。
- 没接麦克风时把 `MIC_ENABLED` 改成 `0`：GPIO39 悬空会乱跳，灯会自己亮。
- **怀疑麦克风坏了**：把顶部 `MIC_SELFTEST` 设为 `1`（默认就是 1），上电后串口会采样
  10 秒，逐条打印 DO 的电平变化，最后给出「一直 HIGH / 一直 LOW / 正常」的结论。
  测完把 `MIC_SELFTEST` 改回 `0`，否则每次开机都要多等这 10 秒。
- 报警期间固件会**临时屏蔽麦克风**，否则蜂鸣器离麦克风太近会自己把自己吹响、无限续命。

### 3.7 烧录

1. Arduino IDE 装开发板 **esp32 by Espressif**（≥ 3.0），库管理器装 **FastLED**（≥ 3.7，
   旧版不支持 ESP32-S3 的 RMT 输出）。
2. 打开 `sketch/sketch.ino`，板型选 **ESP32S3 Dev Module**，其余按板子实际配置：

   | 选项 | 值 |
   | --- | --- |
   | PSRAM | **OPI PSRAM**（N16R8 板）；QSPI 板选 QSPI PSRAM |
   | Flash Size | 16MB |
   | Partition Scheme | **Huge APP (3MB No OTA/1MB SPIFFS)** |
   | USB CDC On Boot | Enabled（想在串口监视器看日志时） |

3. 改顶部的 `ssid` / `password` 为你的 **2.4 GHz** 热点（ESP32 连不上 5 GHz）。
4. 烧录后打开串口监视器（115200），按 3.8 的顺序自检。

> `PSRAM` 选了 `Disabled`、或 `Flash Size` 还停在 4MB，是最常见的两个坑：
> 前者会报 `cam_dma_config(509): frame buffer malloc failed`，
> 后者会报 `Sketch too big`。三项按上表配齐再编译。
>
> 库管理器下载 FastLED 报 `Can't download library` 或 `HEAD request ... EOF` 是网络问题，
> 不是版本问题：换网络，或手动下载 zip 后用 `Sketch → Include Library → Add .ZIP Library` 安装。

### 3.8 上电自检顺序

1. 串口第一行是 `[boot] PSRAM: 已启用，可用 xxxx KB / 总 xxxx KB`。
   显示「未启用」或 0 KB 就是 PSRAM 没开，后面一定跟一条
   `cam_dma_config(509): frame buffer malloc failed`，回 3.7 改设置。
2. 接着应依次出现 `[camera] 初始化成功` → `[wifi] 固定 IP: ...` → `[udp] 开始监听端口 4210`。
   接了麦克风且 `MIC_SELFTEST = 1`（当前默认）时，中间会多花 10 秒跑自检，
   串口以 `[mic-test]` 开头，属于正常现象；不需要就改回 `0`。
3. 浏览器打开 `http://<那个 IP>/stream`，能看到画面 —— 说明摄像头和网络都通了。
4. 电脑端以 `stream` 模式跑起来后，**在预览窗口按空格**，蜂鸣器响 2 秒，
   灯带从暖白逐渐变成红黄流动的警示光。这一步不需要闭眼，
   专门用来验证「电脑 → UDP → ESP32」这条链路。
5. 灯效淡出之后再**拍手 / 吹气**，冷白光由最内环依次向外扩散 —— 验证麦克风那条链路。

### 3.9 接线排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 上电后整条全红常亮 | `RING_COUNTS` 之和 ≠ `TOTAL_LEDS` | 按实际灯环改这两个值 |
| `FastLED.h: No such file or directory` | 没装 FastLED，或版本过旧 | 库管理器装 FastLED ≥ 3.7；下载失败改用 zip 安装 |
| 只亮第一圈（外环 32 颗） | 后面 4 环没供电，或第一跳 `DOUT→DIN` 开路/接反 | 5 环全部接 5V/GND，量第一跳通断 |
| 整条不亮 | GPIO38 接到第一环 `DOUT`，或没接 5V/GND | 改接 `DIN`，确认共地 |
| `cam_dma_config(509): frame buffer malloc failed` | PSRAM 没启用/型号选错，或 Flash 只有 4MB | PSRAM 选 OPI/QSPI，Flash 16MB，Huge APP |
| `Sketch too big` | Partition Scheme 还是 Default 4MB | 改 **Huge APP (3MB No OTA)** |
| 只有第一颗灯亮 / 第一颗颜色乱 | 数据线没串电阻、或电平不够 | 串 390Ω，加电平转换 |
| 灯乱闪、颜色错位 | 5V 电源带不动，或没共地 | 独立电源 + 共地 + 1000µF 电容 |
| 待机时灯带一直亮着 | 这是正常的暖白呼吸待机效果 | 不想要就把 `IDLE_MIN_LEVEL` 和 `IDLE_MAX_LEVEL` 都设成 0 |
| 没人吹气也反复触发绽放 | 麦克风灵敏度太高或脚悬空 | 拧电位器，或 `MIC_ENABLED 0` |
| 蜂鸣器不响 | 无源蜂鸣器用了有源设置 | 改 `BUZZER_TONE_HZ` |
| 蜂鸣器一直响不停 | 电脑端已退出但 UDP 心跳还没超时 | 等 3 秒自动停；仍响则重启 ESP32 |
| 摄像头花屏 / 绿边 | 供电不足或 FPC 排线没压紧 | VGA 已是原生分辨率，先查电源和排线 |
| 浏览器打不开 /stream | IP 不对、连了 5 GHz、不同网段 | 看串口 `[wifi] IP` 那行，改热点到 2.4 GHz |

## 四、命令行

### 4.1 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-s, --source` | `camera` | `camera` / `video` / `image` / `dir` / `stream` |
| `-p, --path` | — | 图片/目录/视频路径；`stream` 填流地址；摄像头可填设备号 |
| `--method` | `ear` | `ear`（眼睑宽高比，推荐）/ `ratio`（眼眶高宽比） |
| `--thresh` | 按判据 | 判为睁眼的阈值；`ear` 默认 0.24、`ratio` 默认 0.22，调大会更倾向判闭眼 |
| `--window` | 9 | 时序平滑窗口帧数 |
| `--pad` | 0.35 | 眼部裁剪框外扩比例 |
| `--alert-after` | 1.5 | 连续闭眼多少秒告警 |
| `--calibrate` | 0 | 开算后 N 秒内做个人化校准，见 4.2；0=关闭 |
| `--landmarker` | `models/face_landmarker.task` | MediaPipe Face Landmarker 模型路径 |
| `--out` | 见下 | `image` / `dir` 模式的输出目录；`dir` 默认同级 `<目录名>_out`，其余 `eyes_out` |
| `--save` | — | `video` 模式的输出视频路径（mp4v，沿用源视频帧率） |
| `--no-display` | 关 | 不弹窗口，只处理并保存 |
| `--show-roi` | 关 | 额外弹窗显示裁出的眼部小图 |
| `--no-hud` | 关 | 不画左上角的 FPS / 阈值 / 状态栏（告警文字仍会显示） |
| `--max-frames` | 0 | 最多处理多少帧后退出，0=不限 |
| `--alarm` | `esp32` | 仅 `stream`：`esp32` / `pc` / `both` / `none` |
| `--esp32-ip` | — | 仅 `stream`：ESP32 的 IP；`--alarm` 为 `esp32` / `both` 时必填 |
| `--udp-port` | 4210 | 仅 `stream`：UDP 端口，与固件 `ALARM_UDP_PORT` 一致 |

参数错误会在启动时被 `_validate` 直接拦下并给出提示，例如：

```text
[error] image 模式必须用 -p 指定路径
[error] stream 模式只接受 http/https 地址，收到 'xxx'
[error] --alarm 用了 ESP32 蜂鸣器，必须同时给 --esp32-ip；只想用电脑喇叭就加 --alarm pc
```

模型路径相关的两个环境变量（都没设时用项目根目录下的 `models/`）：

```bash
set EYE_STATE_HOME=D:\path\to\project          # 项目根目录
set EYE_STATE_MODELS_DIR=D:\some\where\models  # 模型目录，优先于 EYE_STATE_HOME
```

### 4.2 个人化校准

默认阈值 0.24 是在公开数据上调的。换个人、换个摄像头、换个光线就可能不合适，
这时先试自动校准：

```bash
uv run eye-state --calibrate 3        # 对着镜头睁大眼睛 3 秒
```

它会在开始 3 秒内取 EAR 峰值当作"完全睁眼"基线，用峰值的 60% 作为阈值；
校准结束后 `pipeline.calibrated` 为 `True`，阈值写入 `smoother.thresh`。

### 4.3 作为库使用

```python
import cv2
from eye_state import EyeStatePipeline

with EyeStatePipeline("models/face_landmarker.task") as pipeline:
    image = cv2.imread("face.jpg")
    canvas, result = pipeline.process(image)

    print(result.face_found, result.summary(), result.alert)
    for side, eye in result.eyes.items():
        print(side, eye.is_open, f"EAR={eye.metric:.3f}", eye.box)
```

`EyeStatePipeline` 的主要参数：`method`、`video_mode`、`pad`、`window`、`thresh`、
`alert_seconds`、`calibrate_seconds`、`show_roi`、`show_hud`。
`process(bgr, ts_ms=None)` 返回 `(canvas, FrameResult)`。

- `FrameResult(face_found, eyes, alert)`，另有 `both_closed` 与 `summary()`。
- `result.eyes[side]` 是 `EyeResult(side, box, metric, raw, is_open)`：
  `metric` 是平滑后的值（判据用的就是它），`raw` 是本帧原始值。

### 4.4 ESP32 推流 + 报警

```bash
# 默认：ESP32 蜂鸣器 + 灯带
uv run eye-state -s stream -p http://192.168.137.120/stream --esp32-ip 192.168.137.120

# 只用电脑喇叭（不驱动灯带 / 蜂鸣器）
uv run eye-state -s stream -p http://192.168.137.120/stream --alarm pc

# 两边一起响
uv run eye-state -s stream -p http://192.168.137.120/stream --esp32-ip 192.168.137.120 --alarm both
```

窗口打开时，**按空格键**会手动触发 2 秒报警，用来直接验证「检测 → 报警」这条链路，
不用真的闭眼等 1.5 秒。这个按键**与流是否连上无关**：流没连上、中途断开、
ESP32 没上电时，窗口都会照常出现（显示 `waiting for stream...`），程序每秒自动重连一次，
期间空格、`q`、关窗口都照常响应。

| 环节 | 说明 |
| --- | --- |
| 取流 | `stream.py` 按 `Content-Length` 切帧；后台线程取流，主线程只处理最新一帧 |
| 断线 | 连接/读取失败后关闭 reader，每秒重连，断开期间先关报警、不用旧帧判定 |
| 判据 | 与摄像头模式完全相同：EAR 0.24、窗口 9 帧、双眼闭合 1.5 秒告警 |
| 报警 | `remote_alarm.py` 发 UDP；`pc_alarm.py` 用声卡放提示音；两者都是「状态变化才发，报警期间每秒重发」 |
| 固件 | `sketch/sketch.ino`，改 `ssid` / `password` 后烧录 |

为什么报警走 UDP 而不是 HTTP：ESP32 的 WebServer 是单线程的，`/stream` 会一直占着
`loop()`，此时再发 HTTP 请求没有人处理；UDP 由固件里独立的 FreeRTOS 任务接收。

注意点：

- 接线见[第三章](#三硬件与接线)。摄像头 OV5640，固件用 **VGA 640×480 + JPEG 质量 12 + 5 FPS**
  （QVGA 是 OV5640 的降采样模式，明显更糊）。
- ESP32 是 5 FPS，`--window 9` 相当于 1.8 秒的平滑延迟，觉得反应慢就用 `--window 5`。
- 固件里的 `ALARM_UDP_PORT` 和 CLI 的 `--udp-port` 必须一致（默认都是 4210）。
- 空格键依赖预览窗口，**先点一下窗口让它获得焦点**否则按键会被别的窗口吃掉；`--no-display` 下按不到。
- 灯带和蜂鸣器由 ESP32 自己驱动，电脑退出后固件会在 3 秒内结束告警并淡出回待机呼吸（心跳保护）。
- 声音触发那条与电脑无关：ESP32 插上电源、连上热点就能自己工作。

### 4.5 固定 ESP32 的 IP（拔掉 USB 后照常使用）

DHCP 分配的 IP 会随热点重启、租约变化而变，热点也不支持按 MAC 绑定地址。
固件里已经做了处理：**先 DHCP 连一次拿到网关，再把地址改成「网关同网段 + 固定主机号」**
（`STATIC_HOST_OCTET = 120`），改完会重连并核对真实地址，不对就重试，连续失败退回 DHCP。
只要热点本身不变，每次开机都是同一个 IP。

> **前三段永远跟着网关走**。换到另一个网段的热点（比如 10.147.204.x → 10.45.87.x），
> 地址必然会变成 `10.45.87.120` —— 这跟固件无关，静态 IP 必须落在热点自己的网段里。
> 要做到「任何网络都是同一个 IP」，只能固定用一个热点，或在路由器上做 MAC 绑定。

| 热点来源 | ESP32 固定 IP |
| --- | --- |
| Windows 移动热点 | `192.168.137.120` |
| Android 热点 | `192.168.43.120` |
| iPhone 个人热点 | `172.20.10.120` |

所以「电脑当服务器 + 电脑开热点」时，命令是固定的：

```bash
uv run eye-state -s stream -p http://192.168.137.120/stream --esp32-ip 192.168.137.120
```

细节：

- 串口日志里 `[wifi] 固定 IP 生效: ...` 那行才是最终地址；只出现 `[wifi] 下发固定 IP` 说明还在重连核对。
- 如果该地址被占用（一直扫不到），把 `STATIC_HOST_OCTET` 改成别的数字（如 200）重新烧录一次。
  连续 3 次核对不通过时：能靠 DHCP 连上就按 DHCP 用（不反复断网）；完全连不上则持续重试并在串口提示原因。
- 静态地址只存在 RAM 里：换热点后重新上电即可自动按新网关网段重新固定。
- 想退回 DHCP，把 `USE_STATIC_IP` 改成 0。
- Windows 移动热点需要网卡支持（`设置 → 网络和 Internet → 移动热点`），开启后 PC 自身在
  `192.168.137.1`，ESP32 会出现在同一网段。
- Windows 热点默认可能开在 5 GHz，ESP32 只能连 **2.4 GHz**，在热点设置里把「网络频段」改成 2.4 GHz。

## 五、测试与验证

### 5.1 跑测试

```bash
uv run pytest                  # 全部 84 个用例
uv run pytest -m "not smoke"   # 只跑不需要模型的 76 个
```

`test_smoke.py` 需要模型文件和测试图片，缺哪个就跳过对应用例并提示怎么准备：

```bash
python scripts/download_models.py
python scripts/validate_on_dataset.py --per-label 5
```

fixture 会在 `tmp/tests/`（或存在的 `sample_images/`）里找图片；两者都没有时，
8 个 smoke 用例会整体跳过，其余 76 个照常运行。

### 5.2 准确率

在 `MichalMlodawski/closed-open-eyes` 数据集上抽 **160 张真实标注样本**
（闭眼 80 + 睁眼 80，其中包含阈值调优时没见过的样本）实测：

| 指标 | 结果 |
| --- | --- |
| 样本级准确率（双眼多数决） | **98.1%** (157/160) |
| 单眼准确率 | **98.1%** (313/319) |
| 睁眼召回 | 98.8% |
| 闭眼召回 | 97.5% |
| 人脸检出率 | **100%** (160/160) |

复现：

```bash
python scripts/validate_on_dataset.py --per-label 80
```

首次运行会从 HuggingFace 拉样本（约 5 MB，缓存到 `tmp/tests/`），
`datasets-server` 有限流，脚本自带指数退避重试；加 `--no-cache` 可强制重下。
脚本会同时输出样本级 / 单眼级准确率、检出率和一张阈值扫描表。

### 5.3 阈值敏感性

单眼级准确率随阈值的变化 —— **0.22 \~ 0.28 是一段宽平台**，不是刀尖上的最优，
说明这个阈值不需要精调：

| 阈值 | 0.18 | 0.20 | 0.22 | **0.24** | 0.26 | 0.28 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 准确率 | 84.3% | 95.6% | 97.2% | **98.1%** | 97.5% | 97.2% | 95.9% |

## 六、踩过的坑

开发过程中真实踩到的五个坑，都已在代码里修掉：

**1. 滑动窗口跨图污染（最坑）**

图片批量模式下，`EyeSmoother` 的窗口如果不重置，前一张图的结果会污染后一张。
当时表现为"睁眼样本的准确率只有 92%"，而且错的全是**连续几批**样本 ——
因为前面 80 张闭眼图把窗口填满了。修法是在每张图之间调 `smoother.clear()`
（`cli.py` 的 `run_images()` 和 `validate_on_dataset.py` 里都有）。

**2. 裁剪框不外扩会切掉睫毛**

`--pad` 默认 0.35。不往外扩的话，闭眼时上下睫毛正好压在边界上，关键点会退化。

**3. 不能逐帧判定**

眨眼只有 100\~400 ms。不滑动窗口平滑的话，单帧结果会得到 `睁 闭 睁 闭 睁` 的乱跳。
默认 `--window 9`。

**4. 小字号别用描边**

HUD 一开始用"黑描边 + 白字"的常规做法，结果屏幕上一团糊。原因是 OpenCV 的
Hershey 字体在 scale 0.55 下笔画只有约 1px，而描边按惯例用了 `thickness + 2` = 3px，
**描边比字形本身还宽 3 倍**，白字被黑边吞掉。
改为**半透明底板 + 纯色文字**（`render.py`），与背景完全解耦，
过曝 / 暗光 / 杂乱场景都清晰。调版式改 `render.py` 顶部的 `_HUD_SCALE` / `_PANEL_ALPHA` 即可。

**5. MJPEG 不能用 `cv2.VideoCapture(url)`**

它依赖 FFmpeg 后端，面对 ESP32 这种「无总长度、连接一直不关」的流，
在 Windows 上经常卡住或解出坏帧。改成按 `Content-Length` 手动切帧（`stream.py`），
只依赖标准库负责传输、OpenCV 负责解码，行为可预期。

## 七、项目结构

```text
.
├── README.md                 # 本文档
├── pyproject.toml            # 依赖 / 打包 / pytest 配置（唯一真相源）
├── uv.lock                   # 锁定文件
├── .python-version           # 固定解释器版本（3.14）
├── .gitignore                # 已忽略 tmp 系列与构建产物
├── doc.csv                   # 元件清单（对应 README 3.1）
│
├── src/eye_state/            # 主包
│   ├── config.py             #   路径与常量（阈值、颜色、默认参数）
│   ├── geometry.py           #   关键点索引表 + EAR / 眼眶比率 / 外接框
│   ├── smoothing.py          #   滑动窗口平滑 + 疲劳告警
│   ├── stream.py             #   MJPEG over HTTP 读帧（ESP32-CAM）
│   ├── remote_alarm.py       #   UDP 通知 ESP32（ALARM:1 / ALARM:0）
│   ├── pc_alarm.py           #   用电脑声卡播放合成的报警音
│   ├── render.py             #   绘制（纯函数）
│   ├── pipeline.py           #   串联上述环节的 EyeStatePipeline
│   ├── cli.py                #   命令行入口，五种输入源
│   └── __init__.py / __main__.py
│
├── scripts/
│   ├── download_models.py    # 下载模型文件（--force 可重下）
│   └── validate_on_dataset.py# 在真实标注数据集上验证准确率
│
├── sketch/sketch.ino         # ESP32-S3-CAM 固件：推流 + 灯带 + 蜂鸣器 + 麦克风
│
└── tests/                    # pytest（84 个用例）
    ├── conftest.py           #   共享 fixture，缺模型/图片时自动跳过
    ├── test_cli.py           #   参数校验与错误提示（11 个）
    ├── test_geometry.py      #   纯几何，不需要模型（15 个）
    ├── test_smoothing.py     #   纯逻辑，不需要模型（11 个）
    ├── test_stream.py        #   MJPEG 切帧，合成字节流（10 个）
    ├── test_alarm.py         #   报警通道：UDP 发包 + 电脑喇叭（15 个）
    ├── test_render.py        #   绘制与窗口可见性（14 个）
    └── test_smoke.py         #   端到端，需要模型 + 图片（8 个）
```

`models/` 与 `tmp/tests/` 都**不在仓库里**：前者由 `scripts/download_models.py` 创建，
后者由 `validate_on_dataset.py` 在验证时缓存。它们不存在时程序照常运行，
只是 `test_smoke.py` 里依赖它们的用例会跳过。

> 注意 `.gitignore` 目前只覆盖 `tmp` 系列，没有覆盖 `models/`。
> 提交代码时别把 3.6 MB 的 `.task` 模型一起加进去（或自行在 `.gitignore` 补一行 `models/`）。

## 八、依赖与环境

### 8.1 依赖版本

运行只需要三个包，版本下限写在 `pyproject.toml`：

| 包 | 约束 | 用途 |
| --- | --- | --- |
| mediapipe | `>=0.10.9` | Face Landmarker 推理 |
| numpy | `>=1.24` | 关键点数组运算 |
| opencv-python | `>=4.8` | 取帧、绘制、窗口 |

可选 extra：`dev`（`pytest>=8`）、`validate`（`huggingface-hub>=0.20` + `pillow>=10`）。
解释器版本由 `.python-version` 固定为 **3.14**，`requires-python = ">=3.10"`，
classifiers 覆盖 3.10\~3.14；MediaPipe 官方已支持 3.13 / 3.14，不需要降级。

打包使用 hatchling，命令行入口：

```toml
[project.scripts]
eye-state = "eye_state.cli:main"
```

### 8.2 许可

本项目代码按 MIT 授权（`pyproject.toml` 里的 `license` 字段可自行修改）。
依赖的 MediaPipe 为 Apache-2.0，可自由商用。
验证脚本用到的 `MichalMlodawski/closed-open-eyes` 是**数据集**，与模型无关，
只用于跑准确率验证（见 5.2）。

## 附录：维护

依赖是分层的，清理时按需取舍：

| 层级 | 内容 | 大小 | 删了会怎样 |
| --- | --- | --- | --- |
| 最小可运行 | `src/` + `scripts/download_models.py` + `models/face_landmarker.task` | 3.6 MB | — |
| 硬件固件 | `sketch/sketch.ino` | 16 KB | 失去 ESP32 推流与报警 |
| 回归保护 | `tests/` | 20 KB | 失去 84 个用例 |
| 验证数据 | `tmp/tests/`（验证脚本缓存的样本图） | 5.6 MB | 冒烟测试跳过，重下需联网 |

前三项加起来不到 4 MB，建议都留着。最后一项可按需删，`.gitignore` 已经覆盖 `tmp`，不会误提交。

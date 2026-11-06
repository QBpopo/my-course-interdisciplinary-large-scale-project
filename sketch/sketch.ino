// =============================================================================
// eye-state-detector 固件 —— ESP32-S3-CAM
//
// 一块板子包办四件事：
//   1. OV5640 摄像头 → HTTP MJPEG 推流（/stream），电脑端做睁眼/闭眼判断
//   2. WS2812 5050 同心圆灯带（8 + 12 + 16 + 24 + 32 = 92 颗），三套灯效：
//      待机暖白呼吸 / 疲劳时暖白渐变警示流光 / 吹气时冷白由内向外逐环绽放
//      接线是从外环进数据（32 → … → 8），顺序差异在 RING_COUNTS 里处理
//   3. 蜂鸣器模块，报警时鸣响
//   4. 声音传感器模块（DO 数字脚），吹气也能触发自己的灯效 + 蜂鸣
//
// 两个触发源，灯效各不相同：
//   - 电脑（疲劳）：UDP 单包 "ALARM:1" / "ALARM:0"（因为 /stream 会把 loop() 占满，
//     HTTP 请求没人处理，所以单独起一个 FreeRTOS 任务收 UDP）→ 警示流光
//   - 本地（吹气）：声音传感器 DO 有效 → 保持 MIC_ALARM_MS → 冷白逐环绽放
//
// 接线见 README「三、硬件与接线」。烧录：Arduino IDE 选 ESP32S3 Dev Module，
// PSRAM = OPI PSRAM，分区方案 = Huge APP (3MB No OTA/1MB SPIFFS)。
// =============================================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiUdp.h>
#include <FastLED.h>

// =============================
// 摄像头引脚（ESP32-S3-CAM + OV5640，板载 FPC 排线）
// 这 15 个脚被摄像头独占，剩下的可用脚只有 14 / 38 / 39 / 40 / 41
// =============================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5

#define D0_GPIO_NUM       11
#define D1_GPIO_NUM       9
#define D2_GPIO_NUM       8
#define D3_GPIO_NUM       10
#define D4_GPIO_NUM       12
#define D5_GPIO_NUM       18
#define D6_GPIO_NUM       17
#define D7_GPIO_NUM       16

#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

// =============================
// 外设引脚：三个脚各司其职，互不冲突
// =============================
#define LED_PIN            38       // WS2812 数据线：串 330~470Ω，建议加 3.3V→5V 电平转换
#define NUM_RINGS          5        // 同心圆环数
// 按「数据线进入的顺序」排列，不是按物理内外环。
// 本项目数据先进最外环：32 → 24 → 16 → 12 → 8，末端 8 颗的 DOUT 悬空。
// 如果改成从内环进（8 → 12 → 16 → 24 → 32），把数组改回 {8,12,16,24,32} 即可。
const uint16_t RING_COUNTS[NUM_RINGS] = {32, 24, 16, 12, 8};
#define TOTAL_LEDS         92       // 必须等于上面之和，不一致会全亮红色提示
#define MAX_BRIGHTNESS     60       // 0~255；白色是三通道全开，电流约橙红的 3 倍
#define RING_GAP_MS        250      // 吹气时相邻两环的间隔；调大扩散更慢
#define HEARTBEAT_MS       3000     // 电脑超过这么久没发命令视为掉线

// =============================
// 灯效参数
// =============================
#define LED_FRAME_MS       33       // 流光 / 绽放的刷新间隔，约 30fps
#define IDLE_FRAME_MS      50       // 呼吸变化慢，20fps 就够，少打断摄像头
#define IDLE_BREATH_MS     5000     // 待机呼吸一个完整周期的时长
#define IDLE_MIN_LEVEL     10       // 呼吸最暗时的亮度（0~255）
#define IDLE_MAX_LEVEL     130      // 呼吸最亮时的亮度（0~255），比之前更暗
#define IDLE_RING_SKEW     2600     // 相邻环的相位差，形成很慢的径向明暗起伏
#define IDLE_DOT_SKEW      700      // 相邻亮点的相位差，制造轻微的闪烁流动
// 每环点亮的颗数（外 → 内，与 RING_COUNTS 同序）：8、5、3、2、1 正好是斐波那契数列
const uint8_t IDLE_DOTS_PER_RING[NUM_RINGS] = {8, 5, 3, 2, 1};
#define GOLDEN_ANGLE8      98       // 黄金角 137.5°，按 0~255 刻度取 98
#define IDLE_SPIN_MS       20000    // 整圈点阵转一圈的时间；越小转得越快

#define DROWSY_WARM_MS     600      // 疲劳告警开头：纯暖白保持
#define DROWSY_FADE_MS     1600     // 暖白 → 鲜艳流光的过渡时长
#define FLOW_DENSITY       7        // 流光条纹密度，越大条纹越密
#define FLOW_SPEED_MIN     2        // 起始流速（每帧相位增量）
#define FLOW_SPEED_MAX     14       // 过渡完成后的流速，频率明显加快
#define FLOW_HUE_SPAN      64       // 流光色相跨度：红 → 橙 → 黄

#define BLOW_RAMP_MS       260      // 每一环从暗到亮的绽放时间
#define RELEASE_MS         900      // 场景结束后的淡出时长

// 灯效场景。必须放在第一个函数之前：Arduino 会把函数声明自动插到首个函数处，
// 类型定义晚于那里就会报 “LedScene was not declared”。
enum LedScene { SCENE_IDLE, SCENE_DROWSY, SCENE_BLOW, SCENE_RELEASE };

const CRGB IDLE_COLOR(255, 190, 130);      // 待机暖白
const CRGB DROWSY_WARM_COLOR(255, 185, 120);  // 疲劳告警的起始暖白
const CRGB BLOW_COLOR(190, 220, 255);      // 吹气冷白

#define BUZZER_GPIO_NUM    14       // 蜂鸣器模块 S / IO 脚
#define BUZZER_TONE_HZ     0        // 0 = 有源蜂鸣器（高低电平）；无源改成 2730 之类的频率

#define MIC_PIN            39       // 声音传感器模块 D0 → IO39（只读数字量）
#define MIC_ENABLED        1        // 没接麦克风改成 0，否则悬空脚会乱触发
#define MIC_ACTIVE_LEVEL   LOW      // 有声时 DO 的电平；模块相反就改成 HIGH
#define MIC_ALARM_MS       3000     // 声音触发后保持多久
#define MIC_DEBUG          0        // 1 = 串口打印 DO 状态，用来对电位器
#define MIC_SELFTEST       1        // 1 = 上电做一次 10 秒麦克风自检；确认没坏后改回 0
#define MIC_TEST_MS        10000    // 自检采样时长

#define ALARM_UDP_PORT     4210     // 必须和电脑端 --udp-port 一致

// =============================
// 热点信息（只支持 2.4 GHz；ESP32 连不上 5 GHz）
// =============================
const char* ssid     = "Q";
const char* password = "123456789";

// =============================
// 固定 IP
// 手机热点不支持「按 MAC 绑定 IP」，所以固件里自己固定：
// 先用 DHCP 连上拿到网关，再把地址改成「网关前三段 + 固定主机号」。
// 换热点也不用改代码。冲突时把 STATIC_HOST_OCTET 换个数字重新烧录。
//
// 有些 ESP32 核心版本里 WiFi.config() 会被 DHCP 悄悄覆盖，所以下发静态配置后
// 必须重连并核对真实地址；不对就重试，仍旧不对但 DHCP 能连上时就按 DHCP 用，
// 不会把设备反复断网。注意：IP 的前三段永远跟着网关走，换网段必然变化。
// =============================
#define USE_STATIC_IP      1
#define STATIC_HOST_OCTET  120      // 最终地址形如 192.168.43.120
#define STATIC_VERIFY_MS   3000     // 重连后等待多久再核对地址
#define STATIC_MAX_RETRY   3        // 连续失败这么多次就接受 DHCP

bool staticIpApplied = false;   // 静态地址已核对生效
bool staticPending = false;     // 已下发静态配置，等待重连核对
bool staticGaveUp = false;      // 多次失败，已退回 DHCP
uint8_t staticAttempts = 0;
IPAddress staticTarget;
IPAddress staticGateway;
IPAddress staticMask;
unsigned long staticAttemptAt = 0;

// 只有「已连上 且 静态地址已核对（或已放弃）」才算可用，
// 否则会先打印 DHCP 的临时 IP，让人误以为地址已固定。
bool wifiReady() {
#if USE_STATIC_IP
  return WiFi.status() == WL_CONNECTED && (staticIpApplied || staticGaveUp);
#else
  return WiFi.status() == WL_CONNECTED;
#endif
}

// =============================
// HTTP 服务器与推流
// =============================
WebServer server(80);

const int targetFPS = 5;    // VGA 数据量大，降到 5 FPS 保证稳定
const unsigned long frameInterval = 1000 / targetFPS;
unsigned long lastFrameTime = 0;

unsigned long lastReconnectAttempt = 0;

void connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);              // 关省电，降低推流延迟
  WiFi.setHostname("esp32cam");
  WiFi.begin(ssid, password);
  Serial.println("[wifi] 正在连接热点...");
}

// 掉线就重连（非阻塞）。注意：推流期间 loop() 被占住，这里只在空闲时执行。
void maintainWiFi() {
  if (staticPending) return;   // 正在切换静态地址，交给 applyStaticIpOnce 管重连
  if (WiFi.status() == WL_CONNECTED) return;

  if (millis() - lastReconnectAttempt >= 3000) {
    lastReconnectAttempt = millis();
    Serial.println("[wifi] 掉线，重连中...");
    WiFi.disconnect();
    WiFi.begin(ssid, password);
  }
}

#if USE_STATIC_IP
// 下发静态配置：先彻底停掉站点，再设地址，避免 DHCP 把配置覆盖回去
void pushStaticConfig() {
  staticAttemptAt = millis();
  WiFi.disconnect(true);
  delay(200);
  if (!WiFi.config(staticTarget, staticGateway, staticMask, staticGateway)) {
    Serial.println("[wifi] WiFi.config() 返回失败，稍后重试");
  }
  WiFi.begin(ssid, password);
}
#endif

// DHCP 拿到网关后，把地址改成同网段的固定主机号。
// 下发后要重连核对：只有真实地址与目标一致才算成功，否则重试或退回 DHCP。
void applyStaticIpOnce() {
#if USE_STATIC_IP
  if (staticIpApplied || staticGaveUp) return;

  // 阶段一：用 DHCP 的临时连接拿到网关，算出目标地址并下发
  if (!staticPending) {
    if (WiFi.status() != WL_CONNECTED) return;

    IPAddress gw = WiFi.gatewayIP();
    if (gw[0] == 0) return;                 // 还没拿到有效数据

    IPAddress target(gw[0], gw[1], gw[2], STATIC_HOST_OCTET);
    if (target == WiFi.localIP()) {         // DHCP 恰好就分到了这个地址
      staticIpApplied = true;
      Serial.printf("[wifi] 固定 IP 生效: %s\n", target.toString().c_str());
      return;
    }

    staticTarget = target;
    staticGateway = gw;
    staticMask = WiFi.subnetMask();
    staticPending = true;
    staticAttempts = 0;
    Serial.printf("[wifi] 下发固定 IP: %s（网关 %s）\n",
                  staticTarget.toString().c_str(), staticGateway.toString().c_str());
    pushStaticConfig();
    return;
  }

  // 阶段二：等重连完成后核对真实地址
  if (millis() - staticAttemptAt < STATIC_VERIFY_MS) return;

  if (WiFi.localIP() == staticTarget) {
    staticPending = false;
    staticIpApplied = true;
    Serial.printf("[wifi] 固定 IP 生效: %s\n", staticTarget.toString().c_str());
    return;
  }

  staticAttempts++;
  bool connected = WiFi.status() == WL_CONNECTED;

  // 已经用 DHCP 连上、只是地址被 DHCP 覆盖：就用 DHCP，不再反复断网
  if (connected && staticAttempts >= STATIC_MAX_RETRY) {
    staticPending = false;
    staticGaveUp = true;
    Serial.printf("[wifi] 固定 IP 连续 %u 次未生效，改用 DHCP（当前 %s）\n",
                  (unsigned)staticAttempts, WiFi.localIP().toString().c_str());
    return;
  }

  // 还连不上：多半是目标地址被占用，继续重试，并把排查方向打出来
  Serial.printf("[wifi] 固定 IP 未生效（当前 %s），第 %u 次重试\n",
                WiFi.localIP().toString().c_str(), (unsigned)staticAttempts + 1);
  if (!connected) {
    Serial.println("[wifi] 一直连不上时：该地址可能被占用，改 STATIC_HOST_OCTET 后重新烧录");
  }
  pushStaticConfig();
#endif
}

// 一个响应体里连续放多个 JPEG，每个都带 Content-Length。
// 这里一直循环，直到客户端断开 —— 所以 loop() 在整个推流期间不会返回。
void handleStream() {
  WiFiClient client = server.client();
  camera_fb_t *fb = NULL;

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println();

  Serial.println("[stream] 客户端已连接");

  while (true) {
    if (!client.connected()) {
      Serial.println("[stream] 客户端断开");
      client.stop();
      break;
    }

    unsigned long loopStart = millis();

    if (millis() - lastFrameTime >= frameInterval) {
      lastFrameTime = millis();

      fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("[camera] 取帧失败");
        client.stop();
        break;
      }

      client.print("--frame\r\n");
      client.print("Content-Type: image/jpeg\r\n");
      client.printf("Content-Length: %u\r\n\r\n", fb->len);

      // 必须写满整帧：TCP 缓冲满时 write() 只写一部分，这是正常现象，不能当成断开
      size_t sent = 0;
      while (sent < fb->len && client.connected()) {
        size_t n = client.write(fb->buf + sent, fb->len - sent);
        if (n == 0) {
          delay(1);
          continue;
        }
        sent += n;
      }

      if (sent < fb->len) {
        Serial.println("[stream] 发送过程中客户端断开");
        esp_camera_fb_return(fb);
        client.stop();
        break;
      }

      client.print("\r\n");
      esp_camera_fb_return(fb);

      // 每 30 帧打印一次耗时，太频繁会拖慢推流
      static unsigned int framesSent = 0;
      if (++framesSent % 30 == 0) {
        Serial.printf("[stream] 第 %u 帧 | %u 字节 | 总耗时 %lu ms\n",
                      framesSent, sent, millis() - loopStart);
      }

      yield();
    }
  }

  Serial.println("[stream] 已结束");
}

void startCameraServer() {
  server.on("/", []() {
    server.send(200, "text/plain", "MJPEG 服务已启动，访问 /stream 看视频。");
  });
  server.on("/stream", HTTP_GET, handleStream);
  server.begin();
  Serial.println("[http] 服务已启动：/stream");
}

// =============================
// 灯带 + 蜂鸣器 + 麦克风状态机
// 全部放在独立任务里，因为 /stream 会把 loop() 占满。
//
// 灯效分三种场景，由触发源决定：
//   待机（无触发）    暖白呼吸，很慢的径向明暗起伏
//   疲劳（电脑 UDP）  暖白 → 红黄警示流光，流速越来越快
//   吹气（本地麦克风）冷白由内向外逐环绽放，像蒲公英散开
// 场景结束时冻结最后一帧，用 RELEASE_MS 淡出，不会硬切黑。
// =============================
CRGB leds[TOTAL_LEDS];
uint16_t ringStart[NUM_RINGS];   // 每环的起始下标

bool remoteOn = false;           // 电脑发来的报警状态
unsigned long remoteAt = 0;      // 最近一次收到电脑命令的时间
unsigned long localUntil = 0;    // 吹气触发的截止时刻

LedScene scene = SCENE_IDLE;
unsigned long sceneStart = 0;    // 当前场景的起始时刻
unsigned long lastRenderAt = 0;  // 上一帧渲染时刻
uint16_t breathPhase = 0;        // 呼吸相位（sin16 用 u16）
uint16_t idleSpinPhase = 0;      // 点阵旋转相位，高 8 位是当前旋转角
uint8_t flowPhase = 0;           // 流光相位（sin8 用 u8）
CRGB releaseFrame[TOTAL_LEDS];   // 淡出时冻结的画面
bool buzzerActive = false;

void lightRing(uint8_t ring, CRGB color) {
  for (uint16_t i = ringStart[ring]; i < ringStart[ring] + RING_COUNTS[ring]; i++) {
    leds[i] = color;
  }
}

bool ledActive() {
  return scene == SCENE_DROWSY || scene == SCENE_BLOW;
}

void enterScene(LedScene s, unsigned long now) {
  scene = s;
  sceneStart = now;
}

// 有源蜂鸣器直接给电平；无源蜂鸣器要给方波。
// 注意：无源方案依赖 tone() 的 LEDC 通道，与摄像头的 XCLK 通道可能冲突，
// 优先用有源蜂鸣器模块。
void buzzerOn() {
#if BUZZER_TONE_HZ
  tone(BUZZER_GPIO_NUM, BUZZER_TONE_HZ);
#else
  digitalWrite(BUZZER_GPIO_NUM, HIGH);
#endif
}

void buzzerOff() {
#if BUZZER_TONE_HZ
  noTone(BUZZER_GPIO_NUM);
#else
  digitalWrite(BUZZER_GPIO_NUM, LOW);
#endif
}

void setBuzzer(bool on) {
  if (on == buzzerActive) return;
  buzzerActive = on;
  if (on) {
    buzzerOn();
  } else {
    buzzerOff();
  }
}

// 场景结束时把当前画面存下来，之后逐帧压暗，避免直接切黑
void beginRelease(unsigned long now) {
  memcpy(releaseFrame, leds, sizeof(releaseFrame));
  enterScene(SCENE_RELEASE, now);
}

// 待机：只在每环按黄金角排出若干亮点（8/5/3/2/1 颗，斐波那契数列），
// 跨环连成一条向日葵式的螺旋；整圈点阵会缓慢旋转（20 秒一圈），
// 同时 5 秒呼吸一次，环与点之间留一点相位差，所以是「会动的静态光」。
void renderIdle(unsigned long dt) {
  breathPhase += (uint16_t)(dt * 65536UL / IDLE_BREATH_MS);
  idleSpinPhase += (uint16_t)(dt * 65536UL / IDLE_SPIN_MS);
  uint8_t spin = (uint8_t)(idleSpinPhase >> 8);   // 0~255 对应一圈

  fill_solid(leds, TOTAL_LEDS, CRGB::Black);

  uint8_t seed = 0;   // 全局种子序号，决定亮点在螺旋上的位置
  for (uint8_t r = 0; r < NUM_RINGS; r++) {
    uint16_t ringLen = RING_COUNTS[r];

    for (uint8_t k = 0; k < IDLE_DOTS_PER_RING[r]; k++, seed++) {
      // 每个种子比上一个转 137.5°，这是斐波那契排列的核心；spin 让整圈慢慢转动
      uint8_t angle = (uint8_t)(seed * GOLDEN_ANGLE8 + spin);

      // 按 1/256 颗灯的精度算位置，再拆给相邻两颗，转动才不会一格一格地跳
      uint16_t pos = (uint16_t)angle * ringLen;
      uint16_t base = pos >> 8;
      uint8_t frac = pos & 0xFF;
      uint16_t offset = ringStart[r];

      uint16_t p = breathPhase + r * IDLE_RING_SKEW + seed * IDLE_DOT_SKEW;
      uint8_t wave = (uint8_t)((sin16(p) + 32768) >> 8);   // 0~255
      uint8_t level = IDLE_MIN_LEVEL + scale8(wave, IDLE_MAX_LEVEL - IDLE_MIN_LEVEL);

      leds[offset + base] = IDLE_COLOR;
      leds[offset + base].nscale8_video(scale8(level, 255 - frac));

      leds[offset + (base + 1) % ringLen] = IDLE_COLOR;
      leds[offset + (base + 1) % ringLen].nscale8_video(scale8(level, frac));
    }
  }
}

// 疲劳：先全身暖白，再渐进成红黄色相的流动条纹，流速随之加快
void renderDrowsy(unsigned long dt, unsigned long now) {
  unsigned long t = now - sceneStart;

  if (t < DROWSY_WARM_MS) {
    fill_solid(leds, TOTAL_LEDS, DROWSY_WARM_COLOR);
    return;
  }

  unsigned long faded = t - DROWSY_WARM_MS;
  uint8_t p = faded >= DROWSY_FADE_MS
                  ? 255
                  : (uint8_t)(faded * 255UL / DROWSY_FADE_MS);

  // 流速随过渡一起加快：起步舒缓，完全过渡后最快
  uint8_t speed = FLOW_SPEED_MIN + scale8(p, FLOW_SPEED_MAX - FLOW_SPEED_MIN);
  flowPhase += (uint8_t)((dt * speed) / LED_FRAME_MS);

  CRGB warm = DROWSY_WARM_COLOR;
  for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
    uint8_t wave = sin8((uint8_t)(i * FLOW_DENSITY) + flowPhase);
    wave = scale8(wave, wave);   // 亮带收窄、暗带拉长，流动感更清楚

    uint8_t hue = scale8((uint8_t)(flowPhase * 2 + i * 2), FLOW_HUE_SPAN);
    CRGB vivid = hsv2rgb_rainbow(CHSV(hue, 255, 255));

    CRGB c = warm;
    nblend(c, vivid, p);          // 暖白 → 鲜艳流光
    c.nscale8_video(wave);
    leds[i] = c;
  }
}

// 吹气：冷白由内向外逐环绽放，每一环都有一段柔和的亮起过程
void renderBlow(unsigned long now) {
  unsigned long t = now - sceneStart;

  for (uint8_t v = 0; v < NUM_RINGS; v++) {   // v = 0 是最内环
    uint8_t r = NUM_RINGS - 1 - v;            // 数组里外环在前，索引要倒着取
    unsigned long ringAt = (unsigned long)v * RING_GAP_MS;
    uint8_t level = 0;

    if (t >= ringAt) {
      unsigned long since = t - ringAt;
      if (since >= BLOW_RAMP_MS) {
        level = 255;
      } else {
        uint8_t raw = (uint8_t)(since * 255UL / BLOW_RAMP_MS);
        level = scale8(raw, raw);             // 缓起：先慢后快，像花瓣张开
      }
    }

    CRGB c = BLOW_COLOR;
    c.nscale8_video(level);
    lightRing(r, c);
  }
}

// 收场：把冻结的那一帧整体压暗，直到全黑后回到待机呼吸
void renderRelease(unsigned long now) {
  unsigned long t = now - sceneStart;

  if (t >= RELEASE_MS) {
    enterScene(SCENE_IDLE, now);
    fill_solid(leds, TOTAL_LEDS, CRGB::Black);
    return;
  }

  uint8_t level = 255 - (uint8_t)(t * 255UL / RELEASE_MS);
  for (uint16_t i = 0; i < TOTAL_LEDS; i++) {
    CRGB c = releaseFrame[i];
    c.nscale8_video(level);
    leds[i] = c;
  }
}

void renderLeds(unsigned long now) {
  unsigned long interval = (scene == SCENE_IDLE) ? IDLE_FRAME_MS : LED_FRAME_MS;
  unsigned long dt = now - lastRenderAt;
  if (dt < interval) return;

  lastRenderAt = now;
  if (dt > 200) dt = 200;    // 偶发卡顿后别让相位一次跳太远

  switch (scene) {
    case SCENE_IDLE:    renderIdle(dt);         break;
    case SCENE_DROWSY:  renderDrowsy(dt, now);  break;
    case SCENE_BLOW:    renderBlow(now);        break;
    case SCENE_RELEASE: renderRelease(now);     break;
  }

  FastLED.show();
}

#if MIC_SELFTEST
// 上电自检：只读 DO、不触发报警，用来判断麦克风模块和接线是否正常。
// 测完把 MIC_SELFTEST 改回 0，否则每次开机都要等这 10 秒。
void micSelfTest() {
  Serial.println("[mic-test] 接线应为 VCC→3V3、GND→GND、D0→GPIO39，不要接 5V");
  Serial.printf("[mic-test] 开始采样 %d 秒，请拍手或对着麦克风吹气...\n", MIC_TEST_MS / 1000);

  // 判断引脚是否真的被模块驱动：
  // 内部下拉都能读到 HIGH，说明外部有上拉（模块在线）；
  // 下拉读 LOW、上拉读 HIGH，说明 D0 悬空 / 没接到 GPIO39。
  pinMode(MIC_PIN, INPUT_PULLUP);
  delay(2);
  int withPullup = digitalRead(MIC_PIN);
  pinMode(MIC_PIN, INPUT_PULLDOWN);
  delay(2);
  int withPulldown = digitalRead(MIC_PIN);
  pinMode(MIC_PIN, INPUT_PULLUP);

  Serial.printf("[mic-test] 引脚检查：上拉=%s，下拉=%s\n",
                withPullup == HIGH ? "HIGH" : "LOW",
                withPulldown == HIGH ? "HIGH" : "LOW");
  if (withPulldown == HIGH) {
    Serial.println("[mic-test] 判定：D0 有外部驱动，数据线是通的");
  } else {
    Serial.println("[mic-test] 判定：D0 悬空 —— 模块没接上，或 VCC/GND 没电");
  }

  unsigned long start = millis();
  unsigned long lastEdge = start;
  unsigned long highMs = 0, lowMs = 0;
  unsigned int changes = 0;
  int last = digitalRead(MIC_PIN);

  while (millis() - start < MIC_TEST_MS) {
    int level = digitalRead(MIC_PIN);
    if (level != last) {
      unsigned long span = millis() - lastEdge;
      if (last == HIGH) {
        highMs += span;
      } else {
        lowMs += span;
      }
      lastEdge = millis();
      last = level;
      changes++;
      Serial.printf("[mic-test] 第 %u 次变化 → %s（上一电平保持 %lu ms）\n",
                    changes, level == HIGH ? "HIGH" : "LOW", span);
    }
    delay(5);
  }

  // 把最后一段没统计的时长补上
  unsigned long span = millis() - lastEdge;
  if (last == HIGH) {
    highMs += span;
  } else {
    lowMs += span;
  }

  Serial.printf("[mic-test] 结束：%u 次变化，HIGH %lu ms / LOW %lu ms\n",
                changes, highMs, lowMs);

  if (changes == 0 && last == HIGH) {
    Serial.println("[mic-test] 结论：一直是 HIGH —— 查 VCC/GND 是否接好、D0 是否接到 GPIO39，"
                   "以及蓝色电位器是否拧到能触发的方向");
  } else if (changes == 0 && last == LOW) {
    Serial.println("[mic-test] 结论：一直是 LOW —— 这也是固件的触发态，灯会一直报警。"
                   "先拧电位器让安静时回到 HIGH；仍为 LOW 就查 D0 是否对地短路");
  } else {
    Serial.println("[mic-test] 结论：检测到电平变化，模块和接线都正常，"
                   "问题在阈值或供电");
  }
  Serial.println("[mic-test] 自检结束，之后恢复正常报警逻辑");
}
#endif

// 声音传感器用 DO 数字脚：有声音就把截止时间往后推。
// 灵敏度由模块上的蓝色电位器决定，代码里不需要阈值。
void pollMic() {
  if (!MIC_ENABLED) return;
  if (ledActive()) return;  // 报警时蜂鸣器会直接吹到麦克风上，不静音会自我触发、一直响

  int level = digitalRead(MIC_PIN);
  if (level == MIC_ACTIVE_LEVEL) {
    localUntil = millis() + MIC_ALARM_MS;
  }

#if MIC_DEBUG
  static unsigned long lastPrint = 0;
  static int lastLevel = -1;
  if (level != lastLevel || millis() - lastPrint >= 1000) {
    lastLevel = level;
    lastPrint = millis();
    Serial.printf("[mic] DO = %s\n", level == HIGH ? "HIGH" : "LOW");
  }
#endif
}

// 两条触发源对应不同灯效：电脑→警示流光（优先），吹气→冷白逐环绽放
void updateAlarm() {
  unsigned long now = millis();

  // 掉线保护只针对电脑那条；吹气那条靠自己的计时
  if (remoteOn && now - remoteAt > HEARTBEAT_MS) {
    remoteOn = false;
  }

  LedScene want = SCENE_IDLE;
  if (remoteOn) {
    want = SCENE_DROWSY;
  } else if (now < localUntil) {
    want = SCENE_BLOW;
  }

  if (want == SCENE_IDLE) {
    if (scene != SCENE_IDLE && scene != SCENE_RELEASE) {
      beginRelease(now);      // 场景结束：淡出，而不是硬切黑
    }
  } else if (want != scene) {
    enterScene(want, now);
  }

  // 只有真正告警时蜂鸣；淡出和待机都静音
  setBuzzer(want == SCENE_DROWSY || want == SCENE_BLOW);

  renderLeds(now);
}

// 不依赖 WiFi 也能跑：灯带和麦克风立即工作，UDP 等网络就绪后再开。
void alarmTask(void *param) {
  WiFiUDP udp;
  bool udpStarted = false;

  while (true) {
    if (!udpStarted && wifiReady()) {
      udp.begin(ALARM_UDP_PORT);
      udpStarted = true;
      Serial.printf("[udp] 开始监听端口 %d\n", ALARM_UDP_PORT);
    }

    if (udpStarted) {
      int packetSize = udp.parsePacket();
      if (packetSize > 0) {
        char buf[24] = {0};
        if (udp.read(buf, sizeof(buf) - 1) > 0) {
          if (strncmp(buf, "ALARM:1", 7) == 0) {
            remoteOn = true;
            remoteAt = millis();
          } else if (strncmp(buf, "ALARM:0", 7) == 0) {
            remoteOn = false;
            remoteAt = millis();
            localUntil = 0;   // 电脑主动收警就立刻停，不等麦克风计时
          }
        }
      }
    }

    pollMic();
    updateAlarm();
    vTaskDelay(pdMS_TO_TICKS(2));
  }
}

// =============================
// 初始化
// =============================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);   // 关掉 WiFi/lwIP 调试日志，避免拖慢推流
  Serial.println("\n[boot] 初始化摄像头...");

  // PSRAM 诊断：fb_location 设成 PSRAM 后，PSRAM 没启用就会在 cam_dma_config 里 malloc 失败
  Serial.printf("[boot] PSRAM: %s，可用 %u KB / 总 %u KB\n",
                psramFound() ? "已启用" : "未启用",
                (unsigned)(ESP.getFreePsram() / 1024),
                (unsigned)(ESP.getPsramSize() / 1024));

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = D0_GPIO_NUM;
  config.pin_d1 = D1_GPIO_NUM;
  config.pin_d2 = D2_GPIO_NUM;
  config.pin_d3 = D3_GPIO_NUM;
  config.pin_d4 = D4_GPIO_NUM;
  config.pin_d5 = D5_GPIO_NUM;
  config.pin_d6 = D6_GPIO_NUM;
  config.pin_d7 = D7_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 16000000;   // 20MHz 时序偏紧，芯片发热后容易出错；16MHz 更稳
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;    // 640x480 是 OV5640 原生分辨率，QVGA 靠降采样会糊
  config.jpeg_quality = 12;             // 0~63，越小越清晰
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[camera] 初始化失败: 0x%x\n", err);
    return;
  }
  Serial.println("[camera] 初始化成功");

  sensor_t *s = esp_camera_sensor_get();
  if (s != nullptr) {
    Serial.printf("[camera] Sensor PID: 0x%02X（OV5640 应为 0x5640）\n", s->id.PID);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_exposure_ctrl(s, 1);
  }

  connectToWiFi();
  startCameraServer();

  pinMode(BUZZER_GPIO_NUM, OUTPUT);
  buzzerOff();

  // 灯带：先算每环起始下标；总灯数对不上就全红提示
  uint16_t sum = 0;
  for (uint8_t r = 0; r < NUM_RINGS; r++) {
    ringStart[r] = sum;
    sum += RING_COUNTS[r];
  }
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, TOTAL_LEDS);
  FastLED.setBrightness(MAX_BRIGHTNESS);
  fill_solid(leds, TOTAL_LEDS, CRGB::Black);
  FastLED.show();

  if (sum != TOTAL_LEDS) {
    fill_solid(leds, TOTAL_LEDS, CRGB::Red);
    FastLED.show();
    Serial.printf("[led] RING_COUNTS 之和 = %u，与 TOTAL_LEDS = %d 不一致\n",
                  sum, TOTAL_LEDS);
    while (true) {
      delay(100);
    }
  }

  // 声音传感器用开漏输出，模块自带上拉；再开内部上拉只是保险
  if (MIC_ENABLED) {
    pinMode(MIC_PIN, INPUT_PULLUP);
#if MIC_SELFTEST
    micSelfTest();   // 自检放在报警任务启动前，避免灯带/蜂鸣器干扰判断
#endif
  }

  xTaskCreatePinnedToCore(alarmTask, "alarm", 4096, nullptr, 1, nullptr, 0);
}

bool ipPrinted = false;

void loop() {
  maintainWiFi();
  applyStaticIpOnce();

  // 固定地址生效后才打印，避免把 DHCP 的临时 IP 当成固定地址
  if (wifiReady()) {
    if (!ipPrinted) {
      ipPrinted = true;
      Serial.println("[wifi] 已连接");
      Serial.printf("[wifi] IP: %s\n", WiFi.localIP().toString().c_str());
      Serial.printf("[wifi] 流地址: http://%s/stream\n", WiFi.localIP().toString().c_str());
    }
  } else {
    ipPrinted = false;
  }

  server.handleClient();
}

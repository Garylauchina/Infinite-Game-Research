# Infinite Game Backend — Market Camera v2

## 概述

本后端系统是一个**实时市场结构相机**（Market Camera），从 Infinite Game 核心模拟器消费原始 tick 数据，通过滑动窗口聚合器构建**结构帧（Structure Frames）**，并通过 WebSocket 实时流式传输给客户端，同时将数据持久化到磁盘（分段压缩存储）。

**核心特性：**
- ✅ **零核心修改**：不修改 `core_system/`，所有处理在 wrapper 层完成
- ✅ **实时流式传输**：WebSocket 推送 `metric`（轻量）和 `frame`（结构）
- ✅ **历史回放**：分段存储（JSONL.zst），支持完整历史数据回放
- ✅ **前后端解耦**：前端可独立开发，通过 WSS/HTTP 访问数据
- ✅ **滚动存储**：自动清理旧数据，防止磁盘溢出

---

## 架构

```
┌─────────────────┐
│ V5MarketSimulator│ (core_system, 不变)
└────────┬────────┘
         │ 每 tick: s, agents, actions, matches
         ▼
┌─────────────────┐
│  SlidingWindow  │ 滑动窗口聚合器 (W=300 ticks)
│     Camera      │ 构建结构帧: nodes + edges
└────────┬────────┘
         │
         ├─→ metric (每 tick) ──→ WSS ──→ 客户端
         │                        │
         └─→ frame (每 STRIDE) ──→ WSS ──→ 客户端
                                    │
                                    ▼
                            ┌──────────────┐
                            │ ZstdRecorder │ 分段存储
                            │ frames/      │ JSONL.zst
                            │ metrics/     │
                            └──────────────┘
```

---

## 数据结构

### 1. Metric 消息（轻量级，每 tick 或按 downsample）

```json
{
  "type": "metric",
  "t": 12345,
  "s": {
    "price_norm": 0.49875,
    "volatility": 0.003,
    "liquidity": 1.0,
    "imbalance": 0.0
  },
  "N": 10,
  "avg_exp": 0.85,
  "matches_n": 3
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定值 `"metric"` |
| `t` | integer | Tick 编号（从 0 开始递增） |
| `s` | object | 市场状态 |
| `s.price_norm` | float | 归一化价格 (0-1) |
| `s.volatility` | float | 波动率 |
| `s.liquidity` | float | 流动性 (0-1) |
| `s.imbalance` | float | 不平衡度 (-1 到 1) |
| `N` | integer | 当前活跃玩家数量 |
| `avg_exp` | float | 平均经验值 |
| `matches_n` | integer | 本 tick 成交数量 |

---

### 2. Frame 消息（结构帧，每 STRIDE ticks，t >= W 后开始）

```json
{
  "type": "frame",
  "t": 12340,
  "window": {
    "W": 300,
    "t0": 12041,
    "t1": 12340,
    "stride": 10
  },
  "nodes": [
    {
      "node_id": "agent:0@t:12041-12340",
      "agent_id": 0,
      "t0": 12041,
      "t1": 12340,
      "features": {
        "buy_count": 5,
        "sell_count": 2,
        "none_count": 293,
        "match_count": 1,
        "match_rate": 0.00333,
        "quote_price_mean": 50123.5,
        "quote_price_std": 120.1,
        "quote_dist_to_mid_mean": 0.001,
        "experience_start": 0.8,
        "experience_end": 0.82,
        "experience_delta": 0.02,
        "dominant_side": "none"
      }
    }
  ],
  "edges": [
    {
      "src": "agent:0@t:12041-12340",
      "dst": "agent:1@t:12041-12340",
      "type": "match",
      "w": 0.01
    },
    {
      "src": "agent:1@t:12041-12340",
      "dst": "agent:2@t:12041-12340",
      "type": "quote_proximity",
      "w": 0.005
    }
  ],
  "events": {
    "bursts": [],
    "shocks": []
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定值 `"frame"` |
| `t` | integer | 当前 tick（窗口结束点） |
| `window` | object | 窗口信息 |
| `window.W` | integer | 窗口大小（ticks） |
| `window.t0` | integer | 窗口起始 tick |
| `window.t1` | integer | 窗口结束 tick |
| `window.stride` | integer | 帧发送间隔 |
| `nodes` | array | 节点列表（每个 agent 一个节点） |
| `nodes[].node_id` | string | 节点 ID：`"agent:<aid>@t:<t0>-<t1>"` |
| `nodes[].agent_id` | integer | Agent ID |
| `nodes[].t0`, `nodes[].t1` | integer | 时间窗口 |
| `nodes[].features` | object | 节点特征（见下表） |
| `edges` | array | 边列表（最多 `EDGE_CAP` 条） |
| `edges[].src`, `edges[].dst` | string | 源/目标节点 ID |
| `edges[].type` | string | 边类型：`"match"` 或 `"quote_proximity"` |
| `edges[].w` | float | 权重（0-1，归一化） |

**节点特征（`features`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `buy_count` | integer | 窗口内 buy 动作数量 |
| `sell_count` | integer | 窗口内 sell 动作数量 |
| `none_count` | integer | 窗口内 none 动作数量 |
| `match_count` | integer | 窗口内成交次数 |
| `match_rate` | float | 成交率 = `match_count / W` |
| `quote_price_mean` | float | 报价价格均值 |
| `quote_price_std` | float | 报价价格标准差 |
| `quote_dist_to_mid_mean` | float | 报价到中位价的平均距离（归一化） |
| `experience_start` | float | 窗口起始经验值 |
| `experience_end` | float | 窗口结束经验值 |
| `experience_delta` | float | 经验值变化 |
| `dominant_side` | string | 主导方向：`"buy"` / `"sell"` / `"none"` |

**边类型：**
- `match`：成交边（两个 agent 在窗口内发生成交）
- `quote_proximity`：报价邻近边（两个 agent 的报价在价格上接近，距离 ≤ `eps_pct * mid`）

---

## API 接口

### WebSocket 实时流

**端点**: `/ws/stream`  
**协议**: WebSocket (WS/WSS)  
**URL**: 
- 开发: `ws://localhost:8000/ws/stream`
- 生产: `wss://45.76.97.37/ws/stream`

#### 连接示例（JavaScript）

```javascript
const ws = new WebSocket('wss://45.76.97.37/ws/stream');

ws.onopen = () => {
    console.log('✅ WebSocket 连接成功');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    if (msg.type === 'metric') {
        console.log(`Metric t=${msg.t}: N=${msg.N}, matches=${msg.matches_n}`);
    } else if (msg.type === 'frame') {
        console.log(`Frame t=${msg.t}: ${msg.nodes.length} nodes, ${msg.edges.length} edges`);
        // 处理结构帧：构建图、可视化等
    } else if (msg.type === 'heartbeat') {
        console.log(`心跳: t=${msg.t}`);
    }
};

ws.onerror = (error) => {
    console.error('连接错误:', error);
};

ws.onclose = () => {
    console.log('连接已关闭');
};
```

#### 连接示例（Python）

```python
import asyncio
import websockets
import json
import ssl

# 自签名证书：禁用验证
ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def connect():
    uri = "wss://45.76.97.37/ws/stream"
    async with websockets.connect(uri, ssl=ssl_context) as websocket:
        print("✅ WebSocket 连接成功")
        
        async for message in websocket:
            msg = json.loads(message)
            
            if msg["type"] == "metric":
                print(f"Metric t={msg['t']}: N={msg['N']}, matches={msg['matches_n']}")
            elif msg["type"] == "frame":
                print(f"Frame t={msg['t']}: {len(msg['nodes'])} nodes, {len(msg['edges'])} edges")
            elif msg["type"] == "heartbeat":
                print(f"心跳: t={msg['t']}")

asyncio.run(connect())
```

#### 消息类型

| 类型 | 说明 | 频率 |
|------|------|------|
| `metric` | 轻量级指标 | 每 tick（或按 `CAM_METRIC_DOWNSAMPLE`） |
| `frame` | 结构帧 | 每 `STRIDE` ticks（t >= W 后） |
| `heartbeat` | 心跳 | 每 30 秒（无新数据时） |
| `pong` | 响应 ping | 客户端发送 `"ping"` 时 |

---

### HTTP API（只读）

#### 1. 健康检查

```http
GET /health
```

**响应：**
```json
{
  "ok": true,
  "meta": {
    "running": true,
    "start_ts": 1769234208.5683143,
    "ticks": 14466,
    "seed": 0
  },
  "current_segment": null
}
```

#### 2. 获取 Frames Manifest

```http
GET /manifest/frames
```

**响应：** `frames/manifest.json` 内容（段文件列表）

```json
[
  {
    "path": "frames/segment_20260124_120000_1000.jsonl.zst",
    "start_t": 1000,
    "end_t": 5000,
    "lines": 400,
    "bytes": 12345678,
    "created_at": "2026-01-24T12:05:00Z",
    "pending": false
  }
]
```

#### 3. 获取 Metrics Manifest

```http
GET /manifest/metrics
```

**响应：** `metrics/manifest.json` 内容（格式同上）

#### 4. 下载段文件（Frames）

```http
GET /segment/frames/{filename}
```

**示例：**
```bash
curl -O "https://45.76.97.37/segment/frames/segment_20260124_120000_1000.jsonl.zst"
```

**响应：** 原始 `.zst` 文件（`application/octet-stream`）

#### 5. 下载段文件（Metrics）

```http
GET /segment/metrics/{filename}
```

**响应：** 原始 `.zst` 文件

#### 6. API 根路径

```http
GET /
```

**响应：** API 端点列表

```json
{
  "message": "Infinite Game API",
  "version": "2.0",
  "cam_store": "/data/ig_cam",
  "endpoints": {
    "health": "/health",
    "manifest_frames": "/manifest/frames",
    "manifest_metrics": "/manifest/metrics",
    "segment_frames": "/segment/frames/{name}",
    "segment_metrics": "/segment/metrics/{name}",
    "websocket": "/ws/stream"
  }
}
```

---

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IG_CAM_STORE_DIR` | `/data/ig_cam` | 存储根目录（frames/ + metrics/） |
| `IG_CAM_W` | `300` | 滑动窗口大小（ticks） |
| `IG_CAM_STRIDE` | `10` | 帧发送间隔（每 N ticks 发送一次 frame） |
| `IG_CAM_EDGE_CAP` | `2000` | 每帧最大边数 |
| `IG_CAM_PROX_EPS_PCT` | `0.003` | 报价邻近阈值（中位价的 0.3%） |
| `IG_CAM_SEGMENT_SECONDS` | `300` | 段文件滚动间隔（秒，默认 5 分钟） |
| `IG_CAM_ZSTD_LEVEL` | `3` | zstd 压缩级别（1-22） |
| `IG_CAM_METRIC_DOWNSAMPLE` | `1` | Metric 采样率（1 = 每 tick，2 = 每 2 ticks） |
| `IG_SEED` | `0` | 随机种子 |
| `IG_ADJUST_INTERVAL` | `2000` | 玩家调整间隔（ticks） |
| `PORT` | `8000` | HTTP/WebSocket 端口 |

### 配置示例

```bash
export IG_CAM_W=500
export IG_CAM_STRIDE=20
export IG_CAM_EDGE_CAP=3000
export IG_CAM_STORE_DIR=/data/ig_cam
export PORT=8000
```

---

## 存储结构

### 目录布局

```
/data/ig_cam/
├── frames/
│   ├── manifest.json              # 段文件索引（append-only）
│   ├── segment_20260124_120000_1000.jsonl.zst
│   ├── segment_20260124_120500_5000.jsonl.zst
│   └── segment_20260124_121000_10000_pending.jsonl.zst  # 当前写入中
└── metrics/
    ├── manifest.json
    ├── segment_20260124_120000_1000.jsonl.zst
    └── segment_20260124_120500_5000_pending.jsonl.zst
```

### 段文件格式

- **命名**: `segment_<UTC>_<start_t>.jsonl.zst`
  - `UTC`: `YYYYMMDD_HHMMSS`
  - `start_t`: 段起始 tick
  - `_pending`: 正在写入的段（完成后重命名去掉 `_pending`）

- **内容**: 每行一个 JSON 对象（NDJSON），压缩为 zstd
  - Frames: `{"type":"frame", "t":..., "window":..., "nodes":..., "edges":...}`
  - Metrics: `{"type":"metric", "t":..., "s":..., "N":..., ...}`

### Manifest 格式

```json
[
  {
    "path": "frames/segment_20260124_120000_1000.jsonl.zst",
    "start_t": 1000,
    "end_t": 5000,
    "lines": 400,
    "bytes": 12345678,
    "created_at": "2026-01-24T12:05:00Z",
    "pending": false
  }
]
```

**注意**: Manifest 是 **append-only**，不会重写历史记录。

---

## 快速开始

### 1. 安装依赖

```bash
cd /root/Infinite-Game-Research
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn websockets zstandard numpy
```

### 2. 配置环境变量（可选）

```bash
export IG_CAM_STORE_DIR=/data/ig_cam
export IG_CAM_W=300
export IG_CAM_STRIDE=10
export PORT=8000
```

### 3. 启动服务

```bash
cd experiments/live
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

或使用 systemd 服务：

```bash
sudo systemctl start infinite-live
sudo systemctl status infinite-live
```

### 4. 测试连接

```bash
# 健康检查
curl http://localhost:8000/health

# 获取 manifest
curl http://localhost:8000/manifest/frames

# WebSocket 测试（使用 wscat）
npm install -g wscat
wscat -c ws://localhost:8000/ws/stream
```

---

## 数据回放

### 前端回放流程

1. **加载 Manifest**
   ```javascript
   const manifest = await fetch('/manifest/frames').then(r => r.json());
   ```

2. **选择段文件**
   ```javascript
   const segment = manifest.find(s => s.start_t <= target_t && s.end_t >= target_t);
   ```

3. **下载并解压**
   ```javascript
   const response = await fetch(`/segment/frames/${segment.path}`);
   const compressed = await response.arrayBuffer();
   // 使用 zstd 解压（需要 zstd-wasm 或后端解压）
   ```

4. **解析 JSONL**
   ```javascript
   const lines = decompressed.split('\n');
   const frames = lines.map(line => JSON.parse(line));
   ```

5. **按顺序播放**
   ```javascript
   frames.forEach(frame => {
       renderGraph(frame.nodes, frame.edges);
   });
   ```

---

## 模块说明

### `camera_v2.py` — 滑动窗口相机

**类**: `SlidingWindowCamera`

```python
camera = SlidingWindowCamera(
    W=300,              # 窗口大小
    stride=10,          # 帧发送间隔
    eps_pct=0.003,      # 报价邻近阈值
    edge_cap=2000       # 最大边数
)

# 每 tick 调用
camera.ingest(raw_tick)  # raw_tick = {t, mid, agents, actions, matches}

# 检查是否应该发送 frame
frame = camera.maybe_build_frame(t)  # 返回 frame 或 None
```

### `recorder.py` — 分段写入器

**类**: `ZstdSegmentWriter`

```python
recorder = ZstdSegmentWriter(
    kind="frames",              # "frames" 或 "metrics"
    store_dir=Path("/data/ig_cam"),
    segment_seconds=300,        # 滚动间隔（秒）
    level=3                     # zstd 压缩级别
)

# 写入一行
recorder.write_jsonline(obj, current_t)

# 检查是否需要滚动
recorder.roll_if_needed(current_t)

# 关闭当前段
recorder.close()
```

---

## 部署

### Systemd 服务

服务文件：`infinite-live.service`

```ini
[Unit]
Description=Infinite Game Live Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/Infinite-Game-Research
Environment="PATH=/root/Infinite-Game-Research/.venv/bin"
ExecStart=/root/Infinite-Game-Research/.venv/bin/python3 -m experiments.live.server
Restart=always

[Install]
WantedBy=multi-user.target
```

**启用服务：**
```bash
sudo cp infinite-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable infinite-live
sudo systemctl start infinite-live
```

### Nginx 反向代理（HTTPS/WSS）

参考 `HTTPS_WSS_SETUP_COMPLETE.md` 配置 Nginx 反向代理和 SSL 证书。

---

## 性能与限制

### 数据量估算

- **Metric**: ~200 bytes/tick
- **Frame**: ~5-50 KB/frame（取决于节点数和边数）
- **推送频率**: 
  - Metric: 每 tick（或按 downsample）
  - Frame: 每 `STRIDE` ticks（默认每 10 ticks）

### 存储估算

- **压缩比**: zstd level 3 约 5-10:1
- **段文件大小**: 5 分钟约 10-100 MB（取决于活跃度）
- **滚动策略**: 每 5 分钟或按磁盘使用率自动清理

### 并发限制

- **WebSocket 连接**: 理论上无限制（受服务器资源限制）
- **HTTP 请求**: FastAPI 默认并发处理
- **建议**: 监控 CPU/内存/网络带宽

---

## 故障排查

### WebSocket 连接失败

1. 检查服务是否运行：`curl http://localhost:8000/health`
2. 检查防火墙/端口：`netstat -tlnp | grep 8000`
3. 查看日志：`tail -f logs/infinite-live.log`

### 数据不更新

1. 检查模拟器是否运行（`meta.running` 应为 `true`）
2. 检查 `CAM_METRIC_DOWNSAMPLE` 配置
3. 检查 Frame 发送条件：`t >= W` 且 `t % STRIDE == 0`

### 磁盘空间不足

1. 检查磁盘使用率：`df -h`
2. 查看段文件：`ls -lh /data/ig_cam/frames/`
3. 手动清理旧段（保留最近 N 个）

---

## 相关文档

- `MARKET_CAMERA_V2.md` — Market Camera v2 实现总结
- `WSS_ACCESS_GUIDE.md` — WebSocket 访问指南
- `HTTPS_WSS_SETUP_COMPLETE.md` — HTTPS/WSS 配置指南
- `TRADE_FRAME_SCHEMA.md` — 交易帧数据结构（legacy）
- `experiments/live/API_DOCUMENTATION.md` — 详细 API 文档

---

## 许可证

与主项目保持一致。

---

## 更新日志

- **2026-01-24**: Market Camera v2 实现
  - 滑动窗口聚合器
  - Metric + Frame 双消息流
  - 分段存储（frames/ + metrics/）
  - HTTP manifest/segment API

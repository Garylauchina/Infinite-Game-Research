# Infinite Game 实时数据流 API 文档

## 概述

本 API 提供两种方式访问 Infinite Game 的实时 tick 数据：
1. **WebSocket 实时流** - 实时推送每个新 tick
2. **HTTP 增量 API** - 获取历史 tick 数据

**基础 URL**: `http://45.76.97.37:8000`  
**WebSocket URL**: `ws://45.76.97.37:8000/ws/stream`

---

## 数据结构

### Tick 数据格式

每个 tick 是一个 JSON 对象，包含以下字段：

```json
{
  "type": "tick",
  "t": 12345,
  "s": {
    "price_norm": 0.49875,
    "volatility": 0.003,
    "liquidity": 1.0,
    "imbalance": 0.0
  },
  "agents": [
    {
      "id": 0,
      "experience": 0.123
    },
    {
      "id": 1,
      "experience": 0.456
    }
  ],
  "actions": [
    {
      "id": 0,
      "side": "buy",
      "price": 50000.0,
      "size": 1.0
    },
    {
      "id": 1,
      "side": "sell",
      "price": 51000.0,
      "size": 1.0
    }
  ],
  "matches": [
    {
      "a": 0,
      "b": -1,
      "prob": 0.0
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定值 `"tick"` |
| `t` | integer | Tick 编号（从 0 开始递增） |
| `s` | object | 市场状态 |
| `s.price_norm` | float | 归一化价格 (0-1) |
| `s.volatility` | float | 波动率 |
| `s.liquidity` | float | 流动性 (0-1) |
| `s.imbalance` | float | 不平衡度 (-1 到 1) |
| `agents` | array | 活跃玩家列表 |
| `agents[].id` | integer | 玩家 ID |
| `agents[].experience` | float | 玩家经验值 |
| `actions` | array | 本 tick 的所有动作 |
| `actions[].id` | integer | 动作索引 |
| `actions[].side` | string | 方向：`"buy"`, `"sell"`, `"none"` |
| `actions[].price` | float | 价格 |
| `actions[].size` | float | 数量 |
| `matches` | array | 成交记录 |
| `matches[].a` | integer | 成交的动作索引 |
| `matches[].b` | integer | 配对动作索引（-1 表示未配对） |
| `matches[].prob` | float | 成交概率 |

---

## API 端点

### 1. WebSocket 实时流

**端点**: `/ws/stream`  
**协议**: WebSocket  
**URL**: `ws://45.76.97.37:8000/ws/stream`

#### 连接方式

```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');

ws.onopen = () => {
    console.log('WebSocket connected');
};

ws.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    console.log('Tick:', tick.t, tick.agents.length);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('WebSocket closed');
};
```

#### 消息格式

**服务器 → 客户端**:
- **Tick 数据**: 每个新 tick 自动推送
  ```json
  {
    "type": "tick",
    "t": 12345,
    "s": {...},
    "agents": [...],
    "actions": [...],
    "matches": [...]
  }
  ```

- **心跳消息**: 每 30 秒发送一次（如果没有新 tick）
  ```json
  {
    "type": "heartbeat",
    "t": 12345
  }
  ```

- **Pong 响应**: 响应客户端的 ping
  ```json
  {
    "type": "pong"
  }
  ```

**客户端 → 服务器**:
- `"ping"` - 发送心跳，服务器会回复 `{"type": "pong"}`

#### 特性
- ✅ 自动重连：连接断开后需要客户端实现重连逻辑
- ✅ 实时推送：每个新 tick 立即推送（延迟 <100ms）
- ✅ 连接时自动发送最新 tick
- ✅ 心跳保持连接（30 秒超时）

---

### 2. 增量历史数据 API

**端点**: `GET /incremental`  
**URL**: `http://45.76.97.37:8000/incremental`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `from_t` | integer | 否 | 0 | 起始 tick（不包含，返回 > from_t 的数据） |
| `limit` | integer | 否 | 1000 | 最大返回数量 |

#### 请求示例

```bash
# 获取从 tick 1000 开始的数据
curl "http://45.76.97.37:8000/incremental?from_t=1000&limit=100"

# 获取最新 100 个 tick
curl "http://45.76.97.37:8000/incremental?from_t=0&limit=100"
```

#### 响应格式

```json
{
  "from_t": 1000,
  "count": 100,
  "latest_t": 12345,
  "data": [
    {
      "type": "tick",
      "t": 1001,
      "s": {...},
      "agents": [...],
      "actions": [...],
      "matches": [...]
    },
    ...
  ]
}
```

#### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `from_t` | integer | 请求的起始 tick |
| `count` | integer | 实际返回的 tick 数量 |
| `latest_t` | integer | 服务器当前最新的 tick |
| `data` | array | Tick 数据数组（按 t 递增排序） |

#### 使用场景

1. **初始加载**: 客户端连接时先拉取历史数据
2. **断点续传**: WebSocket 断开后，从上次的 tick 继续
3. **批量分析**: 获取特定时间范围的数据

---

### 3. 健康检查

**端点**: `GET /health`  
**URL**: `http://45.76.97.37:8000/health`

#### 响应示例

```json
{
  "ok": true,
  "meta": {
    "running": true,
    "start_ts": 1769192681.9547572,
    "ticks": 12345,
    "seed": 0
  },
  "current_segment": {
    "start_t": 10000,
    "lines": 5000,
    "compressed_bytes": 5000000,
    "pending": false
  }
}
```

---

### 4. 段文件列表

**端点**: `GET /segments`  
**URL**: `http://45.76.97.37:8000/segments`

返回所有已完成的段文件列表（用于历史数据回放）。

---

## 客户端实现示例

### JavaScript (浏览器)

```javascript
class InfiniteGameClient {
    constructor(url = 'ws://45.76.97.37:8000/ws/stream') {
        this.url = url;
        this.ws = null;
        this.lastTick = -1;
        this.onTickCallback = null;
    }
    
    // 连接 WebSocket
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('Connected to Infinite Game');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'tick') {
                this.lastTick = data.t;
                if (this.onTickCallback) {
                    this.onTickCallback(data);
                }
            } else if (data.type === 'heartbeat') {
                console.log('Heartbeat:', data.t);
            } else if (data.type === 'pong') {
                console.log('Pong received');
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket closed, reconnecting...');
            setTimeout(() => this.connect(), 3000);
        };
        
        // 定期发送 ping
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000);
    }
    
    // 设置 tick 回调
    onTick(callback) {
        this.onTickCallback = callback;
    }
    
    // 获取历史数据
    async getHistory(fromTick = 0, limit = 1000) {
        const response = await fetch(
            `http://45.76.97.37:8000/incremental?from_t=${fromTick}&limit=${limit}`
        );
        return await response.json();
    }
    
    // 断开连接
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// 使用示例
const client = new InfiniteGameClient();
client.onTick((tick) => {
    console.log(`Tick ${tick.t}: ${tick.agents.length} agents, ${tick.matches.length} matches`);
});
client.connect();

// 获取历史数据
client.getHistory(0, 100).then(data => {
    console.log(`Loaded ${data.count} ticks`);
});
```

### Python

```python
import asyncio
import websockets
import json
import requests

class InfiniteGameClient:
    def __init__(self, ws_url='ws://45.76.97.37:8000/ws/stream', api_url='http://45.76.97.37:8000'):
        self.ws_url = ws_url
        self.api_url = api_url
        self.last_tick = -1
        
    async def connect(self, on_tick_callback=None):
        """连接 WebSocket 并接收实时数据"""
        async with websockets.connect(self.ws_url) as websocket:
            print("Connected to Infinite Game")
            
            async def ping():
                while True:
                    await asyncio.sleep(30)
                    await websocket.send("ping")
            
            # 启动 ping 任务
            ping_task = asyncio.create_task(ping())
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("type") == "tick":
                        self.last_tick = data["t"]
                        if on_tick_callback:
                            on_tick_callback(data)
                    elif data.get("type") == "heartbeat":
                        print(f"Heartbeat: t={data['t']}")
                    elif data.get("type") == "pong":
                        print("Pong received")
            finally:
                ping_task.cancel()
    
    def get_history(self, from_t=0, limit=1000):
        """获取历史数据"""
        response = requests.get(
            f"{self.api_url}/incremental",
            params={"from_t": from_t, "limit": limit}
        )
        return response.json()

# 使用示例
async def on_tick(tick):
    print(f"Tick {tick['t']}: {tick['agents']} agents, {tick['matches']} matches")

client = InfiniteGameClient()
# 在事件循环中运行
asyncio.run(client.connect(on_tick))

# 获取历史数据
history = client.get_history(0, 100)
print(f"Loaded {history['count']} ticks")
```

---

## 性能考虑

### 数据量
- 每个 tick: 约 1-5 KB（JSON）
- 推送频率: 每个 tick 推送一次（DOWNSAMPLE=1）
- 100 个并发客户端: 约 100-500 KB/s 带宽

### 缓冲区
- 内存缓冲区: 最近 10,000 个 tick（约 10-50 MB）
- 超出缓冲区的数据需要从段文件读取

### 优化建议
1. **批量推送**: 可以修改为每 N 个 tick 推送一次
2. **选择性字段**: 客户端可以请求只接收特定字段
3. **压缩**: WebSocket 支持压缩扩展
4. **连接限制**: 建议限制每个 IP 的连接数

---

## 错误处理

### WebSocket 错误
- 连接断开: 客户端应实现自动重连
- 网络错误: 使用 `get_history()` API 补全丢失的数据
- 超时: 服务器每 30 秒发送心跳，客户端应响应

### HTTP API 错误
- `400 Bad Request`: 参数错误
- `404 Not Found`: 资源不存在
- `500 Internal Server Error`: 服务器错误

---

## 测试命令

### 测试 WebSocket (使用 wscat)

```bash
# 安装 wscat
npm install -g wscat

# 连接 WebSocket
wscat -c ws://45.76.97.37:8000/ws/stream

# 发送 ping
ping
```

### 测试 HTTP API

```bash
# 健康检查
curl http://45.76.97.37:8000/health

# 获取增量数据
curl "http://45.76.97.37:8000/incremental?from_t=0&limit=10"

# 获取段列表
curl http://45.76.97.37:8000/segments
```

---

## 更新日志

- **2026-01-23**: 初始版本
  - WebSocket 实时流
  - 增量历史数据 API
  - 10,000 tick 内存缓冲区

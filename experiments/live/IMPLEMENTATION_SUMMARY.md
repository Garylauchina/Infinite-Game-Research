# WebSocket + 历史数据 API 实现总结

## 实现概述

已实现 WebSocket 实时数据流和 HTTP 增量历史数据 API，**未修改 core_system 代码**。

---

## 修改的文件

### 1. `experiments/live/server.py`

#### 新增导入
```python
from fastapi import WebSocket, WebSocketDisconnect
```

#### 新增全局变量
```python
# WebSocket 连接池和 tick 缓冲区
websocket_connections = set()  # WebSocket 连接集合
tick_buffer = deque(maxlen=10000)  # 环形缓冲区，保留最近 10000 个 tick
tick_buffer_lock = threading.Lock()  # 线程锁
```

#### 修改的函数

**`write_tick_record()`**:
- 在写入段文件后，将 tick 数据添加到内存缓冲区
- 异步广播到所有 WebSocket 连接

**新增函数**:
- `broadcast_tick()`: 异步广播 tick 数据到所有 WebSocket 连接
- `websocket_stream()`: WebSocket 端点处理函数
- `get_incremental()`: HTTP 增量历史数据 API

---

## 新增 API 端点

### 1. WebSocket 实时流

**路径**: `/ws/stream`  
**协议**: WebSocket  
**URL**: `ws://45.76.97.37:8000/ws/stream`

**功能**:
- 实时推送每个新 tick
- 连接时自动发送最新 tick
- 每 30 秒发送心跳保持连接
- 支持客户端 ping/pong

**消息格式**:
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

### 2. HTTP 增量历史数据 API

**路径**: `GET /incremental`  
**URL**: `http://45.76.97.37:8000/incremental`

**参数**:
- `from_t` (int, 可选): 起始 tick，默认 0
- `limit` (int, 可选): 最大返回数量，默认 1000

**响应**:
```json
{
  "from_t": 1000,
  "count": 100,
  "latest_t": 12345,
  "data": [...]
}
```

---

## 数据结构

### Tick 数据格式

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
    {"id": 0, "experience": 0.123}
  ],
  "actions": [
    {"id": 0, "side": "buy", "price": 50000.0, "size": 1.0}
  ],
  "matches": [
    {"a": 0, "b": -1, "prob": 0.0}
  ]
}
```

---

## 技术实现细节

### 1. 线程安全

- `write_tick_record()` 在后台线程（`run_forever`）中运行
- WebSocket 操作在主事件循环（uvicorn）中运行
- 使用 `asyncio.run_coroutine_threadsafe()` 从后台线程调用异步函数

### 2. 内存管理

- 使用 `deque(maxlen=10000)` 作为环形缓冲区
- 自动丢弃最旧的数据，保持内存占用稳定
- 超出缓冲区的数据需要从段文件读取

### 3. 连接管理

- 使用 `set()` 存储 WebSocket 连接
- 自动清理断开的连接
- 支持多客户端同时连接

---

## 使用示例

### JavaScript 客户端

```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');

ws.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    if (tick.type === 'tick') {
        console.log(`Tick ${tick.t}: ${tick.agents.length} agents`);
    }
};
```

### Python 客户端

```python
import asyncio
import websockets
import json

async def connect():
    async with websockets.connect('ws://45.76.97.37:8000/ws/stream') as ws:
        async for message in ws:
            tick = json.loads(message)
            if tick.get('type') == 'tick':
                print(f"Tick {tick['t']}: {len(tick['agents'])} agents")

asyncio.run(connect())
```

### HTTP API 调用

```bash
# 获取从 tick 1000 开始的数据
curl "http://45.76.97.37:8000/incremental?from_t=1000&limit=100"
```

---

## 性能指标

### 数据量
- 每个 tick: 约 1-5 KB（JSON）
- 缓冲区大小: 10,000 个 tick（约 10-50 MB 内存）
- 推送延迟: <100ms

### 并发支持
- 100 个并发客户端: 约 100-500 KB/s 带宽
- 服务器内存: 约 10-50 MB（连接池）

---

## 测试方法

### 1. 测试 WebSocket

```bash
# 使用 wscat
npm install -g wscat
wscat -c ws://45.76.97.37:8000/ws/stream
```

### 2. 测试 HTTP API

```bash
# 健康检查
curl http://45.76.97.37:8000/health

# 增量数据
curl "http://45.76.97.37:8000/incremental?from_t=0&limit=10"
```

### 3. 浏览器测试

打开浏览器控制台：
```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');
ws.onmessage = e => console.log(JSON.parse(e.data));
```

---

## 注意事项

1. **事件循环**: WebSocket 广播使用 uvicorn 创建的事件循环
2. **线程安全**: 缓冲区访问使用锁保护
3. **连接清理**: 断开的连接会自动从连接池移除
4. **内存限制**: 缓冲区限制为 10,000 个 tick，超出需要从段文件读取

---

## 后续优化建议

1. **批量推送**: 可以修改为每 N 个 tick 推送一次，减少网络开销
2. **数据压缩**: 启用 WebSocket 压缩扩展
3. **选择性字段**: 允许客户端订阅特定字段
4. **连接限制**: 限制每个 IP 的连接数，防止滥用
5. **持久化缓冲区**: 将缓冲区数据持久化，支持服务器重启后恢复

---

## 文档

完整 API 文档请参考: `experiments/live/API_DOCUMENTATION.md`

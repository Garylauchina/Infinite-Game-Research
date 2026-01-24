# 实时 Tick 数据流方案设计

## 需求
让所有人通过访问 `45.76.97.37:8000` 可以持续读取到每一 tick 的数据。

## 当前架构
- 后端每 `DOWNSAMPLE` 个 tick（当前=1）写入一次数据到压缩段文件
- 数据格式：JSONL，压缩为 `.jsonl.zst`
- 无实时推送机制，只有历史数据 API

---

## 方案对比

### 方案 1: WebSocket 实时推送 ⭐⭐⭐⭐⭐

**实现方式**:
- 在 `write_tick_record()` 中，每次写入时同时广播到所有连接的 WebSocket 客户端
- 使用 `asyncio` 管理 WebSocket 连接池
- 每个 tick 数据作为 JSON 消息推送

**优点**:
- ✅ 真正的实时推送，延迟最低（<100ms）
- ✅ 双向通信，可以支持客户端请求历史数据
- ✅ 支持多客户端同时连接
- ✅ 浏览器原生支持，无需额外库
- ✅ 可以控制推送频率（如每 N 个 tick 推送一次）

**缺点**:
- ⚠️ 需要维护连接池，内存开销（每个连接约 10-50KB）
- ⚠️ 如果客户端断开，需要重连机制
- ⚠️ 网络不稳定时可能丢消息（需要客户端缓存）

**实现复杂度**: 中等
**性能影响**: 低（广播开销小）

**代码示例**:
```python
# 全局 WebSocket 连接池
websocket_connections = set()

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.add(websocket)
    try:
        while True:
            # 保持连接，等待服务器推送
            await websocket.receive_text()
    except:
        websocket_connections.remove(websocket)

# 在 write_tick_record() 中
async def broadcast_tick(tick_data):
    disconnected = set()
    for ws in websocket_connections:
        try:
            await ws.send_json(tick_data)
        except:
            disconnected.add(ws)
    websocket_connections -= disconnected
```

**适用场景**: 需要实时监控、可视化、多用户同时访问

---

### 方案 2: Server-Sent Events (SSE) ⭐⭐⭐⭐

**实现方式**:
- 使用 FastAPI 的 `StreamingResponse` 实现 SSE
- 客户端通过 `EventSource` API 接收数据流
- 每个 tick 作为 SSE 事件推送

**优点**:
- ✅ 单向推送，实现简单
- ✅ 浏览器原生支持（EventSource API）
- ✅ 自动重连机制
- ✅ HTTP 协议，易于代理和负载均衡
- ✅ 可以设置事件类型，客户端可选择性接收

**缺点**:
- ⚠️ 单向通信，客户端无法发送请求
- ⚠️ 需要维护连接池（类似 WebSocket）
- ⚠️ 某些代理可能不支持长连接

**实现复杂度**: 低
**性能影响**: 低

**代码示例**:
```python
import asyncio
from fastapi.responses import StreamingResponse

tick_queue = asyncio.Queue()

@app.get("/stream")
async def stream_ticks():
    async def event_generator():
        while True:
            tick_data = await tick_queue.get()
            yield f"data: {json.dumps(tick_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# 在 write_tick_record() 中
async def push_tick(tick_data):
    await tick_queue.put(tick_data)
```

**适用场景**: 单向实时数据流，简单监控场景

---

### 方案 3: HTTP 长轮询 (Long Polling) ⭐⭐⭐

**实现方式**:
- 客户端请求 `/poll?last_t=<tick>`，服务器等待新数据
- 有新数据时立即返回，否则等待超时（如 30 秒）
- 客户端收到响应后立即发起下一个请求

**优点**:
- ✅ 实现简单，无需维护连接池
- ✅ 兼容性好，所有 HTTP 代理都支持
- ✅ 可以精确控制客户端接收的数据范围

**缺点**:
- ⚠️ 延迟较高（取决于轮询间隔）
- ⚠️ 服务器需要维护每个客户端的等待状态
- ⚠️ 频繁的 HTTP 请求开销

**实现复杂度**: 低
**性能影响**: 中等（取决于并发客户端数）

**代码示例**:
```python
from asyncio import Queue
tick_buffer = {}  # {t: tick_data}

@app.get("/poll")
async def poll_ticks(last_t: int = 0, timeout: int = 30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 检查是否有新数据
        for t in range(last_t + 1, meta["ticks"] + 1):
            if t in tick_buffer:
                return JSONResponse({
                    "t": t,
                    "data": tick_buffer[t],
                    "next_t": t + 1
                })
        await asyncio.sleep(0.1)  # 避免 CPU 占用过高
    
    return JSONResponse({"status": "timeout", "next_t": last_t})
```

**适用场景**: 低延迟要求不高，需要兼容性好的场景

---

### 方案 4: HTTP 流式响应 (Chunked Transfer) ⭐⭐⭐⭐

**实现方式**:
- 客户端请求 `/stream`，服务器保持连接打开
- 使用 `Transfer-Encoding: chunked` 持续发送 JSONL
- 每个 tick 作为一行 JSON 发送

**优点**:
- ✅ 实现简单，无需额外协议
- ✅ 客户端可以逐行解析（流式处理）
- ✅ 兼容性好，标准 HTTP
- ✅ 可以设置缓冲区，批量发送

**缺点**:
- ⚠️ 需要维护连接状态
- ⚠️ 某些代理可能不支持长连接
- ⚠️ 客户端断开后需要重新连接

**实现复杂度**: 低
**性能影响**: 低

**代码示例**:
```python
from fastapi.responses import StreamingResponse

@app.get("/stream")
async def stream_ticks():
    async def generate():
        last_t = -1
        while True:
            # 检查新数据
            current_t = meta["ticks"]
            if current_t > last_t:
                # 从缓冲区读取新 tick
                for t in range(last_t + 1, current_t + 1):
                    if t in tick_buffer:
                        yield json.dumps(tick_buffer[t]) + "\n"
                last_t = current_t
            await asyncio.sleep(0.01)  # 10ms 检查间隔
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )
```

**适用场景**: 需要流式处理，兼容性要求高的场景

---

### 方案 5: 基于 Ring Buffer 的增量读取 ⭐⭐⭐

**实现方式**:
- 维护一个固定大小的环形缓冲区（如最近 1000 个 tick）
- 客户端请求 `/incremental?from_t=<t>`，返回从该 tick 开始的所有新数据
- 客户端定期轮询（如每秒一次）

**优点**:
- ✅ 实现简单，无需长连接
- ✅ 客户端可以控制读取频率
- ✅ 支持断点续传（从任意 tick 开始）
- ✅ 服务器资源占用低

**缺点**:
- ⚠️ 不是真正的实时（有轮询延迟）
- ⚠️ 缓冲区大小有限，可能丢失历史数据
- ⚠️ 需要客户端实现轮询逻辑

**实现复杂度**: 低
**性能影响**: 低

**代码示例**:
```python
from collections import deque

# 环形缓冲区（保留最近 10000 个 tick）
tick_ring_buffer = deque(maxlen=10000)

@app.get("/incremental")
async def get_incremental(from_t: int):
    result = []
    for tick_data in tick_ring_buffer:
        if tick_data["t"] > from_t:
            result.append(tick_data)
    return JSONResponse(result)
```

**适用场景**: 对实时性要求不高，需要简单实现的场景

---

## 推荐方案组合

### 方案 A: WebSocket + 历史数据 API（推荐）⭐⭐⭐⭐⭐

**架构**:
- WebSocket (`/ws/stream`) 用于实时推送最新 tick
- HTTP API (`/incremental?from_t=<t>`) 用于补全历史数据
- 客户端连接时先拉取历史数据，然后切换到 WebSocket 实时模式

**优点**:
- ✅ 实时性最好
- ✅ 支持断点续传
- ✅ 客户端可以灵活选择模式

**实现**:
```python
# WebSocket 实时流
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.add(websocket)
    # 发送当前最新 tick
    await websocket.send_json(get_latest_tick())
    # 保持连接，等待后续推送
    try:
        while True:
            await websocket.receive_text()
    except:
        websocket_connections.remove(websocket)

# 增量历史数据 API
@app.get("/incremental")
async def get_incremental(from_t: int = 0):
    # 从段文件或缓冲区读取
    return JSONResponse(get_ticks_from_t(from_t))
```

---

### 方案 B: SSE + 历史数据 API ⭐⭐⭐⭐

**架构**:
- SSE (`/stream`) 用于实时推送
- HTTP API (`/range?t0=<start>&t1=<end>`) 用于历史数据
- 客户端先拉取历史，然后连接 SSE

**优点**:
- ✅ 实现简单
- ✅ 浏览器原生支持
- ✅ 自动重连

---

## 性能考虑

### 数据量估算
- 每个 tick 数据大小：约 1-5 KB（JSON）
- 当前 tick 频率：每个 tick 都记录（DOWNSAMPLE=1）
- 假设 100 个并发客户端：
  - 每秒推送：~1000 次（假设每秒 1000 tick）
  - 总带宽：~100-500 KB/s
  - 服务器内存：~10-50 MB（连接池）

### 优化建议
1. **批量推送**: 每 N 个 tick 推送一次（如每 10 个 tick）
2. **数据压缩**: WebSocket 支持压缩扩展
3. **选择性字段**: 客户端可以请求只接收特定字段
4. **连接限制**: 限制每个 IP 的连接数
5. **缓冲区管理**: 使用环形缓冲区，避免内存泄漏

---

## 实现建议

### 阶段 1: 简单实现（SSE）
- 实现 SSE 端点 `/stream`
- 维护一个 tick 队列
- 在 `write_tick_record()` 中推送数据

### 阶段 2: 增强功能（WebSocket）
- 添加 WebSocket 支持
- 实现历史数据补全
- 添加连接管理和重连机制

### 阶段 3: 性能优化
- 实现批量推送
- 添加数据压缩
- 实现选择性字段订阅

---

## 客户端使用示例

### WebSocket 客户端
```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');
ws.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    console.log('Tick:', tick.t, tick.agents.length);
};
```

### SSE 客户端
```javascript
const eventSource = new EventSource('http://45.76.97.37:8000/stream');
eventSource.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    console.log('Tick:', tick.t);
};
```

### HTTP 轮询客户端
```javascript
async function pollTicks(lastT = 0) {
    const res = await fetch(`http://45.76.97.37:8000/poll?last_t=${lastT}`);
    const data = await res.json();
    if (data.t) {
        console.log('Tick:', data.t);
        pollTicks(data.t);
    } else {
        setTimeout(() => pollTicks(lastT), 1000);
    }
}
```

---

## 总结

**推荐方案**: **WebSocket + 历史数据 API**

**理由**:
1. 实时性最好，延迟最低
2. 支持双向通信，功能扩展性强
3. 浏览器原生支持，无需额外库
4. 可以灵活控制推送频率和数据格式

**实施优先级**:
1. 先实现 SSE（简单，快速验证）
2. 再升级到 WebSocket（更好的性能和功能）
3. 最后优化性能（批量推送、压缩等）

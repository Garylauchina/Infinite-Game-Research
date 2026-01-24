# WebSocket Tick 广播问题诊断

## 问题描述

WebSocket 连接后，只在第一次连接时收到一条 tick 消息，之后主要是 heartbeat 消息（每 30 秒一次）。

## 可能原因

### 1. DOWNSAMPLE 设置过大

如果 `DOWNSAMPLE > 1`，`write_tick_record()` 只会每 N 个 tick 调用一次。

**检查方法**:
```bash
# 查看当前 DOWNSAMPLE 值
grep "downsample" /root/Infinite-Game-Research/logs/infinite-live.log | tail -1

# 或查看环境变量
echo $IG_DOWNSAMPLE
```

**解决方案**:
```bash
# 设置 DOWNSAMPLE=1（每个 tick 都记录和广播）
export IG_DOWNSAMPLE=1
# 重启服务
```

### 2. main_event_loop 未正确设置

`main_event_loop` 只在 WebSocket 连接时设置。如果连接在模拟开始后建立，之前的 tick 不会被广播。

**检查方法**:
```bash
# 查看 WebSocket 连接日志
grep "WebSocket connected\|main_event_loop" /root/Infinite-Game-Research/logs/infinite-live.log | tail -10
```

**已修复**: 改进了 `main_event_loop` 的设置逻辑，添加了更详细的日志。

### 3. 广播错误被静默忽略

之前的代码使用 `logger.debug()` 记录错误，可能被忽略。

**已修复**: 
- 将错误日志级别提升到 `warning`
- 添加了详细的异常堆栈跟踪
- 添加了广播成功计数

### 4. 事件循环检查问题

`main_event_loop.is_running()` 检查可能不准确。

**已修复**: 改进了事件循环检查和错误处理。

## 诊断步骤

### 1. 检查日志

```bash
# 查看最近的 WebSocket 相关日志
tail -100 /root/Infinite-Game-Research/logs/infinite-live.log | grep -E "WebSocket|broadcast|tick"

# 查看是否有广播错误
grep "Failed to broadcast\|WebSocket send error" /root/Infinite-Game-Research/logs/infinite-live.log | tail -20
```

### 2. 检查 DOWNSAMPLE

```bash
# 查看当前配置
curl -s https://45.76.97.37/health | python3 -m json.tool | grep -A 5 "meta"

# 或查看环境变量
ps aux | grep "experiments.live.server" | grep -o "IG_DOWNSAMPLE=[0-9]*" || echo "未设置"
```

### 3. 测试 WebSocket 连接

使用测试页面或命令行工具：

```bash
# 使用 wscat（如果已安装）
wscat -c wss://45.76.97.37/ws/stream --no-check

# 或使用 Python 脚本
python3 << 'EOF'
import asyncio
import websockets
import ssl
import json

ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def test():
    async with websockets.connect('wss://45.76.97.37/ws/stream', ssl=ssl_context) as ws:
        print("Connected")
        tick_count = 0
        heartbeat_count = 0
        for i in range(100):
            msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
            data = json.loads(msg)
            if data.get('type') == 'tick':
                tick_count += 1
                print(f"Tick {tick_count}: t={data.get('t')}")
            elif data.get('type') == 'heartbeat':
                heartbeat_count += 1
                if heartbeat_count % 10 == 0:
                    print(f"Heartbeat {heartbeat_count}: t={data.get('t')}")
        print(f"\nSummary: {tick_count} ticks, {heartbeat_count} heartbeats")

asyncio.run(test())
EOF
```

## 修复内容

### 1. 改进错误处理

- 将广播错误日志级别从 `debug` 提升到 `warning`
- 添加异常堆栈跟踪 (`exc_info=True`)
- 添加广播成功计数

### 2. 改进事件循环设置

- 在 WebSocket 连接时强制更新 `main_event_loop`
- 添加日志确认事件循环设置成功
- 改进事件循环运行状态检查

### 3. 添加调试日志

- 在 `write_tick_record()` 中添加调试日志（每 1000 tick）
- 在 `broadcast_tick()` 中添加成功计数日志
- 在 WebSocket 连接时记录初始 tick 发送

## 验证修复

修复后，应该看到：

1. **日志中显示**:
   - `Set main_event_loop for WebSocket broadcasting. Loop running: True`
   - `Broadcast tick t=... to X connections` (调试模式)
   - 如果有错误，会显示 `Failed to broadcast tick t=...` 警告

2. **WebSocket 客户端应该收到**:
   - 连接时：1 条初始 tick 消息
   - 之后：每个新 tick 都会收到消息（根据 DOWNSAMPLE）
   - 每 30 秒：1 条 heartbeat 消息（如果没有新 tick）

## 如果问题仍然存在

1. **检查 DOWNSAMPLE**: 确保 `IG_DOWNSAMPLE=1` 或较小的值
2. **查看完整日志**: 检查是否有 `Failed to broadcast` 警告
3. **检查服务状态**: 确认服务正在运行并生成 tick
4. **重启服务**: 应用修复后需要重启服务

```bash
# 重启服务（根据实际部署方式）
sudo systemctl restart infinite-live
# 或
kill $(ps aux | grep "experiments.live.server" | grep -v grep | awk '{print $2}')
cd /root/Infinite-Game-Research && source .venv/bin/activate && \
  nohup python3 -m experiments.live.server > /tmp/infinite-server.log 2>&1 &
```

---

**最后更新**: 2026-01-23

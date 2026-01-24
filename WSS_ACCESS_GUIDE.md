# WSS 访问指南

## 快速开始

### 访问地址

- **WSS WebSocket**: `wss://45.76.97.37/ws/stream`
- **HTTPS API**: `https://45.76.97.37/health`

---

## 浏览器访问（JavaScript）

### 基本连接

```javascript
const ws = new WebSocket('wss://45.76.97.37/ws/stream');

ws.onopen = () => {
    console.log('✅ 连接成功');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'tick') {
        console.log(`Tick ${data.t}: ${data.agents.length} agents`);
    }
};

ws.onerror = (error) => {
    console.error('连接错误:', error);
};

ws.onclose = () => {
    console.log('连接已关闭');
};
```

### 完整示例

```javascript
const ws = new WebSocket('wss://45.76.97.37/ws/stream');

ws.onopen = () => {
    console.log('✅ WebSocket 连接成功');
};

ws.onmessage = (event) => {
    try {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'tick') {
            // 处理 tick 数据
            console.log(`Tick ${msg.t}:`, {
                agents: msg.agents.length,
                matches: msg.matches.length,
                state: msg.s
            });
        } else if (msg.type === 'heartbeat') {
            // 心跳消息
            console.log(`心跳: t=${msg.t}`);
        }
    } catch (e) {
        console.error('解析错误:', e);
    }
};

ws.onerror = (error) => {
    console.error('连接错误:', error);
};

ws.onclose = () => {
    console.log('连接已关闭');
    // 可选: 自动重连
    setTimeout(() => {
        console.log('尝试重连...');
        // 重新创建连接
    }, 3000);
};
```

---

## Python 访问

### 使用 websockets 库

```python
import asyncio
import websockets
import ssl
import json

# 禁用证书验证（自签名证书）
ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def connect():
    uri = "wss://45.76.97.37/ws/stream"
    
    async with websockets.connect(uri, ssl=ssl_context) as ws:
        print("✅ 连接成功")
        
        async for message in ws:
            data = json.loads(message)
            
            if data['type'] == 'tick':
                print(f"Tick {data['t']}: {len(data['agents'])} agents")
            elif data['type'] == 'heartbeat':
                print(f"心跳: t={data['t']}")

# 运行
asyncio.run(connect())
```

### 使用 requests 访问 HTTPS API

```python
import requests

# 禁用 SSL 验证（自签名证书）
response = requests.get(
    'https://45.76.97.37/health',
    verify=False
)

data = response.json()
print(f"服务状态: {data['ok']}")
print(f"当前 tick: {data['meta']['ticks']}")
```

---

## 数据格式

### Tick 消息

```json
{
    "type": "tick",
    "t": 123456,
    "s": {
        "price_norm": 0.5,
        "volatility": 0.01,
        "liquidity": 0.8,
        "imbalance": 0.3
    },
    "agents": [
        {"id": 0, "experience": 0.85},
        {"id": 1, "experience": 0.92}
    ],
    "actions": [
        {"id": 0, "side": "buy", "price": 50000, "size": 1.0}
    ],
    "matches": [
        {"a": 0, "b": 1, "prob": 0.8}
    ]
}
```

### 心跳消息

```json
{
    "type": "heartbeat",
    "t": 123456
}
```

---

## 重要提示

### ⚠️ 自签名证书警告

浏览器首次访问时会显示"不安全连接"警告，这是正常的（因为使用自签名证书）。

**解决方法**:
1. 点击"高级"或"Advanced"
2. 点击"继续访问"或"Proceed to 45.76.97.37"

### 🔒 安全说明

- 自签名证书仅用于加密传输，不提供身份验证
- 生产环境建议使用 Let's Encrypt 证书（需要域名）
- 当前配置适合内部使用或测试环境

---

## 常见问题

### Q: 连接失败怎么办？

1. **检查网络**: 确保可以访问 `45.76.97.37`
2. **检查端口**: 确保 443 端口未被防火墙阻止
3. **检查证书**: 浏览器需要手动接受自签名证书

### Q: 如何测试连接？

**浏览器控制台**:
```javascript
const ws = new WebSocket('wss://45.76.97.37/ws/stream');
ws.onopen = () => console.log('✅ 连接成功');
ws.onmessage = e => console.log(JSON.parse(e.data));
```

**命令行** (使用 wscat):
```bash
npm install -g wscat
wscat -c wss://45.76.97.37/ws/stream --no-check
```

### Q: 数据更新频率？

- 每个 tick 发送一次数据
- 每 30 秒发送一次心跳（heartbeat）
- 连接断开后需要重新连接

### Q: 如何获取历史数据？

使用 HTTPS API:
```
https://45.76.97.37/incremental?from_t=0&limit=100
```

---

## 更多信息

- **完整 API 文档**: `experiments/live/API_DOCUMENTATION.md`
- **数据格式说明**: `experiments/live/WEBSOCKET_DATA_FORMAT.md`
- **技术实施文档**: `HTTPS_WSS_SETUP_COMPLETE.md`

---

## 快速参考

| 项目 | 地址 |
|------|------|
| WSS WebSocket | `wss://45.76.97.37/ws/stream` |
| HTTPS API | `https://45.76.97.37/health` |
| 增量数据 | `https://45.76.97.37/incremental?from_t=0&limit=10` |

---

**最后更新**: 2026-01-23

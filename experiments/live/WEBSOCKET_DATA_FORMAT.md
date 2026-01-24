# WebSocket 接口输出数据结构

## 连接地址
```
ws://45.76.97.37:8000/ws/stream
```

---

## 消息类型

WebSocket 会发送三种类型的消息：

### 1. Tick 数据（主要消息）

每个新 tick 都会自动推送，格式如下：

```json
{
  "type": "tick",
  "t": 12345,
  "s": {
    "price_norm": 0.49875000041666645,
    "volatility": 0.003226543204437944,
    "liquidity": 1.0,
    "imbalance": 0.3333333888888704
  },
  "agents": [
    {
      "id": 0,
      "experience": 0.7009441879630163
    },
    {
      "id": 1,
      "experience": 0.8013142312573269
    },
    {
      "id": 2,
      "experience": 0.8437582640207304
    }
  ],
  "actions": [
    {
      "id": 0,
      "side": "buy",
      "price": 47729.779622517235,
      "size": 1.0
    },
    {
      "id": 1,
      "side": "sell",
      "price": 53636.40598206967,
      "size": 1.0
    },
    {
      "id": 2,
      "side": "buy",
      "price": 52261.26915768265,
      "size": 1.0
    }
  ],
  "matches": [
    {
      "a": 0,
      "b": -1,
      "prob": 0.0
    },
    {
      "a": 1,
      "b": -1,
      "prob": 0.0
    },
    {
      "a": 2,
      "b": -1,
      "prob": 0.0
    }
  ]
}
```

### 2. 心跳消息（Heartbeat）

每 30 秒发送一次（如果没有新 tick）：

```json
{
  "type": "heartbeat",
  "t": 12345
}
```

### 3. Pong 响应

响应客户端的 ping 消息：

```json
{
  "type": "pong"
}
```

---

## 字段详细说明

### 根字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型：`"tick"`, `"heartbeat"`, `"pong"` |
| `t` | integer | Tick 编号（从 0 开始递增） |

### 市场状态 (`s`)

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `price_norm` | float | 0.0 - 1.0 | 归一化价格 |
| `volatility` | float | ≥ 0.0 | 波动率 |
| `liquidity` | float | 0.0 - 1.0 | 流动性 |
| `imbalance` | float | -1.0 - 1.0 | 买卖不平衡度 |

### 玩家列表 (`agents`)

数组，每个元素包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 玩家唯一 ID |
| `experience` | float | 玩家经验值（通常 ≥ 0） |

### 动作列表 (`actions`)

数组，每个元素包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 动作索引（对应 agents 索引） |
| `side` | string | 方向：`"buy"`, `"sell"`, `"none"` |
| `price` | float | 价格（原始价格，未归一化） |
| `size` | float | 数量 |

### 成交记录 (`matches`)

数组，每个元素包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `a` | integer | 成交的动作索引 |
| `b` | integer | 配对动作索引（-1 表示未配对） |
| `prob` | float | 成交概率（当前版本为 0.0） |

---

## 实际数据示例

### 示例 1: 早期 tick（3 个玩家）

```json
{
  "type": "tick",
  "t": 0,
  "s": {
    "price_norm": 0.49875000041666645,
    "volatility": 0.3,
    "liquidity": 1.0,
    "imbalance": 1.666666111233006e-07
  },
  "agents": [
    {"id": 0, "experience": 0.0261},
    {"id": 1, "experience": 0.0261},
    {"id": 2, "experience": 0.0261}
  ],
  "actions": [
    {"id": 0, "side": "sell", "price": 50976.27, "size": 1.0},
    {"id": 1, "side": "sell", "price": 56885.31, "size": 1.0},
    {"id": 2, "side": "sell", "price": 50897.66, "size": 1.0}
  ],
  "matches": [
    {"a": 0, "b": -1, "prob": 0.0},
    {"a": 1, "b": -1, "prob": 0.0},
    {"a": 2, "b": -1, "prob": 0.0}
  ]
}
```

### 示例 2: 后期 tick（更多玩家）

```json
{
  "type": "tick",
  "t": 24190,
  "s": {
    "price_norm": 0.3467570134253583,
    "volatility": 0.0012703496835609228,
    "liquidity": 0.6,
    "imbalance": 0.2666666822222212
  },
  "agents": [
    {"id": 0, "experience": 0.7009441879630163},
    {"id": 1, "experience": 0.8013142312573269},
    {"id": 2, "experience": 0.8437582640207304},
    {"id": 3, "experience": 0.6730545495059737}
  ],
  "actions": [
    {"id": 0, "side": "buy", "price": 47729.78, "size": 1.0},
    {"id": 1, "side": "sell", "price": 53636.41, "size": 1.0},
    {"id": 2, "side": "buy", "price": 52261.27, "size": 1.0},
    {"id": 3, "side": "sell", "price": 46308.57, "size": 1.0}
  ],
  "matches": [
    {"a": 0, "b": -1, "prob": 0.0},
    {"a": 2, "b": -1, "prob": 0.0}
  ]
}
```

---

## 数据特点

### 1. 数组长度
- `agents`: 长度 = 当前活跃玩家数（通常 3-100+）
- `actions`: 长度 = `agents` 长度（每个玩家一个动作）
- `matches`: 长度 = 本 tick 的成交数（0 到 `actions` 长度）

### 2. 数值范围
- `price_norm`: 通常在 0.3 - 0.7 之间
- `volatility`: 通常在 0.0 - 0.1 之间
- `experience`: 随时间增长，通常 0.0 - 10.0+
- `price`: 原始价格，通常在 30000 - 70000 之间

### 3. 数据更新频率
- 每个 tick 推送一次（DOWNSAMPLE=1）
- 推送延迟：<100ms
- 连接时自动发送最新 tick

---

## 客户端处理示例

### JavaScript

```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'tick':
            console.log(`Tick ${data.t}:`);
            console.log(`  市场: price=${data.s.price_norm.toFixed(3)}, vol=${data.s.volatility.toFixed(4)}`);
            console.log(`  玩家: ${data.agents.length} 个`);
            console.log(`  动作: ${data.actions.length} 个`);
            console.log(`  成交: ${data.matches.length} 个`);
            break;
            
        case 'heartbeat':
            console.log(`心跳: t=${data.t}`);
            break;
            
        case 'pong':
            console.log('收到 pong');
            break;
    }
};
```

### Python

```python
import asyncio
import websockets
import json

async def connect():
    async with websockets.connect('ws://45.76.97.37:8000/ws/stream') as ws:
        async for message in ws:
            data = json.loads(message)
            
            if data.get('type') == 'tick':
                print(f"Tick {data['t']}:")
                print(f"  市场: price={data['s']['price_norm']:.3f}")
                print(f"  玩家: {len(data['agents'])} 个")
                print(f"  动作: {len(data['actions'])} 个")
                print(f"  成交: {len(data['matches'])} 个")
            elif data.get('type') == 'heartbeat':
                print(f"心跳: t={data['t']}")

asyncio.run(connect())
```

---

## 注意事项

1. **JSON 格式**: 所有消息都是有效的 JSON 字符串
2. **编码**: UTF-8
3. **连接时**: 自动发送最新 tick（如果有）
4. **心跳**: 30 秒超时，服务器会发送心跳保持连接
5. **客户端 ping**: 可以发送 `"ping"` 字符串，服务器回复 `{"type": "pong"}`

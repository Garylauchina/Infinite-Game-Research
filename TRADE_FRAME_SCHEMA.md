# 交易展示帧（Trade Frame）数据结构规范

## 概述

WSS 输出已升级为支持"交易展示软件"的数据结构，包含盘口、逐笔、K线等交易展示必需的数据。

---

## 消息结构

### Tick 消息（完整版）

```json
{
  "type": "tick",
  "schema": {
    "name": "ig_wss",
    "version": 1
  },
  "t": 12345,
  "ts_wall": 1706123456.789,
  "s": {
    "price_norm": 0.5,
    "volatility": 0.01,
    "liquidity": 0.8,
    "imbalance": 0.3
  },
  "agents": [
    {"id": 0, "experience": 0.85}
  ],
  "actions": [
    {"id": 0, "side": "buy", "price": 50000.0, "size": 1.0}
  ],
  "matches": [
    {"a": 0, "b": -1, "prob": 0.8, "matched": true}
  ],
  "frame": {
    "depth": {...},
    "trades": [...],
    "ohlcv": {...}
  },
  "metric": {
    "s": {...},
    "N": 10,
    "avg_exp": 0.85
  }
}
```

---

## Frame 结构

### 1. Depth（盘口快照）

```json
{
  "depth": {
    "bids": [[50000.0, 1.0], [49999.0, 2.0], ...],  // [price, size] 降序
    "asks": [[50001.0, 1.5], [50002.0, 2.0], ...],  // [price, size] 升序
    "mid": 50000.5,                                  // (best_bid + best_ask) / 2
    "best_bid": 50000.0,
    "best_ask": 50001.0,
    "spread": 1.0                                    // best_ask - best_bid
  }
}
```

**规则**:
- `bids`: 所有 `side=="buy"` 的订单，按价格降序聚合（同价加和），取 topK（默认 25）
- `asks`: 所有 `side=="sell"` 的订单，按价格升序聚合（同价加和），取 topK（默认 25）
- `mid`: 当 `best_bid` 和 `best_ask` 都存在时计算，否则为 `null`
- `spread`: 当 `best_bid` 和 `best_ask` 都存在时计算，否则为 `null`

### 2. Trades（逐笔成交）

```json
{
  "trades": [
    {
      "trade_id": "t:12345-i:0",
      "t": 12345,
      "ts_wall": 1706123456.789,
      "side": "buy",
      "price": 50000.5,
      "size": 1.0,
      "a": 0,
      "b": -1,
      "prob": 0.8,
      "mode": "external_fill"
    }
  ]
}
```

**规则**:
- 只包含成交的订单（`matched: true`）
- `trade_price`: 优先使用 `depth.mid`，如果 `mid` 为 `null` 则使用 `action.price`
- `trade_size`: 使用 `action.size`
- `trade_side`: 使用 `action.side`
- `mode`: 当前固定为 `"external_fill"`（外部流动性吸收模式）

### 3. OHLCV（聚合K线）

```json
{
  "ohlcv": {
    "tf": 60.0,           // 时间框（秒）
    "open": 50000.0,
    "high": 50010.0,
    "low": 49990.0,
    "close": 50005.0,
    "volume": 100.0,
    "trades": 50,
    "vwap": 50002.5,
    "start_t": 12300,
    "end_t": 12345,
    "start_ts": 1706123400.0,
    "end_ts": 1706123456.789
  }
}
```

**规则**:
- 时间框 `tf` 默认 60 秒（可通过 `IG_TF_SEC` 配置）
- 每次有成交时更新：
  - `close` = 最新成交价
  - `high`/`low` = 最高/最低价
  - `volume` += 成交数量
  - `trades` += 1
  - `vwap` = 加权平均价（sum(price*size) / sum(size)）
- 如果窗口内无成交：
  - 默认不输出 `ohlcv`（前端保持上一根）
  - 如果 `IG_EMIT_EMPTY_BARS=1`，则输出空K线（用 `mid` 填充 OHLC，volume=0）

---

## Matches 语义修订

### 修订前
```json
{
  "matches": [
    {"a": 0, "b": -1, "prob": 0.0}  // 无法区分成交/未成交
  ]
}
```

### 修订后
```json
{
  "matches": [
    {"a": 0, "b": -1, "prob": 0.8, "matched": true}  // 明确标记成交
  ]
}
```

**规则**:
- `matches` 数组只包含**成交的** action
- `matched`: 始终为 `true`（因为未成交的不在数组中）
- `prob`: 如果 core 暴露成交概率，使用真实值；否则为 0.0（TODO: 后续可改进）

---

## Metric（观测指标）

```json
{
  "metric": {
    "s": {
      "price_norm": 0.5,
      "volatility": 0.01,
      "liquidity": 0.8,
      "imbalance": 0.3
    },
    "N": 10,           // 玩家数量
    "avg_exp": 0.85    // 平均经验值
  }
}
```

---

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IG_DEPTH_K` | `25` | 盘口深度档数（bids/asks 各显示多少档） |
| `IG_TF_SEC` | `60.0` | K线时间框（秒） |
| `IG_EMIT_EMPTY_BARS` | `0` | 是否输出空K线（0=否，1=是） |
| `IG_STREAM_MODE` | `full` | 输出模式：`full`（完整）、`frame_only`（仅frame）、`metric_only`（仅metric） |

---

## 输出模式（STREAM_MODE）

### full（完整模式）
包含所有字段：`schema`, `t`, `ts_wall`, `s`, `agents`, `actions`, `matches`, `frame`, `metric`

### frame_only（仅交易展示）
只包含：`schema`, `t`, `ts_wall`, `frame`
- 适用于只需要交易展示的场景（盘口/逐笔/K线）
- 减少网络流量

### metric_only（仅观测指标）
只包含：`schema`, `t`, `ts_wall`, `metric`
- 适用于只需要观测指标的场景
- 减少网络流量

---

## 落盘格式

### 当前实现
- 落盘使用**完整格式**（包含所有字段）
- WebSocket 根据 `STREAM_MODE` 过滤输出
- 保持 append-only 和分段策略不变

### 未来可选升级（D）
- 可以分两类记录：`frame` 和 `metric`
- 同一文件不同 `type`，或分文件存储
- 当前暂不实施，保持兼容

---

## 验收标准

### ✅ 1. Depth 验证
- [x] 每条 tick 都有 `frame.depth`
- [x] `bids`/`asks` 非空时 `mid`/`spread` 有值
- [x] `bids` 按价格降序，`asks` 按价格升序
- [x] 只输出 topK 档（默认 25）

### ✅ 2. Trades 验证
- [x] 有成交时 `frame.trades` 非空
- [x] `trade.price`/`size`/`side` 合法
- [x] `trade_id` 格式正确（`t:<t>-i:<i>`）

### ✅ 3. OHLCV 验证
- [x] `frame.ohlcv` 能随 trades 推进
- [x] `close` 变化，`volume` 累计
- [x] `vwap` 计算正确

### ✅ 4. 前端兼容性
- [x] 前端交易展示软件只靠 `frame` 就能：
  - 画盘口（`depth`）
  - 打逐笔（`trades`）
  - 画K线（`ohlcv`）

### ✅ 5. 回放兼容性
- [x] 落盘文件可回放重建同样的 `depth`/`trades`/`ohlcv`
- [x] 解释层确定性（相同输入产生相同输出）

---

## 注意事项

1. **解释层微结构**: 这是"解释层微结构"，不声称等价真实交易所撮合，但足够支撑展示与回放。

2. **未来升级**: 将来可以将 `external_fill` 升级为 buy/sell pairing 撮合器（仍可保持 core 不动）。

3. **Schema 版本**: `schema.version` 必须加，避免后续升级字段导致前端崩溃。

4. **向后兼容**: 现有字段（`t`, `s`, `agents`, `actions`, `matches`）保持兼容，不会删除。

---

## 使用示例

### JavaScript（浏览器）

```javascript
const ws = new WebSocket('wss://45.76.97.37/ws/stream');

ws.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    if (tick.type === 'tick' && tick.frame) {
        // 画盘口
        const depth = tick.frame.depth;
        console.log(`Best bid: ${depth.best_bid}, Best ask: ${depth.best_ask}, Mid: ${depth.mid}`);
        
        // 打逐笔
        if (tick.frame.trades && tick.frame.trades.length > 0) {
            tick.frame.trades.forEach(trade => {
                console.log(`Trade: ${trade.side} ${trade.size} @ ${trade.price}`);
            });
        }
        
        // 画K线
        if (tick.frame.ohlcv) {
            const ohlcv = tick.frame.ohlcv;
            console.log(`OHLCV: O=${ohlcv.open} H=${ohlcv.high} L=${ohlcv.low} C=${ohlcv.close} V=${ohlcv.volume}`);
        }
    }
};
```

### Python

```python
import asyncio
import websockets
import ssl
import json

ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def connect():
    async with websockets.connect('wss://45.76.97.37/ws/stream', ssl=ssl_context) as ws:
        async for message in ws:
            tick = json.loads(message)
            if tick.get('type') == 'tick' and tick.get('frame'):
                frame = tick['frame']
                
                # 处理 depth
                if frame.get('depth'):
                    depth = frame['depth']
                    print(f"Mid: {depth.get('mid')}, Spread: {depth.get('spread')}")
                
                # 处理 trades
                if frame.get('trades'):
                    for trade in frame['trades']:
                        print(f"Trade: {trade['side']} {trade['size']} @ {trade['price']}")
                
                # 处理 ohlcv
                if frame.get('ohlcv'):
                    ohlcv = frame['ohlcv']
                    print(f"OHLCV: O={ohlcv['open']} H={ohlcv['high']} L={ohlcv['low']} C={ohlcv['close']}")

asyncio.run(connect())
```

---

**最后更新**: 2026-01-23

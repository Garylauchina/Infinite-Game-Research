# Infinite Game 实时数据流 - 快速参考

## API 端点

### WebSocket 实时流
```
ws://45.76.97.37:8000/ws/stream
```

### HTTP API
```
GET /incremental?from_t=<tick>&limit=<count>
GET /health
GET /segments
GET /segment/{filename}
GET /range?t0=<start>&t1=<end>
```

---

## 数据结构

### Tick 数据
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
  "agents": [{"id": 0, "experience": 0.123}],
  "actions": [{"id": 0, "side": "buy", "price": 50000.0, "size": 1.0}],
  "matches": [{"a": 0, "b": -1, "prob": 0.0}]
}
```

---

## 快速测试

### WebSocket (浏览器控制台)
```javascript
const ws = new WebSocket('ws://45.76.97.37:8000/ws/stream');
ws.onmessage = e => console.log(JSON.parse(e.data));
```

### HTTP API
```bash
curl "http://45.76.97.37:8000/incremental?from_t=0&limit=10"
```

---

## 完整文档
- `API_DOCUMENTATION.md` - 完整 API 文档
- `IMPLEMENTATION_SUMMARY.md` - 实现总结

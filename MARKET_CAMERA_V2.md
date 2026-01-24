# Market Camera v2 — Implementation Summary

## Overview

Backend acts as a "camera + recorder": consumes per-tick raw data from the simulator, builds **STRUCTURE FRAMES** via a sliding-window aggregator, streams **metric** + **frame** over WSS, and rolls **metric** / **frame** to disk in segmented JSONL.zst (append-only). **No core_system changes.**

---

## Env Vars

| Variable | Default | Description |
|----------|---------|-------------|
| `IG_CAM_STORE_DIR` | `/data/ig_cam` | Root for frames/ + metrics/ |
| `IG_CAM_W` | `300` | Window size (ticks) |
| `IG_CAM_STRIDE` | `10` | Emit frame every STRIDE ticks |
| `IG_CAM_EDGE_CAP` | `2000` | Max edges per frame |
| `IG_CAM_PROX_EPS_PCT` | `0.003` | Quote proximity epsilon (0.3% of mid) |
| `IG_CAM_SEGMENT_SECONDS` | `300` | Roll segment every 5 min (wall-clock) |
| `IG_CAM_ZSTD_LEVEL` | `3` | zstd level |
| `IG_CAM_METRIC_DOWNSAMPLE` | `1` | Metric every N ticks (1 = every tick) |

---

## WSS (`/ws/stream`)

### 1. `metric` (lightweight, every tick or downsample)

```json
{
  "type": "metric",
  "t": 12345,
  "s": { "price_norm": 0.5, "volatility": 0.01, "liquidity": 0.8, "imbalance": 0.3 },
  "N": 10,
  "avg_exp": 0.85,
  "matches_n": 3
}
```

### 2. `frame` (STRUCTURE, every STRIDE ticks after t >= W)

```json
{
  "type": "frame",
  "t": 12340,
  "window": { "W": 300, "t0": 12041, "t1": 12340, "stride": 10 },
  "nodes": [
    {
      "node_id": "agent:0@t:12041-12340",
      "agent_id": 0,
      "t0": 12041,
      "t1": 12340,
      "features": {
        "buy_count": 5, "sell_count": 2, "none_count": 293,
        "match_count": 1, "match_rate": 0.00333,
        "quote_price_mean": 50123.5, "quote_price_std": 120.1,
        "quote_dist_to_mid_mean": 0.001,
        "experience_start": 0.8, "experience_end": 0.82, "experience_delta": 0.02,
        "dominant_side": "none"
      }
    }
  ],
  "edges": [
    { "src": "agent:0@t:12041-12340", "dst": "agent:1@t:12041-12340", "type": "match", "w": 0.01 },
    { "src": "agent:1@t:12041-12340", "dst": "agent:2@t:12041-12340", "type": "quote_proximity", "w": 0.005 }
  ],
  "events": { "bursts": [], "shocks": [] }
}
```

---

## Storage

- **Directory**: `IG_CAM_STORE_DIR` → `frames/` and `metrics/`
- **Segment files**: `segment_<UTC>_<start_t>.jsonl.zst` (pending → rename on roll)
- **Manifests**: `frames/manifest.json`, `metrics/manifest.json` (append-only)

---

## HTTP (read-only)

- `GET /manifest/frames` → `frames/manifest.json`
- `GET /manifest/metrics` → `metrics/manifest.json`
- `GET /segment/frames/{name}` → raw .zst (e.g. `segment_20260124_120000_1000.jsonl.zst`)
- `GET /segment/metrics/{name}` → raw .zst

---

## Modules

- **`camera_v2.py`**: `SlidingWindowCamera(W, stride, eps_pct, edge_cap)` — `ingest(raw_tick)`, `maybe_build_frame(t)`
- **`recorder.py`**: `ZstdSegmentWriter(kind, store_dir, segment_seconds, level)` — `write_jsonline(obj, current_t)`, `roll_if_needed(current_t)`

---

## Validation Checklist

1. **WSS**: metric every tick (or downsample); frame every STRIDE after t >= W; `frame.window.t0`/`t1` correct; `node_id` includes `t0-t1`; edges ≤ edge_cap, weights in [0,1].
2. **Storage**: segments roll every `SEGMENT_SECONDS`; pending exists for current segment; manifest grows monotonically; zstd decompresses to valid JSONL.
3. **Replay**: frontend can load manifest → fetch segment → decode JSONL.zst → play frames sequentially.

---

## Notes

- **core_system** unchanged.
- If `matches` have `b=-1` mostly, match edges are sparse; quote_proximity edges still provide structure.
- `init_cam_store` removes orphan `*_pending.jsonl.zst` on startup.

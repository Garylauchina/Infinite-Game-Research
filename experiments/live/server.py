import os, time, threading, json, asyncio, hashlib
from pathlib import Path
from datetime import datetime
from collections import deque
import logging
from logging.handlers import RotatingFileHandler

# 线程控制
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fastapi import FastAPI, Query, HTTPException, Path as FPath, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core_system.main import V5MarketSimulator
from core_system.state_engine import MarketState
import numpy as np
import zstandard as zstd

from experiments.live.camera_v2 import SlidingWindowCamera, _matches_pairs, _mid_from_actions
from experiments.live.recorder import ZstdSegmentWriter

# ===========================
# 日志配置
# ===========================
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "infinite-live.log"

logger = logging.getLogger("infinite_live")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=10)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 配置
# ===========================
DATA_DIR = Path(os.environ.get("INFINITE_DATA_DIR", "/data/infinite_game"))
SEGMENTS_DIR = DATA_DIR / "segments"
INDEX_DIR = DATA_DIR / "index"

SEED = int(os.environ.get("IG_SEED", "0"))
ADJUST_INTERVAL = int(os.environ.get("IG_ADJUST_INTERVAL", "2000"))
DOWNSAMPLE = int(os.environ.get("IG_DOWNSAMPLE", "1"))  # 建议 1 或 2
LOG_EVERY_TICKS = int(os.environ.get("IG_LOG_EVERY", "5000"))

# 分段策略（时间或tick数）
SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "300"))  # 5分钟
SEGMENT_TICKS = int(os.environ.get("SEGMENT_TICKS", "200000"))  # 20万tick

ZSTD_LEVEL = int(os.environ.get("ZSTD_LEVEL", "3"))

# 滚动存储配置
DISK_USAGE_THRESHOLD = float(os.environ.get("DISK_USAGE_THRESHOLD", "80.0"))  # 磁盘使用率阈值（%）
DISK_CLEANUP_TARGET = float(os.environ.get("DISK_CLEANUP_TARGET", "70.0"))  # 清理目标使用率（%）
KEEP_MIN_SEGMENTS = int(os.environ.get("KEEP_MIN_SEGMENTS", "10"))  # 至少保留的段数量
KEEP_MIN_DAYS = float(os.environ.get("KEEP_MIN_DAYS", "1.0"))  # 至少保留的天数

# 交易展示帧配置（legacy, kept for reference)
DEPTH_K = int(os.environ.get("IG_DEPTH_K", "25"))
TF_SEC = float(os.environ.get("IG_TF_SEC", "60.0"))
EMIT_EMPTY_BARS = int(os.environ.get("IG_EMIT_EMPTY_BARS", "0"))
STREAM_MODE = os.environ.get("IG_STREAM_MODE", "full")

# Market Camera v2
CAM_STORE_DIR = Path(os.environ.get("IG_CAM_STORE_DIR", "/data/ig_cam"))
CAM_W = int(os.environ.get("IG_CAM_W", "300"))
CAM_STRIDE = int(os.environ.get("IG_CAM_STRIDE", "10"))
CAM_EDGE_CAP = int(os.environ.get("IG_CAM_EDGE_CAP", "2000"))
CAM_PROX_EPS_PCT = float(os.environ.get("IG_CAM_PROX_EPS_PCT", "0.003"))
CAM_SEGMENT_SECONDS = float(os.environ.get("IG_CAM_SEGMENT_SECONDS", "300"))
CAM_ZSTD_LEVEL = int(os.environ.get("IG_CAM_ZSTD_LEVEL", "3"))
CAM_METRIC_DOWNSAMPLE = int(os.environ.get("IG_CAM_METRIC_DOWNSAMPLE", "1"))

# Schema 版本
SCHEMA_VERSION = "structure_tape_v1"
WSS_SCHEMA_NAME = "ig_wss"
WSS_SCHEMA_VERSION = 1

# 全局状态
meta = {"running": False, "start_ts": None, "ticks": 0, "seed": SEED}

# 段写入器状态
current_seg_file = None
current_seg_writer = None
current_seg_path = None
current_seg_pending_path = None
current_seg_start_t = 0
current_seg_start_ts = None
current_seg_lines = 0
current_seg_compressed_bytes = 0
rule_hash = None  # 规则哈希（用于header）

# WebSocket 连接池
websocket_connections = set()
tick_buffer = deque(maxlen=10000)
tick_buffer_lock = threading.Lock()
main_event_loop = None

# Market Camera v2
metric_buffer = deque(maxlen=100)
last_frame = None
cam_buffer_lock = threading.Lock()
camera = None
metric_recorder = None
frame_recorder = None

# OHLCV 聚合器状态（legacy）
class OHLCVAggregator:
    """K线聚合器"""
    def __init__(self, tf_sec: float):
        self.tf_sec = tf_sec
        self.current_bar = None
        self.bar_start_ts = None
    
    def update(self, trade_price: float, trade_size: float, ts_wall: float):
        """更新K线（有成交时调用）"""
        # 检查是否需要开启新bar
        if self.current_bar is None or (self.bar_start_ts is not None and ts_wall - self.bar_start_ts >= self.tf_sec):
            # 开启新bar
            self.current_bar = {
                "open": trade_price,
                "high": trade_price,
                "low": trade_price,
                "close": trade_price,
                "volume": trade_size,
                "trades": 1,
                "vwap": trade_price,
                "vwap_sum": trade_price * trade_size,
                "vwap_size": trade_size
            }
            self.bar_start_ts = ts_wall
        else:
            # 更新当前bar
            self.current_bar["close"] = trade_price
            self.current_bar["high"] = max(self.current_bar["high"], trade_price)
            self.current_bar["low"] = min(self.current_bar["low"], trade_price)
            self.current_bar["volume"] += trade_size
            self.current_bar["trades"] += 1
            self.current_bar["vwap_sum"] += trade_price * trade_size
            self.current_bar["vwap_size"] += trade_size
            if self.current_bar["vwap_size"] > 0:
                self.current_bar["vwap"] = self.current_bar["vwap_sum"] / self.current_bar["vwap_size"]
    
    def get_bar(self, start_t: int, end_t: int, ts_wall: float) -> dict:
        """获取当前bar（格式化输出）"""
        if self.current_bar is None:
            return None
        
        return {
            "tf": self.tf_sec,
            "open": self.current_bar["open"],
            "high": self.current_bar["high"],
            "low": self.current_bar["low"],
            "close": self.current_bar["close"],
            "volume": self.current_bar["volume"],
            "trades": self.current_bar["trades"],
            "vwap": self.current_bar["vwap"],
            "start_t": start_t,
            "end_t": end_t,
            "start_ts": self.bar_start_ts,
            "end_ts": ts_wall
        }

ohlcv_aggregator = OHLCVAggregator(TF_SEC)


def compute_rule_hash():
    """计算规则哈希（简化版：基于关键配置）"""
    rule_str = f"seed={SEED},adjust={ADJUST_INTERVAL},downsample={DOWNSAMPLE}"
    return hashlib.sha256(rule_str.encode()).hexdigest()[:16]


def compute_depth_snapshot(actions_list):
    """B1: 计算盘口快照（depth_snapshot）"""
    bids = {}  # {price: total_size}
    asks = {}  # {price: total_size}
    
    for action in actions_list:
        price = action["price"]
        size = action["size"]
        side = action["side"]
        
        if side == "buy":
            bids[price] = bids.get(price, 0) + size
        elif side == "sell":
            asks[price] = asks.get(price, 0) + size
    
    # 排序并取 topK
    bids_sorted = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:DEPTH_K]
    asks_sorted = sorted(asks.items(), key=lambda x: x[0])[:DEPTH_K]
    
    bids_array = [[float(price), float(size)] for price, size in bids_sorted]
    asks_array = [[float(price), float(size)] for price, size in asks_sorted]
    
    best_bid = bids_array[0][0] if bids_array else None
    best_ask = asks_array[0][0] if asks_array else None
    
    mid = None
    spread = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
    
    return {
        "bids": bids_array,
        "asks": asks_array,
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread
    }


def compute_trades(matches_list, actions_list, depth, t, ts_wall):
    """B2: 计算逐笔成交（trades）"""
    trades = []
    
    for i, match in enumerate(matches_list):
        a_idx = match.get("a", -1)
        if a_idx < 0 or a_idx >= len(actions_list):
            continue
        
        action = actions_list[a_idx]
        matched = match.get("matched", True)  # 默认 True（matches_list 只包含成交的）
        
        if not matched:
            continue  # 跳过未成交的
        
        # 确定成交价：优先用 mid，否则用 action.price
        trade_price = depth.get("mid")
        if trade_price is None:
            trade_price = action["price"]
        
        trade_size = action["size"]
        trade_side = action["side"]
        
        trade = {
            "trade_id": f"t:{t}-i:{i}",
            "t": int(t),
            "ts_wall": ts_wall,
            "side": trade_side,
            "price": float(trade_price),
            "size": float(trade_size),
            "a": a_idx,
            "b": match.get("b", -1),
            "prob": match.get("prob", 0.0),
            "mode": "external_fill"
        }
        trades.append(trade)
    
    return trades


def get_disk_usage():
    """获取磁盘使用率（%）"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(DATA_DIR)
        usage_percent = (used / total) * 100
        return usage_percent, total, used, free
    except Exception as e:
        logger.warning(f"Failed to get disk usage: {e}")
        return None, None, None, None


def cleanup_old_segments():
    """清理旧段文件，释放磁盘空间"""
    global current_seg_start_t
    
    usage_percent, total, used, free = get_disk_usage()
    if usage_percent is None:
        return False, "无法获取磁盘使用率"
    
    if usage_percent < DISK_USAGE_THRESHOLD:
        return False, f"磁盘使用率 {usage_percent:.1f}% < 阈值 {DISK_USAGE_THRESHOLD}%"
    
    logger.warning(f"磁盘使用率 {usage_percent:.1f}% 超过阈值 {DISK_USAGE_THRESHOLD}%，开始清理旧数据...")
    
    # 读取 segments.json
    segments_json = INDEX_DIR / "segments.json"
    if not segments_json.exists():
        return False, "segments.json 不存在"
    
    try:
        with open(segments_json, 'r') as f:
            segments = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read segments.json: {e}")
        return False, f"读取 segments.json 失败: {e}"
    
    if not isinstance(segments, list) or len(segments) == 0:
        return False, "没有可清理的段"
    
    # 按创建时间排序（最旧的在前）
    segments_sorted = sorted(segments, key=lambda x: x.get("created_at", ""))
    
    # 计算保留策略
    cutoff_time = None
    if KEEP_MIN_DAYS > 0:
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(days=KEEP_MIN_DAYS)
    
    deleted_count = 0
    deleted_bytes = 0
    deleted_segments = []
    
    # 确定当前正在写入的段（不能删除）
    current_start_t = current_seg_start_t if current_seg_start_t else 0
    
    # 从最旧的段开始删除，直到达到目标使用率
    for seg in segments_sorted:
        # 检查是否达到目标使用率
        if usage_percent <= DISK_CLEANUP_TARGET:
            break
        
        # 不能删除正在写入的段
        if seg.get("start_t", 0) >= current_start_t:
            continue
        
        # 检查最小保留数量
        remaining_count = len(segments) - deleted_count
        if remaining_count <= KEEP_MIN_SEGMENTS:
            break
        
        # 检查最小保留天数
        if cutoff_time:
            try:
                created_at = datetime.fromisoformat(seg.get("created_at", "").replace("Z", "+00:00"))
                if created_at >= cutoff_time:
                    # 这个段太新，不能删除
                    continue
            except Exception as e:
                logger.warning(f"Failed to parse created_at for segment {seg.get('path', 'unknown')}: {e}")
        
        # 删除段文件
        seg_path = DATA_DIR / seg.get("path", "")
        if seg_path.exists():
            try:
                file_size = seg_path.stat().st_size
                seg_path.unlink()
                deleted_bytes += file_size
                deleted_count += 1
                deleted_segments.append(seg)
                logger.info(f"Deleted segment: {seg_path.name} ({file_size/1024/1024:.2f} MB)")
                
                # 更新使用率估算
                usage_percent = ((used - deleted_bytes) / total) * 100
            except Exception as e:
                logger.error(f"Failed to delete segment {seg_path}: {e}")
        else:
            # 文件不存在，但从索引中移除
            deleted_count += 1
            deleted_segments.append(seg)
            logger.warning(f"Segment file not found: {seg_path}, removing from index")
    
    # 从 segments 列表中移除已删除的段
    if deleted_count > 0:
        for seg in deleted_segments:
            if seg in segments:
                segments.remove(seg)
        
        # 更新 segments.json
        try:
            with open(segments_json, 'w') as f:
                json.dump(segments, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            logger.info(f"清理完成: 删除 {deleted_count} 个段，释放 {deleted_bytes/1024/1024:.2f} MB，"
                       f"剩余 {len(segments)} 个段，磁盘使用率约 {usage_percent:.1f}%")
            
            return True, f"删除 {deleted_count} 个段，释放 {deleted_bytes/1024/1024:.2f} MB"
        except Exception as e:
            logger.error(f"Failed to update segments.json: {e}")
            return False, f"更新索引失败: {e}"
    
    return False, f"无需清理（已保留 {len(segments)} 个段，使用率 {usage_percent:.1f}%）"


def init_data_dir():
    """初始化数据目录（legacy segments）"""
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    segments_json = INDEX_DIR / "segments.json"
    if not segments_json.exists():
        with open(segments_json, 'w') as f:
            json.dump([], f)
    logger.info(f"Data directory initialized: {DATA_DIR}")


def init_cam_store():
    """Market Camera v2: frames/ + metrics/ + manifest.json"""
    CAM_STORE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("frames", "metrics"):
        d = CAM_STORE_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*_pending.jsonl.zst"):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"init_cam_store unlink {f}: {e}")
        m = d / "manifest.json"
        if not m.exists():
            with open(m, "w") as f:
                json.dump([], f)
    logger.info(f"Cam store initialized: {CAM_STORE_DIR}")


def open_segment():
    """打开新的段文件（使用 pending 名称）"""
    global current_seg_file, current_seg_writer, current_seg_path, current_seg_pending_path
    global current_seg_start_t, current_seg_start_ts, current_seg_lines, current_seg_compressed_bytes
    global rule_hash
    
    # 关闭旧段
    if current_seg_writer:
        close_segment()
    
    # 新段文件名（pending）
    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    seg_name = f"segment_{timestamp}_{meta['ticks']}_pending.jsonl.zst"
    current_seg_pending_path = SEGMENTS_DIR / seg_name
    
    # 打开新段（流式压缩）
    current_seg_file = open(current_seg_pending_path, 'wb')
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    current_seg_writer = cctx.stream_writer(current_seg_file)
    
    current_seg_start_t = meta["ticks"]
    current_seg_start_ts = time.time()
    current_seg_lines = 0
    current_seg_compressed_bytes = 0
    rule_hash = compute_rule_hash()
    
    # 写入 segment header
    header = {
        "type": "segment_header",
        "schema": SCHEMA_VERSION,
        "created_at": now.isoformat(),
        "seed": SEED,
        "rule_hash": rule_hash,
        "start_t": current_seg_start_t,
        "fields": {
            "tick": ["t", "s", "agents", "actions", "matches"]
        }
    }
    write_line(header)
    
    logger.info(f"Opened segment: {seg_name} at t={current_seg_start_t}")


def close_segment():
    """关闭当前段：fsync、rename、更新索引"""
    global current_seg_file, current_seg_writer, current_seg_path, current_seg_pending_path
    global current_seg_lines, current_seg_compressed_bytes
    
    if not current_seg_writer:
        return
    
    try:
        # 关闭写入器
        current_seg_writer.close()
    except Exception as e:
        logger.warning(f"Error closing writer: {e}")
    
    try:
        if current_seg_file and not current_seg_file.closed:
            current_seg_file.flush()
            os.fsync(current_seg_file.fileno())
            current_seg_file.close()
    except Exception as e:
        logger.warning(f"Error closing file: {e}")
    
    # 计算文件大小
    file_size = current_seg_pending_path.stat().st_size
    current_seg_compressed_bytes = file_size
    
    # Rename：去掉 _pending（原子提交）
    final_name = current_seg_pending_path.name.replace("_pending", "")
    current_seg_path = current_seg_pending_path.parent / final_name
    current_seg_pending_path.rename(current_seg_path)
    
    # 计算相对路径
    rel_path = current_seg_path.relative_to(DATA_DIR)
    
    # 计算统计信息（简化：从文件读取部分数据）
    end_t = meta["ticks"] - 1
    stats = {
        "ticks": current_seg_lines - 1,  # 减去 header
        "agent_count_mean": 0.0,  # 需要从数据计算
        "match_count_mean": 0.0,
        "match_count_max": 0
    }
    
    # 更新 segments.json
    segments_json = INDEX_DIR / "segments.json"
    segments = []
    if segments_json.exists():
        with open(segments_json, 'r') as f:
            segments = json.load(f)
    
    entry = {
        "path": str(rel_path).replace("\\", "/"),
        "start_t": current_seg_start_t,
        "end_t": end_t,
        "lines": current_seg_lines,
        "bytes": current_seg_compressed_bytes,
        "created_at": datetime.utcnow().isoformat(),
        "stats": stats
    }
    
    segments.append(entry)
    
    with open(segments_json, 'w') as f:
        json.dump(segments, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    logger.info(f"Closed segment: {final_name}, t={current_seg_start_t}..{end_t}, lines={current_seg_lines}, "
                f"compressed={current_seg_compressed_bytes/1024/1024:.2f}MB")
    
    # 检查磁盘使用率，必要时清理旧数据
    try:
        usage_percent, _, _, _ = get_disk_usage()
        if usage_percent is not None:
            logger.debug(f"磁盘使用率: {usage_percent:.1f}%")
            if usage_percent >= DISK_USAGE_THRESHOLD:
                success, msg = cleanup_old_segments()
                if success:
                    logger.info(f"自动清理完成: {msg}")
                else:
                    logger.warning(f"自动清理未执行: {msg}")
    except Exception as e:
        logger.warning(f"磁盘清理检查失败: {e}")
    
    current_seg_writer = None
    current_seg_file = None
    current_seg_path = None
    current_seg_pending_path = None


def should_rotate_segment():
    """判断是否应该切段"""
    if current_seg_start_ts is None:
        return False, None
    
    elapsed = time.time() - current_seg_start_ts
    if elapsed >= SEGMENT_SECONDS:
        return True, "max_seconds"
    
    if meta["ticks"] - current_seg_start_t >= SEGMENT_TICKS:
        return True, "max_ticks"
    
    return False, None


def build_raw_tick(t, s_t, agents, actions, match_indices_actions):
    """Build raw_tick for camera: {t, mid, agents, actions, matches}."""
    agents_map = {}
    actions_map = {}
    for i, p in enumerate(agents):
        agents_map[i] = float(p.experience_score)
    for i, ac in enumerate(actions):
        actions_map[i] = {
            "side": ac.side,
            "price": float(ac.price),
            "size": float(ac.qty) if hasattr(ac, "qty") else 1.0,
        }
    mid = _mid_from_actions(actions_map)
    if mid is None and camera is not None and getattr(camera, "_last_mid", None) is not None:
        mid = camera._last_mid
    matches_pairs = []
    for idx, _ in match_indices_actions:
        matches_pairs.append((idx, -1))  # b=-1; match edges sparse per spec
    return {
        "t": int(t),
        "mid": mid,
        "agents": agents_map,
        "actions": actions_map,
        "matches": matches_pairs,
    }


def cam_tick(t, s_t, agents, actions, matches, match_indices_actions):
    """Market Camera v2: ingest raw_tick, emit metric + frame, persist, broadcast."""
    global camera, metric_recorder, frame_recorder, last_frame, metric_buffer

    if camera is None or metric_recorder is None or frame_recorder is None:
        return

    ti = int(t)
    try:
        metric_recorder.roll_if_needed(ti)
        frame_recorder.roll_if_needed(ti)
    except Exception as e:
        logger.debug(f"roll_if_needed: {e}")

    raw = build_raw_tick(t, s_t, agents, actions, match_indices_actions)
    camera.ingest(raw)
    matches_n = len(match_indices_actions)

    metric_msg = {
        "type": "metric",
        "t": int(t),
        "s": {
            "price_norm": float(s_t.price_norm),
            "volatility": float(s_t.volatility),
            "liquidity": float(s_t.liquidity),
            "imbalance": float(s_t.imbalance),
        },
        "N": len(agents),
        "avg_exp": float(np.mean([p.experience_score for p in agents])) if agents else 0.0,
        "matches_n": matches_n,
    }

    if ti % CAM_METRIC_DOWNSAMPLE == 0:
        try:
            metric_recorder.write_jsonline(metric_msg, ti)
        except Exception as e:
            logger.warning(f"metric_recorder write: {e}")
        if websocket_connections and main_event_loop and main_event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(broadcast_msg(metric_msg), main_event_loop)
            except Exception as e:
                logger.debug(f"broadcast metric: {e}")
        with cam_buffer_lock:
            metric_buffer.append(metric_msg)

    frame = camera.maybe_build_frame(ti)
    if frame is not None:
        try:
            frame_recorder.write_jsonline(frame, ti)
        except Exception as e:
            logger.warning(f"frame_recorder write: {e}")
        if websocket_connections and main_event_loop and main_event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(broadcast_msg(frame), main_event_loop)
            except Exception as e:
                logger.debug(f"broadcast frame: {e}")
        with cam_buffer_lock:
            last_frame = frame


def write_line(obj: dict):
    """写入一行 JSON"""
    global current_seg_lines
    
    if not current_seg_writer:
        return
    
    try:
        line = json.dumps(obj, ensure_ascii=False) + '\n'
        line_bytes = line.encode('utf-8')
        
        current_seg_writer.write(line_bytes)
        current_seg_lines += 1
    except (ValueError, OSError) as e:
        logger.error(f"Failed to write line: {e}, reopening segment")
        # 尝试重新打开段
        if current_seg_writer:
            try:
                current_seg_writer.close()
            except:
                pass
        open_segment()
        # 重试写入
        if current_seg_writer:
            line = json.dumps(obj, ensure_ascii=False) + '\n'
            line_bytes = line.encode('utf-8')
            current_seg_writer.write(line_bytes)
            current_seg_lines += 1


def write_tick_record(t, s_t, agents, actions, matches, match_indices_actions):
    """写入一个 tick 记录（升级版：包含 frame 和 metric）"""
    global ohlcv_aggregator
    ts_wall = time.time()
    
    # 构建 agents 列表
    agents_list = []
    for i, player in enumerate(agents):
        agents_list.append({
            "id": player.id,
            "experience": float(player.experience_score)
        })
    
    # 构建 actions 列表
    actions_list = []
    for i, action in enumerate(actions):
        actions_list.append({
            "id": i,
            "side": action.side,
            "price": float(action.price),
            "size": float(action.qty) if hasattr(action, 'qty') else 1.0
        })
    
    # C1: 修订 matches 语义，添加 matched 字段
    match_indices_set = {idx for idx, _ in match_indices_actions}
    matches_list = []
    
    # 只包含成交的 action（match_indices_actions 本身就是成交集合）
    for idx, action in match_indices_actions:
        matches_list.append({
            "a": idx,
            "b": -1,  # 默认 -1 表示外部流动性池
            "prob": 0.0,  # TODO: 如果 core 暴露 prob，这里应该用真实值
            "matched": True  # 明确标记为成交
        })
    
    # B1: 计算 depth_snapshot
    depth = compute_depth_snapshot(actions_list)
    
    # B2: 计算 trades
    trades = compute_trades(matches_list, actions_list, depth, t, ts_wall)
    
    # B3: 更新 ohlcv（有成交时）
    ohlcv = None
    if trades:
        for trade in trades:
            ohlcv_aggregator.update(trade["price"], trade["size"], ts_wall)
        ohlcv = ohlcv_aggregator.get_bar(t, t, ts_wall)
    elif EMIT_EMPTY_BARS:
        # 输出空K线（用 mid 填充）
        if depth.get("mid") is not None:
            ohlcv = {
                "tf": TF_SEC,
                "open": depth["mid"],
                "high": depth["mid"],
                "low": depth["mid"],
                "close": depth["mid"],
                "volume": 0.0,
                "trades": 0,
                "vwap": depth["mid"],
                "start_t": t,
                "end_t": t,
                "start_ts": ts_wall,
                "end_ts": ts_wall
            }
    
    # 构建 frame
    frame = {
        "depth": depth,
        "trades": trades
    }
    if ohlcv is not None:
        frame["ohlcv"] = ohlcv
    
    # 构建 metric（观测指标）
    metric = {
        "s": {
            "price_norm": float(s_t.price_norm),
            "volatility": float(s_t.volatility),
            "liquidity": float(s_t.liquidity),
            "imbalance": float(s_t.imbalance)
        },
        "N": len(agents),
        "avg_exp": float(np.mean([p.experience_score for p in agents])) if agents else 0.0
    }
    
    # 构建完整的 tick_record
    tick_record = {
        "type": "tick",
        "schema": {
            "name": WSS_SCHEMA_NAME,
            "version": WSS_SCHEMA_VERSION
        },
        "t": int(t),
        "ts_wall": ts_wall,
        "s": {
            "price_norm": float(s_t.price_norm),
            "volatility": float(s_t.volatility),
            "liquidity": float(s_t.liquidity),
            "imbalance": float(s_t.imbalance)
        },
        "agents": agents_list,
        "actions": actions_list,
        "matches": matches_list,
        "frame": frame,
        "metric": metric
    }
    
    # 根据 STREAM_MODE 过滤输出（仅用于 WebSocket，落盘保持完整）
    ws_tick_record = tick_record.copy()
    if STREAM_MODE == "frame_only":
        # 只保留 frame 相关字段
        ws_tick_record = {
            "type": "tick",
            "schema": tick_record["schema"],
            "t": tick_record["t"],
            "ts_wall": tick_record["ts_wall"],
            "frame": tick_record["frame"]
        }
    elif STREAM_MODE == "metric_only":
        # 只保留 metric 相关字段
        ws_tick_record = {
            "type": "tick",
            "schema": tick_record["schema"],
            "t": tick_record["t"],
            "ts_wall": tick_record["ts_wall"],
            "metric": tick_record["metric"]
        }
    
    # 落盘：写入完整记录（包含所有字段）
    write_line(tick_record)
    
    # 添加到缓冲区（使用 WebSocket 过滤后的版本）
    with tick_buffer_lock:
        tick_buffer.append(ws_tick_record)
    
    # 异步广播到所有 WebSocket 连接（在后台线程中调用）
    tick_t = ws_tick_record.get('t', 'unknown')
    
    # 每 1000 个 tick 记录一次状态（用于调试）
    if tick_t != 'unknown' and int(tick_t) % 1000 == 0:
        logger.info(f"write_tick_record t={tick_t}, websocket_connections={len(websocket_connections)}, main_event_loop={main_event_loop is not None}")
    
    if websocket_connections:
        if main_event_loop is None:
            logger.warning(f"main_event_loop is None, cannot broadcast tick t={tick_t}. WebSocket may not be connected yet.")
        else:
            try:
                # 使用保存的事件循环引用
                if main_event_loop.is_running():
                    # 如果循环正在运行，使用 run_coroutine_threadsafe
                    future = asyncio.run_coroutine_threadsafe(broadcast_tick(ws_tick_record), main_event_loop)
                    # 不等待 future，避免阻塞后台线程
                    # 每 1000 个 tick 记录一次广播调度
                    if tick_t != 'unknown' and int(tick_t) % 1000 == 0:
                        logger.info(f"Scheduled broadcast for tick t={tick_t} via run_coroutine_threadsafe")
                    
                    # 检查 future 是否有异常（不阻塞，只检查是否完成）
                    # 注意：这不会等待 future 完成，只是检查状态
                    try:
                        if future.done():
                            exc = future.exception()
                            if exc:
                                logger.warning(f"Broadcast future for tick t={tick_t} raised exception: {exc}")
                    except Exception:
                        pass  # 忽略检查异常
                else:
                    # 如果循环未运行，直接运行（不应该发生，但保险起见）
                    logger.warning(f"main_event_loop is not running for tick t={tick_t}, attempting to run broadcast_tick directly")
                    try:
                        main_event_loop.run_until_complete(broadcast_tick(ws_tick_record))
                    except Exception as e2:
                        logger.error(f"Failed to run broadcast_tick directly for tick t={tick_t}: {e2}", exc_info=True)
            except Exception as e:
                logger.warning(f"Failed to broadcast tick t={tick_t}: {e}", exc_info=True)
    else:
        # 每 10000 个 tick 记录一次（避免日志过多）
        if tick_t != 'unknown' and int(tick_t) % 10000 == 0:
            logger.debug(f"No WebSocket connections for tick t={tick_t}")


def run_forever():
    """主运行循环 — Market Camera v2"""
    global meta, camera, metric_recorder, frame_recorder

    meta["running"] = True
    meta["start_ts"] = time.time()

    init_cam_store()
    camera = SlidingWindowCamera(
        W=CAM_W,
        stride=CAM_STRIDE,
        eps_pct=CAM_PROX_EPS_PCT,
        edge_cap=CAM_EDGE_CAP,
    )
    metric_recorder = ZstdSegmentWriter(
        "metrics",
        CAM_STORE_DIR,
        segment_seconds=CAM_SEGMENT_SECONDS,
        level=CAM_ZSTD_LEVEL,
    )
    frame_recorder = ZstdSegmentWriter(
        "frames",
        CAM_STORE_DIR,
        segment_seconds=CAM_SEGMENT_SECONDS,
        level=CAM_ZSTD_LEVEL,
    )

    np.random.seed(SEED)
    import random
    random.seed(SEED)

    sim = V5MarketSimulator(ticks=1, adjust_interval=ADJUST_INTERVAL, MAX_N=None)
    s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
    t = 0

    logger.info(f"Simulation started: seed={SEED}, cam W={CAM_W} stride={CAM_STRIDE}")

    while True:
        try:
            sim.chaos_factor.update_tick(t)

            actions = [p.decide_action(s_t) for p in sim.active_players]
            match_indices_actions = sim.sample_matches(actions, s_t)
            match_indices = {idx for idx, _ in match_indices_actions}
            matches = [action for _, action in match_indices_actions]

            for i, player in enumerate(sim.active_players):
                matched = i in match_indices
                player.update_experience(matched, s_t)

            s_t = sim.engine.update(s_t, actions, matches)

            state_features = np.array([s_t.price_norm, s_t.volatility, s_t.liquidity, s_t.imbalance])
            sim.structure_metrics.update_trajectory(state_features)

            cam_tick(t, s_t, sim.active_players, actions, matches, match_indices_actions)

            if t % ADJUST_INTERVAL == 0 and t > 0:
                sim.adjust_participation()

            if (t % LOG_EVERY_TICKS) == 0:
                N_log = len(sim.active_players)
                avg_exp_log = float(np.mean([p.experience_score for p in sim.active_players])) if sim.active_players else 0.0
                logger.info(f"t={t} N={N_log} exp={avg_exp_log:.3f} matches={len(matches)}")

            meta["ticks"] = int(t)
            t += 1

            if t % 50 == 0:
                time.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in simulation loop at t={t}: {e}", exc_info=True)


# ===========================
# API 端点
# ===========================

@app.get("/health")
def health():
    """健康检查"""
    return JSONResponse({
        "ok": True,
        "meta": meta,
        "current_segment": {
            "start_t": current_seg_start_t,
            "lines": current_seg_lines,
            "compressed_bytes": current_seg_compressed_bytes,
            "pending": current_seg_pending_path is not None
        } if current_seg_writer else None
    })


@app.get("/disk")
def get_disk_status():
    """获取磁盘使用情况和清理配置"""
    usage_percent, total, used, free = get_disk_usage()
    
    # 读取 segments.json 统计
    segments_json = INDEX_DIR / "segments.json"
    segment_count = 0
    total_segment_bytes = 0
    if segments_json.exists():
        try:
            with open(segments_json, 'r') as f:
                segments = json.load(f)
            if isinstance(segments, list):
                segment_count = len(segments)
                total_segment_bytes = sum(s.get("bytes", 0) for s in segments)
        except Exception as e:
            logger.warning(f"Failed to read segments.json for disk status: {e}")
    
    return JSONResponse({
        "disk": {
            "usage_percent": round(usage_percent, 2) if usage_percent is not None else None,
            "total_gb": round(total / (1024**3), 2) if total else None,
            "used_gb": round(used / (1024**3), 2) if used else None,
            "free_gb": round(free / (1024**3), 2) if free else None,
        },
        "segments": {
            "count": segment_count,
            "total_bytes": total_segment_bytes,
            "total_gb": round(total_segment_bytes / (1024**3), 2),
        },
        "cleanup_config": {
            "threshold_percent": DISK_USAGE_THRESHOLD,
            "target_percent": DISK_CLEANUP_TARGET,
            "keep_min_segments": KEEP_MIN_SEGMENTS,
            "keep_min_days": KEEP_MIN_DAYS,
        },
        "status": {
            "needs_cleanup": usage_percent is not None and usage_percent >= DISK_USAGE_THRESHOLD,
        }
    })


@app.get("/segments")
def get_segments():
    """获取 segments.json"""
    segments_json = INDEX_DIR / "segments.json"
    segments = []
    
    if segments_json.exists():
        try:
            with open(segments_json, 'r') as f:
                content = f.read().strip()
                if content:
                    segments = json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to read segments.json: {e}")
            segments = []
    
    # 确保返回列表
    if not isinstance(segments, list):
        segments = []
    
    return JSONResponse(segments)


@app.get("/segment/{filename:path}")
def get_segment(filename: str, format: str = Query("zst", description="Format: zst or jsonl")):
    """获取段文件"""
    # 处理路径：可能是 "segments/xxx" 或 "xxx"
    if filename.startswith("segments/"):
        # 从 segments.json 来的路径，相对于 DATA_DIR
        seg_path = (DATA_DIR / filename).resolve()
        if not str(seg_path).startswith(str(DATA_DIR.resolve())):
            raise HTTPException(403, "Path traversal not allowed")
    else:
        # 直接文件名，相对于 SEGMENTS_DIR
        seg_path = (SEGMENTS_DIR / filename).resolve()
        if not str(seg_path).startswith(str(SEGMENTS_DIR.resolve())):
            raise HTTPException(403, "Path traversal not allowed")
    
    if not seg_path.exists():
        raise HTTPException(404, f"Segment not found: {filename}")
    
    if format == "jsonl":
        # 解压并返回 NDJSON 流
        def generate():
            dctx = zstd.ZstdDecompressor()
            with open(seg_path, 'rb') as f:
                reader = dctx.stream_reader(f)
                buffer = b""
                while True:
                    chunk = reader.read(8192)
                    if not chunk:
                        if buffer.strip():
                            yield buffer
                        break
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line.strip():
                            yield line + b'\n'
        
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    else:
        # 返回压缩文件
        return FileResponse(seg_path, media_type="application/zstd", filename=seg_path.name)


# Market Camera v2 — manifest / segment (read-only)
@app.get("/manifest/frames")
def get_manifest_frames():
    """返回 frames/manifest.json"""
    p = CAM_STORE_DIR / "frames" / "manifest.json"
    if not p.exists():
        return JSONResponse([])
    try:
        with open(p) as f:
            data = json.load(f)
        return JSONResponse(data if isinstance(data, list) else [])
    except Exception as e:
        logger.warning(f"manifest/frames: {e}")
        return JSONResponse([])


@app.get("/manifest/metrics")
def get_manifest_metrics():
    """返回 metrics/manifest.json"""
    p = CAM_STORE_DIR / "metrics" / "manifest.json"
    if not p.exists():
        return JSONResponse([])
    try:
        with open(p) as f:
            data = json.load(f)
        return JSONResponse(data if isinstance(data, list) else [])
    except Exception as e:
        logger.warning(f"manifest/metrics: {e}")
        return JSONResponse([])


@app.get("/segment/frames/{name:path}")
def get_segment_frames(name: str):
    """返回 frames 目录下 raw .zst 文件 (application/octet-stream)"""
    path = (CAM_STORE_DIR / "frames" / name).resolve()
    if not path.is_file() or not str(path).startswith(str((CAM_STORE_DIR / "frames").resolve())):
        raise HTTPException(404, f"Segment not found: {name}")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/segment/metrics/{name:path}")
def get_segment_metrics(name: str):
    """返回 metrics 目录下 raw .zst 文件 (application/octet-stream)"""
    path = (CAM_STORE_DIR / "metrics" / name).resolve()
    if not path.is_file() or not str(path).startswith(str((CAM_STORE_DIR / "metrics").resolve())):
        raise HTTPException(404, f"Segment not found: {name}")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


async def broadcast_msg(msg: dict):
    """广播任意 JSON 消息到所有 WebSocket 连接（metric / frame）"""
    global websocket_connections

    if not websocket_connections:
        return
    mt = msg.get("type", "?")
    t = msg.get("t", "?")
    disconnected = set()
    for ws in websocket_connections.copy():
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning(f"WebSocket send {mt} t={t}: {e}")
            disconnected.add(ws)
    if disconnected:
        websocket_connections -= disconnected


async def broadcast_tick(tick_data: dict):
    """Legacy; 现用 broadcast_msg 发送 metric/frame"""
    await broadcast_msg(tick_data)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket 实时数据流"""
    global main_event_loop
    
    await websocket.accept()
    websocket_connections.add(websocket)
    
    # 保存事件循环引用（用于后台线程调用）
    # 重要：必须在 WebSocket 连接时设置，以便后台线程可以广播
    try:
        current_loop = asyncio.get_event_loop()
        if main_event_loop is None or main_event_loop != current_loop:
            main_event_loop = current_loop
            logger.info(f"Set main_event_loop for WebSocket broadcasting. Loop running: {main_event_loop.is_running()}")
    except RuntimeError as e:
        logger.error(f"Failed to get event loop: {e}")
        # 如果没有事件循环，创建一个新的（仅用于测试）
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        logger.warning("Created new event loop (this should not happen in production)")
    
    logger.info(f"WebSocket connected. Total connections: {len(websocket_connections)}, main_event_loop set: {main_event_loop is not None}")
    
    try:
        # 发送最新 metric + frame（若有）
        with cam_buffer_lock:
            if last_frame is not None:
                await websocket.send_json(last_frame)
            for m in list(metric_buffer)[-3:]:
                await websocket.send_json(m)
        
        # 保持连接，等待服务器推送
        while True:
            # 接收客户端消息（用于保持连接或发送控制命令）
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # 可以处理客户端命令，如请求历史数据
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # 发送心跳保持连接
                await websocket.send_json({"type": "heartbeat", "t": meta.get("ticks", 0)})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}", exc_info=True)
    finally:
        websocket_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(websocket_connections)}")


@app.get("/incremental")
def get_incremental(from_t: int = Query(0, description="Start tick (inclusive)"), 
                    limit: int = Query(1000, description="Maximum number of ticks to return")):
    """获取增量历史数据（从指定 tick 开始）"""
    result = []
    
    with tick_buffer_lock:
        for tick_data in tick_buffer:
            if tick_data["t"] > from_t:
                result.append(tick_data)
                if len(result) >= limit:
                    break
    
    # 如果缓冲区中没有足够的数据，尝试从段文件读取
    if len(result) < limit and from_t < meta.get("ticks", 0):
        # 从当前段文件读取（如果可能）
        # 这里简化处理，实际可以从段文件读取
        pass
    
    return JSONResponse({
        "from_t": from_t,
        "count": len(result),
        "latest_t": meta.get("ticks", 0),
        "data": result
    })


@app.get("/range")
def get_range(t0: int = Query(..., description="Start tick"), 
              t1: int = Query(..., description="End tick")):
    """按时间范围返回数据（可选）"""
    segments_json = INDEX_DIR / "segments.json"
    if not segments_json.exists():
        return JSONResponse([])
    
    with open(segments_json, 'r') as f:
        segments = json.load(f)
    
    # 找到覆盖时间范围的段
    relevant_segments = [s for s in segments if s["start_t"] <= t1 and s["end_t"] >= t0]
    relevant_segments.sort(key=lambda x: x["start_t"])
    
    def generate():
        dctx = zstd.ZstdDecompressor()
        for seg in relevant_segments:
            seg_path = DATA_DIR / seg["path"]
            if not seg_path.exists():
                continue
            
            with open(seg_path, 'rb') as f:
                reader = dctx.stream_reader(f)
                buffer = b""
                for chunk in iter(lambda: reader.read(8192), b""):
                    if not chunk:
                        break
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line.strip():
                            try:
                                obj = json.loads(line.decode('utf-8'))
                                if obj.get("type") == "tick" and t0 <= obj.get("t", 0) <= t1:
                                    yield json.dumps(obj, ensure_ascii=False) + '\n'
                            except:
                                pass
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


# 静态文件服务（已禁用，前端通过 nginx 提供）
# frontend_dir = Path(__file__).parent / "frontend"
# if frontend_dir.exists():
#     app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/")
def index():
    """API 根路径"""
    return JSONResponse({
        "message": "Infinite Game API",
        "version": "2.0",
        "cam_store": str(CAM_STORE_DIR),
        "endpoints": {
            "health": "/health",
            "disk": "/disk",
            "manifest_frames": "/manifest/frames",
            "manifest_metrics": "/manifest/metrics",
            "segment_frames": "/segment/frames/{name}",
            "segment_metrics": "/segment/metrics/{name}",
            "segments": "/segments",
            "segment": "/segment/{filename}",
            "incremental": "/incremental?from_t=&limit=",
            "range": "/range?t0=<start>&t1=<end>",
            "websocket": "/ws/stream",
            "docs": "/docs"
        }
    })


if __name__ == "__main__":
    init_data_dir()
    init_cam_store()
    
    th = threading.Thread(target=run_forever, daemon=True)
    th.start()
    
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False, log_level="warning")

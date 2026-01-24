#!/usr/bin/env python3
import os, time, json
from pathlib import Path
from collections import deque
import numpy as np

# 重要：不改核心代码，只是import并复用其内部组件
# 注意：实际代码在 core_system 目录，不是 src.v5
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from core_system.main import V5MarketSimulator
from core_system.state_engine import MarketState

RUN_DIR = Path(os.environ.get("IG_RUN_DIR", "runs/continuous_world"))
RUN_DIR = Path(__file__).parent.parent.parent / RUN_DIR
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 参数（可通过环境变量覆盖）
ADJUST_INTERVAL = int(os.environ.get("IG_ADJUST_INTERVAL", "2000"))
LOG_EVERY_TICKS  = int(os.environ.get("IG_LOG_EVERY", "5000"))
FLUSH_EVERY_TICKS = int(os.environ.get("IG_FLUSH_EVERY", "20000"))
RING_SIZE = int(os.environ.get("IG_RING_SIZE", "20000"))  # 内存里最多保留多少个点用于网页
SLEEP_SEC = float(os.environ.get("IG_SLEEP_SEC", "0"))     # 需要降速就调大

# 轨迹降采样（给网页用）
DOWNSAMPLE = int(os.environ.get("IG_DOWNSAMPLE", "10"))

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def main():
    run_id = os.environ.get("IG_RUN_ID", f"world_{now_ts()}")
    out_dir = RUN_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 记录元信息
    meta = {
        "run_id": run_id,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adjust_interval": ADJUST_INTERVAL,
        "log_every_ticks": LOG_EVERY_TICKS,
        "flush_every_ticks": FLUSH_EVERY_TICKS,
        "ring_size": RING_SIZE,
        "downsample": DOWNSAMPLE,
        "env_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 初始化模拟器（ticks 不再作为终止条件使用）
    sim = V5MarketSimulator(ticks=1, adjust_interval=ADJUST_INTERVAL, MAX_N=None)

    # 同一世界的"唯一"状态：放在 wrapper 外部持久化
    s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)

    # ring buffer：给网页实时读取（避免内存爆）
    ring = deque(maxlen=RING_SIZE)
    # 落盘累计（批量写入）
    batch = []

    t = 0
    last_flush = 0

    # 输出文件
    metrics_path = out_dir / "metrics.ndjson"   # 关键指标流（每次 flush 写一批）
    stream_path  = out_dir / "stream.json"      # 网页读取的最新窗口（覆盖写）
    state_path   = out_dir / "latest_state.json"

    def emit_point(t, s, N, avg_exp):
        # 统一输出结构：网页/后处理都可复用
        return {
            "t": int(t),
            "price_norm": float(s.price_norm),
            "volatility": float(s.volatility),
            "liquidity": float(s.liquidity),
            "imbalance": float(s.imbalance),
            "complexity": float(s.complexity),
            "N": int(N),
            "avg_exp": float(avg_exp),
        }

    while True:
        # ========== 单步推进（完全复用核心组件） ==========
        sim.chaos_factor.update_tick(t)

        actions = [p.decide_action(s_t) for p in sim.active_players]

        match_indices_actions = sim.sample_matches(actions, s_t)
        match_indices = {idx for idx, _ in match_indices_actions}
        matches = [action for _, action in match_indices_actions]

        for i, player in enumerate(sim.active_players):
            matched = i in match_indices
            player.update_experience(matched, s_t)

        s_t = sim.engine.update(s_t, actions, matches)

        # 结构密度：沿用 sim 的 StructureMetrics
        state_features = np.array([s_t.price_norm, s_t.volatility, s_t.liquidity, s_t.imbalance])
        sim.structure_metrics.update_trajectory(state_features)

        if t % 500 == 0 and t >= 200:
            cpx = sim.structure_metrics.compute_complexity()
            s_t.complexity = cpx

        # 记录（只做轻量级：网页 ring + 批量落盘）
        avg_exp = float(np.mean([p.experience_score for p in sim.active_players])) if sim.active_players else 0.0
        N = len(sim.active_players)
        if (t % DOWNSAMPLE) == 0:
            pt = emit_point(t, s_t, N, avg_exp)
            ring.append(pt)
            batch.append(pt)

        # 参与度调整：沿用核心逻辑（不改）
        if t % ADJUST_INTERVAL == 0 and t > 0:
            sim.adjust_participation()

        # 日志（stdout 交给 systemd/journal 或 nohup）
        if t % LOG_EVERY_TICKS == 0:
            print(f"t={t} N={N} exp={avg_exp:.3f} liq={s_t.liquidity:.3f} cpx={s_t.complexity:.3f}", flush=True)

        # 周期 flush：写 ndjson + 写一个最新 stream.json（供网页拉取）
        if (t - last_flush) >= FLUSH_EVERY_TICKS and batch:
            with metrics_path.open("a", encoding="utf-8") as f:
                for row in batch:
                    f.write(json.dumps(row) + "\n")
            batch.clear()
            last_flush = t

            # 给网页：写当前窗口（覆盖）
            stream_payload = {"run_id": run_id, "t": int(t), "window": list(ring)}
            stream_path.write_text(json.dumps(stream_payload), encoding="utf-8")

            # 最新状态（覆盖）
            state_path.write_text(json.dumps(emit_point(t, s_t, N, avg_exp)), encoding="utf-8")

        t += 1
        if SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

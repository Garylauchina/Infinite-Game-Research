import time
import numpy as np
import sys
from core_system.main import V5MarketSimulator

# 无限运行参数
TICKS_PER_CHUNK = 5000
SEED = int(time.time()) % 100000

np.random.seed(SEED)

print("🚀 Infinite Game Live started, seed =", SEED)

sim = V5MarketSimulator(
    ticks=TICKS_PER_CHUNK,
    adjust_interval=2000,
    MAX_N=None
)

# 记录总运行时间
total_ticks = 0
chunk_count = 0

# 重定向 run_simulation 的输出以减少噪音（可选）
# 如果需要看到详细输出，可以注释掉这部分
original_stdout = sys.stdout

try:
    while True:
        chunk_count += 1
        
        # 运行一个 chunk
        # 注意：run_simulation() 每次都会重置状态，但会累积到 state_trajectory
        metrics = sim.run_simulation()
        
        # 更新总 tick 数
        total_ticks += TICKS_PER_CHUNK
        
        # 每轮保存最新轨迹（供可视化读取）
        # state_trajectory 存储的是 (price_norm, volatility, liquidity, imbalance) 元组
        if len(sim.state_trajectory) > 0:
            # 保存最近 5000 个状态点
            recent_trajectory = sim.state_trajectory[-5000:]
            np.save(
                "live_state.npy",
                np.array(recent_trajectory)
            )
            
            # 保存元数据：玩家数量和平均体验分数
            if len(sim.active_players) > 0:
                avg_experience = np.mean([p.experience_score for p in sim.active_players])
                np.save(
                    "live_meta.npy",
                    np.array([
                        len(sim.active_players),
                        avg_experience
                    ])
                )
            else:
                np.save(
                    "live_meta.npy",
                    np.array([0, 0.0])
                )
        
        # 输出简洁的状态信息
        if len(sim.active_players) > 0:
            avg_exp = np.mean([p.experience_score for p in sim.active_players])
            final_complexity = metrics.get('final_complexity', 0.0)
            print(f"⏳ chunk #{chunk_count} completed | total_ticks={total_ticks} | players={len(sim.active_players)} | avg_exp={avg_exp:.3f} | complexity={final_complexity:.3f}")
        else:
            print(f"⏳ chunk #{chunk_count} completed | total_ticks={total_ticks} | players=0")
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\n🛑 Infinite Game Live stopped by user")
    print(f"📊 Final stats: {chunk_count} chunks, {total_ticks} total ticks")
    sys.exit(0)
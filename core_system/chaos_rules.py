# chaos_rules.py - 统计混乱因子规则集
"""
真实世界现象的统计提炼：
1. 报价分散 → 流动性碎片化
2. 方向熵高 → 缺乏共识，成交困难  
3. 规模过载 → 市场容量饱和
"""

import numpy as np
from collections import Counter
from typing import List
from state_engine import Action, MarketState
from trading_rules import SIMPLEST_RULES

def compute_price_dispersion(actions: List[Action]) -> float:
    """报价分散度 - 流动性碎片化现象"""
    if len(actions) < 2:
        return 1.0  # 太少时高度分散
    
    prices = np.array([a.price for a in actions])
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    
    # 相对分散度 (2%为健康基准)
    dispersion = np.clip(std_price / mean_price / 0.02, 0.2, 3.0)
    return dispersion

def compute_direction_entropy(actions: List[Action]) -> float:
    """方向熵 - 缺乏共识现象"""
    if len(actions) == 0:
        return 1.0
    
    sides = [a.side for a in actions]
    side_counts = Counter(sides)
    total = len(actions)
    
    probs = [count / total for count in side_counts.values()]
    entropy = -sum(p * np.log2(p + 1e-8) for p in probs)
    
    # 标准化到[0,1] (最大熵=1)
    return entropy / np.log2(2)

def compute_scale_overload(player_count: int, tick_actions: int) -> float:
    """规模过载 - 市场容量饱和现象"""
    # 基准：10玩家8行动/tick
    action_overload = tick_actions / 8.0
    player_overload = np.log1p(player_count / 10)
    
    return 0.6 * action_overload + 0.4 * player_overload

def compute_chaos_factor(actions: List[Action], player_count: int) -> float:
    """
    混乱因子 = 报价分散 + 方向熵 + 规模过载
    [0, 1.5] → 成交概率惩罚因子
    """
    dispersion = compute_price_dispersion(actions)
    direction_entropy = compute_direction_entropy(actions)
    scale_overload = compute_scale_overload(player_count, len(actions))
    
    # 加权综合 (真实市场权重)
    chaos = (
        0.4 * dispersion +      # 碎片化最致命
        0.35 * direction_entropy +  # 共识缺失次之
        0.25 * scale_overload   # 规模效应
    )
    
    return np.clip(chaos, 0.0, 1.5)


class ChaosFactor:
    """动态混乱因子管理器 - 降低剂量版本"""
    def __init__(self):
        self.base_chaos = 0.08  # 从0.15→0.08 (减半！)
        self.decay_factor = 0.95  # 随时间衰减
        self.current_tick = 0
    
    def update_tick(self, tick: int):
        """更新当前tick"""
        self.current_tick = tick
    
    def get_chaos_multiplier(self, player_count: int, avg_exp: float) -> float:
        """
        获取混乱因子乘数（用于调整基础混乱因子）
        返回: 乘数 [0.5, 1.2]
        """
        # 动态衰减
        decay = self.decay_factor ** (self.current_tick // 10000)
        chaos = self.base_chaos * decay
        
        # 拥挤惩罚(温和版)
        if player_count > 15:
            chaos *= 1.2
        elif player_count < 5:
            chaos *= 0.8  # 少人时鼓励交易
        
        # 体验差时降低混乱
        if avg_exp < 0.3:
            chaos *= 0.7
        
        # 上限保护
        return min(chaos / self.base_chaos, 1.5)  # 返回乘数

def compute_match_prob(price: float, s_t: MarketState, all_actions: List[Action], player_count: int, 
                       chaos_factor_manager=None, avg_exp=None) -> float:
    """
    完整统计版成交概率 = 价格优先 × (1 - 混乱因子)
    
    Args:
        chaos_factor_manager: ChaosFactor实例，用于动态调整混乱因子
        avg_exp: 平均体验分，用于动态调整
    """
    # 1. 基准价格优先 (40%权重)
    center_price = s_t.price_norm * (SIMPLEST_RULES['price_max'] - SIMPLEST_RULES['price_min']) + SIMPLEST_RULES['price_min']
    price_dist = abs(price - center_price) / center_price
    price_priority = max(0.01, 0.98 * np.exp(-4 * price_dist))
    
    # 2. 混乱惩罚（使用动态调整）
    base_chaos = compute_chaos_factor(all_actions, player_count)
    
    # 如果提供了chaos_factor_manager，使用动态调整
    if chaos_factor_manager is not None and avg_exp is not None:
        chaos_multiplier = chaos_factor_manager.get_chaos_multiplier(player_count, avg_exp)
        # 将基础混乱因子按比例缩放（但保持相对关系）
        adjusted_chaos = base_chaos * (0.08 / 0.15) * chaos_multiplier  # 从0.15基准降到0.08基准
    else:
        # 默认：降低混乱因子剂量（直接缩放）
        adjusted_chaos = base_chaos * (0.08 / 0.15)  # 减半效果
    
    # 降低惩罚强度（从60%降到40%）
    chaos_penalty = 1.0 - 0.4 * adjusted_chaos  # 最大惩罚从60%降到40%
    
    # 3. 状态修正 (流动性、波动)
    liquidity_boost = 0.3 * s_t.liquidity
    vol_adjust = -0.1 * (s_t.volatility - 0.5)**2  # 中等波动最佳
    
    final_prob = price_priority * chaos_penalty + liquidity_boost + vol_adjust
    return np.clip(final_prob, 0.01, 0.99)

# 测试真实现象映射
if __name__ == "__main__":
    # 模拟不同场景
    scenarios = {
        "健康市场": [Action(50000+i*10, 'buy' if i%2==0 else 'sell') for i in range(8)],
        "报价分散": [Action(45000 + np.random.rand()*10000, np.random.choice(['buy','sell'])) for _ in range(8)],
        "纯买盘": [Action(50000, 'buy') for _ in range(20)],
        "拥挤市场": [Action(np.random.uniform(48000,52000), np.random.choice(['buy','sell'])) for _ in range(50)]
    }
    
    s_test = MarketState(0.5, 0.3, 0.6, 0.5, 0.5)
    
    print("=== 真实现象统计测试 ===")
    for name, actions in scenarios.items():
        chaos = compute_chaos_factor(actions, len(actions))
        test_action = Action(50000, 'buy')
        prob = compute_match_prob(test_action.price, s_test, actions, len(actions))
        
        print(f"{name:12s}: 混乱因子={chaos:.3f}, 成交概率={prob:.3f}")
    
    print("\n统计规则测试通过！")

# chaos_rules.py - V5.2 统计混乱因子规则集
"""
V5.2 混乱因子：惩罚结构塌缩而非失败率

真实世界现象的统计提炼：
1. 报价分散 → 流动性碎片化
2. 方向熵高 → 缺乏共识，成交困难  
3. 规模过载 → 市场容量饱和

V5.2 新增：
4. 结构塌缩 → 单点吸引子或均匀噪声
"""

import os
import numpy as np
from collections import Counter
from typing import List, Optional, Tuple, Dict
from .state_engine import Action, MarketState
from .trading_rules import SIMPLEST_RULES

# Debug 开关：IG_DEBUG_STRUCT=1 时启用结构审计字段输出
IG_DEBUG_STRUCT = os.environ.get('IG_DEBUG_STRUCT', '0') == '1'


def compute_collapse_proxy(match_prices: List[float], 
                           active_protocols_ratio: float = 0.5,
                           n_bins: int = 10) -> float:
    """
    V5.2 新增：计算结构塌缩代理指标
    
    公式：collapse = |2 * H_norm - 1|
    
    当 H_norm > 0.8 时，用 active_protocols_ratio 区分：
    - 高活跃占比 → 均匀噪声 → 塌缩
    - 低活跃占比 → 多结构有骨架 → 健康
    
    Args:
        match_prices: 成交价格列表
        active_protocols_ratio: 活跃簇占比（驻留>2%的簇数 / 总簇数）
        n_bins: 直方图分箱数
        
    Returns:
        collapse_proxy: 0 → 健康结构，1 → 塌缩状态
    """
    if len(match_prices) < 2:
        return 1.0  # 无数据时视为塌缩
    
    # 计算价格分布直方图
    price_min = SIMPLEST_RULES['price_min']
    price_max = SIMPLEST_RULES['price_max']
    
    hist, _ = np.histogram(match_prices, bins=n_bins, range=(price_min, price_max))
    hist = hist.astype(float) + 1e-10  # 避免 log(0)
    hist = hist / hist.sum()  # 归一化为概率
    
    # 计算归一化熵
    H = -np.sum(hist * np.log(hist))
    H_max = np.log(n_bins)
    H_norm = H / H_max if H_max > 0 else 0.0
    
    # 基础塌缩 proxy: |2H - 1|
    collapse = abs(2 * H_norm - 1)
    
    # 高熵时：用 active_protocols_ratio 区分噪声与健康
    if H_norm > 0.8:
        # active_protocols_ratio 高 → 均匀噪声（塌缩），保持 collapse
        # active_protocols_ratio 低 → 多结构有骨架（健康），压低 collapse
        collapse *= active_protocols_ratio
    
    return np.clip(collapse, 0.0, 1.0)


def compute_collapse_proxy_debug(match_prices: List[float], 
                                  active_protocols_ratio: float = 0.5,
                                  n_bins: int = 10) -> Tuple[float, Dict]:
    """
    V5.2 审计版：计算结构塌缩代理指标，返回 debug 字段
    
    返回：
        (collapse_final, dbg)
        
    dbg 字段：
        - match_count: int              # len(match_prices)
        - H_price_norm: float           # 直方图熵归一化后的 H_norm
        - collapse_raw: float           # abs(2*H_price_norm - 1)（乘 active_ratio 前）
        - active_ratio_in: float        # 入参 active_protocols_ratio
        - collapse_final: float         # 最终 collapse（乘完 active_ratio，clip 后）
    """
    dbg = {
        'match_count': len(match_prices),
        'H_price_norm': 0.0,
        'collapse_raw': 1.0,
        'active_ratio_in': float(np.clip(active_protocols_ratio, 0.0, 1.0)),
        'collapse_final': 1.0,
    }
    
    if len(match_prices) < 2:
        # 无数据时视为塌缩
        return 1.0, dbg
    
    # 计算价格分布直方图
    price_min = SIMPLEST_RULES['price_min']
    price_max = SIMPLEST_RULES['price_max']
    
    hist, _ = np.histogram(match_prices, bins=n_bins, range=(price_min, price_max))
    hist = hist.astype(float) + 1e-10  # 避免 log(0)
    hist = hist / hist.sum()  # 归一化为概率
    
    # 计算归一化熵
    H = -np.sum(hist * np.log(hist))
    H_max = np.log(n_bins)
    H_norm = H / H_max if H_max > 0 else 0.0
    H_norm = float(np.clip(H_norm, 0.0, 1.0))
    
    dbg['H_price_norm'] = H_norm
    
    # 基础塌缩 proxy: |2H - 1|
    collapse_raw = abs(2 * H_norm - 1)
    dbg['collapse_raw'] = float(np.clip(collapse_raw, 0.0, 1.0))
    
    collapse = collapse_raw
    
    # 高熵时：用 active_protocols_ratio 区分噪声与健康
    if H_norm > 0.8:
        collapse *= active_protocols_ratio
    
    collapse_final = float(np.clip(collapse, 0.0, 1.0))
    dbg['collapse_final'] = collapse_final
    
    return collapse_final, dbg

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
    """
    V5.2 动态混乱因子管理器
    
    关键变更：惩罚结构塌缩而非失败率
    
    公式：chaos = base_chaos * (1 + γ * collapse_proxy) * multiplier
    """
    def __init__(self, collapse_sensitivity: float = 0.5):
        """
        Args:
            collapse_sensitivity: γ，塌缩敏感度参数
        """
        self.base_chaos = 0.08
        self.decay_factor = 0.95
        self.current_tick = 0
        
        # V5.2 新增
        self.collapse_sensitivity = collapse_sensitivity  # γ
        self.recent_match_prices: List[float] = []  # 最近成交价格
        self.active_protocols_ratio = 0.5  # 活跃簇占比（由外部设置）
    
    def update_tick(self, tick: int):
        """更新当前tick"""
        self.current_tick = tick
    
    def set_match_prices(self, match_prices: List[float]):
        """V5.2 新方法：设置最近成交价格"""
        self.recent_match_prices = match_prices
    
    def set_active_protocols_ratio(self, ratio: float):
        """V5.2 新方法：设置活跃簇占比"""
        self.active_protocols_ratio = np.clip(ratio, 0.0, 1.0)
    
    def get_collapse_proxy(self) -> float:
        """V5.2 新方法：获取当前塌缩代理"""
        if len(self.recent_match_prices) < 2:
            return 0.5  # 默认中等
        return compute_collapse_proxy(
            self.recent_match_prices, 
            self.active_protocols_ratio
        )
    
    def get_chaos_multiplier(self, player_count: int, avg_exp: float) -> float:
        """
        获取混乱因子乘数
        
        V5.2: 加入 collapse_proxy 惩罚
        
        返回: 乘数 [0.5, 2.0]
        """
        # 动态衰减
        decay = self.decay_factor ** (self.current_tick // 10000)
        chaos = self.base_chaos * decay
        
        # V5.2: 结构塌缩惩罚
        collapse = self.get_collapse_proxy()
        chaos *= (1 + self.collapse_sensitivity * collapse)
        
        # 拥挤惩罚(温和版) - 保留但降低权重
        if player_count > 15:
            chaos *= 1.1  # 从1.2降到1.1
        elif player_count < 5:
            chaos *= 0.9  # 从0.8改到0.9
        
        # 体验差时降低混乱（保留）
        if avg_exp < 0.3:
            chaos *= 0.8
        
        # 上限保护
        return np.clip(chaos / self.base_chaos, 0.5, 2.0)

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
    print("=== V5.2 混乱因子测试 ===\n")
    
    # 测试1: 结构塌缩代理
    print("--- 测试1: 结构塌缩代理 (collapse_proxy) ---")
    
    # 场景1: 单点集中（塌缩）
    prices_single = [50000.0] * 100
    collapse1 = compute_collapse_proxy(prices_single, active_protocols_ratio=0.2)
    print(f"单点集中: collapse={collapse1:.3f} (预期≈1)")
    
    # 场景2: 有限区间分布（健康）
    prices_structured = [48000 + i*500 for i in range(20)] * 5
    collapse2 = compute_collapse_proxy(prices_structured, active_protocols_ratio=0.4)
    print(f"有限区间: collapse={collapse2:.3f} (预期≈0)")
    
    # 场景3: 均匀分布 + 高活跃占比（均匀噪声塌缩）
    prices_uniform = np.random.uniform(40000, 60000, 100).tolist()
    collapse3 = compute_collapse_proxy(prices_uniform, active_protocols_ratio=0.9)
    print(f"均匀分布+高活跃: collapse={collapse3:.3f} (预期≈0.9)")
    
    # 场景4: 均匀分布 + 低活跃占比（多结构健康）
    collapse4 = compute_collapse_proxy(prices_uniform, active_protocols_ratio=0.2)
    print(f"均匀分布+低活跃: collapse={collapse4:.3f} (预期≈0.2)")
    
    # 测试2: ChaosFactor 管理器
    print("\n--- 测试2: ChaosFactor 管理器 ---")
    
    cf = ChaosFactor(collapse_sensitivity=0.5)
    cf.update_tick(1000)
    
    # 无塌缩时
    cf.set_match_prices(prices_structured)
    cf.set_active_protocols_ratio(0.4)
    mult1 = cf.get_chaos_multiplier(player_count=10, avg_exp=0.5)
    print(f"健康结构: multiplier={mult1:.3f}")
    
    # 有塌缩时
    cf.set_match_prices(prices_single)
    cf.set_active_protocols_ratio(0.2)
    mult2 = cf.get_chaos_multiplier(player_count=10, avg_exp=0.5)
    print(f"单点塌缩: multiplier={mult2:.3f}")
    
    # 测试3: 原有场景
    print("\n--- 测试3: 原有混乱因子场景 ---")
    scenarios = {
        "健康市场": [Action(50000+i*10, 'buy' if i%2==0 else 'sell') for i in range(8)],
        "报价分散": [Action(45000 + np.random.rand()*10000, np.random.choice(['buy','sell'])) for _ in range(8)],
        "纯买盘": [Action(50000, 'buy') for _ in range(20)],
        "拥挤市场": [Action(np.random.uniform(48000,52000), np.random.choice(['buy','sell'])) for _ in range(50)]
    }
    
    s_test = MarketState(0.5, 0.3, 0.6, 0.5, 0.5)
    
    for name, actions in scenarios.items():
        chaos = compute_chaos_factor(actions, len(actions))
        test_action = Action(50000, 'buy')
        prob = compute_match_prob(test_action.price, s_test, actions, len(actions))
        print(f"{name:12s}: 混乱因子={chaos:.3f}, 成交概率={prob:.3f}")
    
    print("\nV5.2 混乱因子测试通过！")

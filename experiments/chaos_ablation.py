# chaos_ablation.py - P3: 混沌因子消融实验
"""
P3修复：混沌因子作为控制手柄的可证伪化
- chaos_strength ∈ {0, 0.5x, 1x, 2x}
- 分量消融：只开dispersion / 只开entropy / 只开overload
- 诊断：统计final_prob clip前分布与贴顶比例
"""

import numpy as np
from typing import Dict, List, Optional
from core_system import V5MarketSimulator
from core_system.chaos_rules import compute_price_dispersion, compute_direction_entropy, compute_scale_overload

def run_chaos_ablation_experiment(
    seed: int,
    ticks: int,
    chaos_strength: float = 1.0,
    enable_dispersion: bool = True,
    enable_entropy: bool = True,
    enable_overload: bool = True
) -> Dict:
    """
    运行混沌因子消融实验
    
    Args:
        seed: 随机种子
        chaos_strength: 混沌因子强度倍数（0, 0.5, 1.0, 2.0）
        enable_dispersion: 是否启用价格分散
        enable_entropy: 是否启用方向熵
        enable_overload: 是否启用规模过载
    
    Returns:
        包含指标和诊断信息的字典
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 创建模拟器
    sim = V5MarketSimulator(ticks=ticks, adjust_interval=2000, MAX_N=None)
    
    # 修改compute_match_prob以支持消融
    original_compute_match_prob = None
    from core_system.chaos_rules import compute_match_prob as original_cmp
    
    def ablated_compute_match_prob(price, s_t, all_actions, player_count, chaos_factor_manager=None, avg_exp=None):
        """消融版本的compute_match_prob"""
        from core_system.chaos_rules import SIMPLEST_RULES
        
        # 1. 基准价格优先
        center_price = s_t.price_norm * (SIMPLEST_RULES['price_max'] - SIMPLEST_RULES['price_min']) + SIMPLEST_RULES['price_min']
        price_dist = abs(price - center_price) / center_price
        price_priority = max(0.01, 0.98 * np.exp(-4 * price_dist))
        
        # 2. 计算混乱因子（支持分量消融）
        if chaos_strength == 0:
            adjusted_chaos = 0.0
        else:
            dispersion = compute_price_dispersion(all_actions) if enable_dispersion else 0.0
            direction_entropy = compute_direction_entropy(all_actions) if enable_entropy else 0.0
            scale_overload = compute_scale_overload(player_count, len(all_actions)) if enable_overload else 0.0
            
            # 加权综合
            chaos = (
                0.4 * dispersion +
                0.35 * direction_entropy +
                0.25 * scale_overload
            )
            base_chaos = np.clip(chaos, 0.0, 1.5)
            
            # 应用强度倍数
            adjusted_chaos = base_chaos * chaos_strength * (0.08 / 0.15)  # 归一化到0.08基准
        
        # 3. 混乱惩罚
        chaos_penalty = 1.0 - 0.4 * adjusted_chaos
        
        # 4. 状态修正
        liquidity_boost = 0.3 * s_t.liquidity
        vol_adjust = -0.1 * (s_t.volatility - 0.5)**2
        
        # 5. 最终概率（记录clip前的值用于诊断）
        final_prob_unclipped = price_priority * chaos_penalty + liquidity_boost + vol_adjust
        final_prob = np.clip(final_prob_unclipped, 0.01, 0.99)
        
        return final_prob, final_prob_unclipped
    
    # 替换sample_matches方法以支持诊断
    original_sample_matches = sim.sample_matches
    
    prob_unclipped_history = []
    prob_clipped_history = []
    
    def ablated_sample_matches(actions, s_t):
        matches = []
        player_count = len(sim.active_players)
        avg_exp = np.mean([p.experience_score for p in sim.active_players]) if sim.active_players else 0.0
        
        for i, action in enumerate(actions):
            prob, prob_unclipped = ablated_compute_match_prob(
                action.price, s_t, actions, player_count,
                chaos_factor_manager=sim.chaos_factor,
                avg_exp=avg_exp
            )
            
            prob_unclipped_history.append(prob_unclipped)
            prob_clipped_history.append(prob)
            
            if np.random.random() < prob:
                matches.append((i, action))
        
        return matches
    
    sim.sample_matches = ablated_sample_matches
    
    # 运行仿真
    import io
    import contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        metrics = sim.run_simulation()
    
    # 诊断统计
    prob_unclipped_array = np.array(prob_unclipped_history)
    prob_clipped_array = np.array(prob_clipped_history)
    
    # 贴顶/贴底比例
    clip_top_ratio = np.mean(prob_clipped_array >= 0.99)
    clip_bottom_ratio = np.mean(prob_clipped_array <= 0.01)
    
    # 分布统计
    prob_mean = np.mean(prob_unclipped_array)
    prob_std = np.std(prob_unclipped_array)
    prob_min = np.min(prob_unclipped_array)
    prob_max = np.max(prob_unclipped_array)
    
    return {
        **metrics,
        'chaos_strength': chaos_strength,
        'enable_dispersion': enable_dispersion,
        'enable_entropy': enable_entropy,
        'enable_overload': enable_overload,
        'prob_unclipped_mean': prob_mean,
        'prob_unclipped_std': prob_std,
        'prob_unclipped_min': prob_min,
        'prob_unclipped_max': prob_max,
        'clip_top_ratio': clip_top_ratio,
        'clip_bottom_ratio': clip_bottom_ratio,
        'prob_unclipped_history': prob_unclipped_history[:1000],  # 只保存前1000个用于可视化
    }

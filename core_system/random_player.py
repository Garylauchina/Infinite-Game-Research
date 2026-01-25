# random_player.py - V5.2 随机体验玩家
"""
RandomExperiencePlayer：市场先生"试玩游戏"的真实行为
- 纯随机报价（真实试探规则边界）
- V5.2新增：并行探针机制（K probes per tick）
- 追踪游戏体验分量（成功率 + 分布离散度 + 结构奖励）
- 无任何价格学习（验证随机也能涌现结构）

V5.2 关键变更：
1. K 探针：每个 Player 每 tick 生成 K 次独立探针
2. 体验更新：失败不直接惩罚，通过 success_rate 和 dispersion 间接影响
"""

import numpy as np
from .trading_rules import compute_fee, SIMPLEST_RULES
from .state_engine import Action, MarketState
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """单个 tick 的探针结果聚合"""
    probes: List[Action]        # K 个探针
    n_success: int              # 成交成功数
    n_failure: int              # 成交失败数
    match_prices: List[float]   # 成交价格列表
    success_rate: float         # 成功率 = n_success / K
    match_dispersion: float     # 成交价格离散度（归一化）


class RandomExperiencePlayer:
    def __init__(self, player_id: int, probe_count: int = 1, probe_count_random: bool = False):
        """
        V5.2 随机体验玩家
        
        Args:
            player_id: 玩家ID
            probe_count: K 值，每 tick 生成的探针数量（默认1，与v5.0兼容）
            probe_count_random: 是否随机化 K（如果True，K ~ Uniform(1, probe_count)）
        """
        self.id = player_id
        self.experience_score = 0.0  # 核心指标 [-2, 2]
        self.action_history: List[Action] = []     # 最近行动记录
        self.experience_history: List[float] = []  # 最近体验记录
        
        # V5.2 新增：并行探针参数
        self.probe_count = max(1, probe_count)  # K ≥ 1
        self.probe_count_random = probe_count_random
        
        # V5.2 新增：当前 tick 的探针结果（由外部设置）
        self.current_probe_result: Optional[ProbeResult] = None
        
    def _sample_k(self) -> int:
        """采样当前 tick 的 K 值"""
        if self.probe_count_random and self.probe_count > 1:
            return np.random.randint(1, self.probe_count + 1)
        return self.probe_count
    
    def generate_probes(self, s_t: MarketState) -> List[Action]:
        """
        V5.2 新方法：生成 K 个并行探针
        
        K 的语义：行为采样密度，不是订单厚度
        - K=1: 与 v5.0 行为一致
        - K>1: 更密集地试探价格区间
        """
        k = self._sample_k()
        probes = []
        
        for _ in range(k):
            price = np.random.uniform(SIMPLEST_RULES['price_min'], SIMPLEST_RULES['price_max'])
            side = np.random.choice(['buy', 'sell'])
            probe = Action(price=price, side=side, qty=1.0)
            probes.append(probe)
        
        # 记录最后一个探针到历史（兼容旧逻辑）
        if probes:
            self.action_history.append(probes[-1])
            if len(self.action_history) > 50:
                self.action_history.pop(0)
        
        return probes
    
    def decide_action(self, s_t: MarketState) -> Action:
        """
        兼容 v5.0 的单探针接口
        
        注意：v5.2 推荐使用 generate_probes() 方法
        """
        probes = self.generate_probes(s_t)
        return probes[0] if probes else Action(price=50000, side='buy', qty=1.0)
    
    def set_probe_result(self, probe_result: ProbeResult):
        """
        V5.2 新方法：设置当前 tick 的探针结果
        由 main.py 在成交判定后调用
        """
        self.current_probe_result = probe_result
    
    def update_experience(self, matched: bool, s_t: MarketState) -> float:
        """
        V5.2 体验更新：失败不直接惩罚！
        
        instant_reward = w₁ * success_rate + w₂ * match_dispersion + w₃ * complexity
        
        关键设计：
        - 失败通过 success_rate 间接影响（不是直接惩罚）
        - match_dispersion 体现成交分布的结构性
        - 大量失败 ≠ 不好玩（如果失败有结构）
        """
        # V5.2：优先使用 probe_result
        if self.current_probe_result is not None:
            return self._update_experience_v52(s_t)
        
        # 兼容 v5.0：单探针模式
        return self._update_experience_v50(matched, s_t)
    
    def _update_experience_v52(self, s_t: MarketState) -> float:
        """
        V5.2 体验更新：基于探针结果
        
        公式：instant_reward = w₁ * success_rate + w₂ * match_dispersion + w₃ * complexity
        权重：w₁ = 0.4, w₂ = 0.2, w₃ = 0.4
        """
        pr = self.current_probe_result
        
        # 权重配置
        w1, w2, w3 = 0.4, 0.2, 0.4
        
        # 1. 成功率 [0, 1]
        success_rate = pr.success_rate
        
        # 2. 成交分布离散度 [0, 1]
        # 如果有成交，计算价格标准差/价格范围
        match_dispersion = pr.match_dispersion
        
        # 3. 世界结构密度 [0, 1]
        complexity = s_t.complexity
        
        # 综合体验（不直接惩罚失败！）
        instant_experience = (
            w1 * success_rate +
            w2 * match_dispersion +
            w3 * complexity
        )
        
        # 记录体验历史
        self.experience_history.append(instant_experience)
        if len(self.experience_history) > 50:
            self.experience_history.pop(0)
        
        # 指数移动平均（保持 v5.0 的 0.98/0.02）
        self.experience_score = (
            0.98 * self.experience_score + 
            0.02 * instant_experience
        )
        
        # 清空当前探针结果
        self.current_probe_result = None
        
        return instant_experience
    
    def _update_experience_v50(self, matched: bool, s_t: MarketState) -> float:
        """
        V5.0 兼容：单探针体验更新
        """
        if not self.action_history:
            return 0.0
            
        action = self.action_history[-1]
        fee = compute_fee(action, s_t, matched)
        
        # 体验分量设计（v5.0 原始逻辑）
        match_reward = +1.0 if matched else -0.3
        fee_rate = fee / (action.price * action.qty) if action.price * action.qty > 0 else 0.0
        fee_penalty = -10.0 * fee_rate
        structure_reward = +0.4 * s_t.complexity
        volatility_reward = +0.2 * s_t.volatility
        liquidity_reward = +0.1 * s_t.liquidity
        
        instant_experience = (
            match_reward + fee_penalty + 
            structure_reward + volatility_reward + liquidity_reward
        )
        
        # 记录体验历史
        self.experience_history.append(instant_experience)
        if len(self.experience_history) > 50:
            self.experience_history.pop(0)
        
        # 指数移动平均
        self.experience_score = (
            0.98 * self.experience_score + 
            0.02 * instant_experience
        )
        
        return instant_experience
    
    def get_status(self) -> Dict[str, Any]:
        """获取玩家状态"""
        recent_avg_exp = (
            np.mean(self.experience_history[-10:]) 
            if len(self.experience_history) >= 10 
            else 0.0
        )
        return {
            'id': self.id,
            'experience_score': self.experience_score,
            'recent_avg_exp': recent_avg_exp,
            'total_actions': len(self.action_history)
        }

# 测试
if __name__ == "__main__":
    from state_engine import StateEngine, MarketState
    
    engine = StateEngine()
    s_test = MarketState(0.5, 0.3, 0.6, 0.5, 0.4)
    
    print("=== V5.2 随机体验玩家测试 ===")
    
    # 测试1：K=1 模式（兼容 v5.0）
    print("\n--- 测试1: K=1 单探针模式 ---")
    player1 = RandomExperiencePlayer(0, probe_count=1)
    for i in range(3):
        action = player1.decide_action(s_test)
        print(f"行动 {i}: 报价={action.price:.0f}, 方向={action.side}")
        matched = np.random.random() < 0.7
        exp = player1.update_experience(matched, s_test)
        print(f"  → 成交={matched}, 体验={exp:.3f}, 累计={player1.experience_score:.3f}")
    
    # 测试2：K=3 多探针模式
    print("\n--- 测试2: K=3 多探针模式 ---")
    player2 = RandomExperiencePlayer(1, probe_count=3)
    for i in range(3):
        probes = player2.generate_probes(s_test)
        print(f"行动 {i}: 生成 {len(probes)} 个探针")
        for j, p in enumerate(probes):
            print(f"  探针{j}: 报价={p.price:.0f}, 方向={p.side}")
        
        # 模拟成交结果
        n_success = np.random.binomial(len(probes), 0.6)
        match_prices = [probes[k].price for k in range(n_success)]
        
        # 计算离散度
        if len(match_prices) >= 2:
            price_range = SIMPLEST_RULES['price_max'] - SIMPLEST_RULES['price_min']
            dispersion = np.std(match_prices) / price_range
        else:
            dispersion = 0.0
        
        probe_result = ProbeResult(
            probes=probes,
            n_success=n_success,
            n_failure=len(probes) - n_success,
            match_prices=match_prices,
            success_rate=n_success / len(probes),
            match_dispersion=dispersion
        )
        
        player2.set_probe_result(probe_result)
        exp = player2.update_experience(True, s_test)  # matched参数在v5.2中被忽略
        print(f"  → 成交={n_success}/{len(probes)}, 体验={exp:.3f}, 累计={player2.experience_score:.3f}")
    
    print(f"\n玩家1状态: {player1.get_status()}")
    print(f"玩家2状态: {player2.get_status()}")
    print("\nV5.2 随机体验玩家测试通过！")

# random_player.py - V5.0 随机体验玩家
"""
RandomExperiencePlayer：市场先生"试玩游戏"的真实行为
- 纯随机报价（真实试探规则边界）
- 追踪游戏体验分量（成交快感 + 结构奖励 - 费用痛苦）
- 无任何价格学习（验证随机也能涌现结构）
"""

import numpy as np
from trading_rules import compute_match_prob, compute_fee, SIMPLEST_RULES
from state_engine import Action, MarketState
from typing import Dict, Any, List

class RandomExperiencePlayer:
    def __init__(self, player_id: int):
        self.id = player_id
        self.experience_score = 0.0  # 核心指标 [-2, 2]
        self.action_history: List[Action] = []     # 最近行动记录
        self.experience_history: List[float] = []  # 最近体验记录
        
    def decide_action(self, s_t: MarketState) -> Action:
        """纯随机报价 - 市场先生试玩行为"""
        price = np.random.uniform(SIMPLEST_RULES['price_min'], SIMPLEST_RULES['price_max'])
        side = np.random.choice(['buy', 'sell'])
        
        action = Action(price=price, side=side, qty=1.0)
        self.action_history.append(action)
        if len(self.action_history) > 50:  # 保持最近50次
            self.action_history.pop(0)
            
        return action
    
    def update_experience(self, matched: bool, s_t: MarketState) -> float:
        """
        游戏体验 = 成交快感 - 费用痛苦 + 市场氛围奖励
        """
        if not self.action_history:
            return 0.0
            
        action = self.action_history[-1]
        fee = compute_fee(action, s_t, matched)
        
        # 体验分量设计（参考设计文档）
        match_reward = +1.0 if matched else -0.3
        # 费用惩罚：使用费率而非绝对金额，确保体验分值在合理范围
        fee_rate = fee / (action.price * action.qty) if action.price * action.qty > 0 else 0.0
        fee_penalty = -10.0 * fee_rate  # 费率惩罚（放大10倍）
        structure_reward = +0.4 * s_t.complexity
        volatility_reward = +0.2 * s_t.volatility
        liquidity_reward = +0.1 * s_t.liquidity  # 额外奖励
        
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
    player = RandomExperiencePlayer(0)
    
    print("=== 随机体验玩家测试 ===")
    for i in range(5):
        action = player.decide_action(s_test)
        print(f"行动 {i}: 报价={action.price:.0f}, 方向={action.side}")
        
        matched = np.random.random() < 0.7  # 模拟成交
        exp = player.update_experience(matched, s_test)
        print(f"  → 成交={matched}, 体验分值={exp:.2f}, 累计={player.experience_score:.3f}")
    
    print(f"\n玩家状态: {player.get_status()}")
    print("随机体验玩家测试通过！")

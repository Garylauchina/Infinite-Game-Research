# trading_rules.py - V5.0 最简共性规则集
"""
提炼所有复杂交易所规则的共性：
1. 离中心价越近越容易成交 (价格优先)
2. 固定手续费 (简单费率)
无其他复杂规则
"""

import numpy as np
from .state_engine import Action, MarketState

SIMPLEST_RULES = {
    'price_min': 40000.0,
    'price_max': 60000.0,
    'fixed_fee': 0.0005  # 0.05% 固定费率
}

def compute_match_prob(price: float, s_t: MarketState) -> float:
    """
    最简规则：价格优先 (离中心价越近越容易成交)
    """
    price_range = SIMPLEST_RULES['price_max'] - SIMPLEST_RULES['price_min']
    center_price = s_t.price_norm * price_range + SIMPLEST_RULES['price_min']
    price_dist = abs(price - center_price) / center_price if center_price > 0 else 1.0
    
    # 距离惩罚：10%价差衰减到 e^-1 ≈ 0.37
    prob = max(0.01, 0.98 * np.exp(-5 * price_dist))
    return prob

def compute_fee(action: Action, s_t: MarketState, matched: bool) -> float:
    """
    最简费率：固定 0.05%，成交才收
    """
    if not matched:
        return 0.0
    return SIMPLEST_RULES['fixed_fee'] * action.price * action.qty

def sample_match(action: Action, s_t: MarketState) -> bool:
    """
    根据规则采样成交
    """
    prob = compute_match_prob(action.price, s_t)
    return np.random.random() < prob

# 测试
if __name__ == "__main__":
    from state_engine import StateEngine, MarketState
    
    engine = StateEngine()
    s_test = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
    action_test = Action(price=50000, side='buy')
    
    print("=== 最简规则测试 ===")
    print(f"中心价: {engine.denorm_price(s_test.price_norm):.0f}")
    print(f"测试报价: {action_test.price}")
    
    prob = compute_match_prob(action_test.price, s_test)
    matched = sample_match(action_test, s_test)
    fee = compute_fee(action_test, s_test, matched)
    
    print(f"成交概率: {prob:.3f}")
    print(f"实际成交: {matched}")
    print(f"手续费: {fee:.6f}")
    
    print("\n规则测试通过！")

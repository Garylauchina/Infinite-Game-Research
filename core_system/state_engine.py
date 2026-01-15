# state_engine.py - V5.0 Phase 1 核心状态引擎
"""
最小的可运行状态更新模块
验证：随机输入 → 状态演化 → 基本稳定性
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class MarketState:
    """5维状态向量 [0,1]"""
    price_norm: float     # 标准化价格
    volatility: float     # 短期波动率
    liquidity: float      # 成交容易度
    imbalance: float      # 买卖倾斜 [0,1]→[-1,1]
    complexity: float     # 结构密度

@dataclass  
class Action:
    """单个Player行动"""
    price: float          # 绝对价格
    side: str             # 'buy' or 'sell'
    qty: float = 1.0

class StateEngine:
    def __init__(self, price_min: float = 40000, price_max: float = 60000):
        self.price_min = price_min
        self.price_max = price_max
        self.price_range = price_max - price_min
        self.price_history = []  # 用于volatility
        
    def denorm_price(self, price_norm: float) -> float:
        """[0,1] → 绝对价格"""
        return price_norm * self.price_range + self.price_min
        
    def norm_price(self, price: float) -> float:
        """绝对价格 → [0,1]"""
        return np.clip((price - self.price_min) / self.price_range, 0, 1)
    
    def update(self, 
               s_t: MarketState, 
               actions: List[Action], 
               matches: List[Action]) -> MarketState:
        """
        核心状态更新 - 聚合统计
        """
        if len(actions) == 0:
            return s_t  # 无行动，状态不变
            
        # 1. 买卖压力
        buy_qty = sum(a.qty for a in actions if a.side == 'buy')
        sell_qty = sum(a.qty for a in actions if a.side == 'sell')
        total_qty = buy_qty + sell_qty
        
        pressure = (buy_qty - sell_qty) / (total_qty + 1e-6)
        
        # 2. 价格更新
        price_abs = self.denorm_price(s_t.price_norm)
        new_price_abs = price_abs * (1 + 0.0005 * pressure)
        new_price_norm = self.norm_price(new_price_abs)
        
        # 3. 波动率 (20-tick 窗口)
        self.price_history.append(new_price_norm)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history)
            vol = np.std(returns) * np.sqrt(20)  # 年化近似
            new_vol = np.clip(vol, 0, 1)
        else:
            new_vol = 0.3  # 默认中等波动
            
        # 4. 流动性 = 成交率
        new_liq = len(matches) / len(actions)
        
        # 5. 不平衡度
        new_imbalance_raw = pressure  # [-1,1]
        new_imbalance = (new_imbalance_raw + 1) / 2  # [0,1]
        
        # 6. 复杂度暂用占位 (后续实现)
        new_complexity = 0.5  # Phase 1 固定，后续计算
        
        return MarketState(
            price_norm=new_price_norm,
            volatility=new_vol,
            liquidity=new_liq,
            imbalance=new_imbalance,
            complexity=new_complexity
        )

# 测试
if __name__ == "__main__":
    engine = StateEngine()
    
    # 初始状态
    s0 = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
    
    # 模拟100 ticks随机演化
    s = s0
    states = [s]
    
    for t in range(100):
        # 模拟随机行动 (后续由player提供)
        n_actions = np.random.poisson(3)
        actions = [Action(np.random.uniform(40000,60000), np.random.choice(['buy','sell'])) 
                  for _ in range(n_actions)]
        n_matches = np.random.binomial(n_actions, 0.6)
        matches = actions[:n_matches]
        
        s = engine.update(s, actions, matches)
        states.append(s)
        print(f"t={t}: price={s.price_norm:.3f}, liq={s.liquidity:.3f}")
    
    print("状态引擎测试通过！")

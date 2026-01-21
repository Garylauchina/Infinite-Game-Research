# V5.0 基底模型包
"""
V5.0 Phase 1 核心模拟器
冻结的底层机制，仅用于实验运行
"""

from .main import V5MarketSimulator
from .state_engine import StateEngine, MarketState, Action
from .random_player import RandomExperiencePlayer
from .trading_rules import SIMPLEST_RULES, compute_fee
from .chaos_rules import ChaosFactor, compute_match_prob, compute_chaos_factor
from .metrics import StructureMetrics

__all__ = [
    'V5MarketSimulator',
    'StateEngine', 'MarketState', 'Action',
    'RandomExperiencePlayer',
    'SIMPLEST_RULES', 'compute_fee',
    'ChaosFactor', 'compute_match_prob', 'compute_chaos_factor',
    'StructureMetrics'
]

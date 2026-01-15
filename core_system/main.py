# main.py - V5.0 Phase 1 集成主循环
"""
集成测试：随机报价 → 规则判定 → 状态演化 → 参与调整
验证：简单规则下涌现复杂结构
"""

import numpy as np
import os
from state_engine import StateEngine, MarketState, Action
from trading_rules import compute_fee, SIMPLEST_RULES
from chaos_rules import compute_match_prob, ChaosFactor
from random_player import RandomExperiencePlayer
from metrics import StructureMetrics
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # 忽略字体警告

# 设置matplotlib配置目录到可写位置（使用工作目录，避免在核心仿真路径中初始化）
if 'MPLCONFIGDIR' not in os.environ:
    config_dir = os.path.join(os.path.dirname(__file__), '.matplotlib_cache')
    os.makedirs(config_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = config_dir

class V5MarketSimulator:
    def __init__(self, ticks=50000, adjust_interval=1000, MAX_N=None):
        self.engine = StateEngine()
        self.adjust_interval = adjust_interval
        self.ticks = ticks
        
        # 玩家数量上限（None表示无上限）
        self.MAX_N = MAX_N if MAX_N is not None else float('inf')
        
        # 初始 3 个玩家
        self.active_players = [RandomExperiencePlayer(i) for i in range(3)]
        self.player_history = []  # 记录玩家数量变化
        
        # 轨迹记录
        self.state_trajectory = []
        self.complexity_history = []
        self.experience_history = []
        
        # 结构密度计算器
        self.structure_metrics = StructureMetrics(window_size=5000, n_clusters=5)
        
        # 混乱因子管理器
        self.chaos_factor = ChaosFactor()
        
        # 玩家调整阈值（降低版本）
        self.ADD_PLAYER_THRESHOLD = 0.35  # 从0.7→0.35
        self.REMOVE_PLAYER_THRESHOLD = 0.15  # 从0.3→0.15
        
    def sample_matches(self, actions: list[Action], s_t: MarketState) -> list[tuple]:
        """根据规则采样成交，返回 (action_index, action) 元组列表"""
        matches = []
        player_count = len(self.active_players)
        avg_exp = np.mean([p.experience_score for p in self.active_players]) if self.active_players else 0.0
        
        # 使用统计版成交概率（包含动态混乱因子）
        for i, action in enumerate(actions):
            prob = compute_match_prob(
                action.price, s_t, actions, player_count,
                chaos_factor_manager=self.chaos_factor,
                avg_exp=avg_exp
            )
            if np.random.random() < prob:
                matches.append((i, action))
        return matches
    
    def adjust_participation(self):
        """根据平均体验调整玩家数量（降低阈值版本）"""
        if len(self.active_players) == 0:
            return
            
        avg_exp = np.mean([p.experience_score for p in self.active_players])
        
        print(f"调整时刻: 玩家数={len(self.active_players)}, 平均体验={avg_exp:.3f}")
        
        # 最小玩家保护机制
        if len(self.active_players) <= 2 and avg_exp > 0.25:
            new_player = RandomExperiencePlayer(len(self.active_players))
            self.active_players.append(new_player)
            print(f"  ⚠️  激活保护机制，加人！当前体验:{avg_exp:.3f}, 总数 {len(self.active_players)}")
            return
        
        # 无上限模式：降低加人阈值（从0.7→0.35）
        if avg_exp > self.ADD_PLAYER_THRESHOLD and len(self.active_players) < self.MAX_N:
            new_player = RandomExperiencePlayer(len(self.active_players))
            self.active_players.append(new_player)
            print(f"  → 新增玩家，总数 {len(self.active_players)}")
            
        # 降低减人阈值（从0.3→0.15）
        elif avg_exp < self.REMOVE_PLAYER_THRESHOLD and len(self.active_players) > 2:
            # 移除体验最差的
            worst_idx = np.argmin([p.experience_score for p in self.active_players])
            removed = self.active_players.pop(worst_idx)
            print(f"  → 移除玩家 {removed.id} (体验{removed.experience_score:.3f})")
    
    def run_simulation(self) -> dict:
        """完整仿真"""
        print("=== V5.0 Phase 1 仿真开始 ===")
        print(f"规则: 最简价格优先 + 0.05%固定费")
        print(f"时长: {self.ticks} ticks, 参与调整: 每{self.adjust_interval} ticks")
        print("-" * 60)
        
        # 初始状态
        s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
        
        start_time = time.time()
        
        for t in range(self.ticks):
            # 更新混乱因子管理器的tick
            self.chaos_factor.update_tick(t)
            
            # 1. 所有玩家报价
            actions = [p.decide_action(s_t) for p in self.active_players]
            
            # 2. 规则采样成交（返回索引和action）
            match_indices_actions = self.sample_matches(actions, s_t)
            match_indices = {idx for idx, _ in match_indices_actions}
            matches = [action for _, action in match_indices_actions]
            
            # 3. 玩家获得反馈
            for i, player in enumerate(self.active_players):
                matched = i in match_indices
                player.update_experience(matched, s_t)
            
            # 4. 状态演化
            s_t = self.engine.update(s_t, actions, matches)
            
            # 5. 更新结构密度计算
            state_features = np.array([
                s_t.price_norm, s_t.volatility, 
                s_t.liquidity, s_t.imbalance
            ])
            self.structure_metrics.update_trajectory(state_features)
            
            # 优化：降低复杂度计算频率（从每100改为每500 ticks）
            if t % 500 == 0 and t >= 200:
                complexity = self.structure_metrics.compute_complexity()
                # 更新状态中的复杂度
                s_t.complexity = complexity
                self.complexity_history.append(complexity)
            else:
                # 使用上次的复杂度值（避免频繁计算）
                if len(self.complexity_history) > 0:
                    s_t.complexity = self.complexity_history[-1]
                self.complexity_history.append(s_t.complexity)
            
            # 记录
            self.state_trajectory.append((
                s_t.price_norm, s_t.volatility, 
                s_t.liquidity, s_t.imbalance
            ))
            
            if t % 5000 == 0:
                avg_exp = np.mean([p.experience_score for p in self.active_players])
                print(f"t={t:5d} 玩家={len(self.active_players):2d} "
                      f"exp={avg_exp:.3f} liq={s_t.liquidity:.3f} "
                      f"cpx={s_t.complexity:.3f}")
            
            # 6. 参与调整
            if t % self.adjust_interval == 0 and t > 0:
                avg_exp = np.mean([p.experience_score for p in self.active_players])
                self.experience_history.append(avg_exp)
                self.adjust_participation()
                self.player_history.append(len(self.active_players))
        
        elapsed = time.time() - start_time
        print(f"\n仿真完成！用时 {elapsed:.1f}s ({self.ticks/elapsed:.0f} ticks/s)")
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> dict:
        """计算最终指标"""
        trajectory = np.array(self.state_trajectory)
        
        # 计算最终复杂度
        final_complexity = self.structure_metrics.compute_complexity()
        cluster_info = self.structure_metrics.get_cluster_info()
        
        metrics = {
            'final_player_count': len(self.active_players),
            'final_avg_experience': np.mean([p.experience_score for p in self.active_players]),
            'avg_player_count': np.mean(self.player_history) if self.player_history else 3,
            'price_volatility': np.std(trajectory[:, 0]),
            'avg_liquidity': np.mean(trajectory[:, 2]),
            'imbalance_stability': np.std(trajectory[:, 3]),
            'trajectory_length': len(trajectory),
            'final_complexity': final_complexity,
            'avg_complexity': np.mean(self.complexity_history) if self.complexity_history else 0.0,
            'active_protocols': cluster_info.get('active_protocols', 0),
            'n_clusters': cluster_info.get('n_clusters', 0)
        }
        
        print("\n=== 最终指标 ===")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.3f}")
            
        return metrics
    
    def plot_results(self):
        """结果可视化 - 按需导入matplotlib"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        trajectory = np.array(self.state_trajectory)
        
        # 1. 玩家数量 vs 时间
        if self.player_history:
            player_ts = np.arange(0, len(self.player_history)*self.adjust_interval, self.adjust_interval)
            axes[0,0].plot(player_ts, self.player_history, marker='o')
            axes[0,0].set_title('参与度演化 (Player 数量)')
            axes[0,0].set_ylabel('Player Count')
            axes[0,0].set_xlabel('Tick')
        else:
            axes[0,0].text(0.5, 0.5, 'No player history', ha='center', va='center')
            axes[0,0].set_title('参与度演化 (Player 数量)')
        
        # 2. 价格轨迹
        time_axis = np.arange(len(trajectory))
        axes[0,1].plot(time_axis, trajectory[:, 0], alpha=0.7)
        axes[0,1].set_title('标准化价格轨迹')
        axes[0,1].set_ylabel('Price Norm')
        axes[0,1].set_xlabel('Tick')
        
        # 3. 状态空间投影 (price vs liquidity)
        axes[1,0].scatter(trajectory[:, 0], trajectory[:, 2], alpha=0.5, s=1)
        axes[1,0].set_xlabel('Price Norm')
        axes[1,0].set_ylabel('Liquidity')
        axes[1,0].set_title('状态空间轨迹')
        
        # 4. 体验分数分布
        experiences = [p.experience_score for p in self.active_players]
        if experiences:
            axes[1,1].hist(experiences, bins=max(3, len(experiences)//2))
            axes[1,1].set_title('最终体验分数分布')
            axes[1,1].set_xlabel('Experience Score')
            axes[1,1].set_ylabel('Count')
        else:
            axes[1,1].text(0.5, 0.5, 'No players', ha='center', va='center')
            axes[1,1].set_title('最终体验分数分布')
        
        plt.tight_layout()
        plt.savefig('v5_phase1_results.png', dpi=150)
        print("图表已保存: v5_phase1_results.png")
        plt.close()  # 关闭图形以避免在非交互环境中显示

# 运行！
if __name__ == "__main__":
    sim = V5MarketSimulator(ticks=20000, adjust_interval=2000)  # 短跑测试
    metrics = sim.run_simulation()
    sim.plot_results()
    
    print("\nPhase 1 集成测试完成！")
    print("查看 v5_phase1_results.png")

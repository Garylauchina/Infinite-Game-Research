# main.py - V5.2 集成主循环
"""
V5.2 集成测试：并行探针 → 规则判定 → 状态演化 → 参与调整
验证：交互阻抗机制下的结构涌现

V5.2 关键变更：
1. 并行探针机制：每个 Player 每 tick 生成 K 个探针
2. 体验更新：失败不直接惩罚，通过 success_rate 和 dispersion 间接影响
3. Chaos Factor：惩罚结构塌缩而非失败率
"""

import numpy as np
import os
from .state_engine import StateEngine, MarketState, Action
from .trading_rules import compute_fee, SIMPLEST_RULES
from .chaos_rules import compute_match_prob, ChaosFactor
from .random_player import RandomExperiencePlayer, ProbeResult
from .metrics import StructureMetrics
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # 忽略字体警告

# 设置matplotlib配置目录到可写位置
if 'MPLCONFIGDIR' not in os.environ:
    config_dir = os.path.join(os.path.dirname(__file__), '.matplotlib_cache')
    os.makedirs(config_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = config_dir


class V5MarketSimulator:
    def __init__(self, ticks=50000, adjust_interval=1000, MAX_N=None, 
                 probe_count=1, probe_count_random=False):
        """
        V5.2 市场模拟器
        
        Args:
            ticks: 仿真总时长
            adjust_interval: 参与调整间隔
            MAX_N: 玩家数量上限
            probe_count: K 值，每个玩家每 tick 的探针数量（默认1，与v5.0兼容）
            probe_count_random: 是否随机化 K
        """
        self.engine = StateEngine()
        self.adjust_interval = adjust_interval
        self.ticks = ticks
        
        # 玩家数量上限
        self.MAX_N = MAX_N if MAX_N is not None else float('inf')
        
        # V5.2 新增：探针参数
        self.probe_count = probe_count
        self.probe_count_random = probe_count_random
        
        # 初始 3 个玩家（使用 v5.2 参数）
        self.active_players = [
            RandomExperiencePlayer(i, probe_count=probe_count, probe_count_random=probe_count_random) 
            for i in range(3)
        ]
        self.player_history = []
        
        # 轨迹记录
        self.state_trajectory = []
        self.complexity_history = []
        self.experience_history = []
        
        # V5.2 新增：成交价格历史（用于 collapse_proxy）
        self.match_prices_history = []
        
        # 结构密度计算器
        self.structure_metrics = StructureMetrics(window_size=5000, n_clusters=5)
        
        # 混乱因子管理器
        self.chaos_factor = ChaosFactor()
        
        # 玩家调整阈值
        self.ADD_PLAYER_THRESHOLD = 0.35
        self.REMOVE_PLAYER_THRESHOLD = 0.15
        
    def sample_matches(self, actions: list[Action], s_t: MarketState) -> list[tuple]:
        """
        根据规则采样成交，返回 (action_index, action) 元组列表
        兼容 v5.0 单探针模式
        """
        matches = []
        player_count = len(self.active_players)
        avg_exp = np.mean([p.experience_score for p in self.active_players]) if self.active_players else 0.0
        
        for i, action in enumerate(actions):
            prob = compute_match_prob(
                action.price, s_t, actions, player_count,
                chaos_factor_manager=self.chaos_factor,
                avg_exp=avg_exp
            )
            if np.random.random() < prob:
                matches.append((i, action))
        return matches
    
    def sample_probes(self, all_probes: list[list[Action]], s_t: MarketState) -> list[ProbeResult]:
        """
        V5.2 新方法：为每个玩家的探针列表采样成交结果
        
        Args:
            all_probes: 每个玩家的探针列表 [[player0_probes], [player1_probes], ...]
            s_t: 当前市场状态
            
        Returns:
            每个玩家的 ProbeResult 列表
        """
        player_count = len(self.active_players)
        avg_exp = np.mean([p.experience_score for p in self.active_players]) if self.active_players else 0.0
        
        # 收集所有探针用于计算混乱因子
        flat_actions = [probe for probes in all_probes for probe in probes]
        
        results = []
        all_match_prices = []  # 收集所有成交价格（用于 collapse_proxy）
        
        for player_idx, probes in enumerate(all_probes):
            n_success = 0
            match_prices = []
            
            for probe in probes:
                prob = compute_match_prob(
                    probe.price, s_t, flat_actions, player_count,
                    chaos_factor_manager=self.chaos_factor,
                    avg_exp=avg_exp
                )
                if np.random.random() < prob:
                    n_success += 1
                    match_prices.append(probe.price)
                    all_match_prices.append(probe.price)
            
            k = len(probes)
            n_failure = k - n_success
            success_rate = n_success / k if k > 0 else 0.0
            
            # 计算成交价格离散度（归一化）
            if len(match_prices) >= 2:
                price_range = SIMPLEST_RULES['price_max'] - SIMPLEST_RULES['price_min']
                match_dispersion = np.std(match_prices) / price_range
            else:
                match_dispersion = 0.0
            
            probe_result = ProbeResult(
                probes=probes,
                n_success=n_success,
                n_failure=n_failure,
                match_prices=match_prices,
                success_rate=success_rate,
                match_dispersion=match_dispersion
            )
            results.append(probe_result)
        
        # 记录本 tick 所有成交价格（用于 collapse_proxy 计算）
        self.match_prices_history.append(all_match_prices)
        if len(self.match_prices_history) > 100:  # 保留最近 100 ticks
            self.match_prices_history.pop(0)
        
        return results
    
    def adjust_participation(self):
        """根据平均体验调整玩家数量"""
        if len(self.active_players) == 0:
            return
            
        avg_exp = np.mean([p.experience_score for p in self.active_players])
        
        print(f"调整时刻: 玩家数={len(self.active_players)}, 平均体验={avg_exp:.3f}")
        
        # 最小玩家保护机制
        if len(self.active_players) <= 2 and avg_exp > 0.25:
            new_player = RandomExperiencePlayer(
                len(self.active_players),
                probe_count=self.probe_count,
                probe_count_random=self.probe_count_random
            )
            self.active_players.append(new_player)
            print(f"  ⚠️  激活保护机制，加人！当前体验:{avg_exp:.3f}, 总数 {len(self.active_players)}")
            return
        
        # 加人逻辑
        if avg_exp > self.ADD_PLAYER_THRESHOLD and len(self.active_players) < self.MAX_N:
            new_player = RandomExperiencePlayer(
                len(self.active_players),
                probe_count=self.probe_count,
                probe_count_random=self.probe_count_random
            )
            self.active_players.append(new_player)
            print(f"  → 新增玩家，总数 {len(self.active_players)}")
            
        # 减人逻辑
        elif avg_exp < self.REMOVE_PLAYER_THRESHOLD and len(self.active_players) > 2:
            worst_idx = np.argmin([p.experience_score for p in self.active_players])
            removed = self.active_players.pop(worst_idx)
            print(f"  → 移除玩家 {removed.id} (体验{removed.experience_score:.3f})")
    
    def run_simulation(self) -> dict:
        """
        V5.2 完整仿真
        
        关键变更：
        1. 使用 generate_probes 生成 K 个探针
        2. 使用 sample_probes 进行多探针成交判定
        3. 设置 ProbeResult 后更新体验
        """
        version_str = "V5.2" if self.probe_count > 1 else "V5.0/5.2"
        print(f"=== {version_str} 仿真开始 ===")
        print(f"规则: 最简价格优先 + 0.05%固定费")
        print(f"探针: K={self.probe_count} {'(随机)' if self.probe_count_random else '(固定)'}")
        print(f"时长: {self.ticks} ticks, 参与调整: 每{self.adjust_interval} ticks")
        print("-" * 60)
        
        # 初始状态
        s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
        
        start_time = time.time()
        
        for t in range(self.ticks):
            # 更新混乱因子管理器的tick
            self.chaos_factor.update_tick(t)
            
            # V5.2: 更新 chaos_factor 的成交价格和活跃簇占比
            if self.match_prices_history:
                # 使用最近的成交价格
                recent_prices = [p for prices in self.match_prices_history[-10:] for p in prices]
                self.chaos_factor.set_match_prices(recent_prices)
            
            # 从 structure_metrics 获取活跃簇占比
            cluster_info = self.structure_metrics.get_cluster_info()
            if cluster_info['n_clusters'] > 0:
                active_ratio = cluster_info['active_protocols'] / cluster_info['n_clusters']
                self.chaos_factor.set_active_protocols_ratio(active_ratio)
            
            # V5.2: 所有玩家生成探针
            all_probes = [p.generate_probes(s_t) for p in self.active_players]
            
            # V5.2: 多探针成交判定
            probe_results = self.sample_probes(all_probes, s_t)
            
            # 收集所有成交的 action（用于状态更新）
            all_actions = [probe for probes in all_probes for probe in probes]
            all_matches = []
            for pr in probe_results:
                for i, probe in enumerate(pr.probes):
                    if i < pr.n_success:  # 前 n_success 个视为成交
                        all_matches.append(probe)
            
            # V5.2: 设置探针结果并更新体验
            for player, probe_result in zip(self.active_players, probe_results):
                player.set_probe_result(probe_result)
                player.update_experience(True, s_t)  # matched 参数在 v5.2 中被忽略
            
            # 状态演化
            s_t = self.engine.update(s_t, all_actions, all_matches)
            
            # 更新结构密度计算
            state_features = np.array([
                s_t.price_norm, s_t.volatility, 
                s_t.liquidity, s_t.imbalance
            ])
            self.structure_metrics.update_trajectory(state_features)
            
            # 降低复杂度计算频率
            if t % 500 == 0 and t >= 200:
                complexity = self.structure_metrics.compute_complexity()
                s_t.complexity = complexity
                self.complexity_history.append(complexity)
            else:
                if len(self.complexity_history) > 0:
                    s_t.complexity = self.complexity_history[-1]
                self.complexity_history.append(s_t.complexity)
            
            # 记录
            self.state_trajectory.append((
                s_t.price_norm, s_t.volatility, 
                s_t.liquidity, s_t.imbalance
            ))
            
            # 进度输出
            if t % 5000 == 0:
                avg_exp = np.mean([p.experience_score for p in self.active_players])
                total_probes = sum(len(probes) for probes in all_probes)
                total_matches = sum(pr.n_success for pr in probe_results)
                print(f"t={t:5d} N={len(self.active_players):2d} "
                      f"exp={avg_exp:.3f} liq={s_t.liquidity:.3f} "
                      f"cpx={s_t.complexity:.3f} probes={total_probes} matches={total_matches}")
            
            # 参与调整
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
    print("=== V5.2 集成测试 ===\n")
    
    # 测试1: K=1 模式（与 v5.0 兼容）
    print("--- 测试1: K=1 单探针模式 ---")
    sim1 = V5MarketSimulator(ticks=10000, adjust_interval=2000, probe_count=1)
    metrics1 = sim1.run_simulation()
    
    # 测试2: K=3 多探针模式
    print("\n--- 测试2: K=3 多探针模式 ---")
    sim2 = V5MarketSimulator(ticks=10000, adjust_interval=2000, probe_count=3)
    metrics2 = sim2.run_simulation()
    
    # 对比结果
    print("\n=== 对比结果 ===")
    print(f"K=1: 最终玩家={metrics1['final_player_count']}, 平均体验={metrics1['final_avg_experience']:.3f}, 复杂度={metrics1['final_complexity']:.3f}")
    print(f"K=3: 最终玩家={metrics2['final_player_count']}, 平均体验={metrics2['final_avg_experience']:.3f}, 复杂度={metrics2['final_complexity']:.3f}")
    
    print("\nV5.2 集成测试完成！")

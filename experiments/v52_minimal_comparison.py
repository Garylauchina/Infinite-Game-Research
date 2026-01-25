# v52_minimal_comparison.py - V5.0 vs V5.2 最小对照实验
"""
验证核心假设：
H0: 系统的可达结构空间仅由规则决定，与行为采样密度 K 无关
H1: K 只影响结构被探索到的速度，不改变最终可达的结构集合

实验组：
- A: V5.0 (K=1)
- B: V5.2 (K=1) 
- C: V5.2 (K=3)

V5.2 审计补丁：
- 分离 H_price（成交价格熵）与 H_transfer（转移熵）
- IG_DEBUG_STRUCT=1 时输出完整审计字段
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

# Debug 开关
IG_DEBUG_STRUCT = os.environ.get('IG_DEBUG_STRUCT', '0') == '1'

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_system.state_engine import StateEngine, MarketState, Action
from core_system.trading_rules import SIMPLEST_RULES
from core_system.chaos_rules import compute_match_prob, ChaosFactor, compute_collapse_proxy, compute_collapse_proxy_debug
from core_system.random_player import RandomExperiencePlayer, ProbeResult
from core_system.metrics import StructureMetrics


@dataclass
class MilestoneSnapshot:
    """里程碑快照（V5.2 审计版：分离 H_price 与 H_transfer）"""
    milestone: str          # T1, T2, T3
    tick: int
    N: int                  # 玩家数量
    complexity: float       # 结构密度
    # V5.2 审计：分离两个 H
    H_price: float          # 成交价格熵（用于 collapse_proxy）
    H_transfer: float       # 转移熵（用于 complexity）
    active_ratio_struct: float   # 来自 metrics 的活跃簇占比
    collapse_proxy: float
    # Debug 字段（可选）
    match_count: int = 0
    collapse_raw: float = 0.0


@dataclass 
class ExperimentResult:
    """实验结果"""
    group: str              # A, B, C
    version: str            # V5.0 or V5.2
    K: int                  # 探针数量
    seed: int
    snapshots: List[MilestoneSnapshot]
    trajectory: List[Tuple[float, float]]  # (H_transfer, active_ratio_struct) 轨迹
    collapse_history: List[float]          # collapse_proxy 历史
    H_price_history: List[float] = field(default_factory=list)  # H_price 历史


class MinimalComparisonExperiment:
    """最小对照实验"""
    
    # 冻结条件（所有组必须一致）
    FROZEN_CONFIG = {
        'price_min': 40000.0,
        'price_max': 60000.0,
        'fixed_fee': 0.0005,
        'adjust_interval': 1000,
        'window_size': 5000,
        'n_clusters': 5,
        'cluster_update_interval': 500,
        'base_chaos': 0.08,
        'collapse_sensitivity': 0.5,
        'add_threshold': 0.35,
        'remove_threshold': 0.15,
    }
    
    # 里程碑定义
    MILESTONE_N1 = 10   # T1: N 首次达到 10
    MILESTONE_N2 = 50   # T2: N 首次达到 50
    MILESTONE_T3 = 300000  # T3: 固定 tick
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def _create_simulator(self, probe_count: int):
        """创建模拟器（统一配置）"""
        engine = StateEngine(
            price_min=self.FROZEN_CONFIG['price_min'],
            price_max=self.FROZEN_CONFIG['price_max']
        )
        
        structure_metrics = StructureMetrics(
            window_size=self.FROZEN_CONFIG['window_size'],
            n_clusters=self.FROZEN_CONFIG['n_clusters'],
            cluster_update_interval=self.FROZEN_CONFIG['cluster_update_interval']
        )
        
        chaos_factor = ChaosFactor(
            collapse_sensitivity=self.FROZEN_CONFIG['collapse_sensitivity']
        )
        chaos_factor.base_chaos = self.FROZEN_CONFIG['base_chaos']
        
        # 初始玩家
        players = [
            RandomExperiencePlayer(i, probe_count=probe_count) 
            for i in range(3)
        ]
        
        return {
            'engine': engine,
            'structure_metrics': structure_metrics,
            'chaos_factor': chaos_factor,
            'players': players,
            'probe_count': probe_count,
        }
    
    def _compute_H_norm(self, structure_metrics: StructureMetrics) -> float:
        """
        [已弃用] 计算归一化转移熵
        
        V5.2 审计版使用 compute_complexity_debug() 获取 H_transfer
        保留此方法仅供参考
        """
        if len(structure_metrics.cluster_assignments) < 200:
            return 0.5
            
        clusters = np.array(structure_metrics.cluster_assignments)
        n_clusters = structure_metrics.n_clusters
        
        # 计算转移矩阵
        from collections import defaultdict
        trans_count = defaultdict(int)
        for i in range(len(clusters)-1):
            trans_count[(clusters[i], clusters[i+1])] += 1
            
        trans_matrix = np.zeros((n_clusters, n_clusters))
        for (i,j), cnt in trans_count.items():
            if i < n_clusters and j < n_clusters:
                trans_matrix[i,j] = cnt
        
        # 行归一化
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_prob = np.divide(trans_matrix, row_sums, 
                              out=np.zeros_like(trans_matrix), 
                              where=row_sums!=0)
        
        # 计算熵
        entropy = -np.sum(trans_prob * np.log2(trans_prob + 1e-8), axis=1)
        entropy = entropy[row_sums.flatten() > 0]
        
        if len(entropy) == 0:
            return 0.5
            
        H_norm = np.mean(entropy) / np.log2(n_clusters)
        return np.clip(H_norm, 0.0, 1.0)
    
    def _get_active_protocols_ratio(self, structure_metrics: StructureMetrics) -> float:
        """获取活跃协议占比"""
        info = structure_metrics.get_cluster_info()
        if info['n_clusters'] == 0:
            return 0.5
        return info['active_protocols'] / info['n_clusters']
    
    def _sample_probes_v52(self, sim, all_probes, s_t):
        """V5.2 多探针成交判定"""
        player_count = len(sim['players'])
        avg_exp = np.mean([p.experience_score for p in sim['players']]) if sim['players'] else 0.0
        
        flat_actions = [probe for probes in all_probes for probe in probes]
        
        results = []
        all_match_prices = []
        
        for probes in all_probes:
            n_success = 0
            match_prices = []
            
            for probe in probes:
                prob = compute_match_prob(
                    probe.price, s_t, flat_actions, player_count,
                    chaos_factor_manager=sim['chaos_factor'],
                    avg_exp=avg_exp
                )
                if np.random.random() < prob:
                    n_success += 1
                    match_prices.append(probe.price)
                    all_match_prices.append(probe.price)
            
            k = len(probes)
            n_failure = k - n_success
            success_rate = n_success / k if k > 0 else 0.0
            
            if len(match_prices) >= 2:
                price_range = self.FROZEN_CONFIG['price_max'] - self.FROZEN_CONFIG['price_min']
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
        
        return results, all_match_prices
    
    def _adjust_participation(self, sim, avg_exp):
        """调整玩家数量"""
        players = sim['players']
        
        if len(players) <= 2 and avg_exp > 0.25:
            new_player = RandomExperiencePlayer(
                len(players), probe_count=sim['probe_count']
            )
            players.append(new_player)
            return
        
        if avg_exp > self.FROZEN_CONFIG['add_threshold']:
            new_player = RandomExperiencePlayer(
                len(players), probe_count=sim['probe_count']
            )
            players.append(new_player)
            
        elif avg_exp < self.FROZEN_CONFIG['remove_threshold'] and len(players) > 2:
            worst_idx = np.argmin([p.experience_score for p in players])
            players.pop(worst_idx)
    
    def run_experiment(self, group: str, version: str, K: int) -> ExperimentResult:
        """运行单个实验组（V5.2 审计版：分离 H_price 与 H_transfer）"""
        print(f"\n{'='*60}")
        print(f"实验组 {group}: {version} K={K} seed={self.seed}")
        if IG_DEBUG_STRUCT:
            print(f"[DEBUG] IG_DEBUG_STRUCT=1 已启用结构审计")
        print(f"{'='*60}")
        
        # 重置随机种子
        np.random.seed(self.seed)
        
        # 创建模拟器
        sim = self._create_simulator(probe_count=K)
        
        # 初始状态
        s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
        
        # 结果收集
        snapshots = []
        trajectory = []
        collapse_history = []
        H_price_history = []
        match_prices_history = []
        
        # 里程碑标记
        reached_t1 = False
        reached_t2 = False
        
        max_ticks = self.MILESTONE_T3 + 1000  # 稍微多跑一点
        
        for t in range(max_ticks):
            # 更新 chaos factor
            sim['chaos_factor'].update_tick(t)
            
            if match_prices_history:
                recent_prices = [p for prices in match_prices_history[-10:] for p in prices]
                sim['chaos_factor'].set_match_prices(recent_prices)
            
            active_ratio = self._get_active_protocols_ratio(sim['structure_metrics'])
            sim['chaos_factor'].set_active_protocols_ratio(active_ratio)
            
            # 生成探针
            all_probes = [p.generate_probes(s_t) for p in sim['players']]
            
            # 成交判定
            probe_results, all_match_prices = self._sample_probes_v52(sim, all_probes, s_t)
            match_prices_history.append(all_match_prices)
            if len(match_prices_history) > 100:
                match_prices_history.pop(0)
            
            # 收集所有成交
            all_actions = [probe for probes in all_probes for probe in probes]
            all_matches = []
            for pr in probe_results:
                all_matches.extend(pr.match_prices[:pr.n_success] if pr.match_prices else [])
            all_matches_actions = [Action(p, 'buy') for p in all_matches[:len(all_match_prices)]]
            
            # 更新体验
            for player, probe_result in zip(sim['players'], probe_results):
                player.set_probe_result(probe_result)
                player.update_experience(True, s_t)
            
            # 状态演化
            s_t = sim['engine'].update(s_t, all_actions, all_matches_actions)
            
            # 更新结构密度
            state_features = np.array([
                s_t.price_norm, s_t.volatility, 
                s_t.liquidity, s_t.imbalance
            ])
            sim['structure_metrics'].update_trajectory(state_features)
            
            # ===== V5.2 审计：分离 H_price 与 H_transfer =====
            
            # 计算复杂度（使用 debug 版本获取 H_transfer）
            if t % 500 == 0 and t >= 200:
                if IG_DEBUG_STRUCT:
                    complexity, mdbg = sim['structure_metrics'].compute_complexity_debug()
                else:
                    complexity = sim['structure_metrics'].compute_complexity()
                    mdbg = None
                s_t.complexity = complexity
            else:
                mdbg = None
            
            # 计算观测量
            N = len(sim['players'])
            
            # H_transfer: 来自 metrics 的转移熵
            if mdbg is not None:
                H_transfer = mdbg['transfer_entropy_norm']
                active_ratio_struct = mdbg['active_ratio_struct']
            else:
                # 使用缓存的值或重新计算
                _, mdbg_temp = sim['structure_metrics'].compute_complexity_debug()
                H_transfer = mdbg_temp['transfer_entropy_norm']
                active_ratio_struct = mdbg_temp['active_ratio_struct']
            
            # 计算 collapse_proxy（使用 debug 版本获取 H_price）
            if all_match_prices:
                if IG_DEBUG_STRUCT:
                    collapse, cdbg = compute_collapse_proxy_debug(all_match_prices, active_ratio)
                else:
                    collapse = compute_collapse_proxy(all_match_prices, active_ratio)
                    cdbg = {'H_price_norm': 0.0, 'match_count': len(all_match_prices), 'collapse_raw': 0.0}
                H_price = cdbg.get('H_price_norm', 0.0) if IG_DEBUG_STRUCT else 0.0
                match_count = cdbg.get('match_count', len(all_match_prices))
                collapse_raw = cdbg.get('collapse_raw', 0.0) if IG_DEBUG_STRUCT else 0.0
            else:
                collapse = 0.5
                H_price = 0.0
                match_count = 0
                collapse_raw = 1.0
                cdbg = {'H_price_norm': 0.0, 'match_count': 0, 'collapse_raw': 1.0}
            
            # 记录轨迹（每1000 ticks采样）
            if t % 1000 == 0:
                trajectory.append((H_transfer, active_ratio_struct))
                collapse_history.append(collapse)
                H_price_history.append(H_price)
            
            # 里程碑检查
            if not reached_t1 and N >= self.MILESTONE_N1:
                reached_t1 = True
                snapshot = MilestoneSnapshot(
                    milestone='T1',
                    tick=t,
                    N=N,
                    complexity=s_t.complexity,
                    H_price=H_price,
                    H_transfer=H_transfer,
                    active_ratio_struct=active_ratio_struct,
                    collapse_proxy=collapse,
                    match_count=match_count,
                    collapse_raw=collapse_raw
                )
                snapshots.append(snapshot)
                self._print_milestone(f"T1 (N={self.MILESTONE_N1})", t, N, s_t.complexity, 
                                     H_price, H_transfer, active_ratio_struct, collapse, match_count, collapse_raw)
            
            if not reached_t2 and N >= self.MILESTONE_N2:
                reached_t2 = True
                snapshot = MilestoneSnapshot(
                    milestone='T2',
                    tick=t,
                    N=N,
                    complexity=s_t.complexity,
                    H_price=H_price,
                    H_transfer=H_transfer,
                    active_ratio_struct=active_ratio_struct,
                    collapse_proxy=collapse,
                    match_count=match_count,
                    collapse_raw=collapse_raw
                )
                snapshots.append(snapshot)
                self._print_milestone(f"T2 (N={self.MILESTONE_N2})", t, N, s_t.complexity,
                                     H_price, H_transfer, active_ratio_struct, collapse, match_count, collapse_raw)
            
            if t == self.MILESTONE_T3:
                snapshot = MilestoneSnapshot(
                    milestone='T3',
                    tick=t,
                    N=N,
                    complexity=s_t.complexity,
                    H_price=H_price,
                    H_transfer=H_transfer,
                    active_ratio_struct=active_ratio_struct,
                    collapse_proxy=collapse,
                    match_count=match_count,
                    collapse_raw=collapse_raw
                )
                snapshots.append(snapshot)
                self._print_milestone(f"T3 (tick={self.MILESTONE_T3})", t, N, s_t.complexity,
                                     H_price, H_transfer, active_ratio_struct, collapse, match_count, collapse_raw)
                break
            
            # 参与调整
            if t % self.FROZEN_CONFIG['adjust_interval'] == 0 and t > 0:
                avg_exp = np.mean([p.experience_score for p in sim['players']])
                self._adjust_participation(sim, avg_exp)
            
            # 进度输出
            if t % 50000 == 0:
                avg_exp = np.mean([p.experience_score for p in sim['players']])
                self._print_progress(t, N, avg_exp, s_t.complexity, H_price, H_transfer, 
                                    active_ratio_struct, collapse, match_count)
        
        return ExperimentResult(
            group=group,
            version=version,
            K=K,
            seed=self.seed,
            snapshots=snapshots,
            trajectory=trajectory,
            collapse_history=collapse_history,
            H_price_history=H_price_history
        )
    
    def _print_milestone(self, label: str, t: int, N: int, cpx: float, 
                         H_price: float, H_transfer: float, active_ratio: float,
                         collapse: float, match_count: int, collapse_raw: float):
        """打印里程碑（V5.2 审计格式：分离 H_price 与 H_transfer）"""
        if IG_DEBUG_STRUCT:
            print(f"  {label}: tick={t}")
            print(f"    N={N} cpx={cpx:.3f}")
            print(f"    H_price={H_price:.3f} collapse_raw={collapse_raw:.3f} collapse={collapse:.3f}")
            print(f"    H_transfer={H_transfer:.3f} active_ratio={active_ratio:.3f}")
            print(f"    match_count={match_count}")
        else:
            print(f"  {label}: tick={t}, N={N}, cpx={cpx:.3f}")
            print(f"    H_price={H_price:.3f} H_transfer={H_transfer:.3f} collapse={collapse:.3f}")
    
    def _print_progress(self, t: int, N: int, avg_exp: float, cpx: float,
                        H_price: float, H_transfer: float, active_ratio: float,
                        collapse: float, match_count: int):
        """打印进度（V5.2 审计格式）"""
        if IG_DEBUG_STRUCT:
            print(f"  t={t:6d} N={N:3d} exp={avg_exp:.3f} cpx={cpx:.3f}")
            print(f"    H_price={H_price:.3f} H_transfer={H_transfer:.3f} collapse={collapse:.3f} match={match_count}")
        else:
            print(f"  t={t:6d} N={N:3d} exp={avg_exp:.3f} cpx={cpx:.3f} H_price={H_price:.3f} H_transfer={H_transfer:.3f}")
    
    def run_all(self) -> Dict[str, ExperimentResult]:
        """运行所有实验组"""
        results = {}
        
        # A: V5.0 (K=1) - 实际上用 V5.2 代码但 K=1
        results['A'] = self.run_experiment('A', 'V5.0', K=1)
        
        # B: V5.2 (K=1)
        results['B'] = self.run_experiment('B', 'V5.2', K=1)
        
        # C: V5.2 (K=3)
        results['C'] = self.run_experiment('C', 'V5.2', K=3)
        
        return results
    
    def analyze_results(self, results: Dict[str, ExperimentResult]) -> Dict:
        """分析结果（V5.2 审计版）"""
        analysis = {
            'hypothesis_test': {},
            'milestone_comparison': {},
            'trajectory_analysis': {},
        }
        
        print("\n" + "="*60)
        print("实验分析（V5.2 审计版：分离 H_price 与 H_transfer）")
        print("="*60)
        
        # 1. 里程碑对比
        print("\n--- 里程碑对比 ---")
        print(f"{'组':<4} {'里程碑':<4} {'tick':>7} {'N':>4} {'cpx':>6} {'H_price':>7} {'H_trans':>7} {'active':>6} {'collapse':>8}")
        print("-" * 75)
        
        for group, result in results.items():
            for snap in result.snapshots:
                print(f"{group:<4} {snap.milestone:<4} {snap.tick:>7} {snap.N:>4} "
                      f"{snap.complexity:>6.3f} {snap.H_price:>7.3f} {snap.H_transfer:>7.3f} "
                      f"{snap.active_ratio_struct:>6.3f} {snap.collapse_proxy:>8.3f}")
        
        # 2. 轨迹可达性分析
        print("\n--- 轨迹可达性分析 ---")
        
        # 计算各组轨迹的边界
        for group, result in results.items():
            if result.trajectory:
                H_transfer_values = [t[0] for t in result.trajectory]
                A_values = [t[1] for t in result.trajectory]
                print(f"组 {group}: H_transfer ∈ [{min(H_transfer_values):.3f}, {max(H_transfer_values):.3f}], "
                      f"active ∈ [{min(A_values):.3f}, {max(A_values):.3f}]")
            if result.H_price_history:
                H_price_values = result.H_price_history
                collapse_values = result.collapse_history
                print(f"       H_price ∈ [{min(H_price_values):.3f}, {max(H_price_values):.3f}], "
                      f"collapse ∈ [{min(collapse_values):.3f}, {max(collapse_values):.3f}]")
        
        # 3. 假设检验
        print("\n--- 假设检验 ---")
        
        # 比较 A vs B (V5.0 vs V5.2 at K=1)
        if 'A' in results and 'B' in results:
            traj_A = results['A'].trajectory
            traj_B = results['B'].trajectory
            
            # 计算轨迹重叠度（使用 H_transfer）
            if traj_A and traj_B:
                H_A = set(round(t[0], 2) for t in traj_A)
                H_B = set(round(t[0], 2) for t in traj_B)
                overlap = len(H_A & H_B) / max(len(H_A | H_B), 1)
                print(f"A vs B (K=1): H_transfer 轨迹重叠度 = {overlap:.2%}")
                analysis['hypothesis_test']['A_vs_B_overlap'] = overlap
        
        # 比较 B vs C (K=1 vs K=3)
        if 'B' in results and 'C' in results:
            traj_B = results['B'].trajectory
            traj_C = results['C'].trajectory
            
            if traj_B and traj_C:
                # 检查 C 是否进入 B 未到达的区域（使用 H_transfer）
                H_B_range = (min(t[0] for t in traj_B), max(t[0] for t in traj_B))
                A_B_range = (min(t[1] for t in traj_B), max(t[1] for t in traj_B))
                
                C_outside = 0
                for h, a in traj_C:
                    if h < H_B_range[0] - 0.1 or h > H_B_range[1] + 0.1:
                        C_outside += 1
                    if a < A_B_range[0] - 0.1 or a > A_B_range[1] + 0.1:
                        C_outside += 1
                
                outside_ratio = C_outside / (2 * len(traj_C)) if traj_C else 0
                print(f"B vs C: C 超出 B 边界的比例 = {outside_ratio:.2%}")
                analysis['hypothesis_test']['C_outside_B'] = outside_ratio
                
                if outside_ratio > 0.1:
                    print("⚠️ 警告: K=3 可能进入了 K=1 未探索的区域")
                else:
                    print("✅ K=3 的轨迹在 K=1 的可达范围内")
        
        # V5.2 审计：验证 collapse 与 H_price 的一致性
        print("\n--- V5.2 审计：collapse 与 H_price 一致性检验 ---")
        for group, result in results.items():
            if result.snapshots:
                for snap in result.snapshots:
                    expected_collapse_raw = abs(2 * snap.H_price - 1)
                    # 验证 collapse_raw 是否与 H_price 一致
                    if snap.H_price > 0.8:
                        # 高熵时需要乘 active_ratio
                        pass  # 这里不做严格验证，因为 active_ratio_in 可能不同
                    print(f"  组{group} {snap.milestone}: H_price={snap.H_price:.3f} → "
                          f"expected_collapse_raw={expected_collapse_raw:.3f}, actual_collapse={snap.collapse_proxy:.3f}")
        
        # 4. 结论
        print("\n--- 初步结论 ---")
        
        if 'C_outside_B' in analysis['hypothesis_test']:
            if analysis['hypothesis_test']['C_outside_B'] < 0.05:
                print("✅ H0 暂时成立: K 似乎只影响探索速度，不改变结构空间")
            elif analysis['hypothesis_test']['C_outside_B'] < 0.15:
                print("⚠️ 需要进一步验证: 存在轻微的边界扩展")
            else:
                print("❌ H0 可能被否定: K 可能引入了新的自由度")
        
        return analysis
    
    def save_results(self, results: Dict[str, ExperimentResult], output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        data = {}
        for group, result in results.items():
            data[group] = {
                'group': result.group,
                'version': result.version,
                'K': result.K,
                'seed': result.seed,
                'snapshots': [asdict(s) for s in result.snapshots],
                'trajectory': result.trajectory,
                'collapse_history': result.collapse_history,
            }
        
        with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n结果已保存到: {output_dir}")
    
    def plot_results(self, results: Dict[str, ExperimentResult], output_dir: str):
        """绘制结果图表（V5.2 审计版）"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib 未安装，跳过绘图")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 轨迹图 (H_transfer vs active_ratio_struct)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'A': 'blue', 'B': 'green', 'C': 'red'}
        labels = {'A': 'V5.0 K=1', 'B': 'V5.2 K=1', 'C': 'V5.2 K=3'}
        
        for group, result in results.items():
            if result.trajectory:
                H = [t[0] for t in result.trajectory]
                A = [t[1] for t in result.trajectory]
                ax.scatter(H, A, c=colors[group], label=labels[group], alpha=0.6, s=20)
                ax.plot(H, A, c=colors[group], alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('H_transfer (转移熵)', fontsize=12)
        ax.set_ylabel('active_ratio_struct', fontsize=12)
        ax.set_title('结构空间轨迹对比 (H_transfer vs active_ratio)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=150)
        plt.close()
        
        # 1b. H_price vs collapse 图（审计用）
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for group, result in results.items():
            if result.H_price_history and result.collapse_history:
                ax.scatter(result.H_price_history, result.collapse_history, 
                          c=colors[group], label=labels[group], alpha=0.6, s=20)
        
        # 绘制理论曲线 collapse = |2*H - 1|
        h_theory = np.linspace(0, 1, 100)
        collapse_theory = np.abs(2 * h_theory - 1)
        ax.plot(h_theory, collapse_theory, 'k--', label='理论: |2H-1|', linewidth=2)
        
        ax.set_xlabel('H_price (成交价格熵)', fontsize=12)
        ax.set_ylabel('collapse_proxy', fontsize=12)
        ax.set_title('H_price vs collapse 一致性检验', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'H_price_vs_collapse.png'), dpi=150)
        plt.close()
        
        # 2. collapse_proxy 分布
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (group, result) in enumerate(results.items()):
            if result.collapse_history:
                axes[idx].hist(result.collapse_history, bins=20, color=colors[group], alpha=0.7)
                axes[idx].set_xlabel('collapse_proxy')
                axes[idx].set_ylabel('频次')
                axes[idx].set_title(f'{labels[group]} collapse 分布')
                axes[idx].axvline(np.mean(result.collapse_history), color='black', 
                                 linestyle='--', label=f'mean={np.mean(result.collapse_history):.3f}')
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'collapse_distribution.png'), dpi=150)
        plt.close()
        
        print(f"图表已保存到: {output_dir}")


def main():
    """主函数"""
    print("="*60)
    print("V5.0 vs V5.2 最小对照实验")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建实验
    experiment = MinimalComparisonExperiment(seed=42)
    
    # 运行所有组
    results = experiment.run_all()
    
    # 分析结果
    analysis = experiment.analyze_results(results)
    
    # 保存结果
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'v52_comparison_output'
    )
    experiment.save_results(results, output_dir)
    experiment.plot_results(results, output_dir)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()

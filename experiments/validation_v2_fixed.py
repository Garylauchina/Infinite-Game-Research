#!/usr/bin/env python3
"""
复杂系统验证器 v2.1 - 修复版
修复项：
P0-1: nonlinearity测试 - 保存response_curve数据和诊断图
P0-2: reflexivity_population - 多滞后和分箱分析
P1: 线程控制（复现性）
P2: 多尺度complexity观测
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import warnings
import os
from pathlib import Path
import time
from collections import defaultdict
import sys

# P1: 设置线程为1（复现性）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 配置matplotlib缓存目录
cache_dir = Path('.matplotlib_cache')
cache_dir.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache_dir.absolute())

warnings.filterwarnings('ignore')

# P4: 统一导入方式（相对导入）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from core_system import V5MarketSimulator
try:
    from multi_scale_complexity import compute_complexity_sensitivity
except ImportError:
    # 如果直接导入失败，尝试从experiments导入
    import sys
    from pathlib import Path
    experiments_dir = Path(__file__).parent
    sys.path.insert(0, str(experiments_dir))
    from multi_scale_complexity import compute_complexity_sensitivity

class ComplexSystemValidatorV2Fixed:
    def __init__(self, n_seeds=20, ticks=500000, output_dir='validation_output_v2_fixed'):
        self.n_seeds = n_seeds
        self.ticks = ticks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.ADD_PLAYER_THRESHOLD = 0.35
        self.REMOVE_PLAYER_THRESHOLD = 0.15
    
    def run_simulation_wrapper(self, seed=0, fixed_N=None, ticks=None):
        """运行V5.0模拟器并返回关键数据"""
        print(f"  运行seed {seed}...", end=' ', flush=True)
        
        # 设置随机种子
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        actual_ticks = ticks if ticks is not None else self.ticks
        
        sim = V5MarketSimulator(ticks=actual_ticks, adjust_interval=2000, MAX_N=None)
        
        if fixed_N is not None:
            from core_system.random_player import RandomExperiencePlayer
            original_adjust = sim.adjust_participation
            def fixed_adjust():
                current_N = len(sim.active_players)
                if current_N < fixed_N:
                    while len(sim.active_players) < fixed_N:
                        sim.active_players.append(
                            RandomExperiencePlayer(len(sim.active_players))
                        )
                elif current_N > fixed_N:
                    sim.active_players = sim.active_players[:fixed_N]
            sim.adjust_participation = fixed_adjust
        
        # 静默运行
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            metrics = sim.run_simulation()
        
        trajectory = np.array(sim.state_trajectory)
        player_history = np.array(sim.player_history) if sim.player_history else np.array([3])
        experience_history = np.array(sim.experience_history) if sim.experience_history else np.array([0.0])
        complexity_history = np.array(sim.complexity_history) if sim.complexity_history else np.array([0.0])
        
        actual_ticks = len(trajectory)
        metrics['ticks_completed'] = actual_ticks
        metrics['last_adjust_tick'] = (len(player_history) - 1) * sim.adjust_interval if len(player_history) > 0 else 0
        metrics['max_player_count'] = int(np.max(player_history)) if len(player_history) > 0 else len(sim.active_players)
        
        print("✓")
        return trajectory, player_history, experience_history, complexity_history, metrics
    
    def run_n_sweep(self, seed, N_list, sweep_ticks=100000):
        """对给定seed运行N sweep，构造真实的响应曲线数据"""
        results_by_N = {}
        
        print(f"    N sweep: {len(N_list)}个N值...", end=' ', flush=True)
        
        for N in N_list:
            traj, _, exp_hist, complexity_hist, metrics = self.run_simulation_wrapper(
                seed=seed, fixed_N=N, ticks=sweep_ticks
            )
            
            # 使用稳态均值（后50%的数据）
            burn_in = len(exp_hist) // 2
            if burn_in < len(exp_hist):
                steady_exp = np.mean(exp_hist[burn_in:])
                steady_liq = np.mean(traj[burn_in:, 2])
                steady_cpx = np.mean(complexity_hist[burn_in:]) if len(complexity_hist) > burn_in else metrics.get('avg_complexity', 0.0)
            else:
                steady_exp = metrics.get('final_avg_experience', 0.0)
                steady_liq = metrics.get('avg_liquidity', 0.0)
                steady_cpx = metrics.get('avg_complexity', 0.0)
            
            results_by_N[N] = {
                'avg_experience': steady_exp,
                'avg_liquidity': steady_liq,
                'avg_complexity': steady_cpx
            }
        
        print("✓")
        return results_by_N
    
    def test_nonlinearity_response_curve(self, results_by_N, seed, output_dir):
        """
        P0-1修复：基于响应曲线的非线性测试 + 保存数据和诊断图
        """
        if len(results_by_N) < 5:
            return 0.0
        
        N_values = np.array(sorted(results_by_N.keys()), dtype=float)
        
        def get_series(key):
            return np.array([results_by_N[N][key] for N in N_values], dtype=float)
        
        series_list = [
            (get_series('avg_experience'), 'experience'),
            (get_series('avg_liquidity'), 'liquidity'),
            (get_series('avg_complexity'), 'complexity')
        ]
        
        scores = []
        fit_results = []
        
        for y, name in series_list:
            if np.std(y) < 1e-6:
                continue
            
            # 线性模型
            X1 = np.column_stack([N_values])
            reg1 = LinearRegression().fit(X1, y)
            r2_1 = reg1.score(X1, y)
            y_pred_1 = reg1.predict(X1)
            
            # 二次模型
            X2 = np.column_stack([N_values, N_values**2])
            reg2 = LinearRegression().fit(X2, y)
            r2_2 = reg2.score(X2, y)
            y_pred_2 = reg2.predict(X2)
            
            # 调整后的R²
            n = len(y)
            p1, p2 = 1, 2
            adj1 = 1 - (1 - r2_1) * (n - 1) / max(n - p1 - 1, 1)
            adj2 = 1 - (1 - r2_2) * (n - 1) / max(n - p2 - 1, 1)
            
            gain = max(0.0, adj2 - adj1)
            score = 1 - np.exp(-gain / 0.05)
            scores.append(score)
            
            # 保存拟合结果
            fit_results.append({
                'series': name,
                'N_values': N_values,
                'y_values': y,
                'linear_r2': r2_1,
                'quadratic_r2': r2_2,
                'linear_adj_r2': adj1,
                'quadratic_adj_r2': adj2,
                'gain': gain,
                'score': score,
                'y_pred_linear': y_pred_1,
                'y_pred_quadratic': y_pred_2,
                'residuals_linear': y - y_pred_1,
                'residuals_quadratic': y - y_pred_2
            })
        
        # 保存response_curve数据
        response_df = pd.DataFrame({
            'N': N_values,
            **{f'{name}_{key}': [results_by_N[N][key] for N in N_values] 
               for name in ['experience', 'liquidity', 'complexity'] 
               for key in ['avg_experience', 'avg_liquidity', 'avg_complexity'] 
               if key.replace('avg_', '') == name}
        })
        # 简化：直接保存results_by_N
        response_data = []
        for N in N_values:
            response_data.append({
                'N': N,
                'avg_experience': results_by_N[N]['avg_experience'],
                'avg_liquidity': results_by_N[N]['avg_liquidity'],
                'avg_complexity': results_by_N[N]['avg_complexity']
            })
        response_df = pd.DataFrame(response_data)
        response_df.to_csv(output_dir / f'response_curve_{seed}.csv', index=False)
        
        # 绘制诊断图
        fig, axes = plt.subplots(len(fit_results), 2, figsize=(12, 4*len(fit_results)))
        if len(fit_results) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, fit in enumerate(fit_results):
            # 左：数据点+拟合曲线
            ax = axes[idx, 0]
            ax.scatter(fit['N_values'], fit['y_values'], label='Data', alpha=0.7)
            ax.plot(fit['N_values'], fit['y_pred_linear'], '--', label=f'Linear (R²={fit["linear_r2"]:.3f})')
            ax.plot(fit['N_values'], fit['y_pred_quadratic'], '-', label=f'Quadratic (R²={fit["quadratic_r2"]:.3f})')
            ax.set_xlabel('N')
            ax.set_ylabel(fit['series'])
            ax.set_title(f'{fit["series"]}: gain={fit["gain"]:.4f}, score={fit["score"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 右：残差图
            ax = axes[idx, 1]
            ax.scatter(fit['N_values'], fit['residuals_linear'], alpha=0.7, label='Linear residuals')
            ax.scatter(fit['N_values'], fit['residuals_quadratic'], alpha=0.7, label='Quadratic residuals')
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('N')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{fit["series"]} Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'nonlinearity_diagnosis_{seed}.png', dpi=150)
        plt.close()
        
        return float(np.mean(scores)) if scores else 0.0
    
    def test_reflexivity_population(self, N_hist, exp_hist, adjust_interval=2000):
        """
        P0-2修复：增强reflexivity_population测试
        - 多滞后Δ（1,2,3个adjust周期）比较
        - P(add|E分箱) 的单调性/校准曲线
        """
        if len(N_hist) < 3 or len(exp_hist) < 3:
            return 0.0, {}
        
        # 对齐时间序列
        N_array = np.array(N_hist)
        exp_array = np.array(exp_hist)
        
        # 计算dN（玩家数量变化）
        dN = np.diff(N_array)
        
        # 使用exp_array的前len(dN)个值（因为dN少一个）
        exp_for_dN = exp_array[:len(dN)]
        
        # 基础决策一致性（原有逻辑）
        expected_actions = []
        for exp in exp_for_dN:
            if exp > self.ADD_PLAYER_THRESHOLD:
                expected_actions.append(1)  # add
            elif exp < self.REMOVE_PLAYER_THRESHOLD:
                expected_actions.append(-1)  # remove
            else:
                expected_actions.append(0)  # hold
        
        expected_actions = np.array(expected_actions)
        actual_actions = np.sign(dN)  # +1, 0, -1
        
        # 3类决策一致性
        acc = np.mean(expected_actions == actual_actions)
        acc_change = np.mean((expected_actions != 0) == (actual_actions != 0))
        
        # P0-2新增：多滞后分析
        lag_scores = []
        for lag in [1, 2, 3]:
            if len(exp_array) > lag:
                exp_lagged = exp_array[lag:len(dN)+lag] if len(exp_array) >= len(dN)+lag else exp_array[lag:]
                if len(exp_lagged) == len(dN):
                    expected_lagged = []
                    for exp in exp_lagged:
                        if exp > self.ADD_PLAYER_THRESHOLD:
                            expected_lagged.append(1)
                        elif exp < self.REMOVE_PLAYER_THRESHOLD:
                            expected_lagged.append(-1)
                        else:
                            expected_lagged.append(0)
                    expected_lagged = np.array(expected_lagged)
                    acc_lagged = np.mean(expected_lagged == actual_actions)
                    lag_scores.append(acc_lagged)
        
        lag_score = np.mean(lag_scores) if lag_scores else 0.0
        
        # P0-2新增：P(add|E分箱) 单调性和校准曲线
        # 将exp分箱
        n_bins = 10
        exp_bins = np.linspace(exp_array.min(), exp_array.max(), n_bins+1)
        bin_centers = (exp_bins[:-1] + exp_bins[1:]) / 2
        
        # 计算每个分箱的P(add)
        p_add_by_bin = []
        bin_counts = []
        for i in range(n_bins):
            mask = (exp_for_dN >= exp_bins[i]) & (exp_for_dN < exp_bins[i+1])
            if i == n_bins - 1:  # 最后一个bin包含上界
                mask = (exp_for_dN >= exp_bins[i]) & (exp_for_dN <= exp_bins[i+1])
            
            if np.sum(mask) > 0:
                adds_in_bin = np.sum((actual_actions[mask] > 0))
                p_add = adds_in_bin / np.sum(mask)
                p_add_by_bin.append(p_add)
                bin_counts.append(np.sum(mask))
            else:
                p_add_by_bin.append(0.0)
                bin_counts.append(0)
        
        # 单调性检验：P(add)应该随exp增加而增加
        p_add_array = np.array(p_add_by_bin)
        valid_mask = np.array(bin_counts) > 0
        if np.sum(valid_mask) >= 3:
            # 计算Spearman相关系数
            from scipy.stats import spearmanr
            valid_centers = bin_centers[valid_mask]
            valid_p_add = p_add_array[valid_mask]
            if len(valid_centers) > 1 and np.std(valid_p_add) > 1e-6:
                monotonicity, _ = spearmanr(valid_centers, valid_p_add)
                monotonicity = max(0, monotonicity)  # 只考虑正相关
            else:
                monotonicity = 0.0
        else:
            monotonicity = 0.0
        
        # 综合得分
        base_score = 0.5 * acc + 0.5 * acc_change
        enhanced_score = 0.4 * base_score + 0.3 * lag_score + 0.3 * monotonicity
        
        # 保存校准曲线数据（用于后续可视化）
        calibration_data = {
            'bin_centers': bin_centers.tolist(),
            'p_add_by_bin': p_add_by_bin,
            'bin_counts': bin_counts,
            'monotonicity': monotonicity,
            'lag_score': lag_score,
            'base_score': base_score
        }
        
        return float(enhanced_score), calibration_data
    
    def test_emergence(self, trajectory):
        """涌现测试：Silhouette + 转移熵"""
        if len(trajectory) < 100:
            return 0.0
        
        try:
            km = KMeans(5, random_state=42, n_init=3, max_iter=100)
            clusters = km.fit_predict(trajectory[:, :4])
            
            sil = silhouette_score(trajectory[:, :4], clusters)
            sil_score = max(sil, 0)
            
            trans_matrix = self._transition_matrix(clusters, n_states=5)
            cluster_counts = np.bincount(clusters.astype(int), minlength=5)
            pi = cluster_counts / len(clusters)
            
            row_entropies = []
            for i in range(5):
                row = trans_matrix[i, :]
                mask = row > 1e-8
                if np.any(mask):
                    H_i = -np.sum(row[mask] * np.log2(row[mask] + 1e-8))
                else:
                    H_i = 0.0
                row_entropies.append(H_i)
            
            H_weighted = np.sum(pi * np.array(row_entropies))
            max_entropy = np.log2(5)
            trans_score = min(H_weighted / max_entropy, 1.0) if max_entropy > 0 else 0.0
            
            return 0.6 * sil_score + 0.4 * trans_score
        except Exception as e:
            print(f"     涌现测试失败: {e}")
            return 0.0
    
    def test_scale_invariance(self, trajectory):
        """尺度不变性测试"""
        if len(trajectory) < 50:
            return 0.0
        
        k_values = [3, 5, 8, 13]
        sil_scores = []
        
        for k in k_values:
            if len(trajectory) < k * 2:
                continue
            try:
                km = KMeans(k, random_state=42, n_init=3, max_iter=100)
                clusters = km.fit_predict(trajectory[:, :4])
                sil = silhouette_score(trajectory[:, :4], clusters)
                sil_scores.append(sil)
            except:
                continue
        
        if len(sil_scores) < 2:
            return 0.0
        
        # 计算稳定性（标准差越小越好）
        stability = 1.0 - min(np.std(sil_scores) / (np.mean(sil_scores) + 1e-8), 1.0)
        return float(max(stability, 0.0))
    
    def _transition_matrix(self, clusters, n_states=5):
        """转移矩阵计算"""
        matrix = np.zeros((n_states, n_states))
        for i in range(len(clusters) - 1):
            if clusters[i] < n_states and clusters[i+1] < n_states:
                matrix[int(clusters[i]), int(clusters[i+1])] += 1
        
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, 
                          out=np.zeros_like(matrix), 
                          where=row_sums!=0)
        return matrix
    
    def validate_single_seed(self, seed):
        """单seed完整验证（修复版）"""
        try:
            # 运行正常模拟
            traj, N_hist, exp_hist, complexity_hist, metrics = self.run_simulation_wrapper(seed=seed)
            
            # N sweep - 使用指数间隔生成更合理的N列表
            if len(N_hist) > 0:
                min_N = max(2, int(np.min(N_hist)))
                max_N = min(200, int(np.max(N_hist)) + 10)
                # 指数间隔：2, 3, 5, 8, 13, 21, 34, ...
                N_list = [2, 3]
                while N_list[-1] < max_N:
                    next_N = N_list[-1] + N_list[-2] if len(N_list) > 1 else N_list[-1] * 2
                    if next_N <= max_N:
                        N_list.append(next_N)
                    else:
                        break
                # 确保包含max_N
                if max_N not in N_list and max_N > N_list[-1]:
                    N_list.append(max_N)
                N_list = [int(N) for N in N_list if N >= min_N and N <= max_N]
                if len(N_list) < 3:
                    # 如果范围太小，使用线性间隔
                    N_list = list(np.linspace(min_N, max_N, min(5, max_N-min_N+1), dtype=int))
            else:
                N_list = [2, 3, 5, 8, 13, 21]
            
            sweep_ticks = min(100000, self.ticks // 5)
            if sweep_ticks < 50000:
                sweep_ticks = 50000
            
            results_by_N = self.run_n_sweep(seed, N_list, sweep_ticks=sweep_ticks)
            
            # 运行测试
            nonlinearity = self.test_nonlinearity_response_curve(
                results_by_N, seed, self.output_dir
            )
            emergence = self.test_emergence(traj)
            scale_invariance = self.test_scale_invariance(traj)
            reflexivity_pop, reflexivity_calibration = self.test_reflexivity_population(N_hist, exp_hist, adjust_interval=2000)
            
            # P2: 多尺度complexity观测
            complexity_sensitivity = compute_complexity_sensitivity(traj)
            
            # 保存reflexivity校准数据
            import json
            with open(self.output_dir / f'reflexivity_calibration_{seed}.json', 'w') as f:
                json.dump(reflexivity_calibration, f, indent=2)
            
            scores = {
                'seed': seed,
                'nonlinearity': nonlinearity,
                'emergence': emergence,
                'scale_invariance': scale_invariance,
                'reflexivity': reflexivity_pop,
                'final_complexity': metrics.get('final_complexity', 0.0),
                'final_player_count': metrics.get('final_player_count', 0.0),
                **{k: v for k, v in complexity_sensitivity.items() if k.startswith('overall') or k.startswith('stability')}
            }
            return scores
        except Exception as e:
            print(f"  种子{seed}验证失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'seed': seed,
                'nonlinearity': 0.0,
                'emergence': 0.0,
                'scale_invariance': 0.0,
                'reflexivity': 0.0,
            }
    
    def run_full_validation(self):
        """批量验证"""
        print("=" * 80)
        print("🔬 复杂系统验证 v2.1 - 修复版")
        print(f"配置: {self.n_seeds} seeds, {self.ticks} ticks")
        print("=" * 80)
        
        start_time = time.time()
        
        for seed in range(self.n_seeds):
            print(f"验证种子 {seed+1}/{self.n_seeds}...", end=' ', flush=True)
            scores = self.validate_single_seed(seed)
            self.results.append(scores)
        
        elapsed = time.time() - start_time
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'validation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 结果已保存: {csv_path}")
        
        print("\n" + "=" * 80)
        print("🎯 验证结果汇总")
        print("=" * 80)
        print(df[['nonlinearity', 'emergence', 'scale_invariance', 'reflexivity']].mean().round(3))
        print(f"\n总用时: {elapsed:.1f}s ({elapsed/self.n_seeds:.1f}s/seed)")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='复杂系统验证器 v2.1 修复版')
    parser.add_argument('--seeds', type=int, default=20, help='种子数量')
    parser.add_argument('--ticks', type=int, default=500000, help='每个种子的ticks数')
    parser.add_argument('--output', type=str, default='validation_output_v2_fixed', help='输出目录')
    args = parser.parse_args()
    
    validator = ComplexSystemValidatorV2Fixed(args.seeds, args.ticks, args.output)
    results_df = validator.run_full_validation()

if __name__ == "__main__":
    main()
